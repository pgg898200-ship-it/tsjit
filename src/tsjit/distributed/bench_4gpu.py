# Copyright 2026 tsjit contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multi-GPU (W=1, 2, 4) torchrun benchmark for AsyncFuse vs PipeFuse vs AllGather.

Usage:
    torchrun --nproc_per_node=4 -m tsjit.distributed.bench_4gpu --output results/bench_4gpu_W4.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.distributed as dist
    import torch.nn as nn
except ImportError:  # pragma: no cover - optional in pure-CPU environments
    torch = None  # type: ignore[assignment]
    dist = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

from .async_fuse import (
    AsyncFuseMonoid,
    AsyncFuseRunner,
    DistributedStreamCoordinator,
    PipeFuseRunner,
)


def _init_distributed() -> tuple[int, int, int, Any]:
    if torch is None:
        raise RuntimeError("bench_4gpu requires PyTorch.")
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank % max(1, torch.cuda.device_count())))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return rank, world_size, local_rank, device


def _calibrate_gpu_sleep_cycles(device: Any, target_us: float = 500.0) -> tuple[int, float]:
    """Calibrate ``torch.cuda._sleep`` cycles using pure GPU event timing (``t_sleep - t_noop``).

    Returns ``(calibrated_cycles, measured_pure_us)`` so ``rotating_straggler_500us``
    injects an accurate ~``target_us`` pure GPU stall on both W=1 and W=4 without
    host synchronization overhead distorting the calibration.
    """
    if device.type != "cuda" or not hasattr(torch.cuda, "_sleep") or target_us <= 0.0:
        return 0, 0.0
    with torch.cuda.device(device):
        # Warm up kernel launch path before timing.
        torch.cuda._sleep(10_000)
        torch.cuda.synchronize(device)

        def _gpu_us(cycles: int, reps: int = 5) -> float:
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()
            for _ in range(reps):
                if cycles > 0:
                    torch.cuda._sleep(cycles)
            ev1.record()
            torch.cuda.synchronize(device)
            return float(ev0.elapsed_time(ev1)) * 1e3 / float(reps)

        t_noop = _gpu_us(0, reps=5)
        c_lo = 1_000
        c_hi = 2_000_000
        t_lo = _gpu_us(c_lo, reps=5)
        t_hi = _gpu_us(c_hi, reps=5)
        pure_delta_us = max(t_hi - t_lo, 1.0)
        cycles_per_us = float(c_hi - c_lo) / pure_delta_us
        cycles = max(1_000, int(round(target_us * cycles_per_us)))

        # Refine once around the calibrated target to eliminate nonlinearity.
        t_check = max(_gpu_us(cycles, reps=5) - t_noop, 1.0)
        cycles = max(1_000, int(round(cycles * (target_us / t_check))))
        measured_us = max(_gpu_us(cycles, reps=5) - t_noop, 0.0)
        return cycles, float(measured_us)


def _build_backbone_pair(
    in_dim: int,
    embed_dim: int,
    out_dim: int,
    device: Any,
    seed: int = 2026,
) -> tuple[Any, Any]:
    torch.manual_seed(seed)
    pre = (
        nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        .to(device)
        .eval()
    )
    tail = (
        nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, out_dim),
        )
        .to(device)
        .eval()
    )
    return pre, tail


class _AllGatherVariateRunner:
    """Fair pre-allocated + optionally CUDA-Graphed AllGather variate-attention baseline.

    Eliminates per-step tensor allocation and ``torch.cat`` overhead by pre-allocating a
    single static ``all_tokens`` buffer and calling ``dist.all_gather_into_tensor`` (with
    identical CUDA Graph capture for local ``pre`` and ``tail`` compute as ``AsyncFuseRunner``).
    """

    def __init__(
        self,
        pre: Any,
        tail: Any,
        *,
        n_local: int,
        in_dim: int,
        embed_dim: int,
        seq_len: int,
        world_size: int,
        device: Any,
        use_cuda_graphs: bool = True,
    ) -> None:
        self.pre = pre
        self.tail = tail
        self.n_local = int(n_local)
        self.in_dim = int(in_dim)
        self.embed_dim = int(embed_dim)
        self.seq_len = int(seq_len)
        self.world_size = int(world_size)
        self.device = device
        self.use_cuda_graphs = bool(use_cuda_graphs and device.type == "cuda")
        self._is_dist = bool(
            world_size > 1 and dist is not None and dist.is_available() and dist.is_initialized()
        )
        self._has_into_tensor = bool(dist is not None and hasattr(dist, "all_gather_into_tensor"))

        # Pre-allocate static communication buffers ONCE outside the timed loop.
        self._x_buf = torch.zeros((self.n_local, self.in_dim), dtype=torch.float32, device=device)
        self._tokens_local = torch.empty(
            (self.n_local, self.seq_len, self.embed_dim), dtype=torch.float32, device=device
        )
        self._all_tokens = torch.empty(
            (self.world_size, self.n_local, self.seq_len, self.embed_dim),
            dtype=torch.float32,
            device=device,
        )
        self._all_tokens_list = [self._all_tokens[r] for r in range(self.world_size)]
        self._h_buf: Any = None
        self._y_buf: Any = None
        self._g_pre: Any = None
        self._g_tail: Any = None
        self._pool: Any = None
        self._captured = False

        bytes_per_rank = self._tokens_local.numel() * self._tokens_local.element_size()
        self.comm_bytes_per_step = int(bytes_per_rank * self.world_size) if self._is_dist else 0

    def _run_pre_ops(self) -> Any:
        h_local = self.pre(self._x_buf)  # (N_local, D)
        self._tokens_local.copy_(h_local.unsqueeze(1).expand(-1, self.seq_len, -1))
        return h_local

    def _run_tail_ops(self, h_local: Any) -> Any:
        if self._is_dist:
            peer_ctx = self._all_tokens.mean(dim=(0, 1, 2), keepdim=False)
        else:
            peer_ctx = self._tokens_local.mean(dim=(0, 1), keepdim=False)
        h_fused = h_local + 0.05 * peer_ctx.unsqueeze(0)
        return self.tail(h_fused)

    def _ensure_captured(self) -> None:
        if not self.use_cuda_graphs or self._captured:
            return
        with torch.cuda.device(self.device), torch.inference_mode():
            self._pool = torch.cuda.graph_pool_handle()
            side = torch.cuda.Stream(device=self.device)
            side.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(side):
                for _ in range(3):
                    h_tmp = self._run_pre_ops()
                    _ = self._run_tail_ops(h_tmp)
            torch.cuda.current_stream(self.device).wait_stream(side)
            torch.cuda.synchronize(self.device)

            self._g_pre = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._g_pre, pool=self._pool, stream=side):
                self._h_buf = self._run_pre_ops()

            self._g_tail = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._g_tail, pool=self._pool, stream=side):
                self._y_buf = self._run_tail_ops(self._h_buf)
            self._captured = True

    def step(self, x_local: Any) -> Any:
        with torch.inference_mode():
            if self.use_cuda_graphs:
                self._ensure_captured()
                with torch.cuda.device(self.device):
                    self._x_buf.copy_(x_local)
                    self._g_pre.replay()
                    if self._is_dist:
                        if self._has_into_tensor:
                            dist.all_gather_into_tensor(self._all_tokens, self._tokens_local)
                        else:
                            dist.all_gather(self._all_tokens_list, self._tokens_local)
                    self._g_tail.replay()
                    return self._y_buf.clone()

            self._x_buf.copy_(x_local)
            h_local = self._run_pre_ops()
            if self._is_dist:
                if self._has_into_tensor:
                    dist.all_gather_into_tensor(self._all_tokens, self._tokens_local)
                else:
                    dist.all_gather(self._all_tokens_list, self._tokens_local)
            return self._run_tail_ops(h_local)


def _benchmark_config(
    *,
    label: str,
    n_local: int,
    rank: int,
    world_size: int,
    device: Any,
    num_groups: int,
    embed_dim: int,
    fuse_dim: int,
    in_dim: int,
    out_dim: int,
    token_expand: int,
    warmup_steps: int,
    bench_steps: int,
    straggler_cycles: int,
) -> dict[str, Any]:
    pre, tail = _build_backbone_pair(in_dim, embed_dim, out_dim, device)
    torch.manual_seed(2026)
    monoid = AsyncFuseMonoid(embed_dim, fuse_dim, num_groups=num_groups).to(device).eval()

    gen = torch.Generator(device="cpu").manual_seed(1000 + rank)
    x = torch.randn((n_local, in_dim), generator=gen, dtype=torch.float32).to(device)
    gids = (torch.arange(n_local, dtype=torch.int64) % num_groups).to(device)
    sids = (torch.arange(n_local, dtype=torch.int64) + rank * n_local).to(device)

    async_runner = AsyncFuseRunner(
        pre,
        monoid,
        tail,
        batch_size=n_local,
        input_shape=(in_dim,),
        device=device,
        use_cuda_graphs=(device.type == "cuda"),
    )
    async_eager_runner = AsyncFuseRunner(
        pre,
        monoid,
        tail,
        batch_size=n_local,
        input_shape=(in_dim,),
        device=device,
        use_cuda_graphs=False,
    )
    pipe_runner = PipeFuseRunner(pre, monoid, tail, device=device)
    ag_graphed_runner = _AllGatherVariateRunner(
        pre,
        tail,
        n_local=n_local,
        in_dim=in_dim,
        embed_dim=embed_dim,
        seq_len=token_expand,
        world_size=world_size,
        device=device,
        use_cuda_graphs=(device.type == "cuda"),
    )
    ag_eager_runner = _AllGatherVariateRunner(
        pre,
        tail,
        n_local=n_local,
        in_dim=in_dim,
        embed_dim=embed_dim,
        seq_len=token_expand,
        world_size=world_size,
        device=device,
        use_cuda_graphs=False,
    )

    is_dist = bool(
        world_size > 1 and dist is not None and dist.is_available() and dist.is_initialized()
    )
    monoid_comm_bytes = int(monoid.payload_bytes) if is_dist else 0

    def _sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if is_dist:
            dist.barrier()

    def _maybe_straggle(step_idx: int) -> None:
        if straggler_cycles > 0 and (step_idx % world_size) == rank and device.type == "cuda":
            torch.cuda._sleep(straggler_cycles)

    def _measure(
        fn: Any, *, warmup_fn: Any | None = None, flush_fn: Any | None = None
    ) -> dict[str, float]:
        w_fn = fn if warmup_fn is None else warmup_fn
        for i in range(warmup_steps):
            w_fn(i)
        if flush_fn is not None:
            flush_fn()
        _sync()
        lat_ms: list[float] = []
        if device.type == "cuda":
            # Pipeline all steps without per-step host sync so inter-step NCCL overlap
            # and rotating-straggler amortization are preserved accurately.
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(bench_steps)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(bench_steps)]
            stream = torch.cuda.current_stream(device)
            t_start = time.perf_counter()
            for i in range(bench_steps):
                starts[i].record(stream)
                fn(i)
                ends[i].record(stream)
            if flush_fn is not None:
                flush_fn()
            _sync()
            total_s = time.perf_counter() - t_start
            lat_ms = [float(starts[i].elapsed_time(ends[i])) for i in range(bench_steps)]
        else:
            t_start = time.perf_counter()
            for i in range(bench_steps):
                t0 = time.perf_counter()
                fn(i)
                lat_ms.append((time.perf_counter() - t0) * 1e3)
            if flush_fn is not None:
                flush_fn()
            _sync()
            total_s = time.perf_counter() - t_start
        arr = np.asarray(lat_ms, dtype=np.float64)
        n_total = n_local * world_size
        return {
            "p50_ms": float(np.percentile(arr, 50)),
            "p99_ms": float(np.percentile(arr, 99)),
            "mean_ms": float(np.mean(arr)),
            "throughput_series_per_s": float((n_total * bench_steps) / max(total_s, 1e-9)),
        }

    # 1a. AsyncFuse graphed (3-graph ping-pong overlapped ~16 KB comm)
    async_runner.reset_state()

    def _run_async_warmup(i: int) -> None:
        _maybe_straggle(i)
        async_runner.step(x, gids, sids, validate_ids=(i == 0))

    def _run_async(i: int) -> None:
        _maybe_straggle(i)
        # sids/gids are static tensors already validated on warmup step 0.
        async_runner.step(x, gids, sids, validate_ids=False)

    res_async = _measure(_run_async, warmup_fn=_run_async_warmup, flush_fn=async_runner.flush)
    res_async["comm_bytes_per_step"] = monoid_comm_bytes

    # 1b. AsyncFuse eager (no CUDA Graphs — apples-to-apples with eager AllGather & PipeFuse)
    async_eager_runner.reset_state()

    def _run_async_eager_warmup(i: int) -> None:
        _maybe_straggle(i)
        async_eager_runner.step(x, gids, sids, validate_ids=(i == 0))

    def _run_async_eager(i: int) -> None:
        _maybe_straggle(i)
        async_eager_runner.step(x, gids, sids, validate_ids=False)

    res_async_eager = _measure(
        _run_async_eager, warmup_fn=_run_async_eager_warmup, flush_fn=async_eager_runner.flush
    )
    res_async_eager["comm_bytes_per_step"] = monoid_comm_bytes

    # 2. PipeFuse (zero-lag double-buffered panel split)
    def _run_pipe(i: int) -> None:
        _maybe_straggle(i)
        pipe_runner.step(x, gids)

    res_pipe = _measure(_run_pipe)
    res_pipe["comm_bytes_per_step"] = monoid_comm_bytes

    # 3a. AllGather variate-attention baseline (pre-allocated all_gather_into_tensor + CUDA Graphs)
    def _run_allgather_graphed(i: int) -> None:
        _maybe_straggle(i)
        ag_graphed_runner.step(x)

    res_ag = _measure(_run_allgather_graphed)
    res_ag["comm_bytes_per_step"] = ag_graphed_runner.comm_bytes_per_step

    # 3b. AllGather variate-attention eager (pre-allocated all_gather_into_tensor, no CUDA Graphs)
    def _run_allgather_eager(i: int) -> None:
        _maybe_straggle(i)
        ag_eager_runner.step(x)

    res_ag_eager = _measure(_run_allgather_eager)
    res_ag_eager["comm_bytes_per_step"] = ag_eager_runner.comm_bytes_per_step

    ag_comm_bytes = ag_graphed_runner.comm_bytes_per_step
    return {
        "label": label,
        "world_size": world_size,
        "n_local": n_local,
        "n_total": n_local * world_size,
        "num_groups": num_groups,
        "embed_dim": embed_dim,
        "fuse_dim": fuse_dim,
        "seq_len": token_expand,
        "straggler_injected": straggler_cycles > 0,
        "async_fuse": res_async,
        "async_fuse_eager": res_async_eager,
        "pipe_fuse": res_pipe,
        "allgather_variate": res_ag,
        "allgather_variate_eager": res_ag_eager,
        "speedup_async_vs_allgather": float(res_ag["mean_ms"] / max(res_async["mean_ms"], 1e-9)),
        "speedup_async_eager_vs_allgather_eager": float(
            res_ag_eager["mean_ms"] / max(res_async_eager["mean_ms"], 1e-9)
        ),
        "comm_reduction_ratio": (
            float(ag_comm_bytes / max(monoid_comm_bytes, 1))
            if (is_dist and ag_comm_bytes > 0 and monoid_comm_bytes > 0)
            else 1.0
        ),
    }


def _verify_pipefuse_central_parity(
    rank: int,
    world_size: int,
    device: Any,
    *,
    n_local: int = 256,
    num_groups: int = 16,
    embed_dim: int = 128,
    fuse_dim: int = 128,
    in_dim: int = 64,
    out_dim: int = 16,
) -> dict[str, Any]:
    """Verify exact sharded-vs-central parity of PipeFuse across all ranks."""
    pre, tail = _build_backbone_pair(in_dim, embed_dim, out_dim, device, seed=4242)
    torch.manual_seed(4242)
    monoid = AsyncFuseMonoid(embed_dim, fuse_dim, num_groups=num_groups).to(device).eval()
    pipe_runner = PipeFuseRunner(pre, monoid, tail, device=device)

    gen = torch.Generator(device="cpu").manual_seed(777 + rank)
    x_local = torch.randn((n_local, in_dim), generator=gen, dtype=torch.float32).to(device)
    gids_local = torch.randint(0, num_groups, (n_local,), generator=gen, dtype=torch.int64).to(
        device
    )

    y_sharded = pipe_runner.step(x_local, gids_local)

    if world_size > 1 and dist.is_initialized():
        all_x = [torch.empty_like(x_local) for _ in range(world_size)]
        all_g = [torch.empty_like(gids_local) for _ in range(world_size)]
        all_y = [torch.empty_like(y_sharded) for _ in range(world_size)]
        dist.all_gather(all_x, x_local)
        dist.all_gather(all_g, gids_local)
        dist.all_gather(all_y, y_sharded)
        x_full = torch.cat(all_x, dim=0)
        g_full = torch.cat(all_g, dim=0)
        y_sharded_full = torch.cat(all_y, dim=0)
    else:
        x_full = x_local
        g_full = gids_local
        y_sharded_full = y_sharded

    with torch.inference_mode():
        h_full = pre(x_full)
        hf_central, _, _ = monoid(h_full, g_full)
        y_central = tail(hf_central)

    max_diff = float((y_sharded_full - y_central).abs().max().item())
    return {
        "world_size": world_size,
        "n_total": int(x_full.shape[0]),
        "max_abs_diff": max_diff,
        "passed_1e5": max_diff <= 1e-5,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="4xH200 AsyncFuse vs PipeFuse vs AllGather benchmark."
    )
    parser.add_argument("--num-groups", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--fuse-dim", type=int, default=128)
    parser.add_argument("--in-dim", type=int, default=128)
    parser.add_argument("--out-dim", type=int, default=16)
    parser.add_argument(
        "--token-expand",
        "--seq-len",
        dest="token_expand",
        type=int,
        default=16,
        help="Patch sequence length (default 16, matching ctx=512, P=32) for AllGather volume.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--straggler-us", type=float, default=500.0)
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank, world_size, _local_rank, device = _init_distributed()

    try:
        straggler_cycles, measured_straggler_us = _calibrate_gpu_sleep_cycles(
            device, args.straggler_us
        )
        parity = _verify_pipefuse_central_parity(rank, world_size, device)

        # Stream coordinator < 2 KB payload check
        coord = DistributedStreamCoordinator(num_groups=16, device=device)
        z = np.zeros(16, dtype=np.float32)
        ones = np.ones(16, dtype=np.float32)
        coord.submit_async(z, z, ones, z, ones, z)
        coord_state = coord.poll_state()

        weak_res = _benchmark_config(
            label="weak_scaling_4096_per_gpu",
            n_local=4096,
            rank=rank,
            world_size=world_size,
            device=device,
            num_groups=args.num_groups,
            embed_dim=args.embed_dim,
            fuse_dim=args.fuse_dim,
            in_dim=args.in_dim,
            out_dim=args.out_dim,
            token_expand=args.token_expand,
            warmup_steps=args.warmup,
            bench_steps=args.steps,
            straggler_cycles=0,
        )

        strong_n_local = max(1, 16384 // world_size)
        strong_res = _benchmark_config(
            label="strong_scaling_16384_total",
            n_local=strong_n_local,
            rank=rank,
            world_size=world_size,
            device=device,
            num_groups=args.num_groups,
            embed_dim=args.embed_dim,
            fuse_dim=args.fuse_dim,
            in_dim=args.in_dim,
            out_dim=args.out_dim,
            token_expand=args.token_expand,
            warmup_steps=args.warmup,
            bench_steps=args.steps,
            straggler_cycles=0,
        )

        straggler_res = _benchmark_config(
            label=f"rotating_straggler_{int(args.straggler_us)}us",
            n_local=4096,
            rank=rank,
            world_size=world_size,
            device=device,
            num_groups=args.num_groups,
            embed_dim=args.embed_dim,
            fuse_dim=args.fuse_dim,
            in_dim=args.in_dim,
            out_dim=args.out_dim,
            token_expand=args.token_expand,
            warmup_steps=args.warmup,
            bench_steps=args.steps,
            straggler_cycles=straggler_cycles,
        )

        summary = {
            "world_size": world_size,
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "seq_len": args.token_expand,
            "calibrated_straggler_cycles": straggler_cycles,
            "measured_straggler_us": measured_straggler_us,
            "pipefuse_central_parity": parity,
            "stream_coordinator_payload_bytes": coord_state.payload_bytes,
            "benchmarks": [weak_res, strong_res, straggler_res],
        }

        if rank == 0:
            out_path = Path(args.output or f"results/bench_4gpu_W{world_size}.json")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(summary, indent=2))
    finally:
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
