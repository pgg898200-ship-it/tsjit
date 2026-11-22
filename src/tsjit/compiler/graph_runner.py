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

"""Multi-GPU safe bucketed CUDA Graph capture and replay runner."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - optional in pure-CPU environments
    torch = None  # type: ignore[assignment]

from .guards import tf32_precision_context

DEFAULT_BATCH_BUCKETS: tuple[int, ...] = (
    1,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
)


@dataclass
class _CapturedBucket:
    bucket_size: int
    graph: Any
    x_in: Any
    y_out: Any


class BucketedCudaGraphRunner:
    """Fixed-bucket CUDA Graph runner sharing one memory pool across all buckets.

    Guarantees:
    - All buckets share a single ``torch.cuda.graph_pool_handle()``.
    - BOTH warmup/capture AND replay execute inside ``with torch.cuda.device(self.device):``.
    - Runs 3 side-stream warmup iterations on finite ``sin(t)`` inputs before capture.
    - Resets static ``x_in`` to ``NaN`` immediately after capture and pads unused
      batch slots ``x_in[B:]`` with ``NaN`` on every replay to prevent stale leakage.
    """

    def __init__(
        self,
        fn: Callable[[Any], Any],
        *,
        sample_shape: tuple[int, ...] | None = None,
        buckets: Sequence[int] | None = None,
        device: Any = None,
        dtype: Any = None,
        warmup_iters: int = 3,
        enabled: bool = True,
    ) -> None:
        self.fn = fn
        self.sample_shape = None if sample_shape is None else tuple(int(s) for s in sample_shape)
        raw_buckets = (
            DEFAULT_BATCH_BUCKETS
            if buckets is None
            else tuple(sorted({int(b) for b in buckets if int(b) > 0}))
        )
        if len(raw_buckets) == 0:
            raise ValueError("buckets must contain at least one positive integer")
        self.buckets: tuple[int, ...] = raw_buckets
        self.warmup_iters = max(1, int(warmup_iters))
        self.enabled = bool(enabled)

        if torch is not None:
            if device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)
            self.dtype = torch.float32 if dtype is None else dtype
        else:
            self.device = None
            self.dtype = None

        self._pool: Any = None
        self._captured: dict[tuple[int, tuple[int, ...], Any], _CapturedBucket] = {}
        self._uncapturable_keys: set[tuple[int, tuple[int, ...], Any]] = set()
        self.replay_count = 0
        self.capture_count = 0
        self.capture_fallback_count = 0
        self.last_capture_error: str | None = None

    @property
    def is_cuda_active(self) -> bool:
        return bool(
            self.enabled
            and torch is not None
            and torch.cuda.is_available()
            and self.device is not None
            and self.device.type == "cuda"
        )

    def _select_bucket(self, batch_size: int) -> int:
        for b in self.buckets:
            if b >= batch_size:
                return b
        return self.buckets[-1]

    def _make_finite_sin_input(
        self, bucket_size: int, tail_shape: tuple[int, ...], dtype: Any
    ) -> Any:
        numel = bucket_size
        for s in tail_shape:
            numel *= s
        t = torch.linspace(0.1, 10.0, max(numel, 1), device=self.device, dtype=torch.float32)
        vals = torch.sin(t) + 0.25 * torch.cos(0.37 * t)
        return vals.reshape(bucket_size, *tail_shape).to(dtype=dtype).contiguous()

    def _ensure_captured(
        self, bucket_size: int, tail_shape: tuple[int, ...], dtype: Any
    ) -> _CapturedBucket | None:
        key = (bucket_size, tail_shape, dtype)
        if key in self._captured:
            return self._captured[key]
        if key in self._uncapturable_keys:
            return None

        with torch.cuda.device(self.device):
            if self._pool is None:
                self._pool = torch.cuda.graph_pool_handle()

            x_in = self._make_finite_sin_input(bucket_size, tail_shape, dtype)
            side_stream = torch.cuda.Stream(device=self.device)
            side_stream.wait_stream(torch.cuda.current_stream(self.device))

            with torch.inference_mode():
                with torch.cuda.stream(side_stream):
                    for _ in range(self.warmup_iters):
                        _ = self.fn(x_in)
                torch.cuda.current_stream(self.device).wait_stream(side_stream)
                torch.cuda.synchronize(self.device)

                graph = torch.cuda.CUDAGraph()
                try:
                    with torch.cuda.graph(graph, pool=self._pool, stream=side_stream):
                        y_out = self.fn(x_in)
                except torch.cuda.OutOfMemoryError:
                    raise
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        raise
                    # Non-OOM capture incompatibility (e.g., third-party host sync):
                    # record telemetry and fall back to eager execution for this bucket.
                    self._uncapturable_keys.add(key)
                    self.capture_fallback_count += 1
                    self.last_capture_error = f"{type(exc).__name__}: {exc}"
                    return None

                # Reset input buffer to NaN after capture so stale warmup values never linger.
                x_in.fill_(float("nan"))

        entry = _CapturedBucket(bucket_size=bucket_size, graph=graph, x_in=x_in, y_out=y_out)
        self._captured[key] = entry
        self.capture_count += 1
        return entry

    def warmup(
        self,
        buckets: Sequence[int] | None = None,
        sample_shape: tuple[int, ...] | None = None,
    ) -> list[int]:
        """Eagerly capture CUDA Graphs for the requested buckets."""
        if not self.is_cuda_active:
            return []
        tail = sample_shape or self.sample_shape
        if tail is None:
            raise ValueError("sample_shape must be specified for eager warmup")
        targets = tuple(buckets) if buckets is not None else self.buckets
        captured: list[int] = []
        for b in targets:
            if self._ensure_captured(int(b), tuple(tail), self.dtype) is not None:
                captured.append(int(b))
        return captured

    @staticmethod
    def _slice_clone(out: Any, batch_size: int) -> Any:
        if isinstance(out, tuple):
            return tuple(t[:batch_size].clone() for t in out)
        if isinstance(out, list):
            return [t[:batch_size].clone() for t in out]
        return out[:batch_size].clone()

    @staticmethod
    def _concat_outputs(chunks: list[Any]) -> Any:
        first = chunks[0]
        if isinstance(first, tuple):
            return tuple(torch.cat([c[i] for c in chunks], dim=0) for i in range(len(first)))
        if isinstance(first, list):
            return [torch.cat([c[i] for c in chunks], dim=0) for i in range(len(first))]
        return torch.cat(chunks, dim=0)

    def __call__(self, x: Any) -> Any:
        if not self.is_cuda_active or torch is None or not isinstance(x, torch.Tensor):
            if torch is not None:
                with torch.inference_mode():
                    return self.fn(x)
            return self.fn(x)

        if x.ndim < 1:
            raise ValueError(f"Expected batched tensor with ndim >= 1, got {tuple(x.shape)}")
        B = int(x.shape[0])
        if B == 0:
            with torch.inference_mode():
                return self.fn(x)

        max_bucket = self.buckets[-1]
        if B > max_bucket:
            chunks = [self(x[start : start + max_bucket]) for start in range(0, B, max_bucket)]
            return self._concat_outputs(chunks)

        x_dev = x.to(device=self.device, non_blocking=True) if x.device != self.device else x
        tail_shape = tuple(int(s) for s in x_dev.shape[1:])
        b_cap = self._select_bucket(B)
        entry = self._ensure_captured(b_cap, tail_shape, x_dev.dtype)

        with torch.cuda.device(self.device), torch.inference_mode():
            if entry is None:
                return self.fn(x_dev)
            entry.x_in[:B].copy_(x_dev)
            if B < b_cap:
                entry.x_in[B:].fill_(float("nan"))
            entry.graph.replay()
            self.replay_count += 1
            return self._slice_clone(entry.y_out, B)

    def stats(self) -> dict[str, Any]:
        """Return capture and replay telemetry."""
        return {
            "cuda_graph_active": self.is_cuda_active,
            "buckets": list(self.buckets),
            "captured_buckets": sorted({k[0] for k in self._captured}),
            "capture_count": self.capture_count,
            "replay_count": self.replay_count,
            "capture_fallback_count": self.capture_fallback_count,
            "last_capture_error": self.last_capture_error,
        }


class InductorOrGraphRunner:
    """Composable CUDA execution runner: Inductor ``reduce-overhead`` with CUDA-Graph fallback.

    - When ``use_inductor=True`` on CUDA (i.e., zero ``CrestGuardedLinear`` layers in the
      algebraically-optimized core), wraps ``fn`` with ``torch.compile(fn, mode='reduce-overhead')``
      so Inductor fuses elementwise ops + TensorCore TF32/FP32 GEMMs into CUDAGraph trees.
    - When ``use_inductor=False`` (e.g., ``fp16x3`` / ``scaled-fp16`` or outlier-escalated
      ``a10-auto`` containing ``CrestGuardedLinear``) or if Inductor raises a non-OOM error,
      transparently falls back to ``BucketedCudaGraphRunner``.
    - Applies ``tf32_precision_context(precision)`` outside the compiled callable so Inductor
      sees zero context-manager graph breaks while enforcing exact TF32/FP32 semantics.
    """

    def __init__(
        self,
        fn: Callable[[Any], Any],
        *,
        sample_shape: tuple[int, ...] | None = None,
        buckets: Sequence[int] | None = None,
        device: Any = None,
        dtype: Any = None,
        warmup_iters: int = 3,
        enabled: bool = True,
        use_inductor: bool = False,
        precision: str = "auto",
        inductor_mode: str = "default",
        inductor_max_batch: int | None = None,
    ) -> None:
        self.fn = fn
        self.precision = precision
        self.inductor_mode = inductor_mode
        self.inductor_max_batch = (
            None if inductor_max_batch is None else max(1, int(inductor_max_batch))
        )

        def _graph_fn(x_in: Any) -> Any:
            with tf32_precision_context(self.precision):
                return self.fn(x_in)

        self._graph_runner = BucketedCudaGraphRunner(
            _graph_fn,
            sample_shape=sample_shape,
            buckets=buckets,
            device=device,
            dtype=dtype,
            warmup_iters=warmup_iters,
            enabled=enabled,
        )
        self.device = self._graph_runner.device
        self.dtype = self._graph_runner.dtype
        self.buckets = self._graph_runner.buckets
        self.sample_shape = self._graph_runner.sample_shape
        self.enabled = self._graph_runner.enabled

        self._inductor_requested = bool(use_inductor and self._graph_runner.is_cuda_active)
        self._inductor_active = False
        self._compiled_fn: Any = None
        self._inductor_graph_runner: BucketedCudaGraphRunner | None = None
        self.inductor_call_count = 0
        self.inductor_fallback_count = 0
        self.inductor_error: str | None = None

        if (
            self._inductor_requested
            and torch is not None
            and callable(getattr(torch, "compile", None))
        ):
            try:
                dynamo_cfg = getattr(getattr(torch, "_dynamo", None), "config", None)
                if dynamo_cfg is not None and hasattr(dynamo_cfg, "cache_size_limit"):
                    needed = max(16, len(self.buckets) * 2 + 4)
                    if int(getattr(dynamo_cfg, "cache_size_limit", 8)) < needed:
                        dynamo_cfg.cache_size_limit = needed
                # Compose Inductor kernel fusion (mode="default") inside our own isolated
                # BucketedCudaGraphRunner instead of CUDAGraphTrees ("reduce-overhead") to avoid
                # cross-graph pool interference when multiple compiled models coexist.
                compiled = torch.compile(self.fn, mode=self.inductor_mode)
                self._compiled_fn = compiled

                def _inductor_graph_fn(x_in: Any) -> Any:
                    with tf32_precision_context(self.precision):
                        return compiled(x_in)

                ind_buckets = (
                    tuple(b for b in self.buckets if b <= self.inductor_max_batch)
                    if self.inductor_max_batch is not None
                    else self.buckets
                )
                if ind_buckets:
                    self._inductor_graph_runner = BucketedCudaGraphRunner(
                        _inductor_graph_fn,
                        sample_shape=sample_shape,
                        buckets=ind_buckets,
                        device=device,
                        dtype=dtype,
                        warmup_iters=warmup_iters,
                        enabled=enabled,
                    )
                self._inductor_active = True
            except Exception as exc:
                self.inductor_fallback_count += 1
                self.inductor_error = f"{type(exc).__name__}: {exc}"

    @property
    def is_cuda_active(self) -> bool:
        return self._graph_runner.is_cuda_active

    @staticmethod
    def _clone_output(out: Any) -> Any:
        if isinstance(out, tuple):
            return tuple(t.clone() for t in out)
        if isinstance(out, list):
            return [t.clone() for t in out]
        return out.clone()

    def warmup(
        self,
        buckets: Sequence[int] | None = None,
        sample_shape: tuple[int, ...] | None = None,
    ) -> list[int]:
        if not self.is_cuda_active:
            return []
        if not self._inductor_active:
            return self._graph_runner.warmup(buckets=buckets, sample_shape=sample_shape)

        tail = sample_shape or self.sample_shape
        if tail is None:
            raise ValueError("sample_shape must be specified for eager warmup")
        targets = tuple(buckets) if buckets is not None else self.buckets
        warmed: list[int] = []
        for idx, b in enumerate(targets):
            b_int = int(b)
            if not self._inductor_active:
                rem = self._graph_runner.warmup(buckets=targets[idx:], sample_shape=tuple(tail))
                warmed.extend(rem)
                break
            if self.inductor_max_batch is not None and b_int > self.inductor_max_batch:
                warmed.extend(
                    self._graph_runner.warmup(buckets=(b_int,), sample_shape=tuple(tail))
                )
                continue
            x_w = self._graph_runner._make_finite_sin_input(b_int, tuple(tail), self.dtype)
            _ = self(x_w)
            if self._inductor_active:
                _ = self(x_w)
            warmed.append(b_int)
        return warmed

    def __call__(self, x: Any) -> Any:
        if not self.is_cuda_active or torch is None or not isinstance(x, torch.Tensor):
            if torch is not None:
                with torch.inference_mode(), tf32_precision_context(self.precision):
                    return self.fn(x)
            return self.fn(x)

        if x.ndim < 1:
            raise ValueError(f"Expected batched tensor with ndim >= 1, got {tuple(x.shape)}")
        B = int(x.shape[0])
        if B == 0:
            with (
                torch.cuda.device(self.device),
                torch.inference_mode(),
                tf32_precision_context(self.precision),
            ):
                return self.fn(x)

        use_inductor_here = self._inductor_active and (
            self.inductor_max_batch is None or B <= self.inductor_max_batch
        )
        if use_inductor_here:
            try:
                if self._inductor_graph_runner is not None and self.inductor_mode == "default":
                    out = self._inductor_graph_runner(x)
                    self.inductor_call_count += 1
                    return out
                x_dev = (
                    x.to(device=self.device, non_blocking=True) if x.device != self.device else x
                )
                with (
                    torch.cuda.device(self.device),
                    torch.inference_mode(),
                    tf32_precision_context(self.precision),
                ):
                    mark_step = getattr(
                        getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None
                    )
                    if callable(mark_step):
                        mark_step()
                    out = self._compiled_fn(x_dev)
                    self.inductor_call_count += 1
                    return self._clone_output(out)
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as exc:
                if "out of memory" in str(exc).lower():
                    raise
                self._inductor_active = False
                self.inductor_fallback_count += 1
                self.inductor_error = f"{type(exc).__name__}: {exc}"

        return self._graph_runner(x)

    def stats(self) -> dict[str, Any]:
        base = self._graph_runner.stats()
        if self._inductor_active and self._inductor_graph_runner is not None:
            ind_stats = self._inductor_graph_runner.stats()
            base["captured_buckets"] = sorted(
                set(base["captured_buckets"]) | set(ind_stats["captured_buckets"])
            )
            base["capture_count"] = int(base["capture_count"]) + int(ind_stats["capture_count"])
            base["replay_count"] = int(base["replay_count"]) + int(ind_stats["replay_count"])
            base["capture_fallback_count"] = int(base["capture_fallback_count"]) + int(
                ind_stats["capture_fallback_count"]
            )
            backend = "inductor_cuda_graph_buckets"
        elif self._inductor_active:
            backend = "inductor_reduce_overhead"
        elif self._graph_runner.is_cuda_active:
            backend = "cuda_graph_buckets"
        else:
            backend = "eager"
        base.update(
            {
                "backend": backend,
                "inductor_requested": self._inductor_requested,
                "inductor_active": self._inductor_active,
                "inductor_max_batch": self.inductor_max_batch,
                "inductor_call_count": self.inductor_call_count,
                "inductor_fallback_count": self.inductor_fallback_count,
                "inductor_error": self.inductor_error,
            }
        )
        return base
