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

"""Live H200 benchmark for SM90 ADAPT-IPCG (tsjit_ipcg_sm90.cu) vs GPU/CPU Cholesky.

Evaluates:
- N in {256, 1024, 4096, 16384, 32768}, K in {64, 136}, FP64 and FP32.
- tsjit_ipcg_sm90.cu (SMEM-resident persistent grid-stride kernel) vs
  torch.linalg.cholesky + cholesky_solve (GPU) and scipy.linalg.cho_factor /
  numpy.linalg.solve (CPU).
- 100x diagonal shock recovery, ill-conditioned kappa=1e8 Gram verification,
  and selective cholesky_ex fallback behavior.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import scipy.linalg as spla
except ImportError:  # pragma: no cover
    spla = None  # type: ignore[assignment]

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from tsjit.cuda.loader import (
    OnlineGpuRidgeEngine,
    align_k_pad,
    detect_cuda_arch,
    ipcg_solve_batch,
    is_cuda_ipcg_available,
)


def _build_online_ridge_problem(
    N: int,
    K: int,
    dtype: Any,
    device: Any,
    *,
    num_rank1_updates: int = 1,
    shock_scale: float = 1.0,
    seed: int = 2026,
) -> tuple[Any, Any, Any, Any, Any]:
    """Generate batched SPD online ridge update A_m = A_0 + sum_{j=1..m} u_j u_j^T."""
    m_steps = max(1, int(num_rank1_updates))
    gen = torch.Generator(device="cpu").manual_seed(seed)
    # Build well-conditioned SPD A0 on CPU then transfer (deterministic across runs).
    B_mat = torch.randn((N, K, K), generator=gen, dtype=torch.float64) / np.sqrt(K)
    A0 = torch.bmm(B_mat.transpose(1, 2), B_mat)
    eye = torch.eye(K, dtype=torch.float64).unsqueeze(0)
    A0 = A0 + 0.5 * eye
    rhs0 = torch.randn((N, K), generator=gen, dtype=torch.float64)

    A0_dev = A0.to(device=device)
    rhs0_dev = rhs0.to(device=device)
    L0 = torch.linalg.cholesky(A0_dev)
    M0_dev = torch.cholesky_inverse(L0)
    theta0_dev = torch.cholesky_solve(rhs0_dev.unsqueeze(-1), L0).squeeze(-1)

    U = (0.25 * torch.randn((N, m_steps, K), generator=gen, dtype=torch.float64)).to(device=device)
    A1_dev = A0_dev + torch.bmm(U.transpose(1, 2), U)
    if shock_scale != 1.0:
        diag_idx = torch.arange(K, device=device)
        A1_dev[:, diag_idx, diag_idx] *= shock_scale
    rhs1_dev = rhs0_dev + 0.2 * U.sum(dim=1)

    return (
        A1_dev.to(dtype=dtype).contiguous(),
        M0_dev.to(dtype=dtype).contiguous(),
        rhs1_dev.to(dtype=dtype).contiguous(),
        theta0_dev.to(dtype=dtype).contiguous(),
        A1_dev,
    )


def _time_cuda_fn(fn: Any, warmup: int, iters: int, device: Any) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    times_ms: list[float] = []
    for _ in range(iters):
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        fn()
        ev1.record()
        torch.cuda.synchronize(device)
        times_ms.append(float(ev0.elapsed_time(ev1)))
    arr = np.asarray(times_ms, dtype=np.float64)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(np.mean(arr)),
    }


def _time_cpu_cholesky(
    A_np: np.ndarray,
    rhs_np: np.ndarray,
    cpu_max_n: int,
    *,
    total_n: int | None = None,
) -> dict[str, Any]:
    full_n = int(A_np.shape[0] if total_n is None else total_n)
    sample_n = min(int(A_np.shape[0]), int(cpu_max_n))
    t0 = time.perf_counter()
    if spla is not None:
        for i in range(sample_n):
            c, lower = spla.cho_factor(A_np[i], overwrite_a=False, check_finite=False)
            _ = spla.cho_solve((c, lower), rhs_np[i], check_finite=False)
        method = "scipy.linalg.cho_factor"
    else:
        for i in range(sample_n):
            _ = np.linalg.solve(A_np[i], rhs_np[i])
        method = "numpy.linalg.solve"
    dt_ms = (time.perf_counter() - t0) * 1e3
    est_full_ms = dt_ms * (float(full_n) / float(max(sample_n, 1)))
    return {
        "method": method,
        "total_n": full_n,
        "sampled_n": sample_n,
        "sampled_ms": float(dt_ms),
        "estimated_full_ms": float(est_full_ms),
    }


def _run_multistep_drift_sweep(
    device: Any,
    *,
    cuda_ipcg: bool,
    N: int = 4096,
    ks: list[int] | tuple[int, ...] = (64, 136),
    drift_steps: list[int] | tuple[int, ...] = (1, 4, 8, 16),
    dtypes: list[str] | tuple[str, ...] = ("float64", "float32"),
    tol_f64: float = 1e-8,
    tol_f32: float = 1e-5,
    max_iters: int = 24,
    warmup: int = 5,
    iters: int = 20,
) -> list[dict[str, Any]]:
    """Measure frozen-preconditioner drift sweep (m = 1, 4, 8, 16 rank-1 updates from A0)."""
    sweep_rows: list[dict[str, Any]] = []
    print(f"\n### Multi-Step Frozen-Preconditioner Drift Sweep (N={N}, max_iters={max_iters})")
    print(
        "| dtype | K | Drift m | IPCG p50 (ms) | GPU Chol p50 (ms) | Speedup vs GPU Chol | "
        "Mean Iters | IPCG-Only Conv % | Fallbacks | Max Abs Err |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for dtype_str in dtypes:
        dtype = torch.float64 if dtype_str == "float64" else torch.float32
        tol = tol_f64 if dtype == torch.float64 else tol_f32
        is_f64 = dtype == torch.float64
        for K in ks:
            k_pad = align_k_pad(K, is_fp64=is_f64)
            for m in drift_steps:
                A, M0, rhs, x0, _ = _build_online_ridge_problem(
                    N, K, dtype, device, num_rank1_updates=m, seed=2026 + m
                )
                A_pad = torch.zeros((N, K, k_pad), dtype=dtype, device=device)
                M0_pad = torch.zeros((N, K, k_pad), dtype=dtype, device=device)
                rhs_pad = torch.zeros((N, k_pad), dtype=dtype, device=device)
                x0_pad = torch.zeros((N, k_pad), dtype=dtype, device=device)
                A_pad[:, :, :K] = A
                M0_pad[:, :, :K] = M0
                rhs_pad[:, :K] = rhs
                x0_pad[:, :K] = x0

                # Measure raw IPCG-only convergence before fallback (honest accounting!).
                res_raw = ipcg_solve_batch(
                    A_pad,
                    M0_pad,
                    rhs_pad,
                    x0_pad,
                    tol=tol,
                    max_iters=max_iters,
                    use_cuda_kernel=cuda_ipcg,
                    fallback_cholesky=False,
                )
                ipcg_only_conv_pct = float(res_raw.converged.float().mean().item() * 100.0)
                fallback_count = int((~res_raw.converged).sum().item())
                mean_iters = float(res_raw.iters.float().mean().item())

                last_res: Any = None

                def _run_ipcg(
                    a_p: Any = A_pad,
                    m_p: Any = M0_pad,
                    r_p: Any = rhs_pad,
                    x_p: Any = x0_pad,
                    tol_v: float = tol,
                ) -> None:
                    nonlocal last_res
                    last_res = ipcg_solve_batch(
                        a_p,
                        m_p,
                        r_p,
                        x_p,
                        tol=tol_v,
                        max_iters=max_iters,
                        use_cuda_kernel=cuda_ipcg,
                        fallback_cholesky=True,
                    )

                def _run_gpu_chol(a_m: Any = A, r_v: Any = rhs) -> None:
                    L = torch.linalg.cholesky(a_m)
                    _ = torch.cholesky_solve(r_v.unsqueeze(-1), L).squeeze(-1)

                t_ipcg = _time_cuda_fn(_run_ipcg, warmup, iters, device)
                t_chol = _time_cuda_fn(_run_gpu_chol, warmup, iters, device)

                L_ref = torch.linalg.cholesky(A)
                x_ref = torch.cholesky_solve(rhs.unsqueeze(-1), L_ref).squeeze(-1)
                max_err = float((last_res.theta - x_ref).abs().max().item())
                speedup = t_chol["p50_ms"] / max(t_ipcg["p50_ms"], 1e-9)

                sweep_rows.append(
                    {
                        "dtype": dtype_str,
                        "K": K,
                        "N": N,
                        "num_rank1_updates": m,
                        "max_iters": max_iters,
                        "ipcg": t_ipcg,
                        "gpu_cholesky": t_chol,
                        "speedup_vs_gpu_cholesky": float(speedup),
                        "mean_iters": mean_iters,
                        "ipcg_only_converged_pct": ipcg_only_conv_pct,
                        "fallback_count": fallback_count,
                        "post_fallback_converged_pct": float(
                            last_res.converged.float().mean().item() * 100.0
                        ),
                        "post_fallback_max_abs_err": max_err,
                    }
                )
                ipcg_p50 = t_ipcg["p50_ms"]
                chol_p50 = t_chol["p50_ms"]
                print(
                    f"| {dtype_str} | {K} | {m} | {ipcg_p50:.3f} | {chol_p50:.3f} | "
                    f"{speedup:.2f}x | {mean_iters:.2f} | {ipcg_only_conv_pct:.1f}% | "
                    f"{fallback_count} | {max_err:.2e} |"
                )
    return sweep_rows


def _run_shock_and_kappa_checks(device: Any, *, cuda_ipcg: bool) -> dict[str, Any]:
    K = 64
    N = 128
    # 1. 100x diagonal shock recovery: log raw IPCG convergence + fallback count explicitly!
    A_s, M0_s, rhs_s, x0_s, _ = _build_online_ridge_problem(
        N, K, torch.float64, device, shock_scale=100.0, seed=909
    )
    res_shock_raw = ipcg_solve_batch(
        A_s,
        M0_s,
        rhs_s,
        x0_s,
        tol=1e-8,
        max_iters=16,
        use_cuda_kernel=cuda_ipcg,
        fallback_cholesky=False,
    )
    shock_ipcg_conv_pct = float(res_shock_raw.converged.float().mean().item() * 100.0)
    shock_fallback_count = int((~res_shock_raw.converged).sum().item())
    shock_ipcg_mean_iters = float(res_shock_raw.iters.float().mean().item())

    res_shock_fb = ipcg_solve_batch(
        A_s,
        M0_s,
        rhs_s,
        x0_s,
        tol=1e-8,
        max_iters=16,
        use_cuda_kernel=cuda_ipcg,
        fallback_cholesky=True,
    )
    L_ref = torch.linalg.cholesky(A_s)
    ref_shock = torch.cholesky_solve(rhs_s.unsqueeze(-1), L_ref).squeeze(-1)
    shock_max_err = float((res_shock_fb.theta - ref_shock).abs().max().item())

    # 2. LIVE H200 kappa = 1e8 ill-conditioned Gram test (true residual + selective Cholesky).
    n_ill = 32
    gen_ill = torch.Generator(device="cpu").manual_seed(1080)
    Q_raw = torch.randn((n_ill, K, K), generator=gen_ill, dtype=torch.float64)
    Q_orth, _ = torch.linalg.qr(Q_raw)
    eigs = torch.logspace(0.0, -8.0, K, dtype=torch.float64)
    drift = 1.0 + 0.1 * torch.where(
        (torch.arange(K) % 2) == 0,
        torch.ones(K, dtype=torch.float64),
        -torch.ones(K, dtype=torch.float64),
    )
    A_ill = torch.bmm(Q_orth * eigs.view(1, 1, K), Q_orth.transpose(1, 2)).to(device=device)
    M_ill = torch.bmm(Q_orth * (1.0 / (eigs * drift)).view(1, 1, K), Q_orth.transpose(1, 2)).to(
        device=device
    )
    rhs_ill = torch.randn((n_ill, K), generator=gen_ill, dtype=torch.float64).to(device=device)
    x0_ill = torch.zeros((n_ill, K), dtype=torch.float64, device=device)

    L_ill_ref = torch.linalg.cholesky(A_ill)
    ref_ill = torch.cholesky_solve(rhs_ill.unsqueeze(-1), L_ill_ref).squeeze(-1)

    # 2a. FP64 with drifted preconditioner
    res_ill64 = ipcg_solve_batch(
        A_ill,
        M_ill,
        rhs_ill,
        x0_ill,
        tol=1e-8,
        max_iters=32,
        use_cuda_kernel=cuda_ipcg,
        fallback_cholesky=False,
    )
    res_ill64_vec = rhs_ill - torch.bmm(A_ill, res_ill64.theta.unsqueeze(-1)).squeeze(-1)
    ill64_true_rel_res = float(
        (
            torch.linalg.vector_norm(res_ill64_vec, dim=-1)
            / torch.linalg.vector_norm(rhs_ill, dim=-1).clamp(min=1e-30)
        )
        .max()
        .item()
    )
    ill64_max_err = float((res_ill64.theta - ref_ill).abs().max().item())

    # 2b. FP64 with Jacobi (M0=None) and tight budget: verify true-residual rejection + exact
    #     selective Cholesky recovery on GPU (~1e-10 relative error, ~1e-15 true residual).
    x0_ill64_guard = torch.full((n_ill, K), 1.25, dtype=torch.float64, device=device)
    res_ill64_jac_raw = ipcg_solve_batch(
        A_ill,
        None,
        rhs_ill,
        x0_ill64_guard,
        tol=1e-8,
        max_iters=16,
        use_cuda_kernel=cuda_ipcg,
        fallback_cholesky=False,
    )
    unconv64_mask = ~res_ill64_jac_raw.converged
    unconv64_untouched = (
        bool(torch.equal(res_ill64_jac_raw.theta[unconv64_mask], x0_ill64_guard[unconv64_mask]))
        if bool(unconv64_mask.any().item())
        else True
    )
    res_ill64_jac_fb = ipcg_solve_batch(
        A_ill,
        None,
        rhs_ill,
        x0_ill64_guard,
        tol=1e-8,
        max_iters=16,
        use_cuda_kernel=cuda_ipcg,
        fallback_cholesky=True,
    )
    res_ill64_fb_vec = rhs_ill - torch.bmm(A_ill, res_ill64_jac_fb.theta.unsqueeze(-1)).squeeze(-1)
    ill64_fb_rel_res = float(
        (
            torch.linalg.vector_norm(res_ill64_fb_vec, dim=-1)
            / torch.linalg.vector_norm(rhs_ill, dim=-1).clamp(min=1e-30)
        )
        .max()
        .item()
    )
    ill64_fb_rel_err = float(
        (
            torch.linalg.vector_norm(res_ill64_jac_fb.theta - ref_ill, dim=-1)
            / torch.linalg.vector_norm(ref_ill, dim=-1).clamp(min=1e-30)
        )
        .max()
        .item()
    )

    # 2c. FP32 with Jacobi (M0=None) and tight tol=1e-7: verify true-residual rejection + recovery
    #     (comparing against exact Cholesky of A_ill_f32 since kappa=1e8 > 1/eps_f32 ~ 8.4e6).
    A_ill_f32 = A_ill.to(torch.float32)
    rhs_ill_f32 = rhs_ill.to(torch.float32)
    x0_ill_f32 = torch.full((n_ill, K), 1.25, dtype=torch.float32, device=device)
    res_ill32_raw = ipcg_solve_batch(
        A_ill_f32,
        None,
        rhs_ill_f32,
        x0_ill_f32,
        tol=1e-7,
        max_iters=16,
        use_cuda_kernel=cuda_ipcg,
        fallback_cholesky=False,
    )
    unconv_mask = ~res_ill32_raw.converged
    unconv_untouched = (
        bool(torch.equal(res_ill32_raw.theta[unconv_mask], x0_ill_f32[unconv_mask]))
        if bool(unconv_mask.any().item())
        else True
    )
    res_ill32_fb = ipcg_solve_batch(
        A_ill_f32,
        None,
        rhs_ill_f32,
        x0_ill_f32,
        tol=1e-7,
        max_iters=16,
        use_cuda_kernel=cuda_ipcg,
        fallback_cholesky=True,
    )
    L_ill_f32_ref, info_f32 = torch.linalg.cholesky_ex(A_ill_f32.to(torch.float64))
    ref_of_f32_mat = torch.cholesky_solve(
        rhs_ill_f32.to(torch.float64).unsqueeze(-1), L_ill_f32_ref
    ).squeeze(-1)
    ill32_rel_err_vs_f32_mat = (
        float(
            (
                torch.linalg.vector_norm(
                    res_ill32_fb.theta.to(torch.float64) - ref_of_f32_mat, dim=-1
                )
                / torch.linalg.vector_norm(ref_of_f32_mat, dim=-1).clamp(min=1e-30)
            )
            .max()
            .item()
        )
        if bool((info_f32 == 0).all().item())
        else float("nan")
    )

    # 3. LIVE GPU FP32 extreme rhs prescaling check (1e25 and 1e-25)
    A_ext, M0_ext, rhs_ext_base, _, _ = _build_online_ridge_problem(
        16, K, torch.float32, device, seed=2027
    )
    x0_ext = torch.zeros((16, K), dtype=torch.float32, device=device)
    ext_checks: dict[str, Any] = {}
    for tag, sc in (("rhs_1e25", 1e25), ("rhs_1e_25", 1e-25)):
        rhs_sc = rhs_ext_base * sc
        res_ext = ipcg_solve_batch(
            A_ext,
            M0_ext,
            rhs_sc,
            x0_ext,
            tol=1e-5,
            max_iters=16,
            use_cuda_kernel=cuda_ipcg,
            fallback_cholesky=False,
        )
        L_e = torch.linalg.cholesky(A_ext.to(torch.float64))
        ref_e = torch.cholesky_solve(rhs_sc.to(torch.float64).unsqueeze(-1), L_e).squeeze(-1)
        rel_e = float(
            (
                torch.linalg.vector_norm(res_ext.theta.to(torch.float64) - ref_e, dim=-1)
                / torch.linalg.vector_norm(ref_e, dim=-1).clamp(min=1e-300)
            )
            .max()
            .item()
        )
        ext_checks[tag] = {
            "ipcg_only_all_converged": bool(res_ext.converged.all().item()),
            "max_rel_err": rel_e,
        }

    # 4. OnlineGpuRidgeEngine exact FP64-TC int64 Gram accumulation check
    eng = OnlineGpuRidgeEngine(num_series=64, feature_dim=32, ridge_lambda=1e-2, device=device)
    gen = torch.Generator(device="cpu").manual_seed(314)
    X_obs = torch.randn((64, 16, 32), generator=gen, dtype=torch.float64).to(device)
    y_obs = torch.randn((64, 16), generator=gen, dtype=torch.float64).to(device)
    eng.update(X_obs[:, :8, :], y_obs[:, :8])
    eng.update(X_obs[:, 8:, :], y_obs[:, 8:])
    sol = eng.solve(tol=1e-9, max_iters=16)

    # Compare against monolithic single-batch update for bit-exact associativity
    eng_mono = OnlineGpuRidgeEngine(num_series=64, feature_dim=32, ridge_lambda=1e-2, device=device)
    eng_mono.update(X_obs, y_obs)
    gram_bit_exact = bool(
        torch.equal(eng.G_i64, eng_mono.G_i64) and torch.equal(eng.b_i64, eng_mono.b_i64)
    )

    return {
        "shock_100x_ipcg_converged_pct": shock_ipcg_conv_pct,
        "shock_100x_fallback_count": shock_fallback_count,
        "shock_100x_ipcg_mean_iters": shock_ipcg_mean_iters,
        "shock_100x_post_fallback_all_converged": bool(res_shock_fb.converged.all().item()),
        "shock_100x_post_fallback_max_abs_err": shock_max_err,
        "kappa_1e8_gpu_fp64": {
            "ipcg_only_converged_pct": float(res_ill64.converged.float().mean().item() * 100.0),
            "mean_iters": float(res_ill64.iters.float().mean().item()),
            "max_true_rel_residual": ill64_true_rel_res,
            "max_abs_err": ill64_max_err,
        },
        "kappa_1e8_gpu_fp64_recovery": {
            "ipcg_only_converged_pct": float(
                res_ill64_jac_raw.converged.float().mean().item() * 100.0
            ),
            "fallback_count": int(unconv64_mask.sum().item()),
            "non_converged_theta_untouched": unconv64_untouched,
            "post_fallback_all_converged": bool(res_ill64_jac_fb.converged.all().item()),
            "post_fallback_max_true_rel_residual": ill64_fb_rel_res,
            "post_fallback_max_rel_err": ill64_fb_rel_err,
        },
        "kappa_1e8_gpu_fp32_recovery": {
            "ipcg_only_converged_pct": float(res_ill32_raw.converged.float().mean().item() * 100.0),
            "fallback_count": int(unconv_mask.sum().item()),
            "non_converged_theta_untouched": unconv_untouched,
            "post_fallback_all_converged": bool(res_ill32_fb.converged.all().item()),
            "post_fallback_max_rel_err_vs_f32_system": ill32_rel_err_vs_f32_mat,
        },
        "fp32_extreme_rhs_prescale": ext_checks,
        "int64_gram_bit_exact_associative": gram_bit_exact,
        "online_engine_all_converged": bool(sol.converged.all().item()),
        "online_engine_fallbacks": eng.total_fallbacks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live H200 SM90 ADAPT-IPCG benchmark.")
    parser.add_argument("--ns", type=int, nargs="+", default=[256, 1024, 4096, 16384, 32768])
    parser.add_argument("--ks", type=int, nargs="+", default=[64, 136])
    parser.add_argument("--dtypes", type=str, nargs="+", default=["float64", "float32"])
    parser.add_argument("--tol-f64", type=float, default=1e-8)
    parser.add_argument("--tol-f32", type=float, default=1e-5)
    parser.add_argument("--max-iters", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--cpu-max-n", type=int, default=256)
    parser.add_argument("--drift-n", type=int, default=4096)
    parser.add_argument("--drift-steps", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--output", type=str, default="results/bench_h200_ipcg.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError("bench_h200_ipcg requires PyTorch with CUDA.")

    device = torch.device("cuda:0")
    arch = detect_cuda_arch(device)
    cuda_ipcg = is_cuda_ipcg_available()

    rows: list[dict[str, Any]] = []
    print(f"Device: {torch.cuda.get_device_name(device)} | arch={arch} | cuda_ipcg={cuda_ipcg}")
    print(
        "| dtype | K | N | IPCG p50 (ms) | GPU Chol p50 (ms) | Speedup vs GPU Chol | "
        "CPU Est (ms) | Mean Iters | Conv % | Max Abs Err |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for dtype_str in args.dtypes:
        dtype = torch.float64 if dtype_str == "float64" else torch.float32
        tol = args.tol_f64 if dtype == torch.float64 else args.tol_f32
        is_f64 = dtype == torch.float64

        for K in args.ks:
            k_pad = align_k_pad(K, is_fp64=is_f64)
            for N in args.ns:
                A, M0, rhs, x0, _ = _build_online_ridge_problem(N, K, dtype, device)

                # Pre-pad once to measure pure zero-copy kernel speed.
                A_pad = torch.zeros((N, K, k_pad), dtype=dtype, device=device)
                M0_pad = torch.zeros((N, K, k_pad), dtype=dtype, device=device)
                rhs_pad = torch.zeros((N, k_pad), dtype=dtype, device=device)
                x0_pad = torch.zeros((N, k_pad), dtype=dtype, device=device)
                A_pad[:, :, :K] = A
                M0_pad[:, :, :K] = M0
                rhs_pad[:, :K] = rhs
                x0_pad[:, :K] = x0

                last_res: Any = None

                def _run_ipcg(
                    a_p: Any = A_pad,
                    m_p: Any = M0_pad,
                    r_p: Any = rhs_pad,
                    x_p: Any = x0_pad,
                    tol_v: float = tol,
                ) -> None:
                    nonlocal last_res
                    last_res = ipcg_solve_batch(
                        a_p,
                        m_p,
                        r_p,
                        x_p,
                        tol=tol_v,
                        max_iters=args.max_iters,
                        use_cuda_kernel=cuda_ipcg,
                        fallback_cholesky=True,
                    )

                def _run_gpu_chol(a_m: Any = A, r_v: Any = rhs) -> None:
                    L = torch.linalg.cholesky(a_m)
                    _ = torch.cholesky_solve(r_v.unsqueeze(-1), L).squeeze(-1)

                t_ipcg = _time_cuda_fn(_run_ipcg, args.warmup, args.iters, device)
                t_chol = _time_cuda_fn(_run_gpu_chol, args.warmup, args.iters, device)

                L_ref = torch.linalg.cholesky(A)
                x_ref = torch.cholesky_solve(rhs.unsqueeze(-1), L_ref).squeeze(-1)
                max_err = float((last_res.theta - x_ref).abs().max().item())
                conv_pct = float(last_res.converged.float().mean().item() * 100.0)
                mean_iters = float(last_res.iters.float().mean().item())

                cpu_stats = _time_cpu_cholesky(
                    A[: min(N, args.cpu_max_n)].cpu().numpy(),
                    rhs[: min(N, args.cpu_max_n)].cpu().numpy(),
                    args.cpu_max_n,
                    total_n=N,
                )
                speedup_gpu = t_chol["p50_ms"] / max(t_ipcg["p50_ms"], 1e-9)
                speedup_cpu = cpu_stats["estimated_full_ms"] / max(t_ipcg["p50_ms"], 1e-9)

                row = {
                    "dtype": dtype_str,
                    "K": K,
                    "K_PAD": k_pad,
                    "N": N,
                    "ipcg": t_ipcg,
                    "gpu_cholesky": t_chol,
                    "cpu_cholesky": cpu_stats,
                    "speedup_vs_gpu_cholesky": float(speedup_gpu),
                    "speedup_vs_cpu_est": float(speedup_cpu),
                    "mean_iters": mean_iters,
                    "converged_pct": conv_pct,
                    "max_abs_err": max_err,
                }
                rows.append(row)
                ipcg_ms = t_ipcg["p50_ms"]
                chol_ms = t_chol["p50_ms"]
                cpu_full_ms = cpu_stats["estimated_full_ms"]
                print(
                    f"| {dtype_str} | {K} | {N} | {ipcg_ms:.3f} | {chol_ms:.3f} | "
                    f"{speedup_gpu:.2f}x | {cpu_full_ms:.2f} ({speedup_cpu:.0f}x) | "
                    f"{mean_iters:.2f} | {conv_pct:.1f}% | {max_err:.2e} |"
                )

    drift_sweep = _run_multistep_drift_sweep(
        device,
        cuda_ipcg=cuda_ipcg,
        N=args.drift_n,
        ks=args.ks,
        drift_steps=args.drift_steps,
        dtypes=args.dtypes,
        tol_f64=args.tol_f64,
        tol_f32=args.tol_f32,
        max_iters=max(args.max_iters, 24),
        warmup=args.warmup,
        iters=args.iters,
    )
    robustness = _run_shock_and_kappa_checks(device, cuda_ipcg=cuda_ipcg)
    payload = {
        "device": torch.cuda.get_device_name(device),
        "arch": arch,
        "cuda_ipcg_available": cuda_ipcg,
        "results": rows,
        "multistep_drift_sweep": drift_sweep,
        "robustness_checks": robustness,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nSaved JSON report to {out_path}")


if __name__ == "__main__":
    main()
