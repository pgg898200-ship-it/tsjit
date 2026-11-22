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

"""Comprehensive CPU, ASan+UBSan, compiler-pass, distributed, and CUDA-gated tests."""

from __future__ import annotations

import ast
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - optional in pure-CPU environments
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

from tsjit.compiler.guards import (
    DEFAULT_RATIONAL_PERIODS,
    CausalFloat64DCStripper,
    StrictParityValidator,
    extract_rational_lag_features,
    scale_free_norm_floor,
)
from tsjit.cuda.loader import align_k_pad, ipcg_solve_batch, is_cuda_ipcg_available
from tsjit.cuda.scaled_fp16_triton import (
    SM90_GEMM_CONFIGS,
    SM90_VERIFIED_REPORTS,
    check_sm90_budget,
    crest_guarded_linear_numpy,
)
from tsjit.distributed.async_fuse import (
    DistributedStreamCoordinator,
    SeriesLaggedLOOTracker,
    async_fuse_monoid_numpy,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = torch is not None
HAS_CUDA = bool(torch is not None and torch.cuda.is_available())


# ---------------------------------------------------------------------------
# 1. C++20 ASan + UBSan CPU Math Parity Harness (All 4 Contracts)
# ---------------------------------------------------------------------------


def test_cpp_parity_harness_asan_ubsan(tmp_path: Path) -> None:
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("No C++20 compiler found in PATH")

    src = REPO_ROOT / "src" / "tsjit" / "cuda" / "cpu_parity_harness.cpp"
    assert src.is_file(), f"Missing {src}"
    exe = tmp_path / "tsjit_cpu_parity_harness"

    compile_cmd = [
        compiler,
        "-std=c++20",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-fsanitize=address,undefined",
        str(src),
        "-o",
        str(exe),
    ]
    c_proc = subprocess.run(compile_cmd, capture_output=True, text=True, check=False)
    assert c_proc.returncode == 0, f"C++ compile failed:\n{c_proc.stderr}"

    r_proc = subprocess.run([str(exe)], capture_output=True, text=True, check=False)
    assert r_proc.returncode == 0, (
        f"Parity harness failed:\nSTDOUT:\n{r_proc.stdout}\nSTDERR:\n{r_proc.stderr}"
    )
    for contract_idx in (1, 2, 3, 4):
        assert f"[PASS] Contract {contract_idx}:" in r_proc.stdout
    assert "ALL 4 CPU PARITY CONTRACTS PASSED" in r_proc.stdout


# ---------------------------------------------------------------------------
# 2. SM90 Budget Verifier
# ---------------------------------------------------------------------------


def test_sm90_budget_verifier() -> None:
    assert len(SM90_VERIFIED_REPORTS) == len(SM90_GEMM_CONFIGS)
    for cfg in SM90_GEMM_CONFIGS:
        rep = check_sm90_budget(*cfg, strict=True)
        assert rep.valid
        assert rep.smem_bytes <= 227 * 1024
        assert rep.elems_per_thr <= 32
        assert rep.est_regs <= 192

    # Excess SMEM (> 227 KB)
    bad_smem = check_sm90_budget(256, 256, 128, 4, 8, strict=False)
    assert not bad_smem.valid
    with pytest.raises(ValueError, match="smem_bytes"):
        check_sm90_budget(256, 256, 128, 4, 8, strict=True)

    # Excess elems_per_thr (> 32)
    with pytest.raises(ValueError, match="elems_per_thr"):
        check_sm90_budget(128, 128, 64, 3, 4, strict=True)

    # Non-positive args rejected
    with pytest.raises(ValueError):
        check_sm90_budget(0, 64, 64, 3, 4, strict=True)


# ---------------------------------------------------------------------------
# 3. Crest-Guarded Scaled-FP16 & FP16x3 Numerical Battery (CPU/NumPy)
# ---------------------------------------------------------------------------


def test_crest_guarded_precision_adversarial_numpy() -> None:
    rng = np.random.default_rng(2026)

    # 3a. Raw 1e38 near-overflow (without RMSNorm)
    x_hi = (1e38 * (0.1 + 0.05 * rng.standard_normal((16, 256)))).astype(np.float32)
    w_hi = (1e-3 * (0.2 + 0.05 * rng.standard_normal((16, 256)))).astype(np.float32)
    ref_hi = x_hi.astype(np.float64) @ w_hi.astype(np.float64).T
    y_hi_s16, _ = crest_guarded_linear_numpy(x_hi, w_hi, mode="scaled-fp16")
    y_hi_oz, _ = crest_guarded_linear_numpy(x_hi, w_hi, mode="fp16x3")
    assert np.all(np.isfinite(y_hi_s16)) and np.all(np.isfinite(y_hi_oz))
    assert np.max(np.abs(y_hi_s16.astype(np.float64) - ref_hi) / np.abs(ref_hi)) < 2e-3
    assert np.max(np.abs(y_hi_oz.astype(np.float64) - ref_hi) / np.abs(ref_hi)) < 1e-4

    # 3b. Raw 1e-40 subnormals (zero 1/sx -> +Inf blowup)
    x_sub = (1e-40 * (1.0 + 0.3 * rng.standard_normal((16, 256)))).astype(np.float32)
    w_sub = (1e12 * (0.5 + 0.2 * rng.standard_normal((16, 256)))).astype(np.float32)
    ref_sub = x_sub.astype(np.float64) @ w_sub.astype(np.float64).T
    y_sub_oz, _ = crest_guarded_linear_numpy(x_sub, w_sub, mode="fp16x3")
    assert np.all(np.isfinite(y_sub_oz))
    assert np.max(np.abs(y_sub_oz.astype(np.float64) - ref_sub) / np.abs(ref_sub)) < 1e-4

    # 3c. ex + ew > 127 zero-cancelled dot product -> exact 0.0 (never 0*Inf = NaN!)
    x_zc = np.array([[1e25, -1e25, 1e25, -1e25]], dtype=np.float32)
    w_zc = np.array([[1e25, 1e25, 1e25, 1e25]], dtype=np.float32)
    y_zc, _ = crest_guarded_linear_numpy(x_zc, w_zc, mode="scaled-fp16")
    assert np.isfinite(y_zc).all() and float(y_zc[0, 0]) == 0.0

    # 3d. 1000x weight & activation outliers with cancellation (K=512 -> crest=16 > 12)
    K_out = 512
    x_out = (0.5 + 0.1 * rng.standard_normal((8, K_out))).astype(np.float32)
    w_out = (0.5 + 0.1 * rng.standard_normal((8, K_out))).astype(np.float32)
    x_out[:, 0] = 1000.1234
    x_out[:, 1] = -1000.1234
    w_out[:, 0] = 1000.5678
    w_out[:, 1] = 1000.5678
    ref_out = x_out.astype(np.float64) @ w_out.astype(np.float64).T
    y_auto_w, meta_w = crest_guarded_linear_numpy(x_out, w_out, mode="auto")
    assert meta_w["tier"] == "fp16x3-3gemm" and meta_w["escalated_w"]
    assert np.max(np.abs(y_auto_w.astype(np.float64) - ref_out) / np.abs(ref_out)) < 1e-3

    # 3e. Activation-only 1000x outlier -> 2-GEMM act-escalated (zero W1 VRAM)
    w_norm = (0.5 + 0.1 * rng.standard_normal((8, K_out))).astype(np.float32)
    w_norm[:, 0] = 1.0
    w_norm[:, 1] = 1.0
    ref_act = x_out.astype(np.float64) @ w_norm.astype(np.float64).T
    y_auto_x, meta_x = crest_guarded_linear_numpy(x_out, w_norm, mode="auto")
    assert (
        meta_x["tier"] == "act-escalated-2gemm"
        and (not meta_x["escalated_w"])
        and meta_x["escalated_x"]
    )
    assert np.max(np.abs(y_auto_x.astype(np.float64) - ref_act) / np.abs(ref_act)) < 5e-3

    # 3f. Non-pow2 shapes K=688, 1368, 3072 and N % 4 != 0
    for K_np2, N_np2 in ((688, 17), (1368, 689), (3072, 13)):
        xa = rng.standard_normal((7, K_np2)).astype(np.float32)
        wa = rng.standard_normal((N_np2, K_np2)).astype(np.float32)
        ba = rng.standard_normal((N_np2,)).astype(np.float32)
        ya, _ = crest_guarded_linear_numpy(xa, wa, ba, mode="fp16x3")
        refa = xa.astype(np.float64) @ wa.astype(np.float64).T + ba.astype(np.float64)[None, :]
        assert ya.shape == (7, N_np2)
        assert (
            np.max(np.abs(ya.astype(np.float64) - refa)) / max(float(np.max(np.abs(refa))), 1.0)
            < 1e-4
        )

    # 3g. NaN rows (graph_safe=True vs graph_safe=False with > 16 bad rows) + zero row with bias
    x_mix = rng.standard_normal((24, 64)).astype(np.float32)
    w_mix = rng.standard_normal((12, 64)).astype(np.float32)
    b_mix = rng.standard_normal((12,)).astype(np.float32)
    x_mix[0] = 0.0  # zero row
    x_mix[1:20, 3] = np.nan  # 19 bad rows -> exercises 16-row chunking!
    y_gs, _ = crest_guarded_linear_numpy(x_mix, w_mix, b_mix, mode="auto", graph_safe=True)
    y_ng, _ = crest_guarded_linear_numpy(x_mix, w_mix, b_mix, mode="auto", graph_safe=False)
    assert np.allclose(y_gs[0], b_mix, atol=1e-6)
    assert np.all(np.isnan(y_gs[1:20]))
    assert np.all(np.isnan(y_ng[1:20]))
    assert np.all(np.isfinite(y_gs[20:]))

    # 3h. Boundary cases: M=0, N=0, K=0, smallest FP32 subnormal 2^-149, and IEEE 0 * Inf = NaN
    y_m0, _ = crest_guarded_linear_numpy(
        np.zeros((0, 16), np.float32), np.ones((8, 16), np.float32)
    )
    assert y_m0.shape == (0, 8)
    y_n0, _ = crest_guarded_linear_numpy(
        np.ones((3, 16), np.float32), np.zeros((0, 16), np.float32)
    )
    assert y_n0.shape == (3, 0)
    y_k0, _ = crest_guarded_linear_numpy(
        np.zeros((3, 0), np.float32), np.ones((5, 0), np.float32), np.arange(5, dtype=np.float32)
    )
    assert y_k0.shape == (3, 5) and np.allclose(y_k0[0], np.arange(5, dtype=np.float32))

    x_min_sub = np.full((2, 16), np.float32(2.0**-149), dtype=np.float32)
    w_big = np.full((4, 16), np.float32(1e30), dtype=np.float32)
    y_min_sub, _ = crest_guarded_linear_numpy(x_min_sub, w_big, mode="fp16x3")
    ref_min_sub = x_min_sub.astype(np.float64) @ w_big.astype(np.float64).T
    assert np.all(np.isfinite(y_min_sub))
    assert np.max(np.abs(y_min_sub.astype(np.float64) - ref_min_sub) / np.abs(ref_min_sub)) < 1e-4

    x_zero = np.zeros((2, 8), dtype=np.float32)
    w_inf = np.ones((4, 8), dtype=np.float32)
    w_inf[1, 0] = np.inf
    y_zinf, _ = crest_guarded_linear_numpy(x_zero, w_inf, graph_safe=True)
    assert np.all(np.isnan(y_zinf[:, 1])) and np.all(y_zinf[:, 0] == 0.0)


# ---------------------------------------------------------------------------
# 4. ADAPT-IPCG NumPy Contracts & Alignment Helpers
# ---------------------------------------------------------------------------


def test_ipcg_solve_batch_numpy_contracts() -> None:
    assert align_k_pad(1, is_fp64=True) == 2
    assert align_k_pad(1, is_fp64=False) == 4
    assert align_k_pad(136, is_fp64=True) == 136
    assert align_k_pad(136, is_fp64=False) == 136
    with pytest.raises(ValueError):
        align_k_pad(0, is_fp64=True)
    with pytest.raises(ValueError):
        align_k_pad(257, is_fp64=True)

    rng = np.random.default_rng(123)
    N, K = 6, 16
    B = rng.standard_normal((N, K, K)) / np.sqrt(K)
    A = np.einsum("nji,njk->nik", B, B) + 0.5 * np.eye(K)[None, :, :]
    M0 = np.linalg.inv(A)
    # Perturb A slightly (online rank-1 update)
    u = 0.2 * rng.standard_normal((N, K))
    A_upd = A + u[:, :, None] * u[:, None, :]
    rhs = rng.standard_normal((N, K))
    x_init = np.zeros((N, K), dtype=np.float64)

    # Inject NaN in system 4 rhs, and 100x shock in system 5
    rhs[4, 2] = np.nan
    A_upd[5] += 100.0 * np.eye(K)
    x_init[4] = 3.25

    res = ipcg_solve_batch(A_upd, M0, rhs, x_init, tol=1e-9, max_iters=16, fallback_cholesky=True)
    assert bool(res.converged[:4].all())
    assert not bool(res.converged[4])
    assert np.array_equal(res.theta[4], x_init[4])  # untouched!
    assert bool(res.converged[5])  # shock recovered cleanly (via IPCG or selective Cholesky)
    for n in (0, 1, 2, 3, 5):
        ref = np.linalg.solve(A_upd[n], rhs[n])
        assert np.max(np.abs(res.theta[n] - ref)) < 1e-8


# ---------------------------------------------------------------------------
# 5. CausalFloat64DCStripper, scale_free_norm_floor, Parity & Rational Lags
# ---------------------------------------------------------------------------


def test_causal_dc_stripper_2p22e10_and_causality() -> None:
    t = np.linspace(0.0, 10.0, 64, dtype=np.float64)
    signal = np.sin(t) + 0.25 * np.cos(2.0 * t)
    x = 2.22e10 + signal[None, :]

    # Demonstrate FP32 ULP collapse without stripping
    naive_f32 = x.astype(np.float32)
    assert float(np.std(naive_f32)) == 0.0

    # With CausalFloat64DCStripper: full dynamics preserved
    stripper = CausalFloat64DCStripper(enabled=True)
    x_s, dc = stripper.strip(x)
    assert x_s.dtype == np.float32
    assert dc.dtype == np.float64
    assert float(np.std(x_s)) > 0.5
    restored = stripper.restore(x_s, dc)
    assert np.max(np.abs(restored - x)) < 1e-5

    # Trailing NaNs -> causal anchor is last finite value; all-NaN -> anchor 0.0
    x_nan = np.array(
        [
            [2.22e10 + 1.0, 2.22e10 + 3.5, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan],
        ],
        dtype=np.float64,
    )
    xs_n, dc_n = stripper.strip(x_nan)
    assert dc_n[0, 0] == 2.22e10 + 3.5
    assert dc_n[1, 0] == 0.0
    assert np.isclose(xs_n[0, 0], -2.5, atol=1e-6)
    assert np.isnan(xs_n[0, 2])

    # Broadcast restoration onto (B, Q, H), 1D (B,), and (Q, B, H) with batch_dim=1
    q_dummy = np.zeros((2, 9, 16), dtype=np.float32)
    q_rest = stripper.restore(q_dummy, dc_n)
    assert q_rest.shape == (2, 9, 16)
    assert np.all(q_rest[0] == 2.22e10 + 3.5)

    q_1d = np.array([1.0, 2.0], dtype=np.float32)
    q_1d_rest = stripper.restore(q_1d, dc_n)
    assert q_1d_rest.shape == (2,)
    assert q_1d_rest[0] == (2.22e10 + 3.5) + 1.0 and q_1d_rest[1] == 2.0

    q_qbh = np.zeros((9, 2, 8), dtype=np.float32)
    q_qbh_rest = stripper.restore(q_qbh, dc_n, batch_dim=1)
    assert q_qbh_rest.shape == (9, 2, 8)
    assert np.all(q_qbh_rest[:, 0, :] == 2.22e10 + 3.5)
    assert np.all(q_qbh_rest[:, 1, :] == 0.0)


def test_guards_norm_floor_parity_and_rational_lags() -> None:
    # Scale-free across 24 orders of magnitude
    for mag in (1e-12, 1.0, 1e12):
        x = np.array([[mag, 2.0 * mag, 3.0 * mag]], dtype=np.float64)
        sc = scale_free_norm_floor(x, loc=np.array([2.0 * mag]), scale=np.array([0.0]))
        assert np.isclose(sc[0], 1e-4 * 2.0 * mag, rtol=1e-9)

    # All-zero / all-NaN falls back to 1.0
    sc_zero = scale_free_norm_floor(np.zeros((1, 8)), loc=np.array([0.0]), scale=np.array([0.0]))
    assert float(sc_zero[0]) == 1.0
    sc_nan = scale_free_norm_floor(
        np.full((1, 8), np.nan), loc=np.array([np.nan]), scale=np.array([np.nan])
    )
    assert float(sc_nan[0]) == 1.0

    # StrictParityValidator catches drift > 0.05 and NaN mismatches
    rng = np.random.default_rng(7)
    x_val = rng.standard_normal((16, 32))

    def _ref_fn(arr: np.ndarray) -> np.ndarray:
        return arr[:, -8:]

    def _good_fn(arr: np.ndarray) -> np.ndarray:
        return arr[:, -8:] + 0.01

    def _bad_fn(arr: np.ndarray) -> np.ndarray:
        return arr[:, -8:] + 0.5

    validator = StrictParityValidator(tol=0.05)
    rep_ok = validator.audit(_ref_fn, _good_fn, x_val)
    assert rep_ok.passed and rep_ok.num_failed == 0
    rep_bad = validator.audit(_ref_fn, _bad_fn, x_val)
    assert (not rep_bad.passed) and rep_bad.num_failed == 16
    rep_empty = validator.audit(_ref_fn, _good_fn, np.zeros((0, 32), dtype=np.float64))
    assert rep_empty.passed and rep_empty.num_series == 0

    # IEEE rational-lag extraction (5/3, 7/3, 49/25) + L=1 and L=0 guards
    t = np.arange(32, dtype=np.float64)[None, :]  # linear ramp x[i] = i
    lag_vals, lag_ratios = extract_rational_lag_features(t, periods=DEFAULT_RATIONAL_PERIODS)
    expected_lags = np.array(
        [31.0 - num / den for num, den in DEFAULT_RATIONAL_PERIODS], dtype=np.float32
    )
    assert np.allclose(lag_vals[0], expected_lags, atol=1e-5)
    assert np.all(np.isfinite(lag_ratios))
    lag_l1, ratio_l1 = extract_rational_lag_features(np.array([[42.0]], dtype=np.float64))
    assert np.allclose(lag_l1, 42.0) and np.allclose(ratio_l1, 0.0)
    with pytest.raises(ValueError, match="L must be >= 1"):
        extract_rational_lag_features(np.zeros((2, 0), dtype=np.float64))


def test_c5_breaker_dc_2p22e10_error_must_fail_strict_parity() -> None:
    """C5 Breaker regression: +2048.0 error on N(0, 1) + 2.22e10 MUST FAIL StrictParityValidator."""
    rng = np.random.default_rng(202605)
    dc = 2.22e10
    x_dc = dc + rng.standard_normal((16, 512)).astype(np.float64)

    # Confirm scale_free_norm_floor does NOT inflate positive dispersion by 1e-4 * |loc|.
    denom_pos = scale_free_norm_floor(
        x_dc[0], loc=np.array(dc), scale=np.array(1.0, dtype=np.float64)
    )
    assert np.isclose(float(denom_pos), 1.0, rtol=1e-12)

    # Strictly constant series (scale == 0.0) still uses the 1e-4 * |loc| fallback.
    denom_const = scale_free_norm_floor(
        np.full((1, 16), dc, dtype=np.float64),
        loc=np.array([dc], dtype=np.float64),
        scale=np.array([0.0], dtype=np.float64),
    )
    assert np.isclose(float(denom_const[0]), 1e-4 * dc, rtol=1e-12)

    # Higher-rank x (2D/3D) with 0D scalar scale or empty x must NOT crash or loop infinitely!
    denom_2d_scalar = scale_free_norm_floor(np.full((4, 16), 10.0), loc=10.0, scale=0.0)
    assert np.isclose(float(denom_2d_scalar), 1e-3, rtol=1e-12)
    denom_empty = scale_free_norm_floor(np.zeros((0, 16), dtype=np.float64), loc=0.0, scale=0.0)
    assert float(denom_empty) == 1.0

    def _ref_dc(arr: np.ndarray) -> np.ndarray:
        return arr[:, -16:]

    def _corrupted_plus_2048(arr: np.ndarray) -> np.ndarray:
        return arr[:, -16:] + 2048.0

    def _tiny_good_fn(arr: np.ndarray) -> np.ndarray:
        return arr[:, -16:] + 0.01

    validator = StrictParityValidator(tol=0.05)
    rep_fail = validator.audit(_ref_dc, _corrupted_plus_2048, x_dc)
    assert not rep_fail.passed
    assert rep_fail.num_failed == 16
    assert rep_fail.max_scaled_err > 1500.0  # ~2048.0 / ~1.0 >> 0.05!

    # A genuinely small 0.01 error on N(0, 1) + 2.22e10 still passes cleanly.
    rep_ok = validator.audit(_ref_dc, _tiny_good_fn, x_dc)
    assert rep_ok.passed and rep_ok.num_failed == 0
    assert rep_ok.max_scaled_err < 0.02


def test_c6_c13_i1_i3_i5_i75_breaker_hardening_cpu() -> None:
    """Pure-CPU/NumPy Breaker regression tests covering C6, C13, I1, I3, I5(c), and I7.5."""
    import importlib.util
    import threading

    from tsjit.compiler.api import normalize_precision, resolve_family_precision
    from tsjit.compiler.guards import normalize_precision_alias
    from tsjit.cuda.loader import _LIB_LOAD_LOCK, _MAX_Q_BOUND

    # --- C6: Family-aware a10-auto zero-fail precision contract ---
    assert normalize_precision("strict", "exact") == "fp16x3"
    assert normalize_precision("fast-tf32", "exact") == "tf32"
    assert normalize_precision("fast_tf32", "fast") == "tf32"
    assert normalize_precision_alias("a10-auto") == "auto"
    assert normalize_precision_alias("strict") == "fp16x3"
    assert normalize_precision_alias("fast-tf32") == "tf32"
    with pytest.raises(ValueError, match="Unsupported precision"):
        normalize_precision_alias("invalid-prec")
    assert resolve_family_precision("a10-auto", "exact", "toto2") == "fp16x3"
    assert resolve_family_precision("auto", "exact", "toto2") == "fp16x3"
    assert resolve_family_precision("fast-tf32", "exact", "toto2") == "tf32"
    assert resolve_family_precision("a10-auto", "fast", "toto2") == "auto"
    assert resolve_family_precision("a10-auto", "exact", "chronos-2") == "auto"
    assert resolve_family_precision("a10-auto", "exact", "chronos-bolt") == "auto"
    assert resolve_family_precision("strict", "exact", "chronos-2") == "fp16x3"

    # --- C8 constant & I5(c) RLock ---
    assert _MAX_Q_BOUND == 32767
    assert isinstance(_LIB_LOAD_LOCK, type(threading.RLock()))

    # --- I1(4): Extreme FP32 rhs (1e25 and 1e-25) never overflow/underflow rhs_n2 in NumPy ---
    rng = np.random.default_rng(202606)
    K = 16
    B_m = rng.standard_normal((2, K, K)) / np.sqrt(K)
    A_f64 = np.einsum("nji,njk->nik", B_m, B_m) + 0.5 * np.eye(K)[None, :, :]
    M0_f64 = np.linalg.inv(A_f64)
    A_f32 = A_f64.astype(np.float32)
    M0_f32 = M0_f64.astype(np.float32)
    rhs_unit = rng.standard_normal((2, K)).astype(np.float32)
    x0_zero = np.zeros((2, K), dtype=np.float32)
    for sc in (np.float32(1e25), np.float32(1e-25)):
        rhs_sc = (rhs_unit * sc).astype(np.float32)
        x0_use = np.full((2, K), 1e10, dtype=np.float32) if sc < 1.0 else x0_zero
        res = ipcg_solve_batch(
            A_f32, M0_f32, rhs_sc, x0_use, tol=1e-5, max_iters=16, fallback_cholesky=False
        )
        assert bool(res.converged.all()), f"FP32 IPCG failed on rhs scale {sc}"
        for n in range(2):
            ref = np.linalg.solve(A_f64[n], rhs_sc[n].astype(np.float64))
            rel = np.max(np.abs(res.theta[n].astype(np.float64) - ref)) / max(
                float(np.max(np.abs(ref))), 1e-300
            )
            assert rel < 1e-4

    # Stale warm-start x_init = 1e20 on unit rhs must recover via zero-init retry,
    # while NaN in A with zero rhs MUST stay rejected (0 * NaN == NaN).
    res_stale = ipcg_solve_batch(
        A_f32,
        M0_f32,
        rhs_unit,
        np.full((2, K), 1e20, dtype=np.float32),
        tol=1e-5,
        max_iters=16,
        fallback_cholesky=False,
    )
    assert bool(res_stale.converged.all())
    A_nan_z = A_f64[:1].copy()
    A_nan_z[0, 0, 0] = np.nan
    res_nan_z = ipcg_solve_batch(
        A_nan_z,
        M0_f64[:1],
        np.zeros((1, K), dtype=np.float64),
        np.ones((1, K), dtype=np.float64),
        tol=1e-8,
        max_iters=16,
        fallback_cholesky=False,
    )
    assert not bool(res_nan_z.converged[0])

    # --- C13: bench_h200_ipcg._time_cpu_cholesky scales cleanly with total_n ---
    ipcg_path = REPO_ROOT / "benchmarks" / "bench_h200_ipcg.py"
    spec_i = importlib.util.spec_from_file_location("bench_h200_ipcg", ipcg_path)
    assert spec_i is not None and spec_i.loader is not None
    ipcg_mod = importlib.util.module_from_spec(spec_i)
    spec_i.loader.exec_module(ipcg_mod)
    sub_A = np.broadcast_to(np.eye(8, dtype=np.float64), (4, 8, 8)).copy()
    sub_b = np.ones((4, 8), dtype=np.float64)
    cpu_stat = ipcg_mod._time_cpu_cholesky(sub_A, sub_b, cpu_max_n=4, total_n=32768)
    assert cpu_stat["total_n"] == 32768
    assert cpu_stat["sampled_n"] == 4
    expected_full = cpu_stat["sampled_ms"] * (32768.0 / 4.0)
    assert np.isclose(cpu_stat["estimated_full_ms"], expected_full, rtol=1e-9)

    # --- I3: bench_h200_gifteval_stream reports both c11_arrival_only & full_tick_e2e ---
    g_path = REPO_ROOT / "benchmarks" / "bench_h200_gifteval_stream.py"
    spec_g = importlib.util.spec_from_file_location("bench_h200_gifteval_stream_i3", g_path)
    assert spec_g is not None and spec_g.loader is not None
    g_mod = importlib.util.module_from_spec(spec_g)
    spec_g.loader.exec_module(g_mod)
    t_s = np.arange(96, dtype=np.float64)
    ser = 10.0 + np.sin(2.0 * np.pi * t_s / 16.0)
    z9 = np.linspace(-1.28, 1.28, 9, dtype=np.float64)

    def _mock_bb(xb: np.ndarray) -> np.ndarray:
        return np.broadcast_to(xb[:, -1:, None] + z9[None, :, None], (xb.shape[0], 9, 8)).copy()

    m_i3 = g_mod._run_gap_async_stream_single_series(
        ser, _mock_bb, context_length=48, horizon=8, ridge_dim=8, device=None, max_gap_ticks=4
    )
    assert "c11_arrival_only_p50_us" in m_i3 and "full_tick_e2e_p50_us" in m_i3
    assert m_i3["c11_arrival_only_p50_us"] > 0.0
    assert m_i3["full_tick_e2e_p50_us"] >= m_i3["c11_arrival_only_p50_us"]
    assert "e_ville_final" in m_i3 and np.isfinite(m_i3["e_ville_final"])
    agg_i3 = g_mod._aggregate_stream_runs([m_i3])
    assert "e_ville_final_mean" in agg_i3 and "e_ville_final_max" in agg_i3

    # --- I7.5: SeriesLaggedLOOTracker duplicate IDs, OOB groups, 0-D ndim, reset_for_migration ---
    tracker = SeriesLaggedLOOTracker(monoid_dim=4, num_groups=4)
    sids_ok = np.array([10, 20, 30], dtype=np.int64)
    gids_ok = np.array([0, 1, 3], dtype=np.int64)
    rows_ok = np.arange(12, dtype=np.float32).reshape(3, 4)
    tracker.update(sids_ok, gids_ok, rows_ok)

    # 0-D scalar series_ids must raise ValueError cleanly (never IndexError)
    with pytest.raises(ValueError, match="1D series_ids"):
        tracker.update(np.int64(10), np.int64(0), rows_ok[:1])

    # Duplicate series_ids rejected in both update() and lookup_own_prev()
    with pytest.raises(ValueError, match="Duplicate series_ids"):
        tracker.update(np.array([10, 10, 20]), gids_ok, rows_ok)
    with pytest.raises(ValueError, match="Duplicate series_ids"):
        tracker.lookup_own_prev(np.array([10, 10, 20]), gids_ok)

    # Out-of-bounds group_ids rejected (< 0 or >= num_groups)
    with pytest.raises(ValueError, match="out of bounds"):
        tracker.lookup_own_prev(sids_ok, np.array([0, 4, 1]))
    with pytest.raises(ValueError, match="out of bounds"):
        tracker.update(sids_ok, np.array([-1, 0, 1]), rows_ok)

    # Selective reset_for_migration drops ONLY migrated IDs; untouched residents stay exact!
    tracker.reset_for_migration(migrated_series_ids=[20])
    own_after = tracker.lookup_own_prev(sids_ok, gids_ok)
    assert np.array_equal(own_after[0], rows_ok[0])  # 10 retained
    assert np.all(own_after[1] == 0.0)  # 20 migrated -> zeroed!
    assert np.array_equal(own_after[2], rows_ok[2])  # 30 retained

    # Full reset_for_migration clears all history
    tracker.reset_for_migration()
    assert np.all(tracker.lookup_own_prev(sids_ok, gids_ok) == 0.0)

    # async_fuse_monoid_numpy rejects negative / out-of-bounds group_ids
    with pytest.raises(ValueError, match="out of bounds"):
        async_fuse_monoid_numpy(
            np.zeros((2, 4), np.float32),
            np.zeros((2, 4), np.float32),
            np.ones(2, np.float32),
            np.array([-1, 0], np.int64),
            num_groups=2,
        )


# ---------------------------------------------------------------------------
# 6. AsyncFuse Exact Sharded-vs-Central Parity & Dynamic Series-ID LOO Guard
# ---------------------------------------------------------------------------


def test_async_fuse_parity_and_dynamic_loo_tracker() -> None:
    rng = np.random.default_rng(202604)
    D_f = 32
    rho, kappa = 0.25, 1.5

    # W=4 skewed partition [1, 3, 12, 48] = 64 series across G=6 groups
    N, G = 64, 6
    h = rng.standard_normal((N, D_f)).astype(np.float32)
    phi = rng.standard_normal((N, D_f)).astype(np.float32)
    a = rng.uniform(0.1, 5.0, size=N).astype(np.float32)
    gids = np.array([0 if i == 0 else 1 + (i % (G - 1)) for i in range(N)], dtype=np.int64)

    out_central, _, _ = async_fuse_monoid_numpy(h, phi, a, gids, G, rho=rho, kappa=kappa)

    splits = [slice(0, 1), slice(1, 4), slice(4, 16), slice(16, 64)]
    local_tables = []
    for sl in splits:
        _, loc_tab, _ = async_fuse_monoid_numpy(
            h[sl], phi[sl], a[sl], gids[sl], G, rho=rho, kappa=kappa
        )
        local_tables.append(loc_tab)
    global_tab = sum(local_tables)

    out_shards = []
    for sl in splits:
        o_s, _, _ = async_fuse_monoid_numpy(
            h[sl], phi[sl], a[sl], gids[sl], G, rho=rho, kappa=kappa, global_monoid=global_tab
        )
        out_shards.append(o_s)
    out_sharded = np.concatenate(out_shards, axis=0)
    assert float(np.max(np.abs(out_sharded - out_central))) <= 1e-6
    # Singleton group 0 (series 0) -> exact identity h[0]
    assert np.array_equal(out_sharded[0], h[0])

    # Dynamic Series-ID-Keyed Lagged LOO Tracker test
    tracker = SeriesLaggedLOOTracker(monoid_dim=4)
    sids_t0 = np.array([100, 200, 300, 400], dtype=np.int64)
    gids_t0 = np.array([1, 1, 2, 2], dtype=np.int64)
    rows_t0 = np.array(
        [
            [1.0, 0.0, 1.0, 1.0],
            [2.0, 0.0, 2.0, 1.4],
            [3.0, 0.0, 3.0, 1.7],
            [4.0, 0.0, 4.0, 2.0],
        ],
        dtype=np.float32,
    )
    # Cycle 0: empty history -> all zeros
    own0 = tracker.lookup_own_prev(sids_t0, gids_t0)
    assert np.all(own0 == 0.0)
    tracker.update(sids_t0, gids_t0, rows_t0)

    # Cycle 1: reorder (300, 100), regroup (200: 1 -> 2), join (500), drop (400)
    sids_t1 = np.array([300, 200, 500, 100], dtype=np.int64)
    gids_t1 = np.array([2, 2, 1, 1], dtype=np.int64)
    own1 = tracker.lookup_own_prev(sids_t1, gids_t1)
    assert np.array_equal(own1[0], rows_t0[2])  # series 300 same group 2 -> exact match!
    assert np.all(own1[1] == 0.0)  # series 200 changed group 1->2 -> ZERO!
    assert np.all(own1[2] == 0.0)  # series 500 newly joined -> ZERO!
    assert np.array_equal(own1[3], rows_t0[0])  # series 100 reordered, same group 1 -> exact match!

    # DistributedStreamCoordinator payload strictly < 2 KB
    coord = DistributedStreamCoordinator(num_groups=16)
    assert coord.payload_bytes < 2048
    coord.submit_async(
        np.ones(16), np.zeros(16), np.full(16, 10.0), np.zeros(16), np.ones(16), np.zeros(16)
    )
    st = coord.poll_state()
    assert np.allclose(st.miscov_rate_80, 0.1)
    with pytest.raises(ValueError, match="2048"):
        DistributedStreamCoordinator(num_groups=100)


# ---------------------------------------------------------------------------
# 7. Syntax Verification of All Owned Python Files
# ---------------------------------------------------------------------------


def test_all_owned_python_files_parse_cleanly() -> None:
    owned_py = [
        "src/tsjit/cuda/__init__.py",
        "src/tsjit/cuda/loader.py",
        "src/tsjit/cuda/scaled_fp16_triton.py",
        "src/tsjit/compiler/__init__.py",
        "src/tsjit/compiler/guards.py",
        "src/tsjit/compiler/graph_runner.py",
        "src/tsjit/compiler/toto2_pass.py",
        "src/tsjit/compiler/chronos_pass.py",
        "src/tsjit/compiler/generic_pass.py",
        "src/tsjit/compiler/api.py",
        "src/tsjit/distributed/__init__.py",
        "src/tsjit/distributed/async_fuse.py",
        "src/tsjit/distributed/bench_4gpu.py",
        "benchmarks/bench_h200_ipcg.py",
        "benchmarks/bench_h200_precision.py",
        "benchmarks/bench_h200_tsfm.py",
        "benchmarks/bench_h200_gifteval_stream.py",
    ]
    for rel in owned_py:
        path = REPO_ROOT / rel
        assert path.is_file(), f"Missing owned file: {rel}"
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path))
        compile(source, str(path), "exec")


def test_cuda_kernel_cpp20_static_syntax(tmp_path: Path) -> None:
    """Verify tsjit_ipcg_sm90.cu passes strict C++20 syntax checks with and without cuBLAS."""
    import re

    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("No C++20 compiler found in PATH")

    stub_cuda_header = """
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#define __device__
#define __global__
#define __host__
#define __forceinline__ inline
#define __shared__
#define __restrict__ __restrict
#define __launch_bounds__(...)
struct dim3 {
  unsigned int x, y, z;
  dim3(unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1) : x(x_), y(y_), z(z_) {}
};
struct uint3 { unsigned int x, y, z; };
struct alignas(16) float4 { float x, y, z, w; };
struct alignas(16) double2 { double x, y; };
struct alignas(16) uint4 { unsigned int x, y, z, w; };
static thread_local uint3 threadIdx{0, 0, 0}, blockIdx{0, 0, 0};
static thread_local uint3 blockDim{128, 1, 1}, gridDim{1, 1, 1};
typedef void* cudaStream_t;
enum cudaError_t { cudaSuccess = 0 };
enum cudaDeviceAttr { cudaDevAttrMultiProcessorCount = 16 };
enum cudaStreamCaptureStatus {
  cudaStreamCaptureStatusNone = 0,
  cudaStreamCaptureStatusActive = 1
};
inline cudaError_t cudaGetDevice(int* d) { *d = 0; return cudaSuccess; }
inline cudaError_t cudaDeviceGetAttribute(int* v, cudaDeviceAttr, int) {
  *v = 132;
  return cudaSuccess;
}
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline cudaError_t cudaMalloc(void** p, size_t) { *p = nullptr; return cudaSuccess; }
inline cudaError_t cudaFree(void*) { return cudaSuccess; }
inline cudaError_t cudaMemsetAsync(void*, int, size_t, cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamIsCapturing(cudaStream_t, cudaStreamCaptureStatus* st) {
  *st = cudaStreamCaptureStatusNone;
  return cudaSuccess;
}
inline void __syncthreads() {}
inline float __int_as_float(int x) { union { int i; float f; } u{x}; return u.f; }
inline double __longlong_as_double(long long x) {
  union { long long i; double d; } u{x};
  return u.d;
}
template <typename T> inline T __shfl_down_sync(unsigned, T val, int) { return val; }
inline bool isfinite(float x) { return std::isfinite(x); }
inline bool isfinite(double x) { return std::isfinite(x); }
"""
    stub_cublas_header = """
#pragma once
#include <cstddef>
#include "cuda_runtime.h"
typedef void* cublasHandle_t;
enum cublasStatus_t { CUBLAS_STATUS_SUCCESS = 0 };
enum cublasOperation_t { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1 };
enum cublasPointerMode_t { CUBLAS_POINTER_MODE_HOST = 0 };
enum cublasMath_t { CUBLAS_DEFAULT_MATH = 0 };
enum cudaDataType_t { CUDA_R_16F = 2, CUDA_R_32F = 0 };
enum cublasComputeType_t { CUBLAS_COMPUTE_32F = 68 };
enum cublasGemmAlgo_t {
  CUBLAS_GEMM_DEFAULT = -1,
  CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99
};
inline cublasStatus_t cublasCreate(cublasHandle_t* h) {
  *h = reinterpret_cast<void*>(1);
  return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasSetPointerMode(cublasHandle_t, cublasPointerMode_t) {
  return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasSetMathMode(cublasHandle_t, cublasMath_t) {
  return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasSetWorkspace(cublasHandle_t, void*, size_t) {
  return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasSetStream(cublasHandle_t, cudaStream_t) {
  return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasGemmEx(
    cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int, const void*,
    const void*, cudaDataType_t, int,
    const void*, cudaDataType_t, int,
    const void*, void*, cudaDataType_t, int,
    cublasComputeType_t, cublasGemmAlgo_t) {
  return CUBLAS_STATUS_SUCCESS;
}
"""
    cu_path = REPO_ROOT / "src" / "tsjit" / "cuda" / "tsjit_ipcg_sm90.cu"
    cu_src = cu_path.read_text(encoding="utf-8")
    cu_cpp = re.sub(r"<<<[^>]+>>>", "", cu_src)
    (tmp_path / "cuda_runtime.h").write_text(stub_cuda_header, encoding="utf-8")
    (tmp_path / "cublas_v2.h").write_text(stub_cublas_header, encoding="utf-8")
    cpp_file = tmp_path / "tsjit_ipcg_sm90_check.cpp"
    cpp_file.write_text(cu_cpp, encoding="utf-8")

    for extra_flags in ([], ["-DTSJIT_HAS_CUBLAS"]):
        proc = subprocess.run(
            [
                compiler,
                "-std=c++20",
                "-fsyntax-only",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-Wno-unused-variable",
                "-Wno-unused-parameter",
                *extra_flags,
                "-I",
                str(tmp_path),
                str(cpp_file),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, (
            f"CUDA C++ syntax check failed ({extra_flags}):\n{proc.stderr}"
        )


def test_gifteval_gap_async_stream_c11_and_assert_causal() -> None:
    """Exercise C11 SoftRevisionEngine + Pre-Warmed IPCG-Ridge + OnlineACI + EventGate + N=1024."""
    import importlib.util

    bench_path = REPO_ROOT / "benchmarks" / "bench_h200_gifteval_stream.py"
    spec = importlib.util.spec_from_file_location("bench_h200_gifteval_stream", bench_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rng = np.random.default_rng(2026)
    t = np.arange(160, dtype=np.float64)
    series = 50.0 + 5.0 * np.sin(2.0 * np.pi * t / 24.0) + rng.normal(0.0, 0.3, size=160)
    series[120:] += 8.0  # regime shift

    z9 = np.array([-1.28, -0.84, -0.52, -0.25, 0.0, 0.25, 0.52, 0.84, 1.28], dtype=np.float64)

    def _causal_mock_backbone(x_batch: np.ndarray) -> np.ndarray:
        B = x_batch.shape[0]
        out = np.empty((B, 9, 8), dtype=np.float64)
        for b in range(B):
            row = x_batch[b]
            fin = row[np.isfinite(row)]
            center = float(fin[-1]) if fin.size > 0 else 0.0
            sd = max(float(np.std(fin[-16:])) if fin.size > 1 else 1.0, 0.5)
            for h in range(8):
                out[b, :, h] = center + z9 * sd
        return out

    # 1. Event-gated mode (max_gap_ticks=8 -> >50% compute elision + bit-exact causal)
    metrics_gated = mod._run_gap_async_stream_single_series(
        series,
        _causal_mock_backbone,
        context_length=64,
        horizon=8,
        ridge_dim=8,
        device=None,
        max_gap_ticks=8,
    )
    assert np.isfinite(metrics_gated["pinball_crps"])
    assert np.isfinite(metrics_gated["mae"])
    assert 0.0 <= metrics_gated["coverage_80"] <= 1.0
    assert 0.0 <= metrics_gated["coverage_90"] <= 1.0
    assert metrics_gated["gpu_elision_ratio"] > 0.5
    assert metrics_gated["arrival_p50_us"] < 100.0

    # 2. True Gap-Async mode (max_gap_ticks=1 -> age=1, pre-warmed Ridge beats naive persistence!)
    eager = mod._run_standalone_eager_baseline(series, _causal_mock_backbone, context_length=64)
    metrics_gap = mod._run_gap_async_stream_single_series(
        series,
        _causal_mock_backbone,
        context_length=64,
        horizon=8,
        ridge_dim=32,
        device=None,
        max_gap_ticks=1,
    )
    assert np.isfinite(metrics_gap["pinball_crps"])
    assert metrics_gap["pinball_crps"] < eager["pinball_crps"]
    assert metrics_gap["mae"] < eager["mae"]

    # 3. Bit-exact causality verified for BOTH gap_async (1) and event_gated (4, 8),
    #    including cut_step=0 right at the pre-warm boundary!
    for gap_k in (1, 4, 8):
        for cut in (0, None):
            causal_ok = mod.assert_causal_bit_exact(
                series,
                _causal_mock_backbone,
                context_length=64,
                horizon=8,
                ridge_dim=16 if gap_k <= 4 else 8,
                device=None,
                max_gap_ticks=gap_k,
                cut_step=cut,
            )
            assert causal_ok is True

    # 3b. DC-offset invariance at 2.22e10 (shifted-anchor scale must not blow up CRPS)
    dc_series = 2.22e10 + 5.0 * np.sin(2.0 * np.pi * t / 24.0) + rng.normal(0.0, 0.3, size=160)
    dc_metrics = mod._run_gap_async_stream_single_series(
        dc_series,
        _causal_mock_backbone,
        context_length=64,
        horizon=8,
        ridge_dim=16,
        device=None,
        max_gap_ticks=1,
    )
    assert np.isfinite(dc_metrics["pinball_crps"]) and dc_metrics["pinball_crps"] < 2.0
    assert np.isfinite(dc_metrics["mae"]) and dc_metrics["mae"] < 2.0

    # 3c. Micro-scale 1e-8 invariance (no 1e-6 floor distortion: Ridge beats persistence!)
    micro_series = (
        50.0 + 5.0 * np.sin(2.0 * np.pi * t / 24.0) + rng.normal(0.0, 0.2, size=160)
    ) * 1e-9

    def _micro_mock_backbone(x_batch: np.ndarray) -> np.ndarray:
        B = x_batch.shape[0]
        out = np.empty((B, 9, 8), dtype=np.float64)
        for b in range(B):
            row = x_batch[b]
            fin = row[np.isfinite(row)]
            center = float(fin[-1]) if fin.size > 0 else 0.0
            sd = max(float(np.std(fin[-16:])) if fin.size > 1 else 1e-12, 1e-30)
            for h in range(8):
                out[b, :, h] = center + z9 * sd
        return out

    micro_eager = mod._run_standalone_eager_baseline(
        micro_series, _micro_mock_backbone, context_length=64
    )
    micro_gap = mod._run_gap_async_stream_single_series(
        micro_series,
        _micro_mock_backbone,
        context_length=64,
        horizon=8,
        ridge_dim=32,
        device=None,
        max_gap_ticks=1,
    )
    assert np.isfinite(micro_gap["pinball_crps"])
    assert micro_gap["pinball_crps"] < micro_eager["pinball_crps"]

    # 3d. NaN-bearing stream produces finite metrics and stays bit-exact causal
    nan_stream = series.copy()
    nan_stream[10] = np.nan
    nan_stream[74] = np.nan
    nan_gap = mod._run_gap_async_stream_single_series(
        nan_stream,
        _causal_mock_backbone,
        context_length=64,
        horizon=8,
        ridge_dim=16,
        device=None,
        max_gap_ticks=1,
    )
    assert np.isfinite(nan_gap["pinball_crps"]) and np.isfinite(nan_gap["mae"])
    assert mod.assert_causal_bit_exact(
        nan_stream,
        _causal_mock_backbone,
        context_length=64,
        horizon=8,
        ridge_dim=16,
        device=None,
        max_gap_ticks=1,
    )

    # 4. Batched N=1024 C11 arrival benchmark reports true native C11 latency < 1 us/decision
    batched = mod.bench_batched_c11_arrival(n_streams=1024, n_ticks=16)
    assert batched["n_streams"] == 1024
    assert batched["python_ctypes_pwl_us_per_decision"] > 0.0
    if batched["native_c11"].get("status") == "ok":
        assert batched["native_c11"]["pwl_us_per_decision"] < 1.0
        assert batched["native_c11"]["soft_v2_us_per_decision"] < 2.0


def test_family_detection_chronos2_before_bolt_and_toto2_arch_no_torch() -> None:
    """Pure-Python regression tests for family detection & Toto-2.0 layer introspection."""
    from tsjit.compiler.api import _detect_family, normalize_precision
    from tsjit.compiler.chronos_pass import is_chronos2_model, is_chronos_bolt_model
    from tsjit.compiler.toto2_pass import inspect_toto2_architecture

    # 1. Chronos2Pipeline-like model with BOTH output_patch_embedding + chronos_config
    #    MUST be detected as 'chronos-2', NEVER 'chronos-bolt'!
    class GroupSelfAttention:
        pass

    class FakeChronos2Inner:
        def __init__(self, *, with_mask_fn: bool = True, with_gsa: bool = True) -> None:
            self.output_patch_embedding = object()
            self.chronos_config = {"output_patch_size": 16}
            self._gsa = GroupSelfAttention() if with_gsa else None
            self.encoder = SimpleNamespace()
            if with_mask_fn:
                self.encoder._construct_and_invert_group_time_mask = lambda *a, **kw: None

        def modules(self) -> list[Any]:
            return [self, self._gsa] if self._gsa is not None else [self]

    class Chronos2Pipeline:
        def __init__(self, inner: Any) -> None:
            self.model = inner

    pipe_c2 = Chronos2Pipeline(FakeChronos2Inner())
    assert _detect_family(pipe_c2) == "chronos-2"
    assert is_chronos2_model(pipe_c2) is True
    assert is_chronos_bolt_model(pipe_c2) is False

    # Anonymous wrapper without "chronos2" in class name: detected via GSA in modules()
    anon_gsa = SimpleNamespace(model=FakeChronos2Inner(with_mask_fn=False, with_gsa=True))
    assert _detect_family(anon_gsa) == "chronos-2"
    assert is_chronos_bolt_model(anon_gsa) is False

    # Anonymous wrapper without "chronos2" in class name: detected via encoder group-time mask fn
    anon_mask = SimpleNamespace(model=FakeChronos2Inner(with_mask_fn=True, with_gsa=False))
    assert _detect_family(anon_mask) == "chronos-2"
    assert is_chronos_bolt_model(anon_mask) is False

    # True Chronos-Bolt (output_patch_embedding + chronos_config, NO GSA / NO group mask)
    bolt_inner = SimpleNamespace(
        output_patch_embedding=object(),
        chronos_config={"prediction_length": 64},
        encoder=SimpleNamespace(),
    )
    bolt_pipe = SimpleNamespace(model=bolt_inner)
    assert _detect_family(bolt_pipe) == "chronos-bolt"
    assert is_chronos_bolt_model(bolt_pipe) is True
    assert is_chronos2_model(bolt_pipe) is False

    # 2. Toto-2.0 architecture introspection reads layer.attn.is_variate_layer and _Hq
    def _make_toto_layer(is_var: bool, hq: int = 8) -> Any:
        return SimpleNamespace(attn=SimpleNamespace(is_variate_layer=is_var, _Hq=hq))

    toto_4m_like = SimpleNamespace(
        config=SimpleNamespace(model_type="toto2", patch_size=32, embed_dim=256),
        transformer=SimpleNamespace(
            layers=[
                _make_toto_layer(False),
                _make_toto_layer(False),
                _make_toto_layer(False),
                _make_toto_layer(True),
            ]
        ),
    )
    assert _detect_family(toto_4m_like) == "toto2"
    arch_4m = inspect_toto2_architecture(toto_4m_like)
    assert arch_4m.time_layer_indices == (0, 1, 2)
    assert arch_4m.variate_layer_indices == (3,)
    assert arch_4m.final_time_layer_idx == 2
    assert arch_4m.num_heads == 8

    # Multi-group (313m-like) schedule with intermediate variate layer
    toto_mg_like = SimpleNamespace(
        config=SimpleNamespace(model_type="toto2", patch_size=32, embed_dim=256),
        transformer=SimpleNamespace(
            layers=[
                _make_toto_layer(False),
                _make_toto_layer(False),
                _make_toto_layer(True),
                _make_toto_layer(False),
                _make_toto_layer(False),
                _make_toto_layer(True),
            ]
        ),
    )
    arch_mg = inspect_toto2_architecture(toto_mg_like)
    assert arch_mg.time_layer_indices == (0, 1, 3, 4)
    assert arch_mg.variate_layer_indices == (2, 5)
    assert arch_mg.final_time_layer_idx == 4

    # Model detected via config.model_type == 'chronos2' even without class name or GSA
    cfg_c2 = SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(model_type="chronos2"),
            output_patch_embedding=object(),
            chronos_config={"output_patch_size": 16},
        )
    )
    assert _detect_family(cfg_c2) == "chronos-2"
    assert is_chronos_bolt_model(cfg_c2) is False

    # Generic fallback + precision normalization
    assert _detect_family(SimpleNamespace()) == "generic"
    assert normalize_precision("a10-auto", "exact") == "auto"
    assert normalize_precision("fp16x3", "exact") == "fp16x3"
    assert normalize_precision("a10-auto", "fp32") == "fp32"

    # Per-row crest mathematical bound: K=32 cannot exceed sqrt(32)=5.66 < 12; K=256 reaches 16 > 12
    from tsjit.cuda.scaled_fp16_triton import _split_rows_numpy

    w_small = np.full((64, 32), 0.01, dtype=np.float32)
    w_small[0, 0] = 100.0
    _, _, _, _, cw_small = _split_rows_numpy(w_small, need_v1=False)
    assert float(cw_small.max()) < 12.0

    w_wide = np.full((64, 256), 0.01, dtype=np.float32)
    w_wide[0, 0] = 100.0
    _, _, _, _, cw_wide = _split_rows_numpy(w_wide, need_v1=False)
    assert float(cw_wide.max()) > 12.0

    # Fix 2a pure-NumPy shift-equivariance check across 64 multi-regime series
    # (incl. last-patch NaNs, trailing NaN at -1, all-NaN row, and patch 0 outage).
    rng = np.random.default_rng(2026)
    B_sim, L_sim, P_sim = 64, 128, 32
    t_sim = np.linspace(0.0, 12.0, L_sim, dtype=np.float64)
    offsets = rng.uniform(-500.0, 500.0, size=(B_sim, 1))
    x_sim = offsets + np.sin(t_sim)[None, :] + 0.3 * rng.standard_normal((B_sim, L_sim))
    nan_m = rng.random((B_sim, L_sim)) < 0.25
    nan_m[:, -1] = False
    nan_m[:, -4] = True  # guaranteed NaN in final patch [-P:]
    nan_m[1, -1] = True  # trailing NaN at -1
    nan_m[2, :] = True  # all-NaN row
    nan_m[3, :P_sim] = True  # all-missing patch 0 outage
    x_sim[nan_m] = np.nan

    # Reference Toto2Model._prepare_forecast_inputs + PatchedCausalStdScaler in raw space:
    m_raw = np.isfinite(x_sim)
    m_forced = m_raw.copy()
    m_forced[:, -P_sim:] = True
    x_raw_clean = np.where(m_raw, x_sim, 0.0)

    def _causal_patch_scale(x_in: np.ndarray, m_in: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        B0, L0 = x_in.shape
        S0 = L0 // P_sim
        xp = x_in.reshape(B0, S0, P_sim)
        mp = m_in.reshape(B0, S0, P_sim).astype(np.float64)
        xm = np.where(m_in.reshape(B0, S0, P_sim), xp, 0.0)
        cnt = np.maximum(np.cumsum(mp.sum(axis=-1), axis=-1), 1.0)
        loc_p = np.cumsum(xm.sum(axis=-1), axis=-1) / cnt
        var_p = np.maximum(np.cumsum((xm * xm).sum(axis=-1), axis=-1) / cnt - loc_p * loc_p, 0.0)
        sc_p = np.sqrt(var_p + 1e-4)
        loc_full = np.repeat(loc_p, P_sim, axis=-1)
        sc_full = np.repeat(sc_p, P_sim, axis=-1)
        scaled = np.where(m_in, (x_in - loc_full) / sc_full, 0.0)
        return np.arcsinh(scaled), loc_full[:, -1:]

    scaled_ref, loc_ref = _causal_patch_scale(x_raw_clean, m_forced)

    # Fix 2a: compute dc_f64 from last finite, replace NaN -> 0.0 in [-P:] BEFORE subtracting dc_f64
    stripper = CausalFloat64DCStripper(enabled=True)
    dc_sim = stripper._last_finite_numpy(x_sim)
    x_work = x_sim.copy()
    tail = x_work[:, -P_sim:]
    x_work[:, -P_sim:] = np.where(np.isfinite(tail), tail, 0.0)
    x_stripped = (x_work - dc_sim).astype(np.float32).astype(np.float64)
    m_st = np.isfinite(x_stripped)
    m_st[:, -P_sim:] = True
    x_st_clean = np.where(m_st, x_stripped, 0.0)
    scaled_fix2a, loc_fix2a = _causal_patch_scale(x_st_clean, m_st)
    assert np.max(np.abs(scaled_fix2a - scaled_ref)) < 1e-5
    assert np.max(np.abs((loc_fix2a + dc_sim) - loc_ref)) < 1e-4


# ---------------------------------------------------------------------------
# 8. PyTorch Compiler Passes & Distributed Modules (run whenever torch installed)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed in this environment")
def test_generic_pass_transformer_compile() -> None:
    from tsjit.compiler import compile as tsjit_compile
    from tsjit.cuda.scaled_fp16_triton import CrestGuardedLinear

    class ToyTransformerForecaster(nn.Module):
        def __init__(self, ctx: int = 64, d_model: int = 64, horizon: int = 16) -> None:
            super().__init__()
            self.in_proj = nn.Linear(ctx, d_model)
            self.fc1 = nn.Linear(d_model, d_model * 2)
            self.fc2 = nn.Linear(d_model * 2, d_model)
            self.out_head = nn.Linear(d_model, horizon)

        def forward(self, x: Any) -> Any:
            h = torch.relu(self.in_proj(x))
            h = h + self.fc2(torch.nn.functional.gelu(self.fc1(h)))
            return self.out_head(h)

    torch.manual_seed(0)
    model = ToyTransformerForecaster().eval()
    compiled = tsjit_compile(
        model,
        horizon=16,
        context_length=64,
        precision="fp16x3",
        mode="exact",
        dc_strip=True,
        verify=True,
        use_cuda_graphs=False,
    )
    rep = compiled.report()
    assert rep["family"] == "generic"
    assert (
        rep["precision"]["replaced"] == 2
    )  # fc1 & fc2 replaced; in_proj & out_head preserved FP32!
    assert isinstance(compiled._backend.model.fc1, CrestGuardedLinear)
    assert not isinstance(compiled._backend.model.in_proj, CrestGuardedLinear)
    assert not isinstance(compiled._backend.model.out_head, CrestGuardedLinear)
    assert rep["parity_audit"]["passed"]

    # DC offset 2.22e10 preserved
    x_dc = 2.22e10 + torch.randn((4, 64), dtype=torch.float64)
    y_dc = compiled.predict(x_dc)
    assert y_dc.shape == (4, 16)
    assert torch.isfinite(y_dc).all()
    assert float((y_dc - 2.22e10).abs().max().item()) < 50.0


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed in this environment")
def test_chronos_bolt_and_chronos2_passes_mock() -> None:
    from tsjit.compiler import Chronos2Compiler, ChronosBoltCompiler

    # --- Mock Chronos-Bolt (head slicing 9*64 -> 9*16 exactness) ---
    class ResidualBlock(nn.Module):
        def __init__(self, d_in: int, d_out: int) -> None:
            super().__init__()
            self.hidden_layer = nn.Linear(d_in, d_in)
            self.output_layer = nn.Linear(d_in, d_out)
            self.residual_layer = nn.Linear(d_in, d_out)

        def forward(self, x: Any) -> Any:
            return self.output_layer(torch.relu(self.hidden_layer(x))) + self.residual_layer(x)

    class MockChronosBoltModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.chronos_config = SimpleNamespace(
                prediction_length=64,
                quantiles=[0.1 * i for i in range(1, 10)],
            )
            self.enc = nn.Linear(64, 64)
            self.output_patch_embedding = ResidualBlock(64, 9 * 64)

        def forward(self, context: Any) -> Any:
            h = torch.tanh(self.enc(context))
            q = self.output_patch_embedding(h).view(
                context.shape[0], 9, self.chronos_config.prediction_length
            )
            return SimpleNamespace(quantile_preds=q)

    torch.manual_seed(1)
    bolt = MockChronosBoltModel().eval()
    bolt_comp = ChronosBoltCompiler(
        bolt, horizon=16, context_length=64, precision="fp32", dc_strip=False, use_cuda_graphs=False
    )
    assert bolt_comp.head_sliced and bolt_comp.sliced_rows_after == 9 * 16
    x = torch.randn((4, 64), dtype=torch.float32)
    with torch.inference_mode():
        y_ref = bolt(context=x).quantile_preds[:, :, :16].sort(dim=1).values.to(torch.float64)
        y_cmp = bolt_comp.predict_quantiles(x)
    assert float((y_cmp - y_ref).abs().max().item()) < 1e-5

    # --- Mock Chronos-2 (singleton GroupSelfAttention W_ov fold exactness + grouped fallback) ---
    class MockGroupSelfAttention(nn.Module):
        def __init__(self, d: int = 32) -> None:
            super().__init__()
            self.layer_norm = nn.LayerNorm(d)
            self.self_attention = nn.Module()
            self.self_attention.v = nn.Linear(d, d)
            self.self_attention.o = nn.Linear(d, d)
            self.dropout = nn.Identity()

        def forward(
            self,
            hidden_states: Any,
            attention_mask: Any = None,
            output_attentions: bool = False,
            **kw: Any,
        ) -> Any:
            h = self.layer_norm(hidden_states)
            # True cross-batch mean within attention_mask groups; for singleton groups identity!
            v = self.self_attention.v(h)
            if attention_mask is not None:
                v = torch.einsum("ij,jtd->itd", attention_mask, v)
            return hidden_states + self.dropout(self.self_attention.o(v))

    class MockChronos2Encoder(nn.Module):
        def __init__(self, d: int = 32) -> None:
            super().__init__()
            self.block = nn.ModuleList([MockGroupSelfAttention(d) for _ in range(4)])
            self.mask_build_calls = 0

        def _construct_and_invert_group_time_mask(
            self, group_ids: Any, dtype: Any
        ) -> Any:
            self.mask_build_calls += 1
            eq = (group_ids[:, None] == group_ids[None, :]).to(dtype)
            return eq / eq.sum(dim=-1, keepdim=True).clamp(min=1.0)

    class MockChronos2Model(nn.Module):
        def __init__(self, d: int = 32) -> None:
            super().__init__()
            self.in_proj = nn.Linear(16, d)
            self.encoder = MockChronos2Encoder(d)
            self.head = nn.Linear(d, 9 * 16)

        def forward(self, context: Any, group_ids: Any | None = None) -> Any:
            B = context.shape[0]
            h = self.in_proj(context.view(B, 4, 16))
            if group_ids is not None:
                mask = self.encoder._construct_and_invert_group_time_mask(group_ids, h.dtype)
            else:
                mask = None
            for blk in self.encoder.block:
                h = blk(h, attention_mask=mask)
            q = self.head(h[:, -1, :]).view(B, 9, 16)
            return SimpleNamespace(quantile_preds=q)

    torch.manual_seed(2)
    c2 = MockChronos2Model().eval()
    c2_comp = Chronos2Compiler(
        c2, horizon=16, context_length=64, precision="fp32", dc_strip=False, use_cuda_graphs=False
    )
    assert c2_comp.report()["collapsed_group_self_attention_layers"] == 4
    assert c2_comp.report()["group_time_mask_stubbed"] is True
    with torch.inference_mode():
        gids_single = torch.arange(4)
        y2_ref = (
            c2(context=x, group_ids=gids_single).quantile_preds.sort(dim=1).values.to(torch.float64)
        )
        y2_cmp = c2_comp.predict_quantiles(x)
        assert float((y2_cmp - y2_ref).abs().max().item()) < 1e-5
        # Singleton call MUST bypass _construct_and_invert_group_time_mask completely!
        assert c2_comp.model.encoder.mask_build_calls == 0

        # Grouped fallback (non-singleton group_ids=[0, 0, 1, 1]) must match uncollapsed exactly!
        gids_grp = torch.tensor([0, 0, 1, 1])
        y2_grp_ref = (
            c2(context=x, group_ids=gids_grp).quantile_preds.sort(dim=1).values.to(torch.float64)
        )
        y2_grp_cmp = c2_comp.predict_quantiles(x, group_ids=gids_grp)
        assert float((y2_grp_cmp - y2_grp_ref).abs().max().item()) < 1e-5
        assert c2_comp.model.encoder.mask_build_calls == 1

    # Model with 0 collapsed GSA layers must NOT stub _construct_and_invert_group_time_mask!
    c2_nogsa = MockChronos2Model().eval()
    c2_nogsa.encoder.block = nn.ModuleList([])
    c2_nogsa_comp = Chronos2Compiler(
        c2_nogsa,
        horizon=16,
        context_length=64,
        precision="fp32",
        dc_strip=False,
        use_cuda_graphs=False,
    )
    assert c2_nogsa_comp.report()["collapsed_group_self_attention_layers"] == 0
    assert c2_nogsa_comp.report()["group_time_mask_stubbed"] is False


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed in this environment")
def test_toto2_pass_r1_variate_collapse_and_nan_parity() -> None:
    from tsjit.compiler.toto2_pass import compile_toto2, verify_parity

    class MockPatchScaler(nn.Module):
        def forward(self, data: Any, weights: Any, padding_mask: Any) -> tuple[Any, Any, Any]:
            w = weights.to(data.dtype)
            cnt = w.sum(dim=-1, keepdim=True).clamp(min=1.0)
            loc = (data * w).sum(dim=-1, keepdim=True) / cnt
            var = (((data - loc) * w) ** 2).sum(dim=-1, keepdim=True) / cnt
            scale = torch.sqrt(var + 1e-5)
            return (data - loc) / scale, loc, scale

    class MockPatchEmbed(nn.Module):
        def __init__(self, patch_size: int = 16, d: int = 32) -> None:
            super().__init__()
            self.patch_size = patch_size
            self.stride = patch_size
            self.proj = nn.Linear(patch_size, d)

        def forward(self, x_3d: Any) -> Any:
            B, V, L = x_3d.shape
            S = L // self.patch_size
            return self.proj(x_3d.view(B, V, S, self.patch_size))

    class MockTotoLayer(nn.Module):
        def __init__(self, d: int, num_heads: int, is_variate: bool) -> None:
            super().__init__()
            self.is_variate_layer = is_variate
            self.norm1 = nn.LayerNorm(d)
            self.attention = nn.Module()
            self.attention.num_heads = num_heads
            self.attention.wQKV = nn.Linear(d, 3 * d)
            self.attention.wO = nn.Linear(d, d)
            self.norm2 = nn.LayerNorm(d)
            self.mlp = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))

    class MockToto2Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(
                model_type="toto2",
                num_layers=4,
                layer_group_size=2,
                num_variate_layers_per_group=1,
                patch_size=16,
                stride=16,
                embed_dim=32,
                num_heads=4,
            )
            self.scaler = MockPatchScaler()
            self.patch_embed = MockPatchEmbed(16, 32)
            self.transformer = nn.Module()
            self.transformer.layers = nn.ModuleList(
                [
                    MockTotoLayer(32, 4, is_variate=False),
                    MockTotoLayer(32, 4, is_variate=True),
                    MockTotoLayer(32, 4, is_variate=False),  # final time layer (R1)
                    MockTotoLayer(32, 4, is_variate=True),  # final variate layer (collapsed)
                ]
            )
            self.final_norm = nn.LayerNorm(32)
            self.output_distribution = nn.Linear(32, 9 * 16)

    torch.manual_seed(3)
    toto = MockToto2Model().eval()
    compiled = compile_toto2(
        toto,
        horizon=16,
        context_length=64,
        precision="fp16x3",
        dc_strip=True,
        use_cuda_graphs=False,
    )
    rep = compiled.report()
    assert rep["final_time_layer_r1_idx"] == 2
    assert rep["collapsed_variate_layers"] == [1, 3]
    assert rep["dropped_dummy_pred_patch"] is True

    x_test = torch.randn((8, 64), dtype=torch.float64)
    parity = verify_parity(toto, compiled, x_test, tol=0.05, nan_fraction=0.25)
    assert parity["clean"].passed and parity["clean"].num_failed == 0
    assert parity["nan_25pct"].passed and parity["nan_25pct"].num_failed == 0

    # Short context L < patch_size (causal left-pad with NaN) must not crash
    y_short = compiled.predict_quantiles(torch.randn((2, 8), dtype=torch.float64))
    assert y_short.shape == (2, 9, 16) and torch.isfinite(y_short).all()

    # Strict boolean-mask PatchedCausalStdScaler + native .forecast + unit_scaling.Linear subclass
    class StrictBoolPatchedScaler(nn.Module):
        def forward(self, data: Any, mask: Any) -> tuple[Any, Any, Any]:
            # Mirrors toto2/model.py:179 — raises RuntimeError if mask is not boolean!
            mf = mask.to(data.dtype)
            cnt = mf.sum(dim=-1, keepdim=True).clamp(min=1.0)
            loc = (data * mf).sum(dim=-1, keepdim=True) / cnt
            var = (((data - loc) * mf) ** 2).sum(dim=-1, keepdim=True) / cnt
            scale = torch.sqrt(var + 1e-5)
            return torch.where(mask, (data - loc) / scale, torch.zeros_like(data)), loc, scale

    class MockUnitScaledLinear(nn.Linear):
        def forward(self, inp: Any) -> Any:
            return torch.nn.functional.linear(inp, self.weight, self.bias) * (
                1.0 / (self.in_features**0.5)
            )

    class MockUnitScaledVariateAttn(nn.Module):
        """Mimics Toto-2.0 variate attention with Wqkv, unit_scaling factor, and tuple return."""

        def __init__(self, d: int, num_heads: int) -> None:
            super().__init__()
            self.num_heads = num_heads
            self.Wqkv = MockUnitScaledLinear(d, 3 * d)
            self.Wo = MockUnitScaledLinear(d, d)

        def forward(self, x: Any):
            # Unannotated tuple return + 0.5 SDPA output scale tests probe calibration!
            B_s, V, D = x.shape
            qkv = self.Wqkv(x)
            v = qkv[..., 2 * D : 3 * D]
            return self.Wo(0.5 * v), None

    class MockNativeToto2WithForecast(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(
                model_type="toto2",
                num_layers=2,
                layer_group_size=2,
                num_variate_layers_per_group=1,
                patch_size=16,
                stride=16,
                embed_dim=32,
                num_heads=4,
            )
            self.scaler = StrictBoolPatchedScaler()
            self.patch_embed = MockPatchEmbed(16, 32)
            self.transformer = nn.Module()
            self.transformer.layers = nn.ModuleList(
                [
                    MockTotoLayer(32, 4, is_variate=False),
                    MockTotoLayer(32, 4, is_variate=True),
                ]
            )
            # Replace interior MLP & variate attention with unit-scaled / Wqkv modules!
            self.transformer.layers[0].mlp[0] = MockUnitScaledLinear(32, 64)
            self.transformer.layers[1].attention = MockUnitScaledVariateAttn(32, 4)
            self.final_norm = nn.LayerNorm(32)
            self.output_head = nn.Linear(32, 9 * 16)

        def forecast(
            self,
            inputs: dict[str, Any],
            horizon: int = 16,
            has_missing_values: bool = False,
        ) -> Any:
            _ = has_missing_values
            x_3d = inputs["target"]
            m_3d = inputs["target_mask"]
            B, V, L = x_3d.shape
            P = 16
            pad_x = torch.zeros((B, V, P), dtype=x_3d.dtype, device=x_3d.device)
            pad_m = torch.zeros((B, V, P), dtype=torch.bool, device=x_3d.device)
            scaled, loc, scale = self.scaler(
                torch.cat([x_3d, pad_x], dim=-1), torch.cat([m_3d, pad_m], dim=-1)
            )
            h = self.patch_embed(torch.asinh(scaled)).squeeze(1)
            # Apply unit-scaled MLP + singleton variate layer on last context patch (-2:-1)
            h = h + self.transformer.layers[0].mlp(self.transformer.layers[0].norm2(h))
            v_layer = self.transformer.layers[1]
            B_cur, S_cur, D_cur = h.shape
            v_in = v_layer.norm1(h).reshape(B_cur * S_cur, 1, D_cur)
            v_out, _ = v_layer.attention(v_in)
            h = h + v_out.reshape(B_cur, S_cur, D_cur)
            pred_out = self.final_norm(h[:, -2:-1, :])
            raw_q = self.output_head(pred_out).view(B, 9, P).permute(1, 0, 2).unsqueeze(2)
            q = torch.sinh(raw_q) * scale[..., -1:].unsqueeze(0) + loc[..., -1:].unsqueeze(0)
            return q[..., :horizon]

    torch.manual_seed(9)
    native_toto = MockNativeToto2WithForecast().eval()
    compiled_native = compile_toto2(
        native_toto,
        horizon=16,
        context_length=64,
        precision="fp16x3",
        dc_strip=True,
        use_cuda_graphs=False,
    )
    rep_native = compiled_native.report()
    assert rep_native["execution_path"] == "native_variate_collapse"
    assert rep_native["collapsed_variate_layers"] == [1]
    parity_native = verify_parity(native_toto, compiled_native, x_test, tol=0.05, nan_fraction=0.25)
    assert parity_native["clean"].passed and parity_native["clean"].num_failed == 0
    assert parity_native["nan_25pct"].passed and parity_native["nan_25pct"].num_failed == 0


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed in this environment")
def test_chronos2_pipeline_detection_modulelist_collapse_and_a10_auto() -> None:
    """Verify Chronos2Pipeline with output_patch_embedding + ModuleList GSA detects as chronos-2."""
    from tsjit.compiler import compile as tsjit_compile
    from tsjit.compiler.chronos_pass import _PlanGatedGroupSelfAttention
    from tsjit.cuda.scaled_fp16_triton import CrestGuardedLinear

    class TimeSelfAttention(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.layer_norm = nn.LayerNorm(d)
            self.proj = nn.Linear(d, d)

        def forward(self, hidden_states: Any, **kw: Any) -> Any:
            _ = kw
            return (hidden_states + self.proj(self.layer_norm(hidden_states)), None)

    class GroupSelfAttention(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.layer_norm = nn.LayerNorm(d)
            self.self_attention = nn.Module()
            self.self_attention.v = nn.Linear(d, d)
            self.self_attention.o = nn.Linear(d, d)
            self.dropout = nn.Identity()

        def forward(
            self,
            hidden_states: Any,
            attention_mask: Any = None,
            output_attentions: bool = False,
            **kw: Any,
        ) -> tuple[Any, Any]:
            _ = (output_attentions, kw)
            h = self.layer_norm(hidden_states)
            v = self.self_attention.v(h)
            if attention_mask is not None:
                v = torch.einsum("ij,jtd->itd", attention_mask, v)
            return (hidden_states + self.dropout(self.self_attention.o(v)), None)

    class FeedForward(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.fc1 = nn.Linear(d, d * 2)
            self.fc2 = nn.Linear(d * 2, d)

        def forward(self, hidden_states: Any) -> Any:
            return hidden_states + self.fc2(torch.relu(self.fc1(hidden_states)))

    class Chronos2EncoderBlock(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.layer = nn.ModuleList(
                [TimeSelfAttention(d), GroupSelfAttention(d), FeedForward(d)]
            )

        def forward(
            self,
            hidden_states: Any,
            *,
            group_time_mask: Any = None,
            output_attentions: bool = False,
        ) -> Any:
            hidden_states = self.layer[0](hidden_states)[0]
            hidden_states = self.layer[1](
                hidden_states, attention_mask=group_time_mask, output_attentions=output_attentions
            )[0]
            return self.layer[2](hidden_states)

    class Chronos2Encoder(nn.Module):
        def __init__(self, d: int = 32, n_layers: int = 4) -> None:
            super().__init__()
            self.block = nn.ModuleList([Chronos2EncoderBlock(d) for _ in range(n_layers)])
            self.mask_build_calls = 0

        def _construct_and_invert_group_time_mask(
            self, group_ids: Any, attention_mask: Any, dtype: Any
        ) -> Any:
            _ = attention_mask
            self.mask_build_calls += 1
            eq = (group_ids[:, None] == group_ids[None, :]).to(dtype)
            return eq / eq.sum(dim=-1, keepdim=True).clamp(min=1.0)

        def forward(self, inputs_embeds: Any, *, group_ids: Any) -> Any:
            group_time_mask = self._construct_and_invert_group_time_mask(
                group_ids, None, inputs_embeds.dtype
            )
            h = inputs_embeds
            for blk in self.block:
                h = blk(h, group_time_mask=group_time_mask)
            return h

    class MockChronos2WithBoltAttrs(nn.Module):
        """Has output_patch_embedding + chronos_config (like real Chronos-2) + ModuleList GSA."""

        def __init__(self, d: int = 32) -> None:
            super().__init__()
            self.chronos_config = SimpleNamespace(output_patch_size=16, prediction_length=64)
            self.input_patch_embedding = nn.Linear(16, d)
            self.encoder = Chronos2Encoder(d=d, n_layers=4)
            self.output_patch_embedding = nn.Linear(d, 9 * 16)

        def forward(
            self,
            context: Any,
            group_ids: Any | None = None,
            num_output_patches: int = 1,
        ) -> Any:
            _ = num_output_patches
            B = context.shape[0]
            if group_ids is None:
                group_ids = torch.arange(B, dtype=torch.long, device=context.device)
            h = self.input_patch_embedding(context.view(B, 4, 16))
            h = self.encoder(h, group_ids=group_ids)
            q = self.output_patch_embedding(h[:, -1, :]).view(B, 9, 16)
            return SimpleNamespace(quantile_preds=q)

    class MockPipelineWrapper:
        """Anonymous wrapper (no 'Chronos2' in class name) verifying structural detection."""

        def __init__(self, model: nn.Module) -> None:
            self.model = model

    torch.manual_seed(11)
    inner = MockChronos2WithBoltAttrs(d=32).eval()
    pipe = MockPipelineWrapper(inner)
    compiled = tsjit_compile(
        pipe,
        horizon=16,
        context_length=64,
        precision="a10-auto",
        mode="exact",
        dc_strip=True,
        verify=True,
        use_cuda_graphs=False,
    )
    rep = compiled.report()
    assert compiled.family == "chronos-2"
    assert rep["pass"] == "chronos-2"
    assert rep["collapsed_group_self_attention_layers"] == 4
    assert rep["group_time_mask_stubbed"] is True
    assert rep["parity_audit"]["passed"]
    # Well-conditioned weights (crest_w <= 12) stay native nn.Linear under a10-auto!
    assert rep["precision"]["replaced"] == 0
    assert rep["precision"]["kept_native_tf32"] > 0

    for blk in compiled._backend.model.encoder.block:
        assert isinstance(blk.layer[1], _PlanGatedGroupSelfAttention)

    x = torch.randn((4, 64), dtype=torch.float32)
    with torch.inference_mode():
        y_single_ref = (
            inner(context=x, group_ids=torch.arange(4))
            .quantile_preds.sort(dim=1)
            .values.to(torch.float64)
        )
        # Check parity with dc_strip=False to compare directly against raw inner
        comp_nodc = tsjit_compile(
            pipe,
            horizon=16,
            context_length=64,
            precision="a10-auto",
            dc_strip=False,
            use_cuda_graphs=False,
        )
        before_calls = comp_nodc._backend.model.encoder.mask_build_calls
        y_single_cmp = comp_nodc.predict_quantiles(x)
        assert float((y_single_cmp - y_single_ref).abs().max().item()) < 1e-5
        assert comp_nodc._backend.model.encoder.mask_build_calls == before_calls

        gids_grp = torch.tensor([0, 0, 1, 1])
        y_grp_ref = (
            inner(context=x, group_ids=gids_grp).quantile_preds.sort(dim=1).values.to(torch.float64)
        )
        y_grp_cmp = comp_nodc.predict_quantiles(x, group_ids=gids_grp)
        assert float((y_grp_cmp - y_grp_ref).abs().max().item()) < 1e-5
        assert comp_nodc._backend.model.encoder.mask_build_calls == before_calls + 1

    # High-crest outlier linear (crest_w > 12.0 requires K > 144, e.g., d=256 -> crest=16.0 > 12.0)
    # MUST be selectively escalated to fp16x3 while well-conditioned peers stay native nn.Linear!
    inner_out = MockChronos2WithBoltAttrs(d=256).eval()
    with torch.no_grad():
        inner_out.encoder.block[0].layer[2].fc1.weight.fill_(0.01)
        inner_out.encoder.block[0].layer[2].fc1.weight[0, 0] = 100.0
    comp_out = tsjit_compile(
        MockPipelineWrapper(inner_out),
        horizon=16,
        context_length=64,
        precision="a10-auto",
        dc_strip=False,
        use_cuda_graphs=False,
    )
    rep_out = comp_out.report()
    assert rep_out["precision"]["replaced"] == 1
    assert rep_out["precision"]["escalated_w"] == 1
    assert rep_out["precision"]["kept_native_tf32"] > 0
    assert rep_out["precision"]["max_crest_w"] > 12.0
    assert isinstance(comp_out._backend.model.encoder.block[0].layer[2].fc1, CrestGuardedLinear)
    assert comp_out._backend.model.encoder.block[0].layer[2].fc1.mode == "fp16x3"
    assert not isinstance(comp_out._backend.model.encoder.block[0].layer[2].fc2, CrestGuardedLinear)

    # Verify encoder that rejects group_time_mask=None gracefully un-stubs instead of crashing,
    # AND unannotated tuple-returning GroupSelfAttention still works cleanly!
    class StrictMaskChronos2Encoder(Chronos2Encoder):
        def forward(self, inputs_embeds: Any, *, group_ids: Any) -> Any:
            group_time_mask = self._construct_and_invert_group_time_mask(
                group_ids, None, inputs_embeds.dtype
            )
            if group_time_mask is None:
                raise RuntimeError("strict encoder requires non-None group_time_mask")
            h = inputs_embeds
            for blk in self.block:
                h = blk(h, group_time_mask=group_time_mask)
            return h

    strict_inner = MockChronos2WithBoltAttrs(d=32).eval()
    strict_inner.encoder = StrictMaskChronos2Encoder(d=32, n_layers=2)
    comp_strict = tsjit_compile(
        MockPipelineWrapper(strict_inner),
        horizon=16,
        context_length=64,
        precision="a10-auto",
        dc_strip=False,
        use_cuda_graphs=False,
    )
    assert comp_strict.report()["collapsed_group_self_attention_layers"] == 2
    assert comp_strict.report()["group_time_mask_stubbed"] is False
    y_strict = comp_strict.predict_quantiles(x)
    assert y_strict.shape == (4, 9, 16) and torch.isfinite(y_strict).all()


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed in this environment")
def test_c7_chronos2_all_nan_patch_outage_and_concurrent_plan_lock() -> None:
    """C7 Breaker regression: Chronos-2 collapsed GSA MUST match eager on 16-step all-NaN patch."""
    import concurrent.futures

    from tsjit.compiler import Chronos2Compiler

    class PatchMaskGroupSelfAttention(nn.Module):
        """Mirrors real Chronos-2 GroupSelfAttention with (T, 1, B, B) inverted additive mask."""

        def __init__(self, d: int = 32) -> None:
            super().__init__()
            self.layer_norm = nn.LayerNorm(d)
            self.self_attention = nn.Module()
            self.self_attention.v = nn.Linear(d, d, bias=False)
            self.self_attention.o = nn.Linear(d, d, bias=False)
            self.dropout = nn.Identity()

        def forward(
            self,
            hidden_states: Any,
            attention_mask: Any = None,
            output_attentions: bool = False,
            **kw: Any,
        ) -> tuple[Any, Any]:
            _ = (output_attentions, kw)
            B, T, _ = hidden_states.shape
            h = self.layer_norm(hidden_states)
            v = self.self_attention.v(h)  # (B, T, D)
            if attention_mask is not None and attention_mask.ndim == 4:
                # attention_mask: (T, 1, B, B) with 0.0 valid, -inf masked.
                # Eager Chronos-2 masks out all-NaN query/key patches -> attn_out = 0.
                scores = attention_mask[:, 0]  # (T, B, B)
                valid_row = (scores >= -1e-6).any(dim=-1, keepdim=True)  # (T, B, 1)
                safe_scores = torch.where(valid_row, scores, torch.zeros_like(scores))
                weights = torch.softmax(safe_scores, dim=-1)
                weights = torch.where(valid_row, weights, torch.zeros_like(weights))
                v_tbd = v.transpose(0, 1)  # (T, B, D)
                agg = torch.bmm(weights, v_tbd).transpose(0, 1)  # (B, T, D)
            else:
                agg = v
            return (hidden_states + self.dropout(self.self_attention.o(agg)), None)

    class PatchMaskChronos2Encoder(nn.Module):
        def __init__(self, d: int = 32, n_layers: int = 3) -> None:
            super().__init__()
            self.block = nn.ModuleList([PatchMaskGroupSelfAttention(d) for _ in range(n_layers)])
            self.mask_build_calls = 0

        def _construct_and_invert_group_time_mask(
            self, group_ids: Any, attention_mask: Any, dtype: Any
        ) -> Any:
            self.mask_build_calls += 1
            # Real Chronos-2 builds (T, 1, B, B) additive mask combining group_ids & attention_mask
            grp = group_ids[:, None] == group_ids[None, :]  # (B, B)
            # attention_mask: (B, T) bool/float patch validity
            obs = attention_mask > 0  # (B, T)
            # Patch (b_q, t) attends to (b_k, t) iff same group AND both patches observed
            obs_t = obs.transpose(0, 1)  # (T, B)
            allowed = grp.unsqueeze(0) & obs_t.unsqueeze(2) & obs_t.unsqueeze(1)  # (T, B, B)
            inv = torch.where(
                allowed.unsqueeze(1),
                torch.zeros((), dtype=dtype, device=group_ids.device),
                torch.full((), float("-inf"), dtype=dtype, device=group_ids.device),
            )
            return inv

        def forward(self, inputs_embeds: Any, *, attention_mask: Any, group_ids: Any) -> Any:
            group_time_mask = self._construct_and_invert_group_time_mask(
                group_ids, attention_mask, inputs_embeds.dtype
            )
            h = inputs_embeds
            for blk in self.block:
                h = blk(h, attention_mask=group_time_mask)[0]
            return h

    class OutageChronos2Model(nn.Module):
        def __init__(self, d: int = 32) -> None:
            super().__init__()
            self.chronos_config = SimpleNamespace(output_patch_size=16, prediction_length=64)
            self.input_patch_embedding = nn.Linear(16, d)
            self.encoder = PatchMaskChronos2Encoder(d=d, n_layers=3)
            self.output_patch_embedding = nn.Linear(d, 9 * 16)

        def forward(
            self,
            context: Any,
            group_ids: Any | None = None,
            num_output_patches: int = 1,
        ) -> Any:
            _ = num_output_patches
            B, L = context.shape
            T = L // 16
            if group_ids is None:
                group_ids = torch.arange(B, dtype=torch.long, device=context.device)
            patches = context.view(B, T, 16)
            patch_obs = torch.isfinite(patches).any(dim=-1)  # (B, T)
            clean_patches = torch.nan_to_num(patches, nan=0.0)
            h = self.input_patch_embedding(clean_patches)
            h = self.encoder(h, attention_mask=patch_obs.to(h.dtype), group_ids=group_ids)
            # Combine ALL patches (including any all-NaN patch!) into output so mismatch is exposed!
            q = self.output_patch_embedding(h.mean(dim=1)).view(B, 9, 16)
            return SimpleNamespace(quantile_preds=q)

    torch.manual_seed(77)
    model = OutageChronos2Model(d=32).eval()
    comp = Chronos2Compiler(
        model,
        horizon=16,
        context_length=64,
        precision="fp32",
        dc_strip=False,
        use_cuda_graphs=False,
    )
    assert comp.report()["collapsed_group_self_attention_layers"] == 3
    assert comp.report()["group_time_mask_stubbed"] is True

    x_outage = torch.randn((4, 64), dtype=torch.float32)
    x_outage[1, 16:32] = float("nan")  # Entire 16-step patch 1 is all-NaN outage!
    x_outage[2, 0:16] = float("nan")  # Entire 16-step patch 0 is all-NaN outage!

    with torch.inference_mode():
        y_ref = (
            model(context=x_outage, group_ids=torch.arange(4))
            .quantile_preds.sort(dim=1)
            .values.to(torch.float64)
        )
        before_calls = comp.model.encoder.mask_build_calls
        y_cmp = comp.predict_quantiles(x_outage)
        # MUST be O(B*T) (zero calls to _construct_and_invert_group_time_mask) AND bit-exact!
        assert comp.model.encoder.mask_build_calls == before_calls
        assert float((y_cmp - y_ref).abs().max().item()) < 1e-5

    # Concurrent singleton + grouped callers must not race on singleton_plan (_plan_lock).
    gids_grp = torch.tensor([0, 0, 1, 1])
    with torch.inference_mode():
        y_grp_ref = (
            model(context=x_outage, group_ids=gids_grp)
            .quantile_preds.sort(dim=1)
            .values.to(torch.float64)
        )

    def _worker(idx: int) -> float:
        if idx % 2 == 0:
            got = comp.predict_quantiles(x_outage)
            return float((got - y_ref).abs().max().item())
        got = comp.predict_quantiles(x_outage, group_ids=gids_grp)
        return float((got - y_grp_ref).abs().max().item())

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        errs = list(ex.map(_worker, range(16)))
    assert max(errs) < 1e-5

    # Verify 4D boolean mask on _PlanGatedGroupSelfAttention masks False diagonal entries properly
    gsa_gated = comp.model.encoder.block[0]
    h_p = torch.randn((4, 4, 32), dtype=torch.float32)
    bool_mask_4d = torch.ones((4, 1, 4, 4), dtype=torch.bool)
    bool_mask_4d[1, 0, 2, :] = False  # patch t=1 of series b=2 masked out
    y_bool = gsa_gated(h_p, attention_mask=bool_mask_4d)[0]
    assert float((y_bool[2, 1] - h_p[2, 1]).abs().max().item()) < 1e-6
    assert float((y_bool[0, 0] - h_p[0, 0]).abs().max().item()) > 1e-3
    assert comp.warmup(buckets=(1,)) == []  # CPU warmup returns [] safely under _plan_lock

    # Direct Chronos2Compiler precision alias normalization ('strict' -> 'fp16x3')
    comp_strict = Chronos2Compiler(
        model,
        horizon=16,
        context_length=64,
        precision="strict",
        dc_strip=False,
        use_cuda_graphs=False,
    )
    assert comp_strict.precision == "fp16x3"
    assert comp_strict.report()["precision"]["replaced"] > 0


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed in this environment")
def test_c8_i5b_i75_torch_ridge_bounds_state_dict_repack_and_loo_hardening() -> None:
    """Breaker tests for C8 (int64 Gram wraparound), I5(b) (load_state_dict), I7.5 (Torch LOO)."""
    from tsjit.cuda.loader import OnlineGpuRidgeEngine
    from tsjit.cuda.scaled_fp16_triton import CrestGuardedLinear
    from tsjit.distributed.async_fuse import AsyncFuseMonoid, AsyncFuseRunner

    # --- C8: OnlineGpuRidgeEngine validation & int64 wraparound prevention ---
    with pytest.raises(ValueError, match="ridge_lambda"):
        OnlineGpuRidgeEngine(2, 4, ridge_lambda=0.0, device="cpu")
    with pytest.raises(ValueError, match="ridge_lambda"):
        OnlineGpuRidgeEngine(2, 4, ridge_lambda=-1.0, device="cpu")
    with pytest.raises(ValueError, match="q_scale"):
        OnlineGpuRidgeEngine(2, 4, q_scale=0.0, device="cpu")
    with pytest.raises(ValueError, match="q_scale"):
        OnlineGpuRidgeEngine(2, 4, q_scale=float("nan"), device="cpu")
    with pytest.raises(ValueError, match="q_scale"):
        OnlineGpuRidgeEngine(2, 4, q_scale=1e-200, device="cpu")
    with pytest.raises(ValueError, match="q_scale"):
        OnlineGpuRidgeEngine(2, 4, q_scale=1e200, device="cpu")
    with pytest.raises(ValueError, match="num_series"):
        OnlineGpuRidgeEngine(True, 4, device="cpu")

    eng = OnlineGpuRidgeEngine(2, 4, ridge_lambda=1e-2, q_scale=1024.0, device="cpu")
    # Adversarial 1e12 observations repeated 100 times MUST stay strictly clamped to 32767
    # with zero int64 overflow/wraparound!
    x_huge = torch.full((2, 16, 4), 1e12, dtype=torch.float64)
    y_huge = torch.full((2, 16), 1e12, dtype=torch.float64)
    for _ in range(10):
        eng.update(x_huge, y_huge)
    expected_entry = 10 * 16 * (32767 * 32767)
    assert int(eng.G_i64[0, 0, 0].item()) == expected_entry
    assert bool((eng.G_i64[:, :, :4] > 0).all().item())
    sol = eng.solve(tol=1e-8, max_iters=16)
    assert bool(sol.converged.all().item()) and torch.isfinite(sol.theta).all()

    # --- I5(b): CrestGuardedLinear.load_state_dict repacks buffers cleanly ---
    torch.manual_seed(42)
    lin = CrestGuardedLinear(32, 16, bias=True, mode="fp16x3", graph_safe=True)
    lin.repack()
    x_in = torch.randn((4, 32), dtype=torch.float32)
    y_before = lin(x_in)

    new_w = torch.randn((16, 32), dtype=torch.float32) * 3.0
    new_b = torch.randn((16,), dtype=torch.float32)
    lin.load_state_dict({"weight": new_w, "bias": new_b})
    y_after = lin(x_in)
    y_expected = torch.nn.functional.linear(x_in, new_w, new_b)
    assert float((y_after - y_before).abs().max().item()) > 1e-1
    assert float((y_after - y_expected).abs().max().item()) < 1e-3

    # Shape-mismatch load_state_dict must surface PyTorch's RuntimeError cleanly
    with pytest.raises(RuntimeError):
        lin.load_state_dict({"weight": torch.randn((8, 32)), "bias": new_b})

    # --- I7.5: Torch SeriesLaggedLOOTracker duplicate/OOB/0-D rejection + AsyncFuseRunner ---
    pre = nn.Linear(8, 8).eval()
    tail = nn.Linear(8, 4).eval()
    monoid = AsyncFuseMonoid(8, 8, num_groups=4).eval()
    runner = AsyncFuseRunner(pre, monoid, tail, batch_size=4, input_shape=(8,), device="cpu")
    x_af = torch.randn((4, 8), dtype=torch.float32)
    g_ok = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    s_ok = torch.tensor([10, 20, 30, 40], dtype=torch.int64)
    _ = runner.step(x_af, g_ok, s_ok)
    runner.flush()

    with pytest.raises(ValueError, match="1D series_ids"):
        runner.tracker.update(torch.tensor(10), torch.tensor(0), torch.zeros((1, 10)))
    with pytest.raises(ValueError, match="Duplicate series_ids"):
        runner.step(x_af, g_ok, torch.tensor([10, 10, 30, 40], dtype=torch.int64))
    with pytest.raises(ValueError, match="out of bounds"):
        runner.step(x_af, torch.tensor([0, 1, 2, 4], dtype=torch.int64), s_ok)
    runner.reset_for_migration([20])
    own_chk = runner.tracker.lookup_own_prev(s_ok, [0, 1, 2, 3])
    assert float(own_chk[1].abs().max().item()) == 0.0
    assert float(own_chk[0].abs().max().item()) > 0.0

    # --- C5 Torch scale_free_norm_floor higher-rank x with 0D scalar scale must terminate! ---
    d_torch = scale_free_norm_floor(torch.full((4, 16), 10.0), loc=10.0, scale=0.0)
    assert abs(float(d_torch.item()) - 1e-3) < 1e-12


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed in this environment")
def test_toto2_direct_r1_r2_core_nan25_outage_and_multigroup() -> None:
    """Verify real-structure Toto-2.0 direct R1+R2 core, Fix 2a nan_25pct, outage, and 313m."""
    from tsjit.compiler.toto2_pass import _call_toto2_native_forecast, compile_toto2

    class MockUnitScaling:
        @staticmethod
        def residual_split(x: Any, tau: float) -> tuple[Any, Any]:
            _ = tau
            return x, x

        @staticmethod
        def residual_add(branch: Any, skip: Any, tau: float) -> Any:
            return branch * float(tau) + skip * float((1.0 - tau * tau) ** 0.5)

    class MockPatchedCausalStdScaler(nn.Module):
        def __init__(self, patch_size: int) -> None:
            super().__init__()
            self.patch_size = patch_size

        def forward(self, data: Any, mask: Any) -> tuple[Any, Any]:
            # Strict boolean mask + strictly causal patch-level cumulative mean/std -> (loc, scale).
            B, V, L = data.shape
            P = self.patch_size
            S = L // P
            d_p = data.to(torch.float64).view(B, V, S, P)
            m_p = mask.view(B, V, S, P)
            mf = torch.where(m_p, torch.ones_like(d_p), torch.zeros_like(d_p))
            x_m = torch.where(m_p, d_p, torch.zeros_like(d_p))
            c_cnt = mf.sum(dim=-1).cumsum(dim=-1).clamp(min=1.0)
            c_sum = x_m.sum(dim=-1).cumsum(dim=-1)
            c_sq = (x_m * x_m).sum(dim=-1).cumsum(dim=-1)
            loc_p = c_sum / c_cnt
            var_p = (c_sq / c_cnt - loc_p * loc_p).clamp(min=0.0)
            scale_p = torch.sqrt(var_p + 1e-4)
            return (
                loc_p.repeat_interleave(P, dim=-1).to(data.dtype),
                scale_p.repeat_interleave(P, dim=-1).to(data.dtype),
            )

    class MockQKRoPE(nn.Module):
        def __init__(self, head_dim: int) -> None:
            super().__init__()
            inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        def _rot(self, t: Any, pos_ids: Any) -> Any:
            angles = pos_ids.to(t.dtype).unsqueeze(-1) * self.inv_freq.unsqueeze(0)
            cos = torch.cos(angles)[None, None, :, :]
            sin = torch.sin(angles)[None, None, :, :]
            t1 = t[..., 0::2]
            t2 = t[..., 1::2]
            out = torch.empty_like(t)
            out[..., 0::2] = t1 * cos - t2 * sin
            out[..., 1::2] = t1 * sin + t2 * cos
            return out

        def forward(self, q: Any, k: Any, *, query_ids: Any, kv_ids: Any) -> tuple[Any, Any]:
            return self._rot(q, query_ids), self._rot(k, kv_ids)

    class MockRealToto2Attn(nn.Module):
        def __init__(self, d: int, hq: int, hkv: int, is_variate: bool) -> None:
            super().__init__()
            head_dim = d // hq
            dq, dk, dv = hq * head_dim, hkv * head_dim, hkv * head_dim
            self._split_sizes = (dq, dk, dv)
            self._Hq = hq
            self._Hkv = hkv
            self.is_variate_layer = is_variate
            self.config = SimpleNamespace(heads_per_group=hq // hkv)
            self.in_proj = nn.Linear(d, dq + dk + dv)
            self.out_proj = nn.Linear(dq, d)
            self.q_norm = nn.LayerNorm(head_dim)
            self.k_norm = nn.LayerNorm(head_dim)
            self._pds = nn.Parameter(torch.linspace(0.8, 1.2, head_dim))
            self.qk_proj = None if is_variate else MockQKRoPE(head_dim)
            self.mult = 1.0 / float(head_dim**0.5)

        def forward(self, x: Any, seq_ids: Any = None, attn_mask: Any = None) -> Any:
            B, S, _ = x.shape
            dq, dk, dv = self._split_sizes
            dh = dq // self._Hq
            q, k, v = self.in_proj(x).split(self._split_sizes, dim=-1)
            q = q.view(B, S, self._Hq, dh).transpose(1, 2)
            k = k.view(B, S, self._Hkv, dh).transpose(1, 2)
            v = v.view(B, S, self._Hkv, dh).transpose(1, 2)
            q = self.q_norm(q) * self._pds
            k = self.k_norm(k)
            if self.qk_proj is not None and seq_ids is not None:
                q, k = self.qk_proj(q, k, query_ids=seq_ids, kv_ids=seq_ids)
            if self._Hkv != self._Hq:
                g = self.config.heads_per_group
                k = k.unsqueeze(2).expand(B, self._Hkv, g, S, dh).reshape(B, self._Hq, S, dh)
                v = v.unsqueeze(2).expand(B, self._Hkv, g, S, dh).reshape(B, self._Hq, S, dh)
            out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=(attn_mask is None and not self.is_variate_layer),
                scale=self.mult,
            )
            return self.out_proj(out.transpose(1, 2).contiguous().view(B, S, dq))

    class MockRealToto2Layer(nn.Module):
        def __init__(self, d: int, hq: int, hkv: int, is_variate: bool) -> None:
            super().__init__()
            self._us_functional = MockUnitScaling
            self.attn = MockRealToto2Attn(d, hq, hkv, is_variate)
            self.norm1 = nn.LayerNorm(d)
            self.norm2 = nn.LayerNorm(d)
            self.ffn = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))
            self.attn_tau = 0.35
            self.mlp_tau = 0.25
            self.attn_resid_dropout = nn.Identity()
            self.mlp_resid_dropout = nn.Identity()

        def forward(self, x: Any, seq_ids: Any = None, attn_mask: Any = None) -> Any:
            x_in, skip = MockUnitScaling.residual_split(x, self.attn_tau)
            x_attn = self.attn_resid_dropout(
                self.attn(self.norm1(x_in), seq_ids=seq_ids, attn_mask=attn_mask)
            )
            x = MockUnitScaling.residual_add(x_attn, skip, self.attn_tau)
            x_in2, skip2 = MockUnitScaling.residual_split(x, self.mlp_tau)
            x_ffn = self.mlp_resid_dropout(self.ffn(self.norm2(x_in2)))
            return MockUnitScaling.residual_add(x_ffn, skip2, self.mlp_tau)

    class MockQuantileOutputHead(nn.Module):
        def __init__(self, d: int, n_q: int, patch_size: int, nop: int = 2) -> None:
            super().__init__()
            self.n_q = n_q
            self.patch_size = patch_size
            self.num_output_patches = nop
            self.proj = nn.Linear(d, n_q * nop * patch_size)

        def forward(self, x: Any, q: Any = None) -> Any:
            _ = q
            B, S, _ = x.shape
            raw = self.proj(x).view(B, S * self.num_output_patches, self.n_q, self.patch_size)
            return raw.permute(2, 0, 1, 3)  # (Q, B, S * nop, P)

    class MockRealToto2Model(nn.Module):
        def __init__(self, layer_flags: Sequence[bool], *, p: int = 16, d: int = 32) -> None:
            super().__init__()
            self.config = SimpleNamespace(
                model_type="toto2",
                num_layers=len(layer_flags),
                patch_size=p,
                embed_dim=d,
                num_heads=8,
            )
            self.patch_size = p
            self.scaler = MockPatchedCausalStdScaler(p)
            self.patch_proj = nn.Linear(p * 2, d)
            self.transformer = nn.Module()
            self.transformer.layers = nn.ModuleList(
                [MockRealToto2Layer(d, 8, 4, is_var) for is_var in layer_flags]
            )
            self.transformer.out_norm = nn.LayerNorm(d)
            self.output_head = MockQuantileOutputHead(d, n_q=9, patch_size=p, nop=2)

        def _prepare_forecast_inputs(
            self, target: Any, target_mask: Any
        ) -> tuple[Any, Any]:
            # Mirrors real Toto2Model._prepare_forecast_inputs: last patch forced observed!
            m = target_mask.clone().to(torch.bool)
            m[..., -self.patch_size :] = True
            t_clean = torch.where(target_mask, target, torch.zeros_like(target))
            return t_clean, m

        def _embed_patches(self, scaled_ctx: Any, mask: Any, patch_size: int) -> Any:
            B, V, L = scaled_ctx.shape
            S = L // patch_size
            vals = scaled_ctx.view(B, V, S, patch_size)
            mf = mask.to(scaled_ctx.dtype).view(B, V, S, patch_size)
            return self.patch_proj(torch.cat([vals, mf], dim=-1))

        def forecast(
            self,
            inputs: dict[str, Any],
            horizon: int = 16,
            has_missing_values: bool = False,
        ) -> Any:
            _ = has_missing_values
            P = self.patch_size
            t_clean, m_ctx = self._prepare_forecast_inputs(inputs["target"], inputs["target_mask"])
            B, V, L = t_clean.shape
            # Append 1 dummy future patch (matching unmodified Toto2Model.forecast)
            dev = t_clean.device
            full_t = torch.cat(
                [t_clean, torch.zeros((B, V, P), dtype=t_clean.dtype, device=dev)], dim=-1
            )
            full_m = torch.cat(
                [m_ctx, torch.zeros((B, V, P), dtype=torch.bool, device=dev)], dim=-1
            )
            loc, scale = self.scaler(full_t, full_m)
            raw_ctx = (full_t[..., :L] - loc[..., :L]) / scale[..., :L]
            scaled_ctx = torch.where(m_ctx, raw_ctx, torch.zeros_like(raw_ctx)).asinh()
            ctx_x = self._embed_patches(scaled_ctx, m_ctx, P)
            pred_x = torch.zeros((B, V, 1, ctx_x.shape[-1]), dtype=ctx_x.dtype, device=dev)
            x = torch.cat([ctx_x, pred_x], dim=2).squeeze(1)  # (B, S+1, D)
            S_tot = x.shape[1]
            S_ctx = S_tot - 1
            time_ids = torch.arange(S_tot, dtype=torch.long, device=x.device)
            ctx_obs = m_ctx.view(B, S_ctx, P).any(dim=-1)
            key_obs = torch.cat([ctx_obs, torch.ones((B, 1), dtype=torch.bool, device=dev)], dim=1)
            causal = torch.tril(torch.ones((S_tot, S_tot), dtype=torch.bool, device=dev))
            eye = torch.eye(S_tot, dtype=torch.bool, device=dev)
            valid = (causal.unsqueeze(0) & key_obs.unsqueeze(1)) | eye.unsqueeze(0)
            attn_mask = torch.where(
                valid.unsqueeze(1),
                torch.zeros((), dtype=x.dtype, device=dev),
                torch.full((), float("-inf"), dtype=x.dtype, device=dev),
            )
            for layer in self.transformer.layers:
                if layer.attn.is_variate_layer:
                    xv = x.reshape(B * S_tot, 1, -1)
                    x = layer(xv, seq_ids=None, attn_mask=None).reshape(B, S_tot, -1)
                else:
                    x = layer(x, seq_ids=time_ids, attn_mask=attn_mask)
            # Last context token is at index -2 (because dummy pred_x is at -1)
            pred_out = self.transformer.out_norm(x[:, -2:-1, :])
            nop = self.output_head.num_output_patches
            block_q = self.output_head(pred_out, q=None)[..., ::nop, :]  # (Q, B, 1, P)
            loc_a = loc[:, :, -1:].unsqueeze(0)
            scale_a = scale[:, :, -1:].unsqueeze(0)
            q = block_q.sinh() * scale_a + loc_a
            return q[..., :horizon]

    class MockStandaloneUnitScaledLinear(nn.Module):
        """Mimics unit_scaling.Linear inheriting directly from nn.Module (NOT nn.Linear)."""

        def __init__(self, in_features: int, out_features: int) -> None:
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
            self.bias = nn.Parameter(torch.zeros(out_features))

        def forward(self, x: Any) -> Any:
            return torch.nn.functional.linear(x, self.weight, self.bias) * (
                1.0 / (self.in_features**0.5)
            )

    torch.manual_seed(2026)
    validator = StrictParityValidator(tol=0.05)

    for flags, expected_r1, expected_vars, use_standalone_us_linear in (
        ([False, False, False, True], 2, [3], False),  # 4m-like
        ([False, False, False, True], 2, [3], True),  # 4m-like with nn.Module unit_scaling.Linear
        ([False, False, True, False, False, True], 4, [2, 5], False),  # 313m multi-group-like
    ):
        model = MockRealToto2Model(flags, p=16, d=32).eval()
        if use_standalone_us_linear:
            for layer in model.transformer.layers:
                dq, dk, dv = layer.attn._split_sizes
                layer.attn.in_proj = MockStandaloneUnitScaledLinear(32, dq + dk + dv)
                layer.attn.out_proj = MockStandaloneUnitScaledLinear(dq, 32)
        for prec in ("auto", "fp16x3"):
            compiled = compile_toto2(
                model,
                horizon=16,
                context_length=64,
                precision=prec,
                dc_strip=True,
                use_cuda_graphs=False,
            )
            rep = compiled.report()
            assert rep["execution_path"] == "direct_r1_variate"
            assert rep["final_time_layer_r1_idx"] == expected_r1
            assert rep["collapsed_variate_layers"] == expected_vars
            assert rep["forces_last_patch_observed"] is True
            assert rep["dropped_dummy_pred_patch"] is True
            if prec == "auto":
                assert rep["precision"]["replaced"] == 0
                assert rep["precision"]["kept_native_tf32"] > 0
            else:
                assert rep["precision"]["replaced"] > 0
                assert rep["precision"]["escalated_w"] == rep["precision"]["replaced"]

            def _raw_eager_ref(x1: Any, m_: Any = model) -> Any:
                return _call_toto2_native_forecast(m_, x1.to(torch.float32), horizon=16).to(
                    torch.float64
                )

            g = torch.Generator().manual_seed(77)
            # Test BOTH small batch (B=1, sync-free path) AND B=20 (>16, clean is_causal + outage)
            for batch_n in (1, 20):
                x_clean = 100.0 + torch.randn((batch_n, 64), generator=g, dtype=torch.float64)
                x_nan25 = x_clean.clone()
                mask25 = torch.rand((batch_n, 64), generator=g) < 0.25
                mask25[:, -1] = False
                mask25[:, -4] = True  # Guarantee NaNs inside final patch [-P:]!
                x_nan25[mask25] = float("nan")

                x_outage = x_clean.clone()
                x_outage[:, :16] = float("nan")  # Entire patch 0 missing -> exercises masked path!

                aud_clean = validator.audit(_raw_eager_ref, compiled.predict_quantiles, x_clean)
                aud_nan = validator.audit(_raw_eager_ref, compiled.predict_quantiles, x_nan25)
                aud_outage = validator.audit(_raw_eager_ref, compiled.predict_quantiles, x_outage)
                assert aud_clean.passed and aud_clean.num_failed == 0
                assert aud_nan.passed and aud_nan.num_failed == 0
                assert aud_outage.passed and aud_outage.num_failed == 0

            # 2.22e10 DC-offset stress slice against DC-stripped eager reference
            x_dc = 2.22e10 + torch.randn((4, 64), generator=g, dtype=torch.float64)
            stripper = CausalFloat64DCStripper(enabled=True)

            def _dc_ref(x1: Any, m_: Any = model, s_: Any = stripper) -> Any:
                xs, dc = s_.strip(x1)
                qs = _call_toto2_native_forecast(m_, xs, horizon=16)
                return s_.restore(qs, dc).sort(dim=1).values

            aud_dc = validator.audit(_dc_ref, compiled.predict_quantiles, x_dc)
            assert aud_dc.passed and aud_dc.num_failed == 0

    # Verify C6 + I4 via public tsjit.compile API on Toto-2.0:
    # - precision='a10-auto' under mode='exact' escalates to fp16x3 (0/512 guarantee!)
    # - precision='fast-tf32' resolves to tf32 (zero CrestGuardedLinear replacement)
    # - predict_quantiles(has_missing_values=False) hits clean fast path even at B=1!
    from tsjit.compiler import compile as tsjit_compile

    toto_c6 = MockRealToto2Model([False, False, False, True], p=16, d=32).eval()
    c_auto = tsjit_compile(
        toto_c6,
        horizon=16,
        context_length=64,
        precision="a10-auto",
        mode="exact",
        use_cuda_graphs=False,
    )
    assert c_auto.precision_resolved == "fp16x3"
    assert c_auto.report()["precision"]["replaced"] > 0

    c_fast = tsjit_compile(
        toto_c6,
        horizon=16,
        context_length=64,
        precision="fast-tf32",
        mode="fast",
        use_cuda_graphs=False,
    )
    assert c_fast.precision_resolved == "tf32"
    assert c_fast.report()["precision"]["replaced"] == 0

    x_b1 = torch.randn((1, 64), dtype=torch.float64)
    y_b1_fast = c_auto.predict_quantiles(x_b1, has_missing_values=False)
    y_b1_default = c_auto.predict_quantiles(x_b1)
    assert float((y_b1_fast - y_b1_default).abs().max().item()) < 1e-6


# ---------------------------------------------------------------------------
# 9. CUDA / Hopper H200 Hardware Tests (auto-run on GPU nodes)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA GPU required")
def test_cuda_scaled_fp16_triton_and_graph_runner() -> None:
    from tsjit.compiler.graph_runner import BucketedCudaGraphRunner
    from tsjit.cuda.scaled_fp16_triton import CrestGuardedLinear

    device = torch.device("cuda:0")
    lin = CrestGuardedLinear.from_linear(
        nn.Linear(688, 137, bias=True).to(device), mode="fp16x3", graph_safe=True
    )
    # Non-contiguous row-strided input (stride(-1) == 1, stride(0) > K)
    big = torch.randn((32, 700), device=device, dtype=torch.float32)
    x_strided = big[:, :688]
    assert not x_strided.is_contiguous() and x_strided.stride(-1) == 1

    with torch.inference_mode():
        y = lin(x_strided)
        ref = x_strided.to(torch.float64) @ lin.weight.to(torch.float64).t() + lin.bias.to(
            torch.float64
        )
    rel = (y.to(torch.float64) - ref).abs().max().item() / ref.abs().max().item()
    assert rel < 1e-4

    # BucketedCudaGraphRunner multi-bucket capture & replay (including cuBLAS WGMMA + Triton!)
    runner = BucketedCudaGraphRunner(lin, sample_shape=(688,), buckets=(8, 32), device=device)
    y_g = runner(x_strided[:5])
    assert y_g.shape == (5, 137)
    assert float((y_g - y[:5]).abs().max().item()) < 1e-5
    # Second distinct input proves the captured CUDA Graph actually re-executes kernels (non-empty)
    y_g2 = runner(x_strided[5:10])
    assert float((y_g2 - y[5:10]).abs().max().item()) < 1e-5
    assert float((y_g2 - y_g).abs().max().item()) > 1e-2
    assert runner.stats()["captured_buckets"] == [8]
    assert runner.stats()["replay_count"] == 2
    assert runner.stats()["capture_fallback_count"] == 0

    # Verify cuBLAS WGMMA bridge directly when available (including beta=1 accumulation)
    from tsjit.cuda.loader import cublas_scaled_fp16_gemm, is_cublas_fp16_available

    if is_cublas_fp16_available(device):
        a16 = torch.randn((37, 131), device=device, dtype=torch.float16).contiguous()
        b16 = torch.randn((53, 131), device=device, dtype=torch.float16).contiguous()
        c32 = torch.empty((37, 53), device=device, dtype=torch.float32)
        cublas_scaled_fp16_gemm(a16, b16, c32, alpha=1.0, beta=0.0)
        cublas_scaled_fp16_gemm(a16, b16, c32, alpha=0.5, beta=1.0)
        c_ref = 1.5 * (a16.to(torch.float32) @ b16.to(torch.float32).t())
        assert float((c32 - c_ref).abs().max().item()) < 1e-3

        # I5(d): K == 0 with arbitrary beta (2.5) scales C cleanly instead of returning -7.
        a_k0 = torch.empty((16, 0), device=device, dtype=torch.float16).contiguous()
        b_k0 = torch.empty((8, 0), device=device, dtype=torch.float16).contiguous()
        c_k0 = torch.full((16, 8), 3.0, device=device, dtype=torch.float32).contiguous()
        cublas_scaled_fp16_gemm(a_k0, b_k0, c_k0, alpha=1.0, beta=2.5)
        assert float((c_k0 - 7.5).abs().max().item()) < 1e-5

    # IEEE Triton rational-lag kernel vs CPU FP64
    t_gpu = torch.arange(32, dtype=torch.float32, device=device).unsqueeze(0)
    lag_gpu, ratio_gpu = extract_rational_lag_features(t_gpu, use_triton=True)
    lag_cpu, ratio_cpu = extract_rational_lag_features(t_gpu.cpu(), use_triton=False)
    assert float((lag_gpu.cpu() - lag_cpu).abs().max().item()) < 1e-5
    assert float((ratio_gpu.cpu() - ratio_cpu).abs().max().item()) < 1e-5


@pytest.mark.skipif(not (HAS_CUDA and is_cuda_ipcg_available()), reason="CUDA GPU + nvcc required")
def test_cuda_sm90_ipcg_and_online_ridge_engine() -> None:
    from tsjit.cuda.loader import OnlineGpuRidgeEngine

    device = torch.device("cuda:0")
    eng = OnlineGpuRidgeEngine(num_series=32, feature_dim=64, ridge_lambda=1e-2, device=device)
    gen = torch.Generator(device="cpu").manual_seed(42)
    X = torch.randn((32, 16, 64), generator=gen, dtype=torch.float64).to(device)
    y = torch.randn((32, 16), generator=gen, dtype=torch.float64).to(device)
    eng.update(X[:, :8, :], y[:, :8])
    eng.update(X[:, 8:, :], y[:, 8:])
    sol = eng.solve(tol=1e-9, max_iters=16)
    assert bool(sol.converged.all().item())

    A, rhs = eng.build_system()
    L_ref = torch.linalg.cholesky(A[:, :, :64])
    ref = torch.cholesky_solve(rhs[:, :64].unsqueeze(-1), L_ref).squeeze(-1)
    assert float((sol.theta - ref).abs().max().item()) < 1e-8

    # Exercise fused tsjit_eval_ridge_online_f64 C ABI launcher directly
    xq = torch.randn((32, 64), generator=gen, dtype=torch.float64).to(device)
    sol_fused, y_pred = eng.eval_ridge_online(xq, tol=1e-9, max_iters=16)
    assert bool(sol_fused.converged.all().item())
    assert y_pred is not None
    y_ref = (xq * ref).sum(dim=-1)
    assert float((y_pred - y_ref).abs().max().item()) < 1e-7
