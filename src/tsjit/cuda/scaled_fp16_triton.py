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

"""Hardened Crest-Guarded Scaled-FP16 & Exact-FP32 FP16x3 Ozaki TensorCore Engine.

Key numerical invariants:
1. Subnormal-safe integer exponent splitting via exact power-of-2 shifts (zero
   1/scale -> +Inf blowup on 1e-40 subnormals, zero fmax(0, NaN) drop).
2. 3-way balanced split power-of-2 epilogue rescaling so ex + ew > 127 with a
   zero-cancelled accumulator C == 0 never produces 0 * Inf = NaN.
3. Dual weight + online activation crest-factor guard (crest > 12.0):
   - crest_w > 12.0 -> 3-GEMM FP16x3 Ozaki (stores W0, W1).
   - crest_w <= 12.0 and crest_x > 12.0 -> 2-GEMM activation-escalated
     (X0 @ W0^T + 2^-11 * X1 @ W0^T, zero W1 VRAM).
   - crest_w <= 12.0 and crest_x <= 12.0 -> 1-GEMM Scaled-FP16 (X0 @ W0^T).
4. graph_safe=True executes with ZERO .item() host synchronizations for safe
   CUDA Graph capture; graph_safe=False uses chunked 16-row FP32 fallback for
   non-finite rows to avoid 3D (N_bad, K, N) OOM.
5. Import-time Hopper SM90 SMEM / register / occupancy budget verification.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .loader import cublas_scaled_fp16_gemm, is_cublas_fp16_available

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - optional in pure-CPU environments
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - optional in CPU-only environments
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    TRITON_AVAILABLE = False

SM90_MAX_SMEM_BYTES = 227 * 1024  # 227 KB dynamic SMEM limit (1 KB headroom on 228 KB SM90)
SM90_MAX_ELEMS_PER_THR = 32
SM90_MAX_EST_REGS = 192
DEFAULT_CREST_THRESHOLD = 12.0
_FP32_MIN_NORMAL = float(np.float32(2.0**-126))  # 1.1754943508222875e-38
_SUBNORMAL_SHIFT = 24
_SUBNORMAL_SCALE = float(2**_SUBNORMAL_SHIFT)  # 16777216.0
_OZAKI_RES_SCALE = 2048.0  # 2^11
_OZAKI_RES_INV_SCALE = 1.0 / 2048.0  # 2^-11

# Verified Hopper SM90 GEMM tile configs: (BM, BN, BK, num_stages, num_warps)
SM90_GEMM_CONFIGS: tuple[tuple[int, int, int, int, int], ...] = (
    (128, 64, 64, 3, 8),
    (64, 128, 64, 3, 8),
    (64, 64, 64, 3, 4),
    (128, 64, 32, 4, 8),
)


@dataclass(frozen=True)
class SM90BudgetReport:
    """Resource budget evaluation for a single SM90 GEMM tile configuration."""

    bm: int
    bn: int
    bk: int
    num_stages: int
    num_warps: int
    smem_bytes: int
    elems_per_thr: float
    est_regs: int
    valid: bool
    violations: tuple[str, ...]


def check_sm90_budget(
    BM: int,
    BN: int,
    BK: int,
    num_stages: int,
    num_warps: int,
    *,
    strict: bool = True,
) -> SM90BudgetReport:
    """Verify SM90 shared-memory, accumulator-per-thread, and peak register budgets."""
    if min(BM, BN, BK, num_stages, num_warps) <= 0:
        raise ValueError("All tile/stage/warp parameters must be positive integers.")

    threads = num_warps * 32
    # Multi-stage pipelined FP16 A(BM, BK) + B(BK, BN) staging in shared memory.
    smem_bytes = num_stages * (BM * BK + BN * BK) * 2
    elems_per_thr = (BM * BN) / float(threads)
    # Conservative register model for dual-accumulator FP16x3 (2 * elems_per_thr FP32 acc)
    # + MMA operand fragments + scale/pointer/epilogue state.
    mma_frag_regs = math.ceil((BM * 16 + BN * 16) / float(threads))
    est_regs = int(math.ceil(2.0 * elems_per_thr + mma_frag_regs + 48))

    violations: list[str] = []
    if smem_bytes > SM90_MAX_SMEM_BYTES:
        violations.append(
            f"smem_bytes={smem_bytes} exceeds SM90 budget {SM90_MAX_SMEM_BYTES} (227 KB)"
        )
    if elems_per_thr > SM90_MAX_ELEMS_PER_THR:
        violations.append(f"elems_per_thr={elems_per_thr:.1f} exceeds {SM90_MAX_ELEMS_PER_THR}")
    if est_regs > SM90_MAX_EST_REGS:
        violations.append(f"est_regs={est_regs} exceeds {SM90_MAX_EST_REGS}")

    report = SM90BudgetReport(
        bm=BM,
        bn=BN,
        bk=BK,
        num_stages=num_stages,
        num_warps=num_warps,
        smem_bytes=smem_bytes,
        elems_per_thr=elems_per_thr,
        est_regs=est_regs,
        valid=len(violations) == 0,
        violations=tuple(violations),
    )
    if strict and not report.valid:
        raise ValueError(
            f"SM90 budget check failed for ({BM}, {BN}, {BK}, stages={num_stages}, "
            f"warps={num_warps}): {'; '.join(violations)}"
        )
    return report


# Import-time SM90 budget verification across all tuned configurations.
SM90_VERIFIED_REPORTS: tuple[SM90BudgetReport, ...] = tuple(
    check_sm90_budget(*cfg, strict=True) for cfg in SM90_GEMM_CONFIGS
)


# ---------------------------------------------------------------------------
# Pure-NumPy exact mathematical reference (subnormal-safe, 0*Inf-safe)
# ---------------------------------------------------------------------------


def _pow2_f32_np(exp_i32: np.ndarray) -> np.ndarray:
    """Construct exact normal FP32 powers of two ``2**exp_i32`` for ``exp in [-126, 127]``."""
    bits = (exp_i32.astype(np.int32) + np.int32(127)) << np.int32(23)
    return bits.view(np.float32)


def _split_rows_numpy(
    mat: np.ndarray,
    *,
    need_v1: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray]:
    """Subnormal-safe IEEE binary-exponent row split into (v0, v1, exp_e, bad_code, crest)."""
    x = np.asarray(mat, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {x.shape}")
    M, K = x.shape

    abs_x = np.abs(x)
    is_fin = (abs_x < np.float32(np.inf)) & (x == x)
    has_nonfin = ~np.all(is_fin, axis=1)
    x_safe = np.where(is_fin, x, np.float32(0.0))
    amax = np.max(np.abs(x_safe), axis=1) if K > 0 else np.zeros(M, dtype=np.float32)

    bad_code = np.zeros(M, dtype=np.int32)
    bad_code = np.where(amax == np.float32(0.0), np.int32(1), bad_code)
    bad_code = np.where(has_nonfin, np.int32(2), bad_code)

    is_sub = amax < np.float32(_FP32_MIN_NORMAL)
    amax_sub_only = np.where(is_sub, amax, np.float32(0.0))
    amax_norm = np.where(is_sub, amax_sub_only * np.float32(_SUBNORMAL_SCALE), amax)
    sub_bias = np.where(is_sub, np.int32(_SUBNORMAL_SHIFT), np.int32(0))
    amax_for_bits = np.where(bad_code == 0, amax_norm, np.float32(1.0)).astype(np.float32)
    bits = amax_for_bits.view(np.int32)
    exp_e = np.where(
        bad_code == 0,
        ((bits >> np.int32(23)) & np.int32(0xFF)) - np.int32(127) - sub_bias,
        np.int32(0),
    ).astype(np.int32)

    # Exact power-of-2 shift by -exp_e split into two normal factors (handles subnormals).
    sh = -exp_e
    sh1 = sh // 2
    sh2 = sh - sh1
    p1 = _pow2_f32_np(sh1)[:, None]
    p2 = _pow2_f32_np(sh2)[:, None]

    valid_row = (bad_code == 0)[:, None]
    v_s = np.where(valid_row, (x_safe * p1) * p2, np.float32(0.0))

    # Scale-invariant FP64 RMS crest factor computed on shifted v_s in (-2, 2).
    v_s_f64 = v_s.astype(np.float64)
    rms_s = np.sqrt(np.mean(v_s_f64 * v_s_f64, axis=1)) if K > 0 else np.zeros(M, dtype=np.float64)
    amax_s = np.where(bad_code == 0, (amax * p1[:, 0]) * p2[:, 0], np.float32(0.0)).astype(
        np.float64
    )
    crest = np.where(
        (bad_code == 0) & (rms_s > 0.0), amax_s / np.maximum(rms_s, 1e-30), 0.0
    ).astype(np.float32)

    v0 = v_s.astype(np.float16)
    v1: np.ndarray | None = None
    if need_v1:
        res = (v_s - v0.astype(np.float32)) * np.float32(_OZAKI_RES_SCALE)
        v1 = res.astype(np.float16)

    return v0, v1, exp_e, bad_code, crest


def _balanced_rescale_numpy(
    C: np.ndarray,
    ex: np.ndarray,
    ew: np.ndarray,
    bx: np.ndarray,
    bw: np.ndarray,
    bias: np.ndarray | None = None,
) -> np.ndarray:
    """Apply 3-way balanced power-of-2 rescaling + bad_code semantics in FP32."""
    e_tot = np.clip(ex[:, None].astype(np.int32) + ew[None, :].astype(np.int32), -300, 300)
    e1 = e_tot // 3
    rem = e_tot - e1
    e2 = rem // 2
    e3 = rem - e2

    s1 = _pow2_f32_np(e1)
    s2 = _pow2_f32_np(e2)
    s3 = _pow2_f32_np(e3)

    out = ((C.astype(np.float32) * s1) * s2) * s3
    is_bad2 = (bx[:, None] == 2) | (bw[None, :] == 2)
    is_zero = ((bx[:, None] == 1) | (bw[None, :] == 1)) & (~is_bad2)
    out = np.where(is_zero, np.float32(0.0), out)
    if bias is not None:
        out = out + np.asarray(bias, dtype=np.float32)[None, :]
    out = np.where(is_bad2, np.float32(np.nan), out)
    return out


def crest_guarded_linear_numpy(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    *,
    mode: str = "auto",
    crest_threshold: float = DEFAULT_CREST_THRESHOLD,
    graph_safe: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Pure-NumPy exact reference for Crest-Guarded Scaled-FP16 / FP16x3 linear."""
    x_arr = np.asarray(x, dtype=np.float32)
    w_arr = np.asarray(weight, dtype=np.float32)
    orig_shape = x_arr.shape
    K = orig_shape[-1]
    if w_arr.ndim != 2 or w_arr.shape[1] != K:
        raise ValueError(f"Incompatible shapes x={orig_shape}, weight={w_arr.shape}")

    M = int(np.prod(orig_shape[:-1])) if len(orig_shape) > 1 else 1
    N = w_arr.shape[0]
    if M == 0 or N == 0 or K == 0:
        out_empty = np.zeros((*orig_shape[:-1], N), dtype=np.float32)
        if bias is not None and M > 0 and N > 0:
            out_empty = out_empty + np.asarray(bias, dtype=np.float32)
        return out_empty, {
            "tier": "scaled-fp16-1gemm",
            "max_crest_w": 0.0,
            "max_crest_x": 0.0,
            "escalated_w": False,
            "escalated_x": False,
        }

    x_2d = x_arr.reshape(M, K)

    w0, w1, ew, bw, crest_w = _split_rows_numpy(w_arr, need_v1=True)
    max_crest_w = float(np.max(crest_w)) if N > 0 else 0.0
    escalated_w = (mode == "fp16x3") or (mode == "auto" and max_crest_w > crest_threshold)

    x0, x1, ex, bx, crest_x = _split_rows_numpy(x_2d, need_v1=(mode != "scaled-fp16"))
    max_crest_x = float(np.max(crest_x)) if M > 0 else 0.0
    escalated_x = (mode == "auto") and (not escalated_w) and (max_crest_x > crest_threshold)

    x0_f = x0.astype(np.float32)
    w0_f = w0.astype(np.float32)
    C = x0_f @ w0_f.T

    if escalated_w:
        assert x1 is not None and w1 is not None
        C_res = x1.astype(np.float32) @ w0_f.T + x0_f @ w1.astype(np.float32).T
        C = C + np.float32(_OZAKI_RES_INV_SCALE) * C_res
        tier = "fp16x3-3gemm"
    elif escalated_x:
        assert x1 is not None
        C_res = x1.astype(np.float32) @ w0_f.T
        C = C + np.float32(_OZAKI_RES_INV_SCALE) * C_res
        tier = "act-escalated-2gemm"
    else:
        tier = "scaled-fp16-1gemm"

    out = _balanced_rescale_numpy(C, ex, ew, bx, bw, bias=bias)

    if not graph_safe:
        bad_rows = np.nonzero(bx == 2)[0]
        if bad_rows.size > 0 or bool(np.any(bw == 2)):
            rows_to_fix = np.arange(M) if bool(np.any(bw == 2)) else bad_rows
            for start in range(0, rows_to_fix.size, 16):
                idx = rows_to_fix[start : start + 16]
                ref = x_2d[idx] @ w_arr.T
                if bias is not None:
                    ref = ref + np.asarray(bias, dtype=np.float32)[None, :]
                out[idx] = ref

    meta = {
        "tier": tier,
        "max_crest_w": max_crest_w,
        "max_crest_x": max_crest_x,
        "escalated_w": escalated_w,
        "escalated_x": escalated_x,
    }
    return out.reshape(*orig_shape[:-1], N), meta


# ---------------------------------------------------------------------------
# Triton SM90 Kernels (Two-Pass K-Tiled Split + Adaptive/FP16x3 GEMM)
# ---------------------------------------------------------------------------

if TRITON_AVAILABLE:

    @triton.jit
    def _pow2_f32_triton(exp_i32):
        return ((exp_i32.to(tl.int32) + 127) << 23).to(tl.float32, bitcast=True)

    @triton.jit
    def _split_rows_triton_kernel(
        X_ptr,
        V0_ptr,
        V1_ptr,
        EXP_ptr,
        BAD_ptr,
        CREST_ptr,
        M,
        K,
        X_STRIDE,
        V_STRIDE,
        STORE_V1: tl.constexpr,
        BK_TILE: tl.constexpr,
    ):
        row = tl.program_id(0)
        if row >= M:
            return

        row_i64 = row.to(tl.int64)
        row_in = X_ptr + row_i64 * X_STRIDE
        amax = tl.full((), 0.0, dtype=tl.float32)
        bad_cnt = tl.full((), 0, dtype=tl.int32)

        # Pass 1: K-tiled finiteness check + exact amax (never drops NaN via fmax).
        # Uses exact IEEE-754 FLT_MAX (3.4028234663852886e+38) without Python float("inf").
        for k0 in range(0, K, BK_TILE):
            offs_k = k0 + tl.arange(0, BK_TILE)
            mask = offs_k < K
            v = tl.load(row_in + offs_k, mask=mask, other=0.0).to(tl.float32)
            abs_v = tl.abs(v)
            is_fin = (abs_v <= 3.4028234663852886e38) & (v == v)
            bad_cnt += tl.sum(tl.where(mask & (~is_fin), 1, 0), axis=0)
            v_safe = tl.where(mask & is_fin, v, 0.0)
            tile_max = tl.max(tl.abs(v_safe), axis=0)
            amax = tl.maximum(amax, tile_max)

        bad_code = tl.where(bad_cnt > 0, 2, tl.where(amax == 0.0, 1, 0)).to(tl.int32)

        # Subnormal-safe unbiased binary exponent extraction.
        is_sub = amax < 1.1754943508222875e-38
        amax_norm = tl.where(is_sub, amax * 16777216.0, amax)
        sub_bias = tl.where(is_sub, 24, 0)
        amax_safe = tl.where(bad_code == 0, amax_norm, 1.0).to(tl.float32)
        bits = amax_safe.to(tl.int32, bitcast=True)
        exp_e = tl.where(bad_code == 0, ((bits >> 23) & 0xFF) - 127 - sub_bias, 0).to(tl.int32)

        # Split exact shift -exp_e into two normal FP32 powers of two.
        sh = -exp_e
        sh1 = sh // 2
        sh2 = sh - sh1
        p1 = _pow2_f32_triton(sh1)
        p2 = _pow2_f32_triton(sh2)

        row_v0 = V0_ptr + row_i64 * V_STRIDE
        row_v1 = V1_ptr + row_i64 * V_STRIDE
        sum_sq = tl.full((), 0.0, dtype=tl.float32)

        # Pass 2: Power-of-2 shift -> FP16 split (v0, v1) + scale-free crest sum_sq.
        is_valid_row = bad_code == 0
        for k0 in range(0, K, BK_TILE):
            offs_k = k0 + tl.arange(0, BK_TILE)
            mask = offs_k < K
            v = tl.load(row_in + offs_k, mask=mask, other=0.0).to(tl.float32)
            v_clean = tl.where(mask & is_valid_row, v, 0.0)
            v_s = (v_clean * p1) * p2
            sum_sq += tl.sum(tl.where(mask, v_s * v_s, 0.0), axis=0)
            v0 = v_s.to(tl.float16)
            tl.store(row_v0 + offs_k, v0, mask=mask)
            if STORE_V1:
                v1 = ((v_s - v0.to(tl.float32)) * 2048.0).to(tl.float16)
                tl.store(row_v1 + offs_k, v1, mask=mask)

        amax_s = tl.where(is_valid_row, (amax * p1) * p2, 0.0)
        k_f32 = K.to(tl.float32)
        rms_s = tl.sqrt(tl.maximum(sum_sq / k_f32, 1e-30))
        crest = tl.where(is_valid_row, amax_s / rms_s, 0.0)

        tl.store(EXP_ptr + row, exp_e)
        tl.store(BAD_ptr + row, bad_code)
        tl.store(CREST_ptr + row, crest)

    _TRITON_SM90_AUTOTUNE_CONFIGS = [
        triton.Config({"BM": bm, "BN": bn, "BK": bk}, num_stages=st, num_warps=nw)
        for (bm, bn, bk, st, nw) in SM90_GEMM_CONFIGS
    ]

    @triton.autotune(
        configs=_TRITON_SM90_AUTOTUNE_CONFIGS,
        key=["M", "N", "K", "MODE_ID", "HAS_BIAS"],
    )
    @triton.jit
    def _scaled_fp16_gemm_triton_kernel(
        X0_ptr,
        X1_ptr,
        W0_ptr,
        W1_ptr,
        EX_ptr,
        BX_ptr,
        CREST_X_ptr,
        EW_ptr,
        BW_ptr,
        BIAS_ptr,
        OUT_ptr,
        M,
        N,
        K,
        X_STRIDE,
        W_STRIDE,
        OUT_STRIDE,
        CREST_THRESH,
        MODE_ID: tl.constexpr,  # 0=1-GEMM, 1=adaptive 1/2-GEMM, 2=3-GEMM FP16x3
        HAS_BIAS: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_m_i64 = offs_m.to(tl.int64)
        offs_n_i64 = offs_n.to(tl.int64)
        m_mask = offs_m < M
        n_mask = offs_n < N

        acc0 = tl.zeros((BM, BN), dtype=tl.float32)

        if MODE_ID == 2:
            # Single-loop fused 3-GEMM FP16x3 Ozaki: loads X0 and W0 ONCE from HBM.
            acc_res = tl.zeros((BM, BN), dtype=tl.float32)
            for k0 in range(0, K, BK):
                offs_k = k0 + tl.arange(0, BK)
                k_mask = offs_k < K
                x_ptrs = offs_m_i64[:, None] * X_STRIDE + offs_k[None, :]
                w_ptrs = offs_n_i64[None, :] * W_STRIDE + offs_k[:, None]
                xm = m_mask[:, None] & k_mask[None, :]
                wm = n_mask[None, :] & k_mask[:, None]
                a0 = tl.load(X0_ptr + x_ptrs, mask=xm, other=0.0)
                b0 = tl.load(W0_ptr + w_ptrs, mask=wm, other=0.0)
                acc0 = tl.dot(a0, b0, acc=acc0, out_dtype=tl.float32)
                a1 = tl.load(X1_ptr + x_ptrs, mask=xm, other=0.0)
                acc_res = tl.dot(a1, b0, acc=acc_res, out_dtype=tl.float32)
                b1 = tl.load(W1_ptr + w_ptrs, mask=wm, other=0.0)
                acc_res = tl.dot(a0, b1, acc=acc_res, out_dtype=tl.float32)
            acc0 = acc0 + acc_res * (1.0 / 2048.0)
        else:
            # Primary 1-GEMM term: X0 @ W0^T in FP16 Tensor Cores with FP32 accumulation.
            for k0 in range(0, K, BK):
                offs_k = k0 + tl.arange(0, BK)
                k_mask = offs_k < K
                a0 = tl.load(
                    X0_ptr + offs_m_i64[:, None] * X_STRIDE + offs_k[None, :],
                    mask=m_mask[:, None] & k_mask[None, :],
                    other=0.0,
                )
                b0 = tl.load(
                    W0_ptr + offs_n_i64[None, :] * W_STRIDE + offs_k[:, None],
                    mask=n_mask[None, :] & k_mask[:, None],
                    other=0.0,
                )
                acc0 = tl.dot(a0, b0, acc=acc0, out_dtype=tl.float32)

            if MODE_ID == 1:
                # Zero-sync per-tile adaptive activation escalation (graph_safe compatible!).
                # num_stages=1 on the conditional loop avoids SM90 MLIR pipeliner issues on 0-trip.
                cx = tl.load(CREST_X_ptr + offs_m, mask=m_mask, other=0.0)
                tile_needs_x1 = tl.max(cx, axis=0) > CREST_THRESH
                k_end_x1 = tl.where(tile_needs_x1, K, 0)
                acc_res = tl.zeros((BM, BN), dtype=tl.float32)
                for k0 in tl.range(0, k_end_x1, BK, num_stages=1):
                    offs_k = k0 + tl.arange(0, BK)
                    k_mask = offs_k < K
                    a1 = tl.load(
                        X1_ptr + offs_m_i64[:, None] * X_STRIDE + offs_k[None, :],
                        mask=m_mask[:, None] & k_mask[None, :],
                        other=0.0,
                    )
                    b0 = tl.load(
                        W0_ptr + offs_n_i64[None, :] * W_STRIDE + offs_k[:, None],
                        mask=n_mask[None, :] & k_mask[:, None],
                        other=0.0,
                    )
                    acc_res = tl.dot(a1, b0, acc=acc_res, out_dtype=tl.float32)
                acc0 = acc0 + acc_res * (1.0 / 2048.0)

        # Balanced 3-way split power-of-2 rescaling: zero 0*Inf=NaN for ex+ew > 127.
        ex = tl.load(EX_ptr + offs_m, mask=m_mask, other=0)
        bx = tl.load(BX_ptr + offs_m, mask=m_mask, other=1)
        ew = tl.load(EW_ptr + offs_n, mask=n_mask, other=0)
        bw = tl.load(BW_ptr + offs_n, mask=n_mask, other=1)

        e_tot = tl.maximum(tl.minimum(ex[:, None] + ew[None, :], 300), -300)
        e1 = e_tot // 3
        rem = e_tot - e1
        e2 = rem // 2
        e3 = rem - e2

        s1 = _pow2_f32_triton(e1)
        s2 = _pow2_f32_triton(e2)
        s3 = _pow2_f32_triton(e3)

        out = ((acc0 * s1) * s2) * s3
        is_bad2 = (bx[:, None] == 2) | (bw[None, :] == 2)
        is_zero = ((bx[:, None] == 1) | (bw[None, :] == 1)) & (~is_bad2)
        out = tl.where(is_zero, 0.0, out)

        if HAS_BIAS:
            bias = tl.load(BIAS_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
            out = out + bias[None, :]

        qnan_f32 = tl.full((), 0x7FC00000, dtype=tl.int32).to(tl.float32, bitcast=True)
        out = tl.where(is_bad2, qnan_f32, out)
        tl.store(
            OUT_ptr + offs_m_i64[:, None] * OUT_STRIDE + offs_n_i64[None, :],
            out,
            mask=m_mask[:, None] & n_mask[None, :],
        )

    @triton.jit
    def _rescale_epilogue_triton_kernel(
        C_ptr,
        EX_ptr,
        BX_ptr,
        EW_ptr,
        BW_ptr,
        BIAS_ptr,
        M,
        N,
        C_STRIDE,
        HAS_BIAS: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
    ):
        """Fused in-place 3-way balanced power-of-2 rescale + bias + bad-row-NaN epilogue.

        Pairs with Hopper cuBLAS WGMMA (CUBLAS_COMPUTE_32F) GEMM outputs while preserving
        exact subnormal handling, zero 0*Inf = NaN on ex+ew > 127, and NaN/zero semantics.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_m_i64 = offs_m.to(tl.int64)
        offs_n_i64 = offs_n.to(tl.int64)
        m_mask = offs_m < M
        n_mask = offs_n < N
        tile_mask = m_mask[:, None] & n_mask[None, :]

        ptrs = C_ptr + offs_m_i64[:, None] * C_STRIDE + offs_n_i64[None, :]
        acc0 = tl.load(ptrs, mask=tile_mask, other=0.0).to(tl.float32)

        ex = tl.load(EX_ptr + offs_m, mask=m_mask, other=0)
        bx = tl.load(BX_ptr + offs_m, mask=m_mask, other=1)
        ew = tl.load(EW_ptr + offs_n, mask=n_mask, other=0)
        bw = tl.load(BW_ptr + offs_n, mask=n_mask, other=1)

        e_tot = tl.maximum(tl.minimum(ex[:, None] + ew[None, :], 300), -300)
        e1 = e_tot // 3
        rem = e_tot - e1
        e2 = rem // 2
        e3 = rem - e2

        s1 = _pow2_f32_triton(e1)
        s2 = _pow2_f32_triton(e2)
        s3 = _pow2_f32_triton(e3)

        out = ((acc0 * s1) * s2) * s3
        is_bad2 = (bx[:, None] == 2) | (bw[None, :] == 2)
        is_zero = ((bx[:, None] == 1) | (bw[None, :] == 1)) & (~is_bad2)
        out = tl.where(is_zero, 0.0, out)

        if HAS_BIAS:
            bias = tl.load(BIAS_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
            out = out + bias[None, :]

        qnan_f32 = tl.full((), 0x7FC00000, dtype=tl.int32).to(tl.float32, bitcast=True)
        out = tl.where(is_bad2, qnan_f32, out)
        tl.store(ptrs, out, mask=tile_mask)


# ---------------------------------------------------------------------------
# PyTorch CPU / CUDA / Triton Engine & nn.Module Wrapper
# ---------------------------------------------------------------------------


def _pow2_f32_torch(exp_i32: Any) -> Any:
    bits = (exp_i32.to(torch.int32) + 127) << 23
    return bits.view(torch.float32)


def _split_rows_torch(
    mat: Any,
    *,
    need_v1: bool = True,
) -> tuple[Any, Any | None, Any, Any, Any]:
    """Split 2D tensor ``(M, K)`` using Triton on CUDA or exact PyTorch fallback."""
    if mat.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got {tuple(mat.shape)}")
    M, K = mat.shape

    if mat.stride(-1) != 1:
        mat = mat.contiguous()

    if mat.is_cuda and TRITON_AVAILABLE and M > 0 and K > 0:
        with torch.cuda.device(mat.device):
            x_in = mat if mat.dtype == torch.float32 else mat.to(torch.float32)
            v0 = torch.empty((M, K), dtype=torch.float16, device=mat.device)
            v1 = torch.empty((M, K), dtype=torch.float16, device=mat.device) if need_v1 else v0
            exp_e = torch.empty(M, dtype=torch.int32, device=mat.device)
            bad_code = torch.empty(M, dtype=torch.int32, device=mat.device)
            crest = torch.empty(M, dtype=torch.float32, device=mat.device)
            bk_tile = 512 if K >= 512 else 256
            _split_rows_triton_kernel[(M,)](
                x_in,
                v0,
                v1,
                exp_e,
                bad_code,
                crest,
                M,
                K,
                x_in.stride(0),
                v0.stride(0),
                STORE_V1=need_v1,
                BK_TILE=bk_tile,
            )
        return v0, (v1 if need_v1 else None), exp_e, bad_code, crest

    # Pure-PyTorch exact fallback (CPU or CUDA without Triton)
    x = mat.to(torch.float32)
    abs_x = x.abs()
    is_fin = (abs_x < float("inf")) & (x == x)
    has_nonfin = ~is_fin.all(dim=1)
    x_safe = torch.where(is_fin, x, torch.zeros_like(x))
    amax = (
        x_safe.abs().amax(dim=1) if K > 0 else torch.zeros(M, dtype=torch.float32, device=x.device)
    )

    bad_code = torch.zeros(M, dtype=torch.int32, device=x.device)
    bad_code = torch.where(amax == 0.0, torch.ones_like(bad_code), bad_code)
    bad_code = torch.where(has_nonfin, torch.full_like(bad_code, 2), bad_code)

    is_sub = amax < _FP32_MIN_NORMAL
    amax_norm = torch.where(is_sub, amax * _SUBNORMAL_SCALE, amax)
    sub_bias = torch.where(
        is_sub, torch.full_like(bad_code, _SUBNORMAL_SHIFT), torch.zeros_like(bad_code)
    )
    amax_for_bits = torch.where(bad_code == 0, amax_norm, torch.ones_like(amax_norm))
    bits = amax_for_bits.view(torch.int32)
    exp_e = torch.where(
        bad_code == 0,
        ((bits >> 23) & 0xFF) - 127 - sub_bias,
        torch.zeros_like(bad_code),
    ).to(torch.int32)

    sh = -exp_e
    sh1 = torch.div(sh, 2, rounding_mode="floor")
    sh2 = sh - sh1
    p1 = _pow2_f32_torch(sh1).unsqueeze(1)
    p2 = _pow2_f32_torch(sh2).unsqueeze(1)

    valid_row = (bad_code == 0).unsqueeze(1)
    v_s = torch.where(valid_row, (x_safe * p1) * p2, torch.zeros_like(x_safe))

    v_s_f64 = v_s.to(torch.float64)
    rms_s = (
        torch.sqrt((v_s_f64 * v_s_f64).mean(dim=1))
        if K > 0
        else torch.zeros(M, dtype=torch.float64, device=x.device)
    )
    amax_s = torch.where(bad_code == 0, (amax * p1[:, 0]) * p2[:, 0], torch.zeros_like(amax)).to(
        torch.float64
    )
    crest = torch.where(
        (bad_code == 0) & (rms_s > 0.0),
        amax_s / torch.clamp(rms_s, min=1e-30),
        torch.zeros_like(amax_s),
    ).to(torch.float32)

    v0 = v_s.to(torch.float16)
    v1 = ((v_s - v0.to(torch.float32)) * _OZAKI_RES_SCALE).to(torch.float16) if need_v1 else None
    return v0, v1, exp_e, bad_code, crest


@dataclass
class PackedExactWeight:
    """Offline crest-checked and exponent-split weight representation."""

    w0: Any
    w1: Any | None
    exp_w: Any
    bad_w: Any
    crest_w: Any
    max_crest_w: float
    escalated_w: bool
    has_nonfinite_w: bool
    out_features: int
    in_features: int

    @classmethod
    def pack(
        cls,
        weight: Any,
        *,
        mode: str = "auto",
        crest_threshold: float = DEFAULT_CREST_THRESHOLD,
    ) -> PackedExactWeight:
        if torch is None or not isinstance(weight, torch.Tensor):
            w_arr = np.asarray(weight, dtype=np.float32)
            w0, w1, ew, bw, cw = _split_rows_numpy(w_arr, need_v1=True)
            max_cw = float(np.max(cw)) if cw.size > 0 else 0.0
            esc_w = (mode == "fp16x3") or (mode == "auto" and max_cw > crest_threshold)
            return cls(
                w0=w0,
                w1=w1 if esc_w else None,
                exp_w=ew,
                bad_w=bw,
                crest_w=cw,
                max_crest_w=max_cw,
                escalated_w=esc_w,
                has_nonfinite_w=bool(np.any(bw == 2)),
                out_features=int(w_arr.shape[0]),
                in_features=int(w_arr.shape[1]),
            )

        with torch.no_grad():
            w0, w1, ew, bw, cw = _split_rows_torch(weight.detach(), need_v1=True)
            max_cw = float(cw.max().item()) if cw.numel() > 0 else 0.0
            esc_w = (mode == "fp16x3") or (mode == "auto" and max_cw > crest_threshold)
            return cls(
                w0=w0.contiguous(),
                w1=w1.contiguous() if (esc_w and w1 is not None) else None,
                exp_w=ew.contiguous(),
                bad_w=bw.contiguous(),
                crest_w=cw.contiguous(),
                max_crest_w=max_cw,
                escalated_w=esc_w,
                has_nonfinite_w=bool((bw == 2).any().item()),
                out_features=int(weight.shape[0]),
                in_features=int(weight.shape[1]),
            )


def _apply_rescale_epilogue_torch(
    C: Any,
    ex: Any,
    bx: Any,
    ew: Any,
    bw: Any,
    bias: Any | None,
) -> Any:
    """Apply exact 3-way balanced power-of-2 rescale + bias + bad_code semantics in PyTorch."""
    e_tot = torch.clamp(ex.unsqueeze(1) + ew.unsqueeze(0), -300, 300)
    e1 = torch.div(e_tot, 3, rounding_mode="floor")
    rem = e_tot - e1
    e2 = torch.div(rem, 2, rounding_mode="floor")
    e3 = rem - e2
    out = ((C * _pow2_f32_torch(e1)) * _pow2_f32_torch(e2)) * _pow2_f32_torch(e3)

    is_bad2 = (bx.unsqueeze(1) == 2) | (bw.unsqueeze(0) == 2)
    is_zero = ((bx.unsqueeze(1) == 1) | (bw.unsqueeze(0) == 1)) & (~is_bad2)
    out = torch.where(is_zero, torch.zeros_like(out), out)
    if bias is not None:
        out = out + bias.to(torch.float32).unsqueeze(0)
    return torch.where(is_bad2, torch.full_like(out, float("nan")), out)


def _gemm_from_packed_torch(
    x: Any,
    packed: PackedExactWeight,
    bias: Any | None,
    *,
    mode: str,
    crest_threshold: float,
    graph_safe: bool,
    raw_weight_for_fallback: Any | None = None,
) -> Any:
    orig_shape = x.shape
    K = orig_shape[-1]
    if K != packed.in_features:
        raise ValueError(f"Expected last dim {packed.in_features}, got {K}")

    # Preserve non-contiguous row-strided 2D tensors without unnecessary copy.
    if x.ndim == 2:
        x_2d = x if x.stride(-1) == 1 else x.contiguous()
    else:
        x_2d = x.reshape(-1, K)
        if x_2d.stride(-1) != 1:
            x_2d = x_2d.contiguous()

    M = x_2d.shape[0]
    N = packed.out_features
    if M == 0 or N == 0 or K == 0:
        out_empty = torch.zeros((*orig_shape[:-1], N), dtype=torch.float32, device=x.device)
        if bias is not None and M > 0 and N > 0:
            out_empty = out_empty + bias.to(torch.float32)
        return out_empty

    need_x1 = (mode == "fp16x3") or packed.escalated_w or (mode == "auto")
    x0, x1, ex, bx, crest_x = _split_rows_torch(x_2d, need_v1=need_x1)

    if x_2d.is_cuda and is_cublas_fp16_available(x_2d.device):
        with torch.cuda.device(x_2d.device):
            out = torch.empty((M, N), dtype=torch.float32, device=x.device)
            w0_c = packed.w0 if packed.w0.is_contiguous() else packed.w0.contiguous()
            x0_c = x0 if x0.is_contiguous() else x0.contiguous()

            # 1st WGMMA GEMM: C = X0 @ W0^T with exact CUBLAS_COMPUTE_32F accumulation.
            cublas_scaled_fp16_gemm(x0_c, w0_c, out, alpha=1.0, beta=0.0)

            if (packed.escalated_w or mode == "fp16x3") and packed.w1 is not None:
                # 3-GEMM FP16x3 Ozaki: C += 2^-11 * (X1 @ W0^T + X0 @ W1^T).
                assert x1 is not None
                x1_c = x1 if x1.is_contiguous() else x1.contiguous()
                w1_c = packed.w1 if packed.w1.is_contiguous() else packed.w1.contiguous()
                cublas_scaled_fp16_gemm(x1_c, w0_c, out, alpha=_OZAKI_RES_INV_SCALE, beta=1.0)
                cublas_scaled_fp16_gemm(x0_c, w1_c, out, alpha=_OZAKI_RES_INV_SCALE, beta=1.0)
            elif mode == "auto" and x1 is not None:
                # cuBLAS WGMMA mode='auto' contract:
                # - graph_safe=True runs a graph-capturable 2-GEMM activation-outlier-safe path
                #   (zero W1 VRAM, zero host .item() sync); use mode='scaled-fp16' for 1-GEMM.
                # - graph_safe=False dynamically skips the 2nd GEMM when crest_x <= crest_threshold.
                run_x1 = True if graph_safe else bool((crest_x > crest_threshold).any().item())
                if run_x1:
                    x1_c = x1 if x1.is_contiguous() else x1.contiguous()
                    cublas_scaled_fp16_gemm(x1_c, w0_c, out, alpha=_OZAKI_RES_INV_SCALE, beta=1.0)

            # Fused in-place 3-way balanced power-of-2 rescale + bias + bad-row-NaN epilogue.
            if TRITON_AVAILABLE:
                bias_arg = bias.contiguous() if bias is not None else out
                if M * N <= 65536:
                    bm_ep, bn_ep, nw_ep = 16, 64, 2
                else:
                    bm_ep, bn_ep, nw_ep = 32, 128, 4
                grid_ep = (triton.cdiv(M, bm_ep), triton.cdiv(N, bn_ep))
                _rescale_epilogue_triton_kernel[grid_ep](
                    out,
                    ex,
                    bx,
                    packed.exp_w,
                    packed.bad_w,
                    bias_arg,
                    M,
                    N,
                    out.stride(0),
                    HAS_BIAS=(bias is not None),
                    BM=bm_ep,
                    BN=bn_ep,
                    num_warps=nw_ep,
                )
            else:
                out = _apply_rescale_epilogue_torch(out, ex, bx, packed.exp_w, packed.bad_w, bias)
    elif x_2d.is_cuda and TRITON_AVAILABLE:
        with torch.cuda.device(x_2d.device):
            out = torch.empty((M, N), dtype=torch.float32, device=x.device)
            if (packed.escalated_w or mode == "fp16x3") and packed.w1 is not None:
                mode_id = 2
                x1_arg = x1
                w1_arg = packed.w1
            elif mode == "auto":
                mode_id = 1
                x1_arg = x1
                w1_arg = packed.w0  # unused in MODE_ID=1
            else:
                mode_id = 0
                x1_arg = x0  # unused in MODE_ID=0
                w1_arg = packed.w0

            bias_arg = bias.contiguous() if bias is not None else out

            def _grid(meta: dict[str, Any]) -> tuple[int, int]:
                return (triton.cdiv(M, meta["BM"]), triton.cdiv(N, meta["BN"]))

            _scaled_fp16_gemm_triton_kernel[_grid](
                x0,
                x1_arg,
                packed.w0,
                w1_arg,
                ex,
                bx,
                crest_x,
                packed.exp_w,
                packed.bad_w,
                bias_arg,
                out,
                M,
                N,
                K,
                x0.stride(0),
                packed.w0.stride(0),
                out.stride(0),
                float(crest_threshold),
                MODE_ID=mode_id,
                HAS_BIAS=(bias is not None),
            )
    else:
        # Pure-PyTorch fallback (CPU or CUDA without cuBLAS/Triton)
        x0_f = x0.to(torch.float32)
        w0_f = packed.w0.to(torch.float32)
        C = x0_f @ w0_f.T
        if packed.escalated_w or mode == "fp16x3":
            assert x1 is not None and packed.w1 is not None
            C = C + _OZAKI_RES_INV_SCALE * (
                x1.to(torch.float32) @ w0_f.T + x0_f @ packed.w1.to(torch.float32).T
            )
        elif mode == "auto" and x1 is not None:
            high_crest = crest_x > crest_threshold
            if graph_safe:
                mask_f = high_crest.to(torch.float32).unsqueeze(1)
                C = C + (_OZAKI_RES_INV_SCALE * mask_f) * (x1.to(torch.float32) @ w0_f.T)
            elif bool(high_crest.any().item()):
                C = C + _OZAKI_RES_INV_SCALE * (x1.to(torch.float32) @ w0_f.T)

        out = _apply_rescale_epilogue_torch(C, ex, bx, packed.exp_w, packed.bad_w, bias)

    # Chunked 16-row FP32 exact fallback when graph_safe=False (avoids 3D OOM!).
    if (not graph_safe) and raw_weight_for_fallback is not None:
        has_bad_x = bool((bx == 2).any().item())
        if has_bad_x or packed.has_nonfinite_w:
            if packed.has_nonfinite_w:
                bad_rows = torch.arange(M, device=x.device)
            else:
                bad_rows = torch.nonzero(bx == 2, as_tuple=False).squeeze(-1)
            w_f32 = raw_weight_for_fallback.to(torch.float32)
            b_f32 = None if bias is None else bias.to(torch.float32)
            for start in range(0, bad_rows.numel(), 16):
                idx = bad_rows[start : start + 16]
                out[idx] = F.linear(x_2d[idx].to(torch.float32), w_f32, b_f32)

    return out.reshape(*orig_shape[:-1], N)


def scaled_fp16_linear(
    x: Any,
    weight: Any,
    bias: Any | None = None,
    *,
    crest_threshold: float = DEFAULT_CREST_THRESHOLD,
    graph_safe: bool = True,
    mode: str = "scaled-fp16",
) -> Any:
    """Functional Scaled-FP16 (or Crest-Guarded auto) linear projection."""
    if isinstance(x, np.ndarray) or torch is None:
        out, _ = crest_guarded_linear_numpy(
            np.asarray(x),
            np.asarray(weight),
            None if bias is None else np.asarray(bias),
            mode=mode,
            crest_threshold=crest_threshold,
            graph_safe=graph_safe,
        )
        return out
    packed = PackedExactWeight.pack(weight, mode=mode, crest_threshold=crest_threshold)
    return _gemm_from_packed_torch(
        x,
        packed,
        bias,
        mode=mode,
        crest_threshold=crest_threshold,
        graph_safe=graph_safe,
        raw_weight_for_fallback=weight if not graph_safe else None,
    )


def fp16x3_linear(
    x: Any,
    weight: Any,
    bias: Any | None = None,
    *,
    graph_safe: bool = True,
) -> Any:
    """Functional exact-FP32 3-GEMM FP16x3 Ozaki linear projection."""
    return scaled_fp16_linear(
        x,
        weight,
        bias,
        mode="fp16x3",
        graph_safe=graph_safe,
    )


_BaseModule = nn.Module if nn is not None else object


class CrestGuardedLinear(_BaseModule):  # type: ignore[misc]
    """Drop-in ``nn.Linear`` replacement with dual weight + online activation crest guard.

    Execution tiers on CUDA (cuBLAS WGMMA):
    - ``mode='scaled-fp16'``: 1-GEMM Scaled-FP16 (maximum throughput, no W1 VRAM).
    - ``mode='auto'``: 3-GEMM ``fp16x3`` when ``crest_w > crest_threshold``; for well-conditioned
      weights, runs 2-GEMM activation-outlier-safe (zero W1 VRAM, zero host sync) when
      ``graph_safe=True`` or dynamically skips the 2nd GEMM when ``graph_safe=False``.
    - ``mode='fp16x3'``: unconditional 3-GEMM Ozaki exact-FP32 parity.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        mode: str = "auto",
        crest_threshold: float = DEFAULT_CREST_THRESHOLD,
        graph_safe: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        if torch is None:
            raise RuntimeError("CrestGuardedLinear requires PyTorch.")
        super().__init__()
        if mode not in ("auto", "scaled-fp16", "fp16x3", "tf32", "fp32"):
            raise ValueError(f"Unsupported CrestGuardedLinear mode: {mode!r}")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.mode = mode
        self.crest_threshold = float(crest_threshold)
        self.graph_safe = bool(graph_safe)

        factory_kwargs = {"device": device, "dtype": dtype or torch.float32}
        self.weight = nn.Parameter(
            torch.empty((self.out_features, self.in_features), **factory_kwargs),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.out_features, **factory_kwargs),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.in_features) if self.in_features > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

        self._packed: PackedExactWeight | None = None
        self.register_buffer("_w0", torch.empty(0, dtype=torch.float16), persistent=False)
        self.register_buffer("_w1", torch.empty(0, dtype=torch.float16), persistent=False)
        self.register_buffer("_exp_w", torch.empty(0, dtype=torch.int32), persistent=False)
        self.register_buffer("_bad_w", torch.empty(0, dtype=torch.int32), persistent=False)
        self.register_buffer("_crest_w", torch.empty(0, dtype=torch.float32), persistent=False)
        self.escalated_w = False
        self.max_crest_w = 0.0
        self._has_nonfinite_w = False

    def repack(self) -> None:
        """Pack weights offline and evaluate weight crest factor ``max(|W|) / rms_f64(W)``."""
        if self.mode in ("tf32", "fp32"):
            self._packed = None
            return
        packed = PackedExactWeight.pack(
            self.weight.data, mode=self.mode, crest_threshold=self.crest_threshold
        )
        self._w0 = packed.w0
        self._w1 = (
            packed.w1
            if packed.w1 is not None
            else torch.empty(0, dtype=torch.float16, device=self.weight.device)
        )
        self._exp_w = packed.exp_w
        self._bad_w = packed.bad_w
        self._crest_w = packed.crest_w
        self.escalated_w = packed.escalated_w
        self.max_crest_w = packed.max_crest_w
        self._has_nonfinite_w = packed.has_nonfinite_w
        self._packed = packed

    def _load_from_state_dict(
        self,
        state_dict: Any,
        prefix: str,
        local_metadata: Any,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Reload parameters via ``state_dict`` and refresh non-persistent packed FP16 buffers."""
        prev_err_count = len(error_msgs)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self._packed = None
        if (
            len(error_msgs) == prev_err_count
            and self.mode not in ("tf32", "fp32")
            and self.weight.device.type != "meta"
        ):
            self.repack()

    @classmethod
    def from_linear(
        cls,
        linear: Any,
        *,
        mode: str = "auto",
        crest_threshold: float = DEFAULT_CREST_THRESHOLD,
        graph_safe: bool = True,
    ) -> CrestGuardedLinear:
        affine = _extract_effective_affine_params(linear)
        if affine is not None:
            w_eff, b_eff = affine
        else:
            w_eff = linear.weight.detach().to(torch.float32)
            b_eff = None if linear.bias is None else linear.bias.detach().to(torch.float32)
        in_f = int(getattr(linear, "in_features", linear.weight.shape[1]))
        out_f = int(getattr(linear, "out_features", linear.weight.shape[0]))
        layer = cls(
            in_f,
            out_f,
            bias=(b_eff is not None),
            mode=mode,
            crest_threshold=crest_threshold,
            graph_safe=graph_safe,
            device=linear.weight.device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            layer.weight.copy_(w_eff)
            if b_eff is not None and layer.bias is not None:
                layer.bias.copy_(b_eff)
        layer.repack()
        return layer

    def _ensure_packed(self) -> PackedExactWeight:
        needs_repack = (
            self._packed is None
            or self._w0.device != self.weight.device
            or self._w0.dtype != torch.float16
            or ((self.mode == "fp16x3" or self.escalated_w) and self._w1.numel() == 0)
        )
        if needs_repack:
            self.repack()
        assert self._packed is not None
        return PackedExactWeight(
            w0=self._w0,
            w1=self._w1 if (self.escalated_w and self._w1.numel() > 0) else None,
            exp_w=self._exp_w,
            bad_w=self._bad_w,
            crest_w=self._crest_w,
            max_crest_w=self.max_crest_w,
            escalated_w=self.escalated_w,
            has_nonfinite_w=self._has_nonfinite_w,
            out_features=self.out_features,
            in_features=self.in_features,
        )

    def forward(self, x: Any) -> Any:
        if self.mode == "fp32":
            prev_tf32 = (
                torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False
            )
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = False
            try:
                return F.linear(x.to(torch.float32), self.weight, self.bias)
            finally:
                if torch.cuda.is_available():
                    torch.backends.cuda.matmul.allow_tf32 = prev_tf32
        if self.mode == "tf32":
            prev_tf32 = (
                torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False
            )
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
            try:
                return F.linear(x.to(torch.float32), self.weight, self.bias)
            finally:
                if torch.cuda.is_available():
                    torch.backends.cuda.matmul.allow_tf32 = prev_tf32

        packed = self._ensure_packed()
        out = _gemm_from_packed_torch(
            x,
            packed,
            self.bias,
            mode=self.mode,
            crest_threshold=self.crest_threshold,
            graph_safe=self.graph_safe,
            raw_weight_for_fallback=self.weight if not self.graph_safe else None,
        )
        return out.to(x.dtype) if x.dtype != torch.float32 else out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, mode={self.mode!r}, "
            f"escalated_w={self.escalated_w}, max_crest_w={self.max_crest_w:.2f}, "
            f"graph_safe={self.graph_safe}"
        )


def is_linear_module(mod: Any) -> bool:
    """Return True for ``nn.Linear`` and linear-like ``nn.Module`` drop-ins.

    Supports both ``nn.Linear`` subclasses and standalone ``nn.Module`` linear
    wrappers (such as ``unit_scaling.Linear`` / ``dd_unit_scaling.Linear`` that
    inherit directly from ``nn.Module`` while exposing a 2D ``weight`` tensor).
    """
    if torch is None or nn is None or mod is None:
        return False
    if isinstance(mod, CrestGuardedLinear):
        return False
    if isinstance(mod, nn.Linear):
        return True
    if not isinstance(mod, nn.Module):
        return False
    w = getattr(mod, "weight", None)
    if not (isinstance(w, torch.Tensor) and w.ndim == 2):
        return False
    return "linear" in type(mod).__name__.lower()


def _compute_max_row_crest_f64_torch(weight: Any) -> float:
    """Compute exact per-row ``max(|W_i|) / rms_f64(W_i)`` without launching Triton."""
    if weight.ndim != 2 or weight.shape[0] == 0 or weight.shape[1] == 0:
        return 0.0
    with torch.no_grad():
        w64 = weight.detach().to(torch.float64)
        is_fin = torch.isfinite(w64)
        has_nonfin = ~is_fin.all(dim=1)
        w_clean = torch.where(is_fin, w64, torch.zeros_like(w64))
        amax = w_clean.abs().amax(dim=1)
        rms = torch.sqrt((w_clean * w_clean).mean(dim=1))
        valid = (~has_nonfin) & (rms > 0.0)
        cw = torch.where(valid, amax / rms.clamp(min=1e-300), torch.zeros_like(amax))
        return float(cw.max().item()) if cw.numel() > 0 else 0.0


def _extract_effective_affine_params(linear: Any) -> tuple[Any, Any | None] | None:
    """Return exact ``(weight_f32, bias_f32)`` for ``nn.Linear`` and affine drop-ins.

    Preserves forward scaling factors of ``nn.Linear`` subclasses and ``nn.Module``
    linear drop-ins (such as ``unit_scaling.Linear`` / ``dd_unit_scaling.Linear``)
    by probing and verifying exact affine equivalence on two independent inputs;
    returns ``None`` if a module has non-affine forward behavior so replacement
    safely skips it.
    """
    w_attr = getattr(linear, "weight", None)
    if not (isinstance(w_attr, torch.Tensor) and w_attr.ndim == 2):
        return None
    b_attr = getattr(linear, "bias", None)
    w_f32 = w_attr.detach().to(torch.float32)
    b_f32 = None if b_attr is None else b_attr.detach().to(torch.float32)
    if type(linear) is nn.Linear:
        return w_f32, b_f32

    try:
        dev = w_attr.device
        in_f = int(getattr(linear, "in_features", w_attr.shape[1]))
        w_dtype = w_attr.dtype
        gen = torch.Generator(device="cpu").manual_seed(2026)
        probes = torch.randn((2, in_f), generator=gen, dtype=torch.float32).to(
            device=dev, dtype=w_dtype
        )
        zeros = torch.zeros((1, in_f), device=dev, dtype=w_dtype)
        with torch.no_grad():
            y0 = linear(zeros)
            if isinstance(y0, tuple):
                return None
            y0_f32 = y0.to(torch.float32).squeeze(0)
            y_act = linear(probes)
            if isinstance(y_act, tuple):
                return None
            y_act_f32 = y_act.to(torch.float32) - y0_f32.unsqueeze(0)
            y_raw = F.linear(probes.to(torch.float32), w_f32, None)
            denom = float(y_raw.pow(2).sum().item())
            if denom <= 1e-30:
                if float(y_act_f32.abs().max().item()) <= 1e-6:
                    scale = 1.0
                else:
                    return None
            else:
                scale = float((y_act_f32 * y_raw).sum().item()) / denom
            if not math.isfinite(scale):
                return None
            w_eff = w_f32 * scale
            y_check = F.linear(probes.to(torch.float32), w_eff, None)
            max_err = float((y_check - y_act_f32).abs().max().item())
            ref_mag = max(float(y_act_f32.abs().max().item()), 1e-6)
            if max_err / ref_mag > 1e-4:
                return None
            has_eff_bias = (b_attr is not None) or (float(y0_f32.abs().max().item()) > 0.0)
            return w_eff, (y0_f32 if has_eff_bias else None)
    except Exception:
        return None


def _assign_child_module(parent: Any, name: str, child: Any) -> None:
    """Assign a child module safely on both ``nn.Module`` and ``nn.ModuleList``."""
    if isinstance(parent, nn.ModuleList) and name.isdigit():
        parent[int(name)] = child
    else:
        setattr(parent, name, child)


def replace_linears_with_crest_guarded(
    module: Any,
    *,
    mode: str = "auto",
    crest_threshold: float = DEFAULT_CREST_THRESHOLD,
    graph_safe: bool = True,
    skip_names: Iterable[str] = (),
    min_dim: int = 32,
    predicate: Callable[[str, Any], bool] | None = None,
) -> dict[str, Any]:
    """Recursively audit/replace eligible ``nn.Linear`` layers with ``CrestGuardedLinear``.

    - In ``mode='auto'``: audits every eligible weight matrix's crest factor
      ``crest_w = max(|W|) / rms_f64(W)`` at compile time. High-crest outlier
      layers (``crest_w > crest_threshold``) are replaced with
      ``CrestGuardedLinear(mode='fp16x3')`` for cuBLAS WGMMA exact-FP32
      protection, while well-conditioned layers (``crest_w <= crest_threshold``)
      remain native ``nn.Linear`` so PyTorch Inductor can fuse elementwise ops
      and TensorCore TF32 GEMMs.
    - In ``mode='fp16x3'`` or ``mode='scaled-fp16'``: explicitly replaces all
      eligible trunk ``nn.Linear`` layers with ``CrestGuardedLinear``.
    - Always preserves small embedding/norm/head layers matching ``skip_names``
      or dimensions below ``min_dim`` in exact FP32 SIMT.
    """
    if torch is None:
        raise RuntimeError("replace_linears_with_crest_guarded requires PyTorch.")

    skip_tokens = tuple(s.lower() for s in skip_names)
    audited = 0
    replaced = 0
    escalated_w = 0
    kept_native_tf32 = 0
    skipped = 0
    overall_max_crest_w = 0.0

    def _walk(parent: Any, prefix: str) -> None:
        nonlocal audited, replaced, escalated_w, kept_native_tf32, skipped, overall_max_crest_w
        for name, child in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if is_linear_module(child):
                low = full_name.lower()
                in_f = int(getattr(child, "in_features", child.weight.shape[1]))
                out_f = int(getattr(child, "out_features", child.weight.shape[0]))
                is_skipped_name = any(tok in low for tok in skip_tokens)
                is_small = in_f < min_dim or out_f < min_dim
                allowed = (not is_skipped_name) and (not is_small)
                if predicate is not None:
                    allowed = allowed and bool(predicate(full_name, child))
                if allowed:
                    affine = _extract_effective_affine_params(child)
                    if affine is None:
                        skipped += 1
                        continue
                    w_eff, b_eff = affine
                    audited += 1
                    if mode == "auto":
                        max_cw = _compute_max_row_crest_f64_torch(w_eff)
                        if max_cw > overall_max_crest_w:
                            overall_max_crest_w = max_cw
                        if max_cw <= crest_threshold:
                            kept_native_tf32 += 1
                            continue
                        target_mode = "fp16x3"
                    else:
                        target_mode = mode

                    new_layer = CrestGuardedLinear(
                        in_f,
                        out_f,
                        bias=(b_eff is not None),
                        mode=target_mode,
                        crest_threshold=crest_threshold,
                        graph_safe=graph_safe,
                        device=child.weight.device,
                        dtype=torch.float32,
                    )
                    with torch.no_grad():
                        new_layer.weight.copy_(w_eff)
                        if b_eff is not None and new_layer.bias is not None:
                            new_layer.bias.copy_(b_eff)
                    new_layer.repack()
                    if new_layer.max_crest_w > overall_max_crest_w:
                        overall_max_crest_w = new_layer.max_crest_w
                    _assign_child_module(parent, name, new_layer)
                    replaced += 1
                    if new_layer.escalated_w:
                        escalated_w += 1
                else:
                    skipped += 1
            else:
                _walk(child, full_name)

    _walk(module, "")
    return {
        "mode": mode,
        "audited": audited,
        "replaced": replaced,
        "escalated_w": escalated_w,
        "kept_native_tf32": kept_native_tf32,
        "max_crest_w": overall_max_crest_w,
        "skipped": skipped,
    }
