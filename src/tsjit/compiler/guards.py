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

"""Numerical safety guards, DC offset stripping, strict-FP32 parity audit, and IEEE prep kernels."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional in pure-CPU environments
    torch = None  # type: ignore[assignment]

try:
    import triton
    import triton.language as tl

    try:
        from triton.language.extra.cuda import libdevice as _triton_libdevice
    except ImportError:  # pragma: no cover - alternate Triton layout
        try:
            from triton.language.extra import libdevice as _triton_libdevice
        except ImportError:
            _triton_libdevice = None

    _TRITON_IEEE_AVAILABLE = _triton_libdevice is not None
except ImportError:  # pragma: no cover - CPU-only environments
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _triton_libdevice = None
    _TRITON_IEEE_AVAILABLE = False

DEFAULT_RATIONAL_PERIODS: tuple[tuple[int, int], ...] = ((5, 3), (7, 3), (49, 25))

_DIRECT_PRECISION_ALIASES: dict[str, str] = {
    "a10-auto": "auto",
    "auto": "auto",
    "crest-guarded": "auto",
    "scaled-fp16": "scaled-fp16",
    "scaled_fp16": "scaled-fp16",
    "fp16x3": "fp16x3",
    "fp16x3-ozaki": "fp16x3",
    "ozaki": "fp16x3",
    "strict": "fp16x3",
    "strict-fp32-parity": "fp16x3",
    "fast-tf32": "tf32",
    "fast_tf32": "tf32",
    "tf32": "tf32",
    "fp32": "fp32",
}


def normalize_precision_alias(precision: str) -> str:
    """Resolve public precision mode aliases to canonical internal modes."""
    key = precision.strip().lower()
    if key not in _DIRECT_PRECISION_ALIASES:
        raise ValueError(
            f"Unsupported precision {precision!r}. "
            f"Choose from {sorted(_DIRECT_PRECISION_ALIASES)}."
        )
    return _DIRECT_PRECISION_ALIASES[key]


@contextmanager
def tf32_precision_context(precision: str) -> Iterator[None]:
    """Explicitly enforce CUDA TF32 state across compiler precision modes.

    - ``'auto'``, ``'tf32'``, and ``'fast-tf32'`` enable TensorCore TF32 for native
      ``nn.Linear`` layers.
    - ``'fp32'``, ``'fp16x3'``, ``'strict'``, and ``'scaled-fp16'`` disable TF32 so
      non-replaced embeddings/norms/heads retain strict IEEE FP32 parity.
    """
    if (
        torch is None
        or not torch.cuda.is_available()
        or precision
        not in ("fp32", "tf32", "fast-tf32", "auto", "a10-auto", "fp16x3", "strict", "scaled-fp16")
    ):
        yield
        return
    prev_matmul = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    target = precision in ("tf32", "fast-tf32", "auto", "a10-auto")
    if prev_matmul == target and prev_cudnn == target:
        yield
        return
    torch.backends.cuda.matmul.allow_tf32 = target
    torch.backends.cudnn.allow_tf32 = target
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul
        torch.backends.cudnn.allow_tf32 = prev_cudnn


class CausalFloat64DCStripper:
    """Strictly causal per-series FP64 DC anchor stripper and restorer.

    Computes each series' causal anchor as the last finite observation in FP64
    along the time axis (``dim=-1``), subtracts it in FP64 before casting to
    FP32 for neural inference, and restores ``q.astype(float64) + dc_f64`` on
    exit. This completely eliminates FP32 ULP collapse on large DC offsets
    (e.g., ``2.22e10``, where FP32 ULP is 2048.0).
    """

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = bool(enabled)

    @staticmethod
    def _last_finite_numpy(x_f64: np.ndarray) -> np.ndarray:
        T = x_f64.shape[-1]
        if T == 0:
            return np.zeros((*x_f64.shape[:-1], 1), dtype=np.float64)
        is_fin = np.isfinite(x_f64)
        idx = np.arange(T, dtype=np.int64)
        masked_idx = np.where(is_fin, idx, -1)
        last_idx = np.max(masked_idx, axis=-1, keepdims=True)
        has_any = last_idx >= 0
        safe_idx = np.maximum(last_idx, 0)
        gathered = np.take_along_axis(x_f64, safe_idx, axis=-1)
        return np.where(has_any, gathered, 0.0).astype(np.float64)

    @staticmethod
    def _last_finite_torch(x_f64: Any) -> Any:
        T = x_f64.shape[-1]
        if T == 0:
            return torch.zeros((*x_f64.shape[:-1], 1), dtype=torch.float64, device=x_f64.device)
        is_fin = torch.isfinite(x_f64)
        idx = torch.arange(T, dtype=torch.int64, device=x_f64.device)
        masked_idx = torch.where(is_fin, idx, torch.full_like(idx, -1))
        last_idx = masked_idx.amax(dim=-1, keepdim=True)
        has_any = last_idx >= 0
        safe_idx = last_idx.clamp(min=0)
        gathered = torch.gather(x_f64, dim=-1, index=safe_idx)
        return torch.where(has_any, gathered, torch.zeros_like(gathered))

    def strip(self, x: Any) -> tuple[Any, Any]:
        """Return ``(x_stripped_f32, dc_f64)`` using a strictly causal FP64 anchor."""
        if torch is not None and isinstance(x, torch.Tensor):
            x_f64 = x.to(torch.float64)
            if not self.enabled:
                dc = torch.zeros((*x_f64.shape[:-1], 1), dtype=torch.float64, device=x_f64.device)
                return x_f64.to(torch.float32), dc
            dc = self._last_finite_torch(x_f64)
            return (x_f64 - dc).to(torch.float32), dc

        x_f64 = np.asarray(x, dtype=np.float64)
        if not self.enabled:
            dc = np.zeros((*x_f64.shape[:-1], 1), dtype=np.float64)
            return x_f64.astype(np.float32), dc
        dc = self._last_finite_numpy(x_f64)
        return (x_f64 - dc).astype(np.float32), dc

    def restore(self, q: Any, dc_f64: Any, *, batch_dim: int = 0) -> Any:
        """Restore FP64 DC anchor ``q.astype(float64) + dc_f64`` with shape-safe broadcast."""
        if not self.enabled:
            if torch is not None and isinstance(q, torch.Tensor):
                return q.to(torch.float64)
            return np.asarray(q, dtype=np.float64)

        if torch is not None and isinstance(q, torch.Tensor):
            dc = dc_f64.to(device=q.device, dtype=torch.float64)
            while dc.ndim > q.ndim and dc.shape[-1] == 1:
                dc = dc.squeeze(-1)
            if batch_dim != 0 and dc.ndim >= 1 and q.ndim >= 2:
                b_axis = batch_dim if batch_dim >= 0 else (q.ndim + batch_dim)
                b_size = dc.shape[0]
                target_shape = [1] * q.ndim
                target_shape[b_axis] = b_size
                dc = dc.reshape(target_shape)
            else:
                while dc.ndim < q.ndim:
                    dc = dc.unsqueeze(-2)
            return q.to(torch.float64) + dc

        q_f64 = np.asarray(q, dtype=np.float64)
        dc = np.asarray(dc_f64, dtype=np.float64)
        while dc.ndim > q_f64.ndim and dc.shape[-1] == 1:
            dc = np.squeeze(dc, axis=-1)
        if batch_dim != 0 and dc.ndim >= 1 and q_f64.ndim >= 2:
            b_axis = batch_dim if batch_dim >= 0 else (q_f64.ndim + batch_dim)
            b_size = dc.shape[0]
            target_shape = [1] * q_f64.ndim
            target_shape[b_axis] = b_size
            dc = dc.reshape(target_shape)
        else:
            while dc.ndim < q_f64.ndim:
                dc = np.expand_dims(dc, axis=-2)
        return q_f64 + dc

    def wrap_callable(self, fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
        """Wrap ``fn(x_f32) -> y`` with causal FP64 DC stripping and restoration."""

        def _wrapped(x: Any) -> Any:
            x_s, dc = self.strip(x)
            y = fn(x_s)
            return self.restore(y, dc)

        return _wrapped


def scale_free_norm_floor(
    x: Any,
    loc: Any,
    scale: Any,
    *,
    inc: Any | None = None,
) -> Any:
    """Compute a strictly scale-free normalization denominator floor.

    When a positive finite dispersion ``primary = max(scale, |inc|)`` is available,
    returns ``primary`` directly without inflating it by ``1e-4 * |loc|`` or
    ``1e-7 * max(|x|)`` (preventing DC-offset series such as ``N(0, 1) + 2.22e10``
    from masking huge forecast errors under a giant location-proportional denominator).
    Applies the constant-series fallback ``max(1e-4 * |loc|, 1e-7 * max_finite(|x|))``
    ONLY when ``primary == 0.0`` or non-finite, and falls back to ``1.0`` ONLY when
    that fallback is also non-positive or non-finite.
    """
    if torch is not None and isinstance(x, torch.Tensor):
        x_f = x.to(torch.float64)
        loc_f = torch.as_tensor(loc, device=x.device, dtype=torch.float64)
        sc_f = torch.as_tensor(scale, device=x.device, dtype=torch.float64)
        is_fin = torch.isfinite(x_f)
        abs_x = torch.where(is_fin, x_f.abs(), torch.zeros_like(x_f))
        max_abs = (
            abs_x.amax(dim=-1, keepdim=True)
            if x_f.shape[-1] > 0
            else torch.zeros_like(loc_f)
        )
        tol = 1e-7 * max_abs
        while tol.ndim > sc_f.ndim and tol.shape[-1] == 1:
            tol = tol.squeeze(-1)
        if tol.ndim > sc_f.ndim:
            if tol.numel() == 0:
                tol = torch.zeros_like(sc_f)
            elif sc_f.ndim == 0:
                tol = tol.amax()
            else:
                reduce_dims = tuple(range(sc_f.ndim, tol.ndim))
                tol = tol.amax(dim=reduce_dims)
        loc_floor = 1e-4 * torch.where(torch.isfinite(loc_f), loc_f.abs(), torch.zeros_like(loc_f))
        sc_clean = torch.where(
            (sc_f > 0.0) & torch.isfinite(sc_f), sc_f, torch.zeros_like(sc_f)
        )

        primary = sc_clean
        if inc is not None:
            inc_f = torch.as_tensor(inc, device=x.device, dtype=torch.float64)
            inc_clean = torch.where(
                torch.isfinite(inc_f), inc_f.abs(), torch.zeros_like(inc_f)
            )
            primary = torch.maximum(primary, inc_clean)

        has_primary = (primary > 0.0) & torch.isfinite(primary)
        const_floor = torch.maximum(loc_floor, tol)
        chosen = torch.where(has_primary, primary, const_floor)
        valid = (chosen > 0.0) & torch.isfinite(chosen)
        return torch.where(valid, chosen, torch.ones_like(chosen)).to(
            scale.dtype if isinstance(scale, torch.Tensor) else torch.float64
        )

    x_f = np.asarray(x, dtype=np.float64)
    is_fin = np.isfinite(x_f)
    abs_x = np.where(is_fin, np.abs(x_f), 0.0)
    max_abs = (
        np.max(abs_x, axis=-1, keepdims=True)
        if x_f.shape[-1] > 0
        else np.zeros_like(loc, dtype=np.float64)
    )
    tol = 1e-7 * max_abs

    loc_f = np.asarray(loc, dtype=np.float64)
    sc_f = np.asarray(scale, dtype=np.float64)
    while tol.ndim > sc_f.ndim and tol.shape[-1] == 1:
        tol = np.squeeze(tol, axis=-1)
    if tol.ndim > sc_f.ndim:
        if tol.size == 0:
            tol = np.zeros_like(sc_f, dtype=np.float64)
        elif sc_f.ndim == 0:
            tol = np.max(tol)
        else:
            reduce_axes = tuple(range(sc_f.ndim, tol.ndim))
            tol = np.max(tol, axis=reduce_axes)
    loc_floor = 1e-4 * np.where(np.isfinite(loc_f), np.abs(loc_f), 0.0)
    sc_clean = np.where((sc_f > 0.0) & np.isfinite(sc_f), sc_f, 0.0)

    primary = sc_clean
    if inc is not None:
        inc_f = np.asarray(inc, dtype=np.float64)
        inc_clean = np.where(np.isfinite(inc_f), np.abs(inc_f), 0.0)
        primary = np.maximum(primary, inc_clean)

    has_primary = (primary > 0.0) & np.isfinite(primary)
    const_floor = np.maximum(loc_floor, tol)
    chosen = np.where(has_primary, primary, const_floor)
    valid = (chosen > 0.0) & np.isfinite(chosen)
    return np.where(valid, chosen, 1.0)


@dataclass(frozen=True)
class ParityAuditReport:
    """Result of ``StrictParityValidator`` against a serial batch-1 FP32 reference."""

    passed: bool
    num_series: int
    num_failed: int
    max_scaled_err: float
    p99_scaled_err: float
    nonfinite_mismatches: int
    failed_indices: tuple[int, ...]
    tol: float


class StrictParityValidator:
    """Symmetric strict-FP32 parity auditor against serial batch-1 FP32 reference.

    Disables TF32 (``allow_tf32 = False``) while computing the serial batch-1
    reference forecasts and verifies ``max_scaled_err <= tol`` (default 0.05)
    with zero non-finite mismatches.
    """

    def __init__(self, tol: float = 0.05) -> None:
        self.tol = float(tol)

    @staticmethod
    def _to_numpy(arr: Any) -> np.ndarray:
        if torch is not None and isinstance(arr, torch.Tensor):
            return arr.detach().to(torch.float64).cpu().numpy()
        return np.asarray(arr, dtype=np.float64)

    def _run_serial_fp32_ref(
        self,
        ref_fn: Callable[[Any], Any],
        x_batch: Any,
    ) -> np.ndarray:
        if torch is not None and isinstance(x_batch, torch.Tensor):
            if x_batch.shape[0] == 0:
                with torch.inference_mode():
                    return self._to_numpy(ref_fn(x_batch))
            prev_matmul_tf32 = (
                torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False
            )
            prev_cudnn_tf32 = (
                torch.backends.cudnn.allow_tf32 if torch.cuda.is_available() else False
            )
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
            try:
                outs = []
                with torch.inference_mode():
                    for i in range(x_batch.shape[0]):
                        yi = ref_fn(x_batch[i : i + 1])
                        outs.append(self._to_numpy(yi))
                return np.concatenate(outs, axis=0)
            finally:
                if torch.cuda.is_available():
                    torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
                    torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32

        x_np = np.asarray(x_batch)
        if x_np.shape[0] == 0:
            return self._to_numpy(ref_fn(x_np))
        outs = [self._to_numpy(ref_fn(x_np[i : i + 1])) for i in range(x_np.shape[0])]
        return np.concatenate(outs, axis=0)

    def audit(
        self,
        ref_fn: Callable[[Any], Any],
        compiled_fn: Callable[[Any], Any],
        x_batch: Any,
        *,
        precomputed_ref: Any | None = None,
    ) -> ParityAuditReport:
        """Audit ``compiled_fn(x_batch)`` against serial batch-1 strict-FP32 ``ref_fn``."""
        y_ref = (
            self._to_numpy(precomputed_ref)
            if precomputed_ref is not None
            else self._run_serial_fp32_ref(ref_fn, x_batch)
        )
        if torch is not None and isinstance(x_batch, torch.Tensor):
            with torch.inference_mode():
                y_comp = self._to_numpy(compiled_fn(x_batch))
        else:
            y_comp = self._to_numpy(compiled_fn(x_batch))

        if y_ref.shape != y_comp.shape:
            raise ValueError(f"Parity shape mismatch: ref={y_ref.shape} vs compiled={y_comp.shape}")

        x_np = self._to_numpy(x_batch)
        B = y_ref.shape[0]
        scaled_errs = np.zeros(B, dtype=np.float64)
        nonfinite_mismatches = 0
        failed: list[int] = []

        for i in range(B):
            xi = x_np[i].reshape(-1)
            fin_x = xi[np.isfinite(xi)]
            if fin_x.size > 0:
                med_x = float(np.median(fin_x))
                loc = med_x
                # Compute dispersion in FP64 around the causal median anchor so large DC offsets
                # (e.g., N(0, 1) + 2.22e10) retain exact std ~ 1.0 without cancellation.
                std = float(np.std(fin_x - med_x)) if fin_x.size > 1 else 0.0
                if not np.isfinite(std):
                    std = 0.0
            else:
                loc = 0.0
                std = 0.0
            yr = y_ref[i].reshape(-1)
            yc = y_comp[i].reshape(-1)
            fin_r = np.isfinite(yr)
            fin_c = np.isfinite(yc)
            if not np.array_equal(fin_r, fin_c):
                nonfinite_mismatches += 1
                scaled_errs[i] = float("inf")
                failed.append(i)
                continue

            if not np.any(fin_r):
                scaled_errs[i] = 0.0
                continue

            # Symmetric scale combining shifted-anchor context dispersion and reference spread.
            yr_fin = yr[fin_r]
            if yr_fin.size > 1:
                med_r = float(np.median(yr_fin))
                ref_spread = float(np.std(yr_fin - med_r))
                if not np.isfinite(ref_spread):
                    ref_spread = 0.0
            else:
                ref_spread = 0.0
            raw_sc = max(std, ref_spread)
            if raw_sc > 0.0 and np.isfinite(raw_sc):
                denom = raw_sc
            else:
                denom = float(scale_free_norm_floor(xi, loc, 0.0))
            err = float(np.max(np.abs(yc[fin_r] - yr[fin_r]))) / denom
            scaled_errs[i] = err
            if not np.isfinite(err) or err > self.tol:
                failed.append(i)

        max_err = float(np.max(scaled_errs)) if B > 0 else 0.0
        p99_err = float(np.percentile(scaled_errs, 99)) if B > 0 else 0.0
        return ParityAuditReport(
            passed=(len(failed) == 0 and nonfinite_mismatches == 0),
            num_series=B,
            num_failed=len(failed),
            max_scaled_err=max_err,
            p99_scaled_err=p99_err,
            nonfinite_mismatches=nonfinite_mismatches,
            failed_indices=tuple(failed),
            tol=self.tol,
        )


# ---------------------------------------------------------------------------
# IEEE-Safe Rational-Lag Period Extraction (ieee-div-guard: div_rn + no FP fusion)
# ---------------------------------------------------------------------------

if _TRITON_IEEE_AVAILABLE:

    @triton.jit
    def _rational_lag_prep_triton_kernel(
        X_ptr,
        NUM_ptr,
        DEN_ptr,
        OUT_LAG_ptr,
        OUT_RATIO_ptr,
        B,
        L,
        P,
        X_STRIDE,
        OUT_STRIDE,
    ):
        b = tl.program_id(0)
        p_idx = tl.program_id(1)
        if b >= B or p_idx >= P:
            return

        num_f = tl.load(NUM_ptr + p_idx).to(tl.float32)
        den_f = tl.load(DEN_ptr + p_idx).to(tl.float32)
        # Strict IEEE round-to-nearest division (never fast approximate reciprocal).
        period = _triton_libdevice.div_rn(num_f, den_f)

        last_pos = (L - 1).to(tl.float32)
        src_pos = last_pos - period
        lo_i = tl.maximum(tl.minimum(tl.floor(src_pos).to(tl.int32), L - 1), 0)
        hi_i = tl.maximum(tl.minimum(lo_i + 1, L - 1), 0)
        frac = tl.maximum(tl.minimum(src_pos - lo_i.to(tl.float32), 1.0), 0.0)

        b_i64 = b.to(tl.int64)
        row_ptr = X_ptr + b_i64 * X_STRIDE
        x_last = tl.load(row_ptr + (L - 1)).to(tl.float32)
        x_lo = tl.load(row_ptr + lo_i).to(tl.float32)
        x_hi = tl.load(row_ptr + hi_i).to(tl.float32)
        x_lag = x_lo + frac * (x_hi - x_lo)

        denom = tl.maximum(tl.abs(x_last) + tl.abs(x_lag), 1e-7 * tl.maximum(tl.abs(x_last), 1.0))
        ratio = _triton_libdevice.div_rn(x_last - x_lag, denom)

        out_off = b_i64 * OUT_STRIDE + p_idx
        tl.store(OUT_LAG_ptr + out_off, x_lag)
        tl.store(OUT_RATIO_ptr + out_off, ratio)


def extract_rational_lag_features(
    x: Any,
    periods: Sequence[tuple[int, int]] = DEFAULT_RATIONAL_PERIODS,
    *,
    use_triton: bool = True,
) -> tuple[Any, Any]:
    """Extract exact IEEE rational-lag values and normalized lag ratios (e.g., 5/3, 7/3, 49/25).

    On CUDA with Triton, launches with ``libdevice.div_rn`` and
    ``enable_fp_fusion=False`` passed as a launch keyword argument to enforce
    strict IEEE-754 rounding parity; falls back to exact CPU FP64 arithmetic.
    """
    if len(periods) == 0:
        raise ValueError("periods must contain at least one (num, den) pair")
    for num, den in periods:
        if den <= 0 or num < 0:
            raise ValueError(f"Invalid rational period ({num}, {den})")

    if torch is not None and isinstance(x, torch.Tensor):
        if x.ndim != 2:
            raise ValueError(f"Expected 2D (B, L) tensor, got {tuple(x.shape)}")
        B, L = x.shape
        P = len(periods)
        if L == 0:
            raise ValueError("Input sequence length L must be >= 1")
        if x.is_cuda and use_triton and _TRITON_IEEE_AVAILABLE and B > 0:
            with torch.cuda.device(x.device):
                x_f32 = x.contiguous().to(torch.float32)
                nums = torch.tensor([p[0] for p in periods], dtype=torch.int32, device=x.device)
                dens = torch.tensor([p[1] for p in periods], dtype=torch.int32, device=x.device)
                out_lag = torch.empty((B, P), dtype=torch.float32, device=x.device)
                out_ratio = torch.empty((B, P), dtype=torch.float32, device=x.device)
                # Launch kwarg enable_fp_fusion=False enforces ieee-div-guard!
                _rational_lag_prep_triton_kernel[(B, P)](
                    x_f32,
                    nums,
                    dens,
                    out_lag,
                    out_ratio,
                    B,
                    L,
                    P,
                    x_f32.stride(0),
                    out_lag.stride(0),
                    enable_fp_fusion=False,
                )
            return out_lag, out_ratio

        # Exact FP64 CPU/PyTorch fallback
        x_f64 = x.to(torch.float64)
        out_lag = torch.empty((B, P), dtype=torch.float64, device=x.device)
        out_ratio = torch.empty((B, P), dtype=torch.float64, device=x.device)
        x_last = x_f64[:, L - 1]
        for p_idx, (num, den) in enumerate(periods):
            period = float(num) / float(den)
            src_pos = float(L - 1) - period
            lo_i = max(0, min(int(np.floor(src_pos)), L - 1))
            hi_i = max(0, min(lo_i + 1, L - 1))
            frac = max(0.0, min(src_pos - float(lo_i), 1.0))
            x_lag = x_f64[:, lo_i] + frac * (x_f64[:, hi_i] - x_f64[:, lo_i])
            denom = torch.maximum(
                x_last.abs() + x_lag.abs(),
                1e-7 * torch.maximum(x_last.abs(), torch.ones_like(x_last)),
            )
            out_lag[:, p_idx] = x_lag
            out_ratio[:, p_idx] = (x_last - x_lag) / denom
        return out_lag.to(torch.float32), out_ratio.to(torch.float32)

    # Pure-NumPy exact FP64 fallback
    x_f64 = np.asarray(x, dtype=np.float64)
    if x_f64.ndim != 2:
        raise ValueError(f"Expected 2D (B, L) array, got {x_f64.shape}")
    B, L = x_f64.shape
    if L == 0:
        raise ValueError("Input sequence length L must be >= 1")
    P = len(periods)
    out_lag = np.empty((B, P), dtype=np.float64)
    out_ratio = np.empty((B, P), dtype=np.float64)
    x_last = x_f64[:, L - 1]
    for p_idx, (num, den) in enumerate(periods):
        period = float(num) / float(den)
        src_pos = float(L - 1) - period
        lo_i = max(0, min(int(np.floor(src_pos)), L - 1))
        hi_i = max(0, min(lo_i + 1, L - 1))
        frac = max(0.0, min(src_pos - float(lo_i), 1.0))
        x_lag = x_f64[:, lo_i] + frac * (x_f64[:, hi_i] - x_f64[:, lo_i])
        denom = np.maximum(
            np.abs(x_last) + np.abs(x_lag),
            1e-7 * np.maximum(np.abs(x_last), 1.0),
        )
        out_lag[:, p_idx] = x_lag
        out_ratio[:, p_idx] = (x_last - x_lag) / denom
    return out_lag.astype(np.float32), out_ratio.astype(np.float32)
