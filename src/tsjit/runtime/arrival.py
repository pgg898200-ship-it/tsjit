"""Pythonic C11 arrival engines, GapAsyncRuntime, `tsjit.stream(...)`, and `assert_causal`.

Zero mandatory third-party dependencies (works cleanly with plain Python lists,
tuples, and optional NumPy arrays).
"""

from __future__ import annotations

import ctypes
import math
import queue
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..native.build import (
    MARK_NAMES,
    MARK_OK,
    TSJIT_OK,
    CArriveOut,
    CTelemetry,
    load_native_lib,
)
from .calibration import EventGate, OnlineACI, QuantilePitEValueDetector
from .spec import StreamSpec, validate_spec

_Z9: tuple[float, ...] = (
    -1.2815515655446004,
    -0.8416212335729143,
    -0.5244005127080407,
    -0.2533471031357997,
    0.0,
    0.2533471031357997,
    0.5244005127080407,
    0.8416212335729143,
    1.2815515655446004,
)

_C_DOUBLE_9 = ctypes.c_double * 9
_C_DOUBLE_4 = ctypes.c_double * 4
_C_DOUBLE_5 = ctypes.c_double * 5


def _is_real_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _safe_ldexp(val: float, exp: int) -> float:
    if val == 0.0 or not math.isfinite(val):
        return val
    try:
        res = math.ldexp(val, exp)
    except OverflowError:
        return math.copysign(sys.float_info.max, val)
    return res if math.isfinite(res) else math.copysign(sys.float_info.max, val)


def _safe_normal_q9(center: float, scale: float) -> tuple[float, ...]:
    c = float(center) if math.isfinite(float(center)) else 0.0
    sc = float(scale) if (math.isfinite(float(scale)) and float(scale) > 0.0) else 1.0
    max_f = sys.float_info.max
    out: list[float] = []
    for z in _Z9:
        step = z * sc
        v = c + step if math.isfinite(step) else math.copysign(max_f, z)
        if not math.isfinite(v):
            v = math.copysign(max_f, z if z != 0.0 else c)
        out.append(v)
    out.sort()
    return tuple(out)


def _interp_to_q9(seq: Sequence[Any]) -> tuple[float, ...] | None:
    """Linearly interpolate an arbitrary Q >= 2 quantile vector onto 9 deciles [0.1..0.9]."""
    n = len(seq)
    if n < 2 or not all(_is_real_num(v) for v in seq):
        return None
    vals = [float(v) for v in seq]
    if not all(math.isfinite(v) for v in vals):
        return None
    vals.sort()
    if n == 9:
        return tuple(vals)
    # For Q=21 (amazon/chronos-2 [0.01, 0.05, 0.10, ..., 0.90, 0.95, 0.99]) and Q=11,
    # pos = (0.1 * k) * (Q - 1) hits exact decile indices without rounding drift.
    scale = n - 1
    out: list[float] = []
    for k in range(1, 10):
        pos = (0.1 * k) * scale
        lo = int(math.floor(pos))
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        v = vals[lo] * (1.0 - frac) + vals[hi] * frac
        if not math.isfinite(v):
            v = vals[lo]
        out.append(v)
    out.sort()
    return tuple(out)


def _coerce_q9(
    raw: Any,
    fallback_center: float,
    fallback_scale: float,
    *,
    horizon_idx: int = 0,
) -> tuple[float, ...]:
    """Normalize model output (1D/2D/3D quantiles, CompiledTSFM (B,Q,H), dict, or scalar)."""
    explicit_quantiles = False
    explicit_quantile_grid = False
    if isinstance(raw, Mapping):
        for key in ("q9", "quantiles", "forecast", "median", "mean"):
            if key in raw:
                explicit_quantiles = key in ("q9", "quantiles")
                explicit_quantile_grid = key == "quantiles"
                raw = raw[key]
                break
    sc = fallback_scale if (math.isfinite(fallback_scale) and fallback_scale > 0.0) else 1.0

    if _is_real_num(raw):
        c = float(raw) if math.isfinite(float(raw)) else fallback_center
        return _safe_normal_q9(c, sc)

    # Support torch tensors / numpy arrays via detach().cpu().tolist() or tolist()
    if hasattr(raw, "detach"):
        raw = raw.detach()
    if hasattr(raw, "cpu"):
        raw = raw.cpu()
    if hasattr(raw, "tolist"):
        raw = raw.tolist()

    def _from_q9(seq: Sequence[Any]) -> tuple[float, ...] | None:
        if len(seq) == 9 and all(_is_real_num(v) for v in seq):
            vals = [float(v) for v in seq]
            if all(math.isfinite(v) for v in vals):
                vals.sort()
                return tuple(vals)
        return None

    if isinstance(raw, (list, tuple)) and len(raw) > 0:
        # 1D: (9,) quantiles, explicit dict["quantiles"] with Q>=2, or (H,) point forecast
        if _is_real_num(raw[0]):
            q9 = _from_q9(raw)
            if q9 is not None:
                return q9
            if explicit_quantiles and len(raw) >= 2:
                q9 = _interp_to_q9(raw)
                if q9 is not None:
                    return q9
            h_idx = min(max(horizon_idx, 0), len(raw) - 1)
            c = float(raw[h_idx]) if _is_real_num(raw[h_idx]) else fallback_center
            if not math.isfinite(c):
                c = fallback_center
            return _safe_normal_q9(c, sc)

        # 2D: (Q=9, H), (B, Q=9), (Q>=2, H), or (B=1, H)
        if isinstance(raw[0], (list, tuple)) and len(raw[0]) > 0:
            if _is_real_num(raw[0][0]):
                all_rows = all(isinstance(r, (list, tuple)) and len(r) > 0 for r in raw)
                # (Q=9, H)
                if len(raw) == 9 and all_rows:
                    h_idx = min(max(horizon_idx, 0), min(len(r) for r in raw) - 1)
                    q9 = _from_q9([r[h_idx] for r in raw])
                    if q9 is not None:
                        return q9
                # (Q=21, H) e.g. squeezed Chronos-2 (before checking row length == 9!)
                if len(raw) == 21 and all_rows:
                    h_idx = min(max(horizon_idx, 0), min(len(r) for r in raw) - 1)
                    q9 = _interp_to_q9([r[h_idx] for r in raw])
                    if q9 is not None:
                        return q9
                # (B, Q=9)
                q9 = _from_q9(raw[0])
                if q9 is not None:
                    return q9
                # General (Q>=2, H) when single-series model returns 2D quantile grid
                if len(raw) >= 2 and all_rows:
                    h_idx = min(max(horizon_idx, 0), min(len(r) for r in raw) - 1)
                    q9 = _interp_to_q9([r[h_idx] for r in raw])
                    if q9 is not None:
                        return q9
                # (B=1, H) point forecast (or explicit quantiles of shape (1, Q>=2))
                if explicit_quantiles and len(raw[0]) >= 2:
                    q9 = _interp_to_q9(raw[0])
                    if q9 is not None:
                        return q9
                h_idx = min(max(horizon_idx, 0), len(raw[0]) - 1)
                c = float(raw[0][h_idx]) if _is_real_num(raw[0][h_idx]) else fallback_center
                if not math.isfinite(c):
                    c = fallback_center
                return _safe_normal_q9(c, sc)

            # 3D: CompiledTSFM (B, Q, H) (e.g. Chronos-Bolt (B, 9, H), Chronos-2 (B, 21, H),
            # arbitrary Q >= 2) or (B, H, Q=9)
            first_batch = raw[0]
            if isinstance(first_batch[0], (list, tuple)) and len(first_batch[0]) > 0:
                all_rows = all(isinstance(r, (list, tuple)) and len(r) > 0 for r in first_batch)
                # 1. Exact (B, Q=9, H)
                if len(first_batch) == 9 and all_rows:
                    h_idx = min(max(horizon_idx, 0), min(len(r) for r in first_batch) - 1)
                    q9 = _from_q9([r[h_idx] for r in first_batch])
                    if q9 is not None:
                        return q9
                # 2. General (B, Q>=2, H) including Chronos-2 (B, 21, H) and explicit
                # dict["quantiles"] (B, Q>=2, H) even when H == 9
                if (
                    len(first_batch) >= 2
                    and all_rows
                    and (
                        explicit_quantile_grid
                        or len(first_batch) == 21
                        or len(first_batch[0]) != 9
                    )
                ):
                    h_idx = min(max(horizon_idx, 0), min(len(r) for r in first_batch) - 1)
                    q9 = _interp_to_q9([r[h_idx] for r in first_batch])
                    if q9 is not None:
                        return q9
                # 3. (B, H, Q=9)
                h_idx_ax1 = min(max(horizon_idx, 0), len(first_batch) - 1)
                q9 = _from_q9(first_batch[h_idx_ax1])
                if q9 is not None:
                    return q9
                # 4. (B, 1, Q>=2)
                if len(first_batch) == 1 and len(first_batch[0]) >= 2:
                    q9 = _interp_to_q9(first_batch[0])
                    if q9 is not None:
                        return q9

    return _safe_normal_q9(fallback_center, sc)


def _invoke_model(model: Any, ctx: Sequence[float]) -> Any:
    """Invoke a callable forecaster, CompiledTSFM, or object with predict_quantiles/predict."""
    if hasattr(model, "predict_quantiles") and callable(model.predict_quantiles):
        try:
            return model.predict_quantiles(ctx)
        except Exception:
            return model.predict_quantiles([list(ctx)])
    if callable(model):
        return model(ctx)
    if hasattr(model, "predict") and callable(model.predict):
        try:
            return model.predict(ctx)
        except Exception:
            return model.predict([list(ctx)])
    raise TypeError(
        f"model must be callable or expose predict_quantiles/predict (got {type(model).__name__})"
    )


def default_naive_model(context: Sequence[float]) -> tuple[float, ...]:
    """Built-in strictly causal zero-dependency EWMA/rolling quantile forecaster."""
    finite = [float(v) for v in context if math.isfinite(float(v))]
    if not finite:
        return tuple(_Z9)
    n = len(finite)
    max_abs = max(abs(v) for v in finite)
    if max_abs == 0.0:
        return tuple(_Z9)
    if max_abs > 1e140 or max_abs < 1e-140:
        # Power-of-2 prescaling prevents OverflowError on [-1.5e308, 1.5e308] and underflow
        _, e_max = math.frexp(max_abs)
        shift = e_max - 1
        scaled = [math.ldexp(v, -shift) for v in finite]
        mu_s = math.fsum(scaled) / n
        mu = _safe_ldexp(mu_s, shift)
        if n > 1:
            ss_s = math.fsum((vs - mu_s) * (vs - mu_s) for vs in scaled)
            sd_s = math.sqrt(max(ss_s / (n - 1), 0.0))
            sd = _safe_ldexp(sd_s, shift)
        else:
            sd = 0.0
    else:
        anchor = finite[0]
        sum_d = math.fsum(v - anchor for v in finite)
        mu = anchor + sum_d / n
        if n > 1:
            ss = math.fsum((v - mu) * (v - mu) for v in finite)
            sd = math.sqrt(max(ss / (n - 1), 0.0))
        else:
            sd = 0.0
    # Blend level toward latest observation for responsive tracking
    center = 0.5 * mu + 0.5 * finite[-1]
    sc = max(sd, 1e-4 * abs(finite[0])) if (sd > 0.0 or finite[0] != 0.0) else 1.0
    return _safe_normal_q9(center, sc)


@dataclass(frozen=True)
class ArriveStepResult:
    step: int
    key: int
    q9: tuple[float, ...]
    abs_sg: float
    rolling_mean: float
    rolling_std: float
    eff_scale: float
    age: int
    mark: int
    mark_name: str
    pwl_path: int
    family_weights: tuple[float, float, float]


@dataclass(frozen=True)
class StepResult:
    step: int
    key: int
    q9: tuple[float, ...]
    cal_lo: float
    cal_hi: float
    abs_sg: float
    rolling_mean: float
    rolling_std: float
    eff_scale: float
    age: int
    mark: int
    mark_name: str
    pwl_path: int
    family_weights: tuple[float, float, float]
    e_value: float
    e_ville: float
    e_cusum: float
    e_sr: float
    alarm: bool
    dispatched: bool
    trigger_reason: str


class PwlArrivalEngine:
    """Wrapper around C11 hot/cold PWL-ARRIVE engine & Boehm-2012 CAS seqlock."""

    def __init__(
        self,
        n_keys: int = 1,
        window: int = 64,
        *,
        lambda_age: float = 0.15,
        max_age: int = 16,
    ) -> None:
        self._handle: Any = None
        self._lib: Any = None
        self._lib = load_native_lib()
        self.n_keys = int(n_keys)
        self.window = int(window)
        self._handle = self._lib.tsjit_arrive_create(
            self.n_keys, self.window, float(lambda_age), int(max_age)
        )
        if not self._handle:
            raise ValueError(
                f"Failed to create native PwlArrivalEngine(n_keys={n_keys}, window={window})"
            )

    def set_params(self, *, lambda_age: float, max_age: int) -> None:
        if not self._handle:
            raise RuntimeError("PwlArrivalEngine is closed")
        rc = int(self._lib.tsjit_arrive_set_params(self._handle, float(lambda_age), int(max_age)))
        if rc != TSJIT_OK:
            raise ValueError(
                f"tsjit_arrive_set_params(lambda_age={lambda_age}, max_age={max_age}) "
                f"failed (rc={rc})"
            )

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._lib.tsjit_arrive_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> PwlArrivalEngine:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def publish_forecast(
        self,
        key: int,
        forecast_step: int,
        q_slow: Sequence[float],
        q_fast: Sequence[float] | None = None,
        *,
        scale_hint: float = 0.0,
        knots: Sequence[float] | None = None,
        w_fast: Sequence[float] | None = None,
        gain: Sequence[float] | None = None,
    ) -> int:
        if not self._handle:
            raise RuntimeError("PwlArrivalEngine is closed")
        qs_list = [float(v) for v in q_slow]
        if len(qs_list) != 9:
            raise ValueError(f"q_slow must contain exactly 9 quantiles, got {len(qs_list)}")
        qf_src = q_fast if q_fast is not None else qs_list
        qf_list = [float(v) for v in qf_src]
        if len(qf_list) != 9:
            raise ValueError(f"q_fast must contain exactly 9 quantiles, got {len(qf_list)}")
        qs = _C_DOUBLE_9(*qs_list)
        qf = _C_DOUBLE_9(*qf_list)
        if knots is not None:
            k_list = [float(v) for v in knots]
            if len(k_list) != 4:
                raise ValueError(f"knots must contain exactly 4 values, got {len(k_list)}")
            pk = _C_DOUBLE_4(*k_list)
        else:
            pk = None
        if w_fast is not None:
            w_list = [float(v) for v in w_fast]
            if len(w_list) != 5:
                raise ValueError(f"w_fast must contain exactly 5 values, got {len(w_list)}")
            pw = _C_DOUBLE_5(*w_list)
        else:
            pw = None
        if gain is not None:
            g_list = [float(v) for v in gain]
            if len(g_list) != 5:
                raise ValueError(f"gain must contain exactly 5 values, got {len(g_list)}")
            pg = _C_DOUBLE_5(*g_list)
        else:
            pg = None
        rc = int(
            self._lib.tsjit_arrive_publish_forecast(
                self._handle,
                int(key),
                int(forecast_step),
                qs,
                qf,
                float(scale_hint),
                pk,
                pw,
                pg,
            )
        )
        return rc

    def step(self, key: int, step: int, x: float) -> ArriveStepResult:
        if not self._handle:
            raise RuntimeError("PwlArrivalEngine is closed")
        out = CArriveOut()
        rc = int(
            self._lib.tsjit_arrive_step_pwl(
                self._handle, int(key), int(step), float(x), ctypes.byref(out)
            )
        )
        if rc != TSJIT_OK and out.mark != 6:
            raise RuntimeError(f"tsjit_arrive_step_pwl failed with code {rc}")
        mark = int(out.mark)
        return ArriveStepResult(
            step=int(step),
            key=int(key),
            q9=tuple(float(out.q[i]) for i in range(9)),
            abs_sg=float(out.abs_sg),
            rolling_mean=float(out.rolling_mean),
            rolling_std=float(out.rolling_std),
            eff_scale=float(out.eff_scale),
            age=int(out.age),
            mark=mark,
            mark_name=MARK_NAMES.get(mark, f"MARK_{mark}"),
            pwl_path=int(out.pwl_path),
            family_weights=(
                float(out.family_weights[0]),
                float(out.family_weights[1]),
                float(out.family_weights[2]),
            ),
        )

    def get_rolling(self, key: int) -> dict[str, Any]:
        if not self._handle:
            raise RuntimeError("PwlArrivalEngine is closed")
        mean = ctypes.c_double()
        std = ctypes.c_double()
        anchor = ctypes.c_double()
        count = ctypes.c_int64()
        last_step = ctypes.c_int64()
        rc = int(
            self._lib.tsjit_arrive_get_rolling(
                self._handle,
                int(key),
                ctypes.byref(mean),
                ctypes.byref(std),
                ctypes.byref(anchor),
                ctypes.byref(count),
                ctypes.byref(last_step),
            )
        )
        if rc != TSJIT_OK:
            raise IndexError(f"Key {key} out of bounds (rc={rc})")
        return {
            "mean": float(mean.value),
            "std": float(std.value),
            "anchor": float(anchor.value),
            "count": int(count.value),
            "last_step": int(last_step.value),
        }

    def read_telemetry(self) -> dict[str, Any]:
        if not self._handle:
            raise RuntimeError("PwlArrivalEngine is closed")
        telem = CTelemetry()
        rc = int(self._lib.tsjit_read_telemetry(self._handle, ctypes.byref(telem)))
        if rc != TSJIT_OK:
            raise RuntimeError(f"tsjit_read_telemetry failed (rc={rc})")
        return telem.to_dict()


class SoftRevisionEngine(PwlArrivalEngine):
    """Age-aware hierarchical 3-family SoftRevisionV2 engine (equal family prior)."""

    def step_soft(
        self,
        key: int,
        step: int,
        x: float,
        *,
        slow_models: Sequence[Sequence[float]] | None = None,
        slow_losses: Sequence[float] | None = None,
        fast_models: Sequence[Sequence[float]] | None = None,
        fast_losses: Sequence[float] | None = None,
        persist_models: Sequence[Sequence[float]] | None = None,
        persist_losses: Sequence[float] | None = None,
        tau_intra: float = 1.0,
        tau_family: float = 1.0,
        beta_surprise: float = 0.25,
    ) -> ArriveStepResult:
        if not self._handle:
            raise RuntimeError("SoftRevisionEngine is closed")

        def _pack(models: Sequence[Sequence[float]] | None, losses: Sequence[float] | None):
            if not models:
                return 0, None, None
            n = len(models)
            flat = [float(v) for m in models for v in m]
            if len(flat) != n * 9:
                raise ValueError("Each sub-model must supply exactly 9 quantiles")
            q_arr = (ctypes.c_double * len(flat))(*flat)
            if losses is not None:
                loss_list = [float(v) for v in losses]
                if len(loss_list) != n:
                    raise ValueError(
                        f"losses length ({len(loss_list)}) must match models length ({n})"
                    )
                l_arr = (ctypes.c_double * n)(*loss_list)
            else:
                l_arr = (ctypes.c_double * n)(*([0.0] * n))
            return n, q_arr, l_arr

        ns, qs_p, ls_p = _pack(slow_models, slow_losses)
        nf, qf_p, lf_p = _pack(fast_models, fast_losses)
        np_, qp_p, lp_p = _pack(persist_models, persist_losses)

        out = CArriveOut()
        rc = int(
            self._lib.tsjit_arrive_step_soft_v2(
                self._handle,
                int(key),
                int(step),
                float(x),
                ns,
                qs_p,
                ls_p,
                nf,
                qf_p,
                lf_p,
                np_,
                qp_p,
                lp_p,
                float(tau_intra),
                float(tau_family),
                float(beta_surprise),
                ctypes.byref(out),
            )
        )
        if rc != TSJIT_OK and out.mark != 6:
            raise RuntimeError(f"tsjit_arrive_step_soft_v2 failed with code {rc}")
        mark = int(out.mark)
        return ArriveStepResult(
            step=int(step),
            key=int(key),
            q9=tuple(float(out.q[i]) for i in range(9)),
            abs_sg=float(out.abs_sg),
            rolling_mean=float(out.rolling_mean),
            rolling_std=float(out.rolling_std),
            eff_scale=float(out.eff_scale),
            age=int(out.age),
            mark=mark,
            mark_name=MARK_NAMES.get(mark, f"MARK_{mark}"),
            pwl_path=int(out.pwl_path),
            family_weights=(
                float(out.family_weights[0]),
                float(out.family_weights[1]),
                float(out.family_weights[2]),
            ),
        )


class GapAsyncRuntime:
    """High-level causal streaming runtime bridging any forecaster with C11 seqlocked
    PWL/SoftRevision arrival engines, OnlineACI, Quantile-PIT E-Value detection,
    and LAZY-PREP EventGate (sync or background gap worker).
    """

    def __init__(
        self,
        model: Any = None,
        *,
        spec: StreamSpec | Mapping[str, Any] | None = None,
        n_keys: int = 1,
        async_worker: bool = False,
        fast_ridge_lags: int = 4,
    ) -> None:
        # Initialize teardown-safe attributes FIRST so a failed __init__ never triggers
        # AttributeError inside __del__ -> close().
        self._closed: bool = True
        self._stop_event: threading.Event | None = None
        self._worker_thread: threading.Thread | None = None
        self._schur: dict[int, int] = {}
        self.engine: SoftRevisionEngine | None = None  # type: ignore[assignment]
        self._lib: Any = None
        self._state_lock = threading.Lock()
        self._pending_cond = threading.Condition(self._state_lock)
        self._pending_jobs: int = 0
        self._worker_alive: bool = False
        self._closing: bool = False
        self._native_destroyed: bool = False
        self.worker_errors: list[Exception] = []
        self.async_error_count: int = 0
        self.async_publishes: int = 0
        self.async_worker: bool = False

        self.spec = validate_spec(spec) if spec is not None else StreamSpec()
        self.model = model if model is not None else default_naive_model
        if isinstance(n_keys, bool) or not isinstance(n_keys, int) or n_keys < 1:
            raise ValueError(f"n_keys must be an int >= 1, got {n_keys!r}")
        if (
            isinstance(fast_ridge_lags, bool)
            or not isinstance(fast_ridge_lags, int)
            or not (1 <= fast_ridge_lags <= 2047)
        ):
            raise ValueError(
                f"fast_ridge_lags must be an int in [1, 2047], got {fast_ridge_lags!r}"
            )
        self.n_keys = int(n_keys)
        self.async_worker = bool(async_worker)
        self.fast_ridge_lags = int(fast_ridge_lags)

        self.engine = SoftRevisionEngine(
            n_keys=self.n_keys,
            window=self.spec.window,
            lambda_age=self.spec.lambda_age,
            max_age=self.spec.max_age,
        )
        self._lib = load_native_lib()

        # Per-key OnlineACI, E-Value detector, EventGate, and Winsorized Schur Fast-Ridge
        self._aci: dict[int, OnlineACI] = {}
        self._evalue: dict[int, QuantilePitEValueDetector] = {}
        self._gate: dict[int, EventGate] = {}
        self._lag_hist: dict[int, list[float]] = {}
        self._evalue_queue: dict[int, deque[tuple[int, tuple[float, ...]]]] = {}
        self._has_forecast: dict[int, bool] = {}
        self._last_publish_step: dict[int, int | None] = {}

        for k in range(self.n_keys):
            self._aci[k] = OnlineACI(
                alpha=self.spec.alpha,
                gamma=self.spec.aci_gamma,
                delay=0,
                window=max(32, min(self.spec.window, 512)),
            )
            self._evalue[k] = QuantilePitEValueDetector(threshold=self.spec.evalue_threshold)
            self._gate[k] = EventGate(
                surprise_threshold=self.spec.surprise_threshold,
                evalue_threshold=self.spec.evalue_threshold,
                max_age=self.spec.max_age,
                gate_expr=self.spec.gate_expr,
            )
            sh = self._lib.tsjit_schur_create(
                self.fast_ridge_lags + 1, 1, self.spec.ewma_gamma, self.spec.ridge_lambda, 20.0
            )
            if sh:
                self._schur[k] = sh
            self._lag_hist[k] = []
            self._evalue_queue[k] = deque()
            self._has_forecast[k] = False
            self._last_publish_step[k] = None

        self._stop_event = threading.Event()
        self._job_queue: queue.Queue[
            tuple[int, int, list[float], float, tuple[float, ...] | None] | None
        ] = queue.Queue(maxsize=max(8, self.n_keys * 2))
        self._closed = False
        if self.async_worker:
            self._worker_alive = True
            self._worker_thread = threading.Thread(
                target=self._gap_worker_loop, name="tsjit-gap-worker", daemon=True
            )
            try:
                self._worker_thread.start()
            except Exception:
                self._worker_alive = False
                self.close()
                raise

    def _gap_worker_loop(self) -> None:
        stop_ev = self._stop_event
        assert stop_ev is not None
        try:
            while not stop_ev.is_set():
                try:
                    job = self._job_queue.get(timeout=0.02)
                except queue.Empty:
                    continue
                if job is None:
                    self._job_queue.task_done()
                    break
                key, step, ctx, eff_sc, lag_feats = job
                try:
                    self._run_model_and_publish(key, step, ctx, eff_sc, lag_feats=lag_feats)
                    with self._state_lock:
                        self.async_publishes += 1
                except Exception as exc:
                    with self._state_lock:
                        self.async_error_count += 1
                        if len(self.worker_errors) < 64:
                            self.worker_errors.append(exc)
                finally:
                    self._job_queue.task_done()
                    with self._pending_cond:
                        if self._pending_jobs > 0:
                            self._pending_jobs -= 1
                        if self._pending_jobs == 0:
                            self._pending_cond.notify_all()
        finally:
            # Drain any unstarted queued jobs so flush_async() never deadlocks
            drained = 0
            while True:
                try:
                    item = self._job_queue.get_nowait()
                except queue.Empty:
                    break
                self._job_queue.task_done()
                if item is not None:
                    drained += 1
            with self._pending_cond:
                self._pending_jobs = max(0, self._pending_jobs - drained)
                self._worker_alive = False
                self._pending_cond.notify_all()
                should_destroy = self._closing and not self._native_destroyed
                if should_destroy:
                    self._native_destroyed = True
            if should_destroy:
                self._destroy_native_resources()

    def _snapshot_lag_feats(self, key: int) -> tuple[float, ...] | None:
        hist = self._lag_hist[key]
        if len(hist) < self.fast_ridge_lags:
            return None
        return (1.0, *hist[-self.fast_ridge_lags :])

    def _fast_ridge_forecast(
        self,
        key: int,
        center: float,
        eff_sc: float,
        lag_feats: tuple[float, ...] | None = None,
    ) -> tuple[float, ...]:
        sh = self._schur.get(key)
        feats = lag_feats if lag_feats is not None else self._snapshot_lag_feats(key)
        if not sh or feats is None:
            return _safe_normal_q9(center, eff_sc)
        c_x = (ctypes.c_double * len(feats))(*feats)
        c_y = ctypes.c_double(0.0)
        rc = int(self._lib.tsjit_schur_predict(sh, c_x, ctypes.byref(c_y)))
        pred = float(c_y.value) if (rc == TSJIT_OK and math.isfinite(float(c_y.value))) else center
        return _safe_normal_q9(pred, eff_sc)

    def _run_model_and_publish(
        self,
        key: int,
        step: int,
        ctx: Sequence[float],
        eff_sc: float,
        *,
        lag_feats: tuple[float, ...] | None = None,
    ) -> None:
        center = ctx[-1] if ctx else 0.0
        raw = _invoke_model(self.model, ctx)
        q_slow = _coerce_q9(raw, center, eff_sc, horizon_idx=self.spec.horizon - 1)
        with self._state_lock:
            if self._closing or self._native_destroyed or self.engine is None:
                return
        q_fast = self._fast_ridge_forecast(key, center, eff_sc, lag_feats=lag_feats)
        self.engine.publish_forecast(key, step, q_slow, q_fast, scale_hint=eff_sc)
        self._has_forecast[key] = True
        self._last_publish_step[key] = int(step)

    def step(self, step: int, x: float, *, key: int = 0) -> StepResult:
        """Process one strictly causal arrival tick."""
        if self._closed or self.engine is None:
            raise RuntimeError("GapAsyncRuntime is closed")
        if key < 0 or key >= self.n_keys:
            res = self.engine.step(key, step, x)
            return StepResult(
                step=res.step,
                key=res.key,
                q9=res.q9,
                cal_lo=res.q9[0],
                cal_hi=res.q9[8],
                abs_sg=res.abs_sg,
                rolling_mean=res.rolling_mean,
                rolling_std=res.rolling_std,
                eff_scale=res.eff_scale,
                age=res.age,
                mark=res.mark,
                mark_name=res.mark_name,
                pwl_path=res.pwl_path,
                family_weights=res.family_weights,
                e_value=1.0,
                e_ville=1.0,
                e_cusum=1.0,
                e_sr=0.0,
                alarm=False,
                dispatched=False,
                trigger_reason="invalid_key",
            )

        aci = self._aci[key]
        ev = self._evalue[key]
        gate = self._gate[key]
        xf = float(x)

        # 1. Causal arrival through C11 engine (detects DUP/OOO BEFORE mutating ACI/EValue!)
        mode = self.spec.revision_mode
        if mode == "soft_v2":
            arr = self.engine.step_soft(key, step, xf)
        elif mode == "hybrid":
            last_pub = self._last_publish_step[key]
            if last_pub is not None and (int(step) - last_pub) > 1:
                arr = self.engine.step_soft(key, step, xf)
            else:
                arr = self.engine.step(key, step, xf)
        else:
            arr = self.engine.step(key, step, xf)

        if arr.mark in (4, 5):  # MARK_DUP_STEP or MARK_OOO_STEP: zero state mutation!
            return StepResult(
                step=arr.step,
                key=arr.key,
                q9=arr.q9,
                cal_lo=arr.q9[0],
                cal_hi=arr.q9[8],
                abs_sg=arr.abs_sg,
                rolling_mean=arr.rolling_mean,
                rolling_std=arr.rolling_std,
                eff_scale=arr.eff_scale,
                age=arr.age,
                mark=arr.mark,
                mark_name=arr.mark_name,
                pwl_path=arr.pwl_path,
                family_weights=arr.family_weights,
                e_value=ev.last_e,
                e_ville=ev.ville,
                e_cusum=ev.cusum,
                e_sr=ev.shiryaev_roberts,
                alarm=False,
                dispatched=False,
                trigger_reason="rejected_" + arr.mark_name.lower(),
            )

        # 2. Causally update online Schur Fast-Ridge using lagged history prior to step
        if arr.mark == MARK_OK and math.isfinite(xf):
            hist = self._lag_hist[key]
            sh = self._schur.get(key)
            if sh and len(hist) >= self.fast_ridge_lags:
                feats = [1.0, *hist[-self.fast_ridge_lags :]]
                c_x = (ctypes.c_double * len(feats))(*feats)
                c_y = ctypes.c_double(xf)
                self._lib.tsjit_schur_update(sh, c_x, ctypes.byref(c_y), None, None)
            hist.append(xf)
            if len(hist) > max(self.spec.window, self.fast_ridge_lags + 4):
                del hist[: -max(self.spec.window, self.fast_ridge_lags + 4)]

        # 3. Causally score realized xf against due (step - horizon) forecasts in E-Value & ACI
        ev_q = self._evalue_queue[key]
        e_upd = None
        if arr.mark == MARK_OK:
            aci.update(step, xf)
            while ev_q and ev_q[0][0] <= step:
                _due, due_q9 = ev_q.popleft()
                e_upd = ev.update(xf, due_q9)
                if e_upd.alarm:
                    aci.boost_on_alarm(severity=1.0)
        else:
            # Drain expired queue items without scoring NaN observations
            aci.update(step, float("nan"))
            while ev_q and ev_q[0][0] <= step:
                ev_q.popleft()

        e_val = e_upd.e_value if e_upd is not None else 1.0
        e_ville = e_upd.ville if e_upd is not None else ev.ville
        e_cusum = e_upd.cusum if e_upd is not None else ev.cusum
        e_sr = e_upd.shiryaev_roberts if e_upd is not None else ev.shiryaev_roberts
        alarm = e_upd.alarm if e_upd is not None else False

        # 4. LAZY-PREP ingest & EventGate evaluation
        gate.ingest_lazy(step, xf)
        if not self._has_forecast[key] and arr.mark == MARK_OK:
            decision_dispatch = True
            decision_reason = "cold_start"
            gate.dispatched_steps += 1
        else:
            dec = gate.evaluate(
                abs_sg=arr.abs_sg,
                e_value=e_val,
                age=arr.age,
                e_cusum=e_cusum,
                e_ville=e_ville,
                e_sr=e_sr,
            )
            decision_dispatch = dec.dispatch
            decision_reason = dec.reason

        if decision_dispatch and arr.mark == MARK_OK:
            ctx = gate.drain_lazy_context(max_len=self.spec.window)
            lag_snap = self._snapshot_lag_feats(key)
            if self.async_worker:
                with self._state_lock:
                    if not self._closing:
                        try:
                            self._job_queue.put_nowait((key, step, ctx, arr.eff_scale, lag_snap))
                            self._pending_jobs += 1
                        except queue.Full:
                            pass
            else:
                self._run_model_and_publish(key, step, ctx, arr.eff_scale, lag_feats=lag_snap)
            if alarm:
                ev.reset_cusum()

        # 5. Emit CQR-calibrated interval for step + horizon and enqueue for causal E-Value scoring
        cal = aci.predict_interval(
            step + self.spec.horizon,
            arr.q9[0],
            arr.q9[8],
            scale_hint=arr.eff_scale,
            anchor=arr.rolling_mean,
            enqueue=True,
        )
        ev_q.append((int(step) + self.spec.horizon, arr.q9))

        return StepResult(
            step=arr.step,
            key=arr.key,
            q9=arr.q9,
            cal_lo=cal.lo,
            cal_hi=cal.hi,
            abs_sg=arr.abs_sg,
            rolling_mean=arr.rolling_mean,
            rolling_std=arr.rolling_std,
            eff_scale=arr.eff_scale,
            age=arr.age,
            mark=arr.mark,
            mark_name=arr.mark_name,
            pwl_path=arr.pwl_path,
            family_weights=arr.family_weights,
            e_value=e_val,
            e_ville=e_ville,
            e_cusum=e_cusum,
            e_sr=e_sr,
            alarm=alarm,
            dispatched=decision_dispatch,
            trigger_reason=decision_reason,
        )

    def flush_async(self, timeout: float = 2.0) -> bool:
        """Wait up to `timeout` seconds for pending async worker jobs to complete.

        Returns True if all pending jobs drained before the deadline, False on timeout.
        """
        if not self.async_worker:
            return True
        t_val = float(timeout)
        if math.isnan(t_val) or t_val <= 0.0:
            with self._pending_cond:
                return self._pending_jobs == 0
        infinite_wait = math.isinf(t_val) and t_val > 0.0
        deadline = 0.0 if infinite_wait else (time.monotonic() + t_val)
        with self._pending_cond:
            while self._pending_jobs > 0 and self._worker_alive:
                if infinite_wait:
                    self._pending_cond.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        return False
                    self._pending_cond.wait(timeout=remaining)
            return self._pending_jobs == 0

    def telemetry(self) -> dict[str, Any]:
        if self._closed or self.engine is None:
            raise RuntimeError("GapAsyncRuntime is closed")
        return self.engine.read_telemetry()

    def _destroy_native_resources(self) -> None:
        schur_handles = list(self._schur.values())
        self._schur.clear()
        lib = self._lib
        for sh in schur_handles:
            if sh and lib is not None:
                lib.tsjit_schur_destroy(sh)
        eng = self.engine
        if eng is not None:
            eng.close()

    def close(self) -> None:
        if getattr(self, "_closed", True):
            # If __init__ failed midway before _closed became False, still release any
            # partially created native handles safely.
            if (
                hasattr(self, "_pending_cond")
                and not getattr(self, "_worker_alive", False)
                and not getattr(self, "_native_destroyed", True)
            ):
                with self._pending_cond:
                    if not self._worker_alive and not self._native_destroyed:
                        self._native_destroyed = True
                        do_destroy = True
                    else:
                        do_destroy = False
                if do_destroy:
                    self._destroy_native_resources()
            return

        self._closed = True
        stop_ev = getattr(self, "_stop_event", None)
        if stop_ev is not None:
            stop_ev.set()
        with self._state_lock:
            self._closing = True
        wt = getattr(self, "_worker_thread", None)
        if wt is not None and wt.is_alive():
            try:
                self._job_queue.put_nowait(None)
            except queue.Full:
                pass
            wt.join(timeout=2.0)

        # Destroy native handles ONLY once the worker has finished its loop. If a slow (>2s)
        # user model call is still in progress, defer native destruction to the worker's
        # `finally:` block so tsjit_arrive_destroy / tsjit_schur_destroy NEVER UAFs!
        with self._pending_cond:
            if not self._worker_alive and not self._native_destroyed:
                self._native_destroyed = True
                should_destroy = True
            else:
                should_destroy = False
        if should_destroy:
            self._destroy_native_resources()
        self._worker_thread = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> GapAsyncRuntime:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def stream(
    model: Any = None,
    *,
    spec: StreamSpec | Mapping[str, Any] | None = None,
    n_keys: int = 1,
    async_worker: bool = False,
) -> GapAsyncRuntime:
    """Create a causal streaming runtime wrapping any eager or compiled forecaster."""
    return GapAsyncRuntime(model=model, spec=spec, n_keys=n_keys, async_worker=async_worker)


def assert_causal(
    runtime_factory: Callable[[], GapAsyncRuntime],
    series: Sequence[float],
    *,
    split_step: int,
    poison_values: Sequence[float] = (1e12, -1e12, float("nan")),
) -> dict[str, Any]:
    """Audit strict causality: verify 0.0 bit-exact diff for all steps <= split_step
    when future inputs (t > split_step) are poisoned with +1e12, -1e12, and NaN.
    """
    n = len(series)
    if split_step < 0 or split_step >= n - 1:
        raise ValueError(f"split_step={split_step} must be in [0, {n - 2}]")

    with runtime_factory() as rt_clean:
        clean_out = [rt_clean.step(t, float(series[t])) for t in range(n)]

    max_diff = 0.0
    for idx, pval in enumerate(poison_values):
        poisoned = list(series)
        for t in range(split_step + 1, n):
            poisoned[t] = pval
        with runtime_factory() as rt_pois:
            pois_out = [rt_pois.step(t, float(poisoned[t])) for t in range(n)]

        for t in range(split_step + 1):
            c = clean_out[t]
            p = pois_out[t]
            if c.mark != p.mark or c.dispatched != p.dispatched:
                raise AssertionError(
                    f"Causal leakage at step {t} (poison #{idx}={pval}): "
                    f"mark/dispatched mismatch ({c.mark},{c.dispatched}) vs "
                    f"({p.mark},{p.dispatched})"
                )
            for field_name, cv, pv in (
                ("cal_lo", c.cal_lo, p.cal_lo),
                ("cal_hi", c.cal_hi, p.cal_hi),
                ("abs_sg", c.abs_sg, p.abs_sg),
                ("e_value", c.e_value, p.e_value),
                ("e_cusum", c.e_cusum, p.e_cusum),
                *[(f"q9[{q}]", c.q9[q], p.q9[q]) for q in range(9)],
            ):
                if math.isnan(cv) and math.isnan(pv):
                    continue
                diff = abs(cv - pv)
                if diff > max_diff:
                    max_diff = diff
                if diff != 0.0:
                    raise AssertionError(
                        f"Causal leakage at step {t} field {field_name} "
                        f"(poison #{idx}={pval}): clean={cv!r}, poisoned={pv!r}, diff={diff!r}"
                    )

    return {
        "causal": True,
        "split_step": split_step,
        "poison_variants": len(poison_values),
        "max_diff": max_diff,
    }
