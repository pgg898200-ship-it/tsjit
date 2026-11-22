"""Online conformal calibration, distribution-free quantile-PIT e-value detection,
and LAZY-PREP causal event gating.

Zero external runtime dependencies (pure Python stdlib).
"""

from __future__ import annotations

import math
import sys
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .spec import GateExpression

TAU9: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
_MAX_QT_CQR: float = 1e12
_MAX_FLOAT: float = sys.float_info.max


def _finite_or(val: float, default: float) -> float:
    return val if math.isfinite(val) else default


def _empirical_quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n == 1:
        return s[0]
    q_clamped = min(max(q, 0.0), 1.0)
    # Finite-sample conformal quantile index ceil((n+1)*q) - 1, clamped to [0, n-1]
    idx = math.ceil((n + 1) * q_clamped) - 1
    idx = min(max(idx, 0), n - 1)
    return s[idx]


@dataclass(frozen=True)
class CalibratedInterval:
    lo: float
    hi: float
    qt_cqr: float
    alpha_t: float
    eff_sc: float
    inverted_guarded: bool


class OnlineACI:
    """Adaptive Conformal Inference (ACI) with CQR scaling, causal delay queue,
    invert-guard, relative micro-scale floor, and score-driven alarm boosting.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.05,
        *,
        delay: int = 0,
        window: int = 256,
        min_alpha: float = 1e-4,
        max_alpha: float = 0.999,
        boost_factor: float = 1.5,
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if not (0.0 <= gamma <= 1.0):
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        if isinstance(delay, bool) or not isinstance(delay, int) or delay < 0:
            raise ValueError(f"delay must be an int >= 0, got {delay!r}")
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ValueError(f"window must be an int >= 1, got {window!r}")
        if not (0.0 < min_alpha < max_alpha < 1.0):
            raise ValueError(
                f"Expected 0 < min_alpha < max_alpha < 1, got ({min_alpha}, {max_alpha})"
            )
        if not (math.isfinite(boost_factor) and boost_factor > 0.0):
            raise ValueError(f"boost_factor must be finite > 0, got {boost_factor}")

        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.delay = int(delay)
        self.window = int(window)
        self.min_alpha = float(min_alpha)
        self.max_alpha = float(max_alpha)
        self.boost_factor = float(boost_factor)

        self.alpha_t: float = float(alpha)
        self.qt_cqr: float = 0.0
        self._boost_floor: float = 0.0
        self._boost_ttl: int = 0
        self.recent_scores: deque[float] = deque(maxlen=self.window)
        # Queue items: (due_step, raw_lo, raw_hi, cal_lo, cal_hi, eff_sc)
        self.delay_queue: deque[tuple[int, float, float, float, float, float]] = deque()

        self.n_updates: int = 0
        self.n_covered: int = 0
        self.inversions_guarded: int = 0
        self.alarm_boosts: int = 0

    @staticmethod
    def eff_sc(
        q_lo: float,
        q_hi: float,
        *,
        scale_hint: float | None = None,
        anchor: float = 0.0,
    ) -> float:
        """Relative micro-scale-safe scale divisor (preserves 1e-6 scale, fallback 1.0)."""
        if math.isfinite(q_lo) and math.isfinite(q_hi):
            diff = q_hi - q_lo
            if not math.isfinite(diff):
                raw_spread = _MAX_FLOAT if q_hi > q_lo else 0.0
            else:
                raw_spread = max(diff, 0.0)
        else:
            raw_spread = 0.0
        hint = (
            float(scale_hint)
            if (scale_hint is not None and math.isfinite(scale_hint) and scale_hint > 0.0)
            else 0.0
        )
        anch_f = float(anchor)
        abs_anch = abs(anch_f) if math.isfinite(anch_f) else 0.0
        base_sc = hint if hint > 0.0 else raw_spread
        ref = hint if hint > 0.0 else (raw_spread if raw_spread > 0.0 else abs_anch)
        floor_val = max(1e-4 * ref, 1e-14 * abs_anch)
        eff = max(base_sc, floor_val)
        if not math.isfinite(eff) or eff <= 0.0:
            return 1.0
        return eff

    def _current_qt_cqr(self) -> float:
        if self.recent_scores:
            q_val = _empirical_quantile(list(self.recent_scores), 1.0 - self.alpha_t)
        else:
            q_val = 0.0
        if self._boost_ttl > 0:
            q_val = max(q_val, self._boost_floor)
        if not math.isfinite(q_val):
            return _MAX_QT_CQR if q_val > 0.0 else 0.0
        return min(max(q_val, -_MAX_QT_CQR), _MAX_QT_CQR)

    def predict_interval(
        self,
        step: int,
        q_lo: float,
        q_hi: float,
        *,
        scale_hint: float | None = None,
        anchor: float = 0.0,
        enqueue: bool = True,
    ) -> CalibratedInterval:
        """Emit a CQR-calibrated interval with invert-guard and enqueue for causal feedback."""
        q_lo_f = float(q_lo)
        q_hi_f = float(q_hi)
        anch_f = _finite_or(float(anchor), 0.0)
        lo_ok = math.isfinite(q_lo_f)
        hi_ok = math.isfinite(q_hi_f)
        if lo_ok and hi_ok:
            lo, hi = q_lo_f, q_hi_f
        elif lo_ok:
            lo = hi = q_lo_f
        elif hi_ok:
            lo = hi = q_hi_f
        else:
            lo = hi = anch_f
        if lo > hi:
            lo, hi = hi, lo
        sc = self.eff_sc(lo, hi, scale_hint=scale_hint, anchor=anch_f)
        qt = self._current_qt_cqr()
        self.qt_cqr = qt

        margin = qt * sc
        if not math.isfinite(margin):
            margin = math.copysign(_MAX_FLOAT, qt)
        cal_lo = lo - margin
        cal_hi = hi + margin
        if not math.isfinite(cal_lo):
            cal_lo = -_MAX_FLOAT if (margin >= 0.0 or lo < 0.0) else _MAX_FLOAT
        if not math.isfinite(cal_hi):
            cal_hi = _MAX_FLOAT if (margin >= 0.0 or hi > 0.0) else -_MAX_FLOAT
        inverted = False
        if cal_lo > cal_hi:
            mid = 0.5 * cal_lo + 0.5 * cal_hi
            cal_lo = mid
            cal_hi = mid
            inverted = True
            self.inversions_guarded += 1

        if enqueue:
            self.delay_queue.append((int(step) + self.delay, lo, hi, cal_lo, cal_hi, sc))
        if self._boost_ttl > 0:
            self._boost_ttl -= 1
            if self._boost_ttl == 0:
                self._boost_floor = 0.0

        return CalibratedInterval(
            lo=cal_lo,
            hi=cal_hi,
            qt_cqr=qt,
            alpha_t=self.alpha_t,
            eff_sc=sc,
            inverted_guarded=inverted,
        )

    def update(self, step: int, y_obs: float) -> int:
        """Causally score all queued intervals whose due_step <= step."""
        resolved = 0
        obs_finite = math.isfinite(y_obs)
        while self.delay_queue and self.delay_queue[0][0] <= step:
            _due, raw_lo, raw_hi, cal_lo, cal_hi, sc = self.delay_queue.popleft()
            resolved += 1
            if not obs_finite:
                continue
            d_lo = raw_lo - y_obs
            d_hi = y_obs - raw_hi
            raw_err = max(d_lo, d_hi)
            if not math.isfinite(raw_err):
                score = _MAX_QT_CQR
            else:
                score = raw_err / sc
                if not math.isfinite(score):
                    score = math.copysign(_MAX_QT_CQR, raw_err)
                else:
                    score = min(max(score, -_MAX_QT_CQR), _MAX_QT_CQR)
            self.recent_scores.append(score)
            covered = cal_lo <= y_obs <= cal_hi
            err = 0.0 if covered else 1.0
            self.n_updates += 1
            if covered:
                self.n_covered += 1
            self.alpha_t = min(
                max(self.alpha_t + self.gamma * (self.alpha - err), self.min_alpha),
                self.max_alpha,
            )
            self.qt_cqr = self._current_qt_cqr()
        return resolved

    def boost_on_alarm(self, severity: float = 1.0, *, duration: int = 8) -> float:
        """Immediately widen CQR radius using actual recent conformal score magnitudes."""
        sev = min(max(float(severity), 0.1), 1e6) if math.isfinite(severity) else 1.0
        if self.recent_scores:
            abs_recent = [abs(s) for s in self.recent_scores if math.isfinite(s)]
            ref_score = _empirical_quantile(abs_recent, 0.95) if abs_recent else 0.5
            max_pos = max((s for s in self.recent_scores if math.isfinite(s)), default=0.0)
            base_mag = max(ref_score, max_pos, 0.25)
            target = base_mag * self.boost_factor * sev
        else:
            target = 0.5 * self.boost_factor * sev
        if not math.isfinite(target):
            target = _MAX_QT_CQR
        target = min(max(target, 0.0), _MAX_QT_CQR)
        self._boost_floor = min(max(self._boost_floor, target), _MAX_QT_CQR)
        self._boost_ttl = max(self._boost_ttl, int(duration))
        self.alpha_t = max(self.min_alpha, self.alpha_t / (1.0 + 0.5 * sev))
        self.qt_cqr = self._current_qt_cqr()
        self.alarm_boosts += 1
        return self.qt_cqr

    @property
    def empirical_coverage(self) -> float:
        return (self.n_covered / self.n_updates) if self.n_updates > 0 else 1.0


@dataclass(frozen=True)
class EValueUpdate:
    pit: float
    p_value: float
    e_value: float
    ville: float
    cusum: float
    shiryaev_roberts: float
    alarm: bool
    cusum_alarm: bool = False
    ville_alarm: bool = False


class QuantilePitEValueDetector:
    """Distribution-free causal quantile-PIT -> Vovk-Wang e-value sequential detector.

    Simultaneously tracks:
      - Unrestarted Ville test martingale M_t = prod e_i (anytime-valid)
      - Page-CUSUM e-detector C_t = max(1, C_{t-1} * e_t)
      - Shiryaev-Roberts e-detector R_t = (R_{t-1} + 1) * e_t
    """

    def __init__(
        self,
        *,
        kappa: float = 0.5,
        threshold: float = 20.0,
        p_floor: float = 1e-6,
    ) -> None:
        if not (0.0 < kappa < 1.0):
            raise ValueError(f"Vovk-Wang kappa must be in (0, 1), got {kappa}")
        if not (math.isfinite(threshold) and threshold > 1.0):
            raise ValueError(f"threshold must be finite > 1.0, got {threshold}")
        if not (0.0 < p_floor < 0.5):
            raise ValueError(f"p_floor must be in (0, 0.5), got {p_floor}")

        self.kappa = float(kappa)
        self.threshold = float(threshold)
        self.p_floor = float(p_floor)

        self._log_ville: float = 0.0
        self.ville: float = 1.0
        self.cusum: float = 1.0
        self.shiryaev_roberts: float = 0.0
        self.last_e: float = 1.0
        self.last_pit: float = 0.5
        self.n_steps: int = 0
        self.n_alarms: int = 0
        self._cusum_restarted: bool = False

    @staticmethod
    def compute_pit(y: float, q9: Sequence[float]) -> float:
        """Causal piecewise-linear probability integral transform u in (0, 1)."""
        if len(q9) != 9:
            raise ValueError(f"Expected 9 quantiles, got {len(q9)}")
        if not math.isfinite(y) or any(not math.isfinite(v) for v in q9):
            return 0.5
        q = sorted(float(v) for v in q9)
        scale_half = max(abs(y), abs(q[0]), abs(q[8])) > 1e150

        def _diff(a: float, b: float) -> float:
            return (0.5 * a - 0.5 * b) if scale_half else (a - b)

        if y < q[0]:
            half_span = _diff(q[4], q[0])
            full_span = _diff(q[8], q[0])
            s_lo = half_span if half_span > 1e-300 else max(0.5 * full_span, 1e-12)
            arg = min(_diff(q[0], y) / s_lo, 50.0)
            return max(1e-12, 0.1 * math.exp(-arg))
        if y > q[8]:
            half_span = _diff(q[8], q[4])
            full_span = _diff(q[8], q[0])
            s_hi = half_span if half_span > 1e-300 else max(0.5 * full_span, 1e-12)
            arg = min(_diff(y, q[8]) / s_hi, 50.0)
            return min(1.0 - 1e-12, 1.0 - 0.1 * math.exp(-arg))

        for i in range(8):
            if q[i] <= y <= q[i + 1]:
                span = _diff(q[i + 1], q[i])
                if span <= 0.0:
                    return 0.5 * (TAU9[i] + TAU9[i + 1])
                frac = _diff(y, q[i]) / span
                return TAU9[i] + frac * (TAU9[i + 1] - TAU9[i])
        return 0.5

    def vovk_wang_calibrator(self, p_value: float) -> float:
        """Exact Vovk-Wang p-to-e calibrator e(p) = kappa * p^(kappa - 1)."""
        p = min(max(float(p_value), self.p_floor), 1.0)
        e = self.kappa * (p ** (self.kappa - 1.0))
        return e if math.isfinite(e) and e >= 0.0 else 1.0

    def update(self, y: float, q9: Sequence[float]) -> EValueUpdate:
        """Causally score realized observation y against prior forecast q9."""
        if len(q9) != 9:
            raise ValueError(f"Expected 9 quantiles, got {len(q9)}")
        if not math.isfinite(y) or any(not math.isfinite(v) for v in q9):
            return EValueUpdate(
                pit=0.5,
                p_value=1.0,
                e_value=1.0,
                ville=self.ville,
                cusum=self.cusum,
                shiryaev_roberts=self.shiryaev_roberts,
                alarm=False,
                cusum_alarm=False,
                ville_alarm=(self.ville >= self.threshold),
            )

        pit = self.compute_pit(y, q9)
        p_val = min(max(2.0 * min(pit, 1.0 - pit), 0.0), 1.0)
        e_val = self.vovk_wang_calibrator(p_val)

        self.last_pit = pit
        self.last_e = e_val
        self.n_steps += 1

        # Unrestarted Ville test martingale (in log space to prevent overflow)
        if e_val > 0.0:
            self._log_ville = min(max(self._log_ville + math.log(e_val), -700.0), 700.0)
        self.ville = math.exp(self._log_ville)

        # Page-CUSUM e-detector (restarts at 1)
        self.cusum = min(max(1.0, self.cusum * e_val), 1e300)

        # Shiryaev-Roberts e-detector
        self.shiryaev_roberts = min((self.shiryaev_roberts + 1.0) * e_val, 1e300)

        cusum_alarm = self.cusum >= self.threshold
        ville_alarm = self.ville >= self.threshold
        # After reset_cusum(), report sequential cusum_alarm as `alarm` so lifetime
        # unrestarted ville_alarm never causes a sticky alarm storm!
        alarm = cusum_alarm or (ville_alarm and not self._cusum_restarted)
        if alarm:
            self.n_alarms += 1

        return EValueUpdate(
            pit=pit,
            p_value=p_val,
            e_value=e_val,
            ville=self.ville,
            cusum=self.cusum,
            shiryaev_roberts=self.shiryaev_roberts,
            alarm=alarm,
            cusum_alarm=cusum_alarm,
            ville_alarm=ville_alarm,
        )

    def reset_cusum(self) -> None:
        """Reset post-alarm Page-CUSUM / SR detectors while keeping Ville unrestarted."""
        self.cusum = 1.0
        self.shiryaev_roberts = 0.0
        self._cusum_restarted = True

    @staticmethod
    def swarm_mean(e_values: Iterable[float]) -> float:
        """Exact bit-identical commutative arithmetic mean of arbitrarily dependent e-values."""
        vals = [float(v) for v in e_values if math.isfinite(float(v)) and float(v) >= 0.0]
        if not vals:
            return 1.0
        # Sorting prior to math.fsum guarantees 0.0 ULP permutation invariance
        vals.sort()
        max_v = vals[-1]
        if max_v > 1e150:
            _, e_max = math.frexp(max_v)
            shift = e_max - 1
            mean_s = math.fsum(math.ldexp(v, -shift) for v in vals) / len(vals)
            return math.ldexp(mean_s, shift)
        return math.fsum(vals) / len(vals)


@dataclass(frozen=True)
class GateDecision:
    dispatch: bool
    reason: str
    triggers: tuple[str, ...]


class EventGate:
    """Causal dispatch gate with LAZY-PREP ingest-only deferral."""

    def __init__(
        self,
        *,
        surprise_threshold: float = 2.5,
        evalue_threshold: float = 20.0,
        max_age: int = 16,
        gate_expr: str | GateExpression | None = None,
    ) -> None:
        self.surprise_threshold = float(surprise_threshold)
        self.evalue_threshold = float(evalue_threshold)
        self.max_age = int(max_age)
        if isinstance(gate_expr, str):
            self.gate_expr: GateExpression | None = GateExpression.parse(gate_expr)
        else:
            self.gate_expr = gate_expr

        self.ingested_steps: int = 0
        self.deferred_steps: int = 0
        self.dispatched_steps: int = 0
        self.last_reason: str = "init"
        self._lazy_buffer: deque[tuple[int, float]] = deque(maxlen=4096)

    def ingest_lazy(self, step: int, x: float) -> None:
        """LAZY-PREP ingest: record raw observation at zero feature-prep/model cost."""
        self._lazy_buffer.append((int(step), float(x)))
        self.ingested_steps += 1

    def evaluate(
        self,
        *,
        abs_sg: float,
        e_value: float,
        age: int,
        e_cusum: float | None = None,
        e_ville: float | None = None,
        e_sr: float | None = None,
    ) -> GateDecision:
        """Decide causally whether to dispatch Slow-TSFM refresh or defer via LAZY-PREP."""
        ec = e_value if e_cusum is None else e_cusum
        ev = e_value if e_ville is None else e_ville
        esr = e_value if e_sr is None else e_sr

        if self.gate_expr is not None:
            env = {
                "abs_sg": float(abs_sg),
                "e_value": float(e_value),
                "e_cusum": float(ec),
                "e_ville": float(ev),
                "e_sr": float(esr),
                "age": float(age),
                "max_age": float(self.max_age),
                "surprise_threshold": float(self.surprise_threshold),
                "evalue_threshold": float(self.evalue_threshold),
            }
            fire = self.gate_expr.evaluate(env)
            if fire:
                self.dispatched_steps += 1
                self.last_reason = "gate_expr"
                return GateDecision(True, "gate_expr", ("gate_expr",))
            self.deferred_steps += 1
            self.last_reason = "deferred"
            return GateDecision(False, "deferred", ())

        triggers: list[str] = []
        # +Inf surprise or e-value strictly exceeds any finite threshold (only NaN is suppressed)
        if (not math.isnan(abs_sg)) and abs_sg >= self.surprise_threshold:
            triggers.append("surprise")
        e_check = (not math.isnan(e_value)) and e_value >= self.evalue_threshold
        if e_cusum is not None:
            e_check = e_check or ((not math.isnan(ec)) and ec >= self.evalue_threshold)
        elif e_ville is not None:
            e_check = e_check or ((not math.isnan(ev)) and ev >= self.evalue_threshold)
        if e_check:
            triggers.append("evalue")
        if age >= self.max_age:
            triggers.append("max_age")

        if triggers:
            self.dispatched_steps += 1
            reason = "+".join(triggers)
            self.last_reason = reason
            return GateDecision(True, reason, tuple(triggers))

        self.deferred_steps += 1
        self.last_reason = "deferred"
        return GateDecision(False, "deferred", ())

    def drain_lazy_context(self, max_len: int = 256) -> list[float]:
        """Materialize context values only when a dispatch is actually triggered."""
        items = list(self._lazy_buffer)[-max_len:]
        return [v for _s, v in items if math.isfinite(v)]

    @property
    def dispatch_rate(self) -> float:
        total = self.dispatched_steps + self.deferred_steps
        return (self.dispatched_steps / total) if total > 0 else 0.0
