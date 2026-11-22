"""MultiAgentStreamHub wrapping C11 EpochArena + GapAsyncRuntime.

Provides:
  - Shared single-pass rolling channel rings & shifted-anchor moments (zero redundancy)
  - Domain-partitioned NaN-isolated EWMA/Chan-Pebay covariance groups (0.0 NaN bleed)
  - Zero-replay warm-start (`hot_add_agent`) & tombstone preservation across resizes
  - Transactional `tweak()` with atomic rollback on invalid specs + active tweak replay
  - Tamper-evident SHA-256 hash-chained audit ledger
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..native.build import MARK_DUP_STEP, MARK_NAMES, MARK_OOO_STEP, TSJIT_OK, load_native_lib
from .arrival import GapAsyncRuntime, StepResult, _safe_normal_q9
from .calibration import QuantilePitEValueDetector
from .spec import (
    DEFAULT_MAX_CHANNELS,
    SpecValidationError,
    StreamSpec,
    TweakPatch,
    apply_tweak,
    validate_spec,
)

_GENESIS_HASH = "0" * 64


@dataclass(frozen=True)
class LedgerEntry:
    seq: int
    timestamp_ns: int
    event_type: str
    agent_id: int | None
    payload_json: str
    prev_hash: str
    entry_hash: str


class AuditLedger:
    """Tamper-evident SHA-256 hash-chained lifecycle ledger."""

    def __init__(self) -> None:
        self._entries: list[LedgerEntry] = []

    @staticmethod
    def _compute_hash(
        seq: int,
        timestamp_ns: int,
        event_type: str,
        agent_id: int | None,
        payload_json: str,
        prev_hash: str,
    ) -> str:
        canonical = json.dumps(
            {
                "seq": seq,
                "timestamp_ns": timestamp_ns,
                "event_type": event_type,
                "agent_id": agent_id,
                "payload_json": payload_json,
                "prev_hash": prev_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def append(
        self,
        event_type: str,
        agent_id: int | None,
        payload: Mapping[str, Any],
    ) -> LedgerEntry:
        seq = len(self._entries)
        ts = time.time_ns()
        prev_hash = self._entries[-1].entry_hash if self._entries else _GENESIS_HASH
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        entry_hash = self._compute_hash(seq, ts, event_type, agent_id, payload_json, prev_hash)
        entry = LedgerEntry(
            seq=seq,
            timestamp_ns=ts,
            event_type=event_type,
            agent_id=agent_id,
            payload_json=payload_json,
            prev_hash=prev_hash,
            entry_hash=entry_hash,
        )
        self._entries.append(entry)
        return entry

    @property
    def entries(self) -> tuple[LedgerEntry, ...]:
        """Return an immutable snapshot of ledger entries (never exposes internal list)."""
        return tuple(self._entries)

    @property
    def head_hash(self) -> str:
        return self._entries[-1].entry_hash if self._entries else _GENESIS_HASH

    def verify(self) -> bool:
        expected_prev = _GENESIS_HASH
        for idx, entry in enumerate(self._entries):
            if entry.seq != idx:
                return False
            if entry.prev_hash != expected_prev:
                return False
            recomputed = self._compute_hash(
                entry.seq,
                entry.timestamp_ns,
                entry.event_type,
                entry.agent_id,
                entry.payload_json,
                entry.prev_hash,
            )
            if entry.entry_hash != recomputed:
                return False
            expected_prev = entry.entry_hash
        return True


@dataclass
class AgentSubscription:
    """Handle representing one subscribed agent inside a MultiAgentStreamHub."""

    agent_id: int
    spec: StreamSpec
    group_id: int
    tombstoned: bool
    runtime: GapAsyncRuntime
    active_tweaks: list[dict[str, Any]]
    warm_started_at_step: int
    base_spec: StreamSpec | None = None
    last_result: StepResult | None = None


class MultiAgentStreamHub:
    """Shared-arena multi-agent causal streaming hub backed by C11 EpochArena."""

    def __init__(
        self,
        n_channels: int,
        *,
        window: int = 64,
        init_agent_cap: int = 4,
    ) -> None:
        # Initialize teardown-safe fields FIRST so failed __init__ never raises in __del__
        self._agents: dict[int, AgentSubscription] = {}
        self._arena: Any = None
        self._lib: Any = None
        self._closed: bool = True

        if (
            isinstance(n_channels, bool)
            or not isinstance(n_channels, int)
            or n_channels < 1
            or n_channels > DEFAULT_MAX_CHANNELS
        ):
            raise ValueError(
                f"n_channels must be an int in [1, {DEFAULT_MAX_CHANNELS}], got {n_channels!r}"
            )
        if isinstance(window, bool) or not isinstance(window, int) or window < 2:
            raise ValueError(f"window must be an int >= 2, got {window!r}")
        if (
            isinstance(init_agent_cap, bool)
            or not isinstance(init_agent_cap, int)
            or init_agent_cap < 1
            or init_agent_cap > DEFAULT_MAX_CHANNELS
        ):
            raise ValueError(
                f"init_agent_cap must be an int in [1, {DEFAULT_MAX_CHANNELS}], "
                f"got {init_agent_cap!r}"
            )

        self._lib = load_native_lib()
        self.n_channels = int(n_channels)
        self.window = int(window)
        self._arena = self._lib.tsjit_arena_create(
            self.n_channels, int(window), int(max(init_agent_cap, 2))
        )
        if not self._arena:
            raise RuntimeError(
                f"Failed to allocate C11 EpochArena(n_channels={n_channels}, window={window})"
            )

        self._closed = False
        self.ledger = AuditLedger()
        self.step_count: int = 0
        self.last_step: int = -1
        self.ledger.append(
            "hub_init",
            None,
            {"n_channels": self.n_channels, "window": self.window, "init_cap": init_agent_cap},
        )

    def close(self) -> None:
        self._closed = True
        agents = getattr(self, "_agents", None)
        if agents:
            for sub in list(agents.values()):
                sub.runtime.close()
            agents.clear()
        arena = getattr(self, "_arena", None)
        lib = getattr(self, "_lib", None)
        if arena and lib is not None:
            lib.tsjit_arena_destroy(arena)
            self._arena = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> MultiAgentStreamHub:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _require_open(self) -> None:
        if not self._arena:
            raise RuntimeError("MultiAgentStreamHub is closed")

    def _warm_start_runtime(self, sub: AgentSubscription) -> None:
        """O(1) zero-replay warm-start of an agent runtime from shared arena statistics."""
        if self.step_count <= 0:
            return
        ch0 = sub.spec.channels[0]
        stats = self.channel_stats(ch0)
        if stats["count"] > 0 and math.isfinite(stats["mean"]):
            mu = stats["mean"]
            sd = stats["std"]
            sc = (
                sd
                if (math.isfinite(sd) and sd > 0.0)
                else (
                    max(1e-4 * abs(stats["anchor"]), 1e-12)
                    if (math.isfinite(stats["anchor"]) and stats["anchor"] != 0.0)
                    else 1.0
                )
            )
            q_init = _safe_normal_q9(mu, sc)
            sub.runtime.engine.publish_forecast(0, self.last_step, q_init, q_init, scale_hint=sc)
            sub.runtime._has_forecast[0] = True

    def subscribe(
        self,
        spec: StreamSpec | Mapping[str, Any],
        *,
        model: Callable[[Sequence[float]], Any] | None = None,
        hot_add: bool = False,
    ) -> AgentSubscription:
        """Validate spec and subscribe an agent to the shared C11 EpochArena."""
        self._require_open()
        validated = validate_spec(spec, max_channels=self.n_channels)
        dom = (ctypes.c_int32 * len(validated.channels))(*validated.channels)
        out_id = ctypes.c_int32(-1)
        rc = int(
            self._lib.tsjit_arena_add_agent(
                self._arena,
                dom,
                len(validated.channels),
                float(validated.ewma_gamma),
                ctypes.byref(out_id),
            )
        )
        if rc != TSJIT_OK or out_id.value < 0:
            raise RuntimeError(f"tsjit_arena_add_agent failed (rc={rc})")

        aid = int(out_id.value)
        cov_info = self._read_agent_cov_raw(aid, len(validated.channels))
        rt = GapAsyncRuntime(model=model, spec=validated, n_keys=1, async_worker=False)
        sub = AgentSubscription(
            agent_id=aid,
            spec=validated,
            group_id=cov_info["group_id"],
            tombstoned=False,
            runtime=rt,
            active_tweaks=[],
            warm_started_at_step=self.last_step,
            base_spec=validated,
        )
        if hot_add or self.step_count > 0:
            self._warm_start_runtime(sub)

        self._agents[aid] = sub
        self.ledger.append(
            "hot_add" if hot_add else "subscribe",
            aid,
            {
                "name": validated.name,
                "channels": list(validated.channels),
                "group_id": sub.group_id,
                "warm_started_at_step": sub.warm_started_at_step,
            },
        )
        return sub

    def hot_add_agent(
        self,
        spec: StreamSpec | Mapping[str, Any],
        *,
        model: Callable[[Sequence[float]], Any] | None = None,
    ) -> AgentSubscription:
        """Zero-replay warm-start hot-addition of a new agent into a running hub."""
        return self.subscribe(spec, model=model, hot_add=True)

    def tombstone_agent(self, agent_id: int) -> None:
        """Permanently tombstone an agent while preserving its bit across future hot-adds."""
        self._require_open()
        if agent_id not in self._agents:
            raise KeyError(f"Unknown agent_id={agent_id}")
        if self._agents[agent_id].tombstoned:
            return
        rc = int(self._lib.tsjit_arena_tombstone_agent(self._arena, int(agent_id)))
        if rc != TSJIT_OK:
            raise RuntimeError(f"tsjit_arena_tombstone_agent failed (rc={rc})")
        self._agents[agent_id].tombstoned = True
        self.ledger.append("tombstone", agent_id, {"status": "TOMBSTONED"})

    unsubscribe = tombstone_agent

    def agent_status(self, agent_id: int) -> int:
        """Return native arena status bit (0=EMPTY, 1=ALIVE, 2=TOMBSTONED, -1=OOB)."""
        self._require_open()
        return int(self._lib.tsjit_arena_agent_status(self._arena, int(agent_id)))

    def _apply_spec_to_runtime(self, sub: AgentSubscription, new_spec: StreamSpec) -> None:
        """Reconfigure live runtime thresholds/modes and C11 engine params from a StreamSpec."""
        sub.spec = new_spec
        sub.runtime.spec = new_spec
        sub.runtime.engine.set_params(lambda_age=new_spec.lambda_age, max_age=new_spec.max_age)
        gate = sub.runtime._gate[0]
        gate.surprise_threshold = new_spec.surprise_threshold
        gate.evalue_threshold = new_spec.evalue_threshold
        gate.max_age = new_spec.max_age
        gate.gate_expr = new_spec.compiled_gate_expr()
        ev = sub.runtime._evalue[0]
        ev.threshold = new_spec.evalue_threshold
        aci = sub.runtime._aci[0]
        aci.alpha = new_spec.alpha
        aci.gamma = new_spec.aci_gamma

    def tweak(
        self,
        agent_id: int,
        patch: TweakPatch | Mapping[str, Any],
        *,
        _inject_failure: bool = False,
    ) -> StreamSpec:
        """Transactionally tweak an agent's live spec with atomic rollback on error."""
        self._require_open()
        if agent_id not in self._agents:
            raise KeyError(f"Unknown agent_id={agent_id}")
        sub = self._agents[agent_id]
        if sub.tombstoned:
            raise SpecValidationError(f"Cannot tweak tombstoned agent_id={agent_id}")

        old_spec = sub.spec
        old_tweaks = list(sub.active_tweaks)
        patch_dict: dict[str, Any] | None = None

        try:
            if isinstance(patch, TweakPatch):
                patch_dict = dict(patch.updates)
            elif isinstance(patch, Mapping):
                patch_dict = dict(patch)
            else:
                raise SpecValidationError(
                    f"Expected TweakPatch or mapping, got {type(patch).__name__}"
                )
            new_spec = apply_tweak(old_spec, patch, max_channels=self.n_channels)
            self._apply_spec_to_runtime(sub, new_spec)
            if _inject_failure:
                raise RuntimeError("Simulated transactional failure during tweak commit")
            sub.active_tweaks.append(patch_dict)
            self.ledger.append(
                "tweak_commit",
                agent_id,
                {"patch": patch_dict, "active_tweak_count": len(sub.active_tweaks)},
            )
            return new_spec
        except Exception as exc:
            # Atomic rollback to exact prior spec & active tweak history
            self._apply_spec_to_runtime(sub, old_spec)
            sub.active_tweaks = old_tweaks
            logged_patch = (
                {str(k): repr(v) for k, v in patch_dict.items()}
                if patch_dict is not None
                else {"raw": repr(patch)}
            )
            self.ledger.append(
                "tweak_reject",
                agent_id,
                {"patch": logged_patch, "error": str(exc)},
            )
            raise

    def replay_tweaks(self, agent_id: int) -> StreamSpec:
        """Replay all committed active tweaks onto an agent's base spec."""
        self._require_open()
        if agent_id not in self._agents:
            raise KeyError(f"Unknown agent_id={agent_id}")
        sub = self._agents[agent_id]
        if sub.tombstoned:
            raise SpecValidationError(f"Cannot replay tweaks on tombstoned agent_id={agent_id}")
        replayed = sub.base_spec if sub.base_spec is not None else sub.spec
        for p in sub.active_tweaks:
            replayed = apply_tweak(replayed, p, max_channels=self.n_channels)
        self._apply_spec_to_runtime(sub, replayed)
        return replayed

    def ingest(self, step: int, values: Sequence[float]) -> dict[int, StepResult]:
        """Single-pass multivariate ingest updating shared C11 arena and all live agents."""
        self._require_open()
        if len(values) != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channel values, got {len(values)}")

        c_vals = (ctypes.c_double * self.n_channels)(*[float(v) for v in values])
        rc = int(self._lib.tsjit_arena_ingest_tick(self._arena, int(step), c_vals, self.n_channels))
        is_dup_or_ooo = rc in (MARK_DUP_STEP, MARK_OOO_STEP)
        if rc != TSJIT_OK and not is_dup_or_ooo:
            raise RuntimeError(f"tsjit_arena_ingest_tick failed (rc={rc})")

        results: dict[int, StepResult] = {}
        if is_dup_or_ooo:
            # Strict causal rejection: DO NOT step ANY agent runtime (including newly hot-added
            # agents!) so DUP/OOO ticks perform ZERO state mutation across the entire hub.
            mark_name = MARK_NAMES.get(rc, f"MARK_{rc}")
            reason = f"rejected_{mark_name.lower()}"
            for aid, sub in self._agents.items():
                if sub.tombstoned:
                    continue
                prev = sub.last_result
                if prev is not None:
                    results[aid] = StepResult(
                        step=int(step),
                        key=prev.key,
                        q9=prev.q9,
                        cal_lo=prev.cal_lo,
                        cal_hi=prev.cal_hi,
                        abs_sg=prev.abs_sg,
                        rolling_mean=prev.rolling_mean,
                        rolling_std=prev.rolling_std,
                        eff_scale=prev.eff_scale,
                        age=prev.age,
                        mark=rc,
                        mark_name=mark_name,
                        pwl_path=prev.pwl_path,
                        family_weights=prev.family_weights,
                        e_value=prev.e_value,
                        e_ville=prev.e_ville,
                        e_cusum=prev.e_cusum,
                        e_sr=prev.e_sr,
                        alarm=False,
                        dispatched=False,
                        trigger_reason=reason,
                    )
                else:
                    ev = sub.runtime._evalue[0]
                    st = sub.runtime.engine.get_rolling(0)
                    results[aid] = StepResult(
                        step=int(step),
                        key=0,
                        q9=(0.0,) * 9,
                        cal_lo=0.0,
                        cal_hi=0.0,
                        abs_sg=0.0,
                        rolling_mean=st["mean"],
                        rolling_std=st["std"],
                        eff_scale=1.0,
                        age=0,
                        mark=rc,
                        mark_name=mark_name,
                        pwl_path=2,
                        family_weights=(0.0, 0.0, 1.0),
                        e_value=ev.last_e,
                        e_ville=ev.ville,
                        e_cusum=ev.cusum,
                        e_sr=ev.shiryaev_roberts,
                        alarm=False,
                        dispatched=False,
                        trigger_reason=reason,
                    )
            return results

        self.step_count += 1
        self.last_step = int(step)

        c_q9 = ctypes.c_double * 9
        for aid, sub in self._agents.items():
            if sub.tombstoned:
                continue
            # Strictly read ONLY channels in this agent's domain (0.0 cross-channel NaN bleed!)
            primary_ch = sub.spec.channels[0]
            x_primary = float(values[primary_ch])
            res = sub.runtime.step(step, x_primary, key=0)
            results[aid] = res
            if res.mark not in (MARK_DUP_STEP, MARK_OOO_STEP):
                sub.last_result = res
                self._lib.tsjit_arena_publish_agent_output(
                    self._arena,
                    int(aid),
                    int(step),
                    c_q9(*res.q9),
                    float(res.abs_sg),
                    int(res.mark),
                )
        return results

    step = ingest

    def read_agent_snapshot(self, agent_id: int) -> dict[str, Any]:
        """Lock-free EBR-protected snapshot read from native EpochArena."""
        self._require_open()
        q_out = (ctypes.c_double * 9)()
        abs_sg = ctypes.c_double()
        mark = ctypes.c_int32()
        step_v = ctypes.c_int64()
        ver = ctypes.c_uint64()
        rc = int(
            self._lib.tsjit_arena_read_agent_snapshot(
                self._arena,
                int(agent_id),
                q_out,
                ctypes.byref(abs_sg),
                ctypes.byref(mark),
                ctypes.byref(step_v),
                ctypes.byref(ver),
            )
        )
        if rc != TSJIT_OK:
            raise RuntimeError(f"tsjit_arena_read_agent_snapshot failed (rc={rc})")
        return {
            "agent_id": agent_id,
            "q9": tuple(float(q_out[i]) for i in range(9)),
            "abs_sg": float(abs_sg.value),
            "mark": int(mark.value),
            "step": int(step_v.value),
            "version": int(ver.value),
        }

    def channel_stats(self, channel: int) -> dict[str, Any]:
        self._require_open()
        mean = ctypes.c_double()
        std = ctypes.c_double()
        anchor = ctypes.c_double()
        cnt = ctypes.c_uint64()
        rc = int(
            self._lib.tsjit_arena_get_channel_stats(
                self._arena,
                int(channel),
                ctypes.byref(mean),
                ctypes.byref(std),
                ctypes.byref(anchor),
                ctypes.byref(cnt),
            )
        )
        if rc != TSJIT_OK:
            raise IndexError(f"channel={channel} out of bounds (rc={rc})")
        return {
            "mean": float(mean.value),
            "std": float(std.value),
            "anchor": float(anchor.value),
            "count": int(cnt.value),
        }

    def _read_agent_cov_raw(self, agent_id: int, dim: int) -> dict[str, Any]:
        mean_arr = (ctypes.c_double * dim)()
        cov_arr = (ctypes.c_double * (dim * dim))()
        n_val = ctypes.c_uint64()
        gid = ctypes.c_int32()
        rc = int(
            self._lib.tsjit_arena_get_agent_cov(
                self._arena,
                int(agent_id),
                int(dim),
                mean_arr,
                cov_arr,
                ctypes.byref(n_val),
                ctypes.byref(gid),
            )
        )
        if rc != TSJIT_OK:
            raise RuntimeError(f"tsjit_arena_get_agent_cov failed (rc={rc})")
        return {
            "group_id": int(gid.value),
            "n": int(n_val.value),
            "mean": [float(mean_arr[i]) for i in range(dim)],
            "cov": [[float(cov_arr[i * dim + j]) for j in range(dim)] for i in range(dim)],
        }

    def agent_covariance(self, agent_id: int) -> dict[str, Any]:
        self._require_open()
        if agent_id not in self._agents:
            raise KeyError(f"Unknown agent_id={agent_id}")
        spec_ch = self._agents[agent_id].spec.channels
        res = self._read_agent_cov_raw(agent_id, len(spec_ch))
        res["channels"] = sorted(spec_ch)
        return res

    def swarm_e_value(self) -> float:
        """Exact bit-identical commutative mean of active agents' latest e-values."""
        vals = [
            sub.last_result.e_value
            for sub in self._agents.values()
            if (not sub.tombstoned and sub.last_result is not None)
        ]
        return QuantilePitEValueDetector.swarm_mean(vals)

    def force_resize(self, new_capacity: int) -> dict[str, Any]:
        self._require_open()
        rc = int(self._lib.tsjit_arena_force_resize(self._arena, int(new_capacity)))
        if rc != TSJIT_OK:
            raise RuntimeError(f"tsjit_arena_force_resize({new_capacity}) failed (rc={rc})")
        self.ledger.append("resize", None, {"new_capacity": int(new_capacity)})
        return self.ebr_stats()

    def ebr_stats(self) -> dict[str, Any]:
        self._require_open()
        cap = ctypes.c_uint32()
        resizes = ctypes.c_uint64()
        reclaimed = ctypes.c_uint64()
        pending = ctypes.c_uint64()
        rc = int(
            self._lib.tsjit_arena_ebr_stats(
                self._arena,
                ctypes.byref(cap),
                ctypes.byref(resizes),
                ctypes.byref(reclaimed),
                ctypes.byref(pending),
            )
        )
        if rc != TSJIT_OK:
            raise RuntimeError(f"tsjit_arena_ebr_stats failed (rc={rc})")
        return {
            "capacity": int(cap.value),
            "resizes": int(resizes.value),
            "reclaimed": int(reclaimed.value),
            "pending": int(pending.value),
        }

    def verify_ledger(self) -> bool:
        return self.ledger.verify()
