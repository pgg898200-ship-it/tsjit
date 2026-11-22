"""Comprehensive runtime, calibration, MultiAgentStreamHub, CLI, MCP, and clean-room tests."""

from __future__ import annotations

import io
import json
import math
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

import tsjit
from tsjit.mcp_server import handle_request, serve
from tsjit.runtime.arrival import GapAsyncRuntime, assert_causal, stream
from tsjit.runtime.calibration import EventGate, OnlineACI, QuantilePitEValueDetector
from tsjit.runtime.hub import LedgerEntry, MultiAgentStreamHub
from tsjit.runtime.spec import (
    SpecValidationError,
    StreamSpec,
    TweakPatch,
    validate_spec,
    validate_tweak,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_lazy_imports_and_cold_cli_startup_under_25ms() -> None:
    """Verify PEP 562 lazy import keeps torch/numpy unloaded and startup < 25 ms."""
    env = dict(os.environ)
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    # Confirm zero eager numpy/torch/ctypes import on `import tsjit`
    probe_code = (
        "import sys, tsjit; "
        "print(','.join(m for m in ('numpy','torch','triton','ctypes') if m in sys.modules))"
    )
    probe = subprocess.run(
        [sys.executable, "-c", probe_code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert probe.stdout.strip() == "", f"Unexpected eager imports: {probe.stdout.strip()}"

    def median_ms(cmd: list[str]) -> float:
        # Warm OS page cache
        subprocess.run(
            cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        samples = []
        for _ in range(5):
            t0 = time.perf_counter()
            subprocess.run(
                cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
            )
            samples.append((time.perf_counter() - t0) * 1000.0)
        return statistics.median(samples)

    import_ms = median_ms([sys.executable, "-c", "import tsjit"])
    help_ms = median_ms([sys.executable, "-m", "tsjit", "--help"])
    assert import_ms < 25.0, f"import tsjit took {import_ms:.2f} ms (>= 25 ms)"
    assert help_ms < 25.0, f"python3 -m tsjit --help took {help_ms:.2f} ms (>= 25 ms)"


def test_assert_causal_bit_exact_pwl_and_soft_v2() -> None:
    rng = random.Random(31415)
    series = [math.sin(0.15 * t) + 0.1 * rng.gauss(0.0, 1.0) for t in range(80)]

    for mode in ("pwl", "soft_v2", "hybrid"):
        rep = assert_causal(
            lambda m=mode: stream(spec=StreamSpec(name=f"causal_{m}", window=16, revision_mode=m)),
            series,
            split_step=40,
            poison_values=(1e12, -1e12, float("nan")),
        )
        assert rep["causal"] is True
        assert rep["max_diff"] == 0.0


def test_gap_async_worker_concurrent_seqlock_execution() -> None:
    calls = 0

    def slow_mock_model(ctx):
        nonlocal calls
        calls += 1
        time.sleep(0.001)
        c = ctx[-1] if ctx else 0.0
        return [c + z for z in (-1.28, -0.84, -0.52, -0.25, 0.0, 0.25, 0.52, 0.84, 1.28)]

    with GapAsyncRuntime(
        model=slow_mock_model,
        spec=StreamSpec(name="async_test", max_age=5, surprise_threshold=1.5),
        async_worker=True,
    ) as rt:
        for t in range(40):
            res = rt.step(t, math.sin(0.2 * t) + (3.0 if t == 20 else 0.0))
            assert all(math.isfinite(v) for v in res.q9)
            assert res.cal_lo <= res.cal_hi
        rt.flush_async()
        assert rt.async_publishes >= 1
        assert calls >= 1


def test_online_aci_delay_invert_guard_micro_scale_and_alarm_boost() -> None:
    # 1. Delay queue: feedback arrives only after step + delay
    aci = OnlineACI(alpha=0.1, gamma=0.1, delay=3, window=64)
    iv0 = aci.predict_interval(0, -1.0, 1.0)
    assert iv0.lo <= iv0.hi
    assert aci.update(1, 5.0) == 0  # Due at step 3, not yet resolved!
    assert aci.update(2, 5.0) == 0
    assert aci.update(3, 5.0) == 1  # Now resolved!
    assert aci.n_updates == 1
    assert aci.alpha_t < 0.1  # Miss tightened alpha_t

    # 2. Invert-guard forces cal_lo == cal_hi == mid when negative qt_cqr inverts interval
    aci_inv = OnlineACI(alpha=0.2, gamma=0.05, delay=0)
    for t in range(25):
        aci_inv.predict_interval(t, -10.0, 10.0)
        aci_inv.update(t, 0.0)  # Deeply interior -> strongly negative CQR scores (-0.5)
    iv_inv = aci_inv.predict_interval(26, -1.0, 1.0, scale_hint=20.0)
    assert iv_inv.lo <= iv_inv.hi
    assert iv_inv.inverted_guarded is True
    assert aci_inv.inversions_guarded >= 1

    # 3. 1e-6 micro-scale preservation
    sc_micro = OnlineACI.eff_sc(1e-6 - 1e-7, 1e-6 + 1e-7, anchor=1e-6)
    assert abs(sc_micro - 2e-7) < 1e-12

    # 4. boost_on_alarm uses recent conformal scores to widen intervals immediately
    aci_b = OnlineACI(alpha=0.1, gamma=0.05, boost_factor=2.0)
    for t in range(20):
        aci_b.predict_interval(t, -1.0, 1.0)
        aci_b.update(t, 1.4 if t % 4 == 0 else 0.2)
    before_iv = aci_b.predict_interval(21, -1.0, 1.0, enqueue=False)
    aci_b.boost_on_alarm(severity=1.5)
    after_iv = aci_b.predict_interval(22, -1.0, 1.0, enqueue=False)
    assert (after_iv.hi - after_iv.lo) > (before_iv.hi - before_iv.lo)


def test_quantile_pit_evalue_and_commutative_swarm_mean() -> None:
    q_null = [-1.2816, -0.8416, -0.5244, -0.2533, 0.0, 0.2533, 0.5244, 0.8416, 1.2816]
    det = QuantilePitEValueDetector(kappa=0.5, threshold=20.0)

    # Under null N(0, 1), PITs stay near uniform and detectors stay below alarm threshold
    rng = random.Random(2025)
    for _ in range(50):
        det.update(rng.gauss(0.0, 1.0), q_null)
    assert det.cusum < 20.0
    ville_before = det.ville

    # Under abrupt +4 sigma regime shift, Page-CUSUM and Shiryaev-Roberts fire rapidly,
    # and the unrestarted Ville test martingale surges and crosses threshold under sustained shift
    alarmed = False
    for _ in range(15):
        u = det.update(4.0 + rng.gauss(0.0, 0.3), q_null)
        if u.alarm:
            alarmed = True
            break
    assert alarmed is True
    assert det.cusum >= 20.0
    assert det.shiryaev_roberts >= 20.0
    assert det.ville > 10.0 * ville_before
    for _ in range(15):
        det.update(4.0 + rng.gauss(0.0, 0.3), q_null)
    assert det.ville >= 20.0

    # Exact bit-identical commutative swarm_mean across random permutations
    vals = [1e-12, 1e12, 0.3333333333333333, 17.25, 0.000123456789, 42.1, 1e-8]
    ref = QuantilePitEValueDetector.swarm_mean(vals)
    for seed in range(25):
        perm = list(vals)
        random.Random(seed).shuffle(perm)
        assert QuantilePitEValueDetector.swarm_mean(perm) == ref


def test_event_gate_lazy_prep_and_custom_expr() -> None:
    gate = EventGate(
        surprise_threshold=2.5,
        evalue_threshold=20.0,
        max_age=10,
        gate_expr="(abs_sg >= surprise_threshold) or (e_cusum >= 15.0 and age >= 2)",
    )
    gate.ingest_lazy(0, 1.0)
    d0 = gate.evaluate(abs_sg=0.4, e_value=1.0, age=1, e_cusum=1.0)
    assert d0.dispatch is False
    d1 = gate.evaluate(abs_sg=3.0, e_value=1.0, age=1, e_cusum=1.0)
    assert d1.dispatch is True
    d2 = gate.evaluate(abs_sg=0.2, e_value=1.0, age=3, e_cusum=18.0)
    assert d2.dispatch is True
    assert gate.deferred_steps == 1 and gate.dispatched_steps == 2


def test_zero_bypass_spec_and_tweak_adversarial_rejections() -> None:
    # Valid spec passes
    s = validate_spec(
        {
            "name": "prod_agent_1",
            "channels": [0, 2],
            "window": 32,
            "gate_expr": "abs_sg >= 2.0 or (age >= max_age and not e_value < 1.5)",
        }
    )
    assert s.channels == (0, 2)

    bad_specs = [
        {"strict_causality": False},
        {"lookahead": 1},
        {"lookahead": -1},
        {"surprise_threshold": float("nan")},
        {"surprise_threshold": float("inf")},
        {"surprise_threshold": -1.0},
        {"evalue_threshold": 0.5},
        {"ewma_gamma": 0.0},
        {"ewma_gamma": 1.01},
        {"window": True},  # bool masquerading as int!
        {"horizon": False},
        {"channels": []},
        {"channels": [0, 0]},  # duplicate
        {"channels": [-1]},
        {"channels": [65536]},  # OOB (>= DEFAULT_MAX_CHANNELS=65536)
        {"unknown_injected_field": 123},
        {"revision_mode": "unsafe_mode"},
        {"name": "1bad_leading_digit"},
        {"name": "bad;rm -rf"},
        {"gate_expr": "__import__('os').system('id')"},
        {"gate_expr": "abs_sg >= 1.0; open('/etc/passwd')"},
        {"gate_expr": "eval('1+1') > 0"},
        {"gate_expr": "abs_sg.__class__ == 1"},
        {"gate_expr": "unknown_var > 1.0"},
        {"gate_expr": "abs_sg >= nan"},
        {"gate_expr": "abs_sg = 1.0"},
    ]
    for bad in bad_specs:
        with pytest.raises(SpecValidationError):
            validate_spec(bad)

    # Explicit smaller max_channels still enforces OOB (5000 rejected at 4096, allowed at 65536)
    with pytest.raises(SpecValidationError):
        validate_spec({"channels": [5000]}, max_channels=4096)
    assert validate_spec({"channels": [5000]}).channels == (5000,)

    # Disallowed tweak fields rejected
    for bad_patch in (
        {},
        {"strict_causality": False},
        {"lookahead": 2},
        {"channels": (0, 1)},
        {"unknown_key": 1},
        {"surprise_threshold": float("nan")},
    ):
        with pytest.raises(SpecValidationError):
            validate_tweak(bad_patch)


def test_multi_agent_hub_nan_isolation_hot_add_tombstone_tweak_and_ledger() -> None:
    # 1. Domain-partitioned NaN isolation: Ch 2 = 100% NaN causes 0.0 diff on Ch {0,1} agents
    with MultiAgentStreamHub(3, window=16) as h_clean, MultiAgentStreamHub(3, window=16) as h_nan:
        a_c = h_clean.subscribe(StreamSpec(name="a_01", channels=(0, 1)))
        a_n = h_nan.subscribe(StreamSpec(name="a_01", channels=(0, 1)))
        h_clean.subscribe(StreamSpec(name="b_012", channels=(0, 1, 2)))
        h_nan.subscribe(StreamSpec(name="b_012", channels=(0, 1, 2)))

        for t in range(30):
            v0 = 5.0 + math.sin(0.25 * t)
            v1 = -3.0 + math.cos(0.17 * t)
            rc = h_clean.ingest(t, [v0, v1, 100.0 + t])[a_c.agent_id]
            rn = h_nan.ingest(t, [v0, v1, float("nan")])[a_n.agent_id]
            assert rc.q9 == rn.q9
            assert rc.cal_lo == rn.cal_lo and rc.cal_hi == rn.cal_hi
            assert rc.abs_sg == rn.abs_sg and rc.e_value == rn.e_value

        assert (
            h_clean.agent_covariance(a_c.agent_id)["cov"]
            == h_nan.agent_covariance(a_n.agent_id)["cov"]
        )

    # 2. Zero-replay hot_add, tombstone preservation, transactional tweak rollback, SHA-256 ledger
    with MultiAgentStreamHub(4, window=16, init_agent_cap=2) as hub:
        a0 = hub.subscribe(StreamSpec(name="agent_0", channels=(0, 1)))
        a1 = hub.subscribe(StreamSpec(name="agent_1", channels=(0, 1)))
        assert a0.group_id == a1.group_id  # Shared covariance group!

        for t in range(15):
            hub.ingest(t, [10.0 + 0.1 * t, 20.0 - 0.1 * t, 3.0, 4.0])

        # Tombstone agent 1
        hub.tombstone_agent(a1.agent_id)
        assert hub.agent_status(a1.agent_id) == 2

        # Hot-add 10 more agents triggering multiple dynamic EBR snapshot resizes
        hot_ids = []
        for idx in range(10):
            sub = hub.hot_add_agent(StreamSpec(name=f"hot_{idx}", channels=(0, 1)))
            assert sub.warm_started_at_step == 14
            assert sub.group_id == a0.group_id
            hot_ids.append(sub.agent_id)

        # Tombstone bit for agent 1 MUST remain preserved (=2) across all hot-adds & resizes!
        assert hub.agent_status(a1.agent_id) == 2
        assert hub.agent_status(a0.agent_id) == 1

        # Valid transactional tweak commits cleanly
        updated = hub.tweak(a0.agent_id, {"surprise_threshold": 4.25, "max_age": 24})
        assert updated.surprise_threshold == 4.25 and updated.max_age == 24
        assert hub._agents[a0.agent_id].runtime._gate[0].surprise_threshold == 4.25

        # Invalid tweak rolls back atomically (state unchanged)
        with pytest.raises(SpecValidationError):
            hub.tweak(a0.agent_id, {"surprise_threshold": float("nan")})
        assert hub._agents[a0.agent_id].spec.surprise_threshold == 4.25
        assert len(hub._agents[a0.agent_id].active_tweaks) == 1

        # Simulated mid-transaction error also rolls back atomically!
        with pytest.raises(RuntimeError):
            hub.tweak(a0.agent_id, {"surprise_threshold": 9.0}, _inject_failure=True)
        assert hub._agents[a0.agent_id].spec.surprise_threshold == 4.25
        assert hub._agents[a0.agent_id].runtime._gate[0].surprise_threshold == 4.25
        assert len(hub._agents[a0.agent_id].active_tweaks) == 1

        # Step once more & verify lock-free snapshot read + swarm e-value + SHA-256 ledger
        hub.ingest(15, [11.5, 18.5, 3.1, 4.1])
        snap = hub.read_agent_snapshot(a0.agent_id)
        assert snap["step"] == 15 and len(snap["q9"]) == 9
        assert math.isfinite(hub.swarm_e_value())
        assert hub.verify_ledger() is True

        # Ledger entries property returns an immutable tuple snapshot (external mutation blocked),
        # and internal storage tampering is caught by SHA-256 chain verification!
        assert isinstance(hub.ledger.entries, tuple)
        orig = hub.ledger.entries[1]
        with pytest.raises(TypeError):
            hub.ledger.entries[1] = orig  # type: ignore[index]
        hub.ledger._entries[1] = LedgerEntry(
            seq=orig.seq,
            timestamp_ns=orig.timestamp_ns,
            event_type="tampered_event",
            agent_id=orig.agent_id,
            payload_json=orig.payload_json,
            prev_hash=orig.prev_hash,
            entry_hash=orig.entry_hash,
        )
        assert hub.verify_ledger() is False


def test_cli_subcommands_and_mcp_jsonrpc_stdio() -> None:
    env = dict(os.environ)
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    # CLI info & verify JSON output
    r_info = subprocess.run(
        [sys.executable, "-m", "tsjit", "info", "--json"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    info_json = json.loads(r_info.stdout)
    assert info_json["version"] == tsjit.__version__
    assert info_json["native"]["available"] is True

    r_verify = subprocess.run(
        [sys.executable, "-m", "tsjit", "verify", "--json"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    ver_json = json.loads(r_verify.stdout)
    assert ver_json["ok"] is True

    # In-process MCP JSON-RPC 2.0 tests (initialize, notification, tools/list, all 6 tools/call)
    init_resp = handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert init_resp is not None and init_resp["result"]["serverInfo"]["name"] == "tsjit-mcp"
    assert handle_request({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None

    list_resp = handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert list_resp is not None
    tool_names = {t["name"] for t in list_resp["result"]["tools"]}
    assert tool_names == {
        "tsjit_info",
        "tsjit_verify",
        "tsjit_benchmark",
        "tsjit_validate_spec",
        "tsjit_stream_sim",
        "tsjit_hub_demo",
    }

    # Test tools/call for validate_spec, stream_sim, hub_demo, info, verify, benchmark
    c_val_ok = handle_request(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "tsjit_validate_spec",
                "arguments": {"spec": {"name": "ok_spec", "channels": [0]}},
            },
        }
    )
    assert c_val_ok["result"]["isError"] is False
    assert json.loads(c_val_ok["result"]["content"][0]["text"])["valid"] is True

    c_val_bad = handle_request(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "tsjit_validate_spec", "arguments": {"spec": {"lookahead": 3}}},
        }
    )
    assert c_val_bad["result"]["isError"] is True
    assert json.loads(c_val_bad["result"]["content"][0]["text"])["valid"] is False

    c_sim = handle_request(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "tsjit_stream_sim",
                "arguments": {"values": [1.0, 1.1, 1.2, 5.0, 1.3]},
            },
        }
    )
    assert c_sim["result"]["isError"] is False
    assert json.loads(c_sim["result"]["content"][0]["text"])["steps"] == 5

    c_hub = handle_request(
        {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {"name": "tsjit_hub_demo", "arguments": {"n_steps": 20}},
        }
    )
    assert c_hub["result"]["isError"] is False
    hub_data = json.loads(c_hub["result"]["content"][0]["text"])
    assert hub_data["ledger_verified"] is True
    assert hub_data["agents"]["shared_group_clean_and_hot"] is True

    # Real subprocess stdio pipe test via `python3 -m tsjit mcp` (newline-delimited JSON)
    req_lines = (
        "\n".join(
            json.dumps(m)
            for m in (
                {"jsonrpc": "2.0", "id": 10, "method": "initialize", "params": {}},
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 11, "method": "tools/list"},
                {
                    "jsonrpc": "2.0",
                    "id": 12,
                    "method": "tools/call",
                    "params": {"name": "tsjit_info", "arguments": {}},
                },
            )
        )
        + "\n"
    )
    proc = subprocess.run(
        [sys.executable, "-m", "tsjit", "mcp"],
        input=req_lines,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    out_msgs = [json.loads(line) for line in proc.stdout.strip().splitlines()]
    assert [m["id"] for m in out_msgs] == [10, 11, 12]
    assert len(out_msgs[1]["result"]["tools"]) == 6

    # Content-Length framing test in-process
    body = json.dumps({"jsonrpc": "2.0", "id": 99, "method": "ping"}).encode("utf-8")
    in_buf = io.BytesIO(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body)
    out_buf = io.BytesIO()
    assert serve(in_buf, out_buf) == 0
    assert b"Content-Length:" in out_buf.getvalue()
    assert b'"id":99' in out_buf.getvalue()

    # Content-Type before Content-Length & malformed Content-Length must NOT crash server
    ct_first = (
        f"Content-Type: application/vscode-jsonrpc; charset=utf-8\r\n"
        f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body
    )
    out_ct = io.BytesIO()
    assert serve(io.BytesIO(ct_first), out_ct) == 0
    assert b'"id":99' in out_ct.getvalue() and b"-32700" not in out_ct.getvalue()

    out_bad = io.BytesIO()
    assert serve(io.BytesIO(b"Content-Length: notanumber\r\n\r\n"), out_bad) == 0
    assert b"-32700" in out_bad.getvalue()

    # MCP schema bounds enforcement (bool values, out-of-range steps)
    bad_sim = handle_request(
        {
            "jsonrpc": "2.0",
            "id": 100,
            "method": "tools/call",
            "params": {"name": "tsjit_stream_sim", "arguments": {"values": [True, 1.0]}},
        }
    )
    assert bad_sim is not None and bad_sim["result"]["isError"] is True

    bad_bench = handle_request(
        {
            "jsonrpc": "2.0",
            "id": 101,
            "method": "tools/call",
            "params": {"name": "tsjit_benchmark", "arguments": {"steps": 1}},
        }
    )
    assert bad_bench is not None and bad_bench["result"]["isError"] is True

    # CLI subcommand --help & invalid flag handling
    r_vhelp = subprocess.run(
        [sys.executable, "-m", "tsjit", "verify", "--help"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "usage: tsjit verify" in r_vhelp.stdout

    r_bad_steps = subprocess.run(
        [sys.executable, "-m", "tsjit", "bench", "--steps", "bad"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert r_bad_steps.returncode == 2 and "invalid --steps" in r_bad_steps.stderr


def test_runtime_regression_immutability_horizon_hybrid_and_hub_causality() -> None:
    """Regression tests for spec/tweak immutability, deep parens, zero-score ACI boost,
    CompiledTSFM (1,9,H) coercion, multi-horizon EValue alignment, hybrid mode, and hub DUP/OOO.
    """
    # 1. StreamSpec normalizes mutable list to immutable tuple; TweakPatch validates & freezes
    ch = [0, 1]
    s = StreamSpec(channels=ch)  # type: ignore[arg-type]
    ch.append(-999)
    assert s.channels == (0, 1) and isinstance(s.channels, tuple)

    with pytest.raises(SpecValidationError):
        TweakPatch({"surprise_threshold": float("nan")})

    vt = validate_tweak({"surprise_threshold": 3.0})
    with pytest.raises(TypeError):
        vt.updates["strict_causality"] = False  # type: ignore[index]

    # 4 levels of parens allowed (<= _MAX_EXPR_DEPTH=16), 17 rejected
    validate_spec({"gate_expr": "((((abs_sg >= 2.0))))"})
    with pytest.raises(SpecValidationError):
        validate_spec({"gate_expr": "(" * 17 + "abs_sg >= 2.0" + ")" * 17})

    # 2. OnlineACI boost_on_alarm widens even when all recent conformal scores are 0.0
    aci = OnlineACI(alpha=0.1, gamma=0.05)
    for t in range(10):
        aci.predict_interval(t, 0.0, 2.0)
        aci.update(t, 2.0)
    before = aci.predict_interval(11, 0.0, 2.0, enqueue=False)
    aci.boost_on_alarm(severity=2.0)
    after = aci.predict_interval(12, 0.0, 2.0, enqueue=False)
    assert (after.hi - after.lo) > (before.hi - before.lo)

    # 3. CompiledTSFM-compatible (1, 9, H) object with predict_quantiles() & horizon alignment
    class MockCompiledTSFM:
        def predict_quantiles(self, x):
            # Return (B=1, Q=9, H=4) centered at +42.3 for horizon index 3
            return [[[42.0 + 0.5 * (q - 4) + 0.1 * h for h in range(4)] for q in range(9)]]

    with stream(MockCompiledTSFM(), spec=StreamSpec(name="h4", horizon=4, window=16)) as rt:
        for t in range(6):
            res = rt.step(t, 42.0 + 0.05 * t)
            assert rt._aci[0].n_updates == rt._evalue[0].n_steps
        # Forecast published from MockCompiledTSFM (horizon_idx=3 -> median 42.3) was applied!
        assert abs(res.q9[4] - 42.3) < 0.5

    # 4. Hybrid revision mode uses PWL on fresh forecasts and SoftRevisionV2 when aged
    with stream(spec=StreamSpec(name="hyb", revision_mode="hybrid", window=16, max_age=20)) as rt:
        paths = [rt.step(t, 0.05 * t).pwl_path for t in range(8)]
        assert 0 in paths and 3 in paths

    # 5. MultiAgentStreamHub DUP/OOO non-mutation + live tweak(lambda_age, max_age) propagation
    with MultiAgentStreamHub(2, window=16) as hub:
        sub = hub.subscribe(
            StreamSpec(
                name="a0", channels=(1, 0), revision_mode="soft_v2", lambda_age=0.01, max_age=100
            )
        )
        hub.ingest(0, [1.0, 2.0])
        hub.ingest(1, [1.5, 2.5])
        st_before = hub.channel_stats(0)
        cov_before = hub.agent_covariance(sub.agent_id)
        assert cov_before["channels"] == [0, 1]

        dup_res = hub.ingest(1, [999_999.0, 999_999.0])
        ooo_res = hub.ingest(0, [-999_999.0, -999_999.0])
        assert dup_res[sub.agent_id].mark == 4
        assert ooo_res[sub.agent_id].mark == 5
        assert hub.channel_stats(0) == st_before
        assert hub.agent_covariance(sub.agent_id) == cov_before
        assert hub.last_step == 1

        # Live tweak of lambda_age propagates to native C11 engine
        hub.tweak(sub.agent_id, {"lambda_age": 5.0, "max_age": 2})
        r_aged = hub.ingest(10, [1.5, 2.5])[sub.agent_id]
        assert r_aged.family_weights[0] < 1e-4

        # replay_tweaks reconstructs exact tweaked spec from base_spec
        replayed = hub.replay_tweaks(sub.agent_id)
        assert replayed.lambda_age == 5.0 and replayed.max_age == 2


def test_cleanroom_audit_gate_passes_and_catches_leaks() -> None:
    audit_script = REPO_ROOT / "scripts" / "cleanroom_audit.py"
    proc = subprocess.run(
        [sys.executable, str(audit_script), "--json"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"cleanroom_audit failed:\n{proc.stdout}\n{proc.stderr}"
    rep = json.loads(proc.stdout)
    assert rep["ok"] is True and rep["finding_count"] == 0

    # Synthetic violation detection check in isolated temp dir (constructed dynamically)
    with tempfile.TemporaryDirectory() as tmp:
        bad_file = Path(tmp) / "leak.txt"
        synth_ip = ".".join(("10", "20", "30", "40"))
        bad_file.write_text(f"host = {synth_ip}\n", encoding="utf-8")
        proc_bad = subprocess.run(
            [sys.executable, str(audit_script), tmp, "--json"],
            capture_output=True,
            text=True,
        )
        assert proc_bad.returncode == 1
        rep_bad = json.loads(proc_bad.stdout)
        assert rep_bad["ok"] is False and rep_bad["finding_count"] >= 1


def test_wave45_c9_c10_c11_i7_runtime_hub_and_calibration_regressions() -> None:
    """Permanent regression suite for Wave 4.5 C9, C10, C11, and I7.1-I7.6 findings."""
    import gc
    import threading

    from tsjit.runtime.arrival import (
        PwlArrivalEngine,
        SoftRevisionEngine,
        _coerce_q9,
        default_naive_model,
    )
    from tsjit.runtime.hub import AuditLedger
    from tsjit.runtime.spec import GateExpression

    # C9(a): DUP/OOO ingest MUST NOT step newly hot-added agents (zero state mutation!)
    with MultiAgentStreamHub(3, window=16) as hub:
        a0 = hub.subscribe(StreamSpec(name="base", channels=(0, 1)))
        for t in range(5):
            hub.ingest(t, [1.0 + t, 2.0 + t, 3.0])
        hot = hub.hot_add_agent(StreamSpec(name="hot_new", channels=(0, 1)))
        assert hot.runtime.engine.get_rolling(0)["count"] == 0
        assert hot.runtime.engine.get_rolling(0)["last_step"] == -1

        dup = hub.ingest(4, [999.0, 999.0, 999.0])
        ooo = hub.ingest(2, [-999.0, -999.0, -999.0])
        assert dup[a0.agent_id].mark == 4 and dup[hot.agent_id].mark == 4
        assert ooo[a0.agent_id].mark == 5 and ooo[hot.agent_id].mark == 5
        assert hot.runtime.engine.get_rolling(0)["count"] == 0
        assert hot.runtime.engine.get_rolling(0)["last_step"] == -1
        assert hot.runtime._aci[0].n_updates == 0

    # C9(b): Consistent max_channels=65536 across StreamSpec, validate_spec, Hub, and MCP
    assert StreamSpec.max_channels == 65536
    sp_high = StreamSpec(name="high_ch", channels=(5000, 65535))
    assert sp_high.channels == (5000, 65535)
    with MultiAgentStreamHub(5005, window=8) as hub_big:
        sub_big = hub_big.subscribe({"name": "ch5000", "channels": [5000]})
        assert sub_big.spec.channels == (5000,)
    mcp_val = handle_request(
        {
            "jsonrpc": "2.0",
            "id": 900,
            "method": "tools/call",
            "params": {
                "name": "tsjit_validate_spec",
                "arguments": {"spec": {"name": "hi", "channels": [5000, 65535]}},
            },
        }
    )
    assert mcp_val is not None and mcp_val["result"]["isError"] is False

    # C9(c): Failed __init__ MUST NEVER raise AttributeError inside __del__
    unraisable: list[object] = []
    orig_hook = sys.unraisablehook

    def _capture_unraisable(args: object) -> None:
        unraisable.append(args)

    sys.unraisablehook = _capture_unraisable  # type: ignore[assignment]
    try:
        with pytest.raises(ValueError):
            MultiAgentStreamHub(0)
        with pytest.raises(SpecValidationError):
            GapAsyncRuntime(spec={"lookahead": 2})
        with pytest.raises(ValueError):
            PwlArrivalEngine(n_keys=0)
        gc.collect()
    finally:
        sys.unraisablehook = orig_hook
    assert unraisable == [], f"Unexpected __del__ errors on failed __init__: {unraisable}"

    # C10: _coerce_q9 interpolates arbitrary Q >= 2 including amazon/chronos-2 (B, 21, H)
    # Test both H=16 and H=9 (must NOT misread H=9 as Q=9!)
    for h_len in (16, 9):
        # Chronos-2 21-quantile grid: values[q] = 100.0 + q at h=2
        tensor_21 = [[[100.0 + float(q) + 0.01 * h for h in range(h_len)] for q in range(21)]]
        q9 = _coerce_q9(tensor_21, fallback_center=0.0, fallback_scale=1.0, horizon_idx=2)
        # Exact decile indices 2, 4, 6, 8, 10, 12, 14, 16, 18 -> 102.02, 104.02, ..., 118.02
        expected = tuple(100.0 + 2.0 * k + 0.02 for k in range(1, 10))
        for got, exp in zip(q9, expected, strict=True):
            assert abs(got - exp) < 1e-9

    # Arbitrary Q=3 ([0.0, 0.5, 1.0] -> [10, 20, 30]) interpolates linearly onto [0.1..0.9]
    tensor_3 = [[[10.0], [20.0], [30.0]]]
    q9_3 = _coerce_q9(tensor_3, fallback_center=0.0, fallback_scale=1.0, horizon_idx=0)
    assert abs(q9_3[0] - 12.0) < 1e-9 and abs(q9_3[4] - 20.0) < 1e-9 and abs(q9_3[8] - 28.0) < 1e-9

    # Explicit dict["quantiles"] of shape (1, Q=3, H=9) MUST interpolate Q=3 (not read H=9 as Q!)
    q9_dict_3x9 = _coerce_q9(
        {"quantiles": [[[10.0] * 9, [20.0] * 9, [30.0] * 9]]},
        fallback_center=0.0,
        fallback_scale=1.0,
        horizon_idx=0,
    )
    assert abs(q9_dict_3x9[4] - 20.0) < 1e-9

    # GapAsyncRuntime rejects invalid fast_ridge_lags (0 or negative) & handles timeout=nan
    for bad_lags in (0, -2, True):
        with pytest.raises(ValueError):
            GapAsyncRuntime(fast_ridge_lags=bad_lags)  # type: ignore[arg-type]
    with GapAsyncRuntime(async_worker=True) as rt_nan_to:
        rt_nan_to.step(0, 1.0)
        assert isinstance(rt_nan_to.flush_async(timeout=float("nan")), bool)
        assert rt_nan_to.flush_async(timeout=2.0) is True

    # C11: Async worker exception recovery, genuine flush_async timeout, and >2s close() UAF safety
    attempt = 0

    def flaky_model(ctx):
        nonlocal attempt
        attempt += 1
        if attempt == 1:
            raise RuntimeError("transient model failure")
        c = ctx[-1] if ctx else 0.0
        return [c + z for z in (-1.28, -0.84, -0.52, -0.25, 0.0, 0.25, 0.52, 0.84, 1.28)]

    with GapAsyncRuntime(
        model=flaky_model,
        spec=StreamSpec(name="flaky", max_age=2, surprise_threshold=1.0),
        async_worker=True,
    ) as rt:
        for t in range(10):
            rt.step(t, float(t) * 2.0)
        assert rt.flush_async(timeout=2.0) is True
        assert rt.async_error_count >= 1
        assert len(rt.worker_errors) >= 1
        assert rt.async_publishes >= 1

    # Genuine flush_async(timeout=...) & deferred native teardown when worker outlives close()
    release_slow = threading.Event()
    entered_slow = threading.Event()

    def slow_blocking_model(ctx):
        entered_slow.set()
        release_slow.wait(timeout=4.0)
        c = ctx[-1] if ctx else 0.0
        return [c + z for z in (-1.28, -0.84, -0.52, -0.25, 0.0, 0.25, 0.52, 0.84, 1.28)]

    rt_slow = GapAsyncRuntime(
        model=slow_blocking_model,
        spec=StreamSpec(name="slow_uaf", max_age=2),
        async_worker=True,
    )
    rt_slow.step(0, 1.0)
    assert entered_slow.wait(timeout=1.0)
    t0 = time.monotonic()
    flushed = rt_slow.flush_async(timeout=0.05)
    elapsed = time.monotonic() - t0
    assert flushed is False and elapsed < 0.5
    # Close while worker is still blocked (>2s timeout in join) -> must defer teardown (ZERO UAF!)
    wt_ref = rt_slow._worker_thread
    rt_slow.close()
    assert rt_slow._native_destroyed is False  # Deferred while worker alive!
    release_slow.set()
    if wt_ref is not None:
        wt_ref.join(timeout=2.0)
    assert rt_slow._native_destroyed is True  # Destroyed cleanly on worker exit!

    # I7.6: Short losses array in step_soft and short quantiles in publish_forecast raise ValueError
    with SoftRevisionEngine(n_keys=1, window=16) as eng:
        q0 = [float(i) for i in range(9)]
        with pytest.raises(ValueError):
            eng.step_soft(0, 0, 1.0, slow_models=[q0, q0], slow_losses=[0.5])
        with pytest.raises(ValueError):
            eng.publish_forecast(0, 0, [1.0, 2.0, 3.0])

    # I7.1: +Inf surprise or e-value dispatches (True) in BOTH EventGate and GateExpression
    gate = EventGate(surprise_threshold=2.5, evalue_threshold=20.0, max_age=16)
    assert gate.evaluate(abs_sg=float("inf"), e_value=1.0, age=0).dispatch is True
    assert gate.evaluate(abs_sg=0.0, e_value=float("inf"), age=0).dispatch is True
    assert gate.evaluate(abs_sg=0.0, e_value=1.0, age=0, e_cusum=float("inf")).dispatch is True
    assert gate.evaluate(abs_sg=float("nan"), e_value=float("nan"), age=0).dispatch is False

    gexpr = GateExpression.parse("abs_sg >= surprise_threshold or e_cusum >= evalue_threshold")
    base_env = {
        "abs_sg": 0.0,
        "e_value": 1.0,
        "e_cusum": 1.0,
        "e_ville": 1.0,
        "e_sr": 0.0,
        "age": 0.0,
        "max_age": 16.0,
        "surprise_threshold": 2.5,
        "evalue_threshold": 20.0,
    }
    assert gexpr.evaluate({**base_env, "abs_sg": float("inf")}) is True
    assert gexpr.evaluate({**base_env, "e_cusum": float("inf")}) is True
    assert gexpr.evaluate({**base_env, "abs_sg": float("nan")}) is False

    # I7.2: Canonical JSON hashing prevents '|' collision; non-mapping tweak logged; replay guards
    h1 = AuditLedger._compute_hash(0, 123, "a|1", None, "{}", "0" * 64)
    h2 = AuditLedger._compute_hash(0, 123, "a", 1, "{}", "0" * 64)
    assert h1 != h2

    with MultiAgentStreamHub(2, window=16) as hub_l:
        a0 = hub_l.subscribe(StreamSpec(name="l0", channels=(0,)))
        before_n = len(hub_l.ledger.entries)
        with pytest.raises(SpecValidationError):
            hub_l.tweak(a0.agent_id, [1, 2, 3])  # type: ignore[arg-type]
        assert len(hub_l.ledger.entries) == before_n + 1
        assert hub_l.ledger.entries[-1].event_type == "tweak_reject"
        assert hub_l.verify_ledger() is True

        hub_l.tombstone_agent(a0.agent_id)
        with pytest.raises(SpecValidationError):
            hub_l.replay_tweaks(a0.agent_id)
    with pytest.raises(RuntimeError):
        hub_l.replay_tweaks(a0.agent_id)

    # I7.3: OnlineACI guards anchor=inf and 1e308 overflow cleanly
    assert OnlineACI.eff_sc(0.0, 0.0, scale_hint=0.01, anchor=float("inf")) == 0.01
    aci_ext = OnlineACI(alpha=0.1, gamma=0.1)
    iv1 = aci_ext.predict_interval(0, -1e308, 1e308, scale_hint=1e308, anchor=float("inf"))
    assert math.isfinite(iv1.lo) and math.isfinite(iv1.hi) and iv1.lo <= iv1.hi
    aci_ext.update(0, 1e308)
    aci_ext.boost_on_alarm(severity=10.0)
    iv2 = aci_ext.predict_interval(1, -1e308, 1e308, scale_hint=1e308, anchor=1e308)
    assert math.isfinite(iv2.lo) and math.isfinite(iv2.hi) and iv2.lo <= iv2.hi
    assert math.isfinite(aci_ext.qt_cqr)

    # I7.4: reset_cusum() clears sticky alarm while keeping lifetime ville_alarm accurate
    q_null = [-1.2816, -0.8416, -0.5244, -0.2533, 0.0, 0.2533, 0.5244, 0.8416, 1.2816]
    det = QuantilePitEValueDetector(kappa=0.5, threshold=20.0)
    for _ in range(15):
        u = det.update(5.0, q_null)
    assert u.cusum_alarm is True and u.ville_alarm is True and u.alarm is True
    det.reset_cusum()
    u_post = det.update(0.0, q_null)
    assert u_post.cusum_alarm is False
    assert u_post.alarm is False  # NOT sticky!
    assert u_post.ville_alarm is True  # Lifetime Ville preserved!
    assert (
        gate.evaluate(
            abs_sg=0.1,
            e_value=u_post.e_value,
            age=1,
            e_cusum=u_post.cusum,
            e_ville=u_post.ville,
        ).dispatch
        is False
    )

    # I7.4 extra: compute_pit & swarm_mean extreme 1e308 safety (zero NaN / OverflowError)
    pit_ext = QuantilePitEValueDetector.compute_pit(0.9e308, [-1e308] + [1e308] * 8)
    assert math.isfinite(pit_ext) and abs(pit_ext - 0.195) < 1e-6
    pit_tail = QuantilePitEValueDetector.compute_pit(-1.7e308, [-0.9e308] + [0.9e308] * 8)
    assert math.isfinite(pit_tail) and 0.0 < pit_tail < 0.09
    sm_ext = QuantilePitEValueDetector.swarm_mean([1e308, 1e308])
    assert math.isfinite(sm_ext) and abs(sm_ext - 1e308) / 1e308 < 1e-12

    # C12(e): default_naive_model handles [-1e200, 1e200], [-1e308, 1e308], & [-1.5e308, 1.5e308]
    for extreme_ctx in ([-1e200, 1e200], [-1e308, 1e308], [-1.5e308, 1.5e308]):
        q_nat = default_naive_model(extreme_ctx)
        assert len(q_nat) == 9 and all(math.isfinite(v) for v in q_nat)
        assert list(q_nat) == sorted(q_nat)


def test_wave45_c12_c14_c15_mcp_cli_and_packaging_regressions() -> None:
    """Permanent regression suite for Wave 4.5 C12 MCP hardening, CLI info/bounds, and C14/C15."""
    # C12(a): Framed parse error receives framed Content-Length response & continues cleanly
    valid_ping = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}).encode("utf-8")
    stream_a = (
        b"Content-Length: 5\r\n\r\n{bad!"
        + f"Content-Length: {len(valid_ping)}\r\n\r\n".encode("ascii")
        + valid_ping
    )
    out_a = io.BytesIO()
    assert serve(io.BytesIO(stream_a), out_a) == 0
    raw_a = out_a.getvalue()
    assert raw_a.startswith(b"Content-Length:")
    assert b"-32700" in raw_a and b'"id":1' in raw_a

    # C12(b): Oversized Content-Length (> 16 MiB) replies framed error AND aborts (no smuggling!)
    smuggled = b'{"jsonrpc":"2.0","id":8888,"method":"ping"}\n'
    stream_b = b"Content-Length: 99999999\r\n\r\n" + smuggled
    out_b = io.BytesIO()
    assert serve(io.BytesIO(stream_b), out_b) == 0
    raw_b = out_b.getvalue()
    assert raw_b.startswith(b"Content-Length:")
    assert raw_b.count(b"Content-Length:") == 1
    assert b"-32700" in raw_b
    assert b"8888" not in raw_b  # Smuggled payload MUST NOT be executed!

    # C12(b) extra: 5000-digit Content-Length (>4300-digit Python int limit) & duplicate CL
    # headers MUST both reply framed -32700 AND abort (zero request smuggling!)
    stream_b_digits = b"Content-Length: " + b"9" * 5000 + b"\r\n\r\n" + smuggled
    out_b_digits = io.BytesIO()
    assert serve(io.BytesIO(stream_b_digits), out_b_digits) == 0
    raw_b_digits = out_b_digits.getvalue()
    assert raw_b_digits.startswith(b"Content-Length:")
    assert b"-32700" in raw_b_digits and b"8888" not in raw_b_digits

    dup_cl_body = b'{"jsonrpc":"2.0","id":1,"method":"ping"}'
    stream_b_dup = (
        f"Content-Length: {len(dup_cl_body) + len(smuggled)}\r\n"
        f"Content-Length: {len(dup_cl_body)}\r\n\r\n"
    ).encode("ascii") + dup_cl_body + smuggled
    out_b_dup = io.BytesIO()
    assert serve(io.BytesIO(stream_b_dup), out_b_dup) == 0
    raw_b_dup = out_b_dup.getvalue()
    assert raw_b_dup.startswith(b"Content-Length:")
    assert b"-32700" in raw_b_dup and b"8888" not in raw_b_dup

    # Chunked partial pipe/socket reads (3 bytes per read()) must assemble framed messages cleanly
    class _ChunkedReader:
        def __init__(self, data: bytes) -> None:
            self._data = data
            self._pos = 0

        def readline(self, limit: int = -1) -> bytes:
            nl = self._data.find(b"\n", self._pos)
            end = len(self._data) if nl == -1 else nl + 1
            if limit >= 0:
                end = min(end, self._pos + limit)
            out = self._data[self._pos : end]
            self._pos = end
            return out

        def read(self, n: int = -1) -> bytes:
            if self._pos >= len(self._data):
                return b""
            take = 3 if n < 0 else min(n, 3)
            out = self._data[self._pos : self._pos + take]
            self._pos += len(out)
            return out

    framed_chunk_in = (
        f"Content-Length: {len(valid_ping)}\r\n\r\n".encode("ascii") + valid_ping
    )
    out_chunk = io.BytesIO()
    assert serve(_ChunkedReader(framed_chunk_in), out_chunk) == 0  # type: ignore[arg-type]
    assert out_chunk.getvalue().startswith(b"Content-Length:")
    assert b'"id":1' in out_chunk.getvalue()

    # C12(c): Arbitrary colon line does NOT enter framed mode
    stream_c = b'not_a_header:foo\n{"jsonrpc":"2.0","id":2,"method":"ping"}\n'
    out_c = io.BytesIO()
    assert serve(io.BytesIO(stream_c), out_c) == 0
    lines_c = [json.loads(ln) for ln in out_c.getvalue().decode("utf-8").strip().splitlines()]
    assert len(lines_c) == 2
    assert lines_c[0]["error"]["code"] == -32700
    assert lines_c[1]["id"] == 2

    # C12(d): tsjit_stream_sim with [-1e200, 1e200], [-1e308, 1e308], & [-1.5e308, 1.5e308, ...]
    for ext_vals in ([-1e200, 1e200], [-1e308, 1e308], [-1.5e308, 1.5e308, -1.5e308]):
        resp = handle_request(
            {
                "jsonrpc": "2.0",
                "id": 50,
                "method": "tools/call",
                "params": {"name": "tsjit_stream_sim", "arguments": {"values": ext_vals}},
            }
        )
        assert resp is not None and resp["result"]["isError"] is False
        # Strict RFC 8259 check: reject Infinity/NaN constants
        def _no_nan(tok: str) -> None:
            raise ValueError(f"Invalid JSON token: {tok}")

        txt = resp["result"]["content"][0]["text"]
        parsed = json.loads(txt, parse_constant=_no_nan)
        assert parsed["steps"] == len(ext_vals)
        assert all(math.isfinite(q) for q in parsed["last"]["q9"])

    # C12(f): Non-dict arguments ([]), missing method (-32600), missing/unknown params (-32602),
    # and JSON null (-32600 instead of EOF)
    r_arr_args = handle_request(
        {
            "jsonrpc": "2.0",
            "id": 60,
            "method": "tools/call",
            "params": {"name": "tsjit_info", "arguments": []},
        }
    )
    assert r_arr_args is not None and r_arr_args["error"]["code"] == -32602

    r_no_method = handle_request({"jsonrpc": "2.0", "id": 61})
    assert r_no_method is not None and r_no_method["error"]["code"] == -32600

    r_missing_req = handle_request(
        {
            "jsonrpc": "2.0",
            "id": 62,
            "method": "tools/call",
            "params": {"name": "tsjit_validate_spec", "arguments": {}},
        }
    )
    assert r_missing_req is not None and r_missing_req["error"]["code"] == -32602

    r_unknown_arg = handle_request(
        {
            "jsonrpc": "2.0",
            "id": 63,
            "method": "tools/call",
            "params": {"name": "tsjit_info", "arguments": {"unexpected": 1}},
        }
    )
    assert r_unknown_arg is not None and r_unknown_arg["error"]["code"] == -32602

    out_null = io.BytesIO()
    assert serve(io.BytesIO(b'null\n{"jsonrpc":"2.0","id":64,"method":"ping"}\n'), out_null) == 0
    lines_null = [json.loads(ln) for ln in out_null.getvalue().decode("utf-8").strip().splitlines()]
    assert lines_null[0]["error"]["code"] == -32600
    assert lines_null[1]["id"] == 64

    # Invalid JSON-RPC id types (bool/list) -> -32600; non-dict params on ping/tools/list -> -32602
    for bad_id in (True, [1], {"k": 1}):
        r_bad_id = handle_request({"jsonrpc": "2.0", "id": bad_id, "method": "ping"})
        assert r_bad_id is not None and r_bad_id["error"]["code"] == -32600
        assert r_bad_id["id"] is None
    for meth in ("ping", "tools/list"):
        r_bad_p = handle_request({"jsonrpc": "2.0", "id": 65, "method": meth, "params": [1, 2]})
        assert r_bad_p is not None and r_bad_p["error"]["code"] == -32602

    # CLI info dependency-gated pass_status & bench flag upper bounds
    from tsjit.cli import collect_info
    from tsjit.cli import main as cli_main

    info = collect_info()
    assert "pass_status" in info
    opt = info["optional_backends"]
    if not (opt["torch"] and opt["triton"] and opt["cuda_available"]):
        assert "cuda-scaled-fp16-triton" not in info["compiled_passes"]
        assert info["pass_status"]["cuda-scaled-fp16-triton"].startswith("unavailable")
    if not opt["torch"]:
        assert "tsfm-compiler-pass" not in info["compiled_passes"]
        assert info["pass_status"]["tsfm-compiler-pass"].startswith("unavailable")
    assert cli_main(["bench", "--steps", "999999999"]) == 2
    assert cli_main(["bench", "--agents", "99999"]) == 2

    # C14 & C15: .gitignore, README.md, and pyproject.toml metadata hygiene
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    for pat in (
        ".ruff_cache/",
        "__pycache__/",
        "*.pyc",
        ".pytest_cache/",
        "dist/",
        "build/",
        "*.egg-info/",
    ):
        assert pat in gitignore or (pat == "*.pyc" and "*.py[cod]" in gitignore)
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'license = { text = "Apache-2.0" }' in pyproject
    assert "license-files" not in pyproject
    assert 'pythonpath = ["src"]' in pyproject
    assert 'readme = "README.md"' in pyproject
    assert "[project.urls]" in pyproject
    assert '"toto-ts"' in pyproject and '"toto-2"' not in pyproject
    assert (REPO_ROOT / "README.md").is_file()
