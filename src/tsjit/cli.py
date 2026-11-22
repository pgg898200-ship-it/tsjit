"""Fast-startup (<25 ms) CLI for tsjit.

Keeps module-level imports strictly to `sys` and `_version` so `--help` and
`--version` never touch ctypes, json, numpy, or torch.
"""

from __future__ import annotations

import sys

from ._version import __version__

_HELP_TEXT = """usage: tsjit [-h] [-V] {info,verify,bench,mcp} ...

tsjit: Clean-room streaming time-series native runtime and compiler

commands:
  info      Show version, native C11 backend, SIMD ISA, and optional backends
  verify    Run native C11 build, seqlock contention, causality, and NaN checks
  bench     Run local C11 arrival, Schur, and multi-agent streaming benchmarks
  mcp       Start stdio JSON-RPC 2.0 Model Context Protocol (MCP) server

options:
  -h, --help     Show this help message and exit
  -V, --version  Show tsjit version and exit
"""


def collect_info() -> dict:
    """Gather runtime, native C11, SIMD ISA, and optional accelerator status."""
    import importlib.util
    import platform
    from pathlib import Path

    from .native.build import native_info

    nat = native_info()
    pkg_root = Path(__file__).resolve().parent

    def _pkg_available(name: str) -> bool:
        try:
            return importlib.util.find_spec(name) is not None
        except Exception:
            return False

    torch_avail = _pkg_available("torch")
    triton_avail = _pkg_available("triton")
    numpy_avail = _pkg_available("numpy")
    cuda_avail = False
    if torch_avail:
        try:
            import torch  # type: ignore

            cuda_avail = bool(torch.cuda.is_available())
        except Exception:
            cuda_avail = False

    pass_specs = (
        ("native/build.py", "c11-pwl-arrive", bool(nat.get("available")), "native C11 compiler"),
        (
            "native/build.py",
            "c11-schur-winsorized",
            bool(nat.get("available")),
            "native C11 compiler",
        ),
        (
            "native/build.py",
            "c11-epoch-arena-ebr",
            bool(nat.get("available")),
            "native C11 compiler",
        ),
        ("runtime/calibration.py", "online-aci-evalue-gate", True, "stdlib"),
        ("compiler/api.py", "tsfm-compiler-pass", torch_avail, "torch"),
        (
            "cuda/scaled_fp16_triton.py",
            "cuda-scaled-fp16-triton",
            torch_avail and triton_avail and cuda_avail,
            "torch+triton+cuda",
        ),
        (
            "distributed/async_fuse.py",
            "distributed-async-fuse",
            torch_avail and cuda_avail,
            "torch+cuda",
        ),
    )

    passes = []
    pass_status: dict[str, str] = {}
    for rel_path, label, deps_ok, dep_desc in pass_specs:
        file_ok = (pkg_root / rel_path).is_file()
        if file_ok and deps_ok:
            passes.append(label)
            pass_status[label] = "available"
        elif file_ok:
            pass_status[label] = f"unavailable (missing {dep_desc})"
        else:
            pass_status[label] = "unavailable (module absent)"

    return {
        "version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "native": nat,
        "optional_backends": {
            "numpy": numpy_avail,
            "torch": torch_avail,
            "triton": triton_avail,
            "cuda_available": cuda_avail,
        },
        "compiled_passes": passes,
        "pass_status": pass_status,
    }


def run_verification(*, quick: bool = True) -> dict:
    """Run C11 build, seqlock contention, causality, NaN-isolation, and Schur checks."""
    import ctypes
    import math
    import time

    from .native.build import TSJIT_OK, load_native_lib, native_info
    from .runtime.arrival import GapAsyncRuntime, assert_causal
    from .runtime.hub import MultiAgentStreamHub
    from .runtime.spec import StreamSpec

    t0 = time.perf_counter()
    checks: dict[str, dict] = {}
    all_ok = True

    # 1. Native C11 build & ABI
    nat = native_info()
    lib = load_native_lib()
    abi_ok = (
        nat["available"] and nat["pwl_hot_row_bytes"] <= 640 and nat["cacheline_bytes"] in (64, 128)
    )
    checks["native_build"] = {
        "ok": abi_ok,
        "simd_isa": nat["simd_isa"],
        "cacheline_bytes": nat["cacheline_bytes"],
        "pwl_hot_row_bytes": nat["pwl_hot_row_bytes"],
    }
    all_ok = all_ok and abi_ok

    # 2. Multi-writer/multi-reader seqlock contention check
    reads_per_reader = 250_000 if quick else 500_000
    tot_r = ctypes.c_uint64(0)
    torn_r = ctypes.c_uint64(0)
    to_r = ctypes.c_uint64(0)
    rc_seq = int(
        lib.tsjit_seqlock_stress_test(
            2,
            4,
            reads_per_reader,
            0,
            ctypes.byref(tot_r),
            ctypes.byref(torn_r),
            ctypes.byref(to_r),
        )
    )
    tot_u = ctypes.c_uint64(0)
    torn_u = ctypes.c_uint64(0)
    to_u = ctypes.c_uint64(0)
    rc_ung = int(
        lib.tsjit_seqlock_stress_test(
            2,
            4,
            reads_per_reader,
            1,
            ctypes.byref(tot_u),
            ctypes.byref(torn_u),
            ctypes.byref(to_u),
        )
    )
    seq_ok = (
        rc_seq == TSJIT_OK
        and rc_ung == TSJIT_OK
        and tot_r.value >= 1_000_000
        and torn_r.value == 0
        and to_r.value == 0
        and torn_u.value > 1_000
    )
    checks["seqlock_contention"] = {
        "ok": seq_ok,
        "seqlocked_reads": int(tot_r.value),
        "seqlocked_torn": int(torn_r.value),
        "unguarded_control_torn": int(torn_u.value),
    }
    all_ok = all_ok and seq_ok

    # 3. Strict causality bit-exact audit
    series = [math.sin(0.2 * t) + 0.05 * t for t in range(48)]
    causal_report = assert_causal(
        lambda: GapAsyncRuntime(spec=StreamSpec(name="verify_causal", window=16)),
        series,
        split_step=24,
    )
    causal_ok = bool(causal_report["causal"] and causal_report["max_diff"] == 0.0)
    checks["causality_bit_exact"] = {
        "ok": causal_ok,
        "max_diff": causal_report["max_diff"],
        "split_step": causal_report["split_step"],
    }
    all_ok = all_ok and causal_ok

    # 4. Multi-agent domain-partitioned NaN isolation check
    with MultiAgentStreamHub(3, window=16) as h_clean, MultiAgentStreamHub(3, window=16) as h_nan:
        a_clean = h_clean.subscribe(StreamSpec(name="agent_a", channels=(0, 1)))
        a_nan = h_nan.subscribe(StreamSpec(name="agent_a", channels=(0, 1)))
        h_clean.subscribe(StreamSpec(name="agent_b", channels=(0, 1, 2)))
        h_nan.subscribe(StreamSpec(name="agent_b", channels=(0, 1, 2)))
        max_bleed = 0.0
        for t in range(30):
            v0 = 10.0 + math.sin(0.3 * t)
            v1 = 20.0 + math.cos(0.2 * t)
            r_clean = h_clean.ingest(t, [v0, v1, 30.0 + t])[a_clean.agent_id]
            r_nan = h_nan.ingest(t, [v0, v1, float("nan")])[a_nan.agent_id]
            for c_q, n_q in zip(r_clean.q9, r_nan.q9, strict=True):
                max_bleed = max(max_bleed, abs(c_q - n_q))
        cov_clean = h_clean.agent_covariance(a_clean.agent_id)["cov"]
        cov_nan = h_nan.agent_covariance(a_nan.agent_id)["cov"]
        for i in range(2):
            for j in range(2):
                max_bleed = max(max_bleed, abs(cov_clean[i][j] - cov_nan[i][j]))
        ledger_ok = h_clean.verify_ledger() and h_nan.verify_ledger()

    nan_ok = max_bleed == 0.0 and ledger_ok
    checks["nan_isolation_and_ledger"] = {
        "ok": nan_ok,
        "cross_agent_max_diff": max_bleed,
        "ledger_verified": ledger_ok,
    }
    all_ok = all_ok and nan_ok

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "ok": all_ok,
        "elapsed_ms": round(elapsed_ms, 2),
        "checks": checks,
    }


_MAX_BENCH_STEPS = 5_000_000
_MAX_BENCH_AGENTS = 1024


def run_benchmark(*, steps: int = 20_000, n_agents: int = 16) -> dict:
    """Run local CPU/C11 arrival, Schur-complement, and multi-agent hub benchmarks."""
    if (
        isinstance(steps, bool)
        or not isinstance(steps, int)
        or not (1 <= steps <= _MAX_BENCH_STEPS)
    ):
        raise ValueError(f"steps must be an integer in [1, {_MAX_BENCH_STEPS}], got {steps!r}")
    if (
        isinstance(n_agents, bool)
        or not isinstance(n_agents, int)
        or not (1 <= n_agents <= _MAX_BENCH_AGENTS)
    ):
        raise ValueError(
            f"n_agents must be an integer in [1, {_MAX_BENCH_AGENTS}], got {n_agents!r}"
        )

    import ctypes
    import math
    import time

    from .native.build import load_native_lib, native_info
    from .runtime.arrival import PwlArrivalEngine
    from .runtime.hub import MultiAgentStreamHub
    from .runtime.spec import StreamSpec

    lib = load_native_lib()
    nat = native_info()

    # 1. PWL-ARRIVE hot-path throughput & latency
    with PwlArrivalEngine(n_keys=1, window=64) as eng:
        q_slow = [-1.28, -0.84, -0.52, -0.25, 0.0, 0.25, 0.52, 0.84, 1.28]
        q_fast = [-1.10, -0.75, -0.45, -0.20, 0.05, 0.28, 0.58, 0.90, 1.35]
        eng.publish_forecast(0, 0, q_slow, q_fast, scale_hint=1.0)
        xs = [0.2 * math.sin(0.01 * i) for i in range(steps)]
        t0 = time.perf_counter()
        for t, xv in enumerate(xs, start=1):
            eng.step(0, t, xv)
        dt = time.perf_counter() - t0
        telem = eng.read_telemetry()
    pwl_rate = steps / max(dt, 1e-9)
    pwl_us = (dt / max(steps, 1)) * 1e6

    # 2. Shared-core Schur solve (K_core=64, K_aug=4, 10 forms)
    kc, ka, nf = 64, 4, 10
    A_cc = (ctypes.c_double * (kc * kc))()
    for i in range(kc):
        for j in range(kc):
            A_cc[i * kc + j] = 2.0 if i == j else 0.01 / (1 + abs(i - j))
    A_ca = (ctypes.c_double * (nf * kc * ka))(*([0.02] * (nf * kc * ka)))
    A_aa = (ctypes.c_double * (nf * ka * ka))()
    for f in range(nf):
        for i in range(ka):
            for j in range(ka):
                A_aa[f * ka * ka + i * ka + j] = 2.5 if i == j else 0.05
    b_c = (ctypes.c_double * (nf * kc))(*([1.0] * (nf * kc)))
    b_a = (ctypes.c_double * (nf * ka))(*([0.5] * (nf * ka)))
    w_out = (ctypes.c_double * (nf * (kc + ka)))()
    used_linv = ctypes.c_int32(0)
    fb_cnt = ctypes.c_int32(0)
    max_res = ctypes.c_double(0.0)

    n_schur = 200
    t0 = time.perf_counter()
    for _ in range(n_schur):
        lib.tsjit_schur_solve_shared_core(
            kc,
            ka,
            nf,
            A_cc,
            A_ca,
            A_aa,
            b_c,
            b_a,
            w_out,
            ctypes.byref(used_linv),
            ctypes.byref(fb_cnt),
            ctypes.byref(max_res),
        )
    dt_schur = time.perf_counter() - t0
    schur_us = (dt_schur / n_schur) * 1e6

    # 3. Multi-agent shared-arena throughput
    hub_steps = max(200, min(steps // 20, 2000))
    n_ch = 8
    with MultiAgentStreamHub(n_ch, window=64, init_agent_cap=max(4, n_agents)) as hub:
        for a in range(n_agents):
            c0 = a % n_ch
            c1 = (a + 1) % n_ch
            hub.subscribe(StreamSpec(name=f"agent_{a}", channels=(c0, c1)))
        t0 = time.perf_counter()
        for t in range(hub_steps):
            vals = [math.sin(0.05 * t + c) for c in range(n_ch)]
            hub.ingest(t, vals)
        dt_hub = time.perf_counter() - t0
    hub_agent_steps_sec = (hub_steps * n_agents) / max(dt_hub, 1e-9)

    return {
        "simd_isa": nat["simd_isa"],
        "cacheline_bytes": nat["cacheline_bytes"],
        "pwl_arrive": {
            "steps": steps,
            "mean_us": round(pwl_us, 3),
            "arrivals_per_sec": round(pwl_rate, 1),
            "hot_hits": int(telem["pwl_fast_hits"]),
        },
        "schur_shared_core": {
            "k_core": kc,
            "k_aug": ka,
            "n_forms": nf,
            "solves": n_schur,
            "mean_us_per_10form_solve": round(schur_us, 2),
            "used_linv": bool(used_linv.value),
            "fallback_count": int(fb_cnt.value),
            "max_rel_residual": float(max_res.value),
        },
        "multi_agent_hub": {
            "n_channels": n_ch,
            "n_agents": n_agents,
            "ticks": hub_steps,
            "agent_ticks_per_sec": round(hub_agent_steps_sec, 1),
        },
    }


def _print_table(title: str, data: dict) -> None:
    print(f"=== {title} ===")
    for k, v in data.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for sk, sv in v.items():
                print(f"    {sk:<28} {sv}")
        else:
            print(f"  {k:<30} {v}")


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)

    # Ultra-fast path for --help / --version (<25 ms cold startup)
    if not args or args[0] in ("-h", "--help", "help"):
        sys.stdout.write(_HELP_TEXT)
        return 0
    if args[0] in ("-V", "--version"):
        sys.stdout.write(f"tsjit {__version__}\n")
        return 0

    cmd = args[0]
    rest = args[1:]

    if cmd == "mcp":
        if any(a in ("-h", "--help") for a in rest):
            sys.stdout.write("usage: tsjit mcp [-h]\n\nStart stdio JSON-RPC 2.0 MCP server.\n")
            return 0
        if rest:
            sys.stderr.write(f"tsjit mcp: unexpected arguments: {' '.join(rest)}\n")
            return 2
        from .mcp_server import serve_stdio

        return serve_stdio()

    if cmd == "info":
        if any(a in ("-h", "--help") for a in rest):
            sys.stdout.write(
                "usage: tsjit info [-h] [--json]\n\n"
                "Show version, native C11 backend, SIMD ISA, and optional backends.\n"
            )
            return 0
        unknown = [a for a in rest if a != "--json"]
        if unknown:
            sys.stderr.write(f"tsjit info: unknown option(s): {' '.join(unknown)}\n")
            return 2
        import json

        res = collect_info()
        if "--json" in rest:
            print(json.dumps(res, indent=2, sort_keys=True))
        else:
            _print_table("tsjit info", res)
        return 0

    if cmd == "verify":
        if any(a in ("-h", "--help") for a in rest):
            sys.stdout.write(
                "usage: tsjit verify [-h] [--full] [--json]\n\n"
                "Run native C11 build, seqlock contention, causality, and NaN checks.\n"
            )
            return 0
        unknown = [a for a in rest if a not in ("--full", "--json")]
        if unknown:
            sys.stderr.write(f"tsjit verify: unknown option(s): {' '.join(unknown)}\n")
            return 2
        import json

        quick = "--full" not in rest
        res = run_verification(quick=quick)
        if "--json" in rest:
            print(json.dumps(res, indent=2, sort_keys=True))
        else:
            _print_table("tsjit verify", res)
        return 0 if res["ok"] else 1

    if cmd == "bench":
        if any(a in ("-h", "--help") for a in rest):
            sys.stdout.write(
                "usage: tsjit bench [-h] [--steps N] [--agents M] [--json]\n\n"
                "Run local C11 arrival, Schur, and multi-agent streaming benchmarks.\n"
            )
            return 0
        steps = 20_000
        n_agents = 16
        as_json = False
        i = 0
        while i < len(rest):
            tok = rest[i]
            if tok == "--json":
                as_json = True
                i += 1
            elif tok == "--steps":
                if i + 1 >= len(rest):
                    sys.stderr.write("tsjit bench: --steps requires an integer argument\n")
                    return 2
                try:
                    steps = int(rest[i + 1])
                except ValueError:
                    sys.stderr.write(f"tsjit bench: invalid --steps value {rest[i + 1]!r}\n")
                    return 2
                if steps < 1 or steps > _MAX_BENCH_STEPS:
                    sys.stderr.write(
                        f"tsjit bench: --steps must be in [1, {_MAX_BENCH_STEPS}]\n"
                    )
                    return 2
                i += 2
            elif tok == "--agents":
                if i + 1 >= len(rest):
                    sys.stderr.write("tsjit bench: --agents requires an integer argument\n")
                    return 2
                try:
                    n_agents = int(rest[i + 1])
                except ValueError:
                    sys.stderr.write(f"tsjit bench: invalid --agents value {rest[i + 1]!r}\n")
                    return 2
                if n_agents < 1 or n_agents > _MAX_BENCH_AGENTS:
                    sys.stderr.write(
                        f"tsjit bench: --agents must be in [1, {_MAX_BENCH_AGENTS}]\n"
                    )
                    return 2
                i += 2
            else:
                sys.stderr.write(f"tsjit bench: unknown option {tok!r}\n")
                return 2
        import json

        res = run_benchmark(steps=steps, n_agents=n_agents)
        if as_json:
            print(json.dumps(res, indent=2, sort_keys=True))
        else:
            _print_table("tsjit bench", res)
        return 0

    sys.stderr.write(f"tsjit: unknown command {cmd!r}\n\n{_HELP_TEXT}")
    return 2
