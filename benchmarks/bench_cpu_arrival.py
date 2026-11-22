#!/usr/bin/env python3
"""CPU/C11 streaming latency, Schur-complement, multi-agent hub, and CLI startup benchmark."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from tsjit.native.build import load_native_lib, native_info
from tsjit.runtime.arrival import PwlArrivalEngine, SoftRevisionEngine
from tsjit.runtime.hub import MultiAgentStreamHub
from tsjit.runtime.spec import StreamSpec


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(max(int(round((len(sorted_vals) - 1) * p)), 0), len(sorted_vals) - 1)
    return sorted_vals[idx]


def bench_pwl_arrive(steps: int = 50_000) -> dict:
    with PwlArrivalEngine(n_keys=1, window=64) as eng:
        q_slow = [-1.28, -0.84, -0.52, -0.25, 0.0, 0.25, 0.52, 0.84, 1.28]
        q_fast = [-1.15, -0.75, -0.45, -0.20, 0.04, 0.28, 0.58, 0.90, 1.35]
        eng.publish_forecast(0, 0, q_slow, q_fast, scale_hint=1.0)

        # Warmup
        for t in range(1, 501):
            eng.step(0, t, 0.1 * math.sin(0.05 * t))

        xs = [0.25 * math.sin(0.01 * i) for i in range(steps)]
        t0 = time.perf_counter()
        for idx, x in enumerate(xs, start=501):
            eng.step(0, idx, x)
        dt = time.perf_counter() - t0

        # Sampled per-call latency distribution (in microseconds, includes ctypes call overhead)
        sample_n = min(5_000, steps)
        lat_us: list[float] = []
        base = 501 + steps
        for idx in range(sample_n):
            s0 = time.perf_counter_ns()
            eng.step(0, base + idx, 0.2 * math.sin(0.01 * idx))
            lat_us.append((time.perf_counter_ns() - s0) / 1000.0)
        lat_us.sort()
        telem = eng.read_telemetry()

    return {
        "steps": steps,
        "throughput_arrivals_per_sec": round(steps / max(dt, 1e-9), 1),
        "mean_us": round((dt / max(steps, 1)) * 1e6, 3),
        "p50_us": round(_percentile(lat_us, 0.50), 3),
        "p95_us": round(_percentile(lat_us, 0.95), 3),
        "p99_us": round(_percentile(lat_us, 0.99), 3),
        "hot_hits": int(telem["pwl_fast_hits"]),
    }


def bench_soft_revision(steps: int = 20_000) -> dict:
    with SoftRevisionEngine(n_keys=1, window=64) as eng:
        q_slow = [-1.28, -0.84, -0.52, -0.25, 0.0, 0.25, 0.52, 0.84, 1.28]
        q_fast = [-1.15, -0.75, -0.45, -0.20, 0.04, 0.28, 0.58, 0.90, 1.35]
        eng.publish_forecast(0, 0, q_slow, q_fast, scale_hint=1.0)
        t0 = time.perf_counter()
        for t in range(1, steps + 1):
            eng.step_soft(0, t, 0.3 * math.sin(0.02 * t))
        dt = time.perf_counter() - t0
    return {
        "steps": steps,
        "throughput_arrivals_per_sec": round(steps / max(dt, 1e-9), 1),
        "mean_us": round((dt / max(steps, 1)) * 1e6, 3),
    }


def bench_schur_shared_core() -> list[dict]:
    lib = load_native_lib()
    results: list[dict] = []
    for k_core, k_aug, n_forms, n_iters in ((32, 4, 10, 400), (64, 4, 10, 200), (128, 8, 10, 60)):
        A_cc = (ctypes.c_double * (k_core * k_core))()
        for i in range(k_core):
            for j in range(k_core):
                A_cc[i * k_core + j] = 2.0 if i == j else (0.01 / (1 + abs(i - j)))
        A_ca = (ctypes.c_double * (n_forms * k_core * k_aug))(
            *([0.02] * (n_forms * k_core * k_aug))
        )
        A_aa = (ctypes.c_double * (n_forms * k_aug * k_aug))()
        for f in range(n_forms):
            for i in range(k_aug):
                for j in range(k_aug):
                    A_aa[f * k_aug * k_aug + i * k_aug + j] = 2.5 if i == j else 0.05
        b_c = (ctypes.c_double * (n_forms * k_core))(*([1.0] * (n_forms * k_core)))
        b_a = (ctypes.c_double * (n_forms * k_aug))(*([0.5] * (n_forms * k_aug)))
        w_out = (ctypes.c_double * (n_forms * (k_core + k_aug)))()
        used_linv = ctypes.c_int32(0)
        fb = ctypes.c_int32(0)
        max_r = ctypes.c_double(0.0)

        # Warmup
        lib.tsjit_schur_solve_shared_core(
            k_core,
            k_aug,
            n_forms,
            A_cc,
            A_ca,
            A_aa,
            b_c,
            b_a,
            w_out,
            ctypes.byref(used_linv),
            ctypes.byref(fb),
            ctypes.byref(max_r),
        )
        t0 = time.perf_counter()
        for _ in range(n_iters):
            lib.tsjit_schur_solve_shared_core(
                k_core,
                k_aug,
                n_forms,
                A_cc,
                A_ca,
                A_aa,
                b_c,
                b_a,
                w_out,
                ctypes.byref(used_linv),
                ctypes.byref(fb),
                ctypes.byref(max_r),
            )
        dt = time.perf_counter() - t0
        results.append(
            {
                "k_core": k_core,
                "k_aug": k_aug,
                "n_forms": n_forms,
                "mean_us": round((dt / n_iters) * 1e6, 2),
                "used_linv": bool(used_linv.value),
                "fallback_count": int(fb.value),
                "max_rel_residual": float(max_r.value),
            }
        )
    return results


def bench_multi_agent_hub(ticks: int = 500) -> list[dict]:
    n_ch = 16
    rows: list[dict] = []
    for n_agents in (1, 8, 32):
        with MultiAgentStreamHub(n_ch, window=64, init_agent_cap=max(4, n_agents)) as hub:
            for a in range(n_agents):
                dom = ((a % n_ch), ((a + 1) % n_ch), ((a + 2) % n_ch))
                hub.subscribe(StreamSpec(name=f"a_{a}", channels=dom))
            t0 = time.perf_counter()
            for t in range(ticks):
                vals = [math.sin(0.03 * t + c) for c in range(n_ch)]
                hub.ingest(t, vals)
            dt = time.perf_counter() - t0
        rows.append(
            {
                "n_channels": n_ch,
                "n_agents": n_agents,
                "ticks": ticks,
                "tick_us": round((dt / ticks) * 1e6, 2),
                "agent_ticks_per_sec": round((ticks * n_agents) / max(dt, 1e-9), 1),
            }
        )
    return rows


def bench_cli_startup(runs: int = 5) -> dict:
    repo_src = str(Path(__file__).resolve().parents[1] / "src")
    env = dict(os.environ)
    env["PYTHONPATH"] = repo_src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    def _measure(cmd: list[str]) -> float:
        # Warm OS page cache
        subprocess.run(
            cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        samples = []
        for _ in range(runs):
            t0 = time.perf_counter()
            subprocess.run(
                cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
            )
            samples.append((time.perf_counter() - t0) * 1000.0)
        return round(statistics.median(samples), 2)

    return {
        "import_tsjit_median_ms": _measure([sys.executable, "-c", "import tsjit"]),
        "cli_help_median_ms": _measure([sys.executable, "-m", "tsjit", "--help"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="tsjit CPU/C11 arrival & runtime benchmark suite")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    nat_info = dict(native_info())
    for path_key in ("library_path", "compiler"):
        if nat_info.get(path_key):
            nat_info[path_key] = Path(str(nat_info[path_key])).name

    report = {
        "native": nat_info,
        "pwl_arrive": bench_pwl_arrive(args.steps),
        "soft_revision_v2": bench_soft_revision(max(5_000, args.steps // 2)),
        "schur_shared_core": bench_schur_shared_core(),
        "multi_agent_hub": bench_multi_agent_hub(500),
        "cli_startup": bench_cli_startup(5),
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        nat = report["native"]
        print("=== tsjit CPU/C11 Benchmark Report ===")
        print(
            f"SIMD ISA: {nat['simd_isa']} | "
            f"Cacheline: {nat['cacheline_bytes']} B | "
            f"PWL hot row: {nat['pwl_hot_row_bytes']} B"
        )
        p = report["pwl_arrive"]
        print(
            f"PWL-ARRIVE      : {p['throughput_arrivals_per_sec']:>10.1f} arr/s | "
            f"mean={p['mean_us']:.3f} us | p50={p['p50_us']:.3f} us | p99={p['p99_us']:.3f} us"
        )
        s = report["soft_revision_v2"]
        print(
            f"SoftRevisionV2  : {s['throughput_arrivals_per_sec']:>10.1f} arr/s | "
            f"mean={s['mean_us']:.3f} us"
        )
        for row in report["schur_shared_core"]:
            print(
                f"Schur(Kc={row['k_core']:>3},Ka={row['k_aug']},F={row['n_forms']}): "
                f"{row['mean_us']:>8.2f} us | linv={row['used_linv']} | "
                f"fb={row['fallback_count']} | max_res={row['max_rel_residual']:.2e}"
            )
        for row in report["multi_agent_hub"]:
            print(
                f"Hub(C={row['n_channels']},M={row['n_agents']:>2})    : "
                f"{row['tick_us']:>8.2f} us/tick | "
                f"{row['agent_ticks_per_sec']:>10.1f} agent-ticks/s"
            )
        c = report["cli_startup"]
        print(
            f"CLI startup     : import={c['import_tsjit_median_ms']:.2f} ms | "
            f"--help={c['cli_help_median_ms']:.2f} ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
