"""Exhaustive C11 native engine tests including ASan+UBSan and TSan execution."""

from __future__ import annotations

import ctypes
import math
import random
import subprocess
import tempfile
from pathlib import Path

from tsjit.native.build import (
    MARK_DUP_STEP,
    MARK_INVALID_KEY,
    MARK_LAST_VALID,
    MARK_NAN_STALE,
    MARK_OK,
    MARK_OOO_STEP,
    MARK_ZERO,
    TSJIT_ERR_BAD_ARG,
    TSJIT_ERR_OOB,
    TSJIT_ERR_OVERFLOW,
    TSJIT_OK,
    CEwmaMonoid,
    build_c_executable,
    load_native_lib,
    native_info,
)
from tsjit.runtime.arrival import PwlArrivalEngine, SoftRevisionEngine


def _two_pass_mean_std(vals: list[float]) -> tuple[float, float]:
    n = len(vals)
    mu = math.fsum(vals) / n
    ss = math.fsum((v - mu) ** 2 for v in vals)
    return mu, math.sqrt(ss / (n - 1)) if n > 1 else 0.0


def test_platform_abi_and_hot_row_size() -> None:
    info = native_info()
    assert info["available"] is True
    assert info["simd_isa"] in ("NEON", "AVX2", "SCALAR")
    assert info["cacheline_bytes"] in (64, 128)
    assert info["pwl_hot_row_bytes"] <= 640
    assert info["pwl_hot_row_bytes"] % info["cacheline_bytes"] == 0


def test_sort9_zero_one_principle_and_nan_preservation() -> None:
    lib = load_native_lib()
    c_arr = ctypes.c_double * 9

    # Zero-One Principle: verify ALL 2^9 = 512 binary inputs
    for mask in range(512):
        inp = [float((mask >> i) & 1) for i in range(9)]
        buf = c_arr(*inp)
        lib.tsjit_sort9_abi(buf)
        assert list(buf) == sorted(inp)

    # 1 NaN stays exactly 1 NaN (never erased or cloned to 9!)
    for pos in range(9):
        vals = [9.0, 2.0, -3.5, 7.1, 0.0, 4.4, -1.2, 5.5, 8.8]
        vals[pos] = float("nan")
        buf = c_arr(*vals)
        lib.tsjit_sort9_abi(buf)
        out = list(buf)
        nan_cnt = sum(1 for v in out if math.isnan(v))
        assert nan_cnt == 1, f"pos={pos}: expected 1 NaN, got {nan_cnt} in {out}"
        finite = [v for v in out if not math.isnan(v)]
        assert finite == sorted(finite)
        assert math.isnan(out[-1])

    # 3 NaNs stay exactly 3 NaNs at the tail
    vals3 = [5.0, float("nan"), -2.0, float("nan"), 1.0, 3.0, float("nan"), 0.0, 4.0]
    buf3 = c_arr(*vals3)
    lib.tsjit_sort9_abi(buf3)
    out3 = list(buf3)
    assert sum(1 for v in out3 if math.isnan(v)) == 3
    assert [v for v in out3 if not math.isnan(v)] == [-2.0, 0.0, 1.0, 3.0, 4.0, 5.0]


def test_dc_offset_2_22e10_and_micro_scale_1e_6() -> None:
    rng = random.Random(42)
    dc = 2.22e10
    win = 64
    eps = [rng.gauss(0.0, 1.0) for _ in range(120)]
    xs_dc = [dc + e for e in eps]

    with PwlArrivalEngine(n_keys=1, window=win) as eng:
        for t, x in enumerate(xs_dc):
            eng.step(0, t, x)
        st = eng.get_rolling(0)
        ref_mean, ref_std = _two_pass_mean_std(xs_dc[-win:])
        assert abs(st["mean"] - ref_mean) < 1e-4
        assert abs(st["std"] - ref_std) < 1e-5

    # 1e-6 micro-scale preservation (must NOT collapse to 1e-3 or 1.0 floor)
    xs_micro = [1e-6 + 1e-7 * e for e in eps]
    with PwlArrivalEngine(n_keys=1, window=win) as eng:
        q_slow = [
            1e-6 + 1e-7 * z for z in (-1.28, -0.84, -0.52, -0.25, 0.0, 0.25, 0.52, 0.84, 1.28)
        ]
        eng.publish_forecast(0, 0, q_slow, q_slow, scale_hint=1e-7)
        for t, x in enumerate(xs_micro):
            res = eng.step(0, t, x)
        st = eng.get_rolling(0)
        ref_m, ref_s = _two_pass_mean_std(xs_micro[-win:])
        assert abs(st["mean"] - ref_m) < 1e-13
        assert abs(st["std"] - ref_s) < 1e-13
        assert 5e-8 < res.eff_scale < 5e-7
        # Inject a 5-sigma micro jump (+5e-7): abs_sg must be ~5.0
        jump_res = eng.step(0, len(xs_micro), 1e-6 + 5e-7)
        assert abs(jump_res.abs_sg - 5.0) < 0.2


def test_spike_eviction_1e20_and_cross_channel_1e18_isolation() -> None:
    rng = random.Random(123)
    win = 32
    clean = [rng.gauss(0.0, 1.0) for _ in range(100)]
    spiked = list(clean)
    spiked[10] = 1e20  # Evicted at step 10 + 32 = 42

    with PwlArrivalEngine(n_keys=1, window=win) as eng:
        for t, x in enumerate(spiked):
            eng.step(0, t, x)
        st = eng.get_rolling(0)
        telem = eng.read_telemetry()
        ref_m, ref_s = _two_pass_mean_std(spiked[-win:])
        assert telem["roll_refreshes"] >= 1
        assert abs(st["mean"] - ref_m) < 1e-10
        assert abs(st["std"] - ref_s) < 1e-10

    # Single-channel 1e18 cross-spike must leave key 1 bit-for-bit identical (db == 0.0)
    with (
        PwlArrivalEngine(n_keys=2, window=win) as eng_clean,
        PwlArrivalEngine(n_keys=2, window=win) as eng_spike,
    ):
        q0 = [-1.28, -0.84, -0.52, -0.25, 0.0, 0.25, 0.52, 0.84, 1.28]
        for eng in (eng_clean, eng_spike):
            eng.publish_forecast(0, 0, q0, q0, scale_hint=1.0)
            eng.publish_forecast(1, 0, q0, q0, scale_hint=1.0)
        for t in range(50):
            x0_c = math.sin(0.2 * t)
            x0_s = 1e18 if t == 15 else x0_c
            x1 = math.cos(0.3 * t)
            eng_clean.step(0, t, x0_c)
            eng_spike.step(0, t, x0_s)
            rc = eng_clean.step(1, t, x1)
            rs = eng_spike.step(1, t, x1)
            assert rc.q9 == rs.q9
            assert rc.rolling_mean == rs.rolling_mean
            assert rc.rolling_std == rs.rolling_std
            assert rc.abs_sg == rs.abs_sg


def test_schur_winsorization_10000_sigma_and_spread_glitches() -> None:
    lib = load_native_lib()
    k = 4
    sh = lib.tsjit_schur_create(k, 1, 0.98, 1e-3, 20.0)
    assert sh
    try:
        rng = random.Random(7)
        true_w = [1.5, -2.0, 0.75, 0.5]
        c_x = ctypes.c_double * k
        c_y = ctypes.c_double * 1
        c_sp = ctypes.c_double * k
        w_out = ctypes.c_double(0.0)

        # Warm up on 80 clean samples
        for _ in range(80):
            xv = [1.0, rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)]
            yv = sum(a * b for a, b in zip(true_w, xv, strict=True)) + rng.gauss(0.0, 0.02)
            assert (
                lib.tsjit_schur_update(sh, c_x(*xv), c_y(yv), None, ctypes.byref(w_out)) == TSJIT_OK
            )

        probe = [1.0, 0.5, -0.5, 1.0]
        pred_before = ctypes.c_double(0.0)
        assert lib.tsjit_schur_predict(sh, c_x(*probe), ctypes.byref(pred_before)) == TSJIT_OK
        true_probe = sum(a * b for a, b in zip(true_w, probe, strict=True))
        assert abs(pred_before.value - true_probe) < 0.05

        # Inject 10,000-sigma feature & target glitch
        glitch_x = [1.0, 10_000.0, -10_000.0, 5_000.0]
        glitch_y = [10_000.0]
        assert (
            lib.tsjit_schur_update(sh, c_x(*glitch_x), c_y(*glitch_y), None, ctypes.byref(w_out))
            == TSJIT_OK
        )
        assert 0.0 < w_out.value < 1e-4

        # Inject extreme spread glitches (0.0, Inf, NaN, 1e300) + non-finite input -> ZERO 0*Inf!
        bad_spreads = [0.0, float("inf"), float("nan"), 1e300]
        assert (
            lib.tsjit_schur_update(
                sh, c_x(1.0, 10_000.0, 0.1, -0.2), c_y(1.0), c_sp(*bad_spreads), ctypes.byref(w_out)
            )
            == TSJIT_OK
        )
        assert (
            lib.tsjit_schur_update(
                sh,
                c_x(1.0, float("inf"), 0.0, 0.0),
                c_y(1.0),
                c_sp(*bad_spreads),
                ctypes.byref(w_out),
            )
            == TSJIT_OK
        )
        assert w_out.value == 0.0

        pred_after = ctypes.c_double(0.0)
        assert lib.tsjit_schur_predict(sh, c_x(*probe), ctypes.byref(pred_after)) == TSJIT_OK
        assert math.isfinite(pred_after.value)
        assert abs(pred_after.value - true_probe) < 0.15
    finally:
        lib.tsjit_schur_destroy(sh)


def test_higham_ch14_guard_and_extreme_kappa_collinear_fallback() -> None:
    lib = load_native_lib()

    # Higham Ch. 14 bidiagonal L: L_ii = 1.0 (min/max diag ratio == 1.0!), L_{i+1,i} = -2.0
    n_guard = 46
    L = [0.0] * (n_guard * n_guard)
    for i in range(n_guard):
        L[i * n_guard + i] = 1.0
        if i + 1 < n_guard:
            L[(i + 1) * n_guard + i] = -2.0
    allow = ctypes.c_int32(-1)
    dr = ctypes.c_double(0.0)
    kap = ctypes.c_double(0.0)
    assert (
        lib.tsjit_schur_check_higham_guard(
            n_guard,
            (ctypes.c_double * len(L))(*L),
            ctypes.byref(allow),
            ctypes.byref(dr),
            ctypes.byref(kap),
        )
        == TSJIT_OK
    )
    assert dr.value == 1.0
    assert kap.value > 1e12
    assert allow.value == 0  # Higham guard MUST reject L^{-1}!

    # Extreme ill-conditioned A = L L^T with kappa(A) ~ 10^14 (n=24, kappa(L) ~ 1.6e7)
    n = 24
    L2 = [[0.0] * n for _ in range(n)]
    for i in range(n):
        L2[i][i] = 1.0
        if i + 1 < n:
            L2[i + 1][i] = -2.0
    A = [0.0] * (n * n)
    for i in range(n):
        for j in range(n):
            A[i * n + j] = sum(L2[i][p] * L2[j][p] for p in range(n))
    x_true = [(-1.0) ** i * (1.0 + 0.1 * i) for i in range(n)]
    b = [sum(A[i * n + j] * x_true[j] for j in range(n)) for i in range(n)]

    x_out = (ctypes.c_double * n)()
    method = ctypes.c_int32(0)
    rel_r = ctypes.c_double(0.0)
    k_est = ctypes.c_double(0.0)
    assert (
        lib.tsjit_schur_solve_system(
            n,
            (ctypes.c_double * len(A))(*A),
            (ctypes.c_double * n)(*b),
            x_out,
            ctypes.byref(method),
            ctypes.byref(rel_r),
            ctypes.byref(k_est),
        )
        == TSJIT_OK
    )
    x_sol = list(x_out)
    assert all(math.isfinite(v) for v in x_sol)

    # Backward error ||Ax - b||_inf / (||A||_inf ||x||_inf + ||b||_inf) <= 1e-14
    a_inf = max(sum(abs(A[i * n + j]) for j in range(n)) for i in range(n))
    x_inf = max(abs(v) for v in x_sol)
    b_inf = max(abs(v) for v in b)
    r_inf = max(abs(sum(A[i * n + j] * x_sol[j] for j in range(n)) - b[i]) for i in range(n))
    bw_err = r_inf / (a_inf * x_inf + b_inf)
    assert bw_err <= 1e-14, f"Backward error too large: {bw_err}"

    # Collinear shared-core Schur Gram with tiny ridge 1e-12 -> exact fallback & bw_err <= 1e-14
    kc, ka, nf = 6, 2, 3
    K = kc + ka
    # Build rank-deficient Gram where augmented columns duplicate core columns 0, 1
    rng = random.Random(99)
    X_core = [[rng.gauss(0.0, 1.0) for _ in range(kc)] for _ in range(20)]
    A_cc = [0.0] * (kc * kc)
    for i in range(kc):
        for j in range(kc):
            A_cc[i * kc + j] = sum(row[i] * row[j] for row in X_core) + (1e-12 if i == j else 0.0)
    A_ca_all: list[float] = []
    A_aa_all: list[float] = []
    b_c_all: list[float] = []
    b_a_all: list[float] = []
    for f in range(nf):
        # Augmented cols equal core cols 0 and 1 (exact collinearity before 1e-12 ridge!)
        for i in range(kc):
            A_ca_all.append(sum(row[i] * row[0] for row in X_core))
            A_ca_all.append(sum(row[i] * row[1] for row in X_core))
        for i in range(ka):
            for j in range(ka):
                A_aa_all.append(sum(row[i] * row[j] for row in X_core) + (1e-12 if i == j else 0.0))
        w_ref = [0.2 * (p + 1) + 0.1 * f for p in range(K)]
        # Consistent RHS b = A_full * w_ref
        for i in range(kc):
            s = sum(A_cc[i * kc + j] * w_ref[j] for j in range(kc))
            s += sum(row[i] * row[0] for row in X_core) * w_ref[kc]
            s += sum(row[i] * row[1] for row in X_core) * w_ref[kc + 1]
            b_c_all.append(s)
        for i in range(ka):
            s = sum(sum(row[i] * row[j] for row in X_core) * w_ref[j] for j in range(kc))
            for j in range(ka):
                s += (sum(row[i] * row[j] for row in X_core) + (1e-12 if i == j else 0.0)) * w_ref[
                    kc + j
                ]
            b_a_all.append(s)

    w_out = (ctypes.c_double * (nf * K))()
    used_linv = ctypes.c_int32(0)
    fb_cnt = ctypes.c_int32(0)
    max_res = ctypes.c_double(0.0)
    assert (
        lib.tsjit_schur_solve_shared_core(
            kc,
            ka,
            nf,
            (ctypes.c_double * len(A_cc))(*A_cc),
            (ctypes.c_double * len(A_ca_all))(*A_ca_all),
            (ctypes.c_double * len(A_aa_all))(*A_aa_all),
            (ctypes.c_double * len(b_c_all))(*b_c_all),
            (ctypes.c_double * len(b_a_all))(*b_a_all),
            w_out,
            ctypes.byref(used_linv),
            ctypes.byref(fb_cnt),
            ctypes.byref(max_res),
        )
        == TSJIT_OK
    )
    assert all(math.isfinite(w_out[i]) for i in range(nf * K))
    assert max_res.value <= 1e-11


def test_seqlock_contention_torture_over_1m_reads() -> None:
    lib = load_native_lib()
    tot_r = ctypes.c_uint64(0)
    torn_r = ctypes.c_uint64(0)
    to_r = ctypes.c_uint64(0)
    assert (
        lib.tsjit_seqlock_stress_test(
            2, 4, 300_000, 0, ctypes.byref(tot_r), ctypes.byref(torn_r), ctypes.byref(to_r)
        )
        == TSJIT_OK
    )
    assert tot_r.value == 1_200_000
    assert torn_r.value == 0
    assert to_r.value == 0

    # Unguarded control MUST exhibit > 1,000 real snapshot tears under contention
    tot_u = ctypes.c_uint64(0)
    torn_u = ctypes.c_uint64(0)
    to_u = ctypes.c_uint64(0)
    assert (
        lib.tsjit_seqlock_stress_test(
            2, 4, 300_000, 1, ctypes.byref(tot_u), ctypes.byref(torn_u), ctypes.byref(to_u)
        )
        == TSJIT_OK
    )
    assert tot_u.value == 1_200_000
    assert torn_u.value > 1_000


def test_abi_bounds_dup_ooo_degenerate_knots_and_soft_v2() -> None:
    lib = load_native_lib()
    INT64_MAX = (1 << 63) - 1
    INT64_MIN = -(1 << 63)

    # Overflow-safe bounds & safe_age_i64
    assert lib.tsjit_check_bounds_i64(INT64_MAX, 1, INT64_MAX) == TSJIT_ERR_OOB
    assert lib.tsjit_check_bounds_i64(0, 1, 1) == TSJIT_OK
    assert lib.tsjit_check_bounds_i64(-1, 1, 10) == TSJIT_ERR_OOB
    assert lib.tsjit_safe_age_i64_abi(INT64_MAX, INT64_MIN) == INT64_MAX
    assert lib.tsjit_safe_age_i64_abi(INT64_MIN, INT64_MAX) == INT64_MIN

    with SoftRevisionEngine(n_keys=2, window=16, lambda_age=0.25, max_age=8) as eng:
        # Invalid keys (-1 and OOB)
        assert eng.step(-1, 0, 1.0).mark == MARK_INVALID_KEY
        assert eng.step(2, 0, 1.0).mark == MARK_INVALID_KEY

        # NaN before any valid observation -> MARK_ZERO
        assert eng.step(0, 0, float("nan")).mark == MARK_ZERO

        # Valid observations
        q0 = [-1.28, -0.84, -0.52, -0.25, 0.0, 0.25, 0.52, 0.84, 1.28]
        eng.publish_forecast(0, 1, q0, q0, scale_hint=1.0)
        r1 = eng.step(0, 1, 0.2)
        assert r1.mark == MARK_OK and r1.pwl_path == 0  # Hot path!
        r2 = eng.step(0, 2, 0.3)
        assert r2.mark == MARK_OK
        snap_before = eng.get_rolling(0)

        # DUP and OOO must NOT mutate rolling state!
        r_dup = eng.step(0, 2, 9999.0)
        assert r_dup.mark == MARK_DUP_STEP
        assert eng.get_rolling(0) == snap_before

        r_ooo = eng.step(0, 1, -9999.0)
        assert r_ooo.mark == MARK_OOO_STEP
        assert eng.get_rolling(0) == snap_before

        # NaN with valid history -> MARK_LAST_VALID (not corrupting rolling stats)
        r_nan = eng.step(0, 3, float("nan"))
        assert r_nan.mark == MARK_LAST_VALID
        st_after_nan = eng.get_rolling(0)
        assert st_after_nan["mean"] == snap_before["mean"]
        assert st_after_nan["std"] == snap_before["std"]

        # Degenerate knots (un <= 1e-290 and un > 1e290) -> fallback flag active, finite output
        eng.publish_forecast(1, 0, q0, q0, scale_hint=1e-300)
        r_fb1 = eng.step(1, 1, 0.5)
        assert r_fb1.pwl_path == 2 and all(math.isfinite(v) for v in r_fb1.q9)

        eng.publish_forecast(1, 2, q0, q0, scale_hint=1e300)
        r_fb2 = eng.step(1, 2, 0.5)
        assert r_fb2.pwl_path == 2 and all(math.isfinite(v) for v in r_fb2.q9)

        # SoftRevisionV2 hierarchical equal family prior (1 slow vs 10 fast vs 1 persist)
        q_slow = [q0]
        q_fast_10 = [q0] * 10
        q_pers = [q0]
        eng.publish_forecast(0, 10, q0, q0, scale_hint=1.0)
        r_eq = eng.step_soft(
            0,
            10,
            0.0,
            slow_models=q_slow,
            slow_losses=[0.0],
            fast_models=q_fast_10,
            fast_losses=[0.0] * 10,
            persist_models=q_pers,
            persist_losses=[0.0],
            beta_surprise=0.0,
        )
        for fw in r_eq.family_weights:
            assert abs(fw - (1.0 / 3.0)) < 1e-12

        # Exponential age decay: as age grows, Slow-TSFM weight decays strictly
        r_old = eng.step_soft(
            0,
            30,
            0.0,
            slow_models=q_slow,
            slow_losses=[0.0],
            fast_models=q_fast_10,
            fast_losses=[0.0] * 10,
            persist_models=q_pers,
            persist_losses=[0.0],
            beta_surprise=0.0,
        )
        assert r_old.age == 20
        assert r_old.family_weights[0] < 0.05
        assert r_old.family_weights[2] > r_old.family_weights[0]


def test_ewma_monoid_subnormal_extreme_decay_and_aliasing() -> None:
    lib = load_native_lib()

    def make_block(n: int, w: float, mean: list[float], cov: list[float]):
        dim = len(mean)
        m_arr = (ctypes.c_double * dim)(*mean)
        c_arr = (ctypes.c_double * (dim * dim))(*cov)
        obj = CEwmaMonoid(n=n, dim=dim, _pad=0, w=w, mean=m_arr, cov=c_arr)
        return obj, m_arr, c_arr

    # 1. Subnormal decay: gamma = 0.99, nB = 71500 (log2(0.99)*71500 ~ -1036.7)
    A, ma, ca = make_block(100, 63.4, [10.0, -5.0], [2.0, 0.5, 0.5, 3.0])
    B, mb, cb = make_block(71500, 100.0, [1.0, 2.0], [0.5, 0.1, 0.1, 0.8])
    Out1, mo, co = make_block(0, 0.0, [0.0, 0.0], [0.0, 0.0, 0.0, 0.0])
    assert (
        lib.tsjit_ewma_monoid_merge(0.99, ctypes.byref(A), ctypes.byref(B), ctypes.byref(Out1))
        == TSJIT_OK
    )
    assert Out1.n == 71600
    assert all(math.isfinite(v) for v in (*mo, *co))
    assert abs(mo[0] - 1.0) < 1e-12 and abs(co[0] - 0.5) < 1e-12

    # 2. Extreme underflow boundary: gamma = 0.5, nB = 1075 (2^-1075 underflows IEEE double)
    A2, ma2, ca2 = make_block(50, 2.0, [500.0], [100.0])
    B2, mb2, cb2 = make_block(1075, 2.0, [3.0], [0.25])
    O2, mo2, co2 = make_block(0, 0.0, [0.0], [0.0])
    assert (
        lib.tsjit_ewma_monoid_merge(0.5, ctypes.byref(A2), ctypes.byref(B2), ctypes.byref(O2))
        == TSJIT_OK
    )
    assert O2.n == 1125
    assert math.isfinite(mo2[0]) and math.isfinite(co2[0])
    assert abs(mo2[0] - 3.0) < 1e-12 and abs(co2[0] - 0.25) < 1e-12

    # 3. 1e160 + 2000 * 2.0:
    #    (a) gamma=0.95 -> true cov ~ 1.29e274 (finite! naive delta^2=1e320 overflows to Inf!)
    g95 = 0.95
    wB_95 = (1.0 - g95**2000) / (1.0 - g95)
    A3, ma3, ca3 = make_block(1, 1.0, [1e160], [0.0])
    B3, mb3, cb3 = make_block(2000, wB_95, [2.0], [0.0])
    O3, mo3, co3 = make_block(0, 0.0, [0.0], [0.0])
    assert (
        lib.tsjit_ewma_monoid_merge(g95, ctypes.byref(A3), ctypes.byref(B3), ctypes.byref(O3))
        == TSJIT_OK
    )
    assert math.isfinite(mo3[0]) and math.isfinite(co3[0]) and co3[0] > 1e270

    #    (b) gamma=0.5 -> alpha_exp=-2000 < -1080 -> exact 0.0 guard (ZERO 0*Inf NaN!)
    A4, ma4, ca4 = make_block(1, 1.0, [1e160], [0.0])
    B4, mb4, cb4 = make_block(2000, 2.0, [2.0], [0.0])
    O4, mo4, co4 = make_block(0, 0.0, [0.0], [0.0])
    assert (
        lib.tsjit_ewma_monoid_merge(0.5, ctypes.byref(A4), ctypes.byref(B4), ctypes.byref(O4))
        == TSJIT_OK
    )
    assert mo4[0] == 2.0 and co4[0] == 0.0

    # 4. In-place aliasing merge(A, B, B) == out-of-place merge(A, B, O) bit-exact!
    X1, mx1, cx1 = make_block(10, 8.0, [1.5, -2.5], [0.9, 0.2, 0.2, 1.4])
    Y1, my1, cy1 = make_block(15, 11.0, [3.0, 0.5], [1.2, -0.3, -0.3, 0.7])
    Ref, mref, cref = make_block(0, 0.0, [0.0, 0.0], [0.0, 0.0, 0.0, 0.0])
    assert (
        lib.tsjit_ewma_monoid_merge(0.92, ctypes.byref(X1), ctypes.byref(Y1), ctypes.byref(Ref))
        == TSJIT_OK
    )
    assert (
        lib.tsjit_ewma_monoid_merge(0.92, ctypes.byref(X1), ctypes.byref(Y1), ctypes.byref(Y1))
        == TSJIT_OK
    )
    assert Y1.n == Ref.n and Y1.w == Ref.w
    assert list(my1) == list(mref) and list(cy1) == list(cref)

    # 5. O(log P) tree reduction matches sequential left-to-right folding
    rng = random.Random(2026)
    g = 0.93
    pts = [[rng.gauss(0.0, 2.0), rng.gauss(5.0, 1.0)] for _ in range(32)]
    seq_m, seq_ma, seq_ca = make_block(0, 0.0, [0.0, 0.0], [0.0] * 4)
    block_arr = (CEwmaMonoid * len(pts))()
    keep = []
    for idx, pt in enumerate(pts):
        blk, bm, bc = make_block(1, 1.0, pt, [0.0] * 4)
        keep.append((bm, bc))
        block_arr[idx] = blk
        assert (
            lib.tsjit_ewma_monoid_merge(
                g, ctypes.byref(seq_m), ctypes.byref(blk), ctypes.byref(seq_m)
            )
            == TSJIT_OK
        )

    tree_m, tree_ma, tree_ca = make_block(0, 0.0, [0.0, 0.0], [0.0] * 4)
    assert (
        lib.tsjit_ewma_monoid_merge_tree(g, block_arr, len(pts), ctypes.byref(tree_m)) == TSJIT_OK
    )
    assert tree_m.n == seq_m.n
    for a, b in zip(seq_ma, tree_ma, strict=True):
        assert abs(a - b) < 1e-12
    for a, b in zip(seq_ca, tree_ca, strict=True):
        assert abs(a - b) < 1e-12


def test_arena_ebr_150_concurrent_resizes_and_tombstones() -> None:
    lib = load_native_lib()
    reads = ctypes.c_uint64(0)
    reclaimed = ctypes.c_uint64(0)
    errors = ctypes.c_uint64(0)
    assert (
        lib.tsjit_arena_ebr_stress_test(
            4, 150, ctypes.byref(reads), ctypes.byref(reclaimed), ctypes.byref(errors)
        )
        == TSJIT_OK
    )
    assert errors.value == 0
    assert reads.value > 1_000
    assert reclaimed.value >= 140


def test_c11_adversarial_edge_cases_and_numerical_guards() -> None:
    """Regression tests for MARK_NAN_STALE, constant-window refresh, unsorted fallback knots,
    0*Inf finite-extreme guard, Schur nonzero-mean target glitch, GE scaling, and EWMA overflows.
    """
    lib = load_native_lib()

    # 1. MARK_NAN_STALE tracks last_valid_step age, NOT forecast age
    with PwlArrivalEngine(n_keys=1, window=16, max_age=5) as eng:
        eng.step(0, 0, 1.0)
        assert eng.step(0, 100, float("nan")).mark == MARK_NAN_STALE

    with PwlArrivalEngine(n_keys=1, window=16, max_age=5) as eng:
        q0 = [-1.28, -0.84, -0.52, -0.25, 0.0, 0.25, 0.52, 0.84, 1.28]
        eng.publish_forecast(0, 0, q0, q0, scale_hint=1.0)
        for t in range(1, 20):
            eng.step(0, t, 0.1 * t)
        assert eng.step(0, 20, float("nan")).mark == MARK_LAST_VALID

    # 2. Constant non-binary fraction (0.1) after shift must NOT thrash 2-pass refreshes
    with PwlArrivalEngine(n_keys=1, window=3) as eng:
        eng.step(0, 0, 0.0)
        for t in range(1, 100):
            eng.step(0, t, 0.1)
        assert eng.read_telemetry()["roll_refreshes"] <= 2

    # 3. Unsorted knots + degenerate scale_hint=1e-300 must sort knots & guard eff_scale
    with PwlArrivalEngine(n_keys=1, window=16) as eng:
        # Establish rolling_std ~ 1.0 so fallback sg = |2.0 - 0.0| / eff_scale lies in [1.5, 3.0)
        eng.step(0, 0, 0.0)
        eng.step(0, 1, math.sqrt(2.0))
        eng.publish_forecast(
            0,
            1,
            [0.0] * 9,
            [10.0] * 9,
            scale_hint=1e-300,
            knots=[6.0, 3.0, 1.5, 0.75],
            w_fast=[0.0, 0.25, 0.5, 0.75, 1.0],
            gain=[0.0] * 5,
        )
        r_fb = eng.step(0, 2, 2.0)
        assert r_fb.pwl_path == 2
        assert abs(r_fb.q9[4] - 5.0) < 1e-12
        assert abs(r_fb.eff_scale - r_fb.rolling_std) < 1e-12 and r_fb.eff_scale > 0.5

    # 3b. PWL innovation shift must NOT double-count (q_slow - q_fast) on large surprise
    with PwlArrivalEngine(n_keys=1, window=16) as eng:
        eng.publish_forecast(0, 0, [42.0] * 9, [5.0] * 9, scale_hint=1.0)
        r_nodbl = eng.step(0, 1, 5.0)
        assert abs(r_nodbl.q9[4] - 5.0) < 1e-9

    # 4. Extreme finite x=1e308 with med=-1e308 and gain[4]=0.0 must NEVER produce 0*Inf NaN
    with PwlArrivalEngine(n_keys=1, window=16) as eng:
        eng.publish_forecast(
            0,
            0,
            [-1e308] * 9,
            [-1e308] * 9,
            scale_hint=1.0,
            knots=[0.75, 1.5, 3.0, 6.0],
            w_fast=[0.1, 0.25, 0.5, 0.75, 0.9],
            gain=[0.1, 0.2, 0.3, 0.4, 0.0],
        )
        r_ext = eng.step(0, 1, 1e308)
        assert all(math.isfinite(v) for v in r_ext.q9)

    # 5. Nonzero-mean target 10,000-sigma glitch must NOT shrink weights toward 0
    sh = lib.tsjit_schur_create(2, 1, 0.98, 1e-4, 20.0)
    assert sh
    try:
        rng = random.Random(1)
        c_x = ctypes.c_double * 2
        c_y = ctypes.c_double * 1
        for _ in range(100):
            x1 = rng.gauss(0.0, 1.0)
            y = 100.0 + 2.0 * x1 + rng.gauss(0.0, 0.05)
            assert lib.tsjit_schur_update(sh, c_x(1.0, x1), c_y(y), None, None) == TSJIT_OK
        pb = ctypes.c_double()
        assert lib.tsjit_schur_predict(sh, c_x(1.0, 1.0), ctypes.byref(pb)) == TSJIT_OK
        assert lib.tsjit_schur_update(sh, c_x(1.0, 1.0), c_y(10_000.0), None, None) == TSJIT_OK
        pa = ctypes.c_double()
        assert lib.tsjit_schur_predict(sh, c_x(1.0, 1.0), ctypes.byref(pa)) == TSJIT_OK
        assert abs(pa.value - pb.value) < 0.05
    finally:
        lib.tsjit_schur_destroy(sh)

    # 6. Indefinite widely-scaled diagonal diag(-1, 1e-16) solved accurately by GE fallback
    A = (ctypes.c_double * 4)(-1.0, 0.0, 0.0, 1e-16)
    b = (ctypes.c_double * 2)(-1.0, 1e-16)
    x_out = (ctypes.c_double * 2)()
    method = ctypes.c_int32()
    assert (
        lib.tsjit_schur_solve_system(2, A, b, x_out, ctypes.byref(method), None, None) == TSJIT_OK
    )
    assert abs(x_out[0] - 1.0) < 1e-14 and abs(x_out[1] - 1.0) < 1e-14

    # 7. EWMA monoid extreme mean delta (-1e308 vs +1e308), nA+nB overflow, & tree validation
    def mk(n: int, w: float, m: list[float], c: list[float]):
        ma = (ctypes.c_double * len(m))(*m)
        ca = (ctypes.c_double * len(c))(*c)
        return CEwmaMonoid(n=n, dim=len(m), _pad=0, w=w, mean=ma, cov=ca), ma, ca

    A1, ma1, ca1 = mk(10, 10.0, [-1e308], [0.0])
    B1, mb1, cb1 = mk(10, 10.0, [1e308], [0.0])
    O1, mo1, co1 = mk(0, 0.0, [0.0], [0.0])
    assert (
        lib.tsjit_ewma_monoid_merge(1.0, ctypes.byref(A1), ctypes.byref(B1), ctypes.byref(O1))
        == TSJIT_OK
    )
    assert math.isfinite(mo1[0]) and abs(mo1[0]) < 1e295 and math.isfinite(co1[0])

    A_ovf, _, _ = mk((1 << 64) - 5, 1.0, [1.0], [0.0])
    B_ovf, _, _ = mk(10, 1.0, [2.0], [0.0])
    assert (
        lib.tsjit_ewma_monoid_merge(1.0, ctypes.byref(A_ovf), ctypes.byref(B_ovf), ctypes.byref(O1))
        == TSJIT_ERR_OVERFLOW
    )

    blk1 = (CEwmaMonoid * 1)(A1)
    assert (
        lib.tsjit_ewma_monoid_merge_tree(-1.0, blk1, 1, ctypes.byref(O1)) == TSJIT_ERR_BAD_ARG
    )


def test_wave45_adversarial_breaker_c11_regressions() -> None:
    """Permanent regression suite for Wave 4.5 C1-C4 and I6.1-I6.5 C11 breaker findings."""
    lib = load_native_lib()

    # C1: Signed int32 overflow in tsjit_schur_solve_shared_core (k_core=INT32_MAX, k_aug=1)
    dummy = (ctypes.c_double * 4)(1.0, 0.0, 0.0, 1.0)
    assert (
        lib.tsjit_schur_solve_shared_core(
            0x7FFFFFFF, 1, 1, dummy, dummy, dummy, dummy, dummy, dummy, None, None, None
        )
        == TSJIT_ERR_BAD_ARG
    )
    assert (
        lib.tsjit_schur_solve_shared_core(
            2049, 0, 1, dummy, None, None, dummy, None, dummy, None, None, None
        )
        == TSJIT_ERR_BAD_ARG
    )

    # I6.5: Higham guard caps n <= 2048
    allow = ctypes.c_int32(-1)
    assert (
        lib.tsjit_schur_check_higham_guard(2049, dummy, ctypes.byref(allow), None, None)
        == TSJIT_ERR_BAD_ARG
    )
    assert (
        lib.tsjit_schur_check_higham_guard(0x7FFFFFFF, dummy, ctypes.byref(allow), None, None)
        == TSJIT_ERR_BAD_ARG
    )

    # C2: Extreme finite rolling mean/std in BOTH tsjit_arrive.c AND tsjit_arena.c
    # (sample std uses Bessel denominator n - 1, matching _two_pass_mean_std!)
    extreme_cases = [
        ([-1.5e308, 1.5e308, 1.5e308], 5e307, math.sqrt(3.0) * 1e308),
        ([1.5e308, 1.5e308, -1.5e308], 5e307, math.sqrt(3.0) * 1e308),
        ([1e308, 1e308, 1e308, 0.0], 7.5e307, 5.0e307),
        ([-1e308, 1e308], 0.0, math.sqrt(2.0) * 1e308),
    ]
    for seq, exp_mean, exp_std in extreme_cases:
        with PwlArrivalEngine(n_keys=1, window=8) as eng:
            for t, xv in enumerate(seq):
                res = eng.step(0, t, xv)
                assert all(math.isfinite(q) for q in res.q9)
            st = eng.get_rolling(0)
            assert math.isfinite(st["mean"]) and math.isfinite(st["std"])
            tol_m = max(abs(exp_mean) * 1e-12, 1e295)
            tol_s = max(abs(exp_std) * 1e-12, 1e295)
            assert abs(st["mean"] - exp_mean) <= tol_m, f"arrive mean {st['mean']} != {exp_mean}"
            assert abs(st["std"] - exp_std) <= tol_s, f"arrive std {st['std']} != {exp_std}"

        arena = lib.tsjit_arena_create(1, 8, 4)
        assert arena
        try:
            dom = (ctypes.c_int32 * 1)(0)
            aid = ctypes.c_int32(-1)
            assert lib.tsjit_arena_add_agent(arena, dom, 1, 1.0, ctypes.byref(aid)) == TSJIT_OK
            for t, xv in enumerate(seq):
                c_v = (ctypes.c_double * 1)(xv)
                assert lib.tsjit_arena_ingest_tick(arena, t, c_v, 1) == TSJIT_OK
            am = ctypes.c_double()
            asd = ctypes.c_double()
            aanch = ctypes.c_double()
            acnt = ctypes.c_uint64()
            assert (
                lib.tsjit_arena_get_channel_stats(
                    arena,
                    0,
                    ctypes.byref(am),
                    ctypes.byref(asd),
                    ctypes.byref(aanch),
                    ctypes.byref(acnt),
                )
                == TSJIT_OK
            )
            assert math.isfinite(am.value) and math.isfinite(asd.value)
            assert abs(am.value - exp_mean) <= max(abs(exp_mean) * 1e-12, 1e295)
            assert abs(asd.value - exp_std) <= max(abs(exp_std) * 1e-12, 1e295)

            # Warm-starting a new agent after 1e308 ticks must produce finite mean/cov
            aid2 = ctypes.c_int32(-1)
            assert lib.tsjit_arena_add_agent(arena, dom, 1, 0.9, ctypes.byref(aid2)) == TSJIT_OK
            cm = (ctypes.c_double * 1)()
            cc = (ctypes.c_double * 1)()
            cn = ctypes.c_uint64()
            cg = ctypes.c_int32()
            assert (
                lib.tsjit_arena_get_agent_cov(
                    arena, aid2.value, 1, cm, cc, ctypes.byref(cn), ctypes.byref(cg)
                )
                == TSJIT_OK
            )
            assert math.isfinite(cm[0]) and math.isfinite(cc[0])
        finally:
            lib.tsjit_arena_destroy(arena)

    # Evicting 1e308 back to normal data restores exact unscaled mean/std
    with PwlArrivalEngine(n_keys=1, window=4) as eng:
        for t, xv in enumerate([1e308, -1e308, 1.0, 2.0, 3.0, 4.0]):
            eng.step(0, t, xv)
        st = eng.get_rolling(0)
        assert abs(st["mean"] - 2.5) < 1e-12
        assert abs(st["std"] - math.sqrt(5.0 / 3.0)) < 1e-12

    # Tiny/subnormal rolling stats [1e-200, 2e-200, 3e-200] & [1.5e308, 0.0] in arrive & arena
    with PwlArrivalEngine(n_keys=1, window=4) as eng:
        for t, xv in enumerate([1e-200, 2e-200, 3e-200]):
            eng.step(0, t, xv)
        st_sub = eng.get_rolling(0)
        assert abs(st_sub["mean"] - 2e-200) < 1e-212
        assert abs(st_sub["std"] - 1e-200) < 1e-212

    arena_sub = lib.tsjit_arena_create(1, 4, 4)
    assert arena_sub
    try:
        for t, xv in enumerate([1e-200, 2e-200, 3e-200]):
            assert lib.tsjit_arena_ingest_tick(arena_sub, t, (ctypes.c_double * 1)(xv), 1) == 0
        am = ctypes.c_double()
        asd = ctypes.c_double()
        assert (
            lib.tsjit_arena_get_channel_stats(
                arena_sub, 0, ctypes.byref(am), ctypes.byref(asd), None, None
            )
            == TSJIT_OK
        )
        assert abs(am.value - 2e-200) < 1e-212
        assert abs(asd.value - 1e-200) < 1e-212
    finally:
        lib.tsjit_arena_destroy(arena_sub)

    # C3: SoftRevisionV2 NaN q_slow sanitization, NaN loss skipping, all-invalid weight=0,
    # and -1e308 loss + tau_family=1e-310 overflow guard
    with SoftRevisionEngine(n_keys=2, window=16, lambda_age=0.2, max_age=8) as eng:
        # (a) NaN inside published q_slow falls back cleanly to finite causal baseline
        bad_q = [float("nan")] * 9
        good_q = [-1.28, -0.84, -0.52, -0.25, 0.0, 0.25, 0.52, 0.84, 1.28]
        eng.publish_forecast(0, 0, bad_q, good_q, scale_hint=1.0)
        r_nan_slot = eng.step_soft(0, 1, 2.5)
        assert all(math.isfinite(v) for v in r_nan_slot.q9)
        assert all(math.isfinite(w) for w in r_nan_slot.family_weights)

        # (b) NaN loss candidate is skipped (NEVER mapped to loss=0.0!)
        q_good = [10.0 + 0.1 * i for i in range(9)]
        q_poison = [9999.0 + 0.1 * i for i in range(9)]
        r_skip = eng.step_soft(
            0,
            2,
            10.0,
            slow_models=[q_good, q_poison],
            slow_losses=[10.0, float("nan")],
            fast_models=[q_good],
            fast_losses=[10.0],
            persist_models=[q_good],
            persist_losses=[10.0],
            beta_surprise=0.0,
        )
        assert all(math.isfinite(v) for v in r_skip.q9)
        assert abs(r_skip.q9[4] - 10.4) < 1e-6

        # All candidates in slow family invalid -> family_weights[0] == 0.0
        r_inv_fam = eng.step_soft(
            0,
            3,
            10.0,
            slow_models=[q_poison, [float("nan")] * 9],
            slow_losses=[float("nan"), 0.0],
            fast_models=[q_good],
            fast_losses=[1.0],
            persist_models=[q_good],
            persist_losses=[1.0],
            beta_surprise=0.0,
        )
        assert r_inv_fam.family_weights[0] == 0.0
        assert abs((r_inv_fam.family_weights[1] + r_inv_fam.family_weights[2]) - 1.0) < 1e-12
        assert all(math.isfinite(v) for v in r_inv_fam.q9)

        # (c) -1e308 losses + tiny tau_family=1e-310 must NEVER yield Inf - Inf = NaN
        r_ext_loss = eng.step_soft(
            0,
            4,
            10.0,
            slow_models=[q_good],
            slow_losses=[-1e308],
            fast_models=[q_good],
            fast_losses=[-1e308],
            persist_models=[q_good],
            persist_losses=[-1e308],
            tau_intra=1e-310,
            tau_family=1e-310,
            beta_surprise=0.0,
        )
        assert all(math.isfinite(v) for v in r_ext_loss.q9)
        assert all(math.isfinite(w) for w in r_ext_loss.family_weights)
        assert abs(sum(r_ext_loss.family_weights) - 1.0) < 1e-12

    # I6.3: tsjit_arrive_step_pwl honors lambda_age decay on stale forecasts
    with PwlArrivalEngine(n_keys=1, window=16, lambda_age=0.0, max_age=100) as eng_nod:
        with PwlArrivalEngine(n_keys=1, window=16, lambda_age=0.5, max_age=100) as eng_dec:
            q_far = [100.0 + 0.1 * i for i in range(9)]
            eng_nod.publish_forecast(0, 0, q_far, q_far, scale_hint=1.0)
            eng_dec.publish_forecast(0, 0, q_far, q_far, scale_hint=1.0)
            # Step at age=40 with x=0.0: decayed engine blends toward persistence centered at 0.0
            eng_nod.step(0, 1, 0.0)
            eng_dec.step(0, 1, 0.0)
            r_nod = eng_nod.step(0, 40, 0.0)
            r_dec = eng_dec.step(0, 40, 0.0)
            assert r_nod.family_weights[2] == 0.0
            assert r_dec.family_weights[2] > 0.999
            assert abs(r_dec.q9[4] - 0.0) < 1e-3

    # C4: EWMA monoid strict input validation, dim > 4096 guard, and w_out overflow guard
    def mk_m(n: int, w: float, m: list[float], c: list[float]):
        ma = (ctypes.c_double * len(m))(*m)
        ca = (ctypes.c_double * len(c))(*c)
        return CEwmaMonoid(n=n, dim=len(m), _pad=0, w=w, mean=ma, cov=ca), ma, ca

    Ok1, _, _ = mk_m(5, 5.0, [1.0], [0.25])
    Out1, mo1, co1 = mk_m(0, 0.0, [0.0], [0.0])

    # NaN cov rejected (never turned into DBL_MAX), even for n==0, alpha_exp<-1080, or n_blocks==1
    NanCov, _, _ = mk_m(5, 5.0, [1.0], [float("nan")])
    assert (
        lib.tsjit_ewma_monoid_merge(
            0.95, ctypes.byref(Ok1), ctypes.byref(NanCov), ctypes.byref(Out1)
        )
        == TSJIT_ERR_BAD_ARG
    )
    NanCovZero, _, _ = mk_m(0, 0.0, [1.0], [float("nan")])
    assert (
        lib.tsjit_ewma_monoid_merge(
            0.95, ctypes.byref(NanCovZero), ctypes.byref(Ok1), ctypes.byref(Out1)
        )
        == TSJIT_ERR_BAD_ARG
    )
    NanCovHuge, _, _ = mk_m(2000, 2.0, [1.0], [float("nan")])
    assert (
        lib.tsjit_ewma_monoid_merge(
            0.5, ctypes.byref(NanCovHuge), ctypes.byref(Ok1), ctypes.byref(Out1)
        )
        == TSJIT_ERR_BAD_ARG
    )
    NegWZero, _, _ = mk_m(0, -1.0, [1.0], [0.0])
    assert (
        lib.tsjit_ewma_monoid_merge(
            0.95, ctypes.byref(NegWZero), ctypes.byref(Ok1), ctypes.byref(Out1)
        )
        == TSJIT_ERR_BAD_ARG
    )
    blk_bad = (CEwmaMonoid * 1)(NanCov)
    assert (
        lib.tsjit_ewma_monoid_merge_tree(0.95, blk_bad, 1, ctypes.byref(Out1)) == TSJIT_ERR_BAD_ARG
    )
    blk_negw = (CEwmaMonoid * 1)(NegWZero)
    assert (
        lib.tsjit_ewma_monoid_merge_tree(0.95, blk_negw, 1, ctypes.byref(Out1)) == TSJIT_ERR_BAD_ARG
    )

    # dim == 0, dim == 4097, and dim == 0x80000000 rejected cleanly
    d1 = (ctypes.c_double * 1)(1.0)
    for bad_dim in (0, 4097, 0x80000000):
        BadDimA = CEwmaMonoid(n=1, dim=bad_dim, _pad=0, w=1.0, mean=d1, cov=d1)
        BadDimB = CEwmaMonoid(n=1, dim=bad_dim, _pad=0, w=1.0, mean=d1, cov=d1)
        BadDimO = CEwmaMonoid(n=0, dim=bad_dim, _pad=0, w=0.0, mean=d1, cov=d1)
        assert (
            lib.tsjit_ewma_monoid_merge(
                0.95, ctypes.byref(BadDimA), ctypes.byref(BadDimB), ctypes.byref(BadDimO)
            )
            == TSJIT_ERR_BAD_ARG
        )

    # w_out overflow (DBL_MAX + DBL_MAX, gamma=1.0) -> TSJIT_ERR_OVERFLOW (never writes +Inf!)
    MaxA, _, _ = mk_m(10, 1.7976931348623157e308, [1.0], [0.5])
    MaxB, _, _ = mk_m(10, 1.7976931348623157e308, [2.0], [0.5])
    OutW, mow, cow = mk_m(0, 0.0, [0.0], [0.0])
    assert (
        lib.tsjit_ewma_monoid_merge(1.0, ctypes.byref(MaxA), ctypes.byref(MaxB), ctypes.byref(OutW))
        == TSJIT_ERR_OVERFLOW
    )
    assert OutW.w == 0.0 and math.isfinite(OutW.w)

    # I6.5: init_agent_cap > 65536 rejected; publish/read on tombstoned agent -> TSJIT_ERR_BAD_ARG
    assert not lib.tsjit_arena_create(4, 32, 65537)
    assert not lib.tsjit_arena_create(4, 32, 0xFFFFFFFF)
    arena_t = lib.tsjit_arena_create(2, 16, 4)
    assert arena_t
    try:
        dom = (ctypes.c_int32 * 2)(0, 1)
        aid = ctypes.c_int32(-1)
        assert lib.tsjit_arena_add_agent(arena_t, dom, 2, 0.95, ctypes.byref(aid)) == TSJIT_OK
        q9 = (ctypes.c_double * 9)(*range(9))
        assert (
            lib.tsjit_arena_publish_agent_output(arena_t, aid.value, 1, q9, 0.1, MARK_OK)
            == TSJIT_OK
        )
        assert lib.tsjit_arena_tombstone_agent(arena_t, aid.value) == TSJIT_OK
        assert lib.tsjit_arena_agent_status(arena_t, aid.value) == 2
        assert (
            lib.tsjit_arena_publish_agent_output(arena_t, aid.value, 2, q9, 0.1, MARK_OK)
            == TSJIT_ERR_BAD_ARG
        )
        q_rd = (ctypes.c_double * 9)()
        assert (
            lib.tsjit_arena_read_agent_snapshot(arena_t, aid.value, q_rd, None, None, None, None)
            == TSJIT_ERR_BAD_ARG
        )
    finally:
        lib.tsjit_arena_destroy(arena_t)

    # I6.1: Schur update extreme tick-0 (1e200) and gamma=1.0 x=[1e153, 1.0] never overflow G
    sh1 = lib.tsjit_schur_create(2, 1, 0.95, 1e-3, 20.0)
    assert sh1
    try:
        c_x2 = ctypes.c_double * 2
        c_y1 = ctypes.c_double * 1
        assert lib.tsjit_schur_update(sh1, c_x2(1e200, 1.0), c_y1(1e200), None, None) == TSJIT_OK
        py = ctypes.c_double(0.0)
        assert lib.tsjit_schur_predict(sh1, c_x2(1.0, 1.0), ctypes.byref(py)) == TSJIT_OK
        assert math.isfinite(py.value)
    finally:
        lib.tsjit_schur_destroy(sh1)

    sh2 = lib.tsjit_schur_create(2, 1, 1.0, 1e-3, 20.0)
    assert sh2
    try:
        c_x2 = ctypes.c_double * 2
        c_y1 = ctypes.c_double * 1
        for _ in range(10):
            assert lib.tsjit_schur_update(sh2, c_x2(1e153, 1.0), c_y1(1.0), None, None) == TSJIT_OK
        py2 = ctypes.c_double(0.0)
        assert lib.tsjit_schur_predict(sh2, c_x2(1.0, 1.0), ctypes.byref(py2)) == TSJIT_OK
        assert math.isfinite(py2.value)
    finally:
        lib.tsjit_schur_destroy(sh2)

    # I6.4: Subnormal row equilibration (-1e-310 and DBL_TRUE_MIN=5e-324) via frexp/ldexp
    for tiny in (-1e-310, -5e-324):
        A_sub = (ctypes.c_double * 4)(tiny, 0.0, 0.0, 1.0)
        b_sub = (ctypes.c_double * 2)(tiny, 1.0)
        x_sub = (ctypes.c_double * 2)()
        meth = ctypes.c_int32(0)
        assert (
            lib.tsjit_schur_solve_system(2, A_sub, b_sub, x_sub, ctypes.byref(meth), None, None)
            == TSJIT_OK
        )
        assert abs(x_sub[0] - 1.0) < 1e-12 and abs(x_sub[1] - 1.0) < 1e-12

    # C3 extra: tsjit_arrive_step_soft_v2 rejects negative model counts or NULL qs with count > 0
    eng_raw = lib.tsjit_arrive_create(1, 8, 0.1, 8)
    assert eng_raw
    try:
        from tsjit.native.build import CArriveOut

        out_raw = CArriveOut()
        assert (
            lib.tsjit_arrive_step_soft_v2(
                eng_raw,
                0,
                0,
                1.0,
                -1,
                None,
                None,
                0,
                None,
                None,
                0,
                None,
                None,
                1.0,
                1.0,
                0.0,
                ctypes.byref(out_raw),
            )
            == TSJIT_ERR_BAD_ARG
        )
        assert (
            lib.tsjit_arrive_step_soft_v2(
                eng_raw,
                0,
                0,
                1.0,
                2,
                None,
                None,
                0,
                None,
                None,
                0,
                None,
                None,
                1.0,
                1.0,
                0.0,
                ctypes.byref(out_raw),
            )
            == TSJIT_ERR_BAD_ARG
        )
    finally:
        lib.tsjit_arrive_destroy(eng_raw)


def test_c11_whitebox_spin_timeout_and_arena_overflow_recovery() -> None:
    """Deterministic ASan+UBSan C11 white-box verification of I6.3 spin-timeout rollback
    and I6.2 arena covariance group recovery after TSJIT_ERR_OVERFLOW / NaN corruption.
    """
    from tsjit.native.build import NATIVE_DIR, find_c_compiler

    whitebox_code = r"""
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include <assert.h>
#include "tsjit_arrive.c"
#include "tsjit_arena.c"

int main(void) {
    /* 1. I6.3: Deterministic TSJIT_ERR_SPIN_TIMEOUT leaves rolling state untouched */
    tsjit_arrive_engine_t *eng = tsjit_arrive_create(1, 16, 0.2, 8);
    assert(eng != NULL);
    eng->max_spins = 128;
    double q0[9] = {-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0};
    assert(tsjit_arrive_publish_forecast(eng, 0, 0, q0, q0, 1.0, NULL, NULL, NULL) == TSJIT_OK);

    tsjit_arrive_out_t out;
    assert(tsjit_arrive_step_pwl(eng, 0, 1, 0.1, &out) == TSJIT_OK);
    assert(eng->keys[0].count == 1 && eng->keys[0].last_step == 1);

    /* Pin hot slot seqlock ODD to force deterministic TSJIT_ERR_SPIN_TIMEOUT */
    uint64_t saved_seq = atomic_load(&eng->hot_slots[0].seq);
    atomic_store(&eng->hot_slots[0].seq, saved_seq + 1ULL);

    /* Hot-path step, cold-path step, and soft_v2 step MUST timeout with ZERO rolling mutation */
    assert(tsjit_arrive_step_pwl(eng, 0, 2, 0.2, &out) == TSJIT_ERR_SPIN_TIMEOUT);
    assert(eng->keys[0].count == 1 && eng->keys[0].last_step == 1);
    assert(tsjit_arrive_step_pwl(eng, 0, 2, 50.0, &out) == TSJIT_ERR_SPIN_TIMEOUT);
    assert(eng->keys[0].count == 1 && eng->keys[0].last_step == 1);
    assert(tsjit_arrive_step_soft_v2(
        eng, 0, 2, 0.2, 0, NULL, NULL, 0, NULL, NULL, 0, NULL, NULL, 1.0, 1.0, 0.1, &out)
        == TSJIT_ERR_SPIN_TIMEOUT);
    assert(eng->keys[0].count == 1 && eng->keys[0].last_step == 1);

    /* Unpin seqlock: step 2 must now be accepted as MARK_OK (NOT MARK_DUP_STEP!) */
    atomic_store(&eng->hot_slots[0].seq, saved_seq);
    assert(tsjit_arrive_step_pwl(eng, 0, 2, 50.0, &out) == TSJIT_OK);
    assert(out.mark == MARK_OK && eng->keys[0].count == 2 && eng->keys[0].last_step == 2);
    tsjit_arrive_destroy(eng);

    /* 2. I6.2: Arena covariance group recovers cleanly from TSJIT_ERR_OVERFLOW & NaN */
    tsjit_arena_t *arena = tsjit_arena_create(1, 8, 4);
    assert(arena != NULL);
    int32_t dom[1] = {0};
    int32_t aid = -1;
    assert(tsjit_arena_add_agent(arena, dom, 1, 0.95, &aid) == TSJIT_OK);
    double v1[1] = {10.0};
    assert(tsjit_arena_ingest_tick(arena, 1, v1, 1) == TSJIT_OK);

    /* Case (a): n == UINT64_MAX triggers TSJIT_ERR_OVERFLOW -> group must NOT freeze */
    arena->groups[0].n = UINT64_MAX;
    double v2[1] = {50.0};
    assert(tsjit_arena_ingest_tick(arena, 2, v2, 1) == TSJIT_OK);
    assert(arena->groups[0].n < UINT64_MAX);
    assert(isfinite(arena->groups[0].mean[0]) && arena->groups[0].mean[0] > 10.0);

    /* Case (b): NaN covariance triggers TSJIT_ERR_BAD_ARG -> group re-anchors cleanly */
    arena->groups[0].cov[0] = NAN;
    double v3[1] = {75.0};
    assert(tsjit_arena_ingest_tick(arena, 3, v3, 1) == TSJIT_OK);
    assert(arena->groups[0].n == 1ULL && arena->groups[0].w == 1.0);
    assert(arena->groups[0].mean[0] == 75.0 && arena->groups[0].cov[0] == 0.0);
    tsjit_arena_destroy(arena);

    return 0;
}
"""
    cc = find_c_compiler()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        src = tmp_dir / "whitebox.c"
        exe = tmp_dir / "whitebox"
        src.write_text(whitebox_code, encoding="utf-8")
        cmd = [
            cc,
            "-std=c11",
            "-O1",
            "-g",
            "-fno-omit-frame-pointer",
            "-fsanitize=address,undefined",
            "-fno-sanitize-recover=all",
            f"-I{NATIVE_DIR}",
            str(src),
            "-o",
            str(exe),
            "-pthread",
        ]
        import sys as _sys

        if _sys.platform != "darwin":
            cmd.append("-lm")
        comp = subprocess.run(cmd, capture_output=True, text=True)
        assert comp.returncode == 0, f"Whitebox build failed:\n{comp.stderr}"
        run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30)
        assert run.returncode == 0, f"Whitebox run failed:\n{run.stdout}\n{run.stderr}"
        assert "ERROR:" not in run.stderr


def test_c11_asan_ubsan_and_tsan_clean() -> None:
    """Compile and execute a standalone C11 stress harness under ASan+UBSan and TSan."""
    harness_code = r"""
#include "tsjit_common.h"
#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>

typedef struct {
    tsjit_schur_state_t *sh;
    _Atomic int stop;
} schur_race_ctx_t;

static void *schur_pred_worker(void *p) {
    schur_race_ctx_t *c = (schur_race_ctx_t *)p;
    double x[4] = {1.0, 0.5, -0.5, 0.2};
    double y[1] = {0.0};
    while (!atomic_load_explicit(&c->stop, memory_order_acquire)) {
        assert(tsjit_schur_predict(c->sh, x, y) == TSJIT_OK);
    }
    return NULL;
}

typedef struct {
    tsjit_arrive_engine_t *eng;
    _Atomic int stop;
} arr_race_ctx_t;

static void *arr_pub_worker(void *p) {
    arr_race_ctx_t *c = (arr_race_ctx_t *)p;
    double qs[9] = {-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0};
    double qf[9] = {-1.0,-0.8,-0.5,-0.2,0.1,0.3,0.6,0.9,1.2};
    for (int i = 0; i < 2000 && !atomic_load_explicit(&c->stop, memory_order_acquire); ++i) {
        double sc = (i % 3 == 0) ? 1e-300 : 1.0;
        int32_t rc = tsjit_arrive_publish_forecast(
            c->eng, 0, i, qs, qf, sc, NULL, NULL, NULL);
        assert(rc == TSJIT_OK);
    }
    return NULL;
}

static void *arr_telem_worker(void *p) {
    arr_race_ctx_t *c = (arr_race_ctx_t *)p;
    tsjit_telemetry_t tm;
    while (!atomic_load_explicit(&c->stop, memory_order_acquire)) {
        assert(tsjit_read_telemetry(c->eng, &tm) == TSJIT_OK);
    }
    return NULL;
}

int main(void) {
    /* 1. Bounds, safe age, and sort9 with NaN */
    assert(tsjit_check_bounds_i64(INT64_MAX, 1, INT64_MAX) == TSJIT_ERR_OOB);
    assert(tsjit_safe_age_i64_abi(INT64_MAX, INT64_MIN) == INT64_MAX);
    double arr[9] = {5.0, NAN, 1.0, 3.0, -2.0, 4.0, 0.0, 2.0, 6.0};
    tsjit_sort9_abi(arr);
    int nans = 0;
    for (int i = 0; i < 9; ++i) if (isnan(arr[i])) nans++;
    assert(nans == 1 && isnan(arr[8]));

    /* 2. Concurrent multi-writer PWL publish (hot/cold/fallback) + step_pwl + telemetry */
    tsjit_arrive_engine_t *eng = tsjit_arrive_create(2, 16, 0.2, 8);
    assert(eng != NULL);
    arr_race_ctx_t actx = {eng, 0};
    pthread_t pw1, pw2, tw;
    assert(pthread_create(&pw1, NULL, arr_pub_worker, &actx) == 0);
    assert(pthread_create(&pw2, NULL, arr_pub_worker, &actx) == 0);
    assert(pthread_create(&tw, NULL, arr_telem_worker, &actx) == 0);

    tsjit_arrive_out_t out;
    for (int t = 1; t <= 1000; ++t) {
        double v = (t == 5) ? 1e20 : ((t % 2 == 0) ? 0.1 : 5.0);
        assert(tsjit_arrive_step_pwl(eng, 0, t, v, &out) == TSJIT_OK);
        for (int q = 0; q < 9; ++q) assert(isfinite(out.q[q]));
        if (t == 500) assert(tsjit_arrive_set_params(eng, 0.5, 4) == TSJIT_OK);
    }
    assert(tsjit_arrive_step_pwl(eng, 0, 1000, 999.0, &out) == TSJIT_OK &&
           out.mark == MARK_DUP_STEP);
    assert(tsjit_arrive_step_pwl(eng, 0, 999, 999.0, &out) == TSJIT_OK &&
           out.mark == MARK_OOO_STEP);
    assert(tsjit_arrive_step_soft_v2(eng, 0, 1001, 0.2, 0, NULL, NULL, 0, NULL, NULL, 0, NULL, NULL,
                                     1.0, 1.0, 0.2, &out) == TSJIT_OK);
    atomic_store_explicit(&actx.stop, 1, memory_order_release);
    pthread_join(pw1, NULL);
    pthread_join(pw2, NULL);
    pthread_join(tw, NULL);
    tsjit_telemetry_t telem;
    assert(tsjit_read_telemetry(eng, &telem) == TSJIT_OK);
    assert(telem.dup_rejects == 1 && telem.ooo_rejects == 1 && telem.roll_refreshes >= 1);
    tsjit_arrive_destroy(eng);

    /* 3. Multi-writer/multi-reader seqlock stress (both seqlocked & unguarded atomic control) */
    uint64_t r0 = 0, torn0 = 0, to0 = 0;
    assert(tsjit_seqlock_stress_test(2, 3, 40000, 0, &r0, &torn0, &to0) == TSJIT_OK);
    assert(r0 == 120000 && torn0 == 0 && to0 == 0);
    uint64_t r1 = 0, torn1 = 0, to1 = 0;
    assert(tsjit_seqlock_stress_test(2, 3, 40000, 1, &r1, &torn1, &to1) == TSJIT_OK);
    assert(r1 == 120000);

    /* 4. Concurrent Schur update + predict + glitch Winsorization */
    tsjit_schur_state_t *sh = tsjit_schur_create(4, 1, 0.95, 1e-3, 20.0);
    assert(sh != NULL);
    schur_race_ctx_t sctx = {sh, 0};
    pthread_t sp_th;
    assert(pthread_create(&sp_th, NULL, schur_pred_worker, &sctx) == 0);
    for (int t = 0; t < 150; ++t) {
        double x[4] = {1.0, (double)(t % 3), (double)(t % 5), -0.5};
        double y[1] = {2.0 + x[1] - 0.5 * x[2]};
        double w = 0.0;
        assert(tsjit_schur_update(sh, x, y, NULL, &w) == TSJIT_OK);
    }
    double gx[4] = {1.0, 10000.0, -10000.0, 1.0};
    double gy[1] = {10000.0};
    double gsp[4] = {0.0, INFINITY, NAN, 1e300};
    double gw = 0.0;
    assert(tsjit_schur_update(sh, gx, gy, gsp, &gw) == TSJIT_OK);
    atomic_store_explicit(&sctx.stop, 1, memory_order_release);
    pthread_join(sp_th, NULL);
    double py[1] = {0.0};
    double px[4] = {1.0, 1.0, 1.0, -0.5};
    assert(tsjit_schur_predict(sh, px, py) == TSJIT_OK && isfinite(py[0]));
    tsjit_schur_destroy(sh);

    /* 5. EWMA extreme-decay monoid merge, huge nB UBSan guard, & in-place aliasing */
    double ma[2] = {1e160, -1.0}, ca[4] = {0, 0, 0, 0};
    double mb[2] = {2.0, 3.0}, cb[4] = {0.5, 0.1, 0.1, 0.5};
    tsjit_ewma_monoid_t A = {1, 2, 0, 1.0, ma, ca};
    tsjit_ewma_monoid_t B = {2000, 2, 0, 2.0, mb, cb};
    assert(tsjit_ewma_monoid_merge(0.5, &A, &B, &B) == TSJIT_OK);
    assert(isfinite(B.mean[0]) && B.mean[0] == 2.0 && B.cov[0] == 0.5);

    double mh1[1] = {1.0}, ch1[1] = {0.5}, mh2[1] = {2.0}, ch2[1] = {0.25};
    double mho[1] = {0.0}, cho[1] = {0.0};
    tsjit_ewma_monoid_t Ah = {1, 1, 0, 1.0, mh1, ch1};
    tsjit_ewma_monoid_t Bh = {UINT64_MAX - 10ULL, 1, 0, 2.0, mh2, ch2};
    tsjit_ewma_monoid_t Oh = {0, 1, 0, 0.0, mho, cho};
    assert(tsjit_ewma_monoid_merge(0.5, &Ah, &Bh, &Oh) == TSJIT_OK);
    assert(Oh.mean[0] == 2.0 && Oh.cov[0] == 0.25);

    /* 6. Concurrent EBR dynamic output buffer resizing (>120 resizes, 4 lock-free readers) */
    uint64_t ebr_reads = 0, ebr_rec = 0, ebr_err = 0;
    assert(tsjit_arena_ebr_stress_test(4, 120, &ebr_reads, &ebr_rec, &ebr_err) == TSJIT_OK);
    assert(ebr_err == 0 && ebr_reads > 0 && ebr_rec >= 110);

    /* 7. Wave 4.5 UBSan/ASan coverage: C1 overflow, C2 1e308 rolling, C3 -1e308/NaN soft_v2,
     * C4 dim wrap & w_out overflow, I6.1 1e200 schur, I6.4 subnormal GE */
    double d4[4] = {1.0, 0.0, 0.0, 1.0};
    assert(tsjit_schur_solve_shared_core(INT32_MAX, 1, 1, d4, d4, d4, d4, d4, d4,
                                         NULL, NULL, NULL) == TSJIT_ERR_BAD_ARG);
    int32_t allow_hi = 0;
    assert(tsjit_schur_check_higham_guard(2049, d4, &allow_hi, NULL, NULL) == TSJIT_ERR_BAD_ARG);

    double subA[4] = {-1e-310, 0.0, 0.0, 1.0};
    double subB[2] = {-1e-310, 1.0};
    double subX[2] = {0.0, 0.0};
    assert(tsjit_schur_solve_system(2, subA, subB, subX, NULL, NULL, NULL) == TSJIT_OK);
    assert(fabs(subX[0] - 1.0) < 1e-12 && fabs(subX[1] - 1.0) < 1e-12);

    tsjit_schur_state_t *sh_ext = tsjit_schur_create(2, 1, 1.0, 1e-3, 20.0);
    assert(sh_ext != NULL);
    double sx2[2] = {1e200, 1.0}, sy1[1] = {1e200}, spy[1] = {0.0};
    assert(tsjit_schur_update(sh_ext, sx2, sy1, NULL, NULL) == TSJIT_OK);
    assert(tsjit_schur_predict(sh_ext, subB, spy) == TSJIT_OK && isfinite(spy[0]));
    tsjit_schur_destroy(sh_ext);

    tsjit_arrive_engine_t *eng_ext = tsjit_arrive_create(1, 8, 0.5, 16);
    assert(eng_ext != NULL);
    tsjit_arrive_out_t oext;
    assert(tsjit_arrive_step_pwl(eng_ext, 0, 0, -1.5e308, &oext) == TSJIT_OK);
    assert(tsjit_arrive_step_pwl(eng_ext, 0, 1,  1.5e308, &oext) == TSJIT_OK);
    assert(tsjit_arrive_step_pwl(eng_ext, 0, 2,  1.5e308, &oext) == TSJIT_OK);
    assert(isfinite(oext.rolling_mean) && isfinite(oext.rolling_std));
    double qbad[9] = {NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN};
    assert(tsjit_arrive_publish_forecast(
        eng_ext, 0, 2, qbad, qbad, 1.0, NULL, NULL, NULL) == TSJIT_OK);
    double qok[9] = {1,2,3,4,5,6,7,8,9};
    double lneg[1] = {-1e308};
    assert(tsjit_arrive_step_soft_v2(eng_ext, 0, 3, 5.0, 1, qok, lneg, 1, qok, lneg,
                                     1, qok, lneg, 1e-310, 1e-310, 0.0, &oext) == TSJIT_OK);
    for (int q = 0; q < 9; ++q) assert(isfinite(oext.q[q]));
    tsjit_arrive_destroy(eng_ext);

    double xm1[1] = {1.0}, xc1[1] = {0.5}, xo1[1] = {0.0}, xco1[1] = {0.0};
    tsjit_ewma_monoid_t mA = {10, 1, 0, DBL_MAX, xm1, xc1};
    tsjit_ewma_monoid_t mB = {10, 1, 0, DBL_MAX, xm1, xc1};
    tsjit_ewma_monoid_t mO = {0,  1, 0, 0.0,     xo1, xco1};
    assert(tsjit_ewma_monoid_merge(1.0, &mA, &mB, &mO) == TSJIT_ERR_OVERFLOW);
    tsjit_ewma_monoid_t mWrap = {1, 0x80000000U, 0, 1.0, xm1, xc1};
    assert(tsjit_ewma_monoid_merge(0.95, &mWrap, &mWrap, &mWrap) == TSJIT_ERR_BAD_ARG);
    assert(tsjit_arena_create(4, 32, 65537U) == NULL);

    return 0;
}
"""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        harness = tmp_dir / "san_harness.c"
        harness.write_text(harness_code, encoding="utf-8")

        for san_mode in ("asan,ubsan", "tsan"):
            out_bin = tmp_dir / f"harness_{san_mode.replace(',', '_')}"
            build_c_executable(harness, out_bin, sanitizers=san_mode)
            run_cmd = [str(out_bin)]
            if san_mode == "tsan":
                import platform as _plat
                import shutil as _shutil
                import sys as _sys

                if _sys.platform == "linux":
                    setarch = _shutil.which("setarch")
                    if setarch:
                        run_cmd = [setarch, _plat.machine(), "-R", str(out_bin)]
            proc = subprocess.run(run_cmd, capture_output=True, text=True, timeout=60)
            assert proc.returncode == 0, (
                f"{san_mode} failed (rc={proc.returncode}):\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
            assert "ERROR:" not in proc.stderr and "WARNING: ThreadSanitizer" not in proc.stderr, (
                f"{san_mode} reported findings:\n{proc.stderr}"
            )
