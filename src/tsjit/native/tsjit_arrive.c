/*
 * Copyright 2026 TSJIT Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * Portable C11 arrival engine:
 *  - Hardened Boehm-2012 multi-writer CAS seqlock with _Atomic uint64_t payload
 *    words (zero C11 UB, cacheline-aligned per key, bounded lock-free reader)
 *  - Seqlocked telemetry snapshot
 *  - Causal float64 DC-offset stripping & shifted-anchor rolling statistics
 *    with exact 2-pass refresh on eviction shock, cumulative drift, or cancellation
 *  - Hot/cold PWL-ARRIVE engine (<= 640 B hot row, zero divides on hot path)
 *  - Age-aware hierarchical 3-family SoftRevisionV2 (equal family prior)
 */

#include "tsjit_common.h"

#include <pthread.h>
#include <sched.h>

/* Standard normal quantiles at tau = 0.1, 0.2, ..., 0.9 */
static const double TSJIT_Z9[TSJIT_NQ] = {
    -1.2815515655446004,
    -0.8416212335729143,
    -0.5244005127080407,
    -0.2533471031357997,
     0.0,
     0.2533471031357997,
     0.5244005127080407,
     0.8416212335729143,
     1.2815515655446004
};

#define TSJIT_PWL_FLAG_VALID    (1ULL << 0)
#define TSJIT_PWL_FLAG_FALLBACK (1ULL << 1)

/* Hot seqlock slot layout: 46 atomic u64 words + 1 seq word = 376 B -> padded to 384 B <= 640 B */
#define TSJIT_HOT_NWORDS 46

typedef struct tsjit_pwl_hot_slot {
    _Alignas(TSJIT_CACHELINE) _Atomic uint64_t seq;
    _Atomic uint64_t words[TSJIT_HOT_NWORDS];
    uint8_t _pad[TSJIT_CACHELINE - (((1 + TSJIT_HOT_NWORDS) * sizeof(uint64_t)) % TSJIT_CACHELINE)];
} tsjit_pwl_hot_slot_t;

_Static_assert(sizeof(tsjit_pwl_hot_slot_t) <= 640, "PWL hot row must be <= 640 bytes");
_Static_assert(sizeof(tsjit_pwl_hot_slot_t) % TSJIT_CACHELINE == 0,
               "PWL hot slot must be cacheline-multiple sized");

/*
 * Cold PWL & forecast payload (separate memory array from hot slots, synchronized
 * by the matching hot_slot->seq seqlock):
 *  - a_pos_cold[4][9] = 36
 *  - b_pos_cold[4][9] = 36
 *  - a_neg_cold[4][9] = 36
 *  - b_neg_cold[4][9] = 36
 *  - q_slow[9]        = 9
 *  - q_fast[9]        = 9
 *  - knots[4]         = 4
 *  - w_fast[5]        = 5
 *  - gain[5]          = 5
 * Total = 176 u64 words
 */
#define TSJIT_COLD_NWORDS 176

typedef struct tsjit_pwl_cold_slot {
    _Alignas(TSJIT_CACHELINE) _Atomic uint64_t words[TSJIT_COLD_NWORDS];
} tsjit_pwl_cold_slot_t;

#define TSJIT_TELEM_NWORDS 12

typedef struct tsjit_telemetry_slot {
    _Alignas(TSJIT_CACHELINE) _Atomic uint64_t seq;
    _Atomic uint64_t words[TSJIT_TELEM_NWORDS];
    uint8_t _pad[TSJIT_CACHELINE - (((1 + TSJIT_TELEM_NWORDS) * sizeof(uint64_t)) % TSJIT_CACHELINE)];
} tsjit_telemetry_slot_t;

_Static_assert(sizeof(tsjit_telemetry_slot_t) % TSJIT_CACHELINE == 0,
               "Telemetry slot must be cacheline-multiple sized");

typedef struct tsjit_roll_key {
    double  anchor;
    double  sum_d;
    double  sum_d2;
    double  evicted_d2_accum;
    int32_t scale_exp; /* 0 = unscaled; >0 = sum_d/sum_d2 scaled by 2^{-scale_exp} */
    double *ring;
    int64_t window;
    int64_t head;
    int64_t count;
    double  last_valid;
    int64_t last_step;
    int64_t last_valid_step;
    bool    has_step;
    bool    has_valid;
    double  cached_q[TSJIT_NQ];
    double  cached_abs_sg;
    double  cached_eff_sc;
    double  cached_fam_w[3];
    int32_t cached_pwl_path;
} tsjit_roll_key_t;

struct tsjit_arrive_engine {
    int64_t                 n_keys;
    int64_t                 window;
    _Atomic uint64_t        lambda_age_bits;
    _Atomic int64_t         max_age;
    uint64_t                max_spins;
    tsjit_pwl_hot_slot_t   *hot_slots;
    tsjit_pwl_cold_slot_t  *cold_slots;
    tsjit_roll_key_t       *keys;
    double                 *ring_arena;
    tsjit_telemetry_slot_t *telem;
};

/* ---------- Boehm-2012 Multi-Writer CAS Seqlock Helpers ---------- */

static inline void seqlock_backoff(uint64_t spins) {
    if ((spins & 1023ULL) == 0ULL) {
        sched_yield();
    } else {
        tsjit_cpu_relax();
    }
}

static inline uint64_t seqlock_write_begin(_Atomic uint64_t *seq) {
    uint64_t s = atomic_load_explicit(seq, memory_order_relaxed);
    uint64_t spins = 0;
    for (;;) {
        if (s & 1ULL) {
            seqlock_backoff(++spins);
            s = atomic_load_explicit(seq, memory_order_relaxed);
            continue;
        }
        if (atomic_compare_exchange_weak_explicit(
                seq, &s, s + 1ULL,
                memory_order_acq_rel, memory_order_relaxed)) {
            atomic_thread_fence(memory_order_release);
            return s;
        }
        seqlock_backoff(++spins);
    }
}

static inline void seqlock_write_end(_Atomic uint64_t *seq, uint64_t s_even) {
    atomic_store_explicit(seq, s_even + 2ULL, memory_order_release);
}

static inline int32_t seqlock_read_words(
    const _Atomic uint64_t *seq,
    const _Atomic uint64_t *src_words,
    size_t n_words,
    uint64_t *dst_words,
    uint64_t max_spins)
{
    uint64_t spins = 0;
    for (;;) {
        uint64_t s1 = atomic_load_explicit(seq, memory_order_acquire);
        if (TSJIT_UNLIKELY(s1 & 1ULL)) {
            if (++spins >= max_spins) {
                return TSJIT_ERR_SPIN_TIMEOUT;
            }
            seqlock_backoff(spins);
            continue;
        }
        for (size_t i = 0; i < n_words; ++i) {
            dst_words[i] = atomic_load_explicit(&src_words[i], memory_order_relaxed);
        }
        atomic_thread_fence(memory_order_acquire);
        uint64_t s2 = atomic_load_explicit(seq, memory_order_relaxed);
        if (TSJIT_LIKELY(s1 == s2)) {
            return TSJIT_OK;
        }
        if (++spins >= max_spins) {
            return TSJIT_ERR_SPIN_TIMEOUT;
        }
        seqlock_backoff(spins);
    }
}

static inline int32_t seqlock_read_hot_and_cold(
    const tsjit_pwl_hot_slot_t *hot,
    const tsjit_pwl_cold_slot_t *cold,
    uint64_t dst_hot[TSJIT_HOT_NWORDS],
    uint64_t dst_cold[TSJIT_COLD_NWORDS],
    uint64_t max_spins)
{
    uint64_t spins = 0;
    for (;;) {
        uint64_t s1 = atomic_load_explicit(&hot->seq, memory_order_acquire);
        if (TSJIT_UNLIKELY(s1 & 1ULL)) {
            if (++spins >= max_spins) {
                return TSJIT_ERR_SPIN_TIMEOUT;
            }
            seqlock_backoff(spins);
            continue;
        }
        for (size_t i = 0; i < TSJIT_HOT_NWORDS; ++i) {
            dst_hot[i] = atomic_load_explicit(&hot->words[i], memory_order_relaxed);
        }
        for (size_t i = 0; i < TSJIT_COLD_NWORDS; ++i) {
            dst_cold[i] = atomic_load_explicit(&cold->words[i], memory_order_relaxed);
        }
        atomic_thread_fence(memory_order_acquire);
        uint64_t s2 = atomic_load_explicit(&hot->seq, memory_order_relaxed);
        if (TSJIT_LIKELY(s1 == s2)) {
            return TSJIT_OK;
        }
        if (++spins >= max_spins) {
            return TSJIT_ERR_SPIN_TIMEOUT;
        }
        seqlock_backoff(spins);
    }
}

/* Telemetry atomic seqlocked update */
static void telem_record(
    tsjit_telemetry_slot_t *telem,
    uint64_t d_total,
    uint64_t d_valid,
    uint64_t d_dup,
    uint64_t d_ooo,
    uint64_t d_nan_stale,
    uint64_t d_pwl_fast,
    uint64_t d_pwl_cold,
    uint64_t d_pwl_fb,
    uint64_t d_refresh,
    bool update_last,
    int64_t last_step,
    int64_t last_key,
    double last_abs_sg)
{
    uint64_t s = seqlock_write_begin(&telem->seq);
    uint64_t w[TSJIT_TELEM_NWORDS];
    for (int i = 0; i < TSJIT_TELEM_NWORDS; ++i) {
        w[i] = atomic_load_explicit(&telem->words[i], memory_order_relaxed);
    }
    w[0] += d_total;
    w[1] += d_valid;
    w[2] += d_dup;
    w[3] += d_ooo;
    w[4] += d_nan_stale;
    w[5] += d_pwl_fast;
    w[6] += d_pwl_cold;
    w[7] += d_pwl_fb;
    w[8] += d_refresh;
    if (update_last) {
        w[9]  = (uint64_t)last_step;
        w[10] = (uint64_t)last_key;
        w[11] = tsjit_f64_to_u64(last_abs_sg);
    }
    for (int i = 0; i < TSJIT_TELEM_NWORDS; ++i) {
        atomic_store_explicit(&telem->words[i], w[i], memory_order_relaxed);
    }
    seqlock_write_end(&telem->seq, s);
}

/* ---------- Shifted-Anchor Rolling Stats with 2-Pass Refresh ---------- */

static void roll_two_pass_refresh(tsjit_roll_key_t *rk) {
    if (rk->count <= 0) {
        rk->anchor = 0.0;
        rk->sum_d = 0.0;
        rk->sum_d2 = 0.0;
        rk->evicted_d2_accum = 0.0;
        rk->scale_exp = 0;
        return;
    }
    /* Scan active ring entries first to detect constant window and extreme magnitudes */
    double v0 = rk->ring[0];
    bool all_equal = true;
    double max_abs_x = 0.0;
    for (int64_t i = 0; i < rk->count; ++i) {
        double v = rk->ring[i];
        if (v != v0) {
            all_equal = false;
        }
        double av = fabs(v);
        if (av > max_abs_x) {
            max_abs_x = av;
        }
    }
    if (all_equal) {
        rk->anchor = v0;
        rk->sum_d = 0.0;
        rk->sum_d2 = 0.0;
        rk->evicted_d2_accum = 0.0;
        rk->scale_exp = 0;
        return;
    }

    /* Extreme-magnitude path (max |x_i| > 1e140 or 0 < max |x_i| < 1e-140): exact power-of-2
     * prescaling via frexp/ldexp maps every value into [-2, 2] with zero rounding error and
     * zero overflow or square underflow in mean/variance. */
    if ((max_abs_x > 1e140 || (max_abs_x > 0.0 && max_abs_x < 1e-140)) && isfinite(max_abs_x)) {
        int e_max = 0;
        (void)frexp(max_abs_x, &e_max);
        int32_t shift = (int32_t)(e_max - 1);
        double sum_s = 0.0;
        double comp_s = 0.0;
        for (int64_t i = 0; i < rk->count; ++i) {
            double vs = ldexp(rk->ring[i], -shift);
            double t = sum_s + vs;
            if (fabs(sum_s) >= fabs(vs)) {
                comp_s += (sum_s - t) + vs;
            } else {
                comp_s += (vs - t) + sum_s;
            }
            sum_s = t;
        }
        double mu_s = (sum_s + comp_s) / (double)rk->count;
        double mu = ldexp(mu_s, shift);
        if (!isfinite(mu)) {
            mu = copysign(DBL_MAX, mu_s);
        }
        rk->anchor = mu;
        rk->scale_exp = shift;

        double sd_s = 0.0, sd_c = 0.0;
        double sd2_s = 0.0, sd2_c = 0.0;
        for (int64_t i = 0; i < rk->count; ++i) {
            double vs = ldexp(rk->ring[i], -shift);
            double ds = vs - mu_s;
            double td = sd_s + ds;
            if (fabs(sd_s) >= fabs(ds)) {
                sd_c += (sd_s - td) + ds;
            } else {
                sd_c += (ds - td) + sd_s;
            }
            sd_s = td;

            double d2 = ds * ds;
            double td2 = sd2_s + d2;
            if (fabs(sd2_s) >= fabs(d2)) {
                sd2_c += (sd2_s - td2) + d2;
            } else {
                sd2_c += (d2 - td2) + sd2_s;
            }
            sd2_s = td2;
        }
        sd_s += sd_c;
        sd2_s += sd2_c;
        rk->sum_d = sd_s;
        rk->sum_d2 = (sd2_s >= 0.0) ? sd2_s : 0.0;
        rk->evicted_d2_accum = 0.0;
        return;
    }

    /* Normal-magnitude path: Neumaier compensated 2-pass mean and centered sum of squares */
    double sum = 0.0;
    double comp = 0.0;
    for (int64_t i = 0; i < rk->count; ++i) {
        double v = rk->ring[i];
        double t = sum + v;
        if (fabs(sum) >= fabs(v)) {
            comp += (sum - t) + v;
        } else {
            comp += (v - t) + sum;
        }
        sum = t;
    }
    double mu = (sum + comp) / (double)rk->count;
    if (!isfinite(mu)) {
        mu = v0;
    }
    rk->anchor = mu;
    rk->scale_exp = 0;

    double sd = 0.0;
    double sd_c = 0.0;
    double sd2 = 0.0;
    double sd2_c = 0.0;
    for (int64_t i = 0; i < rk->count; ++i) {
        double d = rk->ring[i] - mu;
        double td = sd + d;
        if (fabs(sd) >= fabs(d)) {
            sd_c += (sd - td) + d;
        } else {
            sd_c += (d - td) + sd;
        }
        sd = td;

        double d2 = d * d;
        double td2 = sd2 + d2;
        if (fabs(sd2) >= fabs(d2)) {
            sd2_c += (sd2 - td2) + d2;
        } else {
            sd2_c += (d2 - td2) + sd2;
        }
        sd2 = td2;
    }
    sd += sd_c;
    sd2 += sd2_c;
    rk->sum_d = sd;
    rk->sum_d2 = (sd2 >= 0.0) ? sd2 : 0.0;
    rk->evicted_d2_accum = 0.0;
}

static bool roll_update(tsjit_roll_key_t *rk, double x) {
    bool did_refresh = false;
    if (!rk->has_valid) {
        rk->anchor = x;
        rk->sum_d = 0.0;
        rk->sum_d2 = 0.0;
        rk->evicted_d2_accum = 0.0;
        rk->scale_exp = 0;
        rk->ring[0] = x;
        rk->head = (rk->window > 1) ? 1 : 0;
        rk->count = 1;
        rk->has_valid = true;
        return false;
    }

    if (TSJIT_UNLIKELY(rk->scale_exp != 0 ||
                       fabs(x) > 1e140 || fabs(rk->anchor) > 1e140 ||
                       (x != 0.0 && fabs(x) < 1e-140) ||
                       (rk->anchor != 0.0 && fabs(rk->anchor) < 1e-140))) {
        rk->ring[rk->head] = x;
        rk->head++;
        if (rk->head >= rk->window) {
            rk->head = 0;
        }
        if (rk->count < rk->window) {
            rk->count++;
        }
        roll_two_pass_refresh(rk);
        return true;
    }

    double d_new = x - rk->anchor;
    double d_new2 = d_new * d_new;
    bool need_refresh = false;

    if (rk->count == rk->window) {
        double x_old = rk->ring[rk->head];
        double d_old = x_old - rk->anchor;
        double d_old2 = d_old * d_old;
        double prev_sum_d2 = rk->sum_d2;

        rk->ring[rk->head] = x;
        rk->head++;
        if (rk->head >= rk->window) {
            rk->head = 0;
        }

        rk->sum_d += (d_new - d_old);
        rk->sum_d2 += (d_new2 - d_old2);
        rk->evicted_d2_accum += d_old2;

        double ref_prev = fmax(prev_sum_d2, 1e-30);
        double ref_cur  = fmax(rk->sum_d2, 1e-30);
        if (!isfinite(rk->sum_d) || !isfinite(rk->sum_d2) || rk->sum_d2 < 0.0 ||
            d_old2 > 0.05 * ref_prev ||
            d_old2 > 0.05 * ref_cur ||
            rk->evicted_d2_accum > 500.0 * ref_cur) {
            need_refresh = true;
        }
    } else {
        rk->ring[rk->head] = x;
        rk->head++;
        if (rk->head >= rk->window) {
            rk->head = 0;
        }
        rk->count++;
        rk->sum_d += d_new;
        rk->sum_d2 += d_new2;
        if (!isfinite(rk->sum_d) || !isfinite(rk->sum_d2)) {
            need_refresh = true;
        }
    }

    if (!need_refresh && rk->count > 1) {
        double inv_n = 1.0 / (double)rk->count;
        double mean_d = rk->sum_d * inv_n;
        double raw_m2 = rk->sum_d2 * inv_n;
        double var_pop = raw_m2 - mean_d * mean_d;
        if (!isfinite(var_pop) || var_pop < 0.0 ||
            (raw_m2 > 0.0 && var_pop < 1e-4 * raw_m2)) {
            need_refresh = true;
        }
    }

    if (need_refresh) {
        roll_two_pass_refresh(rk);
        did_refresh = true;
    }
    return did_refresh;
}

static void roll_get_stats(const tsjit_roll_key_t *rk, double *out_mean, double *out_std) {
    if (!rk->has_valid || rk->count <= 0) {
        *out_mean = 0.0;
        *out_std = 0.0;
        return;
    }
    double inv_n = 1.0 / (double)rk->count;
    double mean_d = rk->sum_d * inv_n;
    if (rk->scale_exp == 0) {
        *out_mean = rk->anchor + mean_d;
        if (rk->count <= 1) {
            *out_std = 0.0;
            return;
        }
        double ss = rk->sum_d2 - (rk->sum_d * mean_d);
        if (ss <= 0.0 || !isfinite(ss)) {
            *out_std = 0.0;
        } else {
            *out_std = sqrt(ss / (double)(rk->count - 1));
        }
    } else {
        double m = rk->anchor + ldexp(mean_d, rk->scale_exp);
        *out_mean = isfinite(m) ? m : rk->anchor;
        if (rk->count <= 1) {
            *out_std = 0.0;
            return;
        }
        double ss = rk->sum_d2 - (rk->sum_d * mean_d);
        if (ss <= 0.0 || !isfinite(ss)) {
            *out_std = 0.0;
        } else {
            double std_s = sqrt(ss / (double)(rk->count - 1));
            double sd = ldexp(std_s, rk->scale_exp);
            *out_std = isfinite(sd) ? sd : DBL_MAX;
        }
    }
}

/* Relative scale floor: DC-invariant when rolling_std > 0, micro-scale safe (1e-6), fallback 1.0 */
static inline double compute_eff_scale(double sc_hint, double rolling_std, double anchor) {
    bool valid_rstd = isfinite(rolling_std) && rolling_std > 1e-290 && rolling_std <= 1e290;
    bool valid_hint = isfinite(sc_hint) && sc_hint > 1e-290 && sc_hint <= 1e290;
    bool valid_anch = isfinite(anchor) && fabs(anchor) <= 1e290;
    double abs_anchor = valid_anch ? fabs(anchor) : 0.0;

    double ref_scale = valid_rstd ? rolling_std : abs_anchor;
    double rel_floor = 1e-4 * ref_scale;
    double ulp_floor = 1e-14 * abs_anchor;
    double floor_val = fmax(rel_floor, ulp_floor);

    double sc = valid_hint ? sc_hint : (valid_rstd ? rolling_std : 0.0);

    double eff = fmax(sc, floor_val);
    if (!isfinite(eff) || eff <= 1e-290 || eff > 1e290) {
        eff = 1.0;
    }
    return eff;
}

/* Sort 4 knots ascending */
static void sort4(double k[TSJIT_PWL_N_KNOTS]) {
    for (int i = 1; i < TSJIT_PWL_N_KNOTS; ++i) {
        double v = k[i];
        int j = i - 1;
        while (j >= 0 && k[j] > v) {
            k[j + 1] = k[j];
            --j;
        }
        k[j + 1] = v;
    }
}

/* Fallback evaluation when PWL knots/unit are degenerate */
static void eval_pwl_fallback(
    double x,
    double med,
    const double q_slow[TSJIT_NQ],
    const double q_fast[TSJIT_NQ],
    const double knots[TSJIT_PWL_N_KNOTS],
    const double w_fast[TSJIT_PWL_N_KNOTS + 1],
    const double gain[TSJIT_PWL_N_KNOTS + 1],
    double eff_sc,
    double out_q[TSJIT_NQ],
    double *out_abs_sg)
{
    double dx = x - med;
    double sc = (isfinite(eff_sc) && eff_sc > 1e-290 && eff_sc <= 1e290) ? eff_sc : 1.0;
    double sg = isfinite(dx) ? (fabs(dx) / sc) : 0.0;
    if (!isfinite(sg)) {
        sg = 1e6;
    }
    *out_abs_sg = sg;

    double wf = isfinite(w_fast[0]) ? fmin(fmax(w_fast[0], 0.0), 1.0) : 0.5;
    double g  = isfinite(gain[0]) ? fmin(fmax(gain[0], -2.0), 2.0) : 0.0;
    for (int i = 0; i < TSJIT_PWL_N_KNOTS; ++i) {
        if (isfinite(knots[i]) && sg >= knots[i]) {
            wf = isfinite(w_fast[i + 1]) ? fmin(fmax(w_fast[i + 1], 0.0), 1.0) : wf;
            g  = isfinite(gain[i + 1]) ? fmin(fmax(gain[i + 1], -2.0), 2.0) : g;
        }
    }
    double qs4 = isfinite(q_slow[4]) ? q_slow[4] : med;
    double qf4 = isfinite(q_fast[4]) ? q_fast[4] : qs4;
    double med_blend = (1.0 - wf) * qs4 + wf * qf4;
    double dx_blend = isfinite(med_blend) ? (x - med_blend) : dx;
    double shift = isfinite(dx_blend) ? (g * dx_blend) : 0.0;
    if (!isfinite(shift)) {
        shift = 0.0;
    }
    for (int q = 0; q < TSJIT_NQ; ++q) {
        double qs = isfinite(q_slow[q]) ? q_slow[q] : x;
        double qf = isfinite(q_fast[q]) ? q_fast[q] : x;
        double v = (1.0 - wf) * qs + wf * qf + shift;
        out_q[q] = isfinite(v) ? v : x;
    }
    tsjit_sort9(out_q);
}

/* ---------- Public Platform Utilities ---------- */

TSJIT_API int32_t tsjit_cacheline_size(void) {
    return TSJIT_CACHELINE;
}

TSJIT_API int32_t tsjit_check_bounds_i64(int64_t lo, int64_t n, int64_t cap) {
    return tsjit_bounds_i64(lo, n, cap) ? TSJIT_OK : TSJIT_ERR_OOB;
}

TSJIT_API int64_t tsjit_safe_age_i64_abi(int64_t now_step, int64_t ref_step) {
    return safe_age_i64(now_step, ref_step);
}

TSJIT_API void tsjit_sort9_abi(double a[TSJIT_NQ]) {
    if (a == NULL) {
        return;
    }
    tsjit_sort9(a);
}

TSJIT_API int32_t tsjit_pwl_hot_row_bytes(void) {
    return (int32_t)sizeof(tsjit_pwl_hot_slot_t);
}

TSJIT_API int32_t tsjit_seqlock_slot_bytes(void) {
    return (int32_t)sizeof(tsjit_pwl_hot_slot_t);
}

/* ---------- Engine Lifecycle ---------- */

TSJIT_API tsjit_arrive_engine_t *tsjit_arrive_create(
    int64_t n_keys,
    int64_t window,
    double lambda_age,
    int64_t max_age)
{
    if (n_keys <= 0 || n_keys > (1LL << 24) || window <= 0 || window > (1LL << 24)) {
        return NULL;
    }
    if (!isfinite(lambda_age) || lambda_age < 0.0 || max_age <= 0) {
        return NULL;
    }
    /* Overflow-safe ring allocation size check */
    if ((uint64_t)n_keys > (SIZE_MAX / sizeof(double)) / (uint64_t)window) {
        return NULL;
    }

    tsjit_arrive_engine_t *eng = (tsjit_arrive_engine_t *)calloc(1, sizeof(*eng));
    if (eng == NULL) {
        return NULL;
    }
    eng->n_keys = n_keys;
    eng->window = window;
    atomic_init(&eng->lambda_age_bits, tsjit_f64_to_u64(lambda_age));
    atomic_init(&eng->max_age, max_age);
    eng->max_spins = TSJIT_DEFAULT_MAX_SPINS;

    eng->hot_slots = (tsjit_pwl_hot_slot_t *)tsjit_aligned_calloc(
        TSJIT_CACHELINE, (size_t)n_keys * sizeof(tsjit_pwl_hot_slot_t));
    eng->cold_slots = (tsjit_pwl_cold_slot_t *)tsjit_aligned_calloc(
        TSJIT_CACHELINE, (size_t)n_keys * sizeof(tsjit_pwl_cold_slot_t));
    eng->keys = (tsjit_roll_key_t *)calloc((size_t)n_keys, sizeof(tsjit_roll_key_t));
    eng->ring_arena = (double *)tsjit_aligned_calloc(
        TSJIT_CACHELINE, (size_t)n_keys * (size_t)window * sizeof(double));
    eng->telem = (tsjit_telemetry_slot_t *)tsjit_aligned_calloc(
        TSJIT_CACHELINE, sizeof(tsjit_telemetry_slot_t));

    if (!eng->hot_slots || !eng->cold_slots || !eng->keys || !eng->ring_arena || !eng->telem) {
        tsjit_arrive_destroy(eng);
        return NULL;
    }

    for (int64_t k = 0; k < n_keys; ++k) {
        atomic_init(&eng->hot_slots[k].seq, 0ULL);
        for (size_t i = 0; i < TSJIT_HOT_NWORDS; ++i) {
            atomic_init(&eng->hot_slots[k].words[i], 0ULL);
        }
        for (size_t i = 0; i < TSJIT_COLD_NWORDS; ++i) {
            atomic_init(&eng->cold_slots[k].words[i], 0ULL);
        }
        eng->keys[k].ring = eng->ring_arena + (size_t)k * (size_t)window;
        eng->keys[k].window = window;
    }
    atomic_init(&eng->telem->seq, 0ULL);
    for (int i = 0; i < TSJIT_TELEM_NWORDS; ++i) {
        atomic_init(&eng->telem->words[i], 0ULL);
    }

    return eng;
}

TSJIT_API int32_t tsjit_arrive_set_params(
    tsjit_arrive_engine_t *eng,
    double lambda_age,
    int64_t max_age)
{
    if (eng == NULL || !isfinite(lambda_age) || lambda_age < 0.0 || max_age <= 0) {
        return TSJIT_ERR_BAD_ARG;
    }
    atomic_store_explicit(&eng->lambda_age_bits, tsjit_f64_to_u64(lambda_age), memory_order_release);
    atomic_store_explicit(&eng->max_age, max_age, memory_order_release);
    return TSJIT_OK;
}

TSJIT_API void tsjit_arrive_destroy(tsjit_arrive_engine_t *eng) {
    if (eng == NULL) {
        return;
    }
    free(eng->hot_slots);
    free(eng->cold_slots);
    free(eng->keys);
    free(eng->ring_arena);
    free(eng->telem);
    free(eng);
}

/* ---------- Forecast Compilation & Seqlock Publication ---------- */

TSJIT_API int32_t tsjit_arrive_publish_forecast(
    tsjit_arrive_engine_t *eng,
    int64_t key,
    int64_t forecast_step,
    const double q_slow[TSJIT_NQ],
    const double q_fast[TSJIT_NQ],
    double scale_hint,
    const double knots_in[TSJIT_PWL_N_KNOTS],
    const double w_fast_in[TSJIT_PWL_N_KNOTS + 1],
    const double gain_in[TSJIT_PWL_N_KNOTS + 1])
{
    if (eng == NULL || q_slow == NULL || q_fast == NULL) {
        return TSJIT_ERR_BAD_ARG;
    }
    if (!tsjit_bounds_i64(key, 1, eng->n_keys)) {
        return TSJIT_ERR_OOB;
    }

    /* Default knots/weights/gains if caller passes NULL */
    double knots[TSJIT_PWL_N_KNOTS] = {0.75, 1.5, 3.0, 6.0};
    double w_fast[TSJIT_PWL_N_KNOTS + 1] = {0.10, 0.25, 0.50, 0.75, 0.90};
    double gain[TSJIT_PWL_N_KNOTS + 1]   = {0.15, 0.35, 0.60, 0.85, 1.00};
    if (knots_in != NULL) {
        memcpy(knots, knots_in, sizeof(knots));
    }
    if (w_fast_in != NULL) {
        memcpy(w_fast, w_fast_in, sizeof(w_fast));
    }
    if (gain_in != NULL) {
        memcpy(gain, gain_in, sizeof(gain));
    }

    double qs[TSJIT_NQ];
    double qf[TSJIT_NQ];
    bool any_bad = false;
    for (int i = 0; i < TSJIT_NQ; ++i) {
        qs[i] = q_slow[i];
        qf[i] = q_fast[i];
        if (!isfinite(qs[i]) || !isfinite(qf[i])) {
            any_bad = true;
        }
    }
    if (!any_bad) {
        tsjit_sort9(qs);
        tsjit_sort9(qf);
    }

    double med = isfinite(qs[4]) ? qs[4] : 0.0;
    double raw_iqr = (isfinite(qs[7]) && isfinite(qs[1])) ? ((qs[7] - qs[1]) / 1.6832424671458286) : 0.0;
    double un = scale_hint;
    if (scale_hint == 0.0 && isfinite(raw_iqr) && raw_iqr > 0.0) {
        un = raw_iqr;
    }

    bool fallback = any_bad;
    /* Explicit degenerate knot/unit guard: un <= 1e-290 || un > 1e290 or non-finite */
    if (!isfinite(un) || un <= 1e-290 || un > 1e290) {
        fallback = true;
    }
    bool all_knots_finite = true;
    for (int i = 0; i < TSJIT_PWL_N_KNOTS; ++i) {
        if (!isfinite(knots[i])) {
            all_knots_finite = false;
            fallback = true;
        } else if (knots[i] <= 1e-290 || knots[i] > 1e290) {
            fallback = true;
        }
    }
    if (all_knots_finite) {
        sort4(knots);
    }
    for (int i = 0; i < TSJIT_PWL_N_KNOTS + 1; ++i) {
        if (!isfinite(w_fast[i]) || !isfinite(gain[i])) {
            fallback = true;
        }
    }

    if (!fallback) {
        for (int i = 1; i < TSJIT_PWL_N_KNOTS; ++i) {
            if ((knots[i] - knots[i - 1]) <= 1e-290) {
                fallback = true;
            }
        }
    }

    double bp_dx[TSJIT_PWL_N_KNOTS] = {0.0, 0.0, 0.0, 0.0};
    double a_pos_hot[TSJIT_NQ] = {0};
    double b_pos_hot[TSJIT_NQ] = {0};
    double a_neg_hot[TSJIT_NQ] = {0};
    double b_neg_hot[TSJIT_NQ] = {0};
    double a_pos_cold[4][TSJIT_NQ] = {{0}};
    double b_pos_cold[4][TSJIT_NQ] = {{0}};
    double a_neg_cold[4][TSJIT_NQ] = {{0}};
    double b_neg_cold[4][TSJIT_NQ] = {{0}};

    double inv_sc = 1.0;
    if (!fallback) {
        inv_sc = 1.0 / un;
        if (!isfinite(inv_sc)) {
            fallback = true;
        }
        for (int i = 0; i < TSJIT_PWL_N_KNOTS; ++i) {
            bp_dx[i] = knots[i] * un;
            if (!isfinite(bp_dx[i]) || bp_dx[i] <= 1e-290) {
                fallback = true;
            }
            if (i > 0 && (bp_dx[i] - bp_dx[i - 1]) <= 1e-290) {
                fallback = true;
            }
        }
    }

    if (!fallback) {
        double u[5] = {0.0, bp_dx[0], bp_dx[1], bp_dx[2], bp_dx[3]};
        /* Compile positive (+1) and negative (-1) half-lines into 5 + 5 = 10 affine segments.
         * Compute slopes in DC-cancelled form relative to qs[q] to avoid cancellation at 2.22e10. */
        double dq[TSJIT_NQ];
        for (int q = 0; q < TSJIT_NQ; ++q) {
            dq[q] = qf[q] - qs[q];
        }
        for (int side = 0; side < 2; ++side) {
            double sgn = (side == 0) ? 1.0 : -1.0;
            double y_rel[5][TSJIT_NQ];
            for (int j = 0; j < 5; ++j) {
                double dx_j = sgn * u[j];
                double wf = fmin(fmax(w_fast[j], 0.0), 1.0);
                double g  = gain[j];
                /* Shift innovation relative to blended median so w_fast and gain never double-count */
                double dx_blend = dx_j - wf * dq[4];
                for (int q = 0; q < TSJIT_NQ; ++q) {
                    y_rel[j][q] = wf * dq[q] + g * dx_blend;
                }
            }
            for (int seg = 0; seg < 5; ++seg) {
                double A_seg[TSJIT_NQ];
                double B_seg[TSJIT_NQ];
                if (seg < 4) {
                    double denom = sgn * (u[seg + 1] - u[seg]);
                    double dx0 = sgn * u[seg];
                    for (int q = 0; q < TSJIT_NQ; ++q) {
                        B_seg[q] = (y_rel[seg + 1][q] - y_rel[seg][q]) / denom;
                        A_seg[q] = qs[q] + (y_rel[seg][q] - B_seg[q] * dx0);
                        if (!isfinite(A_seg[q]) || !isfinite(B_seg[q])) {
                            fallback = true;
                        }
                    }
                } else {
                    double dx4 = sgn * u[4];
                    for (int q = 0; q < TSJIT_NQ; ++q) {
                        B_seg[q] = gain[4];
                        A_seg[q] = qs[q] + (y_rel[4][q] - B_seg[q] * dx4);
                        if (!isfinite(A_seg[q]) || !isfinite(B_seg[q])) {
                            fallback = true;
                        }
                    }
                }
                if (seg == 0) {
                    if (side == 0) {
                        memcpy(a_pos_hot, A_seg, sizeof(A_seg));
                        memcpy(b_pos_hot, B_seg, sizeof(B_seg));
                    } else {
                        memcpy(a_neg_hot, A_seg, sizeof(A_seg));
                        memcpy(b_neg_hot, B_seg, sizeof(B_seg));
                    }
                } else {
                    if (side == 0) {
                        memcpy(a_pos_cold[seg - 1], A_seg, sizeof(A_seg));
                        memcpy(b_pos_cold[seg - 1], B_seg, sizeof(B_seg));
                    } else {
                        memcpy(a_neg_cold[seg - 1], A_seg, sizeof(A_seg));
                        memcpy(b_neg_cold[seg - 1], B_seg, sizeof(B_seg));
                    }
                }
            }
        }
    }

    uint64_t flags = TSJIT_PWL_FLAG_VALID | (fallback ? TSJIT_PWL_FLAG_FALLBACK : 0ULL);
    uint64_t hwords[TSJIT_HOT_NWORDS];
    uint64_t cwords[TSJIT_COLD_NWORDS];
    memset(hwords, 0, sizeof(hwords));
    memset(cwords, 0, sizeof(cwords));

    hwords[0] = (uint64_t)forecast_step;
    hwords[1] = flags;
    hwords[2] = tsjit_f64_to_u64(med);
    hwords[3] = tsjit_f64_to_u64(inv_sc);
    hwords[4] = tsjit_f64_to_u64(un);
    hwords[5] = tsjit_f64_to_u64(bp_dx[0]);
    for (int q = 0; q < TSJIT_NQ; ++q) {
        hwords[6 + q]  = tsjit_f64_to_u64(a_pos_hot[q]);
        hwords[15 + q] = tsjit_f64_to_u64(b_pos_hot[q]);
        hwords[24 + q] = tsjit_f64_to_u64(a_neg_hot[q]);
        hwords[33 + q] = tsjit_f64_to_u64(b_neg_hot[q]);
    }
    for (int i = 0; i < TSJIT_PWL_N_KNOTS; ++i) {
        hwords[42 + i] = tsjit_f64_to_u64(bp_dx[i]);
    }

    size_t idx = 0;
    for (int s = 0; s < 4; ++s) {
        for (int q = 0; q < TSJIT_NQ; ++q) {
            cwords[idx++] = tsjit_f64_to_u64(a_pos_cold[s][q]);
        }
    }
    for (int s = 0; s < 4; ++s) {
        for (int q = 0; q < TSJIT_NQ; ++q) {
            cwords[idx++] = tsjit_f64_to_u64(b_pos_cold[s][q]);
        }
    }
    for (int s = 0; s < 4; ++s) {
        for (int q = 0; q < TSJIT_NQ; ++q) {
            cwords[idx++] = tsjit_f64_to_u64(a_neg_cold[s][q]);
        }
    }
    for (int s = 0; s < 4; ++s) {
        for (int q = 0; q < TSJIT_NQ; ++q) {
            cwords[idx++] = tsjit_f64_to_u64(b_neg_cold[s][q]);
        }
    }
    for (int q = 0; q < TSJIT_NQ; ++q) {
        cwords[idx++] = tsjit_f64_to_u64(qs[q]);
    }
    for (int q = 0; q < TSJIT_NQ; ++q) {
        cwords[idx++] = tsjit_f64_to_u64(qf[q]);
    }
    for (int i = 0; i < TSJIT_PWL_N_KNOTS; ++i) {
        cwords[idx++] = tsjit_f64_to_u64(knots[i]);
    }
    for (int i = 0; i < TSJIT_PWL_N_KNOTS + 1; ++i) {
        cwords[idx++] = tsjit_f64_to_u64(w_fast[i]);
    }
    for (int i = 0; i < TSJIT_PWL_N_KNOTS + 1; ++i) {
        cwords[idx++] = tsjit_f64_to_u64(gain[i]);
    }

    tsjit_pwl_hot_slot_t  *hslot = &eng->hot_slots[key];
    tsjit_pwl_cold_slot_t *cslot = &eng->cold_slots[key];
    uint64_t s_even = seqlock_write_begin(&hslot->seq);
    for (size_t i = 0; i < TSJIT_HOT_NWORDS; ++i) {
        atomic_store_explicit(&hslot->words[i], hwords[i], memory_order_relaxed);
    }
    for (size_t i = 0; i < TSJIT_COLD_NWORDS; ++i) {
        atomic_store_explicit(&cslot->words[i], cwords[i], memory_order_relaxed);
    }
    seqlock_write_end(&hslot->seq, s_even);

    return TSJIT_OK;
}

/* ---------- Helper for causal step gating & rolling update ---------- */

static int32_t causal_gate_and_roll(
    tsjit_arrive_engine_t *eng,
    int64_t key,
    int64_t step,
    double x,
    int64_t forecast_step,
    bool has_forecast,
    double *out_x_eff,
    double *out_mean,
    double *out_std,
    int64_t *out_age,
    bool *out_did_refresh,
    tsjit_arrive_out_t *out)
{
    *out_did_refresh = false;
    if (!tsjit_bounds_i64(key, 1, eng->n_keys)) {
        memset(out, 0, sizeof(*out));
        out->mark = MARK_INVALID_KEY;
        telem_record(eng->telem, 1, 0, 0, 0, 0, 0, 0, 0, 0, false, 0, 0, 0.0);
        return MARK_INVALID_KEY;
    }

    tsjit_roll_key_t *rk = &eng->keys[key];
    int64_t age = has_forecast ? safe_age_i64(step, forecast_step) : 0;
    *out_age = age;

    /* DUP / OOO rejection with ZERO state mutation */
    if (rk->has_step) {
        if (step == rk->last_step) {
            roll_get_stats(rk, out_mean, out_std);
            memcpy(out->q, rk->cached_q, sizeof(out->q));
            out->abs_sg = rk->cached_abs_sg;
            out->rolling_mean = *out_mean;
            out->rolling_std = *out_std;
            out->eff_scale = rk->cached_eff_sc;
            out->age = age;
            out->mark = MARK_DUP_STEP;
            out->pwl_path = rk->cached_pwl_path;
            memcpy(out->family_weights, rk->cached_fam_w, sizeof(out->family_weights));
            telem_record(eng->telem, 1, 0, 1, 0, 0, 0, 0, 0, 0, false, 0, 0, 0.0);
            return MARK_DUP_STEP;
        }
        if (step < rk->last_step) {
            roll_get_stats(rk, out_mean, out_std);
            memcpy(out->q, rk->cached_q, sizeof(out->q));
            out->abs_sg = rk->cached_abs_sg;
            out->rolling_mean = *out_mean;
            out->rolling_std = *out_std;
            out->eff_scale = rk->cached_eff_sc;
            out->age = age;
            out->mark = MARK_OOO_STEP;
            out->pwl_path = rk->cached_pwl_path;
            memcpy(out->family_weights, rk->cached_fam_w, sizeof(out->family_weights));
            telem_record(eng->telem, 1, 0, 0, 1, 0, 0, 0, 0, 0, false, 0, 0, 0.0);
            return MARK_OOO_STEP;
        }
    }

    rk->has_step = true;
    rk->last_step = step;

    if (!isfinite(x)) {
        roll_get_stats(rk, out_mean, out_std);
        if (!rk->has_valid) {
            *out_x_eff = 0.0;
            return MARK_ZERO;
        }
        *out_x_eff = rk->last_valid;
        int64_t obs_age = safe_age_i64(step, rk->last_valid_step);
        int64_t max_age = atomic_load_explicit(&eng->max_age, memory_order_acquire);
        if (obs_age > max_age) {
            return MARK_NAN_STALE;
        }
        return MARK_LAST_VALID;
    }

    *out_did_refresh = roll_update(rk, x);
    rk->last_valid = x;
    rk->last_valid_step = step;
    roll_get_stats(rk, out_mean, out_std);
    *out_x_eff = x;
    return MARK_OK;
}

/* ---------- PWL-ARRIVE Hot/Cold Step ---------- */

TSJIT_API int32_t tsjit_arrive_step_pwl(
    tsjit_arrive_engine_t *eng,
    int64_t key,
    int64_t step,
    double x,
    tsjit_arrive_out_t *out)
{
    if (eng == NULL || out == NULL) {
        return TSJIT_ERR_BAD_ARG;
    }
    if (!tsjit_bounds_i64(key, 1, eng->n_keys)) {
        memset(out, 0, sizeof(*out));
        out->mark = MARK_INVALID_KEY;
        telem_record(eng->telem, 1, 0, 0, 0, 0, 0, 0, 0, 0, false, 0, 0, 0.0);
        return TSJIT_ERR_OOB;
    }

    /* Lock-free read of HOT slot only first (BEFORE mutating any causal rolling state!) */
    uint64_t hwords[TSJIT_HOT_NWORDS];
    uint64_t cwords[TSJIT_COLD_NWORDS];
    bool have_cold = false;
    int32_t rc = seqlock_read_words(
        &eng->hot_slots[key].seq,
        eng->hot_slots[key].words,
        TSJIT_HOT_NWORDS,
        hwords,
        eng->max_spins);
    if (TSJIT_UNLIKELY(rc != TSJIT_OK)) {
        return rc;
    }

    int64_t forecast_step = (int64_t)hwords[0];
    uint64_t flags = hwords[1];
    bool has_forecast = (flags & TSJIT_PWL_FLAG_VALID) != 0ULL;

    /* If cold segment or fallback will be needed, acquire consistent hot+cold snapshot NOW
     * BEFORE causal_gate_and_roll() so a TSJIT_ERR_SPIN_TIMEOUT leaves rolling state untouched! */
    const tsjit_roll_key_t *rk_ro = &eng->keys[key];
    if (has_forecast && !(rk_ro->has_step && step <= rk_ro->last_step)) {
        double x_preview = isfinite(x) ? x : (rk_ro->has_valid ? rk_ro->last_valid : 0.0);
        double med_prev  = tsjit_u64_to_f64(hwords[2]);
        double b0_prev   = tsjit_u64_to_f64(hwords[5]);
        double dx_prev   = x_preview - med_prev;
        if (TSJIT_UNLIKELY((flags & TSJIT_PWL_FLAG_FALLBACK) != 0ULL ||
                           !isfinite(dx_prev) || fabs(dx_prev) >= b0_prev)) {
            rc = seqlock_read_hot_and_cold(
                &eng->hot_slots[key], &eng->cold_slots[key],
                hwords, cwords, eng->max_spins);
            if (TSJIT_UNLIKELY(rc != TSJIT_OK)) {
                return rc;
            }
            have_cold = true;
            forecast_step = (int64_t)hwords[0];
            flags = hwords[1];
            has_forecast = (flags & TSJIT_PWL_FLAG_VALID) != 0ULL;
        }
    }

    double x_eff = 0.0;
    double r_mean = 0.0;
    double r_std = 0.0;
    int64_t age = 0;
    bool did_refresh = false;
    int32_t mark = causal_gate_and_roll(
        eng, key, step, x, forecast_step, has_forecast,
        &x_eff, &r_mean, &r_std, &age, &did_refresh, out);

    if (mark == MARK_DUP_STEP || mark == MARK_OOO_STEP) {
        return TSJIT_OK;
    }

    tsjit_roll_key_t *rk = &eng->keys[key];
    double eff_sc = compute_eff_scale(
        has_forecast ? tsjit_u64_to_f64(hwords[4]) : 0.0, r_std, rk->anchor);

    uint64_t d_fast = 0, d_cold = 0, d_fb = 0;
    double abs_sg = 0.0;
    int32_t pwl_path = 0;

    if (!has_forecast) {
        /* Cold start before first forecast: persistence around x_eff with rolling scale */
        double dx = x_eff - r_mean;
        abs_sg = isfinite(dx) ? (fabs(dx) / eff_sc) : 1e6;
        if (!isfinite(abs_sg)) {
            abs_sg = 1e6;
        }
        for (int q = 0; q < TSJIT_NQ; ++q) {
            double v = x_eff + TSJIT_Z9[q] * eff_sc;
            out->q[q] = isfinite(v) ? v : x_eff;
        }
        tsjit_sort9(out->q);
        pwl_path = 2;
        d_fb = 1;
    } else {
        double med    = tsjit_u64_to_f64(hwords[2]);
        double inv_sc = tsjit_u64_to_f64(hwords[3]);
        double b0_dx  = tsjit_u64_to_f64(hwords[5]);
        double dx     = x_eff - med;

        if (TSJIT_LIKELY(!(flags & TSJIT_PWL_FLAG_FALLBACK) && isfinite(dx) && fabs(dx) < b0_dx)) {
            /* HOT segment: [0, b0) or [-b0, 0) — zero divides on hot path! */
            abs_sg = fabs(dx) * inv_sc;
            if (!isfinite(abs_sg)) {
                abs_sg = 1e6;
            }
            size_t a_off = (dx >= 0.0) ? 6U : 24U;
            size_t b_off = (dx >= 0.0) ? 15U : 33U;
            for (int q = 0; q < TSJIT_NQ; ++q) {
                double A = tsjit_u64_to_f64(hwords[a_off + (size_t)q]);
                double B = tsjit_u64_to_f64(hwords[b_off + (size_t)q]);
                double v = A + ((B == 0.0) ? 0.0 : (B * dx));
                out->q[q] = isfinite(v) ? v : x_eff;
            }
            tsjit_sort9(out->q);
            pwl_path = 0;
            d_fast = 1;
        } else if (TSJIT_UNLIKELY(!have_cold)) {
            /* Defensive guard (unreachable in normal flow since preview matches x_eff) */
            double dx_r = x_eff - r_mean;
            abs_sg = isfinite(dx_r) ? (fabs(dx_r) / eff_sc) : 1e6;
            if (!isfinite(abs_sg)) abs_sg = 1e6;
            for (int q = 0; q < TSJIT_NQ; ++q) {
                double v = x_eff + TSJIT_Z9[q] * eff_sc;
                out->q[q] = isfinite(v) ? v : x_eff;
            }
            tsjit_sort9(out->q);
            pwl_path = 2;
            d_fb = 1;
        } else if (TSJIT_UNLIKELY((flags & TSJIT_PWL_FLAG_FALLBACK) != 0ULL)) {
            double qs[TSJIT_NQ], qf[TSJIT_NQ], knots[4], wf[5], gn[5];
            for (int q = 0; q < TSJIT_NQ; ++q) {
                qs[q] = tsjit_u64_to_f64(cwords[144 + q]);
                qf[q] = tsjit_u64_to_f64(cwords[153 + q]);
            }
            for (int i = 0; i < 4; ++i) {
                knots[i] = tsjit_u64_to_f64(cwords[162 + i]);
            }
            for (int i = 0; i < 5; ++i) {
                wf[i] = tsjit_u64_to_f64(cwords[166 + i]);
                gn[i] = tsjit_u64_to_f64(cwords[171 + i]);
            }
            eval_pwl_fallback(x_eff, med, qs, qf, knots, wf, gn, eff_sc, out->q, &abs_sg);
            pwl_path = 2;
            d_fb = 1;
        } else {
            abs_sg = isfinite(dx) ? (fabs(dx) * inv_sc) : 1e6;
            if (!isfinite(abs_sg)) {
                abs_sg = 1e6;
            }
            double bp[4];
            for (int i = 0; i < 4; ++i) {
                bp[i] = tsjit_u64_to_f64(hwords[42 + i]);
            }
            double adx = fabs(dx);
            int cseg = 3; /* [b3, +inf) */
            if (adx < bp[1]) {
                cseg = 0;
            } else if (adx < bp[2]) {
                cseg = 1;
            } else if (adx < bp[3]) {
                cseg = 2;
            }
            size_t a_base = (dx >= 0.0) ? (size_t)(cseg * TSJIT_NQ) : (size_t)(72 + cseg * TSJIT_NQ);
            size_t b_base = (dx >= 0.0) ? (size_t)(36 + cseg * TSJIT_NQ) : (size_t)(108 + cseg * TSJIT_NQ);
            for (int q = 0; q < TSJIT_NQ; ++q) {
                double A = tsjit_u64_to_f64(cwords[a_base + (size_t)q]);
                double B = tsjit_u64_to_f64(cwords[b_base + (size_t)q]);
                double v = A + ((B == 0.0) ? 0.0 : (B * dx));
                out->q[q] = isfinite(v) ? v : x_eff;
            }
            tsjit_sort9(out->q);
            pwl_path = 1;
            d_cold = 1;
        }
    }

    /* Honor lambda_age exponential decay on stale forecast blend (age > 1) */
    double w_age = 1.0;
    if (has_forecast) {
        double lambda_age = tsjit_u64_to_f64(
            atomic_load_explicit(&eng->lambda_age_bits, memory_order_acquire));
        if (age > 1 && isfinite(lambda_age) && lambda_age > 0.0) {
            double decay_arg = lambda_age * (double)(age - 1);
            w_age = (decay_arg >= 700.0) ? 0.0 : exp(-decay_arg);
            double w_pers = 1.0 - w_age;
            for (int q = 0; q < TSJIT_NQ; ++q) {
                double q_pers = x_eff + TSJIT_Z9[q] * eff_sc;
                if (!isfinite(q_pers)) {
                    q_pers = x_eff;
                }
                double blended = w_age * out->q[q] + w_pers * q_pers;
                out->q[q] = isfinite(blended) ? blended : q_pers;
            }
            tsjit_sort9(out->q);
        }
        out->family_weights[0] = 0.5 * w_age;
        out->family_weights[1] = 0.5 * w_age;
        out->family_weights[2] = 1.0 - w_age;
    } else {
        out->family_weights[0] = 0.0;
        out->family_weights[1] = 0.0;
        out->family_weights[2] = 1.0;
    }

    out->abs_sg = abs_sg;
    out->rolling_mean = r_mean;
    out->rolling_std = r_std;
    out->eff_scale = eff_sc;
    out->age = age;
    out->mark = mark;
    out->pwl_path = pwl_path;

    memcpy(rk->cached_q, out->q, sizeof(rk->cached_q));
    rk->cached_abs_sg = abs_sg;
    rk->cached_eff_sc = eff_sc;
    rk->cached_pwl_path = pwl_path;
    memcpy(rk->cached_fam_w, out->family_weights, sizeof(rk->cached_fam_w));

    telem_record(
        eng->telem,
        1,
        (mark == MARK_OK) ? 1ULL : 0ULL,
        0,
        0,
        (mark == MARK_NAN_STALE || mark == MARK_LAST_VALID || mark == MARK_ZERO) ? 1ULL : 0ULL,
        d_fast,
        d_cold,
        d_fb,
        did_refresh ? 1ULL : 0ULL,
        true,
        step,
        key,
        abs_sg);

    return TSJIT_OK;
}

/* ---------- Hierarchical 3-Family Age-Aware SoftRevisionV2 ---------- */

/*
 * Aggregate valid candidates in a single model family:
 *  - Skips any candidate with non-finite loss or any non-finite quantile (NEVER maps NaN loss -> 0.0!)
 *  - Returns false if zero valid candidates exist (caller assigns family weight 0.0)
 *  - Outputs min_loss and log((1/M_valid) * sum exp(-(l - min_loss)/tau_i)) separately so
 *    inter-family scoring shifts by global_min_loss BEFORE dividing by tau (zero Inf - Inf NaN!).
 */
static bool aggregate_family(
    int32_t n_models,
    const double *qs,
    const double *losses,
    double tau_intra,
    double out_q[TSJIT_NQ],
    double *out_min_loss,
    double *out_log_bonus)
{
    if (n_models <= 0 || qs == NULL) {
        return false;
    }

    double min_l = INFINITY;
    int32_t valid_count = 0;
    int32_t best_idx = -1;
    for (int32_t m = 0; m < n_models; ++m) {
        if (losses != NULL && !isfinite(losses[m])) {
            continue;
        }
        const double *mq = qs + (size_t)m * TSJIT_NQ;
        bool q_ok = true;
        for (int q = 0; q < TSJIT_NQ; ++q) {
            if (!isfinite(mq[q])) {
                q_ok = false;
                break;
            }
        }
        if (!q_ok) {
            continue;
        }
        double l = (losses != NULL) ? losses[m] : 0.0;
        if (valid_count == 0 || l < min_l) {
            min_l = l;
            best_idx = m;
        }
        valid_count++;
    }

    if (valid_count == 0 || best_idx < 0 || !isfinite(min_l)) {
        return false;
    }

    double tau_i = fmax(tau_intra, 1e-12);
    double sum_w = 0.0;
    for (int32_t m = 0; m < n_models; ++m) {
        if (losses != NULL && !isfinite(losses[m])) {
            continue;
        }
        const double *mq = qs + (size_t)m * TSJIT_NQ;
        bool q_ok = true;
        for (int q = 0; q < TSJIT_NQ; ++q) {
            if (!isfinite(mq[q])) {
                q_ok = false;
                break;
            }
        }
        if (!q_ok) {
            continue;
        }
        double l = (losses != NULL) ? losses[m] : 0.0;
        double diff = l - min_l;
        double arg = 0.0;
        if (diff > 0.0) {
            double scaled = isfinite(diff) ? (diff / tau_i) : 700.0;
            arg = (!isfinite(scaled) || scaled >= 700.0) ? -700.0 : -scaled;
        }
        sum_w += exp(arg);
    }

    if (sum_w <= 0.0 || !isfinite(sum_w)) {
        memcpy(out_q, qs + (size_t)best_idx * TSJIT_NQ, TSJIT_NQ * sizeof(double));
        tsjit_sort9(out_q);
        *out_min_loss = min_l;
        *out_log_bonus = 0.0;
        return true;
    }

    memset(out_q, 0, TSJIT_NQ * sizeof(double));
    double inv_sw = 1.0 / sum_w;
    for (int32_t m = 0; m < n_models; ++m) {
        if (losses != NULL && !isfinite(losses[m])) {
            continue;
        }
        const double *mq = qs + (size_t)m * TSJIT_NQ;
        bool q_ok = true;
        for (int q = 0; q < TSJIT_NQ; ++q) {
            if (!isfinite(mq[q])) {
                q_ok = false;
                break;
            }
        }
        if (!q_ok) {
            continue;
        }
        double l = (losses != NULL) ? losses[m] : 0.0;
        double diff = l - min_l;
        double arg = 0.0;
        if (diff > 0.0) {
            double scaled = isfinite(diff) ? (diff / tau_i) : 700.0;
            arg = (!isfinite(scaled) || scaled >= 700.0) ? -700.0 : -scaled;
        }
        double wn = exp(arg) * inv_sw;
        if (wn > 0.0) {
            for (int q = 0; q < TSJIT_NQ; ++q) {
                out_q[q] += wn * mq[q];
            }
        }
    }
    const double *best_q = qs + (size_t)best_idx * TSJIT_NQ;
    for (int q = 0; q < TSJIT_NQ; ++q) {
        if (!isfinite(out_q[q])) {
            out_q[q] = best_q[q];
        }
    }
    tsjit_sort9(out_q);

    /* Equal-prior average evidence bonus log(sum_w / M_valid) in [-log(M_valid), 0] */
    double ratio = sum_w / (double)valid_count;
    double bonus = (ratio > 0.0 && isfinite(ratio)) ? log(ratio) : 0.0;
    *out_min_loss = min_l;
    *out_log_bonus = isfinite(bonus) ? bonus : 0.0;
    return true;
}

TSJIT_API int32_t tsjit_arrive_step_soft_v2(
    tsjit_arrive_engine_t *eng,
    int64_t key,
    int64_t step,
    double x,
    int32_t n_slow_models,
    const double *slow_qs,
    const double *slow_losses,
    int32_t n_fast_models,
    const double *fast_qs,
    const double *fast_losses,
    int32_t n_persist_models,
    const double *persist_qs,
    const double *persist_losses,
    double tau_intra,
    double tau_family,
    double beta_surprise,
    tsjit_arrive_out_t *out)
{
    if (eng == NULL || out == NULL ||
        n_slow_models < 0 || n_fast_models < 0 || n_persist_models < 0 ||
        (n_slow_models > 0 && slow_qs == NULL) ||
        (n_fast_models > 0 && fast_qs == NULL) ||
        (n_persist_models > 0 && persist_qs == NULL)) {
        return TSJIT_ERR_BAD_ARG;
    }
    if (!tsjit_bounds_i64(key, 1, eng->n_keys)) {
        memset(out, 0, sizeof(*out));
        out->mark = MARK_INVALID_KEY;
        telem_record(eng->telem, 1, 0, 0, 0, 0, 0, 0, 0, 0, false, 0, 0, 0.0);
        return TSJIT_ERR_OOB;
    }
    if (!isfinite(tau_intra) || tau_intra <= 0.0) {
        tau_intra = 1.0;
    }
    if (!isfinite(tau_family) || tau_family <= 0.0) {
        tau_family = 1.0;
    }
    if (!isfinite(beta_surprise) || beta_surprise < 0.0) {
        beta_surprise = 0.0;
    }

    uint64_t hwords[TSJIT_HOT_NWORDS];
    uint64_t cwords[TSJIT_COLD_NWORDS];
    int32_t rc = seqlock_read_hot_and_cold(
        &eng->hot_slots[key], &eng->cold_slots[key],
        hwords, cwords, eng->max_spins);
    if (TSJIT_UNLIKELY(rc != TSJIT_OK)) {
        return rc;
    }

    int64_t forecast_step = (int64_t)hwords[0];
    uint64_t flags = hwords[1];
    bool has_forecast = (flags & TSJIT_PWL_FLAG_VALID) != 0ULL;

    double x_eff = 0.0;
    double r_mean = 0.0;
    double r_std = 0.0;
    int64_t age = 0;
    bool did_refresh = false;
    int32_t mark = causal_gate_and_roll(
        eng, key, step, x, forecast_step, has_forecast,
        &x_eff, &r_mean, &r_std, &age, &did_refresh, out);

    if (mark == MARK_DUP_STEP || mark == MARK_OOO_STEP) {
        return TSJIT_OK;
    }

    tsjit_roll_key_t *rk = &eng->keys[key];
    double eff_sc = compute_eff_scale(
        has_forecast ? tsjit_u64_to_f64(hwords[4]) : 0.0, r_std, rk->anchor);

    /* Finite causal persistence baseline */
    double baseline[TSJIT_NQ];
    for (int q = 0; q < TSJIT_NQ; ++q) {
        double v = x_eff + TSJIT_Z9[q] * eff_sc;
        baseline[q] = isfinite(v) ? v : x_eff;
    }
    tsjit_sort9(baseline);

    /* Sanitize slot slow/fast quantiles (guard TSJIT_PWL_FLAG_FALLBACK & non-finite quantiles!) */
    double slot_slow[TSJIT_NQ];
    double slot_fast[TSJIT_NQ];
    double def_persist[TSJIT_NQ];
    double zero_loss = 0.0;

    if (has_forecast) {
        bool slow_ok = true;
        bool fast_ok = true;
        for (int q = 0; q < TSJIT_NQ; ++q) {
            slot_slow[q] = tsjit_u64_to_f64(cwords[144 + q]);
            slot_fast[q] = tsjit_u64_to_f64(cwords[153 + q]);
            if (!isfinite(slot_slow[q])) slow_ok = false;
            if (!isfinite(slot_fast[q])) fast_ok = false;
        }
        if (slow_ok) {
            tsjit_sort9(slot_slow);
        } else {
            memcpy(slot_slow, baseline, sizeof(baseline));
        }
        if (fast_ok) {
            tsjit_sort9(slot_fast);
        } else {
            memcpy(slot_fast, baseline, sizeof(baseline));
        }
        if (slow_ok) {
            double slow_med = slot_slow[4];
            bool p_ok = true;
            for (int q = 0; q < TSJIT_NQ; ++q) {
                double pv = x_eff + (slot_slow[q] - slow_med);
                if (!isfinite(pv)) {
                    p_ok = false;
                    break;
                }
                def_persist[q] = pv;
            }
            if (!p_ok) {
                memcpy(def_persist, baseline, sizeof(baseline));
            }
        } else {
            memcpy(def_persist, baseline, sizeof(baseline));
        }
    } else {
        memcpy(slot_slow, baseline, sizeof(baseline));
        memcpy(slot_fast, baseline, sizeof(baseline));
        memcpy(def_persist, baseline, sizeof(baseline));
    }

    double Q_fam[3][TSJIT_NQ];
    double min_l_fam[3] = {0.0, 0.0, 0.0};
    double bonus_fam[3] = {0.0, 0.0, 0.0};
    bool   valid_fam[3] = {false, false, false};

    if (n_slow_models > 0 && slow_qs != NULL) {
        valid_fam[0] = aggregate_family(
            n_slow_models, slow_qs, slow_losses, tau_intra, Q_fam[0], &min_l_fam[0], &bonus_fam[0]);
    } else {
        valid_fam[0] = aggregate_family(
            1, slot_slow, &zero_loss, tau_intra, Q_fam[0], &min_l_fam[0], &bonus_fam[0]);
    }
    if (n_fast_models > 0 && fast_qs != NULL) {
        valid_fam[1] = aggregate_family(
            n_fast_models, fast_qs, fast_losses, tau_intra, Q_fam[1], &min_l_fam[1], &bonus_fam[1]);
    } else {
        valid_fam[1] = aggregate_family(
            1, slot_fast, &zero_loss, tau_intra, Q_fam[1], &min_l_fam[1], &bonus_fam[1]);
    }
    if (n_persist_models > 0 && persist_qs != NULL) {
        valid_fam[2] = aggregate_family(
            n_persist_models, persist_qs, persist_losses, tau_intra, Q_fam[2], &min_l_fam[2], &bonus_fam[2]);
    } else {
        valid_fam[2] = aggregate_family(
            1, def_persist, &zero_loss, tau_intra, Q_fam[2], &min_l_fam[2], &bonus_fam[2]);
    }

    double ref_med = valid_fam[0] ? Q_fam[0][4]
                   : (valid_fam[1] ? Q_fam[1][4]
                   : (valid_fam[2] ? Q_fam[2][4] : r_mean));
    double abs_sg = fabs(x_eff - ref_med) / eff_sc;
    if (!isfinite(abs_sg)) {
        abs_sg = 1e6;
    }

    /* Exponential age decay exp(-lambda_age * max(age - 1, 0)) on Slow-TSFM family */
    double lambda_age = tsjit_u64_to_f64(
        atomic_load_explicit(&eng->lambda_age_bits, memory_order_acquire));
    double age_excess = (age > 1) ? (double)(age - 1) : 0.0;
    double age_decay_log = (isfinite(lambda_age) && lambda_age > 0.0) ? (lambda_age * age_excess) : 0.0;
    if (!isfinite(age_decay_log) || age_decay_log > 700.0) {
        age_decay_log = 700.0;
    }
    double surp_pen = beta_surprise * abs_sg;
    if (!isfinite(surp_pen) || surp_pen > 700.0) {
        surp_pen = 700.0;
    }

    /* Shift all finite family min_losses by global_min_l BEFORE dividing by tau (zero Inf - Inf!) */
    double global_min_l = INFINITY;
    int n_valid_fam = 0;
    for (int f = 0; f < 3; ++f) {
        if (valid_fam[f]) {
            if (n_valid_fam == 0 || min_l_fam[f] < global_min_l) {
                global_min_l = min_l_fam[f];
            }
            n_valid_fam++;
        }
    }

    double w_fam[3] = {0.0, 0.0, 0.0};
    if (n_valid_fam > 0) {
        double tau_i = fmax(tau_intra, 1e-12);
        double tau_f = fmax(tau_family, 1e-12);
        double pen[3] = {
            age_decay_log + surp_pen,
            0.25 * age_decay_log + 0.25 * surp_pen,
            0.0
        };
        double scores[3] = {-INFINITY, -INFINITY, -INFINITY};
        double max_s = -INFINITY;
        for (int f = 0; f < 3; ++f) {
            if (!valid_fam[f]) {
                continue;
            }
            double delta_l = min_l_fam[f] - global_min_l;
            double loss_pen = 0.0;
            if (delta_l > 0.0) {
                double lp = isfinite(delta_l) ? ((delta_l / tau_i) / tau_f) : 1e12;
                loss_pen = (!isfinite(lp) || lp > 1e12) ? 1e12 : lp;
            }
            double bonus_term = bonus_fam[f] / tau_f;
            if (!isfinite(bonus_term) || bonus_term < -1e12) {
                bonus_term = -1e12;
            }
            scores[f] = -loss_pen + bonus_term - pen[f];
            if (scores[f] > max_s) {
                max_s = scores[f];
            }
        }
        double wsum = 0.0;
        for (int f = 0; f < 3; ++f) {
            if (!valid_fam[f]) {
                w_fam[f] = 0.0;
                continue;
            }
            double diff_s = scores[f] - max_s;
            double clamped = fmin(fmax(diff_s, -700.0), 0.0);
            w_fam[f] = exp(clamped);
            wsum += w_fam[f];
        }
        if (wsum > 0.0 && isfinite(wsum)) {
            for (int f = 0; f < 3; ++f) {
                w_fam[f] /= wsum;
            }
        } else {
            double eq = 1.0 / (double)n_valid_fam;
            for (int f = 0; f < 3; ++f) {
                w_fam[f] = valid_fam[f] ? eq : 0.0;
            }
        }

        for (int q = 0; q < TSJIT_NQ; ++q) {
            double acc = 0.0;
            for (int f = 0; f < 3; ++f) {
                if (w_fam[f] > 0.0) {
                    acc += w_fam[f] * Q_fam[f][q];
                }
            }
            out->q[q] = isfinite(acc) ? acc : baseline[q];
        }
        tsjit_sort9(out->q);
    } else {
        /* All 3 families had only invalid candidates: clean finite causal baseline fallback */
        memcpy(out->q, baseline, sizeof(baseline));
    }

    out->abs_sg = abs_sg;
    out->rolling_mean = r_mean;
    out->rolling_std = r_std;
    out->eff_scale = eff_sc;
    out->age = age;
    out->mark = mark;
    out->pwl_path = 3;
    out->family_weights[0] = w_fam[0];
    out->family_weights[1] = w_fam[1];
    out->family_weights[2] = w_fam[2];

    memcpy(rk->cached_q, out->q, sizeof(rk->cached_q));
    rk->cached_abs_sg = abs_sg;
    rk->cached_eff_sc = eff_sc;
    rk->cached_pwl_path = 3;
    memcpy(rk->cached_fam_w, out->family_weights, sizeof(rk->cached_fam_w));

    telem_record(
        eng->telem,
        1,
        (mark == MARK_OK) ? 1ULL : 0ULL,
        0,
        0,
        (mark == MARK_NAN_STALE || mark == MARK_LAST_VALID || mark == MARK_ZERO) ? 1ULL : 0ULL,
        0,
        0,
        0,
        did_refresh ? 1ULL : 0ULL,
        true,
        step,
        key,
        abs_sg);

    return TSJIT_OK;
}

TSJIT_API int32_t tsjit_arrive_get_rolling(
    tsjit_arrive_engine_t *eng,
    int64_t key,
    double *out_mean,
    double *out_std,
    double *out_anchor,
    int64_t *out_count,
    int64_t *out_last_step)
{
    if (eng == NULL) {
        return TSJIT_ERR_BAD_ARG;
    }
    if (!tsjit_bounds_i64(key, 1, eng->n_keys)) {
        return TSJIT_ERR_OOB;
    }
    const tsjit_roll_key_t *rk = &eng->keys[key];
    double m = 0.0, s = 0.0;
    roll_get_stats(rk, &m, &s);
    if (out_mean)      *out_mean = m;
    if (out_std)       *out_std = s;
    if (out_anchor)    *out_anchor = rk->anchor;
    if (out_count)     *out_count = rk->count;
    if (out_last_step) *out_last_step = rk->has_step ? rk->last_step : -1;
    return TSJIT_OK;
}

TSJIT_API int32_t tsjit_read_telemetry(tsjit_arrive_engine_t *eng, tsjit_telemetry_t *out) {
    if (eng == NULL || out == NULL) {
        return TSJIT_ERR_BAD_ARG;
    }
    uint64_t w[TSJIT_TELEM_NWORDS];
    int32_t rc = seqlock_read_words(
        &eng->telem->seq, eng->telem->words, TSJIT_TELEM_NWORDS, w, eng->max_spins);
    if (rc != TSJIT_OK) {
        return rc;
    }
    out->total_arrivals    = w[0];
    out->valid_arrivals    = w[1];
    out->dup_rejects       = w[2];
    out->ooo_rejects       = w[3];
    out->nan_stale_count   = w[4];
    out->pwl_fast_hits     = w[5];
    out->pwl_cold_hits     = w[6];
    out->pwl_fallback_hits = w[7];
    out->roll_refreshes    = w[8];
    out->last_step         = (int64_t)w[9];
    out->last_key          = (int64_t)w[10];
    out->last_abs_surprise = tsjit_u64_to_f64(w[11]);
    return TSJIT_OK;
}

/* ---------- Built-In Multi-Threaded Seqlock Stress Test ---------- */

#define STRESS_NWORDS 16

typedef struct stress_slot {
    _Alignas(TSJIT_CACHELINE) _Atomic uint64_t seq;
    _Atomic uint64_t words[STRESS_NWORDS];
    uint8_t _pad[TSJIT_CACHELINE - (((1 + STRESS_NWORDS) * sizeof(uint64_t)) % TSJIT_CACHELINE)];
} stress_slot_t;

typedef struct stress_ctx {
    stress_slot_t    *slot;
    _Atomic uint32_t  stop_writers;
    _Atomic uint32_t  ready_writers;
    int32_t           unguarded;
    uint64_t          reads_per_reader;
} stress_ctx_t;

typedef struct stress_worker_arg {
    stress_ctx_t *ctx;
    int32_t       id;
    uint64_t      reads;
    uint64_t      torn;
    uint64_t      timeouts;
} stress_worker_arg_t;

static inline uint64_t stress_word_val(uint64_t tag, int i) {
    return tag ^ (0x9E3779B97F4A7C15ULL * (uint64_t)(i + 1));
}

static inline uint64_t stress_checksum(const uint64_t w[STRESS_NWORDS]) {
    uint64_t acc = 0xD1B54A32D192ED03ULL;
    for (int i = 0; i < STRESS_NWORDS - 1; ++i) {
        acc ^= w[i] + 0x9E3779B97F4A7C15ULL + (acc << 6) + (acc >> 2);
    }
    return acc;
}

static void *stress_writer_fn(void *arg_ptr) {
    stress_worker_arg_t *arg = (stress_worker_arg_t *)arg_ptr;
    stress_ctx_t *ctx = arg->ctx;
    stress_slot_t *slot = ctx->slot;

    atomic_fetch_add_explicit(&ctx->ready_writers, 1U, memory_order_release);

    uint64_t iter = 1ULL;
    while (!atomic_load_explicit(&ctx->stop_writers, memory_order_acquire)) {
        uint64_t tag = ((uint64_t)(arg->id + 1) << 48) | iter;
        uint64_t payload[STRESS_NWORDS];
        payload[0] = tag;
        for (int i = 1; i < STRESS_NWORDS - 1; ++i) {
            payload[i] = stress_word_val(tag, i);
        }
        payload[STRESS_NWORDS - 1] = stress_checksum(payload);

        if (!ctx->unguarded) {
            uint64_t s = seqlock_write_begin(&slot->seq);
            for (int i = 0; i < STRESS_NWORDS; ++i) {
                atomic_store_explicit(&slot->words[i], payload[i], memory_order_relaxed);
            }
            seqlock_write_end(&slot->seq, s);
            if ((iter & 15ULL) == 0ULL) {
                sched_yield();
            } else {
                tsjit_cpu_relax();
            }
        } else {
            /* Intentionally unguarded atomic-relaxed stores: zero C11 UB, real snapshot tears */
            for (int i = 0; i < STRESS_NWORDS; ++i) {
                atomic_store_explicit(&slot->words[i], payload[i], memory_order_relaxed);
                if ((i & 3) == 3) {
                    tsjit_cpu_relax();
                }
            }
        }
        ++iter;
    }
    return NULL;
}

static void *stress_reader_fn(void *arg_ptr) {
    stress_worker_arg_t *arg = (stress_worker_arg_t *)arg_ptr;
    stress_ctx_t *ctx = arg->ctx;
    stress_slot_t *slot = ctx->slot;

    uint64_t local_reads = 0;
    uint64_t local_torn = 0;
    uint64_t local_timeouts = 0;
    uint64_t buf[STRESS_NWORDS];

    for (uint64_t r = 0; r < ctx->reads_per_reader; ++r) {
        if (!ctx->unguarded) {
            int32_t rc = seqlock_read_words(
                &slot->seq, slot->words, STRESS_NWORDS, buf, TSJIT_DEFAULT_MAX_SPINS);
            if (rc == TSJIT_ERR_SPIN_TIMEOUT) {
                ++local_timeouts;
                continue;
            }
        } else {
            for (int i = 0; i < STRESS_NWORDS; ++i) {
                buf[i] = atomic_load_explicit(&slot->words[i], memory_order_relaxed);
                if ((i & 3) == 3) {
                    tsjit_cpu_relax();
                }
            }
        }
        ++local_reads;
        uint64_t tag = buf[0];
        bool ok = (tag != 0ULL);
        if (ok) {
            for (int i = 1; i < STRESS_NWORDS - 1; ++i) {
                if (buf[i] != stress_word_val(tag, i)) {
                    ok = false;
                    break;
                }
            }
            if (ok && buf[STRESS_NWORDS - 1] != stress_checksum(buf)) {
                ok = false;
            }
        }
        if (!ok) {
            ++local_torn;
        }
    }

    arg->reads = local_reads;
    arg->torn = local_torn;
    arg->timeouts = local_timeouts;
    return NULL;
}

TSJIT_API int32_t tsjit_seqlock_stress_test(
    int32_t n_writers,
    int32_t n_readers,
    uint64_t reads_per_reader,
    int32_t unguarded_control,
    uint64_t *out_total_reads,
    uint64_t *out_torn_reads,
    uint64_t *out_timeouts)
{
    if (n_writers <= 0 || n_writers > 16 || n_readers <= 0 || n_readers > 32 || reads_per_reader == 0) {
        return TSJIT_ERR_BAD_ARG;
    }

    stress_slot_t *slot = (stress_slot_t *)tsjit_aligned_calloc(TSJIT_CACHELINE, sizeof(stress_slot_t));
    if (slot == NULL) {
        return TSJIT_ERR_ALLOC;
    }

    /* Seed initial valid snapshot */
    atomic_init(&slot->seq, 0ULL);
    uint64_t init_tag = (1ULL << 48) | 1ULL;
    uint64_t init_buf[STRESS_NWORDS];
    init_buf[0] = init_tag;
    for (int i = 1; i < STRESS_NWORDS - 1; ++i) {
        init_buf[i] = stress_word_val(init_tag, i);
    }
    init_buf[STRESS_NWORDS - 1] = stress_checksum(init_buf);
    for (int i = 0; i < STRESS_NWORDS; ++i) {
        atomic_init(&slot->words[i], init_buf[i]);
    }

    stress_ctx_t ctx;
    ctx.slot = slot;
    atomic_init(&ctx.stop_writers, 0U);
    atomic_init(&ctx.ready_writers, 0U);
    ctx.unguarded = unguarded_control ? 1 : 0;
    ctx.reads_per_reader = reads_per_reader;

    pthread_t wthreads[16];
    pthread_t rthreads[32];
    stress_worker_arg_t wargs[16];
    stress_worker_arg_t rargs[32];
    memset(wargs, 0, sizeof(wargs));
    memset(rargs, 0, sizeof(rargs));

    for (int32_t w = 0; w < n_writers; ++w) {
        wargs[w].ctx = &ctx;
        wargs[w].id = w;
        if (pthread_create(&wthreads[w], NULL, stress_writer_fn, &wargs[w]) != 0) {
            atomic_store_explicit(&ctx.stop_writers, 1U, memory_order_release);
            for (int32_t j = 0; j < w; ++j) {
                pthread_join(wthreads[j], NULL);
            }
            free(slot);
            return TSJIT_ERR_ALLOC;
        }
    }

    /* Wait until writers are running to guarantee real contention */
    while (atomic_load_explicit(&ctx.ready_writers, memory_order_acquire) < (uint32_t)n_writers) {
        sched_yield();
    }

    for (int32_t r = 0; r < n_readers; ++r) {
        rargs[r].ctx = &ctx;
        rargs[r].id = r;
        if (pthread_create(&rthreads[r], NULL, stress_reader_fn, &rargs[r]) != 0) {
            for (int32_t j = 0; j < r; ++j) {
                pthread_join(rthreads[j], NULL);
            }
            atomic_store_explicit(&ctx.stop_writers, 1U, memory_order_release);
            for (int32_t w = 0; w < n_writers; ++w) {
                pthread_join(wthreads[w], NULL);
            }
            free(slot);
            return TSJIT_ERR_ALLOC;
        }
    }

    uint64_t tot_reads = 0, tot_torn = 0, tot_to = 0;
    for (int32_t r = 0; r < n_readers; ++r) {
        pthread_join(rthreads[r], NULL);
        tot_reads += rargs[r].reads;
        tot_torn += rargs[r].torn;
        tot_to += rargs[r].timeouts;
    }

    atomic_store_explicit(&ctx.stop_writers, 1U, memory_order_release);
    for (int32_t w = 0; w < n_writers; ++w) {
        pthread_join(wthreads[w], NULL);
    }

    free(slot);
    if (out_total_reads) *out_total_reads = tot_reads;
    if (out_torn_reads)  *out_torn_reads = tot_torn;
    if (out_timeouts)    *out_timeouts = tot_to;
    return TSJIT_OK;
}
