/*
 * Copyright 2026 TSJIT Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * Portable C11 Multi-Agent Shared EpochArena (CausalStreamHub native engine):
 *  - Single contiguous posix_memalign numeric arena for all shared channel rings
 *    (next_pow2 capped at 1<<30) and shifted-anchor rolling moments
 *  - Domain-partitioned NaN-isolated EWMA/Chan-Pebay covariance groups keyed by
 *    (gamma, sorted_domain) -> 0.0 cross-agent NaN bleed
 *  - Subnormal- & extreme-decay-safe O(log P) associative monoid merge using
 *    log2(g)*nB (alpha_mant, alpha_exp) + frexp/ldexp normalization + heavier-side
 *    mean anchor + alpha_exp < -1080 -> 0.0 guard + aliasing-safe in-place merge
 *  - Lock-free snapshot publishing + Safe Epoch-Based Reclamation (EBR) for
 *    dynamic output buffer resizing (>100 concurrent resizes, 0 UAF under ASan/TSan)
 *  - Monotonic agent IDs preserving tombstoned bits across hot-adds & resizes
 */

#include "tsjit_common.h"

#include <pthread.h>
#include <sched.h>

#define TSJIT_EBR_MAX_READERS 128
#define TSJIT_MAX_RING_POW2   (1U << 30)
#define TSJIT_MAX_MONOID_DIM  4096U
#define TSJIT_MAX_AGENT_CAP   65536U
#define TSJIT_AGENT_EMPTY     0U
#define TSJIT_AGENT_ALIVE     1U
#define TSJIT_AGENT_TOMBSTONE 2U

/* Per-agent output slot inside published snapshot: Boehm-2012 seqlocked payload */
#define TSJIT_AGENT_OUT_NWORDS 13 /* q[9] + abs_sg + mark + step + version */

typedef struct tsjit_agent_out_slot {
    _Alignas(TSJIT_CACHELINE) _Atomic uint64_t seq;
    _Atomic uint64_t words[TSJIT_AGENT_OUT_NWORDS];
    uint8_t _pad[TSJIT_CACHELINE - (((1 + TSJIT_AGENT_OUT_NWORDS) * sizeof(uint64_t)) % TSJIT_CACHELINE)];
} tsjit_agent_out_slot_t;

_Static_assert(sizeof(tsjit_agent_out_slot_t) % TSJIT_CACHELINE == 0,
               "Agent output slot must be cacheline-multiple sized");

typedef struct tsjit_hub_snapshot {
    uint64_t                   version;
    uint32_t                   capacity;
    uint32_t                   n_agents;
    uint8_t                   *agent_status; /* length capacity */
    tsjit_agent_out_slot_t    *slots;        /* length capacity */
    uint64_t                   retire_epoch;
    struct tsjit_hub_snapshot *retire_next;
} tsjit_hub_snapshot_t;

typedef struct tsjit_ebr_reader_slot {
    _Alignas(TSJIT_CACHELINE) _Atomic uint64_t active_epoch;
    uint8_t _pad[TSJIT_CACHELINE - sizeof(_Atomic uint64_t)];
} tsjit_ebr_reader_slot_t;

_Static_assert(sizeof(tsjit_ebr_reader_slot_t) == TSJIT_CACHELINE,
               "EBR reader slot must occupy exactly one cacheline");

/* Per-channel shifted-anchor rolling moment state (lives inside contiguous arena) */
typedef struct tsjit_chan_roll {
    double   anchor;
    double   sum_d;
    double   sum_d2;
    double   evicted_d2_accum;
    int32_t  scale_exp; /* 0 = unscaled; >0 = sum_d/sum_d2 scaled by 2^{-scale_exp} */
    uint32_t head;
    uint32_t count;
    uint64_t total_valid;
    double   last_valid;
    bool     has_valid;
    double  *ring;
} tsjit_chan_roll_t;

/* Domain-partitioned EWMA/Chan-Pebay covariance group */
typedef struct tsjit_cov_group {
    double   gamma;
    int32_t  dim;
    int32_t *domain; /* sorted channel indices, length dim */
    uint64_t n;
    double   w;
    double  *mean;   /* length dim */
    double  *cov;    /* length dim * dim */
} tsjit_cov_group_t;

typedef struct tsjit_agent_meta {
    int32_t group_id;
    uint8_t status;
} tsjit_agent_meta_t;

struct tsjit_arena {
    int32_t  n_channels;
    uint32_t window;
    uint32_t ring_cap;
    uint32_t ring_mask;

    /* Single contiguous posix_memalign numeric arena */
    void              *numeric_arena;
    tsjit_chan_roll_t *chans;

    /* Domain-partitioned covariance groups */
    tsjit_cov_group_t *groups;
    int32_t            n_groups;
    int32_t            cap_groups;

    /* Agent metadata (writer-owned under write_mu) */
    tsjit_agent_meta_t *agents;
    uint32_t            n_agents;
    uint32_t            meta_cap;

    /* Causal monotonic tick guard (writer-owned under write_mu) */
    bool                has_step;
    int64_t             last_step;

    /* EBR & lock-free snapshot publication */
    pthread_mutex_t          write_mu;
    _Atomic(tsjit_hub_snapshot_t *) current_snap;
    _Atomic uint64_t         global_epoch;
    tsjit_ebr_reader_slot_t *ebr_readers;
    tsjit_hub_snapshot_t    *retired_head;
    uint64_t                 resize_count;
    uint64_t                 reclaimed_count;
    uint64_t                 pending_count;
    uint64_t                 snap_version;
};

/* ---------- Utilities ---------- */

static uint32_t next_pow2_capped(uint32_t v, bool *overflow) {
    *overflow = false;
    if (v == 0U) {
        *overflow = true;
        return 0U;
    }
    if (v > TSJIT_MAX_RING_POW2) {
        *overflow = true;
        return 0U;
    }
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

static size_t align_up_sz(size_t val, size_t align) {
    size_t rem = val % align;
    return (rem == 0) ? val : (val + (align - rem));
}

static void sort_i32(int32_t *a, int32_t n) {
    for (int32_t i = 1; i < n; ++i) {
        int32_t key = a[i];
        int32_t j = i - 1;
        while (j >= 0 && a[j] > key) {
            a[j + 1] = a[j];
            --j;
        }
        a[j + 1] = key;
    }
}

/* ---------- Per-Channel Shifted-Anchor Rolling Stats ---------- */

static void chan_two_pass_refresh(tsjit_chan_roll_t *cr, uint32_t ring_mask) {
    if (cr->count == 0) {
        cr->anchor = 0.0;
        cr->sum_d = 0.0;
        cr->sum_d2 = 0.0;
        cr->evicted_d2_accum = 0.0;
        cr->scale_exp = 0;
        return;
    }
    uint32_t start = (cr->head - cr->count) & ring_mask;
    double v0 = cr->ring[start];
    bool all_equal = true;
    double max_abs_x = 0.0;
    for (uint32_t i = 0; i < cr->count; ++i) {
        double v = cr->ring[(start + i) & ring_mask];
        if (v != v0) {
            all_equal = false;
        }
        double av = fabs(v);
        if (av > max_abs_x) {
            max_abs_x = av;
        }
    }
    if (all_equal) {
        cr->anchor = v0;
        cr->sum_d = 0.0;
        cr->sum_d2 = 0.0;
        cr->evicted_d2_accum = 0.0;
        cr->scale_exp = 0;
        return;
    }

    /* Extreme-magnitude path (max |x_i| > 1e140 or 0 < max |x_i| < 1e-140): exact power-of-2
     * prescaling via frexp/ldexp prevents both 1e308 overflow and 1e-200 square underflow. */
    if ((max_abs_x > 1e140 || (max_abs_x > 0.0 && max_abs_x < 1e-140)) && isfinite(max_abs_x)) {
        int e_max = 0;
        (void)frexp(max_abs_x, &e_max);
        int32_t shift = (int32_t)(e_max - 1);
        double sum_s = 0.0, comp_s = 0.0;
        for (uint32_t i = 0; i < cr->count; ++i) {
            double vs = ldexp(cr->ring[(start + i) & ring_mask], -shift);
            double t = sum_s + vs;
            comp_s += (fabs(sum_s) >= fabs(vs)) ? ((sum_s - t) + vs) : ((vs - t) + sum_s);
            sum_s = t;
        }
        double mu_s = (sum_s + comp_s) / (double)cr->count;
        double mu = ldexp(mu_s, shift);
        if (!isfinite(mu)) {
            mu = copysign(DBL_MAX, mu_s);
        }
        cr->anchor = mu;
        cr->scale_exp = shift;

        double sd_s = 0.0, sd_c = 0.0, sd2_s = 0.0, sd2_c = 0.0;
        for (uint32_t i = 0; i < cr->count; ++i) {
            double vs = ldexp(cr->ring[(start + i) & ring_mask], -shift);
            double ds = vs - mu_s;
            double td = sd_s + ds;
            sd_c += (fabs(sd_s) >= fabs(ds)) ? ((sd_s - td) + ds) : ((ds - td) + sd_s);
            sd_s = td;

            double d2 = ds * ds;
            double td2 = sd2_s + d2;
            sd2_c += (fabs(sd2_s) >= fabs(d2)) ? ((sd2_s - td2) + d2) : ((d2 - td2) + sd2_s);
            sd2_s = td2;
        }
        sd_s += sd_c;
        sd2_s += sd2_c;
        cr->sum_d = sd_s;
        cr->sum_d2 = (sd2_s >= 0.0) ? sd2_s : 0.0;
        cr->evicted_d2_accum = 0.0;
        return;
    }

    double sum = 0.0, comp = 0.0;
    for (uint32_t i = 0; i < cr->count; ++i) {
        double v = cr->ring[(start + i) & ring_mask];
        double t = sum + v;
        comp += (fabs(sum) >= fabs(v)) ? ((sum - t) + v) : ((v - t) + sum);
        sum = t;
    }
    double mu = (sum + comp) / (double)cr->count;
    if (!isfinite(mu)) {
        mu = v0;
    }
    cr->anchor = mu;
    cr->scale_exp = 0;

    double sd = 0.0, sd_c = 0.0, sd2 = 0.0, sd2_c = 0.0;
    for (uint32_t i = 0; i < cr->count; ++i) {
        double d = cr->ring[(start + i) & ring_mask] - mu;
        double td = sd + d;
        sd_c += (fabs(sd) >= fabs(d)) ? ((sd - td) + d) : ((d - td) + sd);
        sd = td;

        double d2 = d * d;
        double td2 = sd2 + d2;
        sd2_c += (fabs(sd2) >= fabs(d2)) ? ((sd2 - td2) + d2) : ((d2 - td2) + sd2);
        sd2 = td2;
    }
    sd += sd_c;
    sd2 += sd2_c;
    cr->sum_d = sd;
    cr->sum_d2 = (sd2 >= 0.0) ? sd2 : 0.0;
    cr->evicted_d2_accum = 0.0;
}

static void chan_roll_update(tsjit_chan_roll_t *cr, uint32_t window, uint32_t ring_mask, double x) {
    if (!cr->has_valid) {
        cr->anchor = x;
        cr->sum_d = 0.0;
        cr->sum_d2 = 0.0;
        cr->evicted_d2_accum = 0.0;
        cr->scale_exp = 0;
        cr->ring[0] = x;
        cr->head = 1U & ring_mask;
        cr->count = 1U;
        cr->total_valid = 1ULL;
        cr->last_valid = x;
        cr->has_valid = true;
        return;
    }

    if (TSJIT_UNLIKELY(cr->scale_exp != 0 ||
                       fabs(x) > 1e140 || fabs(cr->anchor) > 1e140 ||
                       (x != 0.0 && fabs(x) < 1e-140) ||
                       (cr->anchor != 0.0 && fabs(cr->anchor) < 1e-140))) {
        cr->ring[cr->head] = x;
        cr->head = (cr->head + 1U) & ring_mask;
        if (cr->count < window) {
            cr->count++;
        }
        cr->total_valid++;
        cr->last_valid = x;
        chan_two_pass_refresh(cr, ring_mask);
        return;
    }

    double d_new = x - cr->anchor;
    double d_new2 = d_new * d_new;
    bool need_refresh = false;

    if (cr->count == window) {
        uint32_t oldest_idx = (cr->head - window) & ring_mask;
        double x_old = cr->ring[oldest_idx];
        double d_old = x_old - cr->anchor;
        double d_old2 = d_old * d_old;
        double prev_sum_d2 = cr->sum_d2;

        cr->ring[cr->head] = x;
        cr->head = (cr->head + 1U) & ring_mask;

        cr->sum_d += (d_new - d_old);
        cr->sum_d2 += (d_new2 - d_old2);
        cr->evicted_d2_accum += d_old2;

        double ref_prev = fmax(prev_sum_d2, 1e-30);
        double ref_cur  = fmax(cr->sum_d2, 1e-30);
        if (!isfinite(cr->sum_d) || !isfinite(cr->sum_d2) || cr->sum_d2 < 0.0 ||
            d_old2 > 0.05 * ref_prev ||
            d_old2 > 0.05 * ref_cur ||
            cr->evicted_d2_accum > 500.0 * ref_cur) {
            need_refresh = true;
        }
    } else {
        cr->ring[cr->head] = x;
        cr->head = (cr->head + 1U) & ring_mask;
        cr->count++;
        cr->sum_d += d_new;
        cr->sum_d2 += d_new2;
        if (!isfinite(cr->sum_d) || !isfinite(cr->sum_d2)) {
            need_refresh = true;
        }
    }

    cr->total_valid++;
    cr->last_valid = x;

    if (!need_refresh && cr->count > 1U) {
        double inv_n = 1.0 / (double)cr->count;
        double mean_d = cr->sum_d * inv_n;
        double raw_m2 = cr->sum_d2 * inv_n;
        double var_pop = raw_m2 - mean_d * mean_d;
        if (!isfinite(var_pop) || var_pop < 0.0 ||
            (raw_m2 > 0.0 && var_pop < 1e-4 * raw_m2)) {
            need_refresh = true;
        }
    }

    if (need_refresh) {
        chan_two_pass_refresh(cr, ring_mask);
    }
}

/* ---------- Subnormal- & Extreme-Decay-Safe Associative EWMA Monoid ---------- */

static inline double safe_ldexp_i64(double mant, int64_t exp2_i64) {
    if (mant == 0.0 || !isfinite(mant)) {
        return mant;
    }
    int em = 0;
    double mm = frexp(mant, &em);
    int64_t total_exp = exp2_i64 + (int64_t)em;
    if (total_exp < -1080) {
        return 0.0;
    }
    if (total_exp > 1024) {
        return copysign(DBL_MAX, mant);
    }
    double res = ldexp(mm, (int)total_exp);
    if (!isfinite(res)) {
        return copysign(DBL_MAX, mant);
    }
    return res;
}

static bool validate_ewma_monoid(const tsjit_ewma_monoid_t *M) {
    if (!M || M->dim == 0U || M->dim > TSJIT_MAX_MONOID_DIM || !M->mean || !M->cov) {
        return false;
    }
    if (!isfinite(M->w) || M->w < 0.0) {
        return false;
    }
    if (M->n > 0ULL && M->w <= 0.0) {
        return false;
    }
    uint32_t dim = M->dim;
    for (uint32_t i = 0; i < dim; ++i) {
        if (!isfinite(M->mean[i])) {
            return false;
        }
    }
    size_t dd = (size_t)dim * (size_t)dim;
    for (size_t k = 0; k < dd; ++k) {
        if (!isfinite(M->cov[k])) {
            return false;
        }
    }
    return true;
}

/*
 * Exact associative EWMA Chan-Pebay monoid merge: out = merge(A, B) where A is
 * older (nA samples) and B is newer (nB samples) with decay factor gamma in (0, 1].
 *
 * Guarantees:
 *  - Rejects non-finite/negative w, non-finite mean/cov, dim == 0 or dim > 4096 upfront
 *    on ALL paths (including n == 0 and log2_alpha < -1080)
 *  - log2(g) * nB represented as (alpha_mant, alpha_exp) without underflow
 *  - frexp/ldexp weight normalization (immune to subnormals like g=0.99, nB=71500
 *    and g=0.5, nB=1075) + returns TSJIT_ERR_OVERFLOW if w_out would exceed DBL_MAX
 *  - alpha_exp < -1080 -> exact 0.0 guard (zero 0*Inf NaN)
 *  - Heavier-side mean anchor (minimal catastrophic cancellation)
 *  - Factored frexp/ldexp sqrt(pA * pB) * delta cross-term (immune to delta^2 overflow
 *    on extreme jumps like 1e160 + 2000 * 2.0)
 *  - Aliasing-safe for out == A, out == B, or A == B == out
 */
TSJIT_API int32_t tsjit_ewma_monoid_merge(
    double gamma,
    const tsjit_ewma_monoid_t *A,
    const tsjit_ewma_monoid_t *B,
    tsjit_ewma_monoid_t *out)
{
    if (!A || !B || !out || !isfinite(gamma) || gamma <= 0.0 || gamma > 1.0) {
        return TSJIT_ERR_BAD_ARG;
    }
    if (!validate_ewma_monoid(A) || !validate_ewma_monoid(B)) {
        return TSJIT_ERR_BAD_ARG;
    }
    if (A->dim != B->dim || A->dim != out->dim || !out->mean || !out->cov) {
        return TSJIT_ERR_BAD_ARG;
    }

    uint32_t dim = A->dim;
    size_t dd = (size_t)dim * (size_t)dim;

    /* Trivial empty identity cases (inputs already validated above!) */
    if (A->n == 0ULL) {
        out->n = B->n;
        out->w = B->w;
        if (out->mean != B->mean) memmove(out->mean, B->mean, (size_t)dim * sizeof(double));
        if (out->cov  != B->cov)  memmove(out->cov,  B->cov,  dd * sizeof(double));
        return TSJIT_OK;
    }
    if (B->n == 0ULL) {
        out->n = A->n;
        out->w = A->w;
        if (out->mean != A->mean) memmove(out->mean, A->mean, (size_t)dim * sizeof(double));
        if (out->cov  != A->cov)  memmove(out->cov,  A->cov,  dd * sizeof(double));
        return TSJIT_OK;
    }

    uint64_t nA = A->n;
    uint64_t nB = B->n;
    if (nA > UINT64_MAX - nB) {
        return TSJIT_ERR_OVERFLOW;
    }
    double wA = A->w;
    double wB = B->w;

    /* Compute alpha = gamma^{nB} in (mant, exp2) form; guard BEFORE int64_t cast (zero UB!) */
    double log2_alpha = (gamma == 1.0) ? 0.0 : (log2(gamma) * (double)nB);

    /* Extreme-decay guard: alpha_exp < -1080 => alpha == 0.0 */
    if (!isfinite(log2_alpha) || log2_alpha < -1080.0) {
        out->n = nA + nB;
        out->w = wB;
        if (out->mean != B->mean) memmove(out->mean, B->mean, (size_t)dim * sizeof(double));
        if (out->cov  != B->cov)  memmove(out->cov,  B->cov,  dd * sizeof(double));
        return TSJIT_OK;
    }
    int64_t alpha_exp = (int64_t)floor(log2_alpha);
    if (alpha_exp < -1080) {
        out->n = nA + nB;
        out->w = wB;
        if (out->mean != B->mean) memmove(out->mean, B->mean, (size_t)dim * sizeof(double));
        if (out->cov  != B->cov)  memmove(out->cov,  B->cov,  dd * sizeof(double));
        return TSJIT_OK;
    }

    double alpha_frac = log2_alpha - (double)alpha_exp; /* in [0, 1) */
    double alpha_mant = exp2(alpha_frac);               /* in [1, 2) */

    int e_wA_i = 0, e_wB_i = 0, e_adj = 0;
    double m_wA = frexp(wA, &e_wA_i);
    double m_wB = frexp(wB, &e_wB_i);
    double m_effA = frexp(alpha_mant * m_wA, &e_adj);
    int64_t e_effA = alpha_exp + (int64_t)e_wA_i + (int64_t)e_adj;
    int64_t e_wB = (int64_t)e_wB_i;

    /* Normalize effective weights around e_max so u_tot in [0.5, 2.0] */
    int64_t e_max = (e_effA > e_wB) ? e_effA : e_wB;
    double uA = safe_ldexp_i64(m_effA, e_effA - e_max);
    double uB = safe_ldexp_i64(m_wB,   e_wB   - e_max);
    double u_tot = uA + uB;

    /* Exact normalized (mant, exp) representations of pA = wA_eff / w_tot and pB = wB / w_tot */
    int e_utot_i = 0;
    double m_utot = frexp(u_tot, &e_utot_i);
    int64_t e_wout = e_max + (int64_t)e_utot_i;
    if (e_wout > 1024) {
        return TSJIT_ERR_OVERFLOW;
    }
    double w_out = (e_wout < -1080) ? 0.0 : ldexp(m_utot, (int)e_wout);
    if (!isfinite(w_out) || w_out <= 0.0) {
        return TSJIT_ERR_OVERFLOW;
    }

    int e_pA_adj = 0, e_pB_adj = 0;
    double m_pA = frexp(m_effA / m_utot, &e_pA_adj);
    int64_t e_pA = (e_effA - e_max) - (int64_t)e_utot_i + (int64_t)e_pA_adj;
    double m_pB = frexp(m_wB / m_utot, &e_pB_adj);
    int64_t e_pB = (e_wB - e_max) - (int64_t)e_utot_i + (int64_t)e_pB_adj;

    double pA = safe_ldexp_i64(m_pA, e_pA);
    double pB = safe_ldexp_i64(m_pB, e_pB);

    /* Scratch buffers for aliasing-safe computation (stack fast path for dim <= 32) */
    double stack_mean[32];
    double stack_s[32];
    double stack_cov[32 * 32];
    double *tmp_mean = (dim <= 32) ? stack_mean : (double *)malloc((size_t)dim * sizeof(double));
    double *tmp_s    = (dim <= 32) ? stack_s    : (double *)malloc((size_t)dim * sizeof(double));
    double *tmp_cov  = (dim <= 32) ? stack_cov  : (double *)malloc(dd * sizeof(double));
    if (!tmp_mean || !tmp_s || !tmp_cov) {
        if (dim > 32) { free(tmp_mean); free(tmp_s); free(tmp_cov); }
        return TSJIT_ERR_ALLOC;
    }

    /* Heavier-side mean anchor + frexp/ldexp factored sqrt(pA * pB) * delta */
    int e_prod_adj = 0;
    double m_prod = frexp(m_pA * m_pB, &e_prod_adj);
    int64_t e_prod = e_pA + e_pB + (int64_t)e_prod_adj;
    bool cross_active = (e_prod >= -2160);
    int64_t half_e = (e_prod >= 0) ? (e_prod / 2) : ((e_prod - 1) / 2);
    if (e_prod - 2 * half_e != 0) {
        m_prod *= 2.0;
    }
    double sqrt_mprod = sqrt(m_prod);

    bool b_heavier = (e_wB > e_effA) || (e_wB == e_effA && m_wB >= m_effA);

    for (uint32_t i = 0; i < dim; ++i) {
        double mA = A->mean[i];
        double mB = B->mean[i];
        double delta = mB - mA;

        if (!isfinite(delta)) {
            /* Overflow-safe weighted mean when |mB - mA| > DBL_MAX (e.g. -1e308 vs +1e308) */
            double partA = 0.0, partB = 0.0;
            if (mA != 0.0) {
                int eA = 0;
                double mmA = frexp(mA, &eA);
                partA = safe_ldexp_i64(m_pA * mmA, e_pA + (int64_t)eA);
            }
            if (mB != 0.0) {
                int eB = 0;
                double mmB = frexp(mB, &eB);
                partB = safe_ldexp_i64(m_pB * mmB, e_pB + (int64_t)eB);
            }
            tmp_mean[i] = partA + partB;
            if (!cross_active) {
                tmp_s[i] = 0.0;
            } else {
                double half_delta = 0.5 * mB - 0.5 * mA;
                int ed = 0;
                double md = frexp(half_delta, &ed);
                tmp_s[i] = safe_ldexp_i64(sqrt_mprod * md, half_e + (int64_t)ed + 1);
            }
            continue;
        }

        if (b_heavier) {
            /* mean = mB + pA * (mA - mB) computed in mant/exp space to avoid subnormal loss */
            double diff = mA - mB;
            if (diff == 0.0) {
                tmp_mean[i] = mB;
            } else {
                int ed = 0;
                double md = frexp(diff, &ed);
                double step = safe_ldexp_i64(m_pA * md, e_pA + (int64_t)ed);
                tmp_mean[i] = mB + step;
            }
        } else {
            if (delta == 0.0) {
                tmp_mean[i] = mA;
            } else {
                int ed = 0;
                double md = frexp(delta, &ed);
                double step = safe_ldexp_i64(m_pB * md, e_pB + (int64_t)ed);
                tmp_mean[i] = mA + step;
            }
        }

        if (!cross_active || delta == 0.0) {
            tmp_s[i] = 0.0;
        } else {
            int ed = 0;
            double md = frexp(delta, &ed);
            tmp_s[i] = safe_ldexp_i64(sqrt_mprod * md, half_e + (int64_t)ed);
        }
    }

    for (uint32_t i = 0; i < dim; ++i) {
        for (uint32_t j = i; j < dim; ++j) {
            size_t idx = (size_t)i * (size_t)dim + (size_t)j;
            double cA = A->cov[idx];
            double cB = B->cov[idx];

            double termA = 0.0;
            if (cA != 0.0) {
                int ec = 0;
                double mc = frexp(cA, &ec);
                termA = safe_ldexp_i64(m_pA * mc, e_pA + (int64_t)ec);
            }
            double termB = 0.0;
            if (cB != 0.0) {
                int ec = 0;
                double mc = frexp(cB, &ec);
                termB = safe_ldexp_i64(m_pB * mc, e_pB + (int64_t)ec);
            }
            double cross = 0.0;
            if (tmp_s[i] != 0.0 && tmp_s[j] != 0.0) {
                double prod = tmp_s[i] * tmp_s[j];
                cross = isfinite(prod) ? prod : copysign(DBL_MAX, prod);
            }

            double sum_ab = termA + termB;
            if (!isfinite(sum_ab)) {
                sum_ab = copysign(DBL_MAX, sum_ab);
            }
            double val = sum_ab + cross;
            if (!isfinite(val)) {
                val = copysign(DBL_MAX, val);
            }
            if (i == j && val < 0.0) {
                val = 0.0;
            }
            tmp_cov[idx] = val;
            tmp_cov[(size_t)j * (size_t)dim + (size_t)i] = val;
        }
    }

    (void)pA;
    (void)pB;
    out->n = nA + nB;
    out->w = w_out;
    memcpy(out->mean, tmp_mean, (size_t)dim * sizeof(double));
    memcpy(out->cov, tmp_cov, dd * sizeof(double));

    if (dim > 32) {
        free(tmp_mean);
        free(tmp_s);
        free(tmp_cov);
    }
    return TSJIT_OK;
}

/* O(log P) associative tree reduction across n_blocks summaries */
TSJIT_API int32_t tsjit_ewma_monoid_merge_tree(
    double gamma,
    const tsjit_ewma_monoid_t *blocks,
    int32_t n_blocks,
    tsjit_ewma_monoid_t *out)
{
    if (!blocks || n_blocks <= 0 || !out || !isfinite(gamma) || gamma <= 0.0 || gamma > 1.0) {
        return TSJIT_ERR_BAD_ARG;
    }
    uint32_t dim = out->dim;
    if (dim == 0U || dim > TSJIT_MAX_MONOID_DIM || !out->mean || !out->cov) {
        return TSJIT_ERR_BAD_ARG;
    }
    for (int32_t i = 0; i < n_blocks; ++i) {
        if (blocks[i].dim != dim || !validate_ewma_monoid(&blocks[i])) {
            return TSJIT_ERR_BAD_ARG;
        }
    }
    size_t dd = (size_t)dim * (size_t)dim;
    if (n_blocks == 1) {
        out->n = blocks[0].n;
        out->w = blocks[0].w;
        if (out->mean != blocks[0].mean) {
            memmove(out->mean, blocks[0].mean, (size_t)dim * sizeof(double));
        }
        if (out->cov != blocks[0].cov) {
            memmove(out->cov, blocks[0].cov, dd * sizeof(double));
        }
        return TSJIT_OK;
    }

    if ((size_t)n_blocks > (SIZE_MAX / sizeof(double)) / dd) {
        return TSJIT_ERR_OVERFLOW;
    }
    tsjit_ewma_monoid_t *cur = (tsjit_ewma_monoid_t *)calloc((size_t)n_blocks, sizeof(*cur));
    double *mean_pool = (double *)calloc((size_t)n_blocks * (size_t)dim, sizeof(double));
    double *cov_pool  = (double *)calloc((size_t)n_blocks * dd, sizeof(double));
    if (!cur || !mean_pool || !cov_pool) {
        free(cur); free(mean_pool); free(cov_pool);
        return TSJIT_ERR_ALLOC;
    }
    for (int32_t i = 0; i < n_blocks; ++i) {
        cur[i].n = blocks[i].n;
        cur[i].dim = dim;
        cur[i].w = blocks[i].w;
        cur[i].mean = mean_pool + (size_t)i * (size_t)dim;
        cur[i].cov = cov_pool + (size_t)i * dd;
        memcpy(cur[i].mean, blocks[i].mean, (size_t)dim * sizeof(double));
        memcpy(cur[i].cov, blocks[i].cov, dd * sizeof(double));
    }

    int32_t active = n_blocks;
    while (active > 1) {
        int32_t next_active = 0;
        for (int32_t i = 0; i < active; i += 2) {
            if (i + 1 < active) {
                /* In-place merge into cur[next_active] */
                int32_t rc = tsjit_ewma_monoid_merge(gamma, &cur[i], &cur[i + 1], &cur[next_active]);
                if (rc != TSJIT_OK) {
                    free(cur); free(mean_pool); free(cov_pool);
                    return rc;
                }
            } else if (next_active != i) {
                cur[next_active].n = cur[i].n;
                cur[next_active].w = cur[i].w;
                memcpy(cur[next_active].mean, cur[i].mean, (size_t)dim * sizeof(double));
                memcpy(cur[next_active].cov, cur[i].cov, dd * sizeof(double));
            }
            next_active++;
        }
        active = next_active;
    }

    out->n = cur[0].n;
    out->w = cur[0].w;
    memcpy(out->mean, cur[0].mean, (size_t)dim * sizeof(double));
    memcpy(out->cov, cur[0].cov, dd * sizeof(double));

    free(cur);
    free(mean_pool);
    free(cov_pool);
    return TSJIT_OK;
}

/* ---------- Safe Epoch-Based Reclamation (EBR) Snapshot Engine ---------- */

static tsjit_hub_snapshot_t *snapshot_alloc(uint32_t capacity, uint32_t n_agents, uint64_t version) {
    tsjit_hub_snapshot_t *snap = (tsjit_hub_snapshot_t *)calloc(1, sizeof(*snap));
    if (!snap) return NULL;
    snap->version = version;
    snap->capacity = capacity;
    snap->n_agents = n_agents;
    snap->agent_status = (uint8_t *)calloc((size_t)capacity, sizeof(uint8_t));
    snap->slots = (tsjit_agent_out_slot_t *)tsjit_aligned_calloc(
        TSJIT_CACHELINE, (size_t)capacity * sizeof(tsjit_agent_out_slot_t));
    if (!snap->agent_status || !snap->slots) {
        free(snap->agent_status);
        free(snap->slots);
        free(snap);
        return NULL;
    }
    for (uint32_t i = 0; i < capacity; ++i) {
        atomic_init(&snap->slots[i].seq, 0ULL);
        for (int w = 0; w < TSJIT_AGENT_OUT_NWORDS; ++w) {
            atomic_init(&snap->slots[i].words[w], 0ULL);
        }
    }
    return snap;
}

static void snapshot_free(tsjit_hub_snapshot_t *snap) {
    if (!snap) return;
    free(snap->agent_status);
    free(snap->slots);
    free(snap);
}

/* Lock-free EBR pin: claims a reader slot and publishes global_epoch with seq_cst */
static int32_t ebr_pin(tsjit_arena_t *arena) {
    /* Spread initial probe across slots using stack address hash to reduce cacheline contention */
    uintptr_t h = (uintptr_t)&arena;
    int32_t start = (int32_t)((h ^ (h >> 7) ^ (h >> 14)) & (TSJIT_EBR_MAX_READERS - 1));
    for (uint64_t attempt = 0; attempt < TSJIT_DEFAULT_MAX_SPINS; ++attempt) {
        for (int32_t off = 0; off < TSJIT_EBR_MAX_READERS; ++off) {
            int32_t i = (start + off) & (TSJIT_EBR_MAX_READERS - 1);
            uint64_t zero = 0ULL;
            /* (R1) Mark slot active with sentinel 1 (< all retire_epoch >= 2) */
            if (atomic_compare_exchange_weak_explicit(
                    &arena->ebr_readers[i].active_epoch, &zero, 1ULL,
                    memory_order_seq_cst, memory_order_relaxed)) {
                /* (R2) Load global_epoch with seq_cst */
                uint64_t ep = atomic_load_explicit(&arena->global_epoch, memory_order_seq_cst);
                /* (R3) Publish pinned epoch with seq_cst */
                atomic_store_explicit(&arena->ebr_readers[i].active_epoch, ep, memory_order_seq_cst);
                return i;
            }
        }
        if ((attempt & 63ULL) == 63ULL) {
            sched_yield();
        } else {
            tsjit_cpu_relax();
        }
    }
    return -1;
}

static inline void ebr_unpin(tsjit_arena_t *arena, int32_t slot_idx) {
    /* (R5) Release slot so writer's seq_cst scan establishes happens-before free() */
    atomic_store_explicit(&arena->ebr_readers[slot_idx].active_epoch, 0ULL, memory_order_release);
}

/* Must be called with arena->write_mu held */
static void ebr_reclaim_locked(tsjit_arena_t *arena) {
    uint64_t min_active = UINT64_MAX;
    for (int32_t i = 0; i < TSJIT_EBR_MAX_READERS; ++i) {
        uint64_t ep = atomic_load_explicit(&arena->ebr_readers[i].active_epoch, memory_order_seq_cst);
        if (ep != 0ULL && ep < min_active) {
            min_active = ep;
        }
    }

    tsjit_hub_snapshot_t **pp = &arena->retired_head;
    while (*pp != NULL) {
        tsjit_hub_snapshot_t *node = *pp;
        if (node->retire_epoch < min_active) {
            *pp = node->retire_next;
            snapshot_free(node);
            arena->reclaimed_count++;
            if (arena->pending_count > 0) {
                arena->pending_count--;
            }
        } else {
            pp = &node->retire_next;
        }
    }
}

/* Must be called with arena->write_mu held */
static int32_t arena_resize_snapshot_locked(tsjit_arena_t *arena, uint32_t new_capacity) {
    tsjit_hub_snapshot_t *old_snap = atomic_load_explicit(&arena->current_snap, memory_order_relaxed);
    if (new_capacity < arena->n_agents) {
        return TSJIT_ERR_BAD_ARG;
    }
    if (new_capacity == 0) {
        new_capacity = 4;
    }

    tsjit_hub_snapshot_t *new_snap = snapshot_alloc(new_capacity, arena->n_agents, ++arena->snap_version);
    if (!new_snap) {
        return TSJIT_ERR_ALLOC;
    }

    /* Copy ALL authoritative agent status bits (preserving ALIVE and TOMBSTONED!)
       BEFORE publishing new_snap so agent_status is strictly immutable to readers */
    uint32_t copy_n = (arena->n_agents < new_capacity) ? arena->n_agents : new_capacity;
    for (uint32_t a = 0; a < copy_n; ++a) {
        new_snap->agent_status[a] = arena->agents[a].status;
    }
    if (old_snap != NULL) {
        uint32_t copy_slots = (old_snap->n_agents < copy_n) ? old_snap->n_agents : copy_n;
        for (uint32_t a = 0; a < copy_slots; ++a) {
            for (int w = 0; w < TSJIT_AGENT_OUT_NWORDS; ++w) {
                uint64_t val = atomic_load_explicit(&old_snap->slots[a].words[w], memory_order_relaxed);
                atomic_store_explicit(&new_snap->slots[a].words[w], val, memory_order_relaxed);
            }
        }
    }

    /* (W1) Atomically publish new snapshot */
    tsjit_hub_snapshot_t *prev = atomic_exchange_explicit(
        &arena->current_snap, new_snap, memory_order_seq_cst);
    /* (W2) Advance monotonic global_epoch */
    uint64_t retire_ep = atomic_fetch_add_explicit(&arena->global_epoch, 1ULL, memory_order_seq_cst);
    arena->resize_count++;

    if (prev != NULL) {
        prev->retire_epoch = retire_ep;
        prev->retire_next = arena->retired_head;
        arena->retired_head = prev;
        arena->pending_count++;
    }

    /* (W3-W4) Safe EBR reclamation of snapshots older than all active readers */
    ebr_reclaim_locked(arena);
    return TSJIT_OK;
}

/* ---------- Arena Lifecycle & Agent Management ---------- */

TSJIT_API tsjit_arena_t *tsjit_arena_create(
    int32_t n_channels,
    uint32_t window,
    uint32_t init_agent_cap)
{
    if (n_channels <= 0 || n_channels > 65536 || window == 0U || init_agent_cap > TSJIT_MAX_AGENT_CAP) {
        return NULL;
    }
    bool ovf = false;
    uint32_t ring_cap = next_pow2_capped(window, &ovf);
    if (ovf || ring_cap == 0U) {
        return NULL;
    }
    if ((uint64_t)n_channels > (SIZE_MAX / sizeof(double)) / (uint64_t)ring_cap) {
        return NULL;
    }
    if (init_agent_cap == 0U) {
        init_agent_cap = 4U;
    }

    tsjit_arena_t *arena = (tsjit_arena_t *)calloc(1, sizeof(*arena));
    if (!arena) {
        return NULL;
    }
    arena->n_channels = n_channels;
    arena->window = window;
    arena->ring_cap = ring_cap;
    arena->ring_mask = ring_cap - 1U;

    /* One contiguous posix_memalign numeric arena for chan structs + all channel rings */
    size_t chans_bytes = align_up_sz((size_t)n_channels * sizeof(tsjit_chan_roll_t), TSJIT_CACHELINE);
    size_t rings_bytes = (size_t)n_channels * (size_t)ring_cap * sizeof(double);
    if (chans_bytes > SIZE_MAX - rings_bytes) {
        free(arena);
        return NULL;
    }
    arena->numeric_arena = tsjit_aligned_calloc(TSJIT_CACHELINE, chans_bytes + rings_bytes);
    arena->ebr_readers = (tsjit_ebr_reader_slot_t *)tsjit_aligned_calloc(
        TSJIT_CACHELINE, TSJIT_EBR_MAX_READERS * sizeof(tsjit_ebr_reader_slot_t));
    arena->agents = (tsjit_agent_meta_t *)calloc((size_t)init_agent_cap, sizeof(tsjit_agent_meta_t));
    arena->meta_cap = init_agent_cap;

    tsjit_hub_snapshot_t *init_snap = snapshot_alloc(init_agent_cap, 0U, 1ULL);

    if (!arena->numeric_arena || !arena->ebr_readers || !arena->agents || !init_snap) {
        snapshot_free(init_snap);
        free(arena->numeric_arena);
        free(arena->ebr_readers);
        free(arena->agents);
        free(arena);
        return NULL;
    }

    arena->chans = (tsjit_chan_roll_t *)arena->numeric_arena;
    double *ring_base = (double *)((uint8_t *)arena->numeric_arena + chans_bytes);
    for (int32_t c = 0; c < n_channels; ++c) {
        arena->chans[c].ring = ring_base + (size_t)c * (size_t)ring_cap;
    }

    for (int32_t i = 0; i < TSJIT_EBR_MAX_READERS; ++i) {
        atomic_init(&arena->ebr_readers[i].active_epoch, 0ULL);
    }
    atomic_init(&arena->global_epoch, 2ULL);
    atomic_init(&arena->current_snap, init_snap);
    arena->snap_version = 1ULL;
    pthread_mutex_init(&arena->write_mu, NULL);
    return arena;
}

TSJIT_API void tsjit_arena_destroy(tsjit_arena_t *arena) {
    if (!arena) return;
    pthread_mutex_lock(&arena->write_mu);
    tsjit_hub_snapshot_t *cur = atomic_exchange_explicit(&arena->current_snap, NULL, memory_order_seq_cst);
    snapshot_free(cur);
    tsjit_hub_snapshot_t *node = arena->retired_head;
    while (node != NULL) {
        tsjit_hub_snapshot_t *next = node->retire_next;
        snapshot_free(node);
        node = next;
    }
    arena->retired_head = NULL;
    for (int32_t g = 0; g < arena->n_groups; ++g) {
        free(arena->groups[g].domain);
        free(arena->groups[g].mean);
        free(arena->groups[g].cov);
    }
    free(arena->groups);
    free(arena->agents);
    free(arena->ebr_readers);
    free(arena->numeric_arena);
    pthread_mutex_unlock(&arena->write_mu);
    pthread_mutex_destroy(&arena->write_mu);
    free(arena);
}

static int32_t find_or_create_group_locked(
    tsjit_arena_t *arena,
    const int32_t *sorted_dom,
    int32_t dim,
    double gamma)
{
    for (int32_t g = 0; g < arena->n_groups; ++g) {
        tsjit_cov_group_t *cg = &arena->groups[g];
        if (cg->dim == dim && fabs(cg->gamma - gamma) <= 1e-15) {
            if (memcmp(cg->domain, sorted_dom, (size_t)dim * sizeof(int32_t)) == 0) {
                return g;
            }
        }
    }

    if (arena->n_groups >= arena->cap_groups) {
        int32_t new_cap = (arena->cap_groups == 0) ? 8 : (arena->cap_groups * 2);
        tsjit_cov_group_t *ng = (tsjit_cov_group_t *)realloc(
            arena->groups, (size_t)new_cap * sizeof(tsjit_cov_group_t));
        if (!ng) return -1;
        arena->groups = ng;
        arena->cap_groups = new_cap;
    }

    int32_t gid = arena->n_groups;
    tsjit_cov_group_t *cg = &arena->groups[gid];
    memset(cg, 0, sizeof(*cg));
    cg->gamma = gamma;
    cg->dim = dim;
    cg->domain = (int32_t *)malloc((size_t)dim * sizeof(int32_t));
    cg->mean = (double *)calloc((size_t)dim, sizeof(double));
    cg->cov = (double *)calloc((size_t)dim * (size_t)dim, sizeof(double));
    if (!cg->domain || !cg->mean || !cg->cov) {
        free(cg->domain); free(cg->mean); free(cg->cov);
        return -1;
    }
    memcpy(cg->domain, sorted_dom, (size_t)dim * sizeof(int32_t));

    /* Zero-replay warm-start from already-computed shared channel rolling stats */
    bool all_have_valid = true;
    uint32_t min_cnt = UINT32_MAX;
    for (int32_t i = 0; i < dim; ++i) {
        const tsjit_chan_roll_t *cr = &arena->chans[sorted_dom[i]];
        if (!cr->has_valid || cr->count == 0) {
            all_have_valid = false;
            break;
        }
        if (cr->count < min_cnt) min_cnt = cr->count;
    }
    if (all_have_valid && min_cnt > 0U) {
        /* Warm-start mean & diagonal variance from shared shifted-anchor stats in O(dim) */
        for (int32_t i = 0; i < dim; ++i) {
            const tsjit_chan_roll_t *cr = &arena->chans[sorted_dom[i]];
            double inv_n = 1.0 / (double)cr->count;
            double md = cr->sum_d * inv_n;
            if (cr->scale_exp == 0) {
                double m = cr->anchor + md;
                cg->mean[i] = isfinite(m) ? m : cr->anchor;
                double var = (cr->count > 1U)
                    ? fmax(0.0, (cr->sum_d2 - cr->sum_d * md) / (double)cr->count)
                    : 0.0;
                cg->cov[(size_t)i * (size_t)dim + (size_t)i] = isfinite(var) ? var : DBL_MAX;
            } else {
                double m = cr->anchor + ldexp(md, cr->scale_exp);
                cg->mean[i] = isfinite(m) ? m : cr->anchor;
                double var = 0.0;
                if (cr->count > 1U) {
                    double ss = fmax(0.0, cr->sum_d2 - cr->sum_d * md);
                    double var_s = ss / (double)cr->count;
                    var = safe_ldexp_i64(var_s, 2 * (int64_t)cr->scale_exp);
                }
                cg->cov[(size_t)i * (size_t)dim + (size_t)i] =
                    (isfinite(var) && var >= 0.0) ? var : DBL_MAX;
            }
        }
        cg->n = (uint64_t)min_cnt;
        cg->w = (gamma == 1.0)
            ? (double)min_cnt
            : (-expm1(log(gamma) * (double)min_cnt) / (1.0 - gamma));
        if (!isfinite(cg->w) || cg->w <= 0.0) {
            cg->w = (double)min_cnt;
        }
    }

    arena->n_groups++;
    return gid;
}

TSJIT_API int32_t tsjit_arena_add_agent(
    tsjit_arena_t *arena,
    const int32_t *domain_channels,
    int32_t n_domain,
    double ewma_gamma,
    int32_t *out_agent_id)
{
    if (!arena || !domain_channels || n_domain <= 0 || n_domain > arena->n_channels ||
        (uint32_t)n_domain > TSJIT_MAX_MONOID_DIM ||
        !isfinite(ewma_gamma) || ewma_gamma <= 0.0 || ewma_gamma > 1.0 || !out_agent_id) {
        return TSJIT_ERR_BAD_ARG;
    }

    int32_t *sorted_dom = (int32_t *)malloc((size_t)n_domain * sizeof(int32_t));
    if (!sorted_dom) return TSJIT_ERR_ALLOC;
    for (int32_t i = 0; i < n_domain; ++i) {
        if (domain_channels[i] < 0 || domain_channels[i] >= arena->n_channels) {
            free(sorted_dom);
            return TSJIT_ERR_OOB;
        }
        sorted_dom[i] = domain_channels[i];
    }
    sort_i32(sorted_dom, n_domain);
    for (int32_t i = 1; i < n_domain; ++i) {
        if (sorted_dom[i] == sorted_dom[i - 1]) {
            free(sorted_dom);
            return TSJIT_ERR_BAD_ARG;
        }
    }

    pthread_mutex_lock(&arena->write_mu);

    int32_t gid = find_or_create_group_locked(arena, sorted_dom, n_domain, ewma_gamma);
    free(sorted_dom);
    if (gid < 0) {
        pthread_mutex_unlock(&arena->write_mu);
        return TSJIT_ERR_ALLOC;
    }

    uint32_t aid = arena->n_agents;
    if (aid >= TSJIT_MAX_AGENT_CAP) {
        pthread_mutex_unlock(&arena->write_mu);
        return TSJIT_ERR_OVERFLOW;
    }
    if (aid >= arena->meta_cap) {
        uint32_t new_mcap = arena->meta_cap * 2U;
        if (new_mcap <= aid) new_mcap = aid + 4U;
        tsjit_agent_meta_t *nm = (tsjit_agent_meta_t *)realloc(
            arena->agents, (size_t)new_mcap * sizeof(tsjit_agent_meta_t));
        if (!nm) {
            pthread_mutex_unlock(&arena->write_mu);
            return TSJIT_ERR_ALLOC;
        }
        arena->agents = nm;
        arena->meta_cap = new_mcap;
    }

    arena->agents[aid].group_id = gid;
    arena->agents[aid].status = TSJIT_AGENT_ALIVE;
    arena->n_agents = aid + 1U;

    tsjit_hub_snapshot_t *snap = atomic_load_explicit(&arena->current_snap, memory_order_relaxed);
    uint32_t target_cap = snap->capacity;
    if (arena->n_agents > target_cap) {
        /* Dynamic resize via safe EBR (preserves all previous ALIVE and TOMBSTONED bits!) */
        target_cap = snap->capacity * 2U;
        if (target_cap < arena->n_agents) target_cap = arena->n_agents;
    }
    int32_t rc = arena_resize_snapshot_locked(arena, target_cap);
    if (rc != TSJIT_OK) {
        arena->n_agents = aid;
        pthread_mutex_unlock(&arena->write_mu);
        return rc;
    }

    *out_agent_id = (int32_t)aid;
    pthread_mutex_unlock(&arena->write_mu);
    return TSJIT_OK;
}

TSJIT_API int32_t tsjit_arena_tombstone_agent(tsjit_arena_t *arena, int32_t agent_id) {
    if (!arena) return TSJIT_ERR_BAD_ARG;
    pthread_mutex_lock(&arena->write_mu);
    if (agent_id < 0 || (uint32_t)agent_id >= arena->n_agents) {
        pthread_mutex_unlock(&arena->write_mu);
        return TSJIT_ERR_OOB;
    }
    if (arena->agents[agent_id].status == TSJIT_AGENT_TOMBSTONE) {
        pthread_mutex_unlock(&arena->write_mu);
        return TSJIT_OK;
    }
    arena->agents[agent_id].status = TSJIT_AGENT_TOMBSTONE;
    tsjit_hub_snapshot_t *snap = atomic_load_explicit(&arena->current_snap, memory_order_relaxed);
    int32_t rc = arena_resize_snapshot_locked(arena, snap->capacity);
    pthread_mutex_unlock(&arena->write_mu);
    return rc;
}

TSJIT_API int32_t tsjit_arena_agent_status(tsjit_arena_t *arena, int32_t agent_id) {
    if (!arena || agent_id < 0) return -1;
    int32_t pin = ebr_pin(arena);
    if (pin < 0) return TSJIT_ERR_SPIN_TIMEOUT;
    tsjit_hub_snapshot_t *snap = atomic_load_explicit(&arena->current_snap, memory_order_seq_cst);
    int32_t st = ((uint32_t)agent_id < snap->n_agents) ? (int32_t)snap->agent_status[agent_id] : -1;
    ebr_unpin(arena, pin);
    return st;
}

/* ---------- Shared Tick Ingest (NaN-Isolated per Domain Group) ---------- */

TSJIT_API int32_t tsjit_arena_ingest_tick(
    tsjit_arena_t *arena,
    int64_t step,
    const double *values,
    int32_t n_values)
{
    if (!arena || !values || n_values != arena->n_channels) {
        return TSJIT_ERR_BAD_ARG;
    }

    pthread_mutex_lock(&arena->write_mu);

    /* Strictly enforce causal monotonic steps: reject DUP/OOO with zero state mutation */
    if (arena->has_step) {
        if (step == arena->last_step) {
            pthread_mutex_unlock(&arena->write_mu);
            return MARK_DUP_STEP;
        }
        if (step < arena->last_step) {
            pthread_mutex_unlock(&arena->write_mu);
            return MARK_OOO_STEP;
        }
    }
    arena->has_step = true;
    arena->last_step = step;

    /* 1. Update shared per-channel rolling rings & shifted-anchor stats once */
    for (int32_t c = 0; c < arena->n_channels; ++c) {
        double v = values[c];
        if (isfinite(v)) {
            chan_roll_update(&arena->chans[c], arena->window, arena->ring_mask, v);
        }
    }

    /* 2. Update each domain-partitioned covariance group reading ONLY its domain */
    double stack_x[32];
    double stack_zcov[32 * 32];

    for (int32_t g = 0; g < arena->n_groups; ++g) {
        tsjit_cov_group_t *cg = &arena->groups[g];
        int32_t dim = cg->dim;
        double *x_dom = (dim <= 32) ? stack_x : (double *)malloc((size_t)dim * sizeof(double));
        double *z_cov = (dim <= 32) ? stack_zcov : (double *)calloc((size_t)dim * (size_t)dim, sizeof(double));
        if (!x_dom || !z_cov) {
            if (dim > 32) { free(x_dom); free(z_cov); }
            pthread_mutex_unlock(&arena->write_mu);
            return TSJIT_ERR_ALLOC;
        }
        if (dim <= 32) {
            memset(z_cov, 0, (size_t)dim * (size_t)dim * sizeof(double));
        }

        bool all_finite = true;
        for (int32_t i = 0; i < dim; ++i) {
            double v = values[cg->domain[i]];
            if (!isfinite(v)) {
                all_finite = false;
                break;
            }
            x_dom[i] = v;
        }

        if (all_finite) {
            tsjit_ewma_monoid_t cur = {
                .n = cg->n,
                .dim = (uint32_t)dim,
                ._pad = 0,
                .w = cg->w,
                .mean = cg->mean,
                .cov = cg->cov
            };
            tsjit_ewma_monoid_t single = {
                .n = 1ULL,
                .dim = (uint32_t)dim,
                ._pad = 0,
                .w = 1.0,
                .mean = x_dom,
                .cov = z_cov
            };
            int32_t mrc = tsjit_ewma_monoid_merge(cg->gamma, &cur, &single, &cur);
            if (mrc == TSJIT_OK) {
                cg->n = cur.n;
                cg->w = cur.w;
            } else if (mrc == TSJIT_ERR_ALLOC) {
                if (dim > 32) {
                    free(x_dom);
                    free(z_cov);
                }
                pthread_mutex_unlock(&arena->write_mu);
                return TSJIT_ERR_ALLOC;
            } else {
                /* Never let a bad or overflowed merge permanently poison or freeze the group. */
                bool recovered = false;
                if (mrc == TSJIT_ERR_OVERFLOW) {
                    if (cur.n == UINT64_MAX) {
                        cur.n = UINT64_MAX / 2ULL;
                    }
                    if (isfinite(cur.w) && cur.w > 1e100) {
                        cur.w *= 0.5;
                    }
                    int32_t retry_rc = tsjit_ewma_monoid_merge(cg->gamma, &cur, &single, &cur);
                    if (retry_rc == TSJIT_OK) {
                        cg->n = cur.n;
                        cg->w = cur.w;
                        recovered = true;
                    } else if (retry_rc == TSJIT_ERR_ALLOC) {
                        if (dim > 32) {
                            free(x_dom);
                            free(z_cov);
                        }
                        pthread_mutex_unlock(&arena->write_mu);
                        return TSJIT_ERR_ALLOC;
                    }
                }
                if (!recovered) {
                    /* Re-anchor cleanly onto current finite sample so the group never freezes. */
                    cg->n = 1ULL;
                    cg->w = 1.0;
                    memcpy(cg->mean, x_dom, (size_t)dim * sizeof(double));
                    memset(cg->cov, 0, (size_t)dim * (size_t)dim * sizeof(double));
                }
            }
        }

        if (dim > 32) {
            free(x_dom);
            free(z_cov);
        }
    }

    ebr_reclaim_locked(arena);
    pthread_mutex_unlock(&arena->write_mu);
    return TSJIT_OK;
}

/* ---------- Lock-Free Agent Output Publish & Snapshot Read ---------- */

TSJIT_API int32_t tsjit_arena_publish_agent_output(
    tsjit_arena_t *arena,
    int32_t agent_id,
    int64_t step,
    const double q[TSJIT_NQ],
    double abs_sg,
    int32_t mark)
{
    if (!arena || !q) return TSJIT_ERR_BAD_ARG;
    pthread_mutex_lock(&arena->write_mu);
    if (agent_id < 0 || (uint32_t)agent_id >= arena->n_agents) {
        pthread_mutex_unlock(&arena->write_mu);
        return TSJIT_ERR_OOB;
    }
    if (arena->agents[agent_id].status != TSJIT_AGENT_ALIVE) {
        pthread_mutex_unlock(&arena->write_mu);
        return TSJIT_ERR_BAD_ARG;
    }
    tsjit_hub_snapshot_t *snap = atomic_load_explicit(&arena->current_snap, memory_order_relaxed);
    tsjit_agent_out_slot_t *slot = &snap->slots[agent_id];
    uint64_t s = atomic_load_explicit(&slot->seq, memory_order_relaxed);
    atomic_store_explicit(&slot->seq, s + 1ULL, memory_order_release);
    atomic_thread_fence(memory_order_release);

    for (int i = 0; i < TSJIT_NQ; ++i) {
        atomic_store_explicit(&slot->words[i], tsjit_f64_to_u64(q[i]), memory_order_relaxed);
    }
    atomic_store_explicit(&slot->words[9], tsjit_f64_to_u64(abs_sg), memory_order_relaxed);
    atomic_store_explicit(&slot->words[10], (uint64_t)(int64_t)mark, memory_order_relaxed);
    atomic_store_explicit(&slot->words[11], (uint64_t)step, memory_order_relaxed);
    atomic_store_explicit(&slot->words[12], snap->version, memory_order_relaxed);

    atomic_store_explicit(&slot->seq, s + 2ULL, memory_order_release);
    pthread_mutex_unlock(&arena->write_mu);
    return TSJIT_OK;
}

TSJIT_API int32_t tsjit_arena_read_agent_snapshot(
    tsjit_arena_t *arena,
    int32_t agent_id,
    double out_q[TSJIT_NQ],
    double *out_abs_sg,
    int32_t *out_mark,
    int64_t *out_step,
    uint64_t *out_version)
{
    if (!arena || !out_q || agent_id < 0) return TSJIT_ERR_BAD_ARG;

    int32_t pin = ebr_pin(arena);
    if (pin < 0) return TSJIT_ERR_SPIN_TIMEOUT;

    /* (R4) Acquire current immutable snapshot pointer */
    tsjit_hub_snapshot_t *snap = atomic_load_explicit(&arena->current_snap, memory_order_seq_cst);
    if ((uint32_t)agent_id >= snap->n_agents) {
        ebr_unpin(arena, pin);
        return TSJIT_ERR_OOB;
    }
    if (snap->agent_status[agent_id] != TSJIT_AGENT_ALIVE) {
        ebr_unpin(arena, pin);
        return TSJIT_ERR_BAD_ARG;
    }

    const tsjit_agent_out_slot_t *slot = &snap->slots[agent_id];
    uint64_t w[TSJIT_AGENT_OUT_NWORDS];
    uint64_t spins = 0;
    for (;;) {
        uint64_t s1 = atomic_load_explicit(&slot->seq, memory_order_acquire);
        if (s1 & 1ULL) {
            if (++spins >= TSJIT_DEFAULT_MAX_SPINS) {
                ebr_unpin(arena, pin);
                return TSJIT_ERR_SPIN_TIMEOUT;
            }
            if ((spins & 1023ULL) == 0ULL) sched_yield(); else tsjit_cpu_relax();
            continue;
        }
        for (int i = 0; i < TSJIT_AGENT_OUT_NWORDS; ++i) {
            w[i] = atomic_load_explicit(&slot->words[i], memory_order_relaxed);
        }
        atomic_thread_fence(memory_order_acquire);
        uint64_t s2 = atomic_load_explicit(&slot->seq, memory_order_relaxed);
        if (s1 == s2) {
            break;
        }
        if (++spins >= TSJIT_DEFAULT_MAX_SPINS) {
            ebr_unpin(arena, pin);
            return TSJIT_ERR_SPIN_TIMEOUT;
        }
        if ((spins & 1023ULL) == 0ULL) sched_yield(); else tsjit_cpu_relax();
    }

    ebr_unpin(arena, pin);

    for (int i = 0; i < TSJIT_NQ; ++i) {
        out_q[i] = tsjit_u64_to_f64(w[i]);
    }
    if (out_abs_sg)  *out_abs_sg = tsjit_u64_to_f64(w[9]);
    if (out_mark)    *out_mark = (int32_t)(int64_t)w[10];
    if (out_step)    *out_step = (int64_t)w[11];
    if (out_version) *out_version = w[12];
    return TSJIT_OK;
}

TSJIT_API int32_t tsjit_arena_get_channel_stats(
    tsjit_arena_t *arena,
    int32_t channel,
    double *out_mean,
    double *out_std,
    double *out_anchor,
    uint64_t *out_count)
{
    if (!arena || channel < 0 || channel >= arena->n_channels) {
        return TSJIT_ERR_OOB;
    }
    pthread_mutex_lock(&arena->write_mu);
    const tsjit_chan_roll_t *cr = &arena->chans[channel];
    double mean = 0.0, sd = 0.0;
    if (cr->has_valid && cr->count > 0) {
        double inv_n = 1.0 / (double)cr->count;
        double md = cr->sum_d * inv_n;
        if (cr->scale_exp == 0) {
            mean = cr->anchor + md;
            if (cr->count > 1U) {
                double ss = cr->sum_d2 - cr->sum_d * md;
                sd = (ss > 0.0 && isfinite(ss)) ? sqrt(ss / (double)(cr->count - 1U)) : 0.0;
            }
        } else {
            double m = cr->anchor + ldexp(md, cr->scale_exp);
            mean = isfinite(m) ? m : cr->anchor;
            if (cr->count > 1U) {
                double ss = cr->sum_d2 - cr->sum_d * md;
                if (ss > 0.0 && isfinite(ss)) {
                    double sd_s = sqrt(ss / (double)(cr->count - 1U));
                    double s = ldexp(sd_s, cr->scale_exp);
                    sd = isfinite(s) ? s : DBL_MAX;
                }
            }
        }
    }
    if (out_mean)   *out_mean = mean;
    if (out_std)    *out_std = sd;
    if (out_anchor) *out_anchor = cr->anchor;
    if (out_count)  *out_count = cr->total_valid;
    pthread_mutex_unlock(&arena->write_mu);
    return TSJIT_OK;
}

TSJIT_API int32_t tsjit_arena_get_agent_cov(
    tsjit_arena_t *arena,
    int32_t agent_id,
    int32_t expected_dim,
    double *out_mean,
    double *out_cov,
    uint64_t *out_n,
    int32_t *out_group_id)
{
    if (!arena || agent_id < 0) return TSJIT_ERR_BAD_ARG;
    pthread_mutex_lock(&arena->write_mu);
    if ((uint32_t)agent_id >= arena->n_agents) {
        pthread_mutex_unlock(&arena->write_mu);
        return TSJIT_ERR_OOB;
    }
    int32_t gid = arena->agents[agent_id].group_id;
    const tsjit_cov_group_t *cg = &arena->groups[gid];
    if (expected_dim != cg->dim) {
        pthread_mutex_unlock(&arena->write_mu);
        return TSJIT_ERR_BAD_ARG;
    }
    if (out_mean)     memcpy(out_mean, cg->mean, (size_t)cg->dim * sizeof(double));
    if (out_cov)      memcpy(out_cov, cg->cov, (size_t)cg->dim * (size_t)cg->dim * sizeof(double));
    if (out_n)        *out_n = cg->n;
    if (out_group_id) *out_group_id = gid;
    pthread_mutex_unlock(&arena->write_mu);
    return TSJIT_OK;
}

TSJIT_API int32_t tsjit_arena_force_resize(tsjit_arena_t *arena, uint32_t new_capacity) {
    if (!arena || new_capacity == 0U || new_capacity > (1U << 20)) {
        return TSJIT_ERR_BAD_ARG;
    }
    pthread_mutex_lock(&arena->write_mu);
    int32_t rc = arena_resize_snapshot_locked(arena, new_capacity);
    pthread_mutex_unlock(&arena->write_mu);
    return rc;
}

TSJIT_API int32_t tsjit_arena_ebr_stats(
    tsjit_arena_t *arena,
    uint32_t *out_capacity,
    uint64_t *out_resizes,
    uint64_t *out_reclaimed,
    uint64_t *out_pending)
{
    if (!arena) return TSJIT_ERR_BAD_ARG;
    pthread_mutex_lock(&arena->write_mu);
    tsjit_hub_snapshot_t *snap = atomic_load_explicit(&arena->current_snap, memory_order_relaxed);
    ebr_reclaim_locked(arena);
    if (out_capacity)  *out_capacity = snap ? snap->capacity : 0U;
    if (out_resizes)   *out_resizes = arena->resize_count;
    if (out_reclaimed) *out_reclaimed = arena->reclaimed_count;
    if (out_pending)   *out_pending = arena->pending_count;
    pthread_mutex_unlock(&arena->write_mu);
    return TSJIT_OK;
}

/* ---------- Built-In Concurrent EBR Dynamic Resize Stress Test ---------- */

typedef struct ebr_stress_ctx {
    tsjit_arena_t   *arena;
    _Atomic uint32_t stop;
    _Atomic uint32_t ready;
} ebr_stress_ctx_t;

typedef struct ebr_reader_arg {
    ebr_stress_ctx_t *ctx;
    uint64_t          reads;
    uint64_t          errors;
} ebr_reader_arg_t;

static void *ebr_reader_fn(void *p) {
    ebr_reader_arg_t *arg = (ebr_reader_arg_t *)p;
    ebr_stress_ctx_t *ctx = arg->ctx;
    atomic_fetch_add_explicit(&ctx->ready, 1U, memory_order_release);

    uint64_t reads = 0, errs = 0;
    double q[TSJIT_NQ];
    while (!atomic_load_explicit(&ctx->stop, memory_order_acquire)) {
        double abs_sg = 0.0;
        int32_t mark = 0;
        int64_t step = 0;
        uint64_t ver = 0;
        int32_t rc = tsjit_arena_read_agent_snapshot(ctx->arena, 0, q, &abs_sg, &mark, &step, &ver);
        if (rc != TSJIT_OK) {
            errs++;
        } else {
            /* Check monotonic invariant on published quantiles */
            for (int i = 1; i < TSJIT_NQ; ++i) {
                if (q[i] < q[i - 1]) {
                    errs++;
                    break;
                }
            }
            reads++;
        }
    }
    arg->reads = reads;
    arg->errors = errs;
    return NULL;
}

TSJIT_API int32_t tsjit_arena_ebr_stress_test(
    int32_t n_readers,
    int32_t n_resizes,
    uint64_t *out_reads,
    uint64_t *out_reclaimed,
    uint64_t *out_errors)
{
    if (n_readers <= 0 || n_readers > 32 || n_resizes < 1) {
        return TSJIT_ERR_BAD_ARG;
    }

    tsjit_arena_t *arena = tsjit_arena_create(4, 32, 4);
    if (!arena) return TSJIT_ERR_ALLOC;

    int32_t dom[2] = {0, 1};
    int32_t aid0 = -1;
    if (tsjit_arena_add_agent(arena, dom, 2, 0.95, &aid0) != TSJIT_OK) {
        tsjit_arena_destroy(arena);
        return TSJIT_ERR_ALLOC;
    }
    double init_q[TSJIT_NQ] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    tsjit_arena_publish_agent_output(arena, aid0, 0, init_q, 0.1, MARK_OK);

    ebr_stress_ctx_t ctx = { .arena = arena };
    atomic_init(&ctx.stop, 0U);
    atomic_init(&ctx.ready, 0U);

    pthread_t threads[32];
    ebr_reader_arg_t args[32];
    memset(args, 0, sizeof(args));

    for (int32_t r = 0; r < n_readers; ++r) {
        args[r].ctx = &ctx;
        if (pthread_create(&threads[r], NULL, ebr_reader_fn, &args[r]) != 0) {
            atomic_store_explicit(&ctx.stop, 1U, memory_order_release);
            for (int32_t j = 0; j < r; ++j) pthread_join(threads[j], NULL);
            tsjit_arena_destroy(arena);
            return TSJIT_ERR_ALLOC;
        }
    }
    while (atomic_load_explicit(&ctx.ready, memory_order_acquire) < (uint32_t)n_readers) {
        sched_yield();
    }

    /* Perform >= n_resizes dynamic resizes + concurrent output publishes */
    uint64_t write_errs = 0;
    for (int32_t k = 0; k < n_resizes; ++k) {
        uint32_t target_cap = 4U + (uint32_t)((k % 64) + 1);
        if (tsjit_arena_force_resize(arena, target_cap) != TSJIT_OK) {
            write_errs++;
        }
        double q[TSJIT_NQ];
        for (int i = 0; i < TSJIT_NQ; ++i) {
            q[i] = (double)k + 0.1 * (double)i;
        }
        if (tsjit_arena_publish_agent_output(arena, aid0, (int64_t)(k + 1), q, 0.25, MARK_OK) != TSJIT_OK) {
            write_errs++;
        }
        if ((k & 7) == 0) {
            sched_yield();
        }
    }

    atomic_store_explicit(&ctx.stop, 1U, memory_order_release);
    uint64_t tot_reads = 0, tot_errs = write_errs;
    for (int32_t r = 0; r < n_readers; ++r) {
        pthread_join(threads[r], NULL);
        tot_reads += args[r].reads;
        tot_errs += args[r].errors;
    }

    uint32_t cap = 0;
    uint64_t resizes = 0, reclaimed = 0, pending = 0;
    tsjit_arena_ebr_stats(arena, &cap, &resizes, &reclaimed, &pending);
    tsjit_arena_destroy(arena);

    if (out_reads)     *out_reads = tot_reads;
    if (out_reclaimed) *out_reclaimed = reclaimed;
    if (out_errors)    *out_errors = tot_errs;
    return TSJIT_OK;
}
