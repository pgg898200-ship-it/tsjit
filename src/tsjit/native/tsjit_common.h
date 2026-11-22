/*
 * Copyright 2026 TSJIT Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * Portable C11 core definitions, error codes, causal marks, bounds checks,
 * safe age arithmetic, and NaN-preserving 25-comparator 7-stage sort9.
 */

#ifndef TSJIT_COMMON_H
#define TSJIT_COMMON_H

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdalign.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__APPLE__) && defined(__aarch64__)
#define TSJIT_CACHELINE 128
#else
#define TSJIT_CACHELINE 64
#endif

#if defined(__GNUC__) || defined(__clang__)
#define TSJIT_API __attribute__((visibility("default")))
#define TSJIT_LIKELY(x) __builtin_expect(!!(x), 1)
#define TSJIT_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define TSJIT_API
#define TSJIT_LIKELY(x) (x)
#define TSJIT_UNLIKELY(x) (x)
#endif

#define TSJIT_NQ 9
#define TSJIT_PWL_N_KNOTS 4
#define TSJIT_PWL_N_SEGMENTS 10
#define TSJIT_DEFAULT_MAX_SPINS 10000000ULL

/* Explicit status / error codes */
typedef enum tsjit_status {
    TSJIT_OK               = 0,
    TSJIT_ERR_BAD_ARG      = -1,
    TSJIT_ERR_OOB          = -2,
    TSJIT_ERR_SPIN_TIMEOUT = -3,
    TSJIT_ERR_OVERFLOW     = -4,
    TSJIT_ERR_ALLOC        = -5
} tsjit_status_t;

/* 7 causal arrival marks */
typedef enum tsjit_mark {
    MARK_OK          = 0,
    MARK_NAN_STALE   = 1,
    MARK_LAST_VALID  = 2,
    MARK_ZERO        = 3,
    MARK_DUP_STEP    = 4,
    MARK_OOO_STEP    = 5,
    MARK_INVALID_KEY = 6
} tsjit_mark_t;

#define TSJIT_MARK_OK          MARK_OK
#define TSJIT_MARK_NAN_STALE   MARK_NAN_STALE
#define TSJIT_MARK_LAST_VALID  MARK_LAST_VALID
#define TSJIT_MARK_ZERO        MARK_ZERO
#define TSJIT_MARK_DUP_STEP    MARK_DUP_STEP
#define TSJIT_MARK_OOO_STEP    MARK_OOO_STEP
#define TSJIT_MARK_INVALID_KEY MARK_INVALID_KEY

/* Subtraction-form overflow-safe bounds checks */
static inline bool tsjit_bounds_u64(uint64_t lo, uint64_t n, uint64_t cap) {
    return (lo <= cap) && (n <= (cap - lo));
}

static inline bool tsjit_bounds_i64(int64_t lo, int64_t n, int64_t cap) {
    if (lo < 0 || n < 0 || cap < 0) {
        return false;
    }
    return (lo <= cap) && (n <= (cap - lo));
}

static inline bool tsjit_bounds_sz(size_t lo, size_t n, size_t cap) {
    return (lo <= cap) && (n <= (cap - lo));
}

/* Overflow-safe 64-bit step difference via __int128 */
#if defined(__SIZEOF_INT128__)
__extension__ typedef __int128 tsjit_i128_t;
static inline int64_t safe_age_i64(int64_t now_step, int64_t ref_step) {
    tsjit_i128_t diff = (tsjit_i128_t)now_step - (tsjit_i128_t)ref_step;
    if (diff > (tsjit_i128_t)INT64_MAX) {
        return INT64_MAX;
    }
    if (diff < (tsjit_i128_t)INT64_MIN) {
        return INT64_MIN;
    }
    return (int64_t)diff;
}
#else
static inline int64_t safe_age_i64(int64_t now_step, int64_t ref_step) {
    if (ref_step < 0 && now_step > INT64_MAX + ref_step) {
        return INT64_MAX;
    }
    if (ref_step > 0 && now_step < INT64_MIN + ref_step) {
        return INT64_MIN;
    }
    return now_step - ref_step;
}
#endif

/* Strict-aliasing-safe float64 <-> uint64 bitcasts */
static inline uint64_t tsjit_f64_to_u64(double x) {
    uint64_t u = 0;
    memcpy(&u, &x, sizeof(u));
    return u;
}

static inline double tsjit_u64_to_f64(uint64_t u) {
    double x = 0.0;
    memcpy(&x, &u, sizeof(x));
    return x;
}

/* Spin-wait pause hint */
static inline void tsjit_cpu_relax(void) {
#if defined(__aarch64__) || defined(__arm64__)
    __asm__ __volatile__("yield" ::: "memory");
#elif defined(__x86_64__) || defined(__i386__)
    __asm__ __volatile__("pause" ::: "memory");
#else
    atomic_signal_fence(memory_order_seq_cst);
#endif
}

/* Cacheline-aligned zeroed allocation via posix_memalign */
static inline void *tsjit_aligned_calloc(size_t alignment, size_t size) {
    if (alignment < sizeof(void *) || (alignment & (alignment - 1)) != 0) {
        return NULL;
    }
    if (size == 0) {
        size = alignment;
    }
    /* Round up size to multiple of alignment safely */
    size_t rem = size % alignment;
    if (rem != 0) {
        size_t pad = alignment - rem;
        if (size > SIZE_MAX - pad) {
            return NULL;
        }
        size += pad;
    }
    void *ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0 || ptr == NULL) {
        return NULL;
    }
    memset(ptr, 0, size);
    return ptr;
}

/*
 * 25-comparator, 7-stage optimal sorting network for 9 doubles with explicit
 * any_nan -> insertion-sort guard. Preserves exact NaN multiplicity (1 NaN stays
 * 1 NaN at the tail, never erased or cloned to 9).
 */
static inline void tsjit_sort9(double a[TSJIT_NQ]) {
    bool any_nan = false;
    for (int i = 0; i < TSJIT_NQ; ++i) {
        if (isnan(a[i])) {
            any_nan = true;
            break;
        }
    }
    if (TSJIT_UNLIKELY(any_nan)) {
        for (int i = 1; i < TSJIT_NQ; ++i) {
            double key = a[i];
            bool key_nan = isnan(key);
            int j = i - 1;
            while (j >= 0) {
                bool aj_nan = isnan(a[j]);
                bool aj_gt_key = aj_nan ? (!key_nan) : (!key_nan && a[j] > key);
                if (!aj_gt_key) {
                    break;
                }
                a[j + 1] = a[j];
                --j;
            }
            a[j + 1] = key;
        }
        return;
    }

#define TSJIT_CSWAP(i, j)                                    \
    do {                                                     \
        double _x = a[(i)];                                  \
        double _y = a[(j)];                                  \
        bool _le = (_x <= _y);                               \
        a[(i)] = _le ? _x : _y;                              \
        a[(j)] = _le ? _y : _x;                              \
    } while (0)

    /* Stage 1 (4 comparators) */
    TSJIT_CSWAP(0, 3);
    TSJIT_CSWAP(1, 7);
    TSJIT_CSWAP(2, 5);
    TSJIT_CSWAP(4, 8);
    /* Stage 2 (4 comparators) */
    TSJIT_CSWAP(0, 7);
    TSJIT_CSWAP(2, 4);
    TSJIT_CSWAP(3, 8);
    TSJIT_CSWAP(5, 6);
    /* Stage 3 (4 comparators) */
    TSJIT_CSWAP(0, 2);
    TSJIT_CSWAP(1, 3);
    TSJIT_CSWAP(4, 5);
    TSJIT_CSWAP(7, 8);
    /* Stage 4 (3 comparators) */
    TSJIT_CSWAP(1, 4);
    TSJIT_CSWAP(3, 6);
    TSJIT_CSWAP(5, 7);
    /* Stage 5 (4 comparators) */
    TSJIT_CSWAP(0, 1);
    TSJIT_CSWAP(2, 4);
    TSJIT_CSWAP(3, 5);
    TSJIT_CSWAP(6, 8);
    /* Stage 6 (3 comparators) */
    TSJIT_CSWAP(2, 3);
    TSJIT_CSWAP(4, 5);
    TSJIT_CSWAP(6, 7);
    /* Stage 7 (3 comparators) */
    TSJIT_CSWAP(1, 2);
    TSJIT_CSWAP(3, 4);
    TSJIT_CSWAP(5, 6);

#undef TSJIT_CSWAP
}

/* Telemetry snapshot returned via seqlocked reader */
typedef struct tsjit_telemetry {
    uint64_t total_arrivals;
    uint64_t valid_arrivals;
    uint64_t dup_rejects;
    uint64_t ooo_rejects;
    uint64_t nan_stale_count;
    uint64_t pwl_fast_hits;
    uint64_t pwl_cold_hits;
    uint64_t pwl_fallback_hits;
    uint64_t roll_refreshes;
    int64_t  last_step;
    int64_t  last_key;
    double   last_abs_surprise;
} tsjit_telemetry_t;

/* Result of a single causal arrival step */
typedef struct tsjit_arrive_out {
    double  q[TSJIT_NQ];
    double  abs_sg;
    double  rolling_mean;
    double  rolling_std;
    double  eff_scale;
    int64_t age;
    int32_t mark;
    int32_t pwl_path; /* 0 = hot, 1 = cold, 2 = fallback, 3 = soft_v2 */
    double  family_weights[3]; /* [slow_tsfm, fast_ridge, persistence] */
} tsjit_arrive_out_t;

/* EWMA/Chan-Pebay monoid state (external ABI view) */
typedef struct tsjit_ewma_monoid {
    uint64_t n;
    uint32_t dim;
    uint32_t _pad;
    double   w;
    double  *mean; /* length dim */
    double  *cov;  /* length dim * dim (normalized weighted covariance C = M2 / w) */
} tsjit_ewma_monoid_t;

/* Forward declarations for opaque native handles */
typedef struct tsjit_arrive_engine tsjit_arrive_engine_t;
typedef struct tsjit_schur_state   tsjit_schur_state_t;
typedef struct tsjit_arena         tsjit_arena_t;

/* Exported platform/utility ABI */
TSJIT_API int32_t     tsjit_cacheline_size(void);
TSJIT_API const char *tsjit_simd_isa(void);
TSJIT_API int32_t     tsjit_check_bounds_i64(int64_t lo, int64_t n, int64_t cap);
TSJIT_API int64_t     tsjit_safe_age_i64_abi(int64_t now_step, int64_t ref_step);
TSJIT_API void        tsjit_sort9_abi(double a[TSJIT_NQ]);

/*
 * Exported tsjit_arrive.c ABI
 *
 * Threading & Concurrency Contract:
 *  - tsjit_arrive_publish_forecast() supports arbitrary concurrent multi-writer
 *    threads via the Boehm-2012 CAS seqlock.
 *  - tsjit_arrive_set_params() and tsjit_read_telemetry() are thread-safe for
 *    concurrent callers.
 *  - tsjit_arrive_step_pwl(), tsjit_arrive_step_soft_v2(), and
 *    tsjit_arrive_get_rolling() follow a strict single-writer-per-key contract:
 *    distinct keys may be stepped concurrently across threads, but callers MUST
 *    serialize step/rolling calls targeting the same key.
 */
TSJIT_API tsjit_arrive_engine_t *tsjit_arrive_create(int64_t n_keys, int64_t window,
                                                     double lambda_age, int64_t max_age);
TSJIT_API int32_t tsjit_arrive_set_params(tsjit_arrive_engine_t *eng,
                                          double lambda_age, int64_t max_age);
TSJIT_API void tsjit_arrive_destroy(tsjit_arrive_engine_t *eng);
TSJIT_API int32_t tsjit_arrive_publish_forecast(
    tsjit_arrive_engine_t *eng,
    int64_t key,
    int64_t forecast_step,
    const double q_slow[TSJIT_NQ],
    const double q_fast[TSJIT_NQ],
    double scale_hint,
    const double knots[TSJIT_PWL_N_KNOTS],
    const double w_fast_knots[TSJIT_PWL_N_KNOTS + 1],
    const double gain_knots[TSJIT_PWL_N_KNOTS + 1]);
TSJIT_API int32_t tsjit_arrive_step_pwl(
    tsjit_arrive_engine_t *eng,
    int64_t key,
    int64_t step,
    double x,
    tsjit_arrive_out_t *out);
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
    tsjit_arrive_out_t *out);
TSJIT_API int32_t tsjit_arrive_get_rolling(
    tsjit_arrive_engine_t *eng,
    int64_t key,
    double *out_mean,
    double *out_std,
    double *out_anchor,
    int64_t *out_count,
    int64_t *out_last_step);
TSJIT_API int32_t tsjit_read_telemetry(tsjit_arrive_engine_t *eng, tsjit_telemetry_t *out);
TSJIT_API int32_t tsjit_pwl_hot_row_bytes(void);
TSJIT_API int32_t tsjit_seqlock_slot_bytes(void);
TSJIT_API int32_t tsjit_seqlock_stress_test(
    int32_t n_writers,
    int32_t n_readers,
    uint64_t reads_per_reader,
    int32_t unguarded_control,
    uint64_t *out_total_reads,
    uint64_t *out_torn_reads,
    uint64_t *out_timeouts);

/* Exported tsjit_schur.c ABI */
TSJIT_API tsjit_schur_state_t *tsjit_schur_create(
    int32_t k_features,
    int32_t m_targets,
    double gamma,
    double ridge_lambda,
    double sg_cap);
TSJIT_API void tsjit_schur_destroy(tsjit_schur_state_t *st);
TSJIT_API int32_t tsjit_schur_update(
    tsjit_schur_state_t *st,
    const double *x,
    const double *y,
    const double *local_spread,
    double *out_weight);
TSJIT_API int32_t tsjit_schur_predict(
    tsjit_schur_state_t *st,
    const double *x,
    double *out_y);
TSJIT_API int32_t tsjit_schur_solve_system(
    int32_t n,
    const double *A,
    const double *b,
    double *x_out,
    int32_t *out_method,
    double *out_rel_resid,
    double *out_kappa_est);
TSJIT_API int32_t tsjit_schur_solve_shared_core(
    int32_t k_core,
    int32_t k_aug,
    int32_t n_forms,
    const double *A_cc,
    const double *A_ca_forms,
    const double *A_aa_forms,
    const double *b_c_forms,
    const double *b_a_forms,
    double *w_out_forms,
    int32_t *out_used_linv,
    int32_t *out_fallback_count,
    double *out_max_resid);
TSJIT_API int32_t tsjit_schur_check_higham_guard(
    int32_t n,
    const double *L,
    int32_t *out_allow_inv,
    double *out_diag_ratio,
    double *out_kappa_inf);

/* Exported tsjit_arena.c ABI */
TSJIT_API tsjit_arena_t *tsjit_arena_create(int32_t n_channels, uint32_t window, uint32_t init_agent_cap);
TSJIT_API void           tsjit_arena_destroy(tsjit_arena_t *arena);
TSJIT_API int32_t        tsjit_arena_add_agent(
    tsjit_arena_t *arena,
    const int32_t *domain_channels,
    int32_t n_domain,
    double ewma_gamma,
    int32_t *out_agent_id);
TSJIT_API int32_t tsjit_arena_tombstone_agent(tsjit_arena_t *arena, int32_t agent_id);
TSJIT_API int32_t tsjit_arena_agent_status(tsjit_arena_t *arena, int32_t agent_id);
TSJIT_API int32_t tsjit_arena_ingest_tick(
    tsjit_arena_t *arena,
    int64_t step,
    const double *values,
    int32_t n_values);
TSJIT_API int32_t tsjit_arena_publish_agent_output(
    tsjit_arena_t *arena,
    int32_t agent_id,
    int64_t step,
    const double q[TSJIT_NQ],
    double abs_sg,
    int32_t mark);
TSJIT_API int32_t tsjit_arena_read_agent_snapshot(
    tsjit_arena_t *arena,
    int32_t agent_id,
    double out_q[TSJIT_NQ],
    double *out_abs_sg,
    int32_t *out_mark,
    int64_t *out_step,
    uint64_t *out_version);
TSJIT_API int32_t tsjit_arena_get_channel_stats(
    tsjit_arena_t *arena,
    int32_t channel,
    double *out_mean,
    double *out_std,
    double *out_anchor,
    uint64_t *out_count);
TSJIT_API int32_t tsjit_arena_get_agent_cov(
    tsjit_arena_t *arena,
    int32_t agent_id,
    int32_t expected_dim,
    double *out_mean,
    double *out_cov,
    uint64_t *out_n,
    int32_t *out_group_id);
TSJIT_API int32_t tsjit_arena_force_resize(tsjit_arena_t *arena, uint32_t new_capacity);
TSJIT_API int32_t tsjit_arena_ebr_stats(
    tsjit_arena_t *arena,
    uint32_t *out_capacity,
    uint64_t *out_resizes,
    uint64_t *out_reclaimed,
    uint64_t *out_pending);
TSJIT_API int32_t tsjit_ewma_monoid_merge(
    double gamma,
    const tsjit_ewma_monoid_t *A,
    const tsjit_ewma_monoid_t *B,
    tsjit_ewma_monoid_t *out);
TSJIT_API int32_t tsjit_ewma_monoid_merge_tree(
    double gamma,
    const tsjit_ewma_monoid_t *blocks,
    int32_t n_blocks,
    tsjit_ewma_monoid_t *out);
TSJIT_API int32_t tsjit_arena_ebr_stress_test(
    int32_t n_readers,
    int32_t n_resizes,
    uint64_t *out_reads,
    uint64_t *out_reclaimed,
    uint64_t *out_errors);

#ifdef __cplusplus
}
#endif

#endif /* TSJIT_COMMON_H */
