/*
 * Copyright 2026 TSJIT Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * Portable C11 Huber/Mallows Winsorized EWMA Gram & Shared-Core Schur-Complement
 * Online Ridge Solver with:
 *  - PSD-preserving Mallows weighting w = (sg_cap / r)^2 + local spread clipping
 *  - Higham Ch. 14 guard on L^{-1} (diagonal ratio AND true kappa_inf(L) <= 1e3)
 *  - A posteriori residual check ||Ax - b||_inf <= 1e-11*(||b||_inf + 1e-30)
 *  - Jacobi-equilibrated Cholesky -> Complete-Pivoting Gaussian Elimination fallback
 *  - Apple Silicon NEON & x86-64 AVX2/FMA inner loops + portable scalar fallback
 */

#include "tsjit_common.h"

#include <pthread.h>

#if defined(__aarch64__) || defined(__arm64__)
#include <arm_neon.h>
#define TSJIT_HAVE_NEON 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define TSJIT_HAVE_AVX2 1
#endif

TSJIT_API const char *tsjit_simd_isa(void) {
#if defined(TSJIT_HAVE_NEON)
    return "NEON";
#elif defined(TSJIT_HAVE_AVX2)
    return "AVX2";
#else
    return "SCALAR";
#endif
}

/* ---------- Vectorized Kernels (NEON / AVX2 / Portable Scalar) ---------- */

static inline double simd_dot(const double *a, const double *b, int32_t n) {
    int32_t i = 0;
#if defined(TSJIT_HAVE_NEON)
    float64x2_t acc0 = vdupq_n_f64(0.0);
    float64x2_t acc1 = vdupq_n_f64(0.0);
    for (; i + 4 <= n; i += 4) {
        acc0 = vfmaq_f64(acc0, vld1q_f64(a + i),     vld1q_f64(b + i));
        acc1 = vfmaq_f64(acc1, vld1q_f64(a + i + 2), vld1q_f64(b + i + 2));
    }
    acc0 = vaddq_f64(acc0, acc1);
    double sum = vaddvq_f64(acc0);
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
#elif defined(TSJIT_HAVE_AVX2)
    __m256d acc = _mm256_setzero_pd();
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
#if defined(__FMA__)
        acc = _mm256_fmadd_pd(va, vb, acc);
#else
        acc = _mm256_add_pd(acc, _mm256_mul_pd(va, vb));
#endif
    }
    double tmp[4];
    _mm256_storeu_pd(tmp, acc);
    double sum = (tmp[0] + tmp[1]) + (tmp[2] + tmp[3]);
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
#else
    double sum = 0.0;
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
#endif
}

static inline void simd_axpy(double *y, const double *x, double alpha, int32_t n) {
    int32_t i = 0;
#if defined(TSJIT_HAVE_NEON)
    float64x2_t va = vdupq_n_f64(alpha);
    for (; i + 4 <= n; i += 4) {
        float64x2_t vy0 = vld1q_f64(y + i);
        float64x2_t vy1 = vld1q_f64(y + i + 2);
        vy0 = vfmaq_f64(vy0, va, vld1q_f64(x + i));
        vy1 = vfmaq_f64(vy1, va, vld1q_f64(x + i + 2));
        vst1q_f64(y + i, vy0);
        vst1q_f64(y + i + 2, vy1);
    }
    for (; i < n; ++i) {
        y[i] += alpha * x[i];
    }
#elif defined(TSJIT_HAVE_AVX2)
    __m256d va = _mm256_set1_pd(alpha);
    for (; i + 4 <= n; i += 4) {
        __m256d vy = _mm256_loadu_pd(y + i);
        __m256d vx = _mm256_loadu_pd(x + i);
#if defined(__FMA__)
        vy = _mm256_fmadd_pd(va, vx, vy);
#else
        vy = _mm256_add_pd(vy, _mm256_mul_pd(va, vx));
#endif
        _mm256_storeu_pd(y + i, vy);
    }
    for (; i < n; ++i) {
        y[i] += alpha * x[i];
    }
#else
    for (; i < n; ++i) {
        y[i] += alpha * x[i];
    }
#endif
}

static inline void simd_scale_add_rank1(double *G, double gamma, const double *sx, int32_t k) {
    for (int32_t i = 0; i < k; ++i) {
        double *row = G + (size_t)i * (size_t)k;
        double sxi = sx[i];
        int32_t j = 0;
#if defined(TSJIT_HAVE_NEON)
        float64x2_t vg = vdupq_n_f64(gamma);
        float64x2_t vs = vdupq_n_f64(sxi);
        for (; j + 4 <= k; j += 4) {
            float64x2_t r0 = vmulq_f64(vld1q_f64(row + j), vg);
            float64x2_t r1 = vmulq_f64(vld1q_f64(row + j + 2), vg);
            r0 = vfmaq_f64(r0, vs, vld1q_f64(sx + j));
            r1 = vfmaq_f64(r1, vs, vld1q_f64(sx + j + 2));
            vst1q_f64(row + j, r0);
            vst1q_f64(row + j + 2, r1);
        }
#elif defined(TSJIT_HAVE_AVX2)
        __m256d vg = _mm256_set1_pd(gamma);
        __m256d vs = _mm256_set1_pd(sxi);
        for (; j + 4 <= k; j += 4) {
            __m256d vr = _mm256_mul_pd(_mm256_loadu_pd(row + j), vg);
            __m256d vx = _mm256_loadu_pd(sx + j);
#if defined(__FMA__)
            vr = _mm256_fmadd_pd(vs, vx, vr);
#else
            vr = _mm256_add_pd(vr, _mm256_mul_pd(vs, vx));
#endif
            _mm256_storeu_pd(row + j, vr);
        }
#endif
        for (; j < k; ++j) {
            row[j] = gamma * row[j] + sxi * sx[j];
        }
    }
    /* Symmetrize strictly to prevent floating-point drift asymmetry */
    for (int32_t i = 0; i < k; ++i) {
        for (int32_t j = i + 1; j < k; ++j) {
            double avg = 0.5 * (G[(size_t)i * (size_t)k + (size_t)j] +
                                G[(size_t)j * (size_t)k + (size_t)i]);
            G[(size_t)i * (size_t)k + (size_t)j] = avg;
            G[(size_t)j * (size_t)k + (size_t)i] = avg;
        }
    }
}

/* ---------- Higham Ch. 14 Guard & Linear Solvers ---------- */

static double inf_norm_vec(const double *v, int32_t n) {
    double m = 0.0;
    for (int32_t i = 0; i < n; ++i) {
        double a = fabs(v[i]);
        if (!isfinite(a)) {
            return INFINITY;
        }
        if (a > m) {
            m = a;
        }
    }
    return m;
}

static double residual_inf_norm(int32_t n, const double *A, const double *x, const double *b) {
    double max_r = 0.0;
    for (int32_t i = 0; i < n; ++i) {
        double ax = simd_dot(A + (size_t)i * (size_t)n, x, n);
        double r = fabs(ax - b[i]);
        if (!isfinite(r)) {
            return INFINITY;
        }
        if (r > max_r) {
            max_r = r;
        }
    }
    return max_r;
}

/*
 * Higham Ch. 14 guard: invert lower-triangular L ONLY if BOTH:
 *   1. min(L_ii) > 1e-3 * max(L_ii)
 *   2. true kappa_inf(L) = ||L||_inf * ||L^{-1}||_inf <= 1e3
 */
static int32_t compute_higham_guard_and_linv(
    int32_t n,
    const double *L,
    double *Linv_out, /* optional, may be NULL */
    int32_t *out_allow_inv,
    double *out_diag_ratio,
    double *out_kappa_inf)
{
    if (n <= 0 || n > 2048 || L == NULL || out_allow_inv == NULL) {
        return TSJIT_ERR_BAD_ARG;
    }
    double d_min = INFINITY;
    double d_max = 0.0;
    double norm_L = 0.0;
    for (int32_t i = 0; i < n; ++i) {
        double di = L[(size_t)i * (size_t)n + (size_t)i];
        if (!isfinite(di) || di <= 0.0) {
            *out_allow_inv = 0;
            if (out_diag_ratio) *out_diag_ratio = 0.0;
            if (out_kappa_inf)  *out_kappa_inf = INFINITY;
            return TSJIT_OK;
        }
        if (di < d_min) d_min = di;
        if (di > d_max) d_max = di;
        double row_sum = 0.0;
        for (int32_t j = 0; j <= i; ++j) {
            row_sum += fabs(L[(size_t)i * (size_t)n + (size_t)j]);
        }
        if (row_sum > norm_L) {
            norm_L = row_sum;
        }
    }

    double ratio = (d_max > 0.0) ? (d_min / d_max) : 0.0;
    if (out_diag_ratio) *out_diag_ratio = ratio;

    double *Linv = Linv_out;
    bool own_buf = false;
    if (Linv == NULL) {
        Linv = (double *)calloc((size_t)n * (size_t)n, sizeof(double));
        if (Linv == NULL) {
            return TSJIT_ERR_ALLOC;
        }
        own_buf = true;
    } else {
        memset(Linv, 0, (size_t)n * (size_t)n * sizeof(double));
    }

    bool blew_up = false;
    for (int32_t j = 0; j < n && !blew_up; ++j) {
        double diag = L[(size_t)j * (size_t)n + (size_t)j];
        Linv[(size_t)j * (size_t)n + (size_t)j] = 1.0 / diag;
        for (int32_t i = j + 1; i < n; ++i) {
            double s = 0.0;
            const double *Lrow = L + (size_t)i * (size_t)n;
            for (int32_t p = j; p < i; ++p) {
                s += Lrow[p] * Linv[(size_t)p * (size_t)n + (size_t)j];
            }
            double val = -s / Lrow[i];
            if (!isfinite(val) || fabs(val) > 1e150) {
                blew_up = true;
                break;
            }
            Linv[(size_t)i * (size_t)n + (size_t)j] = val;
        }
    }

    double kappa = INFINITY;
    if (!blew_up) {
        double norm_Linv = 0.0;
        for (int32_t i = 0; i < n; ++i) {
            double row_sum = 0.0;
            for (int32_t j = 0; j <= i; ++j) {
                row_sum += fabs(Linv[(size_t)i * (size_t)n + (size_t)j]);
            }
            if (row_sum > norm_Linv) {
                norm_Linv = row_sum;
            }
        }
        kappa = norm_L * norm_Linv;
    }

    if (out_kappa_inf) *out_kappa_inf = kappa;
    int32_t allow = (ratio > 1e-3 && isfinite(kappa) && kappa <= 1e3) ? 1 : 0;
    *out_allow_inv = allow;

    if (own_buf) {
        free(Linv);
    }
    return TSJIT_OK;
}

TSJIT_API int32_t tsjit_schur_check_higham_guard(
    int32_t n,
    const double *L,
    int32_t *out_allow_inv,
    double *out_diag_ratio,
    double *out_kappa_inf)
{
    return compute_higham_guard_and_linv(n, L, NULL, out_allow_inv, out_diag_ratio, out_kappa_inf);
}

/* Jacobi-equilibrated Cholesky factorization: D A D = L L^T */
static bool jacobi_cholesky_factor(
    int32_t n,
    const double *A,
    double *D_out,
    double *L_out)
{
    for (int32_t i = 0; i < n; ++i) {
        double aii = A[(size_t)i * (size_t)n + (size_t)i];
        if (!isfinite(aii) || aii <= 0.0) {
            return false;
        }
        D_out[i] = 1.0 / sqrt(aii);
        if (!isfinite(D_out[i])) {
            return false;
        }
    }

    memset(L_out, 0, (size_t)n * (size_t)n * sizeof(double));
    for (int32_t i = 0; i < n; ++i) {
        double *Li = L_out + (size_t)i * (size_t)n;
        double di = D_out[i];
        for (int32_t j = 0; j <= i; ++j) {
            const double *Lj = L_out + (size_t)j * (size_t)n;
            double aij = 0.5 * (A[(size_t)i * (size_t)n + (size_t)j] +
                                A[(size_t)j * (size_t)n + (size_t)i]);
            double s = di * aij * D_out[j];
            if (j > 0) {
                s -= simd_dot(Li, Lj, j);
            }
            if (i == j) {
                if (!isfinite(s) || s <= 0.0) {
                    return false;
                }
                Li[i] = sqrt(s);
            } else {
                double val = s / Lj[j];
                if (!isfinite(val)) {
                    return false;
                }
                Li[j] = val;
            }
        }
    }
    return true;
}

static void chol_solve_direct(
    int32_t n,
    const double *D,
    const double *L,
    const double *b,
    double *x_out,
    double *tmp)
{
    /* Forward solve L y = D b */
    for (int32_t i = 0; i < n; ++i) {
        const double *Li = L + (size_t)i * (size_t)n;
        double s = D[i] * b[i];
        if (i > 0) {
            s -= simd_dot(Li, tmp, i);
        }
        tmp[i] = s / Li[i];
    }
    /* Backward solve L^T z = y */
    for (int32_t i = n - 1; i >= 0; --i) {
        double s = tmp[i];
        for (int32_t j = i + 1; j < n; ++j) {
            s -= L[(size_t)j * (size_t)n + (size_t)i] * tmp[j];
        }
        tmp[i] = s / L[(size_t)i * (size_t)n + (size_t)i];
    }
    /* Unscale x = D z */
    for (int32_t i = 0; i < n; ++i) {
        x_out[i] = D[i] * tmp[i];
    }
}

static void chol_solve_linv(
    int32_t n,
    const double *D,
    const double *Linv,
    const double *b,
    double *x_out,
    double *tmp)
{
    /* y = Linv * (D b) */
    for (int32_t i = 0; i < n; ++i) {
        const double *row = Linv + (size_t)i * (size_t)n;
        double s = 0.0;
        for (int32_t j = 0; j <= i; ++j) {
            s += row[j] * (D[j] * b[j]);
        }
        tmp[i] = s;
    }
    /* x = D * (Linv^T * y) */
    for (int32_t i = 0; i < n; ++i) {
        double s = 0.0;
        for (int32_t j = i; j < n; ++j) {
            s += Linv[(size_t)j * (size_t)n + (size_t)i] * tmp[j];
        }
        x_out[i] = D[i] * s;
    }
}

/* Complete-Pivoting Gaussian Elimination with iterative refinement */
static int32_t complete_pivot_ge_solve(
    int32_t n,
    const double *A,
    const double *b,
    double *x_out)
{
    size_t nn = (size_t)n * (size_t)n;
    double *M = (double *)malloc(nn * sizeof(double));
    double *rhs = (double *)malloc((size_t)n * sizeof(double));
    double *xp = (double *)calloc((size_t)n, sizeof(double));
    int32_t *col_perm = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    if (!M || !rhs || !xp || !col_perm) {
        free(M);
        free(rhs);
        free(xp);
        free(col_perm);
        return TSJIT_ERR_ALLOC;
    }

    memcpy(M, A, nn * sizeof(double));
    memcpy(rhs, b, (size_t)n * sizeof(double));
    for (size_t idx = 0; idx < nn; ++idx) {
        if (!isfinite(M[idx])) {
            free(M); free(rhs); free(xp); free(col_perm);
            return TSJIT_ERR_BAD_ARG;
        }
    }
    /* Exact power-of-2 row equilibration via frexp/ldexp so subnormal rows (down to
     * DBL_TRUE_MIN ~ 5e-324) equilibrate with ZERO rounding error and ZERO 1/rmax overflow. */
    for (int32_t i = 0; i < n; ++i) {
        double *row = M + (size_t)i * (size_t)n;
        double rmax = 0.0;
        for (int32_t j = 0; j < n; ++j) {
            double av = fabs(row[j]);
            if (av > rmax) rmax = av;
        }
        if (rmax > 0.0 && isfinite(rmax)) {
            int er = 0;
            (void)frexp(rmax, &er);
            int shift = -er;
            for (int32_t j = 0; j < n; ++j) {
                row[j] = ldexp(row[j], shift);
            }
            rhs[i] = ldexp(rhs[i], shift);
            if (!isfinite(rhs[i])) {
                free(M); free(rhs); free(xp); free(col_perm);
                return TSJIT_ERR_OVERFLOW;
            }
        }
    }
    for (int32_t j = 0; j < n; ++j) {
        col_perm[j] = j;
    }

    double piv_tol = DBL_MIN * 1e6;
    int32_t rank = n;

    for (int32_t p = 0; p < n; ++p) {
        int32_t best_r = p;
        int32_t best_c = p;
        double best_v = 0.0;
        for (int32_t i = p; i < n; ++i) {
            const double *row = M + (size_t)i * (size_t)n;
            for (int32_t j = p; j < n; ++j) {
                double av = fabs(row[j]);
                if (av > best_v) {
                    best_v = av;
                    best_r = i;
                    best_c = j;
                }
            }
        }
        if (best_v <= piv_tol || !isfinite(best_v)) {
            rank = p;
            break;
        }
        if (best_r != p) {
            double *r1 = M + (size_t)p * (size_t)n;
            double *r2 = M + (size_t)best_r * (size_t)n;
            for (int32_t j = 0; j < n; ++j) {
                double t = r1[j]; r1[j] = r2[j]; r2[j] = t;
            }
            double tb = rhs[p]; rhs[p] = rhs[best_r]; rhs[best_r] = tb;
        }
        if (best_c != p) {
            for (int32_t i = 0; i < n; ++i) {
                double *row = M + (size_t)i * (size_t)n;
                double t = row[p]; row[p] = row[best_c]; row[best_c] = t;
            }
            int32_t tc = col_perm[p]; col_perm[p] = col_perm[best_c]; col_perm[best_c] = tc;
        }

        double *rp = M + (size_t)p * (size_t)n;
        double piv = rp[p];
        for (int32_t i = p + 1; i < n; ++i) {
            double *ri = M + (size_t)i * (size_t)n;
            double factor = ri[p] / piv;
            ri[p] = 0.0;
            if (factor != 0.0) {
                simd_axpy(ri + (p + 1), rp + (p + 1), -factor, n - (p + 1));
                rhs[i] -= factor * rhs[p];
            }
        }
    }

    for (int32_t i = rank - 1; i >= 0; --i) {
        const double *ri = M + (size_t)i * (size_t)n;
        double s = rhs[i];
        if (i + 1 < n) {
            s -= simd_dot(ri + (i + 1), xp + (i + 1), n - (i + 1));
        }
        xp[i] = s / ri[i];
    }
    for (int32_t j = 0; j < n; ++j) {
        x_out[col_perm[j]] = isfinite(xp[j]) ? xp[j] : 0.0;
    }

    free(M);
    free(rhs);
    free(xp);
    free(col_perm);
    return TSJIT_OK;
}

/*
 * General safeguarded solve A x = b:
 *   1. Jacobi-equilibrated Cholesky (with optional 1-step iterative refinement)
 *   2. A posteriori residual check ||A x - b||_inf <= 1e-11 * (||b||_inf + 1e-30)
 *   3. Fallback to Complete-Pivoting Gaussian Elimination, picking best residual
 * out_method: 1 = Jacobi Cholesky, 2 = Complete-Pivoting GE fallback
 */
TSJIT_API int32_t tsjit_schur_solve_system(
    int32_t n,
    const double *A,
    const double *b,
    double *x_out,
    int32_t *out_method,
    double *out_rel_resid,
    double *out_kappa_est)
{
    if (n <= 0 || n > 4096 || A == NULL || b == NULL || x_out == NULL) {
        return TSJIT_ERR_BAD_ARG;
    }
    double b_inf = inf_norm_vec(b, n);
    if (!isfinite(b_inf)) {
        return TSJIT_ERR_BAD_ARG;
    }
    double resid_tol = 1e-11 * (b_inf + 1e-30);

    size_t nn = (size_t)n * (size_t)n;
    double *D = (double *)malloc((size_t)n * sizeof(double));
    double *L = (double *)malloc(nn * sizeof(double));
    double *tmp = (double *)malloc((size_t)n * sizeof(double));
    double *x_chol = (double *)malloc((size_t)n * sizeof(double));
    double *r_vec = (double *)malloc((size_t)n * sizeof(double));
    double *dx = (double *)malloc((size_t)n * sizeof(double));
    if (!D || !L || !tmp || !x_chol || !r_vec || !dx) {
        free(D); free(L); free(tmp); free(x_chol); free(r_vec); free(dx);
        return TSJIT_ERR_ALLOC;
    }

    bool chol_ok = jacobi_cholesky_factor(n, A, D, L);
    double r_chol = INFINITY;
    double kappa_est = INFINITY;

    if (chol_ok) {
        /* Estimate condition number via diagonal ratio & Higham check if requested */
        if (out_kappa_est != NULL) {
            int32_t allow_inv = 0;
            double dr = 0.0;
            compute_higham_guard_and_linv(n, L, NULL, &allow_inv, &dr, &kappa_est);
        }
        chol_solve_direct(n, D, L, b, x_chol, tmp);
        r_chol = residual_inf_norm(n, A, x_chol, b);

        /* Up to 2 steps of Cholesky iterative refinement if helpful */
        for (int iter = 0; iter < 2 && isfinite(r_chol) && r_chol > 1e-14 * (b_inf + 1e-30); ++iter) {
            for (int32_t i = 0; i < n; ++i) {
                r_vec[i] = b[i] - simd_dot(A + (size_t)i * (size_t)n, x_chol, n);
            }
            chol_solve_direct(n, D, L, r_vec, dx, tmp);
            for (int32_t i = 0; i < n; ++i) {
                dx[i] = x_chol[i] + dx[i];
            }
            double r_new = residual_inf_norm(n, A, dx, b);
            if (isfinite(r_new) && r_new < r_chol) {
                memcpy(x_chol, dx, (size_t)n * sizeof(double));
                r_chol = r_new;
            } else {
                break;
            }
        }
    }

    int32_t method = 1;
    double final_r = r_chol;
    if (chol_ok && isfinite(r_chol) && r_chol <= resid_tol) {
        memcpy(x_out, x_chol, (size_t)n * sizeof(double));
        method = 1;
        final_r = r_chol;
    } else {
        /* Fallback to Complete-Pivoting Gaussian Elimination */
        int32_t rc = complete_pivot_ge_solve(n, A, b, dx);
        if (rc != TSJIT_OK) {
            free(D); free(L); free(tmp); free(x_chol); free(r_vec); free(dx);
            return rc;
        }
        double r_ge = residual_inf_norm(n, A, dx, b);
        /* One step of GE iterative refinement for extreme-kappa systems */
        if (isfinite(r_ge) && r_ge > 0.0) {
            for (int32_t i = 0; i < n; ++i) {
                r_vec[i] = b[i] - simd_dot(A + (size_t)i * (size_t)n, dx, n);
            }
            double *corr = tmp;
            if (complete_pivot_ge_solve(n, A, r_vec, corr) == TSJIT_OK) {
                for (int32_t i = 0; i < n; ++i) {
                    r_vec[i] = dx[i] + corr[i];
                }
                double r_ge2 = residual_inf_norm(n, A, r_vec, b);
                if (isfinite(r_ge2) && r_ge2 < r_ge) {
                    memcpy(dx, r_vec, (size_t)n * sizeof(double));
                    r_ge = r_ge2;
                }
            }
        }
        if (chol_ok && isfinite(r_chol) && (!isfinite(r_ge) || r_chol < r_ge)) {
            memcpy(x_out, x_chol, (size_t)n * sizeof(double));
            method = 1;
            final_r = r_chol;
        } else {
            memcpy(x_out, dx, (size_t)n * sizeof(double));
            method = 2;
            final_r = r_ge;
        }
    }

    if (out_method)    *out_method = method;
    if (out_rel_resid) *out_rel_resid = final_r / (b_inf + 1e-30);
    if (out_kappa_est) *out_kappa_est = kappa_est;

    free(D); free(L); free(tmp); free(x_chol); free(r_vec); free(dx);
    return TSJIT_OK;
}

/* ---------- Shared-Core Schur-Complement Solver (K <= 136 & General K) ---------- */

TSJIT_API int32_t tsjit_schur_solve_shared_core(
    int32_t k_core,
    int32_t k_aug,
    int32_t n_forms,
    const double *A_cc,
    const double *A_ca_forms, /* [n_forms, k_core, k_aug] */
    const double *A_aa_forms, /* [n_forms, k_aug, k_aug] */
    const double *b_c_forms,  /* [n_forms, k_core] */
    const double *b_a_forms,  /* [n_forms, k_aug] */
    double *w_out_forms,      /* [n_forms, k_core + k_aug] */
    int32_t *out_used_linv,
    int32_t *out_fallback_count,
    double *out_max_resid)
{
    if (k_core <= 0 || k_aug < 0 || n_forms <= 0 ||
        k_core > 2048 || k_aug > 2048 ||
        (int64_t)k_core + (int64_t)k_aug > 2048 ||
        A_cc == NULL || b_c_forms == NULL || w_out_forms == NULL) {
        return TSJIT_ERR_BAD_ARG;
    }
    if (k_aug > 0 && (A_ca_forms == NULL || A_aa_forms == NULL || b_a_forms == NULL)) {
        return TSJIT_ERR_BAD_ARG;
    }

    int32_t K = k_core + k_aug;
    double *D = (double *)malloc((size_t)k_core * sizeof(double));
    double *L = (double *)malloc((size_t)k_core * (size_t)k_core * sizeof(double));
    double *Linv = (double *)malloc((size_t)k_core * (size_t)k_core * sizeof(double));
    double *tmp = (double *)malloc((size_t)k_core * sizeof(double));
    double *col_in = (double *)malloc((size_t)k_core * sizeof(double));
    double *Y = (double *)malloc((size_t)k_core * (size_t)(k_aug > 0 ? k_aug : 1) * sizeof(double));
    double *z = (double *)malloc((size_t)k_core * sizeof(double));
    double *S = (double *)malloc((size_t)(k_aug > 0 ? k_aug : 1) * (size_t)(k_aug > 0 ? k_aug : 1) * sizeof(double));
    double *rhs_a = (double *)malloc((size_t)(k_aug > 0 ? k_aug : 1) * sizeof(double));
    double *A_full = (double *)malloc((size_t)K * (size_t)K * sizeof(double));
    double *b_full = (double *)malloc((size_t)K * sizeof(double));

    if (!D || !L || !Linv || !tmp || !col_in || !Y || !z || !S || !rhs_a || !A_full || !b_full) {
        free(D); free(L); free(Linv); free(tmp); free(col_in);
        free(Y); free(z); free(S); free(rhs_a); free(A_full); free(b_full);
        return TSJIT_ERR_ALLOC;
    }

    bool core_chol_ok = jacobi_cholesky_factor(k_core, A_cc, D, L);
    int32_t allow_linv = 0;
    if (core_chol_ok) {
        double dr = 0.0, kap = 0.0;
        compute_higham_guard_and_linv(k_core, L, Linv, &allow_linv, &dr, &kap);
    }

    int32_t fb_count = 0;
    double max_rel_r = 0.0;

    for (int32_t f = 0; f < n_forms; ++f) {
        const double *A_ca = (k_aug > 0) ? (A_ca_forms + (size_t)f * (size_t)k_core * (size_t)k_aug) : NULL;
        const double *A_aa = (k_aug > 0) ? (A_aa_forms + (size_t)f * (size_t)k_aug * (size_t)k_aug) : NULL;
        const double *b_c  = b_c_forms + (size_t)f * (size_t)k_core;
        const double *b_a  = (k_aug > 0) ? (b_a_forms + (size_t)f * (size_t)k_aug) : NULL;
        double *w_out      = w_out_forms + (size_t)f * (size_t)K;

        /* Assemble full system for exact a posteriori residual verification & fallback */
        for (int32_t i = 0; i < k_core; ++i) {
            memcpy(A_full + (size_t)i * (size_t)K,
                   A_cc + (size_t)i * (size_t)k_core,
                   (size_t)k_core * sizeof(double));
            for (int32_t j = 0; j < k_aug; ++j) {
                double v = A_ca[(size_t)i * (size_t)k_aug + (size_t)j];
                A_full[(size_t)i * (size_t)K + (size_t)(k_core + j)] = v;
                A_full[(size_t)(k_core + j) * (size_t)K + (size_t)i] = v;
            }
            b_full[i] = b_c[i];
        }
        for (int32_t i = 0; i < k_aug; ++i) {
            for (int32_t j = 0; j < k_aug; ++j) {
                A_full[(size_t)(k_core + i) * (size_t)K + (size_t)(k_core + j)] =
                    A_aa[(size_t)i * (size_t)k_aug + (size_t)j];
            }
            b_full[k_core + i] = b_a[i];
        }

        double b_inf = inf_norm_vec(b_full, K);
        double tol = 1e-11 * (b_inf + 1e-30);
        bool schur_ok = false;

        if (core_chol_ok) {
            if (allow_linv) {
                chol_solve_linv(k_core, D, Linv, b_c, z, tmp);
            } else {
                chol_solve_direct(k_core, D, L, b_c, z, tmp);
            }

            if (k_aug == 0) {
                memcpy(w_out, z, (size_t)k_core * sizeof(double));
                schur_ok = true;
            } else {
                /* Solve A_cc * Y[:, j] = A_ca[:, j] for each augmented column */
                for (int32_t j = 0; j < k_aug; ++j) {
                    for (int32_t i = 0; i < k_core; ++i) {
                        col_in[i] = A_ca[(size_t)i * (size_t)k_aug + (size_t)j];
                    }
                    double *yj = Y + (size_t)j * (size_t)k_core; /* column-major storage of Y */
                    if (allow_linv) {
                        chol_solve_linv(k_core, D, Linv, col_in, yj, tmp);
                    } else {
                        chol_solve_direct(k_core, D, L, col_in, yj, tmp);
                    }
                }
                /* Form Schur complement S = A_aa - A_ca^T Y and rhs_a = b_a - A_ca^T z */
                for (int32_t i = 0; i < k_aug; ++i) {
                    double dot_z = 0.0;
                    for (int32_t p = 0; p < k_core; ++p) {
                        dot_z += A_ca[(size_t)p * (size_t)k_aug + (size_t)i] * z[p];
                    }
                    rhs_a[i] = b_a[i] - dot_z;
                    for (int32_t j = 0; j < k_aug; ++j) {
                        const double *yj = Y + (size_t)j * (size_t)k_core;
                        double dot_y = 0.0;
                        for (int32_t p = 0; p < k_core; ++p) {
                            dot_y += A_ca[(size_t)p * (size_t)k_aug + (size_t)i] * yj[p];
                        }
                        S[(size_t)i * (size_t)k_aug + (size_t)j] =
                            A_aa[(size_t)i * (size_t)k_aug + (size_t)j] - dot_y;
                    }
                }
                double *w_a = w_out + k_core;
                if (tsjit_schur_solve_system(k_aug, S, rhs_a, w_a, NULL, NULL, NULL) == TSJIT_OK) {
                    for (int32_t p = 0; p < k_core; ++p) {
                        double acc = z[p];
                        for (int32_t j = 0; j < k_aug; ++j) {
                            acc -= Y[(size_t)j * (size_t)k_core + (size_t)p] * w_a[j];
                        }
                        w_out[p] = acc;
                    }
                    schur_ok = true;
                }
            }
        }

        double r_inf = schur_ok ? residual_inf_norm(K, A_full, w_out, b_full) : INFINITY;
        if (!schur_ok || !isfinite(r_inf) || r_inf > tol) {
            fb_count++;
            double rel_r = 0.0;
            int32_t rc = tsjit_schur_solve_system(K, A_full, b_full, w_out, NULL, &rel_r, NULL);
            if (rc != TSJIT_OK) {
                free(D); free(L); free(Linv); free(tmp); free(col_in);
                free(Y); free(z); free(S); free(rhs_a); free(A_full); free(b_full);
                return rc;
            }
            r_inf = residual_inf_norm(K, A_full, w_out, b_full);
        }
        double rel = r_inf / (b_inf + 1e-30);
        if (rel > max_rel_r) {
            max_rel_r = rel;
        }
    }

    if (out_used_linv)      *out_used_linv = allow_linv;
    if (out_fallback_count) *out_fallback_count = fb_count;
    if (out_max_resid)      *out_max_resid = max_rel_r;

    free(D); free(L); free(Linv); free(tmp); free(col_in);
    free(Y); free(z); free(S); free(rhs_a); free(A_full); free(b_full);
    return TSJIT_OK;
}

/* ---------- Online Huber/Mallows Winsorized EWMA Ridge State ---------- */

struct tsjit_schur_state {
    pthread_mutex_t mu;
    bool            mu_inited;
    int32_t  k;
    int32_t  m;
    double   gamma;
    double   ridge_lambda;
    double   sg_cap;
    uint64_t n_updates;
    bool     weights_dirty;
    double  *G;       /* [k, k] */
    double  *rhs;     /* [k, m] stored column-major by target: rhs[r * k + j] */
    double  *W;       /* [m, k] */
    double  *mu_x;    /* [k] */
    double  *var_x;   /* [k] */
    double  *mu_y;    /* [m] */
    double  *var_y;   /* [m] */
    double  *scratch; /* [3 * k + k * k] */
};

TSJIT_API tsjit_schur_state_t *tsjit_schur_create(
    int32_t k_features,
    int32_t m_targets,
    double gamma,
    double ridge_lambda,
    double sg_cap)
{
    if (k_features <= 0 || k_features > 2048 || m_targets <= 0 || m_targets > 64) {
        return NULL;
    }
    if (!isfinite(gamma) || gamma <= 0.0 || gamma > 1.0) {
        return NULL;
    }
    if (!isfinite(ridge_lambda) || ridge_lambda <= 0.0) {
        return NULL;
    }
    if (!isfinite(sg_cap) || sg_cap <= 0.0) {
        sg_cap = 20.0;
    }

    tsjit_schur_state_t *st = (tsjit_schur_state_t *)calloc(1, sizeof(*st));
    if (!st) {
        return NULL;
    }
    if (pthread_mutex_init(&st->mu, NULL) != 0) {
        free(st);
        return NULL;
    }
    st->mu_inited = true;
    st->k = k_features;
    st->m = m_targets;
    st->gamma = gamma;
    st->ridge_lambda = ridge_lambda;
    st->sg_cap = sg_cap;
    st->weights_dirty = true;

    size_t kk = (size_t)k_features * (size_t)k_features;
    size_t km = (size_t)k_features * (size_t)m_targets;

    st->G = (double *)tsjit_aligned_calloc(TSJIT_CACHELINE, kk * sizeof(double));
    st->rhs = (double *)tsjit_aligned_calloc(TSJIT_CACHELINE, km * sizeof(double));
    st->W = (double *)tsjit_aligned_calloc(TSJIT_CACHELINE, km * sizeof(double));
    st->mu_x = (double *)calloc((size_t)k_features, sizeof(double));
    st->var_x = (double *)calloc((size_t)k_features, sizeof(double));
    st->mu_y = (double *)calloc((size_t)m_targets, sizeof(double));
    st->var_y = (double *)calloc((size_t)m_targets, sizeof(double));
    st->scratch = (double *)tsjit_aligned_calloc(
        TSJIT_CACHELINE, (3 * (size_t)k_features + kk) * sizeof(double));

    if (!st->G || !st->rhs || !st->W || !st->mu_x || !st->var_x ||
        !st->mu_y || !st->var_y || !st->scratch) {
        tsjit_schur_destroy(st);
        return NULL;
    }
    return st;
}

TSJIT_API void tsjit_schur_destroy(tsjit_schur_state_t *st) {
    if (!st) return;
    if (st->mu_inited) {
        pthread_mutex_destroy(&st->mu);
    }
    free(st->G);
    free(st->rhs);
    free(st->W);
    free(st->mu_x);
    free(st->var_x);
    free(st->mu_y);
    free(st->var_y);
    free(st->scratch);
    free(st);
}

static int32_t schur_refresh_weights(tsjit_schur_state_t *st) {
    if (!st->weights_dirty) {
        return TSJIT_OK;
    }
    int32_t k = st->k;
    size_t kk = (size_t)k * (size_t)k;
    double *A = st->scratch + 3 * (size_t)k;
    memcpy(A, st->G, kk * sizeof(double));
    for (int32_t i = 0; i < k; ++i) {
        A[(size_t)i * (size_t)k + (size_t)i] += st->ridge_lambda;
    }
    for (int32_t r = 0; r < st->m; ++r) {
        const double *b = st->rhs + (size_t)r * (size_t)k;
        double *wr = st->W + (size_t)r * (size_t)k;
        int32_t rc = tsjit_schur_solve_system(k, A, b, wr, NULL, NULL, NULL);
        if (rc != TSJIT_OK) {
            return rc;
        }
    }
    st->weights_dirty = false;
    return TSJIT_OK;
}

TSJIT_API int32_t tsjit_schur_update(
    tsjit_schur_state_t *st,
    const double *x,
    const double *y,
    const double *local_spread,
    double *out_weight)
{
    if (!st || !x || !y) {
        return TSJIT_ERR_BAD_ARG;
    }
    int32_t k = st->k;
    int32_t m = st->m;

    /* Reject non-finite inputs BEFORE any arithmetic (zero 0*Inf NaN!) */
    for (int32_t j = 0; j < k; ++j) {
        if (!isfinite(x[j])) {
            if (out_weight) *out_weight = 0.0;
            return TSJIT_OK;
        }
    }
    for (int32_t r = 0; r < m; ++r) {
        if (!isfinite(y[r])) {
            if (out_weight) *out_weight = 0.0;
            return TSJIT_OK;
        }
    }

    pthread_mutex_lock(&st->mu);

    double *z     = st->scratch;
    double *x_win = st->scratch + (size_t)k;
    double *sx    = st->scratch + 2 * (size_t)k;

    double cap = st->sg_cap;
    double max_abs_z = 0.0;

    for (int32_t j = 0; j < k; ++j) {
        double sp_tracked = (st->n_updates >= 2 && st->var_x[j] > 0.0 && isfinite(st->var_x[j]))
            ? sqrt(st->var_x[j])
            : 0.0;
        double base_floor = fmax(1e-4 * fabs(st->mu_x[j]), 1e-12);
        double sp_eff = (sp_tracked > 0.0) ? fmax(sp_tracked, base_floor) : 1.0;

        /* Clip local spread cleanly against 0, Inf, NaN, or 1e300 glitches */
        if (local_spread != NULL && isfinite(local_spread[j]) && local_spread[j] > 0.0) {
            double sp_in = local_spread[j];
            if (st->n_updates >= 5 && sp_tracked > 0.0) {
                double lo = fmax(1e-4 * sp_tracked, 1e-12);
                double hi = cap * fmax(sp_tracked, base_floor);
                sp_eff = fmin(fmax(sp_in, lo), hi);
            } else {
                sp_eff = fmin(fmax(sp_in, 1e-12), 1e12);
            }
        }

        double dx = (st->n_updates >= 2) ? (x[j] - st->mu_x[j]) : 0.0;
        double zj = dx / sp_eff;
        if (!isfinite(zj)) {
            zj = (dx >= 0.0) ? (cap * 1e6) : (-cap * 1e6);
        }
        z[j] = zj;
        double az = fabs(zj);
        if (az > max_abs_z) {
            max_abs_z = az;
        }

        /* Locally Winsorize x_win[j] when warmed up and |z_j| > sg_cap */
        if (st->n_updates >= 2 && az > cap) {
            x_win[j] = st->mu_x[j] + copysign(cap * sp_eff, dx);
        } else {
            x_win[j] = x[j];
        }
    }

    /* Hard magnitude ceiling before squaring into Gram G / rhs (1e70^2 = 1e140 << DBL_MAX) */
    const double max_raw_abs = 1e70;
    for (int32_t j = 0; j < k; ++j) {
        if (!isfinite(x_win[j]) || fabs(x_win[j]) > max_raw_abs) {
            x_win[j] = copysign(max_raw_abs, x_win[j]);
        }
    }

    /* Overflow-safe Euclidean radius r = ||z||_2 */
    double r_norm = 0.0;
    if (max_abs_z > 0.0) {
        double inv_m = 1.0 / max_abs_z;
        double ss = 0.0;
        for (int32_t j = 0; j < k; ++j) {
            double u = z[j] * inv_m;
            ss += u * u;
        }
        r_norm = max_abs_z * sqrt(ss);
    }

    /* PSD-preserving Mallows weight w = (sg_cap / r)^2 when r > sg_cap */
    double w = 1.0;
    if (st->n_updates >= 2 && r_norm > cap) {
        double ratio = cap / r_norm;
        w = ratio * ratio;
        if (!isfinite(w) || w < 0.0) {
            w = 0.0;
        }
    }

    /* Pre-update prediction for causal Huber target Winsorization */
    bool have_pred = (st->n_updates >= 5);
    if (have_pred && st->weights_dirty) {
        if (schur_refresh_weights(st) != TSJIT_OK) {
            have_pred = false;
        }
    }

    /* Compute causal Huber target weights & Winsorized targets BEFORE updating G
     * so G and rhs receive the SAME PSD-preserving effective weight w_eff = w * min(wy)
     * (preventing nonzero-mean shrinkage bias under 10,000-sigma target glitches!). */
    double wy_arr[64];
    double y_eff_arr[64];
    double wy_min = 1.0;
    for (int32_t r = 0; r < m; ++r) {
        double yr = y[r];
        double wy = 1.0;
        double y_eff = yr;
        if (st->n_updates >= 2) {
            double y_ref = have_pred
                ? simd_dot(st->W + (size_t)r * (size_t)k, x_win, k)
                : st->mu_y[r];
            if (!isfinite(y_ref)) {
                y_ref = st->mu_y[r];
            }
            double sp_y = (st->var_y[r] > 0.0 && isfinite(st->var_y[r]))
                ? sqrt(st->var_y[r])
                : fmax(1e-4 * fabs(st->mu_y[r]), 1.0);
            sp_y = fmax(sp_y, fmax(1e-4 * fabs(st->mu_y[r]), 1e-12));
            double err = yr - y_ref;
            double aerr = fabs(err);
            if (!isfinite(aerr) || aerr > cap * sp_y) {
                wy = isfinite(aerr) ? ((cap * sp_y) / aerr) : 0.0;
                y_eff = y_ref + copysign(cap * sp_y, err);
            }
        }
        if (!isfinite(y_eff) || fabs(y_eff) > max_raw_abs) {
            y_eff = copysign(max_raw_abs, isfinite(y_eff) ? y_eff : yr);
        }
        wy_arr[r] = wy;
        y_eff_arr[r] = y_eff;
        if (wy < wy_min) {
            wy_min = wy;
        }
    }

    double w_eff = w * wy_min;
    if (!isfinite(w_eff) || w_eff < 0.0) {
        w_eff = 0.0;
    }

    double sqrt_w = sqrt(w_eff);
    for (int32_t j = 0; j < k; ++j) {
        sx[j] = sqrt_w * x_win[j];
        if (!isfinite(sx[j])) {
            sx[j] = 0.0;
        }
    }

    /* Prevent Gram/rhs overflow under gamma == 1.0 by jointly halving (G, rhs) if near ceiling */
    double max_gram_or_rhs = 0.0;
    for (int32_t j = 0; j < k; ++j) {
        double gii = fabs(st->G[(size_t)j * (size_t)k + (size_t)j]);
        if (gii > max_gram_or_rhs) max_gram_or_rhs = gii;
    }
    for (int32_t r = 0; r < m; ++r) {
        const double *rhs_r = st->rhs + (size_t)r * (size_t)k;
        for (int32_t j = 0; j < k; ++j) {
            double av = fabs(rhs_r[j]);
            if (av > max_gram_or_rhs) max_gram_or_rhs = av;
        }
    }
    if (max_gram_or_rhs > 1e250) {
        size_t kk = (size_t)k * (size_t)k;
        size_t km = (size_t)k * (size_t)m;
        for (size_t idx = 0; idx < kk; ++idx) st->G[idx] *= 0.5;
        for (size_t idx = 0; idx < km; ++idx) st->rhs[idx] *= 0.5;
    }

    /* Update EWMA Gram: G = gamma * G + sx * sx^T (strictly PSD & symmetric) */
    simd_scale_add_rank1(st->G, st->gamma, sx, k);

    /* Update EWMA cross-moment rhs with the exact same effective weight w_eff */
    for (int32_t r = 0; r < m; ++r) {
        double y_eff = y_eff_arr[r];
        double wy = wy_arr[r];
        double eff_scale = w_eff * y_eff;
        if (!isfinite(eff_scale)) {
            eff_scale = 0.0;
        }
        double *rhs_r = st->rhs + (size_t)r * (size_t)k;
        for (int32_t j = 0; j < k; ++j) {
            double next_val = st->gamma * rhs_r[j] + eff_scale * x_win[j];
            if (isfinite(next_val)) {
                rhs_r[j] = next_val;
            }
        }

        /* Robustly update target location/spread (zero DC-squared pollution on first sample!) */
        if (st->n_updates == 0) {
            st->mu_y[r] = y_eff;
            st->var_y[r] = 0.0;
        } else {
            double alpha_y = fmin(1.0 / (double)(st->n_updates + 1), 1.0 - st->gamma + 0.05);
            double eta_y = alpha_y * wy;
            double dy = y_eff - st->mu_y[r];
            double next_mu = st->mu_y[r] + eta_y * dy;
            double next_var = (1.0 - eta_y) * (st->var_y[r] + eta_y * dy * dy);
            if (isfinite(next_mu)) st->mu_y[r] = next_mu;
            if (isfinite(next_var) && next_var >= 0.0) st->var_y[r] = next_var;
        }
    }

    /* Robustly update feature location/spread using Mallows weight */
    if (st->n_updates == 0) {
        for (int32_t j = 0; j < k; ++j) {
            st->mu_x[j] = x_win[j];
            st->var_x[j] = 0.0;
        }
    } else {
        double base_alpha = fmin(1.0 / (double)(st->n_updates + 1), 1.0 - st->gamma + 0.05);
        double eta_x = base_alpha * w;
        for (int32_t j = 0; j < k; ++j) {
            double dx = x_win[j] - st->mu_x[j];
            double next_mu = st->mu_x[j] + eta_x * dx;
            double next_var = (1.0 - eta_x) * (st->var_x[j] + eta_x * dx * dx);
            if (isfinite(next_mu)) st->mu_x[j] = next_mu;
            if (isfinite(next_var) && next_var >= 0.0) st->var_x[j] = next_var;
        }
    }

    st->n_updates++;
    st->weights_dirty = true;
    pthread_mutex_unlock(&st->mu);
    if (out_weight) {
        *out_weight = w_eff;
    }
    return TSJIT_OK;
}

TSJIT_API int32_t tsjit_schur_predict(
    tsjit_schur_state_t *st,
    const double *x,
    double *out_y)
{
    if (!st || !x || !out_y) {
        return TSJIT_ERR_BAD_ARG;
    }
    pthread_mutex_lock(&st->mu);
    int32_t rc = schur_refresh_weights(st);
    if (rc != TSJIT_OK) {
        pthread_mutex_unlock(&st->mu);
        return rc;
    }
    for (int32_t r = 0; r < st->m; ++r) {
        out_y[r] = simd_dot(st->W + (size_t)r * (size_t)st->k, x, st->k);
    }
    pthread_mutex_unlock(&st->mu);
    return TSJIT_OK;
}
