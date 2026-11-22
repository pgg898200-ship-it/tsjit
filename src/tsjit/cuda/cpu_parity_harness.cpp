// Copyright 2026 tsjit contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Standalone C++20 ASan+UBSan math parity verifier testing all 4 tsjit GPU
// numerical contracts on CPU:
//   1. 25-comparator sort9 across all 512 0-1 sequences + NaN preservation.
//   2. ADAPT-IPCG FP64 & FP32 parity vs direct Cholesky (100x diag shock,
//      kappa=1e8 Gram, rz <= 0, NaN in rhs, true-residual verification).
//   3. Direct Scaled-FP16 & FP16x3 Ozaki math parity on raw 1e38, 1e-40
//      subnormals, ex+ew > 127 zero-cancelled dot products, and 1000x
//      weight & activation outliers (without RMSNorm pre-scaling).
//   4. Exact AsyncFuse additive monoid + T3 design-effect LOO sharded-vs-central
//      parity on W=4 skewed and W=8 random partitions (max_diff <= 1e-6).

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace {

int g_failures = 0;

void check_cond(bool cond, const char* msg) {
  if (!cond) {
    std::fprintf(stderr, "[FAIL] %s\n", msg);
    ++g_failures;
  }
}

// ---------------------------------------------------------------------------
// Contract 1: 25-comparator NaN-preserving sort9
// ---------------------------------------------------------------------------

inline void cswap_nan_safe(float& a, float& b) {
  const bool a_nan = std::isnan(a);
  const bool b_nan = std::isnan(b);
  // Push NaNs to the high end while strictly preserving exact NaN count.
  if ((a_nan && !b_nan) || (!a_nan && !b_nan && b < a)) {
    const float tmp = a;
    a = b;
    b = tmp;
  }
}

void sort9(std::array<float, 9>& v) {
  // Optimal 25-comparator sorting network for 9 inputs.
  constexpr std::array<std::pair<int, int>, 25> kNet = {{
      {0, 1}, {3, 4}, {6, 7},
      {1, 2}, {4, 5}, {7, 8},
      {0, 1}, {3, 4}, {6, 7},
      {0, 3}, {3, 6}, {0, 3},
      {1, 4}, {4, 7}, {1, 4},
      {2, 5}, {5, 8}, {2, 5},
      {1, 3}, {5, 7}, {2, 6},
      {4, 6}, {2, 4}, {2, 3}, {5, 6},
  }};
  for (const auto& [i, j] : kNet) {
    cswap_nan_safe(v[static_cast<size_t>(i)], v[static_cast<size_t>(j)]);
  }
}

void verify_contract_1_sort9() {
  int zero_one_fails = 0;
  for (uint32_t mask = 0; mask < 512u; ++mask) {
    std::array<float, 9> v{};
    int ones = 0;
    for (int i = 0; i < 9; ++i) {
      const int bit = static_cast<int>((mask >> i) & 1u);
      v[static_cast<size_t>(i)] = static_cast<float>(bit);
      ones += bit;
    }
    sort9(v);
    bool ok = true;
    int ones_after = 0;
    for (int i = 0; i < 9; ++i) {
      if (v[static_cast<size_t>(i)] == 1.0f) {
        ++ones_after;
      }
      if (i > 0 && v[static_cast<size_t>(i)] < v[static_cast<size_t>(i - 1)]) {
        ok = false;
      }
    }
    if (!ok || ones_after != ones) {
      ++zero_one_fails;
    }
  }
  check_cond(zero_one_fails == 0, "sort9 0-1 principle across all 512 binary sequences");

  // NaN preservation: 1 NaN at every input slot stays exactly 1 NaN at slot 8,
  // with remaining 8 finite elements sorted ascending.
  const float qnan = std::numeric_limits<float>::quiet_NaN();
  const std::array<float, 8> base = {9.5f, -3.25f, 0.0f, 1e20f, -1e20f, 2.5f, 2.5f, -0.0f};
  for (int nan_pos = 0; nan_pos < 9; ++nan_pos) {
    std::array<float, 9> v{};
    int idx = 0;
    for (int i = 0; i < 9; ++i) {
      v[static_cast<size_t>(i)] = (i == nan_pos) ? qnan : base[static_cast<size_t>(idx++)];
    }
    sort9(v);
    int nan_count = 0;
    for (float x : v) {
      if (std::isnan(x)) {
        ++nan_count;
      }
    }
    check_cond(nan_count == 1, "sort9 preserves exact count of 1 NaN");
    check_cond(std::isnan(v[8]), "sort9 places single NaN at index 8");
    for (int i = 1; i < 8; ++i) {
      check_cond(!std::isnan(v[static_cast<size_t>(i)]) &&
                     v[static_cast<size_t>(i)] >= v[static_cast<size_t>(i - 1)],
                 "sort9 sorts finite values around NaN");
    }
  }

  // Multi-NaN + +/-Inf check.
  std::array<float, 9> multi = {qnan, std::numeric_limits<float>::infinity(), -5.0f,
                                qnan, -std::numeric_limits<float>::infinity(), 3.0f,
                                1.0f, 0.0f, 7.0f};
  sort9(multi);
  check_cond(std::isnan(multi[7]) && std::isnan(multi[8]), "sort9 preserves 2 NaNs at end");
  check_cond(multi[0] == -std::numeric_limits<float>::infinity() &&
                 multi[6] == std::numeric_limits<float>::infinity(),
             "sort9 preserves +/-Inf ordering");
  std::printf("[PASS] Contract 1: sort9 (512/512 0-1 sequences, NaN preservation)\n");
}

// ---------------------------------------------------------------------------
// Contract 2: ADAPT-IPCG FP64 & FP32 parity vs direct Cholesky
// ---------------------------------------------------------------------------

bool cholesky_solve_f64(const std::vector<double>& A,
                        const std::vector<double>& rhs,
                        std::vector<double>& x_out,
                        std::vector<double>* inv_out,
                        int K) {
  std::vector<double> L(static_cast<size_t>(K) * static_cast<size_t>(K), 0.0);
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j <= i; ++j) {
      double sum = A[static_cast<size_t>(i) * K + j];
      for (int k = 0; k < j; ++k) {
        sum -= L[static_cast<size_t>(i) * K + k] * L[static_cast<size_t>(j) * K + k];
      }
      if (i == j) {
        if (!(sum > 0.0) || !std::isfinite(sum)) {
          return false;
        }
        L[static_cast<size_t>(i) * K + j] = std::sqrt(sum);
      } else {
        L[static_cast<size_t>(i) * K + j] = sum / L[static_cast<size_t>(j) * K + j];
      }
    }
  }

  auto solve_vec = [&](const std::vector<double>& b, std::vector<double>& x) {
    std::vector<double> y(static_cast<size_t>(K), 0.0);
    for (int i = 0; i < K; ++i) {
      double s = b[static_cast<size_t>(i)];
      for (int k = 0; k < i; ++k) {
        s -= L[static_cast<size_t>(i) * K + k] * y[static_cast<size_t>(k)];
      }
      y[static_cast<size_t>(i)] = s / L[static_cast<size_t>(i) * K + i];
    }
    x.assign(static_cast<size_t>(K), 0.0);
    for (int i = K - 1; i >= 0; --i) {
      double s = y[static_cast<size_t>(i)];
      for (int k = i + 1; k < K; ++k) {
        s -= L[static_cast<size_t>(k) * K + i] * x[static_cast<size_t>(k)];
      }
      x[static_cast<size_t>(i)] = s / L[static_cast<size_t>(i) * K + i];
    }
  };

  solve_vec(rhs, x_out);
  if (inv_out != nullptr) {
    inv_out->assign(static_cast<size_t>(K) * static_cast<size_t>(K), 0.0);
    std::vector<double> e(static_cast<size_t>(K), 0.0);
    std::vector<double> col(static_cast<size_t>(K), 0.0);
    for (int c = 0; c < K; ++c) {
      std::fill(e.begin(), e.end(), 0.0);
      e[static_cast<size_t>(c)] = 1.0;
      solve_vec(e, col);
      for (int r = 0; r < K; ++r) {
        (*inv_out)[static_cast<size_t>(r) * K + c] = col[static_cast<size_t>(r)];
      }
    }
  }
  return true;
}

template <typename T>
struct IpcgCpuResult {
  bool converged = false;
  int iters = 0;
  T true_r2 = std::numeric_limits<T>::quiet_NaN();
};

template <typename T>
void matvec_cpu(const std::vector<T>& M, const std::vector<T>& v, std::vector<T>& out, int K) {
  for (int i = 0; i < K; ++i) {
    T acc = T(0);
    const size_t row_off = static_cast<size_t>(i) * static_cast<size_t>(K);
    for (int j = 0; j < K; ++j) {
      acc = std::fma(M[row_off + static_cast<size_t>(j)], v[static_cast<size_t>(j)], acc);
    }
    out[static_cast<size_t>(i)] = acc;
  }
}

template <typename T>
T dot_cpu(const std::vector<T>& a, const std::vector<T>& b, int K) {
  T acc = T(0);
  for (int i = 0; i < K; ++i) {
    acc = std::fma(a[static_cast<size_t>(i)], b[static_cast<size_t>(i)], acc);
  }
  return acc;
}

template <typename T>
IpcgCpuResult<T> ipcg_solve_cpu(const std::vector<T>& A,
                                const std::vector<T>* M0,
                                const std::vector<T>& rhs,
                                std::vector<T>& theta_inout,
                                int K,
                                T tol,
                                int max_iters) {
  IpcgCpuResult<T> res{};
  std::vector<T> r_s(static_cast<size_t>(K), T(0));
  std::vector<T> z_s(static_cast<size_t>(K), T(0));
  std::vector<T> p_s(static_cast<size_t>(K), T(0));
  std::vector<T> ap_s(static_cast<size_t>(K), T(0));

  T rhs_max = T(0);
  T x0_max = T(0);
  for (int i = 0; i < K; ++i) {
    const T bv = rhs[static_cast<size_t>(i)];
    const T xv = theta_inout[static_cast<size_t>(i)];
    if (!std::isfinite(bv) || !std::isfinite(xv)) {
      return res;
    }
    rhs_max = std::max(rhs_max, std::fabs(bv));
    x0_max = std::max(x0_max, std::fabs(xv));
  }
  const T tol2 = tol * tol;
  if (!std::isfinite(tol2) || tol < T(0)) {
    return res;
  }

  constexpr T kMinSafe = (sizeof(T) == 4) ? T(1e-30) : T(1e-200);
  constexpr T kMaxSafe = (sizeof(T) == 4) ? T(1e30) : T(1e200);
  constexpr T kMaxInitRatio = (sizeof(T) == 4) ? T(1e12) : T(1e100);
  const T ref_mag = (rhs_max > T(0)) ? rhs_max : ((x0_max > T(0)) ? x0_max : T(1));
  const T rhs_scale = (ref_mag > T(0)) ? std::clamp(ref_mag, kMinSafe, kMaxSafe) : T(1);
  const T inv_rhs_scale = T(1) / rhs_scale;

  std::vector<T> rhs_s(static_cast<size_t>(K), T(0));
  std::vector<T> x_s(static_cast<size_t>(K), T(0));
  const T x0_scaled_max = x0_max * inv_rhs_scale;
  const bool x0_scales_ok = std::isfinite(x0_scaled_max) && (x0_scaled_max <= kMaxInitRatio);
  for (int i = 0; i < K; ++i) {
    rhs_s[static_cast<size_t>(i)] = rhs[static_cast<size_t>(i)] * inv_rhs_scale;
    x_s[static_cast<size_t>(i)] = x0_scales_ok ? (theta_inout[static_cast<size_t>(i)] * inv_rhs_scale) : T(0);
  }

  const T rhs_n2 = dot_cpu(rhs_s, rhs_s, K);
  if (!std::isfinite(rhs_n2)) {
    return res;
  }
  const T target_r2 = (rhs_n2 > T(0)) ? (tol2 * rhs_n2) : T(0);
  if (!std::isfinite(target_r2)) {
    return res;
  }

  auto apply_precond = [&](const std::vector<T>& r, std::vector<T>& z) {
    if (M0 != nullptr) {
      matvec_cpu(*M0, r, z, K);
    } else {
      for (int i = 0; i < K; ++i) {
        const T d = A[static_cast<size_t>(i) * K + i];
        z[static_cast<size_t>(i)] =
            (d > T(0) && std::isfinite(d)) ? (r[static_cast<size_t>(i)] / d)
                                           : std::numeric_limits<T>::quiet_NaN();
      }
    }
  };

  matvec_cpu(A, x_s, ap_s, K);
  for (int i = 0; i < K; ++i) {
    r_s[static_cast<size_t>(i)] = rhs_s[static_cast<size_t>(i)] - ap_s[static_cast<size_t>(i)];
  }
  T r_n2 = dot_cpu(r_s, r_s, K);
  if (!std::isfinite(r_n2) && x0_scales_ok && (x0_max > T(0))) {
    std::fill(x_s.begin(), x_s.end(), T(0));
    matvec_cpu(A, x_s, ap_s, K);
    for (int i = 0; i < K; ++i) {
      r_s[static_cast<size_t>(i)] = rhs_s[static_cast<size_t>(i)] - ap_s[static_cast<size_t>(i)];
    }
    r_n2 = dot_cpu(r_s, r_s, K);
  }
  bool verified_true_r2 = false;
  T true_r2_scaled = std::numeric_limits<T>::quiet_NaN();

  if (std::isfinite(r_n2)) {
    if (r_n2 <= target_r2) {
      true_r2_scaled = r_n2;
      verified_true_r2 = true;
    } else if (max_iters > 0) {
      apply_precond(r_s, z_s);
      p_s = z_s;
      T rz = dot_cpu(r_s, z_s, K);
      int restarts = 0;

      if (rz > T(0) && std::isfinite(rz)) {
        for (int it = 1; it <= max_iters; ++it) {
          matvec_cpu(A, p_s, ap_s, K);
          const T pap = dot_cpu(p_s, ap_s, K);
          res.iters = it;
          if (!(pap > T(0) && std::isfinite(pap))) {
            break;
          }
          const T alpha = rz / pap;
          if (!std::isfinite(alpha)) {
            break;
          }
          for (int i = 0; i < K; ++i) {
            x_s[static_cast<size_t>(i)] =
                std::fma(alpha, p_s[static_cast<size_t>(i)], x_s[static_cast<size_t>(i)]);
            r_s[static_cast<size_t>(i)] =
                std::fma(-alpha, ap_s[static_cast<size_t>(i)], r_s[static_cast<size_t>(i)]);
          }
          r_n2 = dot_cpu(r_s, r_s, K);
          if (!std::isfinite(r_n2)) {
            break;
          }
          if (r_n2 <= target_r2) {
            matvec_cpu(A, x_s, ap_s, K);
            for (int i = 0; i < K; ++i) {
              r_s[static_cast<size_t>(i)] = rhs_s[static_cast<size_t>(i)] - ap_s[static_cast<size_t>(i)];
            }
            const T chk = dot_cpu(r_s, r_s, K);
            if (std::isfinite(chk) && chk <= target_r2) {
              true_r2_scaled = chk;
              verified_true_r2 = true;
              break;
            }
            if (!std::isfinite(chk) || restarts >= 1 || it == max_iters) {
              true_r2_scaled = chk;
              verified_true_r2 = true;
              break;
            }
            restarts += 1;
            apply_precond(r_s, z_s);
            p_s = z_s;
            rz = dot_cpu(r_s, z_s, K);
            if (!(rz > T(0) && std::isfinite(rz))) {
              break;
            }
            continue;
          }
          if (it == max_iters) {
            break;
          }
          apply_precond(r_s, z_s);
          const T rz_new = dot_cpu(r_s, z_s, K);
          if (!(rz_new > T(0) && std::isfinite(rz_new))) {
            break;
          }
          const T beta = rz_new / rz;
          if (!std::isfinite(beta)) {
            break;
          }
          for (int i = 0; i < K; ++i) {
            p_s[static_cast<size_t>(i)] =
                std::fma(beta, p_s[static_cast<size_t>(i)], z_s[static_cast<size_t>(i)]);
          }
          rz = rz_new;
        }
      }
    }
  }

  // Final True-Residual Verification
  if (!verified_true_r2) {
    matvec_cpu(A, x_s, ap_s, K);
    for (int i = 0; i < K; ++i) {
      r_s[static_cast<size_t>(i)] = rhs_s[static_cast<size_t>(i)] - ap_s[static_cast<size_t>(i)];
    }
    true_r2_scaled = dot_cpu(r_s, r_s, K);
  }
  bool x_all_fin = true;
  for (int i = 0; i < K; ++i) {
    x_s[static_cast<size_t>(i)] *= rhs_scale;
    if (!std::isfinite(x_s[static_cast<size_t>(i)])) {
      x_all_fin = false;
    }
  }
  res.true_r2 = std::isfinite(true_r2_scaled) ? ((true_r2_scaled * rhs_scale) * rhs_scale) : true_r2_scaled;
  res.converged = std::isfinite(true_r2_scaled) && x_all_fin && (true_r2_scaled <= target_r2);
  if (res.converged) {
    theta_inout = x_s;
  }
  return res;
}

void verify_contract_2_ipcg() {
  std::mt19937_64 rng(20260417ULL);
  std::normal_distribution<double> nd(0.0, 1.0);
  const int K = 24;

  // Build SPD A0 = B^T B / K + 0.5 I and incremental A = A0 + u u^T.
  std::vector<double> B(static_cast<size_t>(K) * K);
  for (double& v : B) v = nd(rng);
  std::vector<double> A0(static_cast<size_t>(K) * K, 0.0);
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < K; ++j) {
      double s = (i == j) ? 0.5 : 0.0;
      for (int t = 0; t < K; ++t) {
        s += B[static_cast<size_t>(t) * K + i] * B[static_cast<size_t>(t) * K + j] / K;
      }
      A0[static_cast<size_t>(i) * K + j] = s;
    }
  }
  std::vector<double> rhs0(static_cast<size_t>(K));
  for (double& v : rhs0) v = nd(rng);
  std::vector<double> theta0, M0;
  check_cond(cholesky_solve_f64(A0, rhs0, theta0, &M0, K), "Cholesky on A0 succeeds");

  // (2a) Incremental low-rank update: FP64 & FP32 parity vs Cholesky.
  std::vector<double> u(static_cast<size_t>(K));
  for (double& v : u) v = 0.25 * nd(rng);
  std::vector<double> A1 = A0;
  std::vector<double> rhs1 = rhs0;
  for (int i = 0; i < K; ++i) {
    rhs1[static_cast<size_t>(i)] += 0.2 * u[static_cast<size_t>(i)];
    for (int j = 0; j < K; ++j) {
      A1[static_cast<size_t>(i) * K + j] += u[static_cast<size_t>(i)] * u[static_cast<size_t>(j)];
    }
  }
  std::vector<double> ref1;
  check_cond(cholesky_solve_f64(A1, rhs1, ref1, nullptr, K), "Cholesky on A1 succeeds");

  std::vector<double> th_f64 = theta0;
  auto r64 = ipcg_solve_cpu<double>(A1, &M0, rhs1, th_f64, K, 1e-10, 16);
  check_cond(r64.converged && r64.iters <= 6, "IPCG FP64 fast convergence on rank-1 update");
  double max_err_64 = 0.0;
  for (int i = 0; i < K; ++i) max_err_64 = std::max(max_err_64, std::abs(th_f64[i] - ref1[i]));
  check_cond(max_err_64 < 1e-9, "IPCG FP64 parity vs direct Cholesky");

  std::vector<float> A1_f(A1.begin(), A1.end()), M0_f(M0.begin(), M0.end()),
      rhs1_f(rhs1.begin(), rhs1.end()), th_f32(theta0.begin(), theta0.end());
  auto r32 = ipcg_solve_cpu<float>(A1_f, &M0_f, rhs1_f, th_f32, K, 1e-5f, 16);
  check_cond(r32.converged, "IPCG FP32 convergence on well-conditioned rank-1 update");
  double max_err_32 = 0.0;
  for (int i = 0; i < K; ++i) max_err_32 = std::max(max_err_32, std::abs(double(th_f32[i]) - ref1[i]));
  check_cond(max_err_32 < 1e-4, "IPCG FP32 parity vs direct Cholesky");

  // (2b) 100x diagonal shock: verify convergence with sufficient iters AND untouched theta on cap.
  std::vector<double> A_shock = A0;
  for (int i = 0; i < K; ++i) {
    A_shock[static_cast<size_t>(i) * K + i] *= 100.0;
  }
  std::vector<double> ref_shock;
  check_cond(cholesky_solve_f64(A_shock, rhs1, ref_shock, nullptr, K), "Cholesky on 100x shock");
  std::vector<double> th_shock = theta0;
  auto r_shock = ipcg_solve_cpu<double>(A_shock, &M0, rhs1, th_shock, K, 1e-9, 48);
  check_cond(r_shock.converged, "IPCG FP64 converges under 100x diagonal shock");
  double shock_err = 0.0;
  for (int i = 0; i < K; ++i) shock_err = std::max(shock_err, std::abs(th_shock[i] - ref_shock[i]));
  check_cond(shock_err < 1e-8, "IPCG FP64 100x diagonal shock parity vs Cholesky");

  std::vector<double> th_cap = theta0;
  auto r_cap = ipcg_solve_cpu<double>(A_shock, &M0, rhs1, th_cap, K, 1e-12, 1);
  check_cond(!r_cap.converged, "IPCG flags non-convergence when max_iters exhausted under shock");
  check_cond(th_cap == theta0, "Non-converged IPCG leaves theta bit-identical untouched");

  // (2c) Ill-conditioned kappa = 1e8 Gram: build Q * diag(1 .. 1e-8) * Q^T via Gram-Schmidt.
  std::vector<double> Q(static_cast<size_t>(K) * K, 0.0);
  for (int c = 0; c < K; ++c) {
    for (int r = 0; r < K; ++r) Q[static_cast<size_t>(r) * K + c] = nd(rng);
    for (int prev = 0; prev < c; ++prev) {
      double dot = 0.0;
      for (int r = 0; r < K; ++r)
        dot += Q[static_cast<size_t>(r) * K + c] * Q[static_cast<size_t>(r) * K + prev];
      for (int r = 0; r < K; ++r)
        Q[static_cast<size_t>(r) * K + c] -= dot * Q[static_cast<size_t>(r) * K + prev];
    }
    double norm = 0.0;
    for (int r = 0; r < K; ++r) norm += Q[static_cast<size_t>(r) * K + c] * Q[static_cast<size_t>(r) * K + c];
    norm = std::sqrt(norm);
    for (int r = 0; r < K; ++r) Q[static_cast<size_t>(r) * K + c] /= norm;
  }
  std::vector<double> A_ill(static_cast<size_t>(K) * K, 0.0);
  std::vector<double> M_ill(static_cast<size_t>(K) * K, 0.0);
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < K; ++j) {
      double sa = 0.0;
      double sm = 0.0;
      for (int d = 0; d < K; ++d) {
        const double lam = std::pow(10.0, -8.0 * double(d) / double(K - 1));
        const double qid = Q[static_cast<size_t>(i) * K + d];
        const double qjd = Q[static_cast<size_t>(j) * K + d];
        sa += lam * qid * qjd;
        // Slightly perturbed preconditioner (10% spectral drift) to mimic incremental update
        const double lam_m = lam * (1.0 + 0.1 * ((d % 2 == 0) ? 1.0 : -1.0));
        sm += (1.0 / lam_m) * qid * qjd;
      }
      A_ill[static_cast<size_t>(i) * K + j] = sa;
      M_ill[static_cast<size_t>(i) * K + j] = sm;
    }
  }
  std::vector<double> ref_ill;
  check_cond(cholesky_solve_f64(A_ill, rhs0, ref_ill, nullptr, K), "Cholesky on kappa=1e8 Gram");
  std::vector<double> th_ill(static_cast<size_t>(K), 0.0);
  auto r_ill64 = ipcg_solve_cpu<double>(A_ill, &M_ill, rhs0, th_ill, K, 1e-8, 32);
  check_cond(r_ill64.converged, "IPCG FP64 solves kappa=1e8 ill-conditioned Gram");
  double ill_rel_res = std::sqrt(r_ill64.true_r2 / dot_cpu(rhs0, rhs0, K));
  check_cond(ill_rel_res <= 1e-8, "IPCG FP64 true residual verified <= 1e-8 on kappa=1e8");

  // In FP32 on kappa=1e8 with Jacobi and tight tol=1e-7, true residual cannot reach 1e-7 -> must reject.
  std::vector<float> A_ill_f(A_ill.begin(), A_ill.end()), rhs0_f(rhs0.begin(), rhs0.end());
  std::vector<float> th_ill_f(static_cast<size_t>(K), 1.25f);
  const std::vector<float> th_ill_f_orig = th_ill_f;
  auto r_ill32 = ipcg_solve_cpu<float>(A_ill_f, nullptr, rhs0_f, th_ill_f, K, 1e-7f, 16);
  check_cond(!r_ill32.converged && th_ill_f == th_ill_f_orig,
             "IPCG FP32 rejects unsolvable kappa=1e8 & leaves theta untouched for Cholesky fallback");

  // (2d) rz <= 0 (negative-definite preconditioner -M0).
  std::vector<double> neg_M0 = M0;
  for (double& v : neg_M0) v = -v;
  std::vector<double> th_neg = theta0;
  auto r_neg = ipcg_solve_cpu<double>(A1, &neg_M0, rhs1, th_neg, K, 1e-8, 16);
  check_cond(!r_neg.converged && th_neg == theta0, "IPCG rejects rz <= 0 and leaves theta untouched");

  // (2e) NaN in rhs & NaN in A.
  std::vector<double> rhs_nan = rhs1;
  rhs_nan[3] = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> th_nan = theta0;
  auto r_nan = ipcg_solve_cpu<double>(A1, &M0, rhs_nan, th_nan, K, 1e-8, 16);
  check_cond(!r_nan.converged && th_nan == theta0, "IPCG rejects NaN in rhs and leaves theta untouched");

  std::vector<double> A_nan = A1;
  A_nan[2 * K + 5] = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> th_anan = theta0;
  auto r_anan = ipcg_solve_cpu<double>(A_nan, &M0, rhs1, th_anan, K, 1e-8, 16);
  check_cond(!r_anan.converged && th_anan == theta0, "IPCG rejects NaN in A and leaves theta untouched");

  // (2f) Extreme FP32 rhs magnitudes (1e25 and 1e-25) AND stale warm-start x_init must NOT overflow!
  for (float scale : {1e25f, 1e-25f}) {
    std::vector<float> rhs_ext(static_cast<size_t>(K));
    std::vector<float> th_ext(static_cast<size_t>(K), (scale < 1.0f) ? 1e10f : 0.0f);
    for (int i = 0; i < K; ++i) {
      rhs_ext[static_cast<size_t>(i)] = rhs1_f[static_cast<size_t>(i)] * scale;
    }
    auto r_ext = ipcg_solve_cpu<float>(A1_f, &M0_f, rhs_ext, th_ext, K, 1e-5f, 16);
    check_cond(r_ext.converged, "IPCG FP32 converges cleanly on extreme rhs (1e25 / 1e-25 + stale x_init)");
    double max_rel = 0.0;
    double ref_norm = 0.0;
    for (int i = 0; i < K; ++i) {
      const double expected = ref1[static_cast<size_t>(i)] * static_cast<double>(scale);
      max_rel = std::max(max_rel, std::abs(static_cast<double>(th_ext[static_cast<size_t>(i)]) - expected));
      ref_norm = std::max(ref_norm, std::abs(expected));
    }
    check_cond(max_rel / std::max(ref_norm, 1e-300) < 1e-4,
               "IPCG FP32 extreme rhs (1e25 / 1e-25) matches scaled Cholesky reference");
  }
  // Zero rhs + NaN in A + non-zero x_init MUST still be rejected (0*NaN=NaN preserved on retry).
  {
    std::vector<double> rhs_zero(static_cast<size_t>(K), 0.0);
    std::vector<double> th_zero_nan(static_cast<size_t>(K), 1.0);
    const std::vector<double> th_zero_nan_orig = th_zero_nan;
    auto r_zn = ipcg_solve_cpu<double>(A_nan, &M0, rhs_zero, th_zero_nan, K, 1e-8, 16);
    check_cond(!r_zn.converged && th_zero_nan == th_zero_nan_orig,
               "IPCG rejects NaN in A even when rhs == 0 and warm-start triggers zero retry");
  }

  std::printf("[PASS] Contract 2: ADAPT-IPCG FP64/FP32 (100x shock, kappa=1e8, rz<=0, NaN rhs/A, rhs 1e+/-25)\n");
}

// ---------------------------------------------------------------------------
// Contract 3: Direct Scaled-FP16 & FP16x3 math parity (IEEE binary16 exact)
// ---------------------------------------------------------------------------

uint16_t f32_to_f16_rne(float f) {
  const uint32_t x = std::bit_cast<uint32_t>(f);
  const uint32_t sign = (x >> 16) & 0x8000u;
  const int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127;
  uint32_t mant = x & 0x007FFFFFu;

  if (exp == 128) {
    if (mant == 0) {
      return static_cast<uint16_t>(sign | 0x7C00u);  // Inf
    }
    return static_cast<uint16_t>(sign | 0x7E00u | (mant >> 13));  // NaN
  }
  if (exp > 15) {
    return static_cast<uint16_t>(sign | 0x7C00u);  // Overflow to Inf
  }
  if (exp < -25) {
    return static_cast<uint16_t>(sign);  // Underflow to signed 0 (exp == -25 can round to 2^-24)
  }
  if (exp < -14) {
    // Subnormal binary16
    mant |= 0x00800000u;
    const uint32_t shift = static_cast<uint32_t>(-14 - exp + 13);
    const uint32_t half = 1u << (shift - 1u);
    const uint32_t mask = (1u << shift) - 1u;
    const uint32_t rem = mant & mask;
    uint32_t q = mant >> shift;
    if (rem > half || (rem == half && (q & 1u))) {
      ++q;
    }
    return static_cast<uint16_t>(sign | q);
  }
  // Normal binary16 with round-to-nearest-even
  const uint32_t rem = mant & 0x1FFFu;
  uint32_t q = mant >> 13;
  if (rem > 0x1000u || (rem == 0x1000u && (q & 1u))) {
    ++q;
    if (q == 0x0400u) {
      q = 0;
      const int32_t e_next = exp + 1;
      if (e_next > 15) {
        return static_cast<uint16_t>(sign | 0x7C00u);
      }
      return static_cast<uint16_t>(sign | (static_cast<uint32_t>(e_next + 15) << 10));
    }
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp + 15) << 10) | q);
}

float f16_to_f32(uint16_t h) {
  const uint32_t sign = (static_cast<uint32_t>(h & 0x8000u)) << 16;
  const uint32_t exp = (h >> 10) & 0x1Fu;
  const uint32_t mant = h & 0x03FFu;
  if (exp == 0) {
    if (mant == 0) {
      return std::bit_cast<float>(sign);
    }
    // Subnormal binary16
    uint32_t m = mant;
    int32_t e = -14;
    while ((m & 0x0400u) == 0) {
      m <<= 1;
      --e;
    }
    m &= 0x03FFu;
    const uint32_t bits = sign | (static_cast<uint32_t>(e + 127) << 23) | (m << 13);
    return std::bit_cast<float>(bits);
  }
  if (exp == 31) {
    const uint32_t bits = sign | 0x7F800000u | (mant << 13);
    return std::bit_cast<float>(bits);
  }
  const uint32_t bits = sign | ((exp + 112u) << 23) | (mant << 13);
  return std::bit_cast<float>(bits);
}

inline float pow2_f32(int32_t e) {
  const uint32_t bits = static_cast<uint32_t>(e + 127) << 23;
  return std::bit_cast<float>(bits);
}

struct SplitRow {
  std::vector<uint16_t> v0;
  std::vector<uint16_t> v1;
  int32_t exp_e = 0;
  int32_t bad_code = 0;  // 0=normal, 1=zero, 2=non-finite
  float crest = 0.0f;
};

SplitRow split_row_cpu(const std::vector<float>& row) {
  const size_t K = row.size();
  SplitRow out;
  out.v0.assign(K, 0);
  out.v1.assign(K, 0);

  bool has_nonfin = false;
  float amax = 0.0f;
  for (size_t k = 0; k < K; ++k) {
    const float v = row[k];
    const float av = std::fabs(v);
    const bool fin = (av < std::numeric_limits<float>::infinity()) && (v == v);
    if (!fin) {
      has_nonfin = true;
    } else if (av > amax) {
      amax = av;
    }
  }

  if (has_nonfin) {
    out.bad_code = 2;
    return out;
  }
  if (amax == 0.0f) {
    out.bad_code = 1;
    return out;
  }

  constexpr float kMinNormal = 1.1754943508222875e-38f;  // 2^-126
  const bool is_sub = amax < kMinNormal;
  const float amax_norm = is_sub ? (amax * 16777216.0f) : amax;
  const int32_t sub_bias = is_sub ? 24 : 0;
  const uint32_t bits = std::bit_cast<uint32_t>(amax_norm);
  out.exp_e = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 - sub_bias;

  const int32_t sh = -out.exp_e;
  const int32_t sh1 = sh / 2;
  const int32_t sh2 = sh - sh1;
  const float p1 = pow2_f32(sh1);
  const float p2 = pow2_f32(sh2);

  double sum_sq = 0.0;
  for (size_t k = 0; k < K; ++k) {
    const float v_s = (row[k] * p1) * p2;
    sum_sq += static_cast<double>(v_s) * static_cast<double>(v_s);
    const uint16_t h0 = f32_to_f16_rne(v_s);
    const float f0 = f16_to_f32(h0);
    const uint16_t h1 = f32_to_f16_rne((v_s - f0) * 2048.0f);
    out.v0[k] = h0;
    out.v1[k] = h1;
  }
  const double amax_s = static_cast<double>((amax * p1) * p2);
  const double rms_s = std::sqrt(sum_sq / static_cast<double>(K));
  out.crest = static_cast<float>(amax_s / std::max(rms_s, 1e-30));
  return out;
}

float dot_split_rows(const SplitRow& rx, const SplitRow& rw, int tier) {
  // tier: 1=Scaled-FP16, 2=Act-Escalated 2-GEMM, 3=FP16x3 3-GEMM
  if (rx.bad_code == 2 || rw.bad_code == 2) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  if (rx.bad_code == 1 || rw.bad_code == 1) {
    return 0.0f;
  }
  const size_t K = rx.v0.size();
  float acc0 = 0.0f;
  float acc_res = 0.0f;
  for (size_t k = 0; k < K; ++k) {
    const float x0 = f16_to_f32(rx.v0[k]);
    const float w0 = f16_to_f32(rw.v0[k]);
    acc0 = std::fma(x0, w0, acc0);
    if (tier >= 2) {
      const float x1 = f16_to_f32(rx.v1[k]);
      acc_res = std::fma(x1, w0, acc_res);
    }
    if (tier >= 3) {
      const float w1 = f16_to_f32(rw.v1[k]);
      acc_res = std::fma(x0, w1, acc_res);
    }
  }
  const float C = (tier >= 2) ? (acc0 + acc_res * (1.0f / 2048.0f)) : acc0;
  const int32_t e_tot = std::clamp(rx.exp_e + rw.exp_e, -300, 300);
  const int32_t e1 = e_tot / 3;
  const int32_t rem = e_tot - e1;
  const int32_t e2 = rem / 2;
  const int32_t e3 = rem - e2;
  return ((C * pow2_f32(e1)) * pow2_f32(e2)) * pow2_f32(e3);
}

void verify_contract_3_scaled_fp16() {
  std::mt19937_64 rng(991177ULL);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  const size_t K = 256;

  // (3a) Raw 1e38 inputs (NO RMSNorm pre-scaling!)
  std::vector<float> x_hi(K), w_hi(K);
  double ref_hi = 0.0;
  for (size_t k = 0; k < K; ++k) {
    x_hi[k] = 1e38f * (0.1f + 0.05f * nd(rng));
    w_hi[k] = 1e-3f * (0.2f + 0.05f * nd(rng));
    ref_hi += static_cast<double>(x_hi[k]) * static_cast<double>(w_hi[k]);
  }
  auto sx_hi = split_row_cpu(x_hi);
  auto sw_hi = split_row_cpu(w_hi);
  const float y1_hi = dot_split_rows(sx_hi, sw_hi, 1);
  const float y3_hi = dot_split_rows(sx_hi, sw_hi, 3);
  check_cond(std::isfinite(y1_hi) && std::isfinite(y3_hi), "Raw 1e38 stays finite");
  check_cond(std::abs(double(y1_hi) - ref_hi) / std::abs(ref_hi) < 2e-3,
             "Raw 1e38 Scaled-FP16 relative accuracy < 2e-3");
  check_cond(std::abs(double(y3_hi) - ref_hi) / std::abs(ref_hi) < 1e-5,
             "Raw 1e38 FP16x3 Ozaki relative accuracy < 1e-5");

  // (3b) Raw 1e-40 subnormal inputs (zero 1/sx -> +Inf blowup!)
  std::vector<float> x_sub(K), w_sub(K);
  double ref_sub = 0.0;
  for (size_t k = 0; k < K; ++k) {
    x_sub[k] = 1e-40f * (1.0f + 0.3f * nd(rng));
    w_sub[k] = 1e12f * (0.5f + 0.2f * nd(rng));
    ref_sub += static_cast<double>(x_sub[k]) * static_cast<double>(w_sub[k]);
  }
  auto sx_sub = split_row_cpu(x_sub);
  auto sw_sub = split_row_cpu(w_sub);
  const float y1_sub = dot_split_rows(sx_sub, sw_sub, 1);
  const float y3_sub = dot_split_rows(sx_sub, sw_sub, 3);
  check_cond(std::isfinite(y1_sub) && std::isfinite(y3_sub), "Raw 1e-40 subnormal stays finite");
  check_cond(std::abs(double(y1_sub) - ref_sub) / std::abs(ref_sub) < 2e-3,
             "Raw 1e-40 subnormal Scaled-FP16 relative accuracy < 2e-3");
  check_cond(std::abs(double(y3_sub) - ref_sub) / std::abs(ref_sub) < 1e-5,
             "Raw 1e-40 subnormal FP16x3 relative accuracy < 1e-5");

  // (3c) ex + ew > 127 zero-cancelled dot product (zero 0*Inf = NaN!)
  std::vector<float> x_zc = {1e25f, -1e25f, 1e25f, -1e25f};
  std::vector<float> w_zc = {1e25f, 1e25f, 1e25f, 1e25f};
  auto sx_zc = split_row_cpu(x_zc);
  auto sw_zc = split_row_cpu(w_zc);
  check_cond(sx_zc.exp_e + sw_zc.exp_e > 127, "Zero-cancel test has ex + ew > 127");
  const float y_zc1 = dot_split_rows(sx_zc, sw_zc, 1);
  const float y_zc3 = dot_split_rows(sx_zc, sw_zc, 3);
  check_cond(y_zc1 == 0.0f && !std::isnan(y_zc1), "ex+ew > 127 zero-cancel Scaled-FP16 is exact 0.0");
  check_cond(y_zc3 == 0.0f && !std::isnan(y_zc3), "ex+ew > 127 zero-cancel FP16x3 is exact 0.0");

  // (3d) 1000x weight & activation outliers with cancellation (without RMSNorm!)
  // Use K_out=512 so 2 cancelling outliers have crest = sqrt(512/2) = 16.0 > 12.0.
  const size_t K_out = 512;
  std::vector<float> x_out(K_out, 0.0f), w_out(K_out, 0.0f);
  for (size_t k = 2; k < K_out; ++k) {
    x_out[k] = 0.5f + 0.1f * nd(rng);
    w_out[k] = 0.5f + 0.1f * nd(rng);
  }
  x_out[0] = 1000.1234f;
  x_out[1] = -1000.1234f;
  w_out[0] = 1000.5678f;
  w_out[1] = 1000.5678f;
  double ref_out = 0.0;
  for (size_t k = 0; k < K_out; ++k) ref_out += double(x_out[k]) * double(w_out[k]);
  auto sx_out = split_row_cpu(x_out);
  auto sw_out = split_row_cpu(w_out);
  check_cond(sx_out.crest > 12.0f && sw_out.crest > 12.0f,
             "Crest detector flags 1000x weight & activation outliers (> 12.0)");
  const float y3_out = dot_split_rows(sx_out, sw_out, 3);
  const double rel_out3 = std::abs(double(y3_out) - ref_out) / std::abs(ref_out);
  check_cond(rel_out3 < 1e-3, "FP16x3 Ozaki accurately resolves 1000x outlier cancellation");

  // Also test activation-only 1000x outlier (crest_w <= 12, crest_x > 12) -> 2-GEMM act-escalated.
  std::vector<float> w_norm(K_out, 0.0f);
  for (size_t k = 0; k < K_out; ++k) w_norm[k] = 0.5f + 0.1f * nd(rng);
  w_norm[0] = 1.0f;
  w_norm[1] = 1.0f;
  double ref_act = 0.0;
  for (size_t k = 0; k < K_out; ++k) ref_act += double(x_out[k]) * double(w_norm[k]);
  auto sw_norm = split_row_cpu(w_norm);
  check_cond(sw_norm.crest <= 12.0f && sx_out.crest > 12.0f,
             "Activation-only 1000x outlier triggers 2-GEMM crest guard without weight escalation");
  const float y2_act = dot_split_rows(sx_out, sw_norm, 2);
  const double rel_act2 = std::abs(double(y2_act) - ref_act) / std::abs(ref_act);
  check_cond(rel_act2 < 5e-3, "2-GEMM act-escalated accurately resolves activation-only 1000x outlier");

  // (3e) NaN propagation check (bad_code == 2 -> NaN, never dropped!)
  std::vector<float> x_nan = x_hi;
  x_nan[17] = std::numeric_limits<float>::quiet_NaN();
  auto sx_nan = split_row_cpu(x_nan);
  check_cond(sx_nan.bad_code == 2, "NaN sets bad_code == 2");
  check_cond(std::isnan(dot_split_rows(sx_nan, sw_hi, 1)), "NaN propagates to output");

  // (3f) IEEE binary16 subnormal RNE boundary check (1.5 * 2^-25 rounds to 2^-24 = 0x0001).
  check_cond(f32_to_f16_rne(1.5f * pow2_f32(-25)) == 0x0001u,
             "IEEE binary16 subnormal RNE rounds 1.5*2^-25 up to 2^-24");
  check_cond(f32_to_f16_rne(pow2_f32(-25)) == 0x0000u,
             "IEEE binary16 subnormal RNE rounds tie 0.5*2^-24 to even 0");

  std::printf("[PASS] Contract 3: Scaled-FP16 & FP16x3 (1e38, 1e-40, ex+ew>127 zero-cancel, 1000x outliers)\n");
}

// ---------------------------------------------------------------------------
// Contract 4: AsyncFuse additive monoid + T3 design-effect LOO parity
// ---------------------------------------------------------------------------

struct SeriesItem {
  int group_id = 0;
  float a = 1.0f;
  std::vector<float> h;
  std::vector<float> phi;
};

std::vector<std::vector<float>> async_fuse_central(const std::vector<SeriesItem>& items,
                                                   int G,
                                                   int D,
                                                   float rho,
                                                   float kappa) {
  std::vector<std::vector<float>> S_phi(static_cast<size_t>(G), std::vector<float>(static_cast<size_t>(D), 0.0f));
  std::vector<float> A_tot(static_cast<size_t>(G), 0.0f);
  std::vector<float> Q_tot(static_cast<size_t>(G), 0.0f);

  for (const auto& s : items) {
    const size_t g = static_cast<size_t>(s.group_id);
    for (int d = 0; d < D; ++d) {
      S_phi[g][static_cast<size_t>(d)] += s.a * s.phi[static_cast<size_t>(d)];
    }
    A_tot[g] += s.a;
    Q_tot[g] += std::sqrt(s.a);
  }

  std::vector<std::vector<float>> out(items.size(), std::vector<float>(static_cast<size_t>(D), 0.0f));
  for (size_t i = 0; i < items.size(); ++i) {
    const auto& s = items[i];
    const size_t g = static_cast<size_t>(s.group_id);
    const float A_p = std::max(A_tot[g] - s.a, 0.0f);
    const float Q_p = std::max(Q_tot[g] - std::sqrt(s.a), 0.0f);
    float gamma = 0.0f;
    if (A_p > 1e-12f) {
      const float n_eff = (Q_p * Q_p) / A_p;
      const float A_eff = A_p / (1.0f + std::max(n_eff - 1.0f, 0.0f) * rho);
      gamma = A_eff / (A_eff + kappa);
    }
    for (int d = 0; d < D; ++d) {
      const float s_loo = S_phi[g][static_cast<size_t>(d)] - s.a * s.phi[static_cast<size_t>(d)];
      const float mu_loo = (A_p > 1e-12f) ? (s_loo / A_p) : 0.0f;
      out[i][static_cast<size_t>(d)] = s.h[static_cast<size_t>(d)] + gamma * mu_loo;
    }
  }
  return out;
}

std::vector<std::vector<float>> async_fuse_sharded(const std::vector<SeriesItem>& items,
                                                   const std::vector<int>& shard_of,
                                                   int W,
                                                   int G,
                                                   int D,
                                                   float rho,
                                                   float kappa) {
  // Each worker builds its local FP32 additive monoid M_{w,g} = [a*phi, a, sqrt(a)].
  std::vector<std::vector<std::vector<float>>> local_S(
      static_cast<size_t>(W),
      std::vector<std::vector<float>>(static_cast<size_t>(G),
                                      std::vector<float>(static_cast<size_t>(D), 0.0f)));
  std::vector<std::vector<float>> local_A(static_cast<size_t>(W),
                                          std::vector<float>(static_cast<size_t>(G), 0.0f));
  std::vector<std::vector<float>> local_Q(static_cast<size_t>(W),
                                          std::vector<float>(static_cast<size_t>(G), 0.0f));

  for (size_t i = 0; i < items.size(); ++i) {
    const size_t w = static_cast<size_t>(shard_of[i]);
    const size_t g = static_cast<size_t>(items[i].group_id);
    for (int d = 0; d < D; ++d) {
      local_S[w][g][static_cast<size_t>(d)] += items[i].a * items[i].phi[static_cast<size_t>(d)];
    }
    local_A[w][g] += items[i].a;
    local_Q[w][g] += std::sqrt(items[i].a);
  }

  // Simulated FP32 AllReduce(SUM) across W workers.
  std::vector<std::vector<float>> S_phi(static_cast<size_t>(G), std::vector<float>(static_cast<size_t>(D), 0.0f));
  std::vector<float> A_tot(static_cast<size_t>(G), 0.0f);
  std::vector<float> Q_tot(static_cast<size_t>(G), 0.0f);
  for (int w = 0; w < W; ++w) {
    for (int g = 0; g < G; ++g) {
      for (int d = 0; d < D; ++d) {
        S_phi[static_cast<size_t>(g)][static_cast<size_t>(d)] +=
            local_S[static_cast<size_t>(w)][static_cast<size_t>(g)][static_cast<size_t>(d)];
      }
      A_tot[static_cast<size_t>(g)] += local_A[static_cast<size_t>(w)][static_cast<size_t>(g)];
      Q_tot[static_cast<size_t>(g)] += local_Q[static_cast<size_t>(w)][static_cast<size_t>(g)];
    }
  }

  std::vector<std::vector<float>> out(items.size(), std::vector<float>(static_cast<size_t>(D), 0.0f));
  for (size_t i = 0; i < items.size(); ++i) {
    const auto& s = items[i];
    const size_t g = static_cast<size_t>(s.group_id);
    const float A_p = std::max(A_tot[g] - s.a, 0.0f);
    const float Q_p = std::max(Q_tot[g] - std::sqrt(s.a), 0.0f);
    float gamma = 0.0f;
    if (A_p > 1e-12f) {
      const float n_eff = (Q_p * Q_p) / A_p;
      const float A_eff = A_p / (1.0f + std::max(n_eff - 1.0f, 0.0f) * rho);
      gamma = A_eff / (A_eff + kappa);
    }
    for (int d = 0; d < D; ++d) {
      const float s_loo = S_phi[g][static_cast<size_t>(d)] - s.a * s.phi[static_cast<size_t>(d)];
      const float mu_loo = (A_p > 1e-12f) ? (s_loo / A_p) : 0.0f;
      out[i][static_cast<size_t>(d)] = s.h[static_cast<size_t>(d)] + gamma * mu_loo;
    }
  }
  return out;
}

void verify_contract_4_async_fuse() {
  std::mt19937_64 rng(774422ULL);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::uniform_real_distribution<float> ud(0.1f, 5.0f);

  const int D = 32;
  const float rho = 0.25f;
  const float kappa = 1.5f;

  // (4a) W=4 skewed partition [1, 3, 12, 48] = 64 series across G=6 groups (including a singleton group).
  {
    const int N = 64;
    const int G = 6;
    std::vector<SeriesItem> items(static_cast<size_t>(N));
    std::vector<int> shard_of(static_cast<size_t>(N));
    const std::array<int, 4> counts = {1, 3, 12, 48};
    int cur = 0;
    for (int w = 0; w < 4; ++w) {
      for (int c = 0; c < counts[static_cast<size_t>(w)]; ++c) {
        shard_of[static_cast<size_t>(cur++)] = w;
      }
    }
    for (int i = 0; i < N; ++i) {
      items[static_cast<size_t>(i)].group_id = (i == 0) ? 0 : (1 + (i % (G - 1)));
      items[static_cast<size_t>(i)].a = ud(rng);
      items[static_cast<size_t>(i)].h.resize(static_cast<size_t>(D));
      items[static_cast<size_t>(i)].phi.resize(static_cast<size_t>(D));
      for (int d = 0; d < D; ++d) {
        items[static_cast<size_t>(i)].h[static_cast<size_t>(d)] = nd(rng);
        items[static_cast<size_t>(i)].phi[static_cast<size_t>(d)] = nd(rng);
      }
    }
    auto central = async_fuse_central(items, G, D, rho, kappa);
    auto sharded = async_fuse_sharded(items, shard_of, 4, G, D, rho, kappa);
    float max_diff = 0.0f;
    for (int i = 0; i < N; ++i) {
      for (int d = 0; d < D; ++d) {
        max_diff = std::max(max_diff, std::fabs(central[i][d] - sharded[i][d]));
      }
    }
    check_cond(max_diff <= 1e-6f, "AsyncFuse W=4 skewed partition sharded-vs-central max_diff <= 1e-6");
    // Singleton group (series 0 in group 0) must equal h_0 exactly.
    check_cond(sharded[0] == items[0].h, "AsyncFuse singleton group yields exact gamma=0 identity");
  }

  // (4b) W=8 random partition on 256 series across G=12 groups.
  {
    const int N = 256;
    const int G = 12;
    std::uniform_int_distribution<int> wdist(0, 7);
    std::uniform_int_distribution<int> gdist(0, G - 1);
    std::vector<SeriesItem> items(static_cast<size_t>(N));
    std::vector<int> shard_of(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) {
      shard_of[static_cast<size_t>(i)] = wdist(rng);
      items[static_cast<size_t>(i)].group_id = gdist(rng);
      items[static_cast<size_t>(i)].a = ud(rng);
      items[static_cast<size_t>(i)].h.resize(static_cast<size_t>(D));
      items[static_cast<size_t>(i)].phi.resize(static_cast<size_t>(D));
      for (int d = 0; d < D; ++d) {
        items[static_cast<size_t>(i)].h[static_cast<size_t>(d)] = nd(rng);
        items[static_cast<size_t>(i)].phi[static_cast<size_t>(d)] = nd(rng);
      }
    }
    auto central = async_fuse_central(items, G, D, rho, kappa);
    auto sharded = async_fuse_sharded(items, shard_of, 8, G, D, rho, kappa);
    float max_diff = 0.0f;
    for (int i = 0; i < N; ++i) {
      for (int d = 0; d < D; ++d) {
        max_diff = std::max(max_diff, std::fabs(central[i][d] - sharded[i][d]));
      }
    }
    check_cond(max_diff <= 1e-6f, "AsyncFuse W=8 random partition sharded-vs-central max_diff <= 1e-6");
  }

  std::printf("[PASS] Contract 4: AsyncFuse monoid + T3 LOO parity (W=4 skewed & W=8 random <= 1e-6)\n");
}

}  // namespace

int main() {
  verify_contract_1_sort9();
  verify_contract_2_ipcg();
  verify_contract_3_scaled_fp16();
  verify_contract_4_async_fuse();

  if (g_failures != 0) {
    std::fprintf(stderr, "CPU parity harness FAILED with %d failure(s).\n", g_failures);
    return 1;
  }
  std::printf("ALL 4 CPU PARITY CONTRACTS PASSED (0 failures).\n");
  return 0;
}
