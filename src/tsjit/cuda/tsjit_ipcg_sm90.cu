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

// Hardened Persistent SMEM-Resident Batched Exact Online Ridge (ADAPT-IPCG)
// CUDA Kernel for Hopper SM90+ (and SM80+/SM89+ fallback).
//
// Recommended compilation flags:
//   nvcc -O3 -arch=sm_90a --prec-div=true --prec-sqrt=true -shared -Xcompiler -fPIC

#include <cuda_runtime.h>
#ifdef TSJIT_HAS_CUBLAS
#include <cublas_v2.h>
#endif
#include <cmath>
#include <cstdint>
#include <limits>

namespace tsjit {
namespace cuda {

static constexpr int kBlockThreads = 128;  // 4 warps per block
static constexpr int kNumWarps = kBlockThreads / 32;
static constexpr int kMinBlocksPerSm = 8;
static constexpr int kMaxK = 256;

template <typename T>
struct VecTraits;

template <>
struct VecTraits<float> {
  using VecType = float4;
  static constexpr int kWidth = 4;
  static constexpr float kMinSafeScale = 1e-30f;
  static constexpr float kMaxSafeScale = 1e30f;
  static constexpr float kMaxInitRatio = 1e12f;
};

template <>
struct VecTraits<double> {
  using VecType = double2;
  static constexpr int kWidth = 2;
  static constexpr double kMinSafeScale = 1e-200;
  static constexpr double kMaxSafeScale = 1e200;
  static constexpr double kMaxInitRatio = 1e100;
};

__device__ __forceinline__ float abs_val(float x) {
  return fabsf(x);
}

__device__ __forceinline__ double abs_val(double x) {
  return fabs(x);
}

__device__ __forceinline__ float fma_val(float a, float b, float c) {
  return fmaf(a, b, c);
}

__device__ __forceinline__ double fma_val(double a, double b, double c) {
  return fma(a, b, c);
}

__device__ __forceinline__ bool isfinite_val(float x) {
  return ::isfinite(x);
}

__device__ __forceinline__ bool isfinite_val(double x) {
  return ::isfinite(x);
}

__device__ __forceinline__ float quiet_nan_val(float) {
  return __int_as_float(0x7fc00000);
}

__device__ __forceinline__ double quiet_nan_val(double) {
  return __longlong_as_double(0x7ff8000000000000ULL);
}

__device__ __forceinline__ float vec_dot_fma(float4 a, float4 b, float acc) {
  acc = fmaf(a.x, b.x, acc);
  acc = fmaf(a.y, b.y, acc);
  acc = fmaf(a.z, b.z, acc);
  acc = fmaf(a.w, b.w, acc);
  return acc;
}

__device__ __forceinline__ double vec_dot_fma(double2 a, double2 b, double acc) {
  acc = fma(a.x, b.x, acc);
  acc = fma(a.y, b.y, acc);
  return acc;
}

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
  val += __shfl_down_sync(0xffffffffu, val, 16);
  val += __shfl_down_sync(0xffffffffu, val, 8);
  val += __shfl_down_sync(0xffffffffu, val, 4);
  val += __shfl_down_sync(0xffffffffu, val, 2);
  val += __shfl_down_sync(0xffffffffu, val, 1);
  return val;
}

// Warp-per-row 16-byte vectorized matrix-vector product: out_s = M * v_s.
// Rows of M have 16-byte aligned stride K_PAD, and base pointers are validated
// 16-byte aligned in the host launcher. Padding columns [K, K_PAD) are never
// read so uninitialized or NaN padding cannot contaminate the dot product.
template <typename T>
__device__ __forceinline__ void parallel_matvec(
    const T* __restrict__ M,
    const T* __restrict__ v_s,
    T* __restrict__ out_s,
    int K,
    int K_PAD) {
  using VecType = typename VecTraits<T>::VecType;
  constexpr int kWidth = VecTraits<T>::kWidth;

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int k_vec = K / kWidth;
  const int tail_start = k_vec * kWidth;

  for (int row = warp_id; row < K; row += kNumWarps) {
    const T* row_ptr = M + static_cast<size_t>(row) * static_cast<size_t>(K_PAD);
    const VecType* row_vec = reinterpret_cast<const VecType*>(row_ptr);
    const VecType* v_vec = reinterpret_cast<const VecType*>(v_s);

    T acc = T(0);
    for (int idx = lane; idx < k_vec; idx += 32) {
      acc = vec_dot_fma(row_vec[idx], v_vec[idx], acc);
    }
    const int tail_col = tail_start + lane;
    if (tail_col < K) {
      acc = fma_val(row_ptr[tail_col], v_s[tail_col], acc);
    }
    acc = warp_reduce_sum<T>(acc);
    if (lane == 0) {
      out_s[row] = acc;
    }
  }
  __syncthreads();
}

// Apply SPD preconditioner: z_s = M0 * r_s (dense KxK) or Jacobi diag(A)^{-1} * r_s
// when M0 is nullptr.
template <typename T>
__device__ __forceinline__ void apply_precond(
    const T* __restrict__ M0,
    const T* __restrict__ A,
    const T* __restrict__ r_s,
    T* __restrict__ z_s,
    int K,
    int K_PAD) {
  if (M0 != nullptr) {
    parallel_matvec<T>(M0, r_s, z_s, K, K_PAD);
    return;
  }
  const int tid = static_cast<int>(threadIdx.x);
  for (int i = tid; i < K; i += kBlockThreads) {
    const T diag = A[static_cast<size_t>(i) * static_cast<size_t>(K_PAD) + i];
    if (diag > T(0) && isfinite_val(diag)) {
      z_s[i] = r_s[i] / diag;
    } else {
      z_s[i] = quiet_nan_val(T(0));
    }
  }
  __syncthreads();
}

template <typename T>
__device__ __forceinline__ T max_nan_prop(T a, T b) {
  if (!isfinite_val(a) || !isfinite_val(b)) {
    return quiet_nan_val(T(0));
  }
  return (a > b) ? a : b;
}

template <typename T>
__device__ __forceinline__ T warp_reduce_max_nan_prop(T val) {
  val = max_nan_prop<T>(val, __shfl_down_sync(0xffffffffu, val, 16));
  val = max_nan_prop<T>(val, __shfl_down_sync(0xffffffffu, val, 8));
  val = max_nan_prop<T>(val, __shfl_down_sync(0xffffffffu, val, 4));
  val = max_nan_prop<T>(val, __shfl_down_sync(0xffffffffu, val, 2));
  val = max_nan_prop<T>(val, __shfl_down_sync(0xffffffffu, val, 1));
  return val;
}

// Deterministic block-wide NaN/Inf-propagating infinity-norm: returns max_i |v[i]|
// if every element is finite, else quiet NaN.
template <typename T>
__device__ __forceinline__ T parallel_abs_max_finite(
    const T* __restrict__ v,
    int K,
    T* __restrict__ smem_red) {
  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid >> 5;
  const int lane = tid & 31;

  T acc = T(0);
  for (int i = tid; i < K; i += kBlockThreads) {
    const T val = v[i];
    if (!isfinite_val(val)) {
      acc = quiet_nan_val(T(0));
    } else if (isfinite_val(acc)) {
      const T av = abs_val(val);
      if (av > acc) {
        acc = av;
      }
    }
  }
  acc = warp_reduce_max_nan_prop<T>(acc);
  if (lane == 0) {
    smem_red[warp_id] = acc;
  }
  __syncthreads();
  const T total = max_nan_prop<T>(
      max_nan_prop<T>(smem_red[0], smem_red[1]),
      max_nan_prop<T>(smem_red[2], smem_red[3]));
  __syncthreads();
  return total;
}

// Deterministic block-wide prescaled squared L2 norm: sum_i (v[i] * scale)^2.
template <typename T>
__device__ __forceinline__ T parallel_scaled_norm2(
    const T* __restrict__ v,
    T scale,
    int K,
    T* __restrict__ smem_red) {
  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid >> 5;
  const int lane = tid & 31;

  T acc = T(0);
  for (int i = tid; i < K; i += kBlockThreads) {
    const T sv = v[i] * scale;
    acc = fma_val(sv, sv, acc);
  }
  acc = warp_reduce_sum<T>(acc);
  if (lane == 0) {
    smem_red[warp_id] = acc;
  }
  __syncthreads();
  const T total = (smem_red[0] + smem_red[1]) + (smem_red[2] + smem_red[3]);
  __syncthreads();
  return total;
}

// Deterministic block-wide inner product across 4 warps.
template <typename T>
__device__ __forceinline__ T parallel_dot(
    const T* __restrict__ a,
    const T* __restrict__ b,
    int K,
    T* __restrict__ smem_red) {
  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid >> 5;
  const int lane = tid & 31;

  T acc = T(0);
  for (int i = tid; i < K; i += kBlockThreads) {
    acc = fma_val(a[i], b[i], acc);
  }
  acc = warp_reduce_sum<T>(acc);
  if (lane == 0) {
    smem_red[warp_id] = acc;
  }
  __syncthreads();
  const T total = (smem_red[0] + smem_red[1]) + (smem_red[2] + smem_red[3]);
  __syncthreads();
  return total;
}

// Persistent SMEM-resident batched ADAPT-IPCG kernel.
// Shared memory layout (5 * K_PAD + 4 elements of T):
//   x_s[K_PAD], r_s[K_PAD], z_s[K_PAD], p_s[K_PAD], ap_s[K_PAD], red[4].
// For K=136: FP32 uses ~2.7 KB SMEM; FP64 uses ~5.5 KB SMEM.
template <typename T>
__global__ __launch_bounds__(kBlockThreads, kMinBlocksPerSm) void tsjit_ipcg_kernel(
    const T* __restrict__ A,
    const T* __restrict__ M0,
    const T* __restrict__ rhs,
    const T* __restrict__ x_init,
    T* __restrict__ x_out,
    uint8_t* __restrict__ conv_out,
    int32_t* __restrict__ iters_out,
    T* __restrict__ res_out,
    uint32_t N,
    int32_t K,
    int32_t K_PAD,
    T tol,
    int32_t max_iters) {
  extern __shared__ uint4 smem_raw[];
  T* smem = reinterpret_cast<T*>(smem_raw);
  T* x_s = smem;
  T* r_s = x_s + K_PAD;
  T* z_s = r_s + K_PAD;
  T* p_s = z_s + K_PAD;
  T* ap_s = p_s + K_PAD;
  T* smem_red = ap_s + K_PAD;

  const int tid = static_cast<int>(threadIdx.x);
  const size_t mat_stride = static_cast<size_t>(K) * static_cast<size_t>(K_PAD);
  const size_t vec_stride = static_cast<size_t>(K_PAD);
  const T tol2 = tol * tol;

  for (uint32_t n = blockIdx.x; n < N; n += gridDim.x) {
    const T* A_n = A + static_cast<size_t>(n) * mat_stride;
    const T* M0_n = (M0 != nullptr) ? (M0 + static_cast<size_t>(n) * mat_stride) : nullptr;
    const T* rhs_n = rhs + static_cast<size_t>(n) * vec_stride;
    const T* x_in_n = x_init + static_cast<size_t>(n) * vec_stride;
    T* x_out_n = x_out + static_cast<size_t>(n) * vec_stride;

    // Explicit element-wise finiteness + infinity-norm check on rhs and x_init.
    // Prescale the linear system A * (x * inv_rhs_scale) = rhs * inv_rhs_scale so
    // FP32 rhs >= 1e20 or <= 1e-20 never overflows or underflows rhs_n2 / target_r2.
    const T rhs_max = parallel_abs_max_finite<T>(rhs_n, K, smem_red);
    const T x_init_max = parallel_abs_max_finite<T>(x_in_n, K, smem_red);

    T rhs_scale = T(1);
    T inv_rhs_scale = T(1);
    const T ref_mag = (isfinite_val(rhs_max) && rhs_max > T(0))
                          ? rhs_max
                          : ((isfinite_val(x_init_max) && x_init_max > T(0)) ? x_init_max : T(1));
    if (isfinite_val(ref_mag) && ref_mag > T(0)) {
      rhs_scale = ref_mag;
      if (rhs_scale < VecTraits<T>::kMinSafeScale) {
        rhs_scale = VecTraits<T>::kMinSafeScale;
      } else if (rhs_scale > VecTraits<T>::kMaxSafeScale) {
        rhs_scale = VecTraits<T>::kMaxSafeScale;
      }
      inv_rhs_scale = T(1) / rhs_scale;
    }

    const T x_init_scaled_max = x_init_max * inv_rhs_scale;
    const bool x_init_scales_ok = isfinite_val(x_init_max) && isfinite_val(x_init_scaled_max) &&
                                  (x_init_scaled_max <= VecTraits<T>::kMaxInitRatio);

    // Load prescaled warm-start x_init into SMEM and zero padding lanes.
    for (int i = tid; i < K_PAD; i += kBlockThreads) {
      x_s[i] = (i < K && x_init_scales_ok) ? (x_in_n[i] * inv_rhs_scale) : T(0);
      r_s[i] = T(0);
      z_s[i] = T(0);
      p_s[i] = T(0);
      ap_s[i] = T(0);
    }
    __syncthreads();

    const T rhs_n2_scaled = parallel_scaled_norm2<T>(rhs_n, inv_rhs_scale, K, smem_red);

    bool valid_inputs = isfinite_val(rhs_max) && isfinite_val(x_init_max) &&
                        isfinite_val(rhs_n2_scaled) && isfinite_val(tol2) && (tol >= T(0));
    T target_r2 = (rhs_n2_scaled > T(0)) ? (tol2 * rhs_n2_scaled) : T(0);
    if (!isfinite_val(target_r2)) {
      valid_inputs = false;
    }

    bool verified_true_r2 = false;
    int32_t iter_done = 0;
    T true_r2_scaled = quiet_nan_val(T(0));

    if (valid_inputs) {
      // Initial residual in prescaled coordinates: r = rhs * inv_rhs_scale - A * x_0.
      parallel_matvec<T>(A_n, x_s, ap_s, K, K_PAD);
      for (int i = tid; i < K; i += kBlockThreads) {
        r_s[i] = (rhs_n[i] * inv_rhs_scale) - ap_s[i];
      }
      __syncthreads();

      T r_n2 = parallel_dot<T>(r_s, r_s, K, smem_red);
      if (!isfinite_val(r_n2) && x_init_scales_ok && (x_init_max > T(0))) {
        // Stale warm-start overflowed A*x_s or ||r_s||^2; retry cleanly from x_s = 0.
        // Re-running parallel_matvec(A_n, 0) preserves IEEE 0*NaN/Inf = NaN rejection.
        for (int i = tid; i < K; i += kBlockThreads) {
          x_s[i] = T(0);
        }
        __syncthreads();
        parallel_matvec<T>(A_n, x_s, ap_s, K, K_PAD);
        for (int i = tid; i < K; i += kBlockThreads) {
          r_s[i] = (rhs_n[i] * inv_rhs_scale) - ap_s[i];
        }
        __syncthreads();
        r_n2 = parallel_dot<T>(r_s, r_s, K, smem_red);
      }
      if (isfinite_val(r_n2)) {
        if (r_n2 <= target_r2) {
          true_r2_scaled = r_n2;
          verified_true_r2 = true;
        } else if (max_iters > 0) {
          apply_precond<T>(M0_n, A_n, r_s, z_s, K, K_PAD);
          for (int i = tid; i < K; i += kBlockThreads) {
            p_s[i] = z_s[i];
          }
          __syncthreads();

          T rz = parallel_dot<T>(r_s, z_s, K, smem_red);
          int restarts = 0;

          if (rz > T(0) && isfinite_val(rz)) {
            for (int32_t it = 1; it <= max_iters; ++it) {
              parallel_matvec<T>(A_n, p_s, ap_s, K, K_PAD);
              const T pap = parallel_dot<T>(p_s, ap_s, K, smem_red);
              if (!(pap > T(0) && isfinite_val(pap))) {
                iter_done = it;
                break;
              }
              const T alpha = rz / pap;
              if (!isfinite_val(alpha)) {
                iter_done = it;
                break;
              }

              for (int i = tid; i < K; i += kBlockThreads) {
                x_s[i] = fma_val(alpha, p_s[i], x_s[i]);
                r_s[i] = fma_val(-alpha, ap_s[i], r_s[i]);
              }
              __syncthreads();

              r_n2 = parallel_dot<T>(r_s, r_s, K, smem_red);
              iter_done = it;
              if (!isfinite_val(r_n2)) {
                break;
              }

              if (r_n2 <= target_r2) {
                // Compute fresh true residual to guard against recursive drift.
                parallel_matvec<T>(A_n, x_s, ap_s, K, K_PAD);
                for (int i = tid; i < K; i += kBlockThreads) {
                  r_s[i] = (rhs_n[i] * inv_rhs_scale) - ap_s[i];
                }
                __syncthreads();
                const T check_r2 = parallel_dot<T>(r_s, r_s, K, smem_red);
                if (isfinite_val(check_r2) && check_r2 <= target_r2) {
                  true_r2_scaled = check_r2;
                  verified_true_r2 = true;
                  break;
                }
                // Allow one clean residual replacement restart if budget remains.
                if (!isfinite_val(check_r2) || restarts >= 1 || it == max_iters) {
                  true_r2_scaled = check_r2;
                  verified_true_r2 = true;
                  break;
                }
                restarts += 1;
                apply_precond<T>(M0_n, A_n, r_s, z_s, K, K_PAD);
                for (int i = tid; i < K; i += kBlockThreads) {
                  p_s[i] = z_s[i];
                }
                __syncthreads();
                rz = parallel_dot<T>(r_s, z_s, K, smem_red);
                if (!(rz > T(0) && isfinite_val(rz))) {
                  break;
                }
                continue;
              }

              if (it == max_iters) {
                break;
              }

              apply_precond<T>(M0_n, A_n, r_s, z_s, K, K_PAD);
              const T rz_new = parallel_dot<T>(r_s, z_s, K, smem_red);
              if (!(rz_new > T(0) && isfinite_val(rz_new))) {
                break;
              }
              const T beta = rz_new / rz;
              if (!isfinite_val(beta)) {
                break;
              }
              for (int i = tid; i < K; i += kBlockThreads) {
                p_s[i] = fma_val(beta, p_s[i], z_s[i]);
              }
              __syncthreads();
              rz = rz_new;
            }
          }
        }
      }

      // Final True-Residual Verification: whenever not already freshly computed
      // on the final x_s, compute exact parallel_matvec(A_n, x_s, ap_s) and
      // verify ||rhs_scaled - A*x_scaled||_2^2 <= target_r2 && isfinite(true_r2_scaled).
      if (!verified_true_r2) {
        parallel_matvec<T>(A_n, x_s, ap_s, K, K_PAD);
        for (int i = tid; i < K; i += kBlockThreads) {
          r_s[i] = (rhs_n[i] * inv_rhs_scale) - ap_s[i];
        }
        __syncthreads();
        true_r2_scaled = parallel_dot<T>(r_s, r_s, K, smem_red);
      }
    }

    // Unscale x_s *= rhs_scale and check element-wise finiteness (no squaring overflow).
    for (int i = tid; i < K; i += kBlockThreads) {
      x_s[i] = x_s[i] * rhs_scale;
    }
    __syncthreads();
    const T x_final_max = parallel_abs_max_finite<T>(x_s, K, smem_red);
    const bool converged = valid_inputs &&
                           isfinite_val(true_r2_scaled) && isfinite_val(x_final_max) &&
                           (true_r2_scaled <= target_r2);
    const T true_r2_out =
        isfinite_val(true_r2_scaled) ? ((true_r2_scaled * rhs_scale) * rhs_scale) : true_r2_scaled;

    // Non-converged systems leave theta untouched (and if out-of-place, preserve x_init).
    if (converged) {
      for (int i = tid; i < K; i += kBlockThreads) {
        x_out_n[i] = x_s[i];
      }
    } else if (x_out_n != x_in_n) {
      for (int i = tid; i < K; i += kBlockThreads) {
        x_out_n[i] = x_in_n[i];
      }
    }

    if (tid == 0) {
      if (conv_out != nullptr) {
        conv_out[n] = converged ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0);
      }
      if (iters_out != nullptr) {
        iters_out[n] = iter_done;
      }
      if (res_out != nullptr) {
        res_out[n] = true_r2_out;
      }
    }
    __syncthreads();
  }
}

// Companion exact int64 Gram accumulator directly from quantized int32 features:
//   G_i64[n, i, j] += sum_{b=0}^{B_bucket-1} int64(X_q[n, b, i]) * int64(X_q[n, b, j])
//   b_i64[n, i]    += sum_{b=0}^{B_bucket-1} int64(X_q[n, b, i]) * int64(y_q[n, b])
__global__ void tsjit_gram_accum_i64_kernel(
    const int32_t* __restrict__ X_q,
    const int32_t* __restrict__ y_q,
    int64_t* __restrict__ G_i64,
    int64_t* __restrict__ b_i64,
    uint32_t N,
    int32_t B_bucket,
    int32_t K,
    int32_t K_PAD) {
  const uint32_t n = blockIdx.x;
  if (n >= N) {
    return;
  }
  const int32_t idx = static_cast<int32_t>(blockIdx.y * blockDim.x + threadIdx.x);
  const int32_t total_g = K * K;
  const int32_t* X_n = X_q + static_cast<size_t>(n) * static_cast<size_t>(B_bucket) * static_cast<size_t>(K);
  const int32_t* y_n = y_q + static_cast<size_t>(n) * static_cast<size_t>(B_bucket);

  if (idx < total_g) {
    const int32_t row = idx / K;
    const int32_t col = idx - row * K;
    int64_t acc = 0;
    for (int32_t b = 0; b < B_bucket; ++b) {
      const int64_t xv_r = static_cast<int64_t>(X_n[static_cast<size_t>(b) * K + row]);
      const int64_t xv_c = static_cast<int64_t>(X_n[static_cast<size_t>(b) * K + col]);
      acc += xv_r * xv_c;
    }
    G_i64[static_cast<size_t>(n) * static_cast<size_t>(K) * static_cast<size_t>(K_PAD) +
          static_cast<size_t>(row) * static_cast<size_t>(K_PAD) + col] += acc;
  } else if (idx < total_g + K) {
    const int32_t row = idx - total_g;
    int64_t acc = 0;
    for (int32_t b = 0; b < B_bucket; ++b) {
      const int64_t xv_r = static_cast<int64_t>(X_n[static_cast<size_t>(b) * K + row]);
      const int64_t yv = static_cast<int64_t>(y_n[b]);
      acc += xv_r * yv;
    }
    b_i64[static_cast<size_t>(n) * static_cast<size_t>(K_PAD) + row] += acc;
  }
}

// Companion exact FP64-TC -> int64 Gram accumulator (used when B_bucket * max_abs_q^2 < 2^53).
__global__ void tsjit_gram_accum_f64_to_i64_kernel(
    const double* __restrict__ G_delta_f64,
    const double* __restrict__ b_delta_f64,
    int64_t* __restrict__ G_i64,
    int64_t* __restrict__ b_i64,
    uint32_t N,
    int32_t K,
    int32_t K_PAD) {
  const uint32_t n = blockIdx.x;
  if (n >= N) {
    return;
  }
  const int32_t idx = static_cast<int32_t>(blockIdx.y * blockDim.x + threadIdx.x);
  const int32_t total_g = K * K;
  if (idx < total_g) {
    const int32_t row = idx / K;
    const int32_t col = idx - row * K;
    const size_t src_off = static_cast<size_t>(n) * static_cast<size_t>(total_g) + static_cast<size_t>(idx);
    const size_t dst_off = static_cast<size_t>(n) * static_cast<size_t>(K) * static_cast<size_t>(K_PAD) +
                           static_cast<size_t>(row) * static_cast<size_t>(K_PAD) + col;
    G_i64[dst_off] += static_cast<int64_t>(::llround(G_delta_f64[src_off]));
  } else if (idx < total_g + K) {
    const int32_t row = idx - total_g;
    const size_t src_off = static_cast<size_t>(n) * static_cast<size_t>(K) + static_cast<size_t>(row);
    const size_t dst_off = static_cast<size_t>(n) * static_cast<size_t>(K_PAD) + static_cast<size_t>(row);
    b_i64[dst_off] += static_cast<int64_t>(::llround(b_delta_f64[src_off]));
  }
}

// Finalize dequantized ridge system A = inv_s2 * G_i64 + ridge_lambda * I, rhs = inv_sxy * b_i64.
__global__ void tsjit_gram_finalize_ridge_f64_kernel(
    const int64_t* __restrict__ G_i64,
    const int64_t* __restrict__ b_i64,
    double inv_s2,
    double inv_sxy,
    double ridge_lambda,
    double* __restrict__ A_out,
    double* __restrict__ rhs_out,
    uint32_t N,
    int32_t K,
    int32_t K_PAD) {
  const uint32_t n = blockIdx.x;
  if (n >= N) {
    return;
  }
  const int32_t idx = static_cast<int32_t>(blockIdx.y * blockDim.x + threadIdx.x);
  const int32_t total_g = K * K_PAD;
  if (idx < total_g) {
    const int32_t row = idx / K_PAD;
    const int32_t col = idx - row * K_PAD;
    const size_t off = static_cast<size_t>(n) * static_cast<size_t>(total_g) + static_cast<size_t>(idx);
    if (col < K) {
      double val = static_cast<double>(G_i64[off]) * inv_s2;
      if (row == col) {
        val += ridge_lambda;
      }
      A_out[off] = val;
    } else {
      A_out[off] = 0.0;
    }
  } else if (idx < total_g + K_PAD) {
    const int32_t row = idx - total_g;
    const size_t off = static_cast<size_t>(n) * static_cast<size_t>(K_PAD) + static_cast<size_t>(row);
    rhs_out[off] = (row < K) ? (static_cast<double>(b_i64[off]) * inv_sxy) : 0.0;
  }
}

// Batched ridge prediction kernel: y_pred[n] = dot(x_query[n], theta[n]).
template <typename T>
__global__ void tsjit_ridge_predict_kernel(
    const T* __restrict__ x_query,
    const T* __restrict__ theta,
    T* __restrict__ y_pred,
    uint32_t N,
    int32_t K,
    int32_t K_PAD) {
  const uint32_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) {
    return;
  }
  const T* xq = x_query + static_cast<size_t>(n) * static_cast<size_t>(K_PAD);
  const T* th = theta + static_cast<size_t>(n) * static_cast<size_t>(K_PAD);
  T acc = T(0);
  for (int32_t i = 0; i < K; ++i) {
    acc = fma_val(xq[i], th[i], acc);
  }
  y_pred[n] = acc;
}

// Scale output buffer C[0..total) *= beta for K == 0 with arbitrary beta.
__global__ void tsjit_scale_f32_kernel(float* __restrict__ C, size_t total, float beta) {
  const size_t stride = static_cast<size_t>(gridDim.x) * static_cast<size_t>(blockDim.x);
  for (size_t idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) +
                    static_cast<size_t>(threadIdx.x);
       idx < total;
       idx += stride) {
    C[idx] *= beta;
  }
}

template <typename T>
inline bool is_aligned_16(const T* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & static_cast<uintptr_t>(15)) == 0;
}

template <typename T>
static int launch_ipcg_impl(
    const T* A,
    const T* M0,
    const T* rhs,
    const T* x_init,
    T* x_out,
    uint8_t* conv_out,
    int32_t* iters_out,
    T* res_out,
    uint32_t N,
    int32_t K,
    int32_t K_PAD,
    T tol,
    int32_t max_iters,
    cudaStream_t stream) {
  constexpr int kWidth = VecTraits<T>::kWidth;

  if (N == 0) {
    return 0;
  }
  if (K < 1 || K > kMaxK || K_PAD < K || (K_PAD % kWidth) != 0 || max_iters < 0) {
    return -1;
  }
  if (A == nullptr || rhs == nullptr || x_init == nullptr || x_out == nullptr) {
    return -2;
  }
  if (!is_aligned_16(A) || !is_aligned_16(rhs) || !is_aligned_16(x_init) || !is_aligned_16(x_out)) {
    return -2;
  }
  if (M0 != nullptr && !is_aligned_16(M0)) {
    return -2;
  }

  int dev = 0;
  int num_sms = 1;
  if (cudaGetDevice(&dev) != cudaSuccess ||
      cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess) {
    return -3;
  }

  const uint32_t max_grid = static_cast<uint32_t>(num_sms) * 16u;
  const uint32_t grid = (N < max_grid) ? N : max_grid;
  const size_t raw_smem = static_cast<size_t>(5 * K_PAD + kNumWarps) * sizeof(T);
  const size_t smem_bytes = (raw_smem + 15u) & ~static_cast<size_t>(15u);

  tsjit_ipcg_kernel<T><<<grid, kBlockThreads, smem_bytes, stream>>>(
      A, M0, rhs, x_init, x_out, conv_out, iters_out, res_out, N, K, K_PAD, tol, max_iters);

  const cudaError_t err = cudaGetLastError();
  return (err == cudaSuccess) ? 0 : -4;
}

#ifdef TSJIT_HAS_CUBLAS

static constexpr int kMaxCublasDevices = 16;
static constexpr size_t kCublasWorkspaceBytes = 32u * 1024u * 1024u;  // 32 MiB Hopper WGMMA ws

struct ThreadLocalCublasCache {
  cublasHandle_t handles[kMaxCublasDevices]{};
  void* workspaces[kMaxCublasDevices]{};
  bool initialized[kMaxCublasDevices]{};

  ~ThreadLocalCublasCache() {
    // Intentionally leak CUDA/cuBLAS handles on process teardown to avoid
    // calling into an already-unloaded CUDA driver context during exit().
  }
};

static cublasHandle_t get_thread_local_cublas_handle(int dev, cudaStream_t stream) {
  static thread_local ThreadLocalCublasCache cache;
  if (dev < 0 || dev >= kMaxCublasDevices) {
    return nullptr;
  }
  if (!cache.initialized[dev]) {
    cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &cap_status) != cudaSuccess ||
        cap_status != cudaStreamCaptureStatusNone) {
      // Never call cublasCreate / cudaMalloc inside (or implicit to) an active CUDA Graph capture.
      return nullptr;
    }
    cublasHandle_t h = nullptr;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) {
      return nullptr;
    }
    cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST);
    cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);
    void* ws = nullptr;
    if (cudaMalloc(&ws, kCublasWorkspaceBytes) == cudaSuccess && ws != nullptr) {
      if (cublasSetWorkspace(h, ws, kCublasWorkspaceBytes) == CUBLAS_STATUS_SUCCESS) {
        cache.workspaces[dev] = ws;
      } else {
        cudaFree(ws);
      }
    }
    cache.handles[dev] = h;
    cache.initialized[dev] = true;
  }
  return cache.handles[dev];
}

#endif  // TSJIT_HAS_CUBLAS

}  // namespace cuda
}  // namespace tsjit

extern "C" {

int tsjit_ipcg_solve_f64(
    const double* A,
    const double* M0,
    const double* rhs,
    const double* x_init,
    double* x_out,
    uint8_t* conv_out,
    int32_t* iters_out,
    double* res_out,
    uint32_t N,
    int32_t K,
    int32_t K_PAD,
    double tol,
    int32_t max_iters,
    cudaStream_t stream) {
  return tsjit::cuda::launch_ipcg_impl<double>(
      A, M0, rhs, x_init, x_out, conv_out, iters_out, res_out, N, K, K_PAD, tol, max_iters, stream);
}

int tsjit_ipcg_solve_f32(
    const float* A,
    const float* M0,
    const float* rhs,
    const float* x_init,
    float* x_out,
    uint8_t* conv_out,
    int32_t* iters_out,
    float* res_out,
    uint32_t N,
    int32_t K,
    int32_t K_PAD,
    float tol,
    int32_t max_iters,
    cudaStream_t stream) {
  return tsjit::cuda::launch_ipcg_impl<float>(
      A, M0, rhs, x_init, x_out, conv_out, iters_out, res_out, N, K, K_PAD, tol, max_iters, stream);
}

int tsjit_gram_accum_i64(
    const int32_t* X_q,
    const int32_t* y_q,
    int64_t* G_i64,
    int64_t* b_i64,
    uint32_t N,
    int32_t B_bucket,
    int32_t K,
    int32_t K_PAD,
    cudaStream_t stream) {
  if (N == 0 || B_bucket == 0) {
    return 0;
  }
  if (K < 1 || K > tsjit::cuda::kMaxK || K_PAD < K || B_bucket < 0) {
    return -1;
  }
  if (X_q == nullptr || y_q == nullptr || G_i64 == nullptr || b_i64 == nullptr) {
    return -2;
  }
  const int32_t total = K * K + K;
  const int32_t threads = 256;
  const dim3 grid(N, static_cast<uint32_t>((total + threads - 1) / threads));
  tsjit::cuda::tsjit_gram_accum_i64_kernel<<<grid, threads, 0, stream>>>(
      X_q, y_q, G_i64, b_i64, N, B_bucket, K, K_PAD);
  return (cudaGetLastError() == cudaSuccess) ? 0 : -4;
}

int tsjit_gram_accum_f64_to_i64(
    const double* G_delta_f64,
    const double* b_delta_f64,
    int64_t* G_i64,
    int64_t* b_i64,
    uint32_t N,
    int32_t K,
    int32_t K_PAD,
    cudaStream_t stream) {
  if (N == 0) {
    return 0;
  }
  if (K < 1 || K > tsjit::cuda::kMaxK || K_PAD < K) {
    return -1;
  }
  if (G_delta_f64 == nullptr || b_delta_f64 == nullptr || G_i64 == nullptr || b_i64 == nullptr) {
    return -2;
  }
  const int32_t total = K * K + K;
  const int32_t threads = 256;
  const dim3 grid(N, static_cast<uint32_t>((total + threads - 1) / threads));
  tsjit::cuda::tsjit_gram_accum_f64_to_i64_kernel<<<grid, threads, 0, stream>>>(
      G_delta_f64, b_delta_f64, G_i64, b_i64, N, K, K_PAD);
  return (cudaGetLastError() == cudaSuccess) ? 0 : -4;
}

int tsjit_eval_ridge_online_f64(
    const int64_t* G_i64,
    const int64_t* b_i64,
    double inv_s2,
    double inv_sxy,
    double ridge_lambda,
    double* A_work,
    const double* M0,
    double* rhs_work,
    const double* x_init,
    double* theta_out,
    const double* x_query,
    double* y_pred_out,
    uint8_t* conv_out,
    int32_t* iters_out,
    double* res_out,
    uint32_t N,
    int32_t K,
    int32_t K_PAD,
    double tol,
    int32_t max_iters,
    cudaStream_t stream) {
  if (N == 0) {
    return 0;
  }
  if (K < 1 || K > tsjit::cuda::kMaxK || K_PAD < K || (K_PAD % 2) != 0) {
    return -1;
  }
  if (G_i64 == nullptr || b_i64 == nullptr || A_work == nullptr || rhs_work == nullptr ||
      x_init == nullptr || theta_out == nullptr) {
    return -2;
  }
  if (!tsjit::cuda::is_aligned_16(A_work) || !tsjit::cuda::is_aligned_16(rhs_work) ||
      !tsjit::cuda::is_aligned_16(x_init) || !tsjit::cuda::is_aligned_16(theta_out)) {
    return -2;
  }
  if (M0 != nullptr && !tsjit::cuda::is_aligned_16(M0)) {
    return -2;
  }
  if (x_query != nullptr && !tsjit::cuda::is_aligned_16(x_query)) {
    return -2;
  }

  const int32_t total_fin = K * K_PAD + K_PAD;
  const int32_t threads_fin = 256;
  const dim3 grid_fin(N, static_cast<uint32_t>((total_fin + threads_fin - 1) / threads_fin));
  tsjit::cuda::tsjit_gram_finalize_ridge_f64_kernel<<<grid_fin, threads_fin, 0, stream>>>(
      G_i64, b_i64, inv_s2, inv_sxy, ridge_lambda, A_work, rhs_work, N, K, K_PAD);
  if (cudaGetLastError() != cudaSuccess) {
    return -4;
  }

  const int solve_rc = tsjit::cuda::launch_ipcg_impl<double>(
      A_work, M0, rhs_work, x_init, theta_out, conv_out, iters_out, res_out,
      N, K, K_PAD, tol, max_iters, stream);
  if (solve_rc != 0) {
    return solve_rc;
  }

  if (x_query != nullptr && y_pred_out != nullptr) {
    const uint32_t pred_threads = 256u;
    const uint32_t pred_grid = (N + pred_threads - 1u) / pred_threads;
    tsjit::cuda::tsjit_ridge_predict_kernel<double><<<pred_grid, pred_threads, 0, stream>>>(
        x_query, theta_out, y_pred_out, N, K, K_PAD);
    if (cudaGetLastError() != cudaSuccess) {
      return -4;
    }
  }
  return 0;
}

int tsjit_cublas_available() {
#ifdef TSJIT_HAS_CUBLAS
  return 1;
#else
  return 0;
#endif
}

// Pre-warm thread-local per-device cuBLAS handle and 32 MiB WGMMA workspace outside
// any CUDA Graph capture so first-time graph capture never fails with -5 and broken
// runtime libcublas installations cleanly fall back to pure-Triton.
int tsjit_cublas_warmup(cudaStream_t stream) {
#ifdef TSJIT_HAS_CUBLAS
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) {
    return -3;
  }
  cublasHandle_t handle = tsjit::cuda::get_thread_local_cublas_handle(dev, stream);
  return (handle != nullptr) ? 0 : -5;
#else
  (void)stream;
  return -10;
#endif
}

// Single Hopper WGMMA FP16->FP32 GEMM (row-major C[M, N] = alpha * X0[M, K] @ W0[N, K]^T + beta * C[M, N])
// with exact CUBLAS_COMPUTE_32F FP32 accumulation and CUDA-Graph-capture-safe handle/workspace reuse.
int tsjit_cublas_scaled_fp16_gemm(
    int32_t M,
    int32_t N,
    int32_t K,
    const void* X0_f16,
    const void* W0_f16,
    float* C_f32,
    float alpha,
    float beta,
    cudaStream_t stream) {
#ifdef TSJIT_HAS_CUBLAS
  if (M < 0 || N < 0 || K < 0) {
    return -1;
  }
  if (M == 0 || N == 0) {
    return 0;
  }
  if (C_f32 == nullptr || (K > 0 && (X0_f16 == nullptr || W0_f16 == nullptr))) {
    return -2;
  }
  if (K == 0) {
    if (beta == 0.0f) {
      const size_t bytes = static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(float);
      if (cudaMemsetAsync(C_f32, 0, bytes, stream) != cudaSuccess) {
        return -4;
      }
      return 0;
    }
    if (beta == 1.0f) {
      return 0;
    }
    const size_t total = static_cast<size_t>(M) * static_cast<size_t>(N);
    const uint32_t threads = 256u;
    const size_t blocks_needed = (total + static_cast<size_t>(threads) - 1u) /
                                 static_cast<size_t>(threads);
    const uint32_t grid = static_cast<uint32_t>(blocks_needed < 65535u ? blocks_needed : 65535u);
    tsjit::cuda::tsjit_scale_f32_kernel<<<grid, threads, 0, stream>>>(C_f32, total, beta);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -4;
  }

  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) {
    return -3;
  }
  cublasHandle_t handle = tsjit::cuda::get_thread_local_cublas_handle(dev, stream);
  if (handle == nullptr) {
    return -5;
  }
  if (cublasSetStream(handle, stream) != CUBLAS_STATUS_SUCCESS) {
    return -6;
  }

  // Row-major C(M,N) = alpha * X0(M,K) @ W0(N,K)^T + beta * C(M,N)
  // <-> Col-major C^T(N,M) = alpha * W0(N,K) @ X0^T(K,M) + beta * C^T(N,M).
  cublasStatus_t st = cublasGemmEx(
      handle,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      N,
      M,
      K,
      &alpha,
      W0_f16,
      CUDA_R_16F,
      K,
      X0_f16,
      CUDA_R_16F,
      K,
      &beta,
      C_f32,
      CUDA_R_32F,
      N,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (st != CUBLAS_STATUS_SUCCESS) {
    st = cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        N,
        M,
        K,
        &alpha,
        W0_f16,
        CUDA_R_16F,
        K,
        X0_f16,
        CUDA_R_16F,
        K,
        &beta,
        C_f32,
        CUDA_R_32F,
        N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);
  }
  return (st == CUBLAS_STATUS_SUCCESS) ? 0 : -7;
#else
  (void)M;
  (void)N;
  (void)K;
  (void)X0_f16;
  (void)W0_f16;
  (void)C_f32;
  (void)alpha;
  (void)beta;
  (void)stream;
  return -10;
#endif
}

}  // extern "C"
