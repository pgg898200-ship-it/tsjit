# Copyright 2026 tsjit contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hopper SM90 CUDA & Triton numerical kernels for tsjit."""

from .loader import (
    IPCGResult,
    OnlineGpuRidgeEngine,
    align_k_pad,
    align_up,
    cublas_scaled_fp16_gemm,
    detect_cuda_arch,
    discover_cxx,
    discover_nvcc,
    ipcg_solve_batch,
    is_cublas_fp16_available,
    is_cuda_ipcg_available,
    load_ipcg_cuda_lib,
)
from .scaled_fp16_triton import (
    DEFAULT_CREST_THRESHOLD,
    SM90_GEMM_CONFIGS,
    SM90_VERIFIED_REPORTS,
    TRITON_AVAILABLE,
    CrestGuardedLinear,
    PackedExactWeight,
    SM90BudgetReport,
    check_sm90_budget,
    crest_guarded_linear_numpy,
    fp16x3_linear,
    replace_linears_with_crest_guarded,
    scaled_fp16_linear,
)

__all__ = [
    "DEFAULT_CREST_THRESHOLD",
    "IPCGResult",
    "OnlineGpuRidgeEngine",
    "PackedExactWeight",
    "SM90BudgetReport",
    "SM90_GEMM_CONFIGS",
    "SM90_VERIFIED_REPORTS",
    "TRITON_AVAILABLE",
    "CrestGuardedLinear",
    "align_k_pad",
    "align_up",
    "check_sm90_budget",
    "crest_guarded_linear_numpy",
    "cublas_scaled_fp16_gemm",
    "detect_cuda_arch",
    "discover_cxx",
    "discover_nvcc",
    "fp16x3_linear",
    "ipcg_solve_batch",
    "is_cublas_fp16_available",
    "is_cuda_ipcg_available",
    "load_ipcg_cuda_lib",
    "replace_linears_with_crest_guarded",
    "scaled_fp16_linear",
]
