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

"""JIT compiler, loader, and PyTorch wrappers for the SM90 ADAPT-IPCG CUDA kernel."""

from __future__ import annotations

import ctypes
import hashlib
import math
import os
import shutil
import subprocess
import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple, TypeVar

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional in pure-CPU environments
    torch = None  # type: ignore[assignment]

_F = TypeVar("_F", bound=Callable[..., Any])


def _inference_mode(fn: _F) -> _F:
    if torch is not None:
        return torch.inference_mode()(fn)  # type: ignore[return-value]
    return fn


_EXACT_FP64_INT_LIMIT = 1 << 53  # 9,007,199,254,740,992
_MAX_K = 256
_MAX_Q_BOUND = 32767  # |q| <= 32767 -> per-step outer product <= 1.07e9 (>8.5B updates in int64)
_LIB_LOAD_LOCK = threading.RLock()
_LOADED_LIBS: dict[str, ctypes.CDLL] = {}
_CUBLAS_AVAIL_CACHE: dict[tuple[str, int, int], bool] = {}


class IPCGResult(NamedTuple):
    """Result container returned by ``ipcg_solve_batch``."""

    theta: Any
    converged: Any
    iters: Any
    residual_sq: Any


def align_up(value: int, alignment: int) -> int:
    """Round ``value`` up to a multiple of ``alignment``."""
    return ((int(value) + alignment - 1) // alignment) * alignment


def align_k_pad(k: int, is_fp64: bool = True) -> int:
    """Return the 16-byte aligned padded row stride ``K_PAD`` for feature dim ``k``."""
    if k < 1 or k > _MAX_K:
        raise ValueError(f"Feature dimension K must be in [1, {_MAX_K}], got {k}")
    return align_up(k, 2 if is_fp64 else 4)


def discover_nvcc() -> str | None:
    """Discover the ``nvcc`` compiler from environment or PATH."""
    candidates: list[str] = []
    if env_nvcc := os.environ.get("TSJIT_NVCC"):
        candidates.append(env_nvcc)
    for home_var in ("CUDA_HOME", "CUDA_PATH"):
        if cuda_home := os.environ.get(home_var):
            candidates.append(str(Path(cuda_home) / "bin" / "nvcc"))
    if found := shutil.which("nvcc"):
        candidates.append(found)
    candidates.append("/usr/local/cuda/bin/nvcc")
    for cand in candidates:
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def discover_cxx() -> str | None:
    """Discover the host C++ compiler (``g++``) for nvcc ``-ccbin``."""
    if env_cxx := os.environ.get("TSJIT_CXX"):
        if os.path.isfile(env_cxx) and os.access(env_cxx, os.X_OK):
            return env_cxx
    for name in ("g++", "c++", "clang++"):
        if found := shutil.which(name):
            return found
    return None


def detect_cuda_arch(device: Any = None) -> str:
    """Detect target CUDA architecture (e.g. ``sm_90a``, ``sm_89``, ``sm_80``)."""
    if override := os.environ.get("TSJIT_CUDA_ARCH"):
        return override.strip()
    if torch is not None and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(device)
        if (major, minor) == (9, 0):
            return "sm_90a"
        if (major, minor) == (10, 0):
            return "sm_100a"
        return f"sm_{major}{minor}"
    return "sm_90a"


def _cache_dir() -> Path:
    if env_dir := os.environ.get("TSJIT_CACHE_DIR"):
        return Path(env_dir) / "cuda"
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else (Path.home() / ".cache")
    return base / "tsjit" / "cuda"


def _configure_c_signatures(lib: ctypes.CDLL) -> None:
    c_void_p = ctypes.c_void_p
    c_uint32 = ctypes.c_uint32
    c_int32 = ctypes.c_int32
    c_double = ctypes.c_double
    c_float = ctypes.c_float

    lib.tsjit_ipcg_solve_f64.argtypes = [
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_uint32,
        c_int32,
        c_int32,
        c_double,
        c_int32,
        c_void_p,
    ]
    lib.tsjit_ipcg_solve_f64.restype = ctypes.c_int

    lib.tsjit_ipcg_solve_f32.argtypes = [
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_uint32,
        c_int32,
        c_int32,
        c_float,
        c_int32,
        c_void_p,
    ]
    lib.tsjit_ipcg_solve_f32.restype = ctypes.c_int

    lib.tsjit_gram_accum_i64.argtypes = [
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_uint32,
        c_int32,
        c_int32,
        c_int32,
        c_void_p,
    ]
    lib.tsjit_gram_accum_i64.restype = ctypes.c_int

    lib.tsjit_gram_accum_f64_to_i64.argtypes = [
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_uint32,
        c_int32,
        c_int32,
        c_void_p,
    ]
    lib.tsjit_gram_accum_f64_to_i64.restype = ctypes.c_int

    lib.tsjit_eval_ridge_online_f64.argtypes = [
        c_void_p,
        c_void_p,
        c_double,
        c_double,
        c_double,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_uint32,
        c_int32,
        c_int32,
        c_double,
        c_int32,
        c_void_p,
    ]
    lib.tsjit_eval_ridge_online_f64.restype = ctypes.c_int

    if hasattr(lib, "tsjit_cublas_available"):
        lib.tsjit_cublas_available.argtypes = []
        lib.tsjit_cublas_available.restype = ctypes.c_int

    if hasattr(lib, "tsjit_cublas_warmup"):
        lib.tsjit_cublas_warmup.argtypes = [c_void_p]
        lib.tsjit_cublas_warmup.restype = ctypes.c_int

    if hasattr(lib, "tsjit_cublas_scaled_fp16_gemm"):
        lib.tsjit_cublas_scaled_fp16_gemm.argtypes = [
            c_int32,
            c_int32,
            c_int32,
            c_void_p,
            c_void_p,
            c_void_p,
            c_float,
            c_float,
            c_void_p,
        ]
        lib.tsjit_cublas_scaled_fp16_gemm.restype = ctypes.c_int


def _discover_cublas_paths(
    cuda_roots: list[Path],
) -> tuple[list[Path], list[Path], Path | None]:
    """Locate ``cublas_v2.h`` and ``libcublas.so`` across CUDA Toolkit and PyTorch/nvidia wheels."""
    inc_candidates: list[Path] = []
    lib_candidates: list[Path] = []

    for root in cuda_roots:
        inc_candidates.extend(
            [
                root / "include",
                root / "targets" / "x86_64-linux" / "include",
                root / "targets" / "sbsa-linux" / "include",
            ]
        )
        lib_candidates.extend(
            [
                root / "lib64",
                root / "lib",
                root / "targets" / "x86_64-linux" / "lib",
                root / "targets" / "sbsa-linux" / "lib",
            ]
        )

    if torch is not None and getattr(torch, "__file__", None):
        torch_dir = Path(torch.__file__).resolve().parent
        lib_candidates.append(torch_dir / "lib")
        site_dir = torch_dir.parent
        nvidia_cublas = site_dir / "nvidia" / "cublas"
        inc_candidates.append(nvidia_cublas / "include")
        lib_candidates.append(nvidia_cublas / "lib")

    inc_dirs: list[Path] = []
    seen_inc: set[str] = set()
    has_header = False
    for d in inc_candidates:
        if d.is_dir():
            s = str(d)
            if s not in seen_inc:
                inc_dirs.append(d)
                seen_inc.add(s)
            if (d / "cublas_v2.h").is_file():
                has_header = True

    lib_dirs: list[Path] = []
    seen_lib: set[str] = set()
    cublas_so: Path | None = None
    for d in lib_candidates:
        if d.is_dir():
            s = str(d)
            if s not in seen_lib:
                lib_dirs.append(d)
                seen_lib.add(s)
            if cublas_so is None:
                direct = d / "libcublas.so"
                if direct.is_file():
                    cublas_so = direct
                else:
                    versioned = sorted(d.glob("libcublas.so.*"))
                    if versioned:
                        cublas_so = versioned[0]

    if not has_header:
        return inc_dirs, lib_dirs, None
    return inc_dirs, lib_dirs, cublas_so


def _compile_cuda_so(
    *,
    nvcc: str,
    cxx: str | None,
    target_arch: str,
    cu_path: Path,
    so_path: Path,
    inc_dirs: list[Path],
    lib_dirs: list[Path],
    cublas_so: Path | None,
) -> subprocess.CompletedProcess[str]:
    cache_root = so_path.parent
    tmp_so = cache_root / f".tmp_{os.getpid()}_{uuid.uuid4().hex}.so"
    cmd = [
        nvcc,
        "-O3",
        f"-arch={target_arch}",
        "--prec-div=true",
        "--prec-sqrt=true",
        "-shared",
        "-Xcompiler",
        "-fPIC",
    ]
    if cxx is not None:
        cmd.extend(["-ccbin", cxx])
    if cublas_so is not None:
        cmd.append("-DTSJIT_HAS_CUBLAS")
    for inc in inc_dirs:
        cmd.append(f"-I{inc}")
    for libd in lib_dirs:
        cmd.append(f"-L{libd}")
        if cublas_so is not None:
            cmd.extend(["-Xlinker", f"-rpath={libd}"])
    cmd.extend([str(cu_path), "-o", str(tmp_so)])
    if cublas_so is not None:
        if cublas_so.name == "libcublas.so":
            cmd.append("-lcublas")
        else:
            cmd.extend(["-Xlinker", str(cublas_so)])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            os.replace(tmp_so, so_path)
        return proc
    finally:
        if tmp_so.exists():
            tmp_so.unlink(missing_ok=True)


def load_ipcg_cuda_lib(arch: str | None = None, *, force_rebuild: bool = False) -> ctypes.CDLL:
    """Compile (if needed) and load the SM90 ADAPT-IPCG + cuBLAS WGMMA CUDA shared library."""
    with _LIB_LOAD_LOCK:
        target_arch = arch or detect_cuda_arch()
        if not force_rebuild and target_arch in _LOADED_LIBS:
            return _LOADED_LIBS[target_arch]
        if force_rebuild:
            for k in [key for key in _CUBLAS_AVAIL_CACHE if key[0] == target_arch]:
                _CUBLAS_AVAIL_CACHE.pop(k, None)

        cu_path = Path(__file__).with_name("tsjit_ipcg_sm90.cu")
        if not cu_path.is_file():
            raise FileNotFoundError(f"Missing CUDA kernel source: {cu_path}")

        nvcc = discover_nvcc()
        if nvcc is None:
            raise RuntimeError(
                "nvcc not found. Set TSJIT_NVCC or CUDA_HOME, or install the CUDA Toolkit."
            )
        cxx = discover_cxx()

        cuda_roots: list[Path] = []
        for env_var in ("CUDA_HOME", "CUDA_PATH"):
            if env_val := os.environ.get(env_var):
                cuda_roots.append(Path(env_val))
        cuda_roots.append(Path(nvcc).resolve().parent.parent)
        if Path("/usr/local/cuda").is_dir():
            cuda_roots.append(Path("/usr/local/cuda"))

        inc_dirs, lib_dirs, cublas_so = _discover_cublas_paths(cuda_roots)
        allow_cublas = (cublas_so is not None) and (
            os.environ.get("TSJIT_DISABLE_CUBLAS_FP16") != "1"
        )

        source_bytes = cu_path.read_bytes()
        cache_root = _cache_dir()
        cache_root.mkdir(parents=True, exist_ok=True)

        def _so_path_for(use_cublas: bool) -> Path:
            hasher = hashlib.sha256()
            hasher.update(source_bytes)
            hasher.update(
                f"|arch={target_arch}|nvcc={nvcc}|cxx={cxx or ''}|cublas={int(use_cublas)}".encode()
            )
            return cache_root / f"libtsjit_ipcg_{target_arch}_{hasher.hexdigest()[:16]}.so"

        so_path = _so_path_for(allow_cublas)
        built_with_cublas = allow_cublas

        if force_rebuild or not so_path.is_file():
            proc = _compile_cuda_so(
                nvcc=nvcc,
                cxx=cxx,
                target_arch=target_arch,
                cu_path=cu_path,
                so_path=so_path,
                inc_dirs=inc_dirs,
                lib_dirs=lib_dirs,
                cublas_so=cublas_so if allow_cublas else None,
            )
            if proc.returncode != 0 and allow_cublas:
                # Fall back to pure-IPCG build (pure-Triton Scaled-FP16) if -lcublas fails.
                built_with_cublas = False
                so_path = _so_path_for(False)
                if force_rebuild or not so_path.is_file():
                    proc = _compile_cuda_so(
                        nvcc=nvcc,
                        cxx=cxx,
                        target_arch=target_arch,
                        cu_path=cu_path,
                        so_path=so_path,
                        inc_dirs=inc_dirs,
                        lib_dirs=lib_dirs,
                        cublas_so=None,
                    )
                else:
                    proc = subprocess.CompletedProcess(
                        args=[], returncode=0, stdout="", stderr=""
                    )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"nvcc compilation failed (rc={proc.returncode}):\n"
                    f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
                )

        try:
            lib = ctypes.CDLL(str(so_path))
        except OSError:
            if built_with_cublas and cublas_so is not None:
                try:
                    for lt_cand in sorted(cublas_so.parent.glob("libcublasLt.so*")):
                        if lt_cand.is_file():
                            ctypes.CDLL(str(lt_cand), mode=ctypes.RTLD_GLOBAL)
                            break
                    ctypes.CDLL(str(cublas_so), mode=ctypes.RTLD_GLOBAL)
                    lib = ctypes.CDLL(str(so_path))
                except OSError:
                    # Fall back to no-cuBLAS build if libcublas.so cannot be loaded at runtime.
                    so_nocb = _so_path_for(False)
                    if not so_nocb.is_file():
                        proc = _compile_cuda_so(
                            nvcc=nvcc,
                            cxx=cxx,
                            target_arch=target_arch,
                            cu_path=cu_path,
                            so_path=so_nocb,
                            inc_dirs=inc_dirs,
                            lib_dirs=lib_dirs,
                            cublas_so=None,
                        )
                        if proc.returncode != 0:
                            raise RuntimeError(
                                f"nvcc fallback compilation failed (rc={proc.returncode}):\n"
                                f"{proc.stderr}"
                            ) from None
                    lib = ctypes.CDLL(str(so_nocb))
            else:
                raise

        _configure_c_signatures(lib)
        _LOADED_LIBS[target_arch] = lib
        return lib


def _pad_mat_torch(mat: Any, k: int, k_pad: int) -> Any:
    if mat.shape[-1] == k_pad and mat.is_contiguous():
        return mat
    out = torch.zeros((*mat.shape[:-2], k, k_pad), dtype=mat.dtype, device=mat.device)
    out[..., :k] = mat[..., :k]
    return out


def _pad_vec_torch(vec: Any, k: int, k_pad: int) -> Any:
    if vec.shape[-1] == k_pad and vec.is_contiguous():
        return vec
    out = torch.zeros((*vec.shape[:-1], k_pad), dtype=vec.dtype, device=vec.device)
    out[..., :k] = vec[..., :k]
    return out


def _ipcg_solve_numpy(
    A: np.ndarray,
    M0: np.ndarray | None,
    rhs: np.ndarray,
    x_init: np.ndarray,
    tol: float,
    max_iters: int,
    fallback_cholesky: bool,
) -> IPCGResult:
    """Exact CPU NumPy mirror of ADAPT-IPCG with true-residual check and Cholesky fallback."""
    dtype = np.float64 if A.dtype == np.float64 else np.float32
    A_arr = np.asarray(A, dtype=dtype)
    rhs_arr = np.asarray(rhs, dtype=dtype)
    x0_arr = np.asarray(x_init, dtype=dtype)
    M0_arr = None if M0 is None else np.asarray(M0, dtype=dtype)

    N, K, _ = A_arr.shape
    theta = x0_arr.copy()
    conv = np.zeros(N, dtype=np.bool_)
    iters = np.zeros(N, dtype=np.int32)
    res_sq = np.full(N, np.nan, dtype=dtype)
    tol2 = dtype(tol) * dtype(tol)

    min_safe = dtype(1e-200) if dtype is np.float64 else dtype(1e-30)
    max_safe = dtype(1e200) if dtype is np.float64 else dtype(1e30)
    max_init_ratio = dtype(1e100) if dtype is np.float64 else dtype(1e12)

    for n in range(N):
        An = A_arr[n, :K, :K]
        bn = rhs_arr[n, :K]
        xn_orig = x0_arr[n, :K]
        Mn = None if M0_arr is None else M0_arr[n, :K, :K]

        if not (
            np.all(np.isfinite(bn))
            and np.all(np.isfinite(xn_orig))
            and np.isfinite(tol)
            and tol >= 0.0
        ):
            continue

        rhs_max = dtype(np.max(np.abs(bn))) if K > 0 else dtype(0)
        x0_max = dtype(np.max(np.abs(xn_orig))) if K > 0 else dtype(0)
        ref_mag = rhs_max if rhs_max > dtype(0) else (x0_max if x0_max > dtype(0) else dtype(1))
        rhs_scale = dtype(np.clip(ref_mag, min_safe, max_safe)) if ref_mag > dtype(0) else dtype(1)
        inv_rhs_scale = dtype(dtype(1) / rhs_scale)

        bn_s = (bn * inv_rhs_scale).astype(dtype, copy=False)
        x0_scaled_max = dtype(x0_max * inv_rhs_scale)
        x0_scales_ok = bool(np.isfinite(x0_scaled_max) and (x0_scaled_max <= max_init_ratio))
        if x0_scales_ok:
            xn = (xn_orig * inv_rhs_scale).astype(dtype, copy=True)
        else:
            xn = np.zeros(K, dtype=dtype)

        rhs_n2 = dtype(np.dot(bn_s, bn_s))
        if not np.isfinite(rhs_n2):
            continue
        target_r2 = dtype(tol2 * rhs_n2) if rhs_n2 > dtype(0) else dtype(0)
        if not np.isfinite(target_r2):
            continue

        r = bn_s - An @ xn
        r_n2 = dtype(np.dot(r, r))
        if (not np.isfinite(r_n2)) and x0_scales_ok and (x0_max > dtype(0)):
            xn = np.zeros(K, dtype=dtype)
            r = bn_s - An @ xn
            r_n2 = dtype(np.dot(r, r))
        verified_true_r2 = False
        true_r2_scaled = dtype(np.nan)
        it_done = 0

        if np.isfinite(r_n2):
            if r_n2 <= target_r2:
                true_r2_scaled = r_n2
                verified_true_r2 = True
            elif max_iters > 0:
                if Mn is not None:
                    z = Mn @ r
                else:
                    diag = np.diag(An)
                    z = np.where((diag > 0) & np.isfinite(diag), r / diag, np.nan).astype(dtype)
                p = z.copy()
                rz = dtype(np.dot(r, z))
                restarts = 0
                if rz > dtype(0) and np.isfinite(rz):
                    for it in range(1, max_iters + 1):
                        ap = An @ p
                        pap = dtype(np.dot(p, ap))
                        it_done = it
                        if not (pap > dtype(0) and np.isfinite(pap)):
                            break
                        alpha = dtype(rz / pap)
                        if not np.isfinite(alpha):
                            break
                        xn = xn + alpha * p
                        r = r - alpha * ap
                        r_n2 = dtype(np.dot(r, r))
                        if not np.isfinite(r_n2):
                            break
                        if r_n2 <= target_r2:
                            r_chk = bn_s - An @ xn
                            chk_r2 = dtype(np.dot(r_chk, r_chk))
                            if np.isfinite(chk_r2) and chk_r2 <= target_r2:
                                true_r2_scaled = chk_r2
                                verified_true_r2 = True
                                break
                            if (not np.isfinite(chk_r2)) or restarts >= 1 or it == max_iters:
                                true_r2_scaled = chk_r2
                                verified_true_r2 = True
                                break
                            restarts += 1
                            r = r_chk
                            if Mn is not None:
                                z = Mn @ r
                            else:
                                diag = np.diag(An)
                                z = np.where(
                                    (diag > 0) & np.isfinite(diag), r / diag, np.nan
                                ).astype(dtype)
                            p = z.copy()
                            rz = dtype(np.dot(r, z))
                            if not (rz > dtype(0) and np.isfinite(rz)):
                                break
                            continue
                        if it == max_iters:
                            break
                        if Mn is not None:
                            z = Mn @ r
                        else:
                            diag = np.diag(An)
                            z = np.where((diag > 0) & np.isfinite(diag), r / diag, np.nan).astype(
                                dtype
                            )
                        rz_new = dtype(np.dot(r, z))
                        if not (rz_new > dtype(0) and np.isfinite(rz_new)):
                            break
                        beta = dtype(rz_new / rz)
                        if not np.isfinite(beta):
                            break
                        p = z + beta * p
                        rz = rz_new

        if not verified_true_r2:
            true_r = bn_s - An @ xn
            true_r2_scaled = dtype(np.dot(true_r, true_r))
        xn_unscaled = (xn * rhs_scale).astype(dtype, copy=False)
        x_fin = bool(np.all(np.isfinite(xn_unscaled)))
        ok = bool(np.isfinite(true_r2_scaled) and x_fin and (true_r2_scaled <= target_r2))
        iters[n] = it_done
        res_sq[n] = (
            dtype(dtype(true_r2_scaled * rhs_scale) * rhs_scale)
            if np.isfinite(true_r2_scaled)
            else true_r2_scaled
        )
        if ok:
            theta[n, :K] = xn_unscaled
            conv[n] = True
        elif fallback_cholesky:
            try:
                if np.all(np.isfinite(An)) and np.all(np.isfinite(bn)):
                    L = np.linalg.cholesky(An.astype(np.float64))
                    y = np.linalg.solve(L, bn.astype(np.float64))
                    x_chol = np.linalg.solve(L.T, y).astype(dtype)
                    if np.all(np.isfinite(x_chol)):
                        theta[n, :K] = x_chol
                        conv[n] = True
                        r_c = bn - An @ x_chol
                        res_sq[n] = dtype(np.dot(r_c, r_c))
            except np.linalg.LinAlgError:
                pass

    return IPCGResult(theta=theta, converged=conv, iters=iters, residual_sq=res_sq)


def _selective_cholesky_fallback_torch(
    A: Any,
    rhs: Any,
    theta: Any,
    conv: Any,
    res_sq: Any,
) -> Any:
    """Run selective ``torch.linalg.cholesky_ex`` only for non-converged entries."""
    bad_mask = ~conv
    if not bool(bad_mask.any()):
        return None
    bad_idx = torch.nonzero(bad_mask, as_tuple=False).squeeze(-1)
    K = A.shape[1]
    A_bad = A[bad_idx, :K, :K]
    rhs_bad = rhs[bad_idx, :K]

    fin_mask = torch.isfinite(A_bad).flatten(1).all(dim=1) & torch.isfinite(rhs_bad).all(dim=1)
    if not bool(fin_mask.any()):
        return bad_idx

    valid_idx = bad_idx[fin_mask]
    A_val = A_bad[fin_mask].to(torch.float64)
    rhs_val = rhs_bad[fin_mask].to(torch.float64)

    L, info = torch.linalg.cholesky_ex(A_val)
    ok_chol = info == 0
    if bool(ok_chol.any()):
        solved_idx = valid_idx[ok_chol]
        L_ok = L[ok_chol]
        x_chol = torch.cholesky_solve(rhs_val[ok_chol].unsqueeze(-1), L_ok).squeeze(-1)
        x_chol_cast = x_chol.to(dtype=theta.dtype)
        x_fin = torch.isfinite(x_chol_cast).all(dim=1)
        if bool(x_fin.any()):
            final_idx = solved_idx[x_fin]
            x_ok = x_chol_cast[x_fin]
            theta[final_idx, :K] = x_ok
            conv[final_idx] = True
            r_c = rhs[final_idx, :K] - torch.bmm(A[final_idx, :K, :K], x_ok.unsqueeze(-1)).squeeze(
                -1
            )
            res_sq[final_idx] = (r_c * r_c).sum(dim=-1)
            return final_idx, L_ok[x_fin]
    return None


def ipcg_solve_batch(
    A: Any,
    M0: Any | None,
    rhs: Any,
    x_init: Any,
    tol: float = 1e-8,
    max_iters: int = 16,
    *,
    use_cuda_kernel: bool = True,
    fallback_cholesky: bool = True,
) -> IPCGResult:
    """Solve batched ``N`` SPD systems ``A[n] x[n] = rhs[n]`` via SMEM-resident ADAPT-IPCG.

    Non-converged systems leave ``theta`` untouched by the CUDA kernel and optionally
    invoke selective ``torch.linalg.cholesky_ex`` fallback solely on failed indices.
    """
    if isinstance(A, np.ndarray) or torch is None:
        return _ipcg_solve_numpy(
            np.asarray(A),
            None if M0 is None else np.asarray(M0),
            np.asarray(rhs),
            np.asarray(x_init),
            tol=tol,
            max_iters=max_iters,
            fallback_cholesky=fallback_cholesky,
        )

    if not isinstance(A, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor or np.ndarray for A, got {type(A)}")
    if A.ndim != 3:
        raise ValueError(f"A must have shape (N, K, K) or (N, K, K_PAD), got {tuple(A.shape)}")
    N, K, cols = A.shape
    if K < 1 or K > _MAX_K or cols < K:
        raise ValueError(f"Invalid A shape {tuple(A.shape)} (require 1 <= K <= {_MAX_K})")
    if A.dtype not in (torch.float64, torch.float32):
        raise ValueError(f"A must be float64 or float32, got {A.dtype}")

    if not isinstance(rhs, torch.Tensor) or not isinstance(x_init, torch.Tensor):
        raise TypeError("rhs and x_init must be torch.Tensor when A is torch.Tensor")
    if rhs.ndim != 2 or rhs.shape[0] != N or rhs.shape[1] < K:
        raise ValueError(f"Invalid rhs shape {tuple(rhs.shape)} for N={N}, K={K}")
    if x_init.ndim != 2 or x_init.shape[0] != N or x_init.shape[1] < K:
        raise ValueError(f"Invalid x_init shape {tuple(x_init.shape)} for N={N}, K={K}")
    if rhs.dtype != A.dtype or x_init.dtype != A.dtype:
        raise ValueError(f"Mismatched dtypes: A={A.dtype}, rhs={rhs.dtype}, x_init={x_init.dtype}")
    if rhs.device != A.device or x_init.device != A.device:
        raise ValueError(
            f"Mismatched devices: A={A.device}, rhs={rhs.device}, x_init={x_init.device}"
        )
    if M0 is not None:
        if not isinstance(M0, torch.Tensor) or M0.ndim != 3 or M0.shape[:2] != (N, K):
            raise ValueError(f"Invalid M0 shape for N={N}, K={K}")
        if M0.shape[2] < K or M0.dtype != A.dtype or M0.device != A.device:
            raise ValueError("M0 must match A's dtype, device, and have cols >= K")

    is_fp64 = A.dtype == torch.float64
    k_pad = align_k_pad(K, is_fp64=is_fp64)

    if A.is_cuda and use_cuda_kernel:
        lib = load_ipcg_cuda_lib(detect_cuda_arch(A.device))
        with torch.cuda.device(A.device):
            A_pad = _pad_mat_torch(A, K, k_pad)
            M0_pad = None if M0 is None else _pad_mat_torch(M0, K, k_pad)
            rhs_pad = _pad_vec_torch(rhs, K, k_pad)
            x0_pad = _pad_vec_torch(x_init, K, k_pad)
            x_out_pad = x0_pad.clone()

            for name, tensor in (
                ("A", A_pad),
                ("rhs", rhs_pad),
                ("x_init", x0_pad),
                ("x_out", x_out_pad),
            ):
                if (tensor.data_ptr() & 15) != 0:
                    raise RuntimeError(f"Tensor {name} is not 16-byte aligned")
            if M0_pad is not None and (M0_pad.data_ptr() & 15) != 0:
                raise RuntimeError("Tensor M0 is not 16-byte aligned")

            conv_u8 = torch.zeros(N, dtype=torch.uint8, device=A.device)
            iters = torch.zeros(N, dtype=torch.int32, device=A.device)
            res_sq = torch.empty(N, dtype=A.dtype, device=A.device)
            stream_ptr = int(torch.cuda.current_stream(A.device).cuda_stream)

            fn = lib.tsjit_ipcg_solve_f64 if is_fp64 else lib.tsjit_ipcg_solve_f32
            rc = fn(
                ctypes.c_void_p(A_pad.data_ptr()),
                ctypes.c_void_p(0 if M0_pad is None else M0_pad.data_ptr()),
                ctypes.c_void_p(rhs_pad.data_ptr()),
                ctypes.c_void_p(x0_pad.data_ptr()),
                ctypes.c_void_p(x_out_pad.data_ptr()),
                ctypes.c_void_p(conv_u8.data_ptr()),
                ctypes.c_void_p(iters.data_ptr()),
                ctypes.c_void_p(res_sq.data_ptr()),
                ctypes.c_uint32(N),
                ctypes.c_int32(K),
                ctypes.c_int32(k_pad),
                ctypes.c_double(tol) if is_fp64 else ctypes.c_float(tol),
                ctypes.c_int32(max_iters),
                ctypes.c_void_p(stream_ptr),
            )
            if rc != 0:
                raise RuntimeError(f"tsjit_ipcg_solve CUDA launcher failed with code {rc}")

            conv = conv_u8.bool()
            theta = x_out_pad[:, :K].contiguous()
            if fallback_cholesky and not bool(conv.all()):
                _selective_cholesky_fallback_torch(A_pad, rhs_pad, theta, conv, res_sq)
            return IPCGResult(theta=theta, converged=conv, iters=iters, residual_sq=res_sq)

    # Pure-PyTorch fallback (CPU or CUDA with use_cuda_kernel=False)
    A_c = A[:, :K, :K]
    rhs_c = rhs[:, :K]
    x0_c = x_init[:, :K]
    M0_c = None if M0 is None else M0[:, :K, :K]

    res_np = _ipcg_solve_numpy(
        A_c.detach().cpu().numpy(),
        None if M0_c is None else M0_c.detach().cpu().numpy(),
        rhs_c.detach().cpu().numpy(),
        x0_c.detach().cpu().numpy(),
        tol=tol,
        max_iters=max_iters,
        fallback_cholesky=False,
    )
    theta = torch.from_numpy(res_np.theta).to(device=A.device, dtype=A.dtype)
    conv = torch.from_numpy(res_np.converged).to(device=A.device)
    iters = torch.from_numpy(res_np.iters).to(device=A.device)
    res_sq = torch.from_numpy(res_np.residual_sq).to(device=A.device, dtype=A.dtype)
    if fallback_cholesky and not bool(conv.all()):
        _selective_cholesky_fallback_torch(A_c, rhs_c, theta, conv, res_sq)
    return IPCGResult(theta=theta, converged=conv, iters=iters, residual_sq=res_sq)


class OnlineGpuRidgeEngine:
    """Exact online batched ridge engine with FP64-TC int64 Gram accumulation and ADAPT-IPCG.

    Guarantees associative, bit-exact integer Gram accumulation by explicitly verifying
    ``B_bucket * max_abs_q^2 < 2^53`` before dispatching Hopper FP64 Tensor Core GEMM;
    overflowing buckets are automatically split or routed through exact int64 kernels.
    """

    def __init__(
        self,
        num_series: int,
        feature_dim: int,
        *,
        ridge_lambda: float = 1e-2,
        q_scale: float = 1024.0,
        dtype: Any = None,
        device: Any = "cuda",
        use_cuda_kernel: bool = True,
    ) -> None:
        if torch is None:
            raise RuntimeError("OnlineGpuRidgeEngine requires PyTorch.")
        if isinstance(num_series, bool) or num_series < 1:
            raise ValueError(f"num_series must be >= 1, got {num_series}")
        if isinstance(feature_dim, bool) or feature_dim < 1 or feature_dim > _MAX_K:
            raise ValueError(f"feature_dim must be in [1, {_MAX_K}], got {feature_dim}")
        if not (math.isfinite(ridge_lambda) and ridge_lambda > 0.0):
            raise ValueError(f"ridge_lambda must be finite and > 0, got {ridge_lambda}")
        if not (
            math.isfinite(q_scale)
            and q_scale > 0.0
            and math.isfinite(q_scale * q_scale)
            and (q_scale * q_scale) > 0.0
        ):
            raise ValueError(
                f"q_scale must be finite with finite positive q_scale^2, got {q_scale}"
            )

        self.num_series = int(num_series)
        self.K = int(feature_dim)
        self.ridge_lambda = float(ridge_lambda)
        self.q_scale = float(q_scale)
        self.max_q_bound = _MAX_Q_BOUND
        self.dtype = torch.float64 if dtype is None else dtype
        self.is_fp64 = self.dtype == torch.float64
        self.k_pad = align_k_pad(self.K, is_fp64=self.is_fp64)
        self.device = torch.device(
            device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        )
        self.use_cuda_kernel = bool(use_cuda_kernel and self.device.type == "cuda")

        self.G_i64 = torch.zeros(
            (self.num_series, self.K, self.k_pad), dtype=torch.int64, device=self.device
        )
        self.b_i64 = torch.zeros(
            (self.num_series, self.k_pad), dtype=torch.int64, device=self.device
        )
        self.theta = torch.zeros(
            (self.num_series, self.k_pad), dtype=self.dtype, device=self.device
        )
        self.M0 = torch.zeros(
            (self.num_series, self.K, self.k_pad), dtype=self.dtype, device=self.device
        )
        eye = torch.eye(self.K, dtype=self.dtype, device=self.device) * (1.0 / self.ridge_lambda)
        self.M0[:, :, : self.K] = eye.unsqueeze(0)
        self.total_fallbacks = 0
        self.total_solves = 0

    def _accum_subbucket(
        self, X_q: Any, y_q: Any, *, known_max_abs: int | None = None
    ) -> None:
        """Accumulate integer-valued ``X_q [N, B, K]`` and ``y_q [N, B]`` exactly into int64.

        When ``known_max_abs`` is supplied (e.g., ``32767`` from clamped ``update()``),
        verifies ``B * known_max_abs^2 < 2^53`` purely on host metadata with ZERO
        ``.item()`` D2H GPU synchronizations.
        """
        N, B, K = X_q.shape
        if B == 0:
            return
        if known_max_abs is not None:
            max_abs = int(known_max_abs)
        else:
            max_x = int(X_q.abs().max().item()) if X_q.numel() > 0 else 0
            max_y = int(y_q.abs().max().item()) if y_q.numel() > 0 else 0
            max_abs = max(max_x, max_y)

        # Verify exact FP64-TC integer representability: B * max_abs^2 < 2^53.
        if max_abs == 0:
            return
        sq_bound = max_abs * max_abs
        if B * sq_bound < _EXACT_FP64_INT_LIMIT:
            X_f64 = X_q.to(torch.float64)
            y_f64 = y_q.to(torch.float64)
            G_delta = torch.bmm(X_f64.transpose(1, 2), X_f64)
            b_delta = torch.bmm(X_f64.transpose(1, 2), y_f64.unsqueeze(-1)).squeeze(-1)

            if self.use_cuda_kernel:
                lib = load_ipcg_cuda_lib(detect_cuda_arch(self.device))
                with torch.cuda.device(self.device):
                    # Bind contiguous buffers to locals so they stay alive through async launch.
                    G_delta_c = G_delta.contiguous()
                    b_delta_c = b_delta.contiguous()
                    stream_ptr = int(torch.cuda.current_stream(self.device).cuda_stream)
                    rc = lib.tsjit_gram_accum_f64_to_i64(
                        ctypes.c_void_p(G_delta_c.data_ptr()),
                        ctypes.c_void_p(b_delta_c.data_ptr()),
                        ctypes.c_void_p(self.G_i64.data_ptr()),
                        ctypes.c_void_p(self.b_i64.data_ptr()),
                        ctypes.c_uint32(N),
                        ctypes.c_int32(K),
                        ctypes.c_int32(self.k_pad),
                        ctypes.c_void_p(stream_ptr),
                    )
                    if rc != 0:
                        raise RuntimeError(f"tsjit_gram_accum_f64_to_i64 failed with code {rc}")
            else:
                self.G_i64[:, :, :K].add_(torch.round(G_delta).to(torch.int64))
                self.b_i64[:, :K].add_(torch.round(b_delta).to(torch.int64))
            return

        # If a single sample fits below 2^53, split B_bucket into exact sub-buckets.
        if sq_bound < _EXACT_FP64_INT_LIMIT:
            safe_b = max(1, (_EXACT_FP64_INT_LIMIT - 1) // sq_bound)
            for start in range(0, B, safe_b):
                self._accum_subbucket(
                    X_q[:, start : start + safe_b, :],
                    y_q[:, start : start + safe_b],
                    known_max_abs=max_abs,
                )
            return

        # Exact int64 fallback when max_abs^2 >= 2^53.
        if self.use_cuda_kernel and max_abs < (1 << 31):
            lib = load_ipcg_cuda_lib(detect_cuda_arch(self.device))
            with torch.cuda.device(self.device):
                X_i32 = X_q.to(torch.int32).contiguous()
                y_i32 = y_q.to(torch.int32).contiguous()
                stream_ptr = int(torch.cuda.current_stream(self.device).cuda_stream)
                rc = lib.tsjit_gram_accum_i64(
                    ctypes.c_void_p(X_i32.data_ptr()),
                    ctypes.c_void_p(y_i32.data_ptr()),
                    ctypes.c_void_p(self.G_i64.data_ptr()),
                    ctypes.c_void_p(self.b_i64.data_ptr()),
                    ctypes.c_uint32(N),
                    ctypes.c_int32(B),
                    ctypes.c_int32(K),
                    ctypes.c_int32(self.k_pad),
                    ctypes.c_void_p(stream_ptr),
                )
                if rc != 0:
                    raise RuntimeError(f"tsjit_gram_accum_i64 failed with code {rc}")
        else:
            X_i64 = X_q.to(torch.int64)
            y_i64 = y_q.to(torch.int64)
            # Chunked int64 accumulation avoiding large 4D temporaries.
            for b_idx in range(B):
                xb = X_i64[:, b_idx, :]
                yb = y_i64[:, b_idx]
                self.G_i64[:, :, :K].add_(xb.unsqueeze(2) * xb.unsqueeze(1))
                self.b_i64[:, :K].add_(xb * yb.unsqueeze(1))

    @_inference_mode
    def update(self, X: Any, y: Any) -> None:
        """Quantize and exactly accumulate observations ``X [N, B, K]``, ``y [N, B]``."""
        X_t = torch.as_tensor(X, device=self.device, dtype=torch.float64)
        y_t = torch.as_tensor(y, device=self.device, dtype=torch.float64)
        if X_t.ndim == 2:
            X_t = X_t.unsqueeze(1)
        if y_t.ndim == 1:
            y_t = y_t.unsqueeze(1)
        if X_t.shape[0] != self.num_series or X_t.shape[2] != self.K:
            raise ValueError(
                f"Expected X shape ({self.num_series}, B, {self.K}), got {tuple(X_t.shape)}"
            )
        if y_t.shape != (self.num_series, X_t.shape[1]):
            raise ValueError(
                f"Expected y shape ({self.num_series}, {X_t.shape[1]}), got {tuple(y_t.shape)}"
            )

        # Guard against NaN/Inf or int64-overflowing observations corrupting int64 accumulators.
        valid = torch.isfinite(X_t).all(dim=-1) & torch.isfinite(y_t)
        X_clean = torch.where(valid.unsqueeze(-1), X_t, torch.zeros_like(X_t))
        y_clean = torch.where(valid, y_t, torch.zeros_like(y_t))

        # Clamp strictly to [-32767, 32767]: per-step outer product <= 1.0737e9
        # (>8.5B updates before int64 overflow) and B_bucket * 32767^2 << 2^53
        # for exact FP64-TC bmm with zero per-step .item() D2H synchronizations.
        max_q_bound = float(self.max_q_bound)
        X_scaled = torch.clamp(torch.round(X_clean * self.q_scale), -max_q_bound, max_q_bound)
        y_scaled = torch.clamp(torch.round(y_clean * self.q_scale), -max_q_bound, max_q_bound)
        X_q = X_scaled.to(torch.int64)
        y_q = y_scaled.to(torch.int64)
        self._accum_subbucket(X_q, y_q, known_max_abs=self.max_q_bound)

    @_inference_mode
    def build_system(self) -> tuple[Any, Any]:
        """Materialize dequantized ``(A, rhs)`` from the exact int64 Gram accumulators."""
        inv_s2 = 1.0 / (self.q_scale * self.q_scale)
        A = torch.zeros((self.num_series, self.K, self.k_pad), dtype=self.dtype, device=self.device)
        A[:, :, : self.K] = self.G_i64[:, :, : self.K].to(self.dtype) * inv_s2
        diag_idx = torch.arange(self.K, device=self.device)
        A[:, diag_idx, diag_idx] += self.ridge_lambda
        rhs = torch.zeros((self.num_series, self.k_pad), dtype=self.dtype, device=self.device)
        rhs[:, : self.K] = self.b_i64[:, : self.K].to(self.dtype) * inv_s2
        return A, rhs

    @_inference_mode
    def solve(self, tol: float = 1e-8, max_iters: int = 16) -> IPCGResult:
        """Solve current online ridge system via ADAPT-IPCG + selective Cholesky refresh."""
        A, rhs = self.build_system()
        res = ipcg_solve_batch(
            A,
            self.M0,
            rhs,
            self.theta,
            tol=tol,
            max_iters=max_iters,
            use_cuda_kernel=self.use_cuda_kernel,
            fallback_cholesky=False,
        )
        self.total_solves += self.num_series
        self.theta[:, : self.K] = res.theta
        conv = res.converged
        res_sq = res.residual_sq

        if not bool(conv.all()):
            num_bad = int((~conv).sum().item())
            self.total_fallbacks += num_bad
            fallback_out = _selective_cholesky_fallback_torch(
                A, rhs, self.theta[:, : self.K], conv, res_sq
            )
            if isinstance(fallback_out, tuple):
                solved_idx, L_ok = fallback_out
                # Selectively refresh preconditioner M0 = A^{-1} for recovered series.
                self.M0[solved_idx, :, : self.K] = torch.cholesky_inverse(L_ok).to(
                    dtype=self.dtype
                )

        return IPCGResult(
            theta=self.theta[:, : self.K].clone(),
            converged=conv,
            iters=res.iters,
            residual_sq=res_sq,
        )

    @_inference_mode
    def eval_ridge_online(
        self,
        x_query: Any | None = None,
        *,
        tol: float = 1e-8,
        max_iters: int = 16,
    ) -> tuple[IPCGResult, Any | None]:
        """Fused finalize + ADAPT-IPCG solve + optional batched prediction."""
        if self.use_cuda_kernel and self.is_fp64:
            lib = load_ipcg_cuda_lib(detect_cuda_arch(self.device))
            inv_s2 = 1.0 / (self.q_scale * self.q_scale)
            with torch.cuda.device(self.device):
                A_work = torch.empty(
                    (self.num_series, self.K, self.k_pad), dtype=torch.float64, device=self.device
                )
                rhs_work = torch.empty(
                    (self.num_series, self.k_pad), dtype=torch.float64, device=self.device
                )
                theta_out = self.theta.clone()
                conv_u8 = torch.zeros(self.num_series, dtype=torch.uint8, device=self.device)
                iters = torch.zeros(self.num_series, dtype=torch.int32, device=self.device)
                res_sq = torch.empty(self.num_series, dtype=torch.float64, device=self.device)

                xq_pad = None
                y_pred = None
                if x_query is not None:
                    xq_t = torch.as_tensor(x_query, device=self.device, dtype=torch.float64)
                    xq_pad = _pad_vec_torch(xq_t, self.K, self.k_pad)
                    y_pred = torch.empty(self.num_series, dtype=torch.float64, device=self.device)

                stream_ptr = int(torch.cuda.current_stream(self.device).cuda_stream)
                rc = lib.tsjit_eval_ridge_online_f64(
                    ctypes.c_void_p(self.G_i64.data_ptr()),
                    ctypes.c_void_p(self.b_i64.data_ptr()),
                    ctypes.c_double(inv_s2),
                    ctypes.c_double(inv_s2),
                    ctypes.c_double(self.ridge_lambda),
                    ctypes.c_void_p(A_work.data_ptr()),
                    ctypes.c_void_p(self.M0.data_ptr()),
                    ctypes.c_void_p(rhs_work.data_ptr()),
                    ctypes.c_void_p(self.theta.data_ptr()),
                    ctypes.c_void_p(theta_out.data_ptr()),
                    ctypes.c_void_p(0 if xq_pad is None else xq_pad.data_ptr()),
                    ctypes.c_void_p(0 if y_pred is None else y_pred.data_ptr()),
                    ctypes.c_void_p(conv_u8.data_ptr()),
                    ctypes.c_void_p(iters.data_ptr()),
                    ctypes.c_void_p(res_sq.data_ptr()),
                    ctypes.c_uint32(self.num_series),
                    ctypes.c_int32(self.K),
                    ctypes.c_int32(self.k_pad),
                    ctypes.c_double(tol),
                    ctypes.c_int32(max_iters),
                    ctypes.c_void_p(stream_ptr),
                )
                if rc != 0:
                    raise RuntimeError(f"tsjit_eval_ridge_online_f64 failed with code {rc}")

                self.total_solves += self.num_series
                self.theta[:, : self.K] = theta_out[:, : self.K]
                conv = conv_u8.bool()
                if not bool(conv.all()):
                    num_bad = int((~conv).sum().item())
                    self.total_fallbacks += num_bad
                    fallback_out = _selective_cholesky_fallback_torch(
                        A_work, rhs_work, self.theta[:, : self.K], conv, res_sq
                    )
                    if isinstance(fallback_out, tuple):
                        solved_idx, L_ok = fallback_out
                        self.M0[solved_idx, :, : self.K] = torch.cholesky_inverse(L_ok).to(
                            dtype=self.dtype
                        )
                        if xq_pad is not None and y_pred is not None:
                            y_pred[solved_idx] = (
                                xq_pad[solved_idx, : self.K] * self.theta[solved_idx, : self.K]
                            ).sum(dim=-1)
                res = IPCGResult(
                    theta=self.theta[:, : self.K].clone(),
                    converged=conv,
                    iters=iters,
                    residual_sq=res_sq,
                )
                return res, y_pred

        res = self.solve(tol=tol, max_iters=max_iters)
        y_pred = None if x_query is None else self.predict(x_query)
        return res, y_pred

    @_inference_mode
    def predict(self, x_query: Any) -> Any:
        """Compute batched ridge predictions ``y_hat[n] = dot(x_query[n], theta[n])``."""
        xq = torch.as_tensor(x_query, device=self.device, dtype=self.dtype)
        return (xq * self.theta[:, : self.K]).sum(dim=-1)


def is_cuda_ipcg_available() -> bool:
    """Return True if PyTorch CUDA and ``nvcc`` are both available."""
    return bool(torch is not None and torch.cuda.is_available() and discover_nvcc() is not None)


_DEVICE_ARCH_CACHE: dict[int, str] = {}


def _resolve_device_index(device: Any) -> int:
    if device is not None and hasattr(device, "index") and device.index is not None:
        return int(device.index)
    if isinstance(device, int):
        return int(device)
    if isinstance(device, str) and ":" in device:
        try:
            return int(device.split(":", 1)[1])
        except ValueError:
            pass
    if torch is not None and torch.cuda.is_available():
        return int(torch.cuda.current_device())
    return 0


def _cached_arch_for_device(device: Any) -> str:
    dev_idx = _resolve_device_index(device)
    with _LIB_LOAD_LOCK:
        arch = _DEVICE_ARCH_CACHE.get(dev_idx)
        if arch is None:
            arch = detect_cuda_arch(device)
            _DEVICE_ARCH_CACHE[dev_idx] = arch
        return arch


def is_cublas_fp16_available(device: Any = None) -> bool:
    """Return True if the compiled CUDA library supports cuBLAS WGMMA FP16->FP32 GEMM.

    Also pre-warms the thread-local per-device ``cublasHandle_t`` + 32 MiB WGMMA
    workspace outside CUDA Graph capture so first-time graph capture never fails
    and broken runtime ``libcublas`` environments cleanly fall back to pure-Triton.
    """
    if os.environ.get("TSJIT_DISABLE_CUBLAS_FP16") == "1":
        return False
    if torch is None or not torch.cuda.is_available() or discover_nvcc() is None:
        return False
    dev_idx = _resolve_device_index(device)
    arch = _cached_arch_for_device(device)
    cache_key = (arch, dev_idx, threading.get_ident())
    with _LIB_LOAD_LOCK:
        cached = _CUBLAS_AVAIL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        lib = load_ipcg_cuda_lib(arch)
        avail = bool(
            hasattr(lib, "tsjit_cublas_available") and int(lib.tsjit_cublas_available()) == 1
        )
        if avail and hasattr(lib, "tsjit_cublas_warmup"):
            dev_obj = torch.device(f"cuda:{dev_idx}")
            with torch.cuda.device(dev_obj):
                is_capturing_fn = getattr(torch.cuda, "is_current_stream_capturing", None)
                capturing = bool(is_capturing_fn()) if callable(is_capturing_fn) else False
                if capturing:
                    # Do not cache False while capturing; allow warmup once outside capture.
                    return False
                stream_ptr = int(torch.cuda.current_stream(dev_obj).cuda_stream)
                avail = int(lib.tsjit_cublas_warmup(ctypes.c_void_p(stream_ptr))) == 0
    except Exception:
        avail = False
    with _LIB_LOAD_LOCK:
        _CUBLAS_AVAIL_CACHE[cache_key] = avail
    return avail


def cublas_scaled_fp16_gemm(
    X0_f16: Any,
    W0_f16: Any,
    C_f32: Any,
    alpha: float = 1.0,
    beta: float = 0.0,
    *,
    lib: ctypes.CDLL | None = None,
) -> None:
    """Execute ``C_f32 = alpha * X0_f16 @ W0_f16.T + beta * C_f32`` via cuBLAS WGMMA FP32-acc."""
    if torch is None:
        raise RuntimeError("cublas_scaled_fp16_gemm requires PyTorch with CUDA.")
    if not (X0_f16.is_cuda and W0_f16.is_cuda and C_f32.is_cuda):
        raise ValueError("X0_f16, W0_f16, and C_f32 must be CUDA tensors.")
    if X0_f16.device != W0_f16.device or X0_f16.device != C_f32.device:
        raise ValueError(
            f"All tensors must reside on the same CUDA device: "
            f"X0={X0_f16.device}, W0={W0_f16.device}, C={C_f32.device}"
        )
    if X0_f16.ndim != 2 or W0_f16.ndim != 2 or C_f32.ndim != 2:
        raise ValueError(
            f"Expected 2D tensors, got X0={tuple(X0_f16.shape)}, "
            f"W0={tuple(W0_f16.shape)}, C={tuple(C_f32.shape)}"
        )
    if X0_f16.dtype != torch.float16 or W0_f16.dtype != torch.float16:
        raise ValueError(
            f"Expected float16 inputs, got X0={X0_f16.dtype}, W0={W0_f16.dtype}"
        )
    if C_f32.dtype != torch.float32:
        raise ValueError(f"Expected float32 output C_f32, got {C_f32.dtype}")
    if not (X0_f16.is_contiguous() and W0_f16.is_contiguous() and C_f32.is_contiguous()):
        raise ValueError("X0_f16, W0_f16, and C_f32 must be contiguous.")

    M, K = int(X0_f16.shape[0]), int(X0_f16.shape[1])
    N, K_w = int(W0_f16.shape[0]), int(W0_f16.shape[1])
    if K_w != K or tuple(C_f32.shape) != (M, N):
        raise ValueError(
            f"Incompatible GEMM shapes: X0={tuple(X0_f16.shape)}, "
            f"W0={tuple(W0_f16.shape)}, C={tuple(C_f32.shape)}"
        )

    dev = X0_f16.device
    if lib is None:
        lib = load_ipcg_cuda_lib(_cached_arch_for_device(dev))
    with torch.cuda.device(dev):
        stream_ptr = int(torch.cuda.current_stream(dev).cuda_stream)
        rc = int(
            lib.tsjit_cublas_scaled_fp16_gemm(
                ctypes.c_int32(M),
                ctypes.c_int32(N),
                ctypes.c_int32(K),
                ctypes.c_void_p(X0_f16.data_ptr()),
                ctypes.c_void_p(W0_f16.data_ptr()),
                ctypes.c_void_p(C_f32.data_ptr()),
                ctypes.c_float(alpha),
                ctypes.c_float(beta),
                ctypes.c_void_p(stream_ptr),
            )
        )
    if rc != 0:
        raise RuntimeError(f"tsjit_cublas_scaled_fp16_gemm failed with code {rc}")
