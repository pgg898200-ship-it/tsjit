"""On-demand C11 compiler & ctypes loader for tsjit native engines.

Compiles tsjit_arrive.c, tsjit_schur.c, and tsjit_arena.c with strict
`-std=c11 -O3 -Wall -Wextra -Werror` into a content-hashed cache directory,
supporting macOS ARM64 (Apple clang / NEON) and Linux x86-64 (gcc/clang / AVX2),
plus optional ASan+UBSan and TSan builds for verification.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

NATIVE_DIR = Path(__file__).resolve().parent
C_SOURCES = ("tsjit_arrive.c", "tsjit_schur.c", "tsjit_arena.c")
C_HEADERS = ("tsjit_common.h",)

# Status codes matching tsjit_common.h
TSJIT_OK = 0
TSJIT_ERR_BAD_ARG = -1
TSJIT_ERR_OOB = -2
TSJIT_ERR_SPIN_TIMEOUT = -3
TSJIT_ERR_OVERFLOW = -4
TSJIT_ERR_ALLOC = -5

# Causal arrival marks matching tsjit_common.h
MARK_OK = 0
MARK_NAN_STALE = 1
MARK_LAST_VALID = 2
MARK_ZERO = 3
MARK_DUP_STEP = 4
MARK_OOO_STEP = 5
MARK_INVALID_KEY = 6

MARK_NAMES = {
    MARK_OK: "OK",
    MARK_NAN_STALE: "NAN_STALE",
    MARK_LAST_VALID: "LAST_VALID",
    MARK_ZERO: "ZERO",
    MARK_DUP_STEP: "DUP_STEP",
    MARK_OOO_STEP: "OOO_STEP",
    MARK_INVALID_KEY: "INVALID_KEY",
}


class CTelemetry(ctypes.Structure):
    _fields_ = [
        ("total_arrivals", ctypes.c_uint64),
        ("valid_arrivals", ctypes.c_uint64),
        ("dup_rejects", ctypes.c_uint64),
        ("ooo_rejects", ctypes.c_uint64),
        ("nan_stale_count", ctypes.c_uint64),
        ("pwl_fast_hits", ctypes.c_uint64),
        ("pwl_cold_hits", ctypes.c_uint64),
        ("pwl_fallback_hits", ctypes.c_uint64),
        ("roll_refreshes", ctypes.c_uint64),
        ("last_step", ctypes.c_int64),
        ("last_key", ctypes.c_int64),
        ("last_abs_surprise", ctypes.c_double),
    ]

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name, _ in self._fields_}


class CArriveOut(ctypes.Structure):
    _fields_ = [
        ("q", ctypes.c_double * 9),
        ("abs_sg", ctypes.c_double),
        ("rolling_mean", ctypes.c_double),
        ("rolling_std", ctypes.c_double),
        ("eff_scale", ctypes.c_double),
        ("age", ctypes.c_int64),
        ("mark", ctypes.c_int32),
        ("pwl_path", ctypes.c_int32),
        ("family_weights", ctypes.c_double * 3),
    ]


class CEwmaMonoid(ctypes.Structure):
    _fields_ = [
        ("n", ctypes.c_uint64),
        ("dim", ctypes.c_uint32),
        ("_pad", ctypes.c_uint32),
        ("w", ctypes.c_double),
        ("mean", ctypes.POINTER(ctypes.c_double)),
        ("cov", ctypes.POINTER(ctypes.c_double)),
    ]


def find_c_compiler() -> str:
    """Locate a C11 compiler (prefer clang on macOS, gcc/clang on Linux)."""
    for env_key in ("TSJIT_CC", "CC"):
        cc = os.environ.get(env_key)
        if cc:
            resolved = shutil.which(cc)
            if resolved:
                return resolved
    candidates = ("clang", "gcc", "cc") if sys.platform == "darwin" else ("gcc", "clang", "cc")
    for cand in candidates:
        resolved = shutil.which(cand)
        if resolved:
            return resolved
    raise RuntimeError("No C11 compiler found in TSJIT_CC, CC, or PATH")


def _default_cache_dir() -> Path:
    env_dir = os.environ.get("TSJIT_CACHE_DIR")
    if env_dir:
        base = Path(env_dir)
    else:
        base = Path.home() / ".cache" / "tsjit" / "native"
    try:
        base.mkdir(parents=True, exist_ok=True)
        return base
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "tsjit_native_cache"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def _linux_has_avx2_fma() -> bool:
    if sys.platform != "linux" or platform.machine().lower() not in ("x86_64", "amd64"):
        return False
    try:
        txt = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    for line in txt.splitlines():
        if line.startswith("flags"):
            flags = set(line.split())
            return "avx2" in flags and "fma" in flags
    return False


def _compile_flags(sanitizers: str | None, *, shared: bool) -> list[str]:
    flags = ["-std=c11", "-Wall", "-Wextra", "-Werror", f"-I{NATIVE_DIR}"]
    if sanitizers is None:
        flags.append("-O3")
        if _linux_has_avx2_fma():
            flags.extend(["-mavx2", "-mfma"])
    else:
        san = sanitizers.strip().lower()
        flags.extend(["-O1", "-g", "-fno-omit-frame-pointer"])
        if san in ("asan,ubsan", "ubsan,asan", "asan+ubsan"):
            flags.extend(["-fsanitize=address,undefined", "-fno-sanitize-recover=all"])
        elif san == "asan":
            flags.append("-fsanitize=address")
        elif san == "ubsan":
            flags.extend(["-fsanitize=undefined", "-fno-sanitize-recover=all"])
        elif san == "tsan":
            flags.append("-fsanitize=thread")
            if "gcc" in Path(find_c_compiler()).name.lower():
                flags.append("-Wno-tsan")
        else:
            raise ValueError(f"Unsupported sanitizer mode: {sanitizers!r}")

    if shared:
        flags.append("-fPIC")
        if sys.platform == "darwin":
            flags.append("-dynamiclib")
        else:
            flags.append("-shared")
    return flags


def _link_flags() -> list[str]:
    if sys.platform == "darwin":
        return ["-pthread"]
    return ["-pthread", "-lm"]


def _content_hash(cc: str, flags: list[str]) -> str:
    h = hashlib.sha256()
    h.update(f"{sys.platform}|{platform.machine()}|{cc}|{' '.join(flags)}".encode())
    for name in (*C_HEADERS, *C_SOURCES):
        p = NATIVE_DIR / name
        h.update(name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def build_native(
    *,
    sanitizers: str | None = None,
    force: bool = False,
    cache_dir: Path | None = None,
) -> Path:
    """Compile and cache libtsjit_native.{dylib,so}, returning its path."""
    cc = find_c_compiler()
    flags = _compile_flags(sanitizers, shared=True)
    digest = _content_hash(cc, flags)
    ext = "dylib" if sys.platform == "darwin" else "so"
    san_tag = "" if sanitizers is None else f"_{sanitizers.replace(',', '_')}"
    out_dir = cache_dir if cache_dir is not None else _default_cache_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"libtsjit_native{san_tag}_{digest}.{ext}"

    if target.exists() and not force:
        return target

    sources = [str(NATIVE_DIR / name) for name in C_SOURCES]
    tmp_target = out_dir / f".tmp_{os.getpid()}_{threading.get_ident()}_{target.name}"
    cmd = [cc, *flags, *sources, "-o", str(tmp_target), *_link_flags()]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        if tmp_target.exists():
            tmp_target.unlink(missing_ok=True)
        raise RuntimeError(
            f"Native C11 build failed (rc={proc.returncode}):\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stderr:\n{proc.stderr}"
        )
    os.replace(tmp_target, target)
    return target


def build_c_executable(
    harness_c: Path,
    out_bin: Path,
    *,
    sanitizers: str | None = None,
) -> Path:
    """Compile a standalone C11 executable linked against all tsjit native sources."""
    cc = find_c_compiler()
    flags = _compile_flags(sanitizers, shared=False)
    sources = [str(NATIVE_DIR / name) for name in C_SOURCES] + [str(harness_c)]
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    cmd = [cc, *flags, *sources, "-o", str(out_bin), *_link_flags()]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"C11 executable build failed (rc={proc.returncode}):\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stderr:\n{proc.stderr}"
        )
    return out_bin


_CACHED_LIB: ctypes.CDLL | None = None
_CACHED_LIB_PATH: Path | None = None
_BUILD_LOCK = threading.Lock()


def _configure_signatures(lib: ctypes.CDLL) -> None:
    c_void_p = ctypes.c_void_p
    c_i32 = ctypes.c_int32
    c_u32 = ctypes.c_uint32
    c_i64 = ctypes.c_int64
    c_u64 = ctypes.c_uint64
    c_f64 = ctypes.c_double
    p_i32 = ctypes.POINTER(c_i32)
    p_u32 = ctypes.POINTER(c_u32)
    p_i64 = ctypes.POINTER(c_i64)
    p_u64 = ctypes.POINTER(c_u64)
    p_f64 = ctypes.POINTER(c_f64)

    # Platform & utilities
    lib.tsjit_cacheline_size.argtypes = []
    lib.tsjit_cacheline_size.restype = c_i32

    lib.tsjit_simd_isa.argtypes = []
    lib.tsjit_simd_isa.restype = ctypes.c_char_p

    lib.tsjit_check_bounds_i64.argtypes = [c_i64, c_i64, c_i64]
    lib.tsjit_check_bounds_i64.restype = c_i32

    lib.tsjit_safe_age_i64_abi.argtypes = [c_i64, c_i64]
    lib.tsjit_safe_age_i64_abi.restype = c_i64

    lib.tsjit_sort9_abi.argtypes = [p_f64]
    lib.tsjit_sort9_abi.restype = None

    lib.tsjit_pwl_hot_row_bytes.argtypes = []
    lib.tsjit_pwl_hot_row_bytes.restype = c_i32

    lib.tsjit_seqlock_slot_bytes.argtypes = []
    lib.tsjit_seqlock_slot_bytes.restype = c_i32

    # tsjit_arrive.c
    lib.tsjit_arrive_create.argtypes = [c_i64, c_i64, c_f64, c_i64]
    lib.tsjit_arrive_create.restype = c_void_p

    lib.tsjit_arrive_set_params.argtypes = [c_void_p, c_f64, c_i64]
    lib.tsjit_arrive_set_params.restype = c_i32

    lib.tsjit_arrive_destroy.argtypes = [c_void_p]
    lib.tsjit_arrive_destroy.restype = None

    lib.tsjit_arrive_publish_forecast.argtypes = [
        c_void_p,
        c_i64,
        c_i64,
        p_f64,
        p_f64,
        c_f64,
        p_f64,
        p_f64,
        p_f64,
    ]
    lib.tsjit_arrive_publish_forecast.restype = c_i32

    lib.tsjit_arrive_step_pwl.argtypes = [c_void_p, c_i64, c_i64, c_f64, ctypes.POINTER(CArriveOut)]
    lib.tsjit_arrive_step_pwl.restype = c_i32

    lib.tsjit_arrive_step_soft_v2.argtypes = [
        c_void_p,
        c_i64,
        c_i64,
        c_f64,
        c_i32,
        p_f64,
        p_f64,
        c_i32,
        p_f64,
        p_f64,
        c_i32,
        p_f64,
        p_f64,
        c_f64,
        c_f64,
        c_f64,
        ctypes.POINTER(CArriveOut),
    ]
    lib.tsjit_arrive_step_soft_v2.restype = c_i32

    lib.tsjit_arrive_get_rolling.argtypes = [c_void_p, c_i64, p_f64, p_f64, p_f64, p_i64, p_i64]
    lib.tsjit_arrive_get_rolling.restype = c_i32

    lib.tsjit_read_telemetry.argtypes = [c_void_p, ctypes.POINTER(CTelemetry)]
    lib.tsjit_read_telemetry.restype = c_i32

    lib.tsjit_seqlock_stress_test.argtypes = [c_i32, c_i32, c_u64, c_i32, p_u64, p_u64, p_u64]
    lib.tsjit_seqlock_stress_test.restype = c_i32

    # tsjit_schur.c
    lib.tsjit_schur_create.argtypes = [c_i32, c_i32, c_f64, c_f64, c_f64]
    lib.tsjit_schur_create.restype = c_void_p

    lib.tsjit_schur_destroy.argtypes = [c_void_p]
    lib.tsjit_schur_destroy.restype = None

    lib.tsjit_schur_update.argtypes = [c_void_p, p_f64, p_f64, p_f64, p_f64]
    lib.tsjit_schur_update.restype = c_i32

    lib.tsjit_schur_predict.argtypes = [c_void_p, p_f64, p_f64]
    lib.tsjit_schur_predict.restype = c_i32

    lib.tsjit_schur_solve_system.argtypes = [c_i32, p_f64, p_f64, p_f64, p_i32, p_f64, p_f64]
    lib.tsjit_schur_solve_system.restype = c_i32

    lib.tsjit_schur_solve_shared_core.argtypes = [
        c_i32,
        c_i32,
        c_i32,
        p_f64,
        p_f64,
        p_f64,
        p_f64,
        p_f64,
        p_f64,
        p_i32,
        p_i32,
        p_f64,
    ]
    lib.tsjit_schur_solve_shared_core.restype = c_i32

    lib.tsjit_schur_check_higham_guard.argtypes = [c_i32, p_f64, p_i32, p_f64, p_f64]
    lib.tsjit_schur_check_higham_guard.restype = c_i32

    # tsjit_arena.c
    lib.tsjit_arena_create.argtypes = [c_i32, c_u32, c_u32]
    lib.tsjit_arena_create.restype = c_void_p

    lib.tsjit_arena_destroy.argtypes = [c_void_p]
    lib.tsjit_arena_destroy.restype = None

    lib.tsjit_arena_add_agent.argtypes = [c_void_p, p_i32, c_i32, c_f64, p_i32]
    lib.tsjit_arena_add_agent.restype = c_i32

    lib.tsjit_arena_tombstone_agent.argtypes = [c_void_p, c_i32]
    lib.tsjit_arena_tombstone_agent.restype = c_i32

    lib.tsjit_arena_agent_status.argtypes = [c_void_p, c_i32]
    lib.tsjit_arena_agent_status.restype = c_i32

    lib.tsjit_arena_ingest_tick.argtypes = [c_void_p, c_i64, p_f64, c_i32]
    lib.tsjit_arena_ingest_tick.restype = c_i32

    lib.tsjit_arena_publish_agent_output.argtypes = [c_void_p, c_i32, c_i64, p_f64, c_f64, c_i32]
    lib.tsjit_arena_publish_agent_output.restype = c_i32

    lib.tsjit_arena_read_agent_snapshot.argtypes = [
        c_void_p,
        c_i32,
        p_f64,
        p_f64,
        p_i32,
        p_i64,
        p_u64,
    ]
    lib.tsjit_arena_read_agent_snapshot.restype = c_i32

    lib.tsjit_arena_get_channel_stats.argtypes = [c_void_p, c_i32, p_f64, p_f64, p_f64, p_u64]
    lib.tsjit_arena_get_channel_stats.restype = c_i32

    lib.tsjit_arena_get_agent_cov.argtypes = [c_void_p, c_i32, c_i32, p_f64, p_f64, p_u64, p_i32]
    lib.tsjit_arena_get_agent_cov.restype = c_i32

    lib.tsjit_arena_force_resize.argtypes = [c_void_p, c_u32]
    lib.tsjit_arena_force_resize.restype = c_i32

    lib.tsjit_arena_ebr_stats.argtypes = [c_void_p, p_u32, p_u64, p_u64, p_u64]
    lib.tsjit_arena_ebr_stats.restype = c_i32

    lib.tsjit_ewma_monoid_merge.argtypes = [
        c_f64,
        ctypes.POINTER(CEwmaMonoid),
        ctypes.POINTER(CEwmaMonoid),
        ctypes.POINTER(CEwmaMonoid),
    ]
    lib.tsjit_ewma_monoid_merge.restype = c_i32

    lib.tsjit_ewma_monoid_merge_tree.argtypes = [
        c_f64,
        ctypes.POINTER(CEwmaMonoid),
        c_i32,
        ctypes.POINTER(CEwmaMonoid),
    ]
    lib.tsjit_ewma_monoid_merge_tree.restype = c_i32

    lib.tsjit_arena_ebr_stress_test.argtypes = [c_i32, c_i32, p_u64, p_u64, p_u64]
    lib.tsjit_arena_ebr_stress_test.restype = c_i32


def load_native_lib(*, force: bool = False) -> ctypes.CDLL:
    """Build (if needed) and load the cached C11 shared library."""
    global _CACHED_LIB, _CACHED_LIB_PATH
    if _CACHED_LIB is not None and not force:
        return _CACHED_LIB
    with _BUILD_LOCK:
        if _CACHED_LIB is not None and not force:
            return _CACHED_LIB
        lib_path = build_native(force=force)
        lib = ctypes.CDLL(str(lib_path))
        _configure_signatures(lib)
        _CACHED_LIB = lib
        _CACHED_LIB_PATH = lib_path
        return lib


def native_info() -> dict[str, Any]:
    """Return metadata about the compiled native C11 backend."""
    lib = load_native_lib()
    return {
        "available": True,
        "library_path": str(_CACHED_LIB_PATH),
        "compiler": find_c_compiler(),
        "simd_isa": lib.tsjit_simd_isa().decode("ascii"),
        "cacheline_bytes": int(lib.tsjit_cacheline_size()),
        "pwl_hot_row_bytes": int(lib.tsjit_pwl_hot_row_bytes()),
        "seqlock_slot_bytes": int(lib.tsjit_seqlock_slot_bytes()),
    }
