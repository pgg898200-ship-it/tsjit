"""tsjit: Clean-room streaming time-series runtime and compiler.

Uses PEP 562 lazy attribute loading so package import and CLI startup stay
under 25 ms with zero eager numpy/torch/ctypes imports.
"""

from ._version import __version__, __version_info__

_LAZY_IMPORTS = {
    # Compiler API
    "compile": ("tsjit.compiler.api", "compile"),
    "CompileConfig": ("tsjit.compiler.api", "CompileConfig"),
    "CompiledTSFM": ("tsjit.compiler.api", "CompiledTSFM"),
    # Arrival & streaming runtime
    "stream": ("tsjit.runtime.arrival", "stream"),
    "GapAsyncRuntime": ("tsjit.runtime.arrival", "GapAsyncRuntime"),
    "PwlArrivalEngine": ("tsjit.runtime.arrival", "PwlArrivalEngine"),
    "SoftRevisionEngine": ("tsjit.runtime.arrival", "SoftRevisionEngine"),
    # Online calibration & sequential testing
    "OnlineACI": ("tsjit.runtime.calibration", "OnlineACI"),
    "QuantilePitEValueDetector": ("tsjit.runtime.calibration", "QuantilePitEValueDetector"),
    "EventGate": ("tsjit.runtime.calibration", "EventGate"),
    # Multi-agent streaming hub
    "MultiAgentStreamHub": ("tsjit.runtime.hub", "MultiAgentStreamHub"),
    "AgentSubscription": ("tsjit.runtime.hub", "AgentSubscription"),
    # Specification & validation
    "StreamSpec": ("tsjit.runtime.spec", "StreamSpec"),
    "SpecValidationError": ("tsjit.runtime.spec", "SpecValidationError"),
    "validate_spec": ("tsjit.runtime.spec", "validate_spec"),
    # Distributed associative fusion
    "distributed_stream": ("tsjit.distributed.async_fuse", "distributed_stream"),
    "AsyncFuseMonoid": ("tsjit.distributed.async_fuse", "AsyncFuseMonoid"),
    # CUDA/Triton scaled FP16 kernels
    "scaled_fp16_linear": ("tsjit.cuda.scaled_fp16_triton", "scaled_fp16_linear"),
    "fp16x3_linear": ("tsjit.cuda.scaled_fp16_triton", "fp16x3_linear"),
    "CrestGuardedLinear": ("tsjit.cuda.scaled_fp16_triton", "CrestGuardedLinear"),
}

__all__ = [
    "__version__",
    "__version_info__",
    *_LAZY_IMPORTS.keys(),
]


def __getattr__(name: str):
    target = _LAZY_IMPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'tsjit' has no attribute {name!r}")
    mod_name, attr_name = target
    mod = __import__(mod_name, fromlist=[attr_name])
    val = getattr(mod, attr_name)
    globals()[name] = val
    return val


def __dir__():
    return sorted(set(globals().keys()) | set(__all__))
