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

"""Unified public TSFM compiler entrypoint ``tsjit.compile``."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - optional in pure-CPU environments
    torch = None  # type: ignore[assignment]

from .chronos_pass import (
    Chronos2Compiler,
    ChronosBoltCompiler,
    _get_chronos2_output_patch_size,
    is_chronos2_model,
    is_chronos_bolt_model,
)
from .generic_pass import GenericTSFMCompiler
from .guards import (
    CausalFloat64DCStripper,
    ParityAuditReport,
    StrictParityValidator,
    tf32_precision_context,
)
from .toto2_pass import Toto2CompiledEngine, _run_uncontracted_eager_toto2, is_toto2_model


@dataclass(frozen=True)
class CompileConfig:
    """Declarative configuration for ``tsjit.compile``."""

    horizon: int = 16
    context_length: int = 512
    precision: str = "a10-auto"
    mode: str = "exact"
    batch_buckets: tuple[int, ...] | None = None
    dc_strip: bool = True
    verify: bool = False
    verify_tol: float = 0.05
    use_cuda_graphs: bool = True


_PRECISION_ALIASES: dict[str, str] = {
    "a10-auto": "auto",
    "auto": "auto",
    "crest-guarded": "auto",
    "scaled-fp16": "scaled-fp16",
    "scaled_fp16": "scaled-fp16",
    "fp16x3": "fp16x3",
    "fp16x3-ozaki": "fp16x3",
    "ozaki": "fp16x3",
    "strict": "fp16x3",
    "strict-fp32-parity": "fp16x3",
    "fast-tf32": "tf32",
    "fast_tf32": "tf32",
    "tf32": "tf32",
    "fp32": "fp32",
}


def normalize_precision(precision: str, mode: str) -> str:
    """Resolve user precision aliases while honoring ``mode='exact'`` invariants."""
    key = precision.strip().lower()
    if key not in _PRECISION_ALIASES:
        raise ValueError(
            f"Unsupported precision {precision!r}. Choose from {sorted(_PRECISION_ALIASES)}."
        )
    resolved = _PRECISION_ALIASES[key]
    if mode not in ("exact", "fast", "fp32"):
        raise ValueError(f"Unsupported mode {mode!r}. Expected 'exact', 'fast', or 'fp32'.")
    if mode == "fp32":
        return "fp32"
    return resolved


def resolve_family_precision(precision: str, mode: str, family: str) -> str:
    """Resolve family-aware precision upholding the zero-fail ``a10-auto`` contract.

    - On ``toto2`` under ``mode='exact'``, ``a10-auto`` / ``auto`` escalates trunk
      linears to ``fp16x3`` because Toto-2.0's ``asinh``/``sinh`` scaling amplifies
      10-bit TF32 rounding under 25%-NaN inputs; explicit ``precision='fast-tf32'``
      / ``'tf32'`` or ``mode='fast'`` retains maximum-throughput TF32 execution.
    - On ``chronos-2``, ``chronos-bolt``, and ``generic``, ``a10-auto`` retains
      crest-guarded ``auto`` (native TF32 for ``crest_w <= 12``, selective ``fp16x3``
      for ``crest_w > 12``), which passes 0/512 strict FP32 parity across all stress slices.
    """
    resolved = normalize_precision(precision, mode)
    if family == "toto2" and mode == "exact" and resolved == "auto":
        return "fp16x3"
    return resolved


class CompiledTSFM:
    """Unified handle returned by ``tsjit.compile(...)``."""

    def __init__(
        self,
        backend: Any,
        *,
        family: str,
        precision_requested: str,
        precision_resolved: str,
        mode: str,
        parity_report: ParityAuditReport | None = None,
    ) -> None:
        self._backend = backend
        self.family = family
        self.precision_requested = precision_requested
        self.precision_resolved = precision_resolved
        self.mode = mode
        self.parity_report = parity_report

    def predict(self, x: Any, **kwargs: Any) -> Any:
        """Return point forecast ``(B, H)``."""
        return self._backend.predict(x, **kwargs)

    def predict_quantiles(self, x: Any, **kwargs: Any) -> Any:
        """Return monotonic quantile forecast ``(B, Q, H)``."""
        return self._backend.predict_quantiles(x, **kwargs)

    def warmup(self, buckets: Sequence[int] | None = None) -> list[int]:
        """Eagerly capture CUDA Graphs for requested batch buckets."""
        backend_warmup = getattr(self._backend, "warmup", None)
        if callable(backend_warmup):
            return backend_warmup(buckets=buckets)
        runner = getattr(self._backend, "graph_runner", None)
        if runner is None:
            return []
        return runner.warmup(buckets=buckets)

    def report(self) -> dict[str, Any]:
        """Return structured compilation, pass, precision, graph, and parity telemetry."""
        base = dict(self._backend.report())
        base["family"] = self.family
        base["precision_requested"] = self.precision_requested
        base["precision_resolved"] = self.precision_resolved
        base["mode"] = self.mode
        if self.parity_report is not None:
            base["parity_audit"] = {
                "passed": self.parity_report.passed,
                "num_series": self.parity_report.num_series,
                "num_failed": self.parity_report.num_failed,
                "max_scaled_err": self.parity_report.max_scaled_err,
                "p99_scaled_err": self.parity_report.p99_scaled_err,
                "nonfinite_mismatches": self.parity_report.nonfinite_mismatches,
                "tol": self.parity_report.tol,
            }
        return base

    def __call__(self, x: Any, **kwargs: Any) -> Any:
        return self.predict_quantiles(x, **kwargs)


def _build_unmodified_eager_ref(
    model: Any,
    family: str,
    *,
    horizon: int,
    dc_strip: bool,
    quantiles: Sequence[float] | None = None,
    quantile_indices: Sequence[int] | None = None,
) -> Callable[[Any], Any]:
    """Build an unmodified strict-FP32 eager reference (zero structural or precision passes)."""
    stripper = CausalFloat64DCStripper(enabled=dc_strip)

    if family == "toto2":
        q_lvls = tuple(Toto2CompiledEngine.DEFAULT_QUANTILES if quantiles is None else quantiles)

        def _ref_toto(x_one: Any) -> Any:
            return _run_uncontracted_eager_toto2(
                model, x_one, horizon=horizon, quantiles=q_lvls, dc_strip=dc_strip
            )

        return _ref_toto

    inner = (
        model.model
        if (hasattr(model, "model") and isinstance(model.model, torch.nn.Module))
        else model
    )

    if family == "chronos-bolt":

        def _ref_bolt(x_one: Any) -> Any:
            x_s, dc_f64 = stripper.strip(x_one)
            with tf32_precision_context("fp32"):
                out = inner(context=x_s)
            q = out.quantile_preds if hasattr(out, "quantile_preds") else out
            q = q[:, :, :horizon].sort(dim=1).values
            return stripper.restore(q, dc_f64).sort(dim=1).values

        return _ref_bolt

    if family == "chronos-2":
        import inspect

        fwd_params = set(inspect.signature(inner.forward).parameters.keys())
        patch_len = _get_chronos2_output_patch_size(inner)
        n_patches = max(1, (horizon + patch_len - 1) // patch_len)
        q_idx = None if quantile_indices is None else tuple(int(i) for i in quantile_indices)

        def _ref_c2(x_one: Any) -> Any:
            x_s, dc_f64 = stripper.strip(x_one)
            B = x_s.shape[0]
            kw: dict[str, Any] = {}
            if "context" in fwd_params:
                kw["context"] = x_s
            if "group_ids" in fwd_params:
                kw["group_ids"] = torch.arange(B, dtype=torch.long, device=x_s.device)
            if "num_output_patches" in fwd_params:
                kw["num_output_patches"] = n_patches
            with tf32_precision_context("fp32"):
                out = inner(**kw) if kw else inner(x_s)
            q = out.quantile_preds if hasattr(out, "quantile_preds") else out
            if q_idx is not None:
                idx_t = torch.tensor(q_idx, dtype=torch.long, device=q.device)
                q = q.index_select(1, idx_t)
            q = q[:, :, :horizon].sort(dim=1).values
            return stripper.restore(q, dc_f64).sort(dim=1).values

        return _ref_c2

    def _ref_generic(x_one: Any) -> Any:
        x_s, dc_f64 = stripper.strip(x_one)
        with tf32_precision_context("fp32"):
            out = inner(x_s)
        if isinstance(out, tuple):
            out = out[0]
        if out.ndim >= 2 and out.shape[-1] > horizon:
            out = out[..., :horizon]
        y_rest = stripper.restore(out, dc_f64)
        if y_rest.ndim == 2:
            return y_rest.unsqueeze(1)
        return y_rest.sort(dim=1).values

    return _ref_generic


def _detect_family(model: Any) -> str:
    """Auto-detect TSFM family, checking ``chronos-2`` strictly BEFORE ``chronos-bolt``."""
    if is_toto2_model(model):
        return "toto2"
    if is_chronos2_model(model):
        return "chronos-2"
    if is_chronos_bolt_model(model):
        return "chronos-bolt"
    return "generic"


def compile(
    model: Any,
    *,
    config: CompileConfig | None = None,
    horizon: int = 16,
    context_length: int = 512,
    precision: str = "a10-auto",
    mode: str = "exact",
    batch_buckets: Sequence[int] | None = None,
    dc_strip: bool = True,
    verify: bool = False,
    verify_samples: Any | None = None,
    verify_tol: float = 0.05,
    use_cuda_graphs: bool = True,
    **kwargs: Any,
) -> CompiledTSFM:
    """Compile a public time-series foundation model or PyTorch forecaster.

    Auto-detects:
    - ``Toto2Model`` / ``Toto`` / ``TotoBackbone`` (4m / 22m / 313m)
    - ``Chronos2Pipeline`` / ``Chronos2Model``
    - ``ChronosBoltPipeline`` / ``ChronosBoltModelForForecasting``
    - Any arbitrary ``torch.nn.Module`` forecaster
    """
    if torch is None:
        raise RuntimeError("tsjit.compile requires PyTorch.")

    if config is not None:
        horizon = config.horizon
        context_length = config.context_length
        precision = config.precision
        mode = config.mode
        batch_buckets = config.batch_buckets
        dc_strip = config.dc_strip
        verify = config.verify
        verify_tol = config.verify_tol
        use_cuda_graphs = config.use_cuda_graphs

    family = _detect_family(model)
    resolved_prec = resolve_family_precision(precision, mode, family)

    if family == "toto2":
        backend: Any = Toto2CompiledEngine(
            model,
            horizon=horizon,
            context_length=context_length,
            precision=resolved_prec,
            batch_buckets=batch_buckets,
            dc_strip=dc_strip,
            use_cuda_graphs=use_cuda_graphs,
            quantiles=kwargs.get("quantiles"),
        )
    elif family == "chronos-2":
        backend = Chronos2Compiler(
            model,
            horizon=horizon,
            context_length=context_length,
            precision=resolved_prec,
            batch_buckets=batch_buckets,
            dc_strip=dc_strip,
            quantile_indices=kwargs.get("quantile_indices"),
            use_cuda_graphs=use_cuda_graphs,
        )
    elif family == "chronos-bolt":
        backend = ChronosBoltCompiler(
            model,
            horizon=horizon,
            context_length=context_length,
            precision=resolved_prec,
            batch_buckets=batch_buckets,
            dc_strip=dc_strip,
            use_cuda_graphs=use_cuda_graphs,
        )
    else:
        backend = GenericTSFMCompiler(
            model,
            horizon=horizon,
            context_length=context_length,
            precision=resolved_prec,
            batch_buckets=batch_buckets,
            dc_strip=dc_strip,
            use_cuda_graphs=use_cuda_graphs,
        )

    parity_report: ParityAuditReport | None = None
    if verify:
        device = getattr(backend, "device", torch.device("cpu"))
        if verify_samples is None:
            g = torch.Generator(device="cpu").manual_seed(2026)
            t = torch.linspace(0.0, 12.0, context_length, dtype=torch.float64)
            base = torch.sin(t).unsqueeze(0).repeat(8, 1)
            noise = 0.1 * torch.randn((8, context_length), generator=g, dtype=torch.float64)
            x_val = (base + noise).to(device)
        else:
            x_val = torch.as_tensor(verify_samples, device=device, dtype=torch.float64)

        # Audit against the UNMODIFIED strict-FP32 eager model (verifies BOTH structural
        # compiler passes AND Crest-Guarded precision lowering!).
        ref_fn = _build_unmodified_eager_ref(
            model,
            family,
            horizon=horizon,
            dc_strip=dc_strip,
            quantiles=kwargs.get("quantiles"),
            quantile_indices=kwargs.get("quantile_indices"),
        )
        validator = StrictParityValidator(tol=verify_tol)
        parity_report = validator.audit(ref_fn, backend.predict_quantiles, x_val)

        # In mode='exact' with auto precision, escalate to FP16x3 if parity audit fails.
        if (
            (not parity_report.passed)
            and mode == "exact"
            and resolved_prec in ("auto", "scaled-fp16")
        ):
            return compile(
                model,
                horizon=horizon,
                context_length=context_length,
                precision="fp16x3",
                mode="exact",
                batch_buckets=batch_buckets,
                dc_strip=dc_strip,
                verify=True,
                verify_samples=x_val,
                verify_tol=verify_tol,
                use_cuda_graphs=use_cuda_graphs,
                **kwargs,
            )

    return CompiledTSFM(
        backend,
        family=family,
        precision_requested=precision,
        precision_resolved=resolved_prec,
        mode=mode,
        parity_report=parity_report,
    )
