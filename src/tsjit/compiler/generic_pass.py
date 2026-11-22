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

"""Generic compiler pass for arbitrary PyTorch time-series forecasting modules."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - optional in pure-CPU environments
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

from ..cuda.scaled_fp16_triton import is_linear_module, replace_linears_with_crest_guarded
from .graph_runner import InductorOrGraphRunner
from .guards import CausalFloat64DCStripper, normalize_precision_alias


class GenericTSFMCompiler:
    """Compile any arbitrary ``torch.nn.Module`` time-series forecaster.

    Applies:
    1. ``CausalFloat64DCStripper`` (optional, default enabled).
    2. ``CrestGuardedLinear`` replacement on interior ``nn.Linear`` layers while
       strictly preserving the first/last linear layers, norms, embeddings, and
       projection heads in exact FP32 SIMT.
    3. ``BucketedCudaGraphRunner`` on CUDA or ``torch.inference_mode()`` on CPU/MPS.
    """

    def __init__(
        self,
        model: Any,
        *,
        horizon: int | None = None,
        context_length: int = 512,
        precision: str = "auto",
        batch_buckets: Sequence[int] | None = None,
        dc_strip: bool = True,
        use_cuda_graphs: bool = True,
        copy_model: bool = True,
    ) -> None:
        if torch is None:
            raise RuntimeError("GenericTSFMCompiler requires PyTorch.")
        if not isinstance(model, nn.Module):
            raise TypeError(f"Expected torch.nn.Module, got {type(model)}")

        self.model = copy.deepcopy(model) if copy_model else model
        self.model.eval()
        self.horizon = None if horizon is None else int(horizon)
        self.context_length = int(context_length)
        precision = normalize_precision_alias(precision)
        self.precision = precision
        self.dc_stripper = CausalFloat64DCStripper(enabled=dc_strip)

        # Preserve first and last nn.Linear layers plus norm/embed/head layers in FP32 SIMT.
        linear_names = [
            name for name, mod in self.model.named_modules() if is_linear_module(mod)
        ]
        boundary_names: set[str] = set()
        if linear_names:
            boundary_names.add(linear_names[0])
            boundary_names.add(linear_names[-1])

        self.precision_summary: dict[str, Any] = {
            "mode": precision,
            "replaced": 0,
            "escalated_w": 0,
        }
        if precision in ("auto", "scaled-fp16", "fp16x3"):
            self.precision_summary = replace_linears_with_crest_guarded(
                self.model,
                mode=precision,
                graph_safe=True,
                skip_names=(
                    "norm",
                    "embed",
                    "head",
                    "proj_in",
                    "proj_out",
                    "input_proj",
                    "output_proj",
                ),
                predicate=lambda full_name, _mod: full_name not in boundary_names,
            )

        first_param = next(self.model.parameters(), None)
        self.device = first_param.device if first_param is not None else torch.device("cpu")

        has_crest_guarded = int(self.precision_summary.get("replaced", 0)) > 0
        use_inductor = (
            self.device.type == "cuda"
            and use_cuda_graphs
            and (not has_crest_guarded)
            and self.precision in ("auto", "tf32", "fp32")
        )
        self.graph_runner = InductorOrGraphRunner(
            self._forward_e2e,
            sample_shape=(self.context_length,),
            buckets=batch_buckets,
            device=self.device,
            dtype=torch.float32,
            enabled=(use_cuda_graphs and self.device.type == "cuda"),
            use_inductor=use_inductor,
            precision=self.precision,
        )

    def _forward_stripped_f32(self, x_f32: Any) -> Any:
        out = self.model(x_f32)
        if isinstance(out, tuple):
            out = out[0]
        if self.horizon is not None and out.ndim >= 2 and out.shape[-1] > self.horizon:
            out = out[..., : self.horizon]
        return out.contiguous()

    def _forward_e2e(self, x_2d: Any) -> Any:
        x_s, dc_f64 = self.dc_stripper.strip(x_2d)
        y_s = self._forward_stripped_f32(x_s)
        return self.dc_stripper.restore(y_s, dc_f64).contiguous()

    def _prepare_input(self, x: Any) -> Any:
        if (
            isinstance(x, torch.Tensor)
            and x.device == self.device
            and x.dtype in (torch.float32, torch.float64)
        ):
            return x.unsqueeze(0) if x.ndim == 1 else x
        x_t = torch.as_tensor(x, device=self.device, dtype=torch.float64)
        return x_t.unsqueeze(0) if x_t.ndim == 1 else x_t

    def predict_quantiles(self, x: Any) -> Any:
        """Predict quantiles ``(B, Q, H)`` (promotes point ``(B, H)`` to ``(B, 1, H)``)."""
        y_restored = self.graph_runner(self._prepare_input(x))
        if y_restored.ndim == 2:
            return y_restored.unsqueeze(1)
        return y_restored.sort(dim=1).values

    def predict(self, x: Any) -> Any:
        """Predict point forecast ``(B, H)`` (median quantile if multi-quantile output)."""
        y_restored = self.graph_runner(self._prepare_input(x))
        if y_restored.ndim == 3:
            q_sorted = y_restored.sort(dim=1).values
            return q_sorted[:, q_sorted.shape[1] // 2, :]
        return y_restored

    def report(self) -> dict[str, Any]:
        return {
            "pass": "generic",
            "horizon": self.horizon,
            "context_length": self.context_length,
            "dc_strip": self.dc_stripper.enabled,
            "precision": self.precision_summary,
            "cuda_graphs": self.graph_runner.stats(),
        }


def compile_generic(
    model: Any,
    *,
    horizon: int | None = None,
    context_length: int = 512,
    precision: str = "auto",
    batch_buckets: Sequence[int] | None = None,
    dc_strip: bool = True,
    use_cuda_graphs: bool = True,
) -> GenericTSFMCompiler:
    """Compile an arbitrary ``torch.nn.Module`` forecaster."""
    return GenericTSFMCompiler(
        model,
        horizon=horizon,
        context_length=context_length,
        precision=precision,
        batch_buckets=batch_buckets,
        dc_strip=dc_strip,
        use_cuda_graphs=use_cuda_graphs,
    )
