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

"""Compiler passes for Chronos-Bolt and Chronos-2 foundation models.

- ``ChronosBoltCompiler``: exact dead output-head row slicing for horizon < 64,
  CausalFloat64DCStripper, CrestGuardedLinear, and BucketedCudaGraphRunner.
- ``Chronos2Compiler``: plan-gated singleton ``GroupSelfAttention`` collapse
  (folding W_ov = W_o @ W_v in FP64 -> FP32 across all encoder layers,
  eliminating (B, T, D) <-> (T, B, D) transposes and [T, 1, B, B] masks when
  group_ids == arange(B)), optional quantile subset slicing,
  CausalFloat64DCStripper, CrestGuardedLinear, and BucketedCudaGraphRunner.
"""

from __future__ import annotations

import copy
import importlib
import inspect
import threading
from collections.abc import Sequence
from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - optional in pure-CPU environments
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from ..cuda.scaled_fp16_triton import replace_linears_with_crest_guarded
from .graph_runner import InductorOrGraphRunner
from .guards import CausalFloat64DCStripper, normalize_precision_alias, tf32_precision_context


def is_chronos2_model(model: Any) -> bool:
    """Detect ``Chronos2Pipeline`` or ``Chronos2Model``."""
    names = [type(model).__name__.lower()]
    inner = getattr(model, "model", None)
    if inner is not None:
        names.append(type(inner).__name__.lower())
    if any("chronos2" in n or "chronos_2" in n for n in names):
        return True
    target = inner if inner is not None else model
    cfg = getattr(target, "config", getattr(model, "config", None))
    if cfg is not None:
        mtype = str(getattr(cfg, "model_type", "")).lower()
        if "chronos2" in mtype or "chronos_2" in mtype:
            return True
    encoder = getattr(target, "encoder", None)
    if hasattr(encoder, "_construct_and_invert_group_time_mask") or hasattr(
        target, "_construct_and_invert_group_time_mask"
    ):
        return True
    modules_fn = getattr(target, "modules", None)
    if callable(modules_fn):
        for mod in modules_fn():
            if "groupselfattention" in type(mod).__name__.lower():
                return True
    if encoder is not None and hasattr(encoder, "block"):
        for blk in encoder.block:
            if "groupselfattention" in type(blk).__name__.lower():
                return True
            if hasattr(blk, "layer"):
                for sub in blk.layer:
                    if "groupselfattention" in type(sub).__name__.lower():
                        return True
    return False


def is_chronos_bolt_model(model: Any) -> bool:
    """Detect ``ChronosBoltPipeline`` or ``ChronosBoltModelForForecasting``."""
    if is_chronos2_model(model):
        return False
    names = [type(model).__name__.lower()]
    inner = getattr(model, "model", None)
    if inner is not None:
        names.append(type(inner).__name__.lower())
    if any("chronosbolt" in n or "chronos_bolt" in n for n in names):
        return True
    target = inner if inner is not None else model
    return hasattr(target, "output_patch_embedding") and hasattr(target, "chronos_config")


def _slice_linear_out_rows(linear: Any, row_indices: Any) -> Any:
    """Create a sliced ``nn.Linear`` keeping only ``row_indices`` output rows."""
    idx = row_indices.to(device=linear.weight.device, dtype=torch.long)
    new_lin = nn.Linear(
        linear.in_features,
        int(idx.numel()),
        bias=(linear.bias is not None),
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    with torch.no_grad():
        new_lin.weight.copy_(linear.weight.index_select(0, idx))
        if linear.bias is not None and new_lin.bias is not None:
            new_lin.bias.copy_(linear.bias.index_select(0, idx))
    return new_lin


def _get_chronos_bolt_meta(model: Any) -> tuple[int, list[float]]:
    ccfg = getattr(model, "chronos_config", None)
    if ccfg is None and hasattr(model, "config"):
        ccfg = getattr(model.config, "chronos_config", None)
    if isinstance(ccfg, dict):
        pred_len = int(ccfg.get("prediction_length", 64))
        quantiles = [float(q) for q in ccfg.get("quantiles", [0.1 * i for i in range(1, 10)])]
    elif ccfg is not None:
        pred_len = int(getattr(ccfg, "prediction_length", 64))
        quantiles = [float(q) for q in getattr(ccfg, "quantiles", [0.1 * i for i in range(1, 10)])]
    else:
        pred_len = 64
        quantiles = [0.1 * i for i in range(1, 10)]
    return pred_len, quantiles


class ChronosBoltCompiler:
    """Compiler for ``autogluon/chronos-bolt-*`` models."""

    def __init__(
        self,
        model: Any,
        *,
        horizon: int = 16,
        context_length: int = 512,
        precision: str = "auto",
        batch_buckets: Sequence[int] | None = None,
        dc_strip: bool = True,
        use_cuda_graphs: bool = True,
        copy_model: bool = True,
    ) -> None:
        if torch is None:
            raise RuntimeError("ChronosBoltCompiler requires PyTorch.")

        raw_model = (
            model.model if hasattr(model, "model") and isinstance(model.model, nn.Module) else model
        )
        self.model = copy.deepcopy(raw_model) if copy_model else raw_model
        self.model.eval()

        self.orig_prediction_length, self.quantiles = _get_chronos_bolt_meta(self.model)
        self.num_quantiles = len(self.quantiles)
        self.horizon = int(horizon)
        self.context_length = int(context_length)
        precision = normalize_precision_alias(precision)
        self.precision = precision
        self.dc_stripper = CausalFloat64DCStripper(enabled=dc_strip)

        # Pass 1: Exact dead output-head row slicing when horizon < orig_prediction_length (64).
        self.head_sliced = False
        self.sliced_rows_before = self.num_quantiles * self.orig_prediction_length
        self.sliced_rows_after = self.sliced_rows_before
        if 0 < self.horizon < self.orig_prediction_length:
            self._slice_output_head(self.horizon)

        # Pass 2: Crest-Guarded precision on encoder/decoder body (keep norms & patch heads FP32).
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
                    "input_patch_embedding",
                    "output_patch_embedding",
                    "norm",
                    "lm_head",
                    "embed",
                ),
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

    def _slice_output_head(self, target_h: int) -> None:
        head = getattr(self.model, "output_patch_embedding", None)
        if head is None:
            return
        P_out = self.orig_prediction_length
        Q = self.num_quantiles
        expected_out = Q * P_out

        row_idx = torch.tensor(
            [q * P_out + h for q in range(Q) for h in range(target_h)],
            dtype=torch.long,
        )

        sliced_any = False
        if isinstance(head, nn.Linear) and head.out_features == expected_out:
            self.model.output_patch_embedding = _slice_linear_out_rows(head, row_idx)
            sliced_any = True
        else:
            for attr in ("output_layer", "residual_layer"):
                sub = getattr(head, attr, None)
                if isinstance(sub, nn.Linear) and sub.out_features == expected_out:
                    setattr(head, attr, _slice_linear_out_rows(sub, row_idx))
                    sliced_any = True

        if sliced_any:
            self.head_sliced = True
            self.sliced_rows_after = Q * target_h
            ccfg = getattr(self.model, "chronos_config", None)
            if isinstance(ccfg, dict):
                ccfg["prediction_length"] = target_h
            elif ccfg is not None and hasattr(ccfg, "prediction_length"):
                ccfg.prediction_length = target_h
            if hasattr(self.model, "config") and hasattr(self.model.config, "chronos_config"):
                raw_cc = self.model.config.chronos_config
                if isinstance(raw_cc, dict):
                    raw_cc["prediction_length"] = target_h
                elif hasattr(raw_cc, "prediction_length"):
                    raw_cc.prediction_length = target_h
            if hasattr(self.model, "model_prediction_length"):
                self.model.model_prediction_length = target_h

    def _forward_stripped_f32(self, x_2d_f32: Any) -> Any:
        out = self.model(context=x_2d_f32)
        if hasattr(out, "quantile_preds"):
            q = out.quantile_preds
        elif isinstance(out, tuple):
            q = out[0]
        else:
            q = out
        # Output shape (B, Q, H) -> slice to requested horizon.
        return q[:, :, : self.horizon]

    def _forward_e2e(self, x_2d: Any) -> Any:
        x_s, dc_f64 = self.dc_stripper.strip(x_2d)
        q_s = self._forward_stripped_f32(x_s)
        return self.dc_stripper.restore(q_s, dc_f64).sort(dim=1).values.contiguous()

    def predict_quantiles(self, x: Any) -> Any:
        """Predict monotonic quantiles ``(B, Q, H)`` in FP64 with causal DC stripping."""
        if (
            isinstance(x, torch.Tensor)
            and x.device == self.device
            and x.dtype in (torch.float32, torch.float64)
        ):
            x_t = x.unsqueeze(0) if x.ndim == 1 else x
        else:
            x_t = torch.as_tensor(x, device=self.device, dtype=torch.float64)
            if x_t.ndim == 1:
                x_t = x_t.unsqueeze(0)
        return self.graph_runner(x_t)

    def predict(self, x: Any) -> Any:
        """Predict point forecast ``(B, H)`` (median quantile)."""
        q = self.predict_quantiles(x)
        return q[:, q.shape[1] // 2, :]

    def report(self) -> dict[str, Any]:
        return {
            "pass": "chronos-bolt",
            "horizon": self.horizon,
            "orig_prediction_length": self.orig_prediction_length,
            "head_sliced": self.head_sliced,
            "head_rows": f"{self.sliced_rows_before}->{self.sliced_rows_after}",
            "dc_strip": self.dc_stripper.enabled,
            "precision": self.precision_summary,
            "cuda_graphs": self.graph_runner.stats(),
        }


def _find_v_and_o_linears(gsa: Any) -> tuple[Any, Any]:
    """Locate value and output linear projections inside a Chronos-2 ``GroupSelfAttention``."""
    attn = getattr(gsa, "self_attention", getattr(gsa, "attn", gsa))
    v_lin = getattr(attn, "v", getattr(attn, "v_proj", getattr(attn, "value", None)))
    o_lin = getattr(attn, "o", getattr(attn, "o_proj", getattr(attn, "out_proj", None)))
    if not isinstance(v_lin, nn.Linear) or not isinstance(o_lin, nn.Linear):
        raise AttributeError(f"Cannot locate (v, o) nn.Linear layers on {type(gsa).__name__}")
    return v_lin, o_lin


def _resolve_attention_output_cls(gsa: Any) -> Any:
    try:
        mod = importlib.import_module(gsa.__class__.__module__)
        return getattr(mod, "AttentionOutput", None)
    except Exception:
        return None


_BaseModule = nn.Module if nn is not None else object


class _PlanGatedGroupSelfAttention(_BaseModule):  # type: ignore[misc]
    """Plan-gated replacement for Chronos-2 ``GroupSelfAttention``.

    When ``singleton_plan`` is True (independent series ``group_ids == arange(B)``),
    replaces cross-batch attention + (B, T, D) <-> (T, B, D) transposes + [T, 1, B, B]
    masking with exact fused ``hidden_states + dropout(W_ov(layer_norm(hidden_states)))``
    where ``W_ov = W_o @ W_v`` is folded offline in FP64 -> FP32.
    """

    def __init__(self, orig_gsa: Any, *, singleton_plan: bool = True) -> None:
        super().__init__()
        self.orig_gsa = orig_gsa
        self.singleton_plan = bool(singleton_plan)
        self._patch_mask_stub_active = False
        self.layer_norm = getattr(orig_gsa, "layer_norm", None)
        self.dropout = getattr(orig_gsa, "dropout", None)
        self._attn_out_cls = _resolve_attention_output_cls(orig_gsa)

        v_lin, o_lin = _find_v_and_o_linears(orig_gsa)
        if v_lin.out_features != o_lin.in_features:
            raise ValueError(
                f"GroupSelfAttention v_out={v_lin.out_features} != o_in={o_lin.in_features}"
            )
        with torch.no_grad():
            # Fold W_ov = W_o @ W_v in FP64 -> FP32 for exact mathematical parity.
            w_v64 = v_lin.weight.detach().to(torch.float64)
            w_o64 = o_lin.weight.detach().to(torch.float64)
            w_ov = (w_o64 @ w_v64).to(dtype=v_lin.weight.dtype, device=v_lin.weight.device)
            b_ov = None
            if v_lin.bias is not None or o_lin.bias is not None:
                b64 = torch.zeros(
                    o_lin.out_features, dtype=torch.float64, device=v_lin.weight.device
                )
                if v_lin.bias is not None:
                    b64 = b64 + (w_o64 @ v_lin.bias.detach().to(torch.float64))
                if o_lin.bias is not None:
                    b64 = b64 + o_lin.bias.detach().to(torch.float64)
                b_ov = b64.to(dtype=v_lin.weight.dtype, device=v_lin.weight.device)

        self.w_ov = nn.Linear(
            v_lin.in_features,
            o_lin.out_features,
            bias=(b_ov is not None),
            device=v_lin.weight.device,
            dtype=v_lin.weight.dtype,
        )
        with torch.no_grad():
            self.w_ov.weight.copy_(w_ov)
            if b_ov is not None and self.w_ov.bias is not None:
                self.w_ov.bias.copy_(b_ov)

        # Inspect original return style once if possible (annotation + lightweight probe fallback).
        self._returns_tuple = False
        self._tuple_len = 2
        sig = inspect.signature(orig_gsa.forward)
        ret_ann = str(sig.return_annotation).lower()
        if "tuple" in ret_ann:
            self._returns_tuple = True

        if self._attn_out_cls is None and not self._returns_tuple:
            try:
                probe_x = torch.zeros(
                    (1, 1, v_lin.in_features),
                    device=v_lin.weight.device,
                    dtype=v_lin.weight.dtype,
                )
                with torch.no_grad():
                    y_probe = orig_gsa(probe_x)
                if isinstance(y_probe, tuple):
                    self._returns_tuple = True
                    self._tuple_len = max(len(y_probe), 2)
                elif not isinstance(y_probe, torch.Tensor) and hasattr(y_probe, "hidden_states"):
                    self._attn_out_cls = type(y_probe)
            except Exception:
                pass

        # Inspect AttentionOutput constructor once for Inductor-friendly wrapping.
        self._attn_out_arity = -1
        self._attn_out_use_kwargs = False
        if self._attn_out_cls is not None:
            try:
                ao_params = [
                    p
                    for p in inspect.signature(self._attn_out_cls).parameters.values()
                    if p.name != "self"
                    and p.kind
                    not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                ]
                ao_names = {p.name for p in ao_params}
                if "hidden_states" in ao_names and "attn_weights" in ao_names:
                    self._attn_out_use_kwargs = True
                self._attn_out_arity = len(ao_params)
            except Exception:
                self._attn_out_arity = 0

    def forward(
        self,
        hidden_states: Any,
        attention_mask: Any | None = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Any:
        if not self.singleton_plan or output_attentions:
            if attention_mask is not None or "group_time_mask" not in kwargs:
                return self.orig_gsa(
                    hidden_states,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                    **kwargs,
                )
            return self.orig_gsa(
                hidden_states,
                output_attentions=output_attentions,
                **kwargs,
            )

        normed = self.layer_norm(hidden_states) if self.layer_norm is not None else hidden_states
        proj = self.w_ov(normed)
        if self.dropout is not None:
            proj = self.dropout(proj)

        mask = attention_mask if attention_mask is not None else kwargs.get("group_time_mask", None)
        if mask is not None and isinstance(mask, torch.Tensor) and hidden_states.ndim == 3:
            if (
                self._patch_mask_stub_active
                and mask.ndim == 2
                and mask.shape[0] == hidden_states.shape[0]
                and mask.shape[1] == hidden_states.shape[1]
            ):
                # O(B * T) exact patch-validity gate: 1.0 on observed patches, 0.0 on all-NaN
                # outages, matching eager Chronos2Model (hidden_states + 0) with zero O(B^2) cost.
                valid_bool = mask if mask.dtype == torch.bool else (mask > 0)
                valid = valid_bool.to(device=proj.device, dtype=proj.dtype).unsqueeze(-1)
                out = hidden_states + proj * valid
            elif (
                mask.ndim == 4
                and mask.shape[0] == hidden_states.shape[1]
                and mask.shape[1] == 1
                and mask.shape[2] == hidden_states.shape[0]
                and mask.shape[3] == hidden_states.shape[0]
            ):
                # Un-stubbed Chronos-2 inverted additive mask (T, 1, B, B) or boolean mask:
                # 0.0 / True valid, < 0 / False masked.
                diag = torch.diagonal(mask[:, 0], dim1=-2, dim2=-1).transpose(0, 1)
                valid_bool = diag if diag.dtype == torch.bool else (diag >= -1e-6)
                valid = valid_bool.to(device=proj.device, dtype=proj.dtype).unsqueeze(-1)
                out = hidden_states + proj * valid
            else:
                out = hidden_states + proj
        else:
            out = hidden_states + proj

        if self._attn_out_cls is not None:
            if self._attn_out_use_kwargs:
                return self._attn_out_cls(hidden_states=out, attn_weights=None)
            if self._attn_out_arity >= 2:
                return self._attn_out_cls(out, None)
            if self._attn_out_arity == 1:
                return self._attn_out_cls(out)
            return self._attn_out_cls(hidden_states=out, attn_weights=None)
        if self._returns_tuple:
            return (out,) + (None,) * max(self._tuple_len - 1, 1)
        return out


def _get_chronos2_output_patch_size(model: Any) -> int:
    ccfg = getattr(model, "chronos_config", None)
    if ccfg is None and hasattr(model, "config"):
        ccfg = getattr(model.config, "chronos_config", None)
    if isinstance(ccfg, dict):
        return max(int(ccfg.get("output_patch_size", 16)), 1)
    if ccfg is not None:
        return max(int(getattr(ccfg, "output_patch_size", 16)), 1)
    return 16


class Chronos2Compiler:
    """Compiler for ``amazon/chronos-2`` models with plan-gated GroupSelfAttention collapse."""

    def __init__(
        self,
        model: Any,
        *,
        horizon: int = 16,
        context_length: int = 512,
        precision: str = "auto",
        batch_buckets: Sequence[int] | None = None,
        dc_strip: bool = True,
        quantile_indices: Sequence[int] | None = None,
        use_cuda_graphs: bool = True,
        copy_model: bool = True,
    ) -> None:
        if torch is None:
            raise RuntimeError("Chronos2Compiler requires PyTorch.")

        raw_model = (
            model.model if hasattr(model, "model") and isinstance(model.model, nn.Module) else model
        )
        self.model = copy.deepcopy(raw_model) if copy_model else raw_model
        self.model.eval()

        self.horizon = int(horizon)
        self.context_length = int(context_length)
        precision = normalize_precision_alias(precision)
        self.precision = precision
        self.dc_stripper = CausalFloat64DCStripper(enabled=dc_strip)
        self.quantile_indices = (
            None if quantile_indices is None else tuple(int(i) for i in quantile_indices)
        )

        # Pass 1: Dynamically discover and collapse ALL GroupSelfAttention layers,
        # and replace encoder._construct_and_invert_group_time_mask under singleton plans
        # with an O(B * T) pass-through of the 2D patch-validity attention_mask to eliminate
        # quadratic [T, 1, B, B] mask allocation & compute while remaining bit-exact on
        # all-NaN patch outages.
        self._plan_lock = threading.RLock()
        self._singleton_active: bool = True
        self._group_mask_stubbed: bool = False
        self._group_mask_stub_target: tuple[Any, Any] | None = None
        self._gated_gsa_modules: list[_PlanGatedGroupSelfAttention] = []
        self._collapse_group_self_attention()
        if self._gated_gsa_modules:
            self._stub_group_time_mask()

        first_param = next(self.model.parameters(), None)
        self.device = first_param.device if first_param is not None else torch.device("cpu")
        self._forward_params = set(inspect.signature(self.model.forward).parameters.keys())
        self._output_patch_size = _get_chronos2_output_patch_size(self.model)
        self._quantile_indices_t = (
            None
            if self.quantile_indices is None
            else torch.tensor(self.quantile_indices, dtype=torch.long, device=self.device)
        )

        # Verify at compile time that Chronos2Encoder.forward cleanly accepts the O(B*T)
        # 2D patch-validity attention_mask pass-through.
        if self._group_mask_stubbed:
            self._verify_group_time_mask_stub()

        # Pass 2: Crest-Guarded precision on encoder projections (norms/embeddings/head stay FP32).
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
                    "input_patch_embedding",
                    "output_patch_embedding",
                    "orig_gsa",
                    "norm",
                    "embed",
                    "head",
                ),
            )

        has_crest_guarded = int(self.precision_summary.get("replaced", 0)) > 0
        use_inductor = (
            self.device.type == "cuda"
            and use_cuda_graphs
            and (not has_crest_guarded)
            and self.precision in ("auto", "tf32", "fp32")
        )
        self.graph_runner = InductorOrGraphRunner(
            self._forward_singleton_e2e,
            sample_shape=(self.context_length,),
            buckets=batch_buckets,
            device=self.device,
            dtype=torch.float32,
            enabled=(use_cuda_graphs and self.device.type == "cuda"),
            use_inductor=use_inductor,
            precision=self.precision,
        )

    def _collapse_group_self_attention(self) -> None:
        def _walk(parent: Any) -> None:
            for name, child in list(parent.named_children()):
                if isinstance(child, _PlanGatedGroupSelfAttention):
                    continue
                cls_low = type(child).__name__.lower()
                if "groupselfattention" in cls_low or name == "group_self_attention":
                    gated = _PlanGatedGroupSelfAttention(child, singleton_plan=True)
                    if isinstance(parent, nn.ModuleList) and name.isdigit():
                        parent[int(name)] = gated
                    else:
                        setattr(parent, name, gated)
                    self._gated_gsa_modules.append(gated)
                else:
                    _walk(child)

        _walk(self.model)

    def _stub_group_time_mask(self) -> None:
        """Bypass quadratic [T, 1, B, B] group-time mask construction under singleton plans.

        Returns the existing 2D ``(B, T)`` ``attention_mask`` directly (zero allocation,
        zero ``O(B^2)`` memory) so ``_PlanGatedGroupSelfAttention`` masks all-NaN patch
        outages identically to eager ``Chronos2Model``.
        """
        candidates = []
        encoder = getattr(self.model, "encoder", None)
        if encoder is not None:
            candidates.append(encoder)
        candidates.append(self.model)

        for target in candidates:
            orig_fn = getattr(target, "_construct_and_invert_group_time_mask", None)
            if callable(orig_fn):

                def _gated_group_time_mask(
                    *args: Any,
                    _orig: Any = orig_fn,
                    **kwargs: Any,
                ) -> Any:
                    if self._singleton_active:
                        attn_mask = kwargs.get("attention_mask", None)
                        if attn_mask is None:
                            for cand in args:
                                if isinstance(cand, torch.Tensor) and cand.ndim == 2:
                                    attn_mask = cand
                                    break
                        return attn_mask
                    return _orig(*args, **kwargs)

                target._construct_and_invert_group_time_mask = _gated_group_time_mask
                self._group_mask_stub_target = (target, orig_fn)
                self._group_mask_stubbed = True
                for mod in self._gated_gsa_modules:
                    mod._patch_mask_stub_active = True
                break

    def _verify_group_time_mask_stub(self) -> None:
        """Confirm ``Chronos2Encoder.forward`` accepts the O(B*T) mask pass-through."""
        if not self._group_mask_stubbed or self._group_mask_stub_target is None:
            return
        try:
            t = torch.linspace(
                0.0, 4.0, max(self.context_length, 1), dtype=torch.float32, device=self.device
            )
            probe_x = torch.stack([torch.sin(t), torch.cos(t)], dim=0)
            with torch.inference_mode(), tf32_precision_context("fp32"):
                _ = self._forward_singleton_stripped_f32(probe_x)
        except Exception:
            target, orig_fn = self._group_mask_stub_target
            target._construct_and_invert_group_time_mask = orig_fn
            self._group_mask_stubbed = False
            self._group_mask_stub_target = None
            for mod in self._gated_gsa_modules:
                mod._patch_mask_stub_active = False

    def _set_singleton_plan(self, active: bool) -> None:
        with self._plan_lock:
            self._singleton_active = bool(active)
            for mod in self._gated_gsa_modules:
                mod.singleton_plan = bool(active)

    def _extract_quantile_preds(self, out: Any) -> Any:
        if hasattr(out, "quantile_preds"):
            q = out.quantile_preds
        elif isinstance(out, tuple):
            q = out[0]
        else:
            q = out
        if self._quantile_indices_t is not None:
            idx = (
                self._quantile_indices_t
                if self._quantile_indices_t.device == q.device
                else self._quantile_indices_t.to(device=q.device)
            )
            q = q.index_select(1, idx)
        return q[:, :, : self.horizon]

    def _call_chronos2_model(self, x_2d_f32: Any, group_ids: Any) -> Any:
        kwargs: dict[str, Any] = {}
        if "context" in self._forward_params:
            kwargs["context"] = x_2d_f32
        if "group_ids" in self._forward_params:
            kwargs["group_ids"] = group_ids
        if "num_output_patches" in self._forward_params:
            patch_len = max(self._output_patch_size, 1)
            kwargs["num_output_patches"] = max(1, (self.horizon + patch_len - 1) // patch_len)
        if kwargs:
            return self.model(**kwargs)
        return self.model(x_2d_f32)

    def _forward_singleton_stripped_f32(self, x_2d_f32: Any) -> Any:
        B = x_2d_f32.shape[0]
        group_ids = torch.arange(B, dtype=torch.long, device=x_2d_f32.device)
        out = self._call_chronos2_model(x_2d_f32, group_ids)
        return self._extract_quantile_preds(out)

    def _forward_singleton_e2e(self, x_2d: Any) -> Any:
        x_s, dc_f64 = self.dc_stripper.strip(x_2d)
        q_s = self._forward_singleton_stripped_f32(x_s)
        return self.dc_stripper.restore(q_s, dc_f64).sort(dim=1).values.contiguous()

    def predict_quantiles(self, x: Any, *, group_ids: Any | None = None) -> Any:
        """Predict monotonic quantiles ``(B, Q, H)`` with plan-gated singleton collapse."""
        if (
            isinstance(x, torch.Tensor)
            and x.device == self.device
            and x.dtype in (torch.float32, torch.float64)
        ):
            x_t = x.unsqueeze(0) if x.ndim == 1 else x
        else:
            x_t = torch.as_tensor(x, device=self.device, dtype=torch.float64)
            if x_t.ndim == 1:
                x_t = x_t.unsqueeze(0)
        B = x_t.shape[0]

        is_singleton = True
        gid_t = None
        if group_ids is not None:
            gid_t = torch.as_tensor(group_ids, device=self.device, dtype=torch.long)
            if gid_t.ndim != 1 or gid_t.shape[0] != B:
                raise ValueError(f"Expected 1D group_ids of length {B}, got {tuple(gid_t.shape)}")
            is_singleton = bool(torch.unique(gid_t).numel() == B)

        with self._plan_lock:
            if is_singleton:
                self._set_singleton_plan(True)
                return self.graph_runner(x_t)

            self._set_singleton_plan(False)
            try:
                with torch.inference_mode(), tf32_precision_context(self.precision):
                    x_s, dc_f64 = self.dc_stripper.strip(x_t)
                    out = self._call_chronos2_model(x_s, gid_t)
                    q_s = self._extract_quantile_preds(out)
                    return self.dc_stripper.restore(q_s, dc_f64).sort(dim=1).values
            finally:
                self._set_singleton_plan(True)

    def warmup(self, buckets: Sequence[int] | None = None) -> list[int]:
        """Thread-safely warm up singleton CUDA/Inductor graphs under ``_plan_lock``."""
        with self._plan_lock:
            self._set_singleton_plan(True)
            return self.graph_runner.warmup(buckets=buckets)

    def predict(self, x: Any, *, group_ids: Any | None = None) -> Any:
        """Predict point forecast ``(B, H)`` (median quantile)."""
        q = self.predict_quantiles(x, group_ids=group_ids)
        return q[:, q.shape[1] // 2, :]

    def report(self) -> dict[str, Any]:
        return {
            "pass": "chronos-2",
            "horizon": self.horizon,
            "collapsed_group_self_attention_layers": len(self._gated_gsa_modules),
            "group_time_mask_stubbed": self._group_mask_stubbed,
            "quantile_subset_size": None
            if self.quantile_indices is None
            else len(self.quantile_indices),
            "dc_strip": self.dc_stripper.enabled,
            "precision": self.precision_summary,
            "cuda_graphs": self.graph_runner.stats(),
        }
