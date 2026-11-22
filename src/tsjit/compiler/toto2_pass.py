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

"""Dynamic compiler pass for Toto-2.0 (4m / 22m / 313m) models.

Implements:
- Dynamic architecture introspection (num_layers, layer_group_size,
  num_variate_layers_per_group, patch_size — never hardcoded).
- Exact missing-value & PatchedCausalStdScaler parity on 3D (B, 1, L) context
  with forward-fill (static_loc[..., -1:], static_scale[..., -1:]) and
  valid = (causal & patch_obs) | eye(S) all-NaN softmax guard.
- Pass 1 (R1) anchor-query contraction on the final time-attention layer
  (preserving all unit_scaling.functional calls).
- Pass 2 (Singleton Variate Collapse) replacing length-1 variate SDPA +
  rearrange with direct out_proj(v_proj(norm1(x))), eliminating CUDA gridDim
  crashes at B >= 1009.
- CausalFloat64DCStripper + CrestGuardedLinear + non-finite clamp + monotonic
  quantile sorting + BucketedCudaGraphRunner.
"""

from __future__ import annotations

import copy
import importlib
import inspect
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - optional in pure-CPU environments
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from ..cuda.scaled_fp16_triton import (
    _extract_effective_affine_params,
    is_linear_module,
    replace_linears_with_crest_guarded,
)
from .graph_runner import InductorOrGraphRunner
from .guards import (
    CausalFloat64DCStripper,
    ParityAuditReport,
    StrictParityValidator,
    normalize_precision_alias,
    tf32_precision_context,
)

_us_functional: Any = None
for _us_mod_name in ("dd_unit_scaling.functional", "unit_scaling.functional"):
    try:
        _us_functional = importlib.import_module(_us_mod_name)
        break
    except ImportError:  # pragma: no cover - optional unless Toto-2.0 runtime installed
        continue


@dataclass(frozen=True)
class Toto2ArchitectureInfo:
    """Dynamically inspected Toto-2.0 architecture parameters."""

    num_layers: int
    layer_group_size: int
    num_variate_layers_per_group: int
    patch_size: int
    stride: int
    embed_dim: int
    num_heads: int
    variate_layer_indices: tuple[int, ...]
    time_layer_indices: tuple[int, ...]
    final_time_layer_idx: int


def unwrap_toto2_backbone(model: Any) -> Any:
    """Unwrap ``Toto`` / ``TotoForecaster`` / ``Toto2Model`` to its core backbone."""
    if hasattr(model, "model") and (
        hasattr(model.model, "transformer") or hasattr(model.model, "layers")
    ):
        return model.model
    return model


def is_toto2_model(model: Any) -> bool:
    """Heuristically identify Toto-2.0 models (4m / 22m / 313m) or compatible backbones."""
    cls_name = type(model).__name__.lower()
    if "toto" in cls_name:
        return True
    backbone = unwrap_toto2_backbone(model)
    b_name = type(backbone).__name__.lower()
    if "toto" in b_name:
        return True
    cfg = getattr(backbone, "config", getattr(model, "config", None))
    if cfg is not None:
        model_type = str(getattr(cfg, "model_type", "")).lower()
        if "toto" in model_type:
            return True
        if hasattr(cfg, "layer_group_size") and hasattr(cfg, "num_variate_layers_per_group"):
            return True
    return False


def _get_transformer_layers(backbone: Any) -> list[Any]:
    if hasattr(backbone, "transformer"):
        if hasattr(backbone.transformer, "layers"):
            return list(backbone.transformer.layers)
        if hasattr(backbone.transformer, "blocks"):
            return list(backbone.transformer.blocks)
    if hasattr(backbone, "layers"):
        return list(backbone.layers)
    if hasattr(backbone, "blocks"):
        return list(backbone.blocks)
    raise ValueError("Cannot locate transformer layers on Toto-2.0 backbone")


def inspect_toto2_architecture(model: Any) -> Toto2ArchitectureInfo:
    """Dynamically inspect Toto-2.0 layer schedule and patch dimensions."""
    backbone = unwrap_toto2_backbone(model)
    cfg = getattr(backbone, "config", getattr(model, "config", None))
    layers = _get_transformer_layers(backbone)
    num_layers = int(getattr(cfg, "num_layers", len(layers))) if cfg is not None else len(layers)
    if len(layers) != num_layers and len(layers) > 0:
        num_layers = len(layers)

    layer_group_size = int(
        getattr(
            cfg,
            "layer_group_size",
            getattr(
                cfg, "spacewise_every_n_layers", getattr(backbone, "spacewise_every_n_layers", 2)
            ),
        )
    )
    if layer_group_size <= 0:
        layer_group_size = 1

    num_var_per_group = int(
        getattr(cfg, "num_variate_layers_per_group", 1) if cfg is not None else 1
    )
    patch_embed = getattr(backbone, "patch_embed", None)
    patch_size = int(
        getattr(
            cfg,
            "patch_size",
            getattr(patch_embed, "patch_size", getattr(backbone, "patch_size", 32)),
        )
    )
    stride = int(
        getattr(
            cfg,
            "stride",
            getattr(patch_embed, "stride", patch_size),
        )
    )
    embed_dim = int(
        getattr(
            cfg,
            "embed_dim",
            getattr(cfg, "d_model", getattr(backbone, "embed_dim", 128)),
        )
    )
    first_attn = None
    if layers:
        first_attn = getattr(
            layers[0],
            "attn",
            getattr(layers[0], "attention", getattr(layers[0], "self_attn", None)),
        )
    num_heads = int(
        getattr(
            cfg,
            "num_heads",
            getattr(
                cfg,
                "n_heads",
                getattr(
                    backbone,
                    "num_heads",
                    getattr(first_attn, "_Hq", getattr(first_attn, "num_heads", 4)),
                ),
            ),
        )
    )

    variate_indices: list[int] = []
    time_indices: list[int] = []
    for idx, layer in enumerate(layers):
        axis = getattr(layer, "attention_axis", None)
        is_var_attr = getattr(layer, "is_variate_layer", None)
        if is_var_attr is None:
            attn_sub = getattr(
                layer, "attn", getattr(layer, "attention", getattr(layer, "self_attn", None))
            )
            if attn_sub is not None:
                is_var_attr = getattr(attn_sub, "is_variate_layer", None)
        if isinstance(axis, str):
            is_var = axis.lower() in ("space", "variate", "channel", "spacewise")
        elif is_var_attr is not None:
            is_var = bool(is_var_attr)
        elif hasattr(layer, "spacewise") and isinstance(layer.spacewise, bool):
            is_var = layer.spacewise
        else:
            pos_in_group = idx % layer_group_size
            is_var = pos_in_group >= (layer_group_size - num_var_per_group)
        if is_var:
            variate_indices.append(idx)
        else:
            time_indices.append(idx)

    if not time_indices:
        raise ValueError("Toto-2.0 backbone must contain at least one time-attention layer")

    return Toto2ArchitectureInfo(
        num_layers=num_layers,
        layer_group_size=layer_group_size,
        num_variate_layers_per_group=num_var_per_group,
        patch_size=patch_size,
        stride=stride,
        embed_dim=embed_dim,
        num_heads=num_heads,
        variate_layer_indices=tuple(variate_indices),
        time_layer_indices=tuple(time_indices),
        final_time_layer_idx=time_indices[-1],
    )


def _call_toto_scaler(scaler: Any, x_3d: Any, obs_mask_3d: Any) -> tuple[Any, Any, Any]:
    """Call Toto-2.0 scaler on exact 3D ``(B, 1, L)`` context + mask, preserving forward-fill.

    Passes ``obs_mask_3d.to(torch.bool)`` FIRST (required by Toto-2.0
    ``PatchedCausalStdScaler``'s ``torch.where(mask, ...)``), falling back to
    ``obs_mask_3d.to(x.dtype)`` ONLY if boolean invocation raises TypeError/RuntimeError.
    """
    x_clean = torch.nan_to_num(x_3d, nan=0.0, posinf=0.0, neginf=0.0)
    mask_bool = obs_mask_3d.to(dtype=torch.bool)
    mask_f = obs_mask_3d.to(dtype=x_clean.dtype)
    weights = mask_f

    if scaler is None:
        # Clean causal mean/std fallback when a custom model has no scaler submodule.
        cnt = mask_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
        loc = (x_clean * mask_f).sum(dim=-1, keepdim=True) / cnt
        var = (((x_clean - loc) * mask_f) ** 2).sum(dim=-1, keepdim=True) / cnt
        scale = torch.sqrt(var + 1e-6).clamp(min=1e-6)
        scaled = torch.where(mask_bool, (x_clean - loc) / scale, torch.zeros_like(x_clean))
        return scaled, loc, scale

    sig = inspect.signature(scaler.forward)
    params = [
        p
        for p in sig.parameters.values()
        if p.name != "self"
        and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]

    if len(params) == 1:
        res = scaler(x_clean)
    elif len(params) == 2:
        try:
            res = scaler(x_clean, mask_bool)
        except (TypeError, RuntimeError):
            res = scaler(x_clean, mask_f)
    else:
        kwargs: dict[str, Any] = {}
        mask_keys: set[str] = set()
        for p in params[1:]:
            if p.name == "weights":
                kwargs["weights"] = weights
            elif p.name == "id_mask":
                kwargs["id_mask"] = torch.zeros_like(mask_bool, dtype=torch.long)
            elif "mask" in p.name.lower():
                kwargs[p.name] = mask_bool
                mask_keys.add(p.name)
            elif p.default is inspect.Parameter.empty:
                kwargs[p.name] = mask_bool
                mask_keys.add(p.name)
        if kwargs:
            try:
                res = scaler(x_clean, **kwargs)
            except (TypeError, RuntimeError):
                kwargs_f = {
                    k: (mask_f if k in mask_keys else v)
                    for k, v in kwargs.items()
                }
                res = scaler(x_clean, **kwargs_f)
        elif params[1].name == "weights":
            try:
                res = scaler(x_clean, weights, mask_bool)
            except (TypeError, RuntimeError):
                res = scaler(x_clean, weights, mask_f)
        else:
            try:
                res = scaler(x_clean, mask_bool)
            except (TypeError, RuntimeError):
                res = scaler(x_clean, mask_f)

    if isinstance(res, tuple) and len(res) >= 3:
        scaled, loc, scale = res[0], res[1], res[2]
    elif isinstance(res, tuple) and len(res) == 2:
        scaled, (loc, scale) = res[0], res[1]
    else:
        raise ValueError("Unexpected return signature from Toto-2.0 scaler")
    return scaled, loc, scale


def _scale_context_with_horizon_carry(
    scaler: Any,
    x_3d: Any,
    obs_mask_3d: Any,
    patch_size: int,
) -> tuple[Any, Any, Any]:
    """Scale context and carry forward horizon ``(static_loc, static_scale)``.

    Appends one masked-out future patch (``full_mask[..., L:] = False``) so
    ``PatchedCausalStdScaler`` carries forward the exact last context position's
    causal ``loc``/``scale`` onto the forecast horizon (matching ``Toto2Model.forecast``).
    """
    B, V, L = x_3d.shape
    pad_x = torch.zeros((B, V, patch_size), dtype=x_3d.dtype, device=x_3d.device)
    pad_m = torch.zeros((B, V, patch_size), dtype=torch.bool, device=x_3d.device)
    full_x = torch.cat([torch.nan_to_num(x_3d, nan=0.0, posinf=0.0, neginf=0.0), pad_x], dim=-1)
    full_m = torch.cat([obs_mask_3d.to(dtype=torch.bool), pad_m], dim=-1)
    scaled_full, loc_full, scale_full = _call_toto_scaler(scaler, full_x, full_m)
    return scaled_full[..., :L], loc_full[..., -1:], scale_full[..., -1:]


def _build_toto_time_mask(
    ctx_patch_obs: Any,
    total_patches: int,
) -> Any:
    """Build exact Toto-2.0 causal time mask with all-NaN softmax guard.

    Matches vendor ``base_gids[..., :initial_patches][ctx_patch_obs == 0] = -1``
    with ``valid = (causal & patch_obs) | eye(S)`` so rows with zero observed
    prior patches attend cleanly to self instead of producing NaN softmax.
    """
    B, S_ctx = ctx_patch_obs.shape
    S = int(total_patches)
    device = ctx_patch_obs.device

    causal = torch.tril(torch.ones((S, S), dtype=torch.bool, device=device)).view(1, 1, S, S)
    key_obs = torch.ones((B, S), dtype=torch.bool, device=device)
    key_obs[:, :S_ctx] = ctx_patch_obs
    patch_obs = key_obs.view(B, 1, 1, S)
    eye_s = torch.eye(S, dtype=torch.bool, device=device).view(1, 1, S, S)
    return (causal & patch_obs) | eye_s


def _get_attn_module(layer: Any) -> Any:
    for name in ("attention", "attn", "self_attn"):
        if hasattr(layer, name):
            return getattr(layer, name)
    raise AttributeError(f"Layer {type(layer).__name__} has no recognized attention submodule")


def _get_mlp_module(layer: Any) -> Any:
    for name in ("mlp", "feed_forward", "ffn"):
        if hasattr(layer, name):
            return getattr(layer, name)
    return None


def _get_norm1(layer: Any) -> Any:
    for name in ("norm1", "attention_norm", "pre_norm", "input_layernorm"):
        if hasattr(layer, name):
            return getattr(layer, name)
    return None


def _get_norm2(layer: Any) -> Any:
    for name in ("norm2", "ffn_norm", "post_attention_layernorm", "mlp_norm"):
        if hasattr(layer, name):
            return getattr(layer, name)
    return None


def _unit_residual_split(layer: Any, x: Any) -> tuple[Any, Any]:
    tau = getattr(layer, "residual_tau", getattr(layer, "tau", None))
    if _us_functional is not None and hasattr(_us_functional, "residual_split") and tau is not None:
        return _us_functional.residual_split(x, tau=tau)
    return x, x


def _unit_residual_add(layer: Any, residual: Any, branch: Any) -> Any:
    tau = getattr(layer, "residual_tau", getattr(layer, "tau", None))
    if _us_functional is not None and hasattr(_us_functional, "residual_add") and tau is not None:
        return _us_functional.residual_add(residual, branch, tau=tau)
    return residual + branch


def _get_fused_qkv(attn: Any) -> Any | None:
    for name in ("in_proj", "wQKV", "Wqkv", "W_qkv", "qkv_proj", "qkv", "c_attn"):
        cand = getattr(attn, name, None)
        if cand is not None:
            return cand
    return None


def _infer_qkv_dims(attn: Any, d_model: int, default_heads: int) -> tuple[int, int, int, int, int]:
    """Return ``(n_heads, n_kv_heads, head_dim, d_q, d_kv)`` supporting MHA and GQA."""
    split_sizes = getattr(attn, "_split_sizes", None)
    hq_attr = getattr(attn, "_Hq", None)
    hkv_attr = getattr(attn, "_Hkv", None)
    if (
        split_sizes is not None
        and len(split_sizes) == 3
        and hq_attr is not None
        and hkv_attr is not None
    ):
        d_q = int(split_sizes[0])
        d_k = int(split_sizes[1])
        d_v = int(split_sizes[2])
        n_heads = int(hq_attr)
        n_kv_heads = int(hkv_attr)
        head_dim = d_q // max(n_heads, 1)
        if d_k == d_v:
            return n_heads, n_kv_heads, head_dim, d_q, d_v

    n_heads = int(getattr(attn, "num_heads", getattr(attn, "n_heads", default_heads)))
    head_dim = int(getattr(attn, "head_dim", d_model // max(n_heads, 1)))
    d_q = n_heads * head_dim
    n_kv_attr = getattr(attn, "num_kv_heads", getattr(attn, "n_kv_heads", None))
    if n_kv_attr is not None:
        n_kv_heads = int(n_kv_attr)
        d_kv = n_kv_heads * head_dim
        return n_heads, n_kv_heads, head_dim, d_q, d_kv

    fused = _get_fused_qkv(attn)
    if fused is not None and hasattr(fused, "weight"):
        d_total = int(fused.weight.shape[0])
        if d_total > d_q and ((d_total - d_q) % 2) == 0:
            d_kv = (d_total - d_q) // 2
            n_kv_heads = max(1, d_kv // max(head_dim, 1))
            return n_heads, n_kv_heads, head_dim, d_q, d_kv
    return n_heads, n_heads, head_dim, d_q, d_q


def _slice_linear_rows(linear: Any, start: int, end: int) -> Any:
    affine = _extract_effective_affine_params(linear)
    if affine is not None:
        w_src, b_src = affine
    else:
        w_src = linear.weight.detach()
        b_src = None if getattr(linear, "bias", None) is None else linear.bias.detach()

    in_f = int(getattr(linear, "in_features", linear.weight.shape[1]))
    out_dtype = linear.weight.dtype
    new_lin = nn.Linear(
        in_f,
        end - start,
        bias=(b_src is not None),
        device=linear.weight.device,
        dtype=out_dtype,
    )
    with torch.no_grad():
        new_lin.weight.copy_(w_src[start:end].to(dtype=out_dtype))
        if b_src is not None and new_lin.bias is not None:
            new_lin.bias.copy_(b_src[start:end].to(dtype=out_dtype))
    return new_lin


def _prepare_compiled_qkv_projections(
    layers: list[Any], arch: Toto2ArchitectureInfo
) -> None:
    """Materialize sliced linear views before Crest-Guarded replacement so zero layers bypass it."""
    var_set = set(arch.variate_layer_indices)
    for idx, layer in enumerate(layers):
        attn = _get_attn_module(layer)
        fused = _get_fused_qkv(attn)
        if fused is None or not is_linear_module(fused):
            continue
        d_in = int(getattr(fused, "in_features", fused.weight.shape[1]))
        _, _, _, d_q, d_kv = _infer_qkv_dims(attn, d_in, arch.num_heads)
        if idx in var_set and not hasattr(attn, "v_proj"):
            attn.v_proj = _slice_linear_rows(fused, d_q + d_kv, d_q + 2 * d_kv)
        elif idx == arch.final_time_layer_idx and not hasattr(attn, "q_proj"):
            attn.q_proj = _slice_linear_rows(fused, 0, d_q)
            attn.k_proj = _slice_linear_rows(fused, d_q, d_q + d_kv)
            attn.v_proj = _slice_linear_rows(fused, d_q + d_kv, d_q + 2 * d_kv)


def _project_qkv(
    attn: Any,
    x_q: Any,
    x_kv: Any,
    *,
    d_q: int,
    d_kv: int,
) -> tuple[Any, Any, Any]:
    """Extract (q, k, v) supporting separate q/k/v projections, MHA, and GQA fused wQKV."""
    if hasattr(attn, "q_proj") and hasattr(attn, "k_proj") and hasattr(attn, "v_proj"):
        return attn.q_proj(x_q), attn.k_proj(x_kv), attn.v_proj(x_kv)

    fused = _get_fused_qkv(attn)
    if fused is None:
        raise AttributeError(f"Attention module {type(attn).__name__} missing QKV projections")

    if x_q.data_ptr() == x_kv.data_ptr() and x_q.shape == x_kv.shape:
        qkv = fused(x_kv)
        return qkv[..., :d_q], qkv[..., d_q : d_q + d_kv], qkv[..., d_q + d_kv : d_q + 2 * d_kv]

    w = fused.weight
    b = fused.bias
    q = F.linear(x_q, w[:d_q], None if b is None else b[:d_q])
    kv = F.linear(x_kv, w[d_q : d_q + 2 * d_kv], None if b is None else b[d_q : d_q + 2 * d_kv])
    k, v = kv.split([d_kv, d_kv], dim=-1)
    return q, k, v


def _project_v_only(attn: Any, x: Any, *, d_q: int, d_kv: int) -> Any:
    """Compute ONLY the value projection V(x) for Singleton Variate Collapse."""
    if hasattr(attn, "v_proj"):
        return attn.v_proj(x)
    fused = _get_fused_qkv(attn)
    if fused is None:
        raise AttributeError(f"Attention module {type(attn).__name__} missing V projection")
    w_v = fused.weight[d_q + d_kv : d_q + 2 * d_kv]
    b_v = None if fused.bias is None else fused.bias[d_q + d_kv : d_q + 2 * d_kv]
    return F.linear(x, w_v, b_v)


def _project_out(attn: Any, x: Any) -> Any:
    for name in ("wO", "Wo", "W_o", "out_proj", "o_proj", "output_proj", "proj"):
        if hasattr(attn, name):
            return getattr(attn, name)(x)
    raise AttributeError(f"Attention module {type(attn).__name__} missing output projection")


def _apply_rotary_if_present(attn: Any, q: Any, k: Any, q_start_pos: int) -> tuple[Any, Any]:
    """Apply rotary positional embeddings with exact offset ``[q_start_pos .. S-1]`` for Q."""
    rotary = getattr(attn, "rotary_emb", getattr(attn, "rope", None))
    if rotary is None:
        return q, k
    # q: (B, H, nq, Dh), k: (B, H, S, Dh)
    S = k.shape[-2]
    nq = q.shape[-2]
    if hasattr(rotary, "rotate_queries_and_keys"):
        if nq == S:
            return rotary.rotate_queries_and_keys(q, k)
        # Pad Q left to S to reuse vendor RoPE exact phases, then slice last nq.
        q_full = torch.zeros_like(k)
        q_full[:, :, q_start_pos : q_start_pos + nq, :] = q
        q_rot, k_rot = rotary.rotate_queries_and_keys(q_full, k)
        return q_rot[:, :, q_start_pos : q_start_pos + nq, :], k_rot
    if callable(rotary):
        sig = inspect.signature(rotary)
        if len(sig.parameters) >= 2:
            if nq == S:
                return rotary(q, k)
            q_full = torch.zeros_like(k)
            q_full[:, :, q_start_pos : q_start_pos + nq, :] = q
            q_rot, k_rot = rotary(q_full, k)
            return q_rot[:, :, q_start_pos : q_start_pos + nq, :], k_rot
    return q, k


def _run_time_layer(
    layer: Any,
    x: Any,
    valid_mask: Any,
    num_heads: int,
    *,
    anchor_nq: int | None = None,
) -> Any:
    """Run a Toto-2.0 time layer with optional Pass 1 (R1) anchor-query contraction."""
    B, S, D = x.shape
    nq = S if anchor_nq is None else min(int(anchor_nq), S)
    q_start = S - nq

    norm1 = _get_norm1(layer)
    attn = _get_attn_module(layer)
    norm2 = _get_norm2(layer)
    mlp = _get_mlp_module(layer)

    res1, branch_in = _unit_residual_split(layer, x)
    if nq < S:
        res1 = res1[:, q_start:, :]

    h_kv = norm1(branch_in) if norm1 is not None else branch_in
    h_q = h_kv[:, q_start:, :] if nq < S else h_kv

    n_heads, n_kv_heads, head_dim, d_q, d_kv = _infer_qkv_dims(attn, D, num_heads)
    q, k, v = _project_qkv(attn, h_q, h_kv, d_q=d_q, d_kv=d_kv)

    q = q.view(B, nq, n_heads, head_dim).transpose(1, 2)
    k = k.view(B, S, n_kv_heads, head_dim).transpose(1, 2)
    v = v.view(B, S, n_kv_heads, head_dim).transpose(1, 2)

    if hasattr(attn, "q_norm") and attn.q_norm is not None:
        q = attn.q_norm(q)
    if hasattr(attn, "k_norm") and attn.k_norm is not None:
        k = attn.k_norm(k)

    q, k = _apply_rotary_if_present(attn, q, k, q_start)

    if n_kv_heads != n_heads:
        rep = n_heads // n_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

    mask_slice = valid_mask[:, :, q_start:, :] if nq < S else valid_mask
    attn_scale = getattr(attn, "scale", 1.0 / math.sqrt(head_dim))
    if (
        _us_functional is not None
        and hasattr(_us_functional, "mult")
        and hasattr(attn, "unit_scale_mult")
    ):
        q = _us_functional.mult(q, attn.unit_scale_mult)

    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask_slice,
        dropout_p=0.0,
        is_causal=False,
        scale=attn_scale,
    )
    out = out.transpose(1, 2).contiguous().view(B, nq, d_q)
    out = _project_out(attn, out)
    x_mid = _unit_residual_add(layer, res1, out)

    if mlp is not None:
        res2, mlp_in = _unit_residual_split(layer, x_mid)
        h_mlp = norm2(mlp_in) if norm2 is not None else mlp_in
        x_mid = _unit_residual_add(layer, res2, mlp(h_mlp))
    return x_mid


def _run_singleton_variate_layer(layer: Any, x: Any, num_heads: int) -> Any:
    """Pass 2 (Singleton Variate Collapse): length-1 variate SDPA -> out_proj(v_proj(norm1(x))).

    Eliminates length-1 SDPA + (B, V, S, D) <-> (B*S, V, D) rearranges and avoids
    vendor Toto-2.0 CUDA gridDim crashes at B >= 1009.
    """
    B, S, D = x.shape
    norm1 = _get_norm1(layer)
    attn = _get_attn_module(layer)
    norm2 = _get_norm2(layer)
    mlp = _get_mlp_module(layer)

    res1, branch_in = _unit_residual_split(layer, x)
    h = norm1(branch_in) if norm1 is not None else branch_in
    n_heads, n_kv_heads, head_dim, d_q, d_kv = _infer_qkv_dims(attn, D, num_heads)
    v = _project_v_only(attn, h, d_q=d_q, d_kv=d_kv)
    if n_kv_heads != n_heads:
        rep = n_heads // n_kv_heads
        v = v.view(B, S, n_kv_heads, head_dim).repeat_interleave(rep, dim=2).reshape(B, S, d_q)
    out = _project_out(attn, v)
    x_mid = _unit_residual_add(layer, res1, out)

    if mlp is not None:
        res2, mlp_in = _unit_residual_split(layer, x_mid)
        h_mlp = norm2(mlp_in) if norm2 is not None else mlp_in
        x_mid = _unit_residual_add(layer, res2, mlp(h_mlp))
    return x_mid


def _decode_toto_quantiles(
    backbone: Any,
    hidden_nq: Any,
    static_loc: Any,
    static_scale: Any,
    horizon: int,
    patch_size: int,
    quantiles: Sequence[float],
) -> Any:
    """Decode Toto-2.0 output distribution or quantile head into ``(Q, B, H)`` tensor."""
    B, nq, D = hidden_nq.shape
    Q = len(quantiles)
    q_levels = torch.tensor(quantiles, dtype=torch.float32, device=hidden_nq.device)

    unembed = getattr(backbone, "unembed", getattr(backbone, "output_proj", None))
    head_in = unembed(hidden_nq) if unembed is not None else hidden_nq

    dist_head = getattr(
        backbone,
        "output_distribution",
        getattr(
            backbone,
            "output_head",
            getattr(backbone, "quantile_head", getattr(backbone, "head", None)),
        ),
    )
    if dist_head is None:
        raise ValueError(
            "Toto-2.0 backbone has no output_distribution, output_head, or head module"
        )

    out = dist_head(head_in)
    # Case 1: Distribution object supporting .icdf / .sample / .mean
    if hasattr(out, "icdf"):
        # Compute exact analytical quantiles via icdf: shape (Q, B, nq, patch_size) or (Q, B, H)
        q_shape = (Q,) + (1,) * len(out.batch_shape)
        raw_q = out.icdf(q_levels.view(q_shape))
        raw_q = raw_q.reshape(Q, B, -1)[:, :, :horizon]
    elif hasattr(out, "sample"):
        # Deterministic median/quantile fallback from distribution samples if icdf unavailable
        samples = out.sample((256,))
        raw_q = torch.quantile(samples.to(torch.float32), q_levels, dim=0)
        raw_q = raw_q.reshape(Q, B, -1)[:, :, :horizon]
    elif isinstance(out, torch.Tensor):
        # Case 2: Direct quantile head tensor (B, nq, Q*P), (Q, B, nq, P), (B, Q, nq*P), etc.
        if out.ndim == 5 and out.shape[0] == Q and out.shape[1] == B:
            raw_q = out.reshape(Q, B, -1)[:, :, :horizon]
        elif out.ndim == 4 and out.shape[0] == Q and out.shape[1] == B:
            raw_q = out.reshape(Q, B, -1)[:, :, :horizon]
        elif out.ndim == 3 and out.shape[-1] == Q * patch_size:
            raw_q = (
                out.view(B, nq, Q, patch_size)
                .permute(2, 0, 1, 3)
                .reshape(Q, B, nq * patch_size)[:, :, :horizon]
            )
        elif out.ndim == 3 and out.shape[1] == Q:
            raw_q = out.permute(1, 0, 2)[:, :, :horizon]
        elif out.ndim == 4 and out.shape[2] == Q:
            raw_q = out.permute(2, 0, 1, 3).reshape(Q, B, -1)[:, :, :horizon]
        else:
            flat = out.reshape(B, -1)[:, :horizon]
            raw_q = flat.unsqueeze(0).expand(Q, B, -1).clone()
    else:
        raise TypeError(f"Unsupported Toto-2.0 output head return type: {type(out)}")

    # Forward-fill static_loc[..., -1:] and static_scale[..., -1:] onto prediction horizon.
    loc_last = static_loc[..., -1:].reshape(1, B, 1).to(raw_q.dtype)
    scale_last = static_scale[..., -1:].reshape(1, B, 1).to(raw_q.dtype)
    return raw_q * scale_last + loc_last


def _has_native_toto2_forecast(model: Any) -> bool:
    """Return True if ``model`` exposes the native ``Toto2Model.forecast(...)`` API."""
    return callable(getattr(model, "forecast", None))


def _clamp_nonfinite_bqh(q_bqh: Any) -> Any:
    """Branchless CUDA-Graph-safe replacement of non-finite entries on ``(B, Q, H)``."""
    is_fin = torch.isfinite(q_bqh)
    q_zero = torch.where(is_fin, q_bqh, torch.zeros_like(q_bqh))
    cnt = is_fin.to(q_bqh.dtype).sum(dim=1, keepdim=True).clamp(min=1.0)
    fallback = q_zero.sum(dim=1, keepdim=True) / cnt
    return torch.where(is_fin, q_bqh, fallback)


def _normalize_toto_forecast_output(
    raw: Any,
    *,
    batch_size: int,
    horizon: int,
    num_quantiles_hint: int = 9,
) -> Any:
    """Normalize ``Toto2Model.forecast`` outputs into ``(B, Q, H)`` tensor."""
    if isinstance(raw, dict):
        for key in ("quantiles", "quantile_preds", "forecast", "samples", "mean"):
            if key in raw:
                raw = raw[key]
                break
    elif not isinstance(raw, torch.Tensor):
        for attr in ("quantiles", "quantile_preds", "forecast", "samples", "mean"):
            if hasattr(raw, attr):
                raw = getattr(raw, attr)
                break

    if not isinstance(raw, torch.Tensor):
        raise TypeError(f"Unsupported Toto2Model.forecast return type: {type(raw)}")

    t = raw
    B = int(batch_size)
    H = int(horizon)
    Qh = int(num_quantiles_hint)

    if t.ndim == 4:
        s0, s1, s2, s3 = t.shape
        # Standard Toto2Model.forecast output: (Q, B, n_var=1, H)
        if s1 == B and s2 == 1 and s3 >= H and (s0 == Qh or B != 1 or s0 != 1):
            return t[:, :, 0, :H].permute(1, 0, 2)
        # (B, n_var=1, Q, H)
        if s0 == B and s1 == 1 and s3 >= H:
            return t[:, 0, :, :H]
        # (B, Q, n_var=1, H)
        if s0 == B and s2 == 1 and s3 >= H:
            return t[:, :, 0, :H]
        # (B, n_var=1, H, Q)
        if s0 == B and s1 == 1 and s2 >= H:
            return t[:, 0, :H, :].permute(0, 2, 1)
    elif t.ndim == 3:
        s0, s1, s2 = t.shape
        if s0 == B and s2 >= H and (s1 == Qh or s0 != Qh):
            return t[:, :, :H]
        if s1 == B and s2 >= H:
            return t[:, :, :H].permute(1, 0, 2)
        if s0 == B and s1 >= H:
            return t[:, :H, :].permute(0, 2, 1)
    elif t.ndim == 2 and t.shape[0] == B and t.shape[1] >= H:
        return t[:, :H].unsqueeze(1)

    raise ValueError(
        f"Cannot normalize Toto2Model.forecast output shape {tuple(t.shape)} "
        f"for B={B}, H={H}"
    )


def _call_toto2_native_forecast(
    model: Any,
    x_2d_f32: Any,
    horizon: int,
    *,
    num_quantiles_hint: int = 9,
    has_missing_override: bool | None = None,
) -> Any:
    """Invoke unmodified ``Toto2Model.forecast({'target': x_3d, ...}, horizon=horizon)``."""
    B, L = x_2d_f32.shape
    device = x_2d_f32.device
    x_f32 = x_2d_f32.to(dtype=torch.float32)
    obs_mask_2d = torch.isfinite(x_f32)
    x_clean = torch.nan_to_num(x_f32, nan=0.0, posinf=0.0, neginf=0.0)

    inputs = {
        "target": x_clean.unsqueeze(1),  # (B, 1, L)
        "target_mask": obs_mask_2d.unsqueeze(1).to(dtype=torch.bool),  # (B, 1, L) bool
        "series_ids": torch.zeros((B, 1), dtype=torch.long, device=device),
    }

    sig = inspect.signature(model.forecast)
    params = sig.parameters
    kwargs: dict[str, Any] = {}
    if "horizon" in params:
        kwargs["horizon"] = int(horizon)
    elif "prediction_length" in params:
        kwargs["prediction_length"] = int(horizon)
    if "has_missing_values" in params:
        if has_missing_override is not None:
            kwargs["has_missing_values"] = bool(has_missing_override)
        else:
            kwargs["has_missing_values"] = bool((~obs_mask_2d).any().item())

    raw = model.forecast(inputs, **kwargs)
    q_bqh = _normalize_toto_forecast_output(
        raw, batch_size=B, horizon=horizon, num_quantiles_hint=num_quantiles_hint
    )
    q_bqh = _clamp_nonfinite_bqh(q_bqh)
    return q_bqh.sort(dim=1).values.contiguous()


_BaseModule = nn.Module if nn is not None else object


def _slice_linear_indices(linear: Any, row_indices: Sequence[int]) -> Any:
    affine = _extract_effective_affine_params(linear)
    if affine is not None:
        w_src, b_src = affine
    else:
        w_src = linear.weight.detach()
        b_src = None if getattr(linear, "bias", None) is None else linear.bias.detach()

    in_f = int(getattr(linear, "in_features", linear.weight.shape[1]))
    out_dtype = linear.weight.dtype
    idx = torch.tensor(list(row_indices), dtype=torch.long, device=w_src.device)
    new_lin = nn.Linear(
        in_f,
        int(idx.numel()),
        bias=(b_src is not None),
        device=linear.weight.device,
        dtype=out_dtype,
    )
    with torch.no_grad():
        new_lin.weight.copy_(w_src.index_select(0, idx).to(dtype=out_dtype))
        if b_src is not None and new_lin.bias is not None:
            new_lin.bias.copy_(b_src.index_select(0, idx).to(dtype=out_dtype))
    return new_lin


def _build_ov_module(
    v_lin: Any,
    o_lin: Any,
    *,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    d_q: int,
    alpha: float = 1.0,
    scale_o_bias: bool = False,
) -> tuple[Any | None, Any | None, Any | None]:
    """Fold ``o_lin(alpha * v_lin(x))`` into ``w_ov`` (MHA) or ``(v_proj, o_proj)`` (GQA)."""
    dev = v_lin.weight.device
    dtype = v_lin.weight.dtype
    if n_kv_heads == n_heads:
        with torch.no_grad():
            w_v64 = v_lin.weight.detach().to(torch.float64) * float(alpha)
            w_o64 = o_lin.weight.detach().to(torch.float64)
            w_ov = (w_o64 @ w_v64).to(dtype=dtype, device=dev)
            b_ov = None
            if v_lin.bias is not None or o_lin.bias is not None:
                b64 = torch.zeros(o_lin.out_features, dtype=torch.float64, device=dev)
                if v_lin.bias is not None:
                    b64 = b64 + (w_o64 @ (v_lin.bias.detach().to(torch.float64) * float(alpha)))
                if o_lin.bias is not None:
                    b_o64 = o_lin.bias.detach().to(torch.float64)
                    b64 = b64 + (b_o64 * float(alpha) if scale_o_bias else b_o64)
                b_ov = b64.to(dtype=dtype, device=dev)
        w_ov_lin = nn.Linear(
            v_lin.in_features,
            o_lin.out_features,
            bias=(b_ov is not None),
            device=dev,
            dtype=dtype,
        )
        with torch.no_grad():
            w_ov_lin.weight.copy_(w_ov)
            if b_ov is not None and w_ov_lin.bias is not None:
                w_ov_lin.bias.copy_(b_ov)
        return w_ov_lin, None, None

    v_scaled = nn.Linear(
        v_lin.in_features,
        v_lin.out_features,
        bias=(v_lin.bias is not None),
        device=dev,
        dtype=dtype,
    )
    o_scaled = nn.Linear(
        o_lin.in_features,
        o_lin.out_features,
        bias=(o_lin.bias is not None),
        device=dev,
        dtype=dtype,
    )
    with torch.no_grad():
        v_scaled.weight.copy_(v_lin.weight.detach() * float(alpha))
        if v_lin.bias is not None and v_scaled.bias is not None:
            v_scaled.bias.copy_(v_lin.bias.detach() * float(alpha))
        o_scaled.weight.copy_(o_lin.weight.detach())
        if o_lin.bias is not None and o_scaled.bias is not None:
            o_scaled.bias.copy_(
                o_lin.bias.detach() * float(alpha) if scale_o_bias else o_lin.bias.detach()
            )
    _ = (head_dim, d_q)
    return None, v_scaled, o_scaled


class _SingletonVariateAttnWrapper(_BaseModule):  # type: ignore[misc]
    """In-place Singleton Variate Collapse wrapper for native ``Toto2Model`` variate attention.

    When the variate dimension is singleton (``V == 1``), replaces length-1 SDPA
    (which overflows CUDA ``gridDim.y`` at ``B >= 1009``) with exact folded
    ``o_proj(alpha * v_proj(x))``, calibrated and verified against the real
    ``orig_attn`` forward output (preserving ``unit_scaling`` factors, fused QKV
    layouts, and tuple-vs-tensor return conventions).
    """

    def __init__(
        self,
        orig_attn: Any,
        default_heads: int,
        *,
        probe_io: tuple[Any, Any] | None = None,
    ) -> None:
        super().__init__()
        self.orig_attn = orig_attn
        self.enabled = True

        # Locate V candidate(s) and O projection as exact affine nn.Linear modules.
        fused = _get_fused_qkv(orig_attn)
        v_candidates: list[Any] = []
        if hasattr(orig_attn, "v_proj") and is_linear_module(orig_attn.v_proj):
            v_out_f = int(
                getattr(orig_attn.v_proj, "out_features", orig_attn.v_proj.weight.shape[0])
            )
            v_lin = _slice_linear_rows(orig_attn.v_proj, 0, v_out_f)
            d_in = v_lin.in_features
            n_heads, n_kv_heads, head_dim, d_q, d_kv = _infer_qkv_dims(
                orig_attn, d_in, default_heads
            )
            v_candidates.append(v_lin)
        elif fused is not None and is_linear_module(fused):
            d_in = int(getattr(fused, "in_features", fused.weight.shape[1]))
            f_out = int(getattr(fused, "out_features", fused.weight.shape[0]))
            n_heads, n_kv_heads, head_dim, d_q, d_kv = _infer_qkv_dims(
                orig_attn, d_in, default_heads
            )
            # Candidate A: contiguous [Q, K, V]
            if d_q + 2 * d_kv <= f_out:
                v_candidates.append(_slice_linear_rows(fused, d_q + d_kv, d_q + 2 * d_kv))
            # Candidate B: head-interleaved (h, 3, head_dim)
            if n_kv_heads == n_heads and f_out == 3 * n_heads * head_dim:
                inter_idx = [
                    h * 3 * head_dim + 2 * head_dim + d
                    for h in range(n_heads)
                    for d in range(head_dim)
                ]
                v_candidates.append(_slice_linear_indices(fused, inter_idx))
        else:
            raise AttributeError("Cannot locate V projection for singleton variate collapse")

        if not v_candidates:
            raise AttributeError("No valid V projection slice found for singleton variate collapse")

        out_lin_orig = None
        for name in ("wO", "Wo", "W_o", "out_proj", "o_proj", "output_proj", "proj"):
            cand = getattr(orig_attn, name, None)
            if is_linear_module(cand):
                out_lin_orig = cand
                break
        if out_lin_orig is None:
            raise AttributeError("Cannot locate output projection for singleton variate collapse")

        o_out_f = int(getattr(out_lin_orig, "out_features", out_lin_orig.weight.shape[0]))
        o_lin = _slice_linear_rows(out_lin_orig, 0, o_out_f)
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.d_q = d_q
        self.d_kv = d_kv

        sig = inspect.signature(orig_attn.forward)
        self._returns_tuple = "tuple" in str(sig.return_annotation).lower()
        self._tuple_len = 2 if self._returns_tuple else 0
        self._expected_ndim: int | None = None
        self._singleton_dim: int | None = None

        best_choice: tuple[Any, float, bool] = (v_candidates[0], 1.0, False)
        if probe_io is not None:
            x_probe, y_probe_raw = probe_io
            if isinstance(y_probe_raw, tuple) and len(y_probe_raw) >= 1:
                self._returns_tuple = True
                self._tuple_len = len(y_probe_raw)
                y_probe = y_probe_raw[0]
            else:
                self._returns_tuple = False
                self._tuple_len = 0
                y_probe = y_probe_raw

            if isinstance(x_probe, torch.Tensor) and isinstance(y_probe, torch.Tensor):
                self._expected_ndim = int(x_probe.ndim)
                if x_probe.ndim == 3 and x_probe.shape[1] == 1:
                    self._singleton_dim = 1
                elif x_probe.ndim == 4:
                    if x_probe.shape[1] == 1 and x_probe.shape[2] != 1:
                        self._singleton_dim = 1
                    elif x_probe.shape[2] == 1 and x_probe.shape[1] != 1:
                        self._singleton_dim = 2
                    elif x_probe.shape[1] == 1:
                        self._singleton_dim = 1

                # Calibrate candidate layout + optional unit_scaling scalar multiplier on real I/O.
                x_p = x_probe.detach().to(dtype=v_candidates[0].weight.dtype)
                y_t = y_probe.detach().to(torch.float32)
                b_o = (
                    o_lin.bias.detach().to(torch.float32)
                    if o_lin.bias is not None
                    else torch.zeros(o_lin.out_features, dtype=torch.float32, device=y_t.device)
                )
                ref_mag = max(float(y_t.abs().max().item()), 1e-6)
                best_err = float("inf")
                found_valid = False

                with torch.no_grad():
                    for v_cand in v_candidates:
                        for scale_o_bias in (False, True):
                            w_ov_c, vp_c, op_c = _build_ov_module(
                                v_cand,
                                o_lin,
                                n_heads=n_heads,
                                n_kv_heads=n_kv_heads,
                                head_dim=head_dim,
                                d_q=d_q,
                                alpha=1.0,
                                scale_o_bias=scale_o_bias,
                            )
                            if w_ov_c is not None:
                                y_base = w_ov_c(x_p).to(torch.float32)
                            else:
                                assert vp_c is not None and op_c is not None
                                v_tmp = vp_c(x_p)
                                rep = n_heads // n_kv_heads
                                v_tmp = v_tmp.view(*x_p.shape[:-1], n_kv_heads, head_dim)
                                v_tmp = v_tmp.repeat_interleave(rep, dim=-2).reshape(
                                    *x_p.shape[:-1], d_q
                                )
                                y_base = op_c(v_tmp).to(torch.float32)

                            if scale_o_bias:
                                y_lin = y_base
                                y_tgt = y_t
                            else:
                                y_lin = y_base - b_o
                                y_tgt = y_t - b_o

                            denom = float((y_lin * y_lin).sum().item())
                            if denom > 1e-30:
                                alpha_fit = float((y_tgt * y_lin).sum().item()) / denom
                            else:
                                alpha_fit = 1.0
                            if not math.isfinite(alpha_fit):
                                continue
                            y_fit = (
                                alpha_fit * y_lin
                                if scale_o_bias
                                else (alpha_fit * y_lin + b_o)
                            )
                            rel_err = float((y_fit - y_t).abs().max().item()) / ref_mag
                            if rel_err < best_err:
                                best_err = rel_err
                                best_choice = (v_cand, alpha_fit, scale_o_bias)
                                found_valid = rel_err <= 1e-3

                if not found_valid:
                    raise ValueError(
                        f"Singleton variate probe mismatch (best rel_err={best_err:.4e})"
                    )

        v_best, alpha_best, scale_o_bias_best = best_choice
        self.w_ov, self.v_proj, self.o_proj = _build_ov_module(
            v_best,
            o_lin,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            d_q=d_q,
            alpha=alpha_best,
            scale_o_bias=scale_o_bias_best,
        )

    def _apply_singleton(self, x: Any) -> Any:
        if self.w_ov is not None:
            return self.w_ov(x)
        v = self.v_proj(x)
        rep = self.n_heads // self.n_kv_heads
        v = v.view(*x.shape[:-1], self.n_kv_heads, self.head_dim)
        v = v.repeat_interleave(rep, dim=-2).reshape(*x.shape[:-1], self.d_q)
        return self.o_proj(v)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if not self.enabled:
            return self.orig_attn(*args, **kwargs)

        x = args[0] if len(args) > 0 else kwargs.get("x", kwargs.get("hidden_states"))
        if not isinstance(x, torch.Tensor):
            return self.orig_attn(*args, **kwargs)

        if self._expected_ndim is not None and self._singleton_dim is not None:
            is_singleton_var = (
                x.ndim == self._expected_ndim and x.shape[self._singleton_dim] == 1
            )
        else:
            is_singleton_var = (x.ndim == 3 and x.shape[1] == 1) or (
                x.ndim == 4 and (x.shape[1] == 1 or x.shape[2] == 1)
            )
        if not is_singleton_var:
            return self.orig_attn(*args, **kwargs)

        out = self._apply_singleton(x)
        if self._returns_tuple:
            return (out,) + (None,) * max(self._tuple_len - 1, 1)
        return out


def _capture_variate_attn_probe_io(
    backbone: Any,
    layers: list[Any],
    arch: Toto2ArchitectureInfo,
    probe_x: Any,
    horizon: int,
    num_quantiles_hint: int,
) -> dict[int, tuple[Any, Any]]:
    """Capture ground-truth ``(x_in, y_out)`` on variate attention layers during a tiny probe."""
    captured: dict[int, tuple[Any, Any]] = {}
    handles: list[Any] = []
    for idx in arch.variate_layer_indices:
        layer = layers[idx]
        for attr_name in ("attention", "attn", "self_attn"):
            cand = getattr(layer, attr_name, None)
            if cand is not None and isinstance(cand, nn.Module):

                def _hook_kw(
                    _mod: Any,
                    inp: tuple[Any, ...],
                    kw: dict[str, Any],
                    out: Any,
                    _idx: int = idx,
                ) -> None:
                    if _idx in captured:
                        return
                    x_val = inp[0] if len(inp) > 0 else kw.get("x", kw.get("hidden_states"))
                    if isinstance(x_val, torch.Tensor):
                        captured[_idx] = (x_val.detach(), out)

                def _hook_pos(
                    _mod: Any,
                    inp: tuple[Any, ...],
                    out: Any,
                    _idx: int = idx,
                ) -> None:
                    if _idx not in captured and len(inp) > 0 and isinstance(inp[0], torch.Tensor):
                        captured[_idx] = (inp[0].detach(), out)

                try:
                    handles.append(cand.register_forward_hook(_hook_kw, with_kwargs=True))
                except TypeError:
                    handles.append(cand.register_forward_hook(_hook_pos))
                break

    try:
        with torch.inference_mode(), tf32_precision_context("fp32"):
            _call_toto2_native_forecast(
                backbone,
                probe_x,
                horizon,
                num_quantiles_hint=num_quantiles_hint,
                has_missing_override=True,
            )
    except Exception:
        captured.clear()
    finally:
        for h in handles:
            h.remove()
    return captured


def _install_native_singleton_variate_collapse(
    backbone: Any,
    layers: list[Any],
    arch: Toto2ArchitectureInfo,
    *,
    probe_io_map: dict[int, tuple[Any, Any]] | None = None,
) -> list[tuple[Any, str, _SingletonVariateAttnWrapper, int]]:
    """Wrap variate attention submodules in-place for singleton variate collapse."""
    _ = backbone
    installed: list[tuple[Any, str, _SingletonVariateAttnWrapper, int]] = []
    for idx in arch.variate_layer_indices:
        layer = layers[idx]
        for attr_name in ("attention", "attn", "self_attn"):
            cand = getattr(layer, attr_name, None)
            if cand is not None and not isinstance(cand, _SingletonVariateAttnWrapper):
                try:
                    probe_io = None if probe_io_map is None else probe_io_map.get(idx)
                    wrapper = _SingletonVariateAttnWrapper(
                        cand, arch.num_heads, probe_io=probe_io
                    )
                    setattr(layer, attr_name, wrapper)
                    installed.append((layer, attr_name, wrapper, idx))
                except Exception:
                    pass
                break
    return installed


def _strip_toto2_inputs_f64(
    stripper: CausalFloat64DCStripper,
    x_2d: Any,
    patch_size: int,
    forces_last_patch_observed: bool,
) -> tuple[Any, Any]:
    """Shift-equivariant FP64 DC stripping matching ``Toto2Model._prepare_forecast_inputs``.

    Real ``Toto2Model._prepare_forecast_inputs`` forces ``target_mask[..., -P:] = True``
    and zero-fills missing entries in raw space BEFORE calling ``self.scaler``. To remain
    strictly shift-equivariant on NaN-bearing series, replace non-finite entries in the
    final ``patch_size`` positions with ``0.0`` BEFORE subtracting ``dc_f64`` so
    ``(0.0 - dc_f64) + dc_f64 == 0.0``. Earlier positions keep ``target_mask = isfinite(x)``
    (``False`` at NaNs, zero-filled after stripping).
    """
    x_f64 = x_2d.to(torch.float64)
    if not stripper.enabled:
        dc = torch.zeros((*x_f64.shape[:-1], 1), dtype=torch.float64, device=x_f64.device)
        return x_f64.to(torch.float32), dc

    dc = stripper._last_finite_torch(x_f64)
    if forces_last_patch_observed and x_f64.shape[-1] > 0:
        p_tail = min(int(patch_size), int(x_f64.shape[-1]))
        tail = x_f64[..., -p_tail:]
        tail_clean = torch.where(torch.isfinite(tail), tail, torch.zeros_like(tail))
        if x_f64.shape[-1] > p_tail:
            x_work = torch.cat([x_f64[..., :-p_tail], tail_clean], dim=-1)
        else:
            x_work = tail_clean
    else:
        x_work = x_f64

    return (x_work - dc).to(torch.float32), dc


def _resolve_layer_unit_scaling(layer: Any) -> Any:
    """Resolve the exact ``unit_scaling.functional`` module used by a Toto-2.0 layer."""
    for attr in ("_us_functional", "us_functional"):
        cand = getattr(layer, attr, None)
        if cand is not None and hasattr(cand, "residual_split") and hasattr(cand, "residual_add"):
            return cand
    try:
        mod = importlib.import_module(type(layer).__module__)
        for attr in ("U", "F_us", "us_functional"):
            cand = getattr(mod, attr, None)
            if (
                cand is not None
                and hasattr(cand, "residual_split")
                and hasattr(cand, "residual_add")
            ):
                return cand
        for val in vars(mod).values():
            if hasattr(val, "residual_split") and hasattr(val, "residual_add"):
                return val
    except Exception:
        pass
    return _us_functional


def _us_residual_split(us_mod: Any, x: Any, tau: Any) -> tuple[Any, Any]:
    """Call ``U.residual_split(x, tau)`` preserving exact Toto-2.0 ``(x_in, skip)`` order."""
    if us_mod is not None and tau is not None and hasattr(us_mod, "residual_split"):
        return us_mod.residual_split(x, tau)
    return x, x


def _us_residual_add(us_mod: Any, branch: Any, skip: Any, tau: Any) -> Any:
    """Call ``U.residual_add(branch, skip, tau)`` preserving exact Toto-2.0 arg order."""
    if us_mod is not None and tau is not None and hasattr(us_mod, "residual_add"):
        return us_mod.residual_add(branch, skip, tau)
    return branch + skip


def _inspect_layer_call_sig(mod: Any) -> tuple[bool, bool, bool]:
    """Return ``(takes_seq_ids, takes_attn_mask, takes_is_causal)`` for a Toto-2.0 layer."""
    try:
        params = inspect.signature(mod.forward).parameters
    except (TypeError, ValueError):
        return True, True, False
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    takes_seq_ids = ("seq_ids" in params) or has_var_kw
    takes_attn_mask = ("attn_mask" in params) or has_var_kw
    takes_is_causal = "is_causal" in params
    return takes_seq_ids, takes_attn_mask, takes_is_causal


def _invoke_layer_with_sig(
    mod: Any,
    x: Any,
    seq_ids: Any,
    attn_mask: Any,
    takes_seq_ids: bool,
    takes_attn_mask: bool,
    takes_is_causal: bool,
    *,
    is_causal_override: bool | None = None,
) -> Any:
    kwargs: dict[str, Any] = {}
    if takes_seq_ids:
        kwargs["seq_ids"] = seq_ids
    if takes_attn_mask:
        kwargs["attn_mask"] = attn_mask
    if takes_is_causal:
        kwargs["is_causal"] = (
            (attn_mask is None) if is_causal_override is None else bool(is_causal_override)
        )
    out = mod(x, **kwargs) if kwargs else mod(x)
    return out[0] if isinstance(out, tuple) else out


def _is_direct_toto2_architecture(
    backbone: Any, layers: list[Any], arch: Toto2ArchitectureInfo
) -> bool:
    """Return True if ``backbone`` matches standard Toto-2.0 (4m/22m/313m) layout."""
    if getattr(backbone, "scaler", None) is None:
        return False
    if not callable(getattr(backbone, "_embed_patches", None)):
        return False
    transformer = getattr(backbone, "transformer", None)
    if (
        transformer is None
        or not hasattr(transformer, "layers")
        or not hasattr(transformer, "out_norm")
    ):
        return False
    if getattr(backbone, "output_head", None) is None:
        return False
    if len(layers) < 2 or arch.final_time_layer_idx < 0:
        return False

    for layer in layers:
        attn = getattr(layer, "attn", None)
        if attn is None:
            return False
        if not (
            hasattr(layer, "norm1")
            and hasattr(layer, "norm2")
            and hasattr(layer, "ffn")
            and is_linear_module(getattr(attn, "in_proj", None))
            and is_linear_module(getattr(attn, "out_proj", None))
            and hasattr(attn, "_split_sizes")
            and hasattr(attn, "_Hq")
            and hasattr(attn, "_Hkv")
            and hasattr(attn, "is_variate_layer")
        ):
            return False
        split_sizes = attn._split_sizes
        if len(split_sizes) != 3 or int(split_sizes[1]) != int(split_sizes[2]):
            return False
    return True


def _apply_pds_tensor(t: Any, pds: Any, layout: str) -> Any:
    """Broadcast a per-dim scale tensor onto ``(B, H, L, Dh)`` or ``(B, L, H, Dh)``."""
    if pds.ndim == 0:
        return t * pds
    if pds.ndim == 1:
        if pds.shape[0] == t.shape[-1]:
            return t * pds
        if layout == "bhsd" and pds.shape[0] == t.shape[1]:
            return t * pds.view(1, pds.shape[0], 1, 1)
        if layout == "bshd" and pds.shape[0] == t.shape[2]:
            return t * pds.view(1, 1, pds.shape[0], 1)
        return t * pds
    if pds.ndim == 2:
        if layout == "bhsd" and pds.shape[0] == t.shape[1]:
            return t * pds.view(1, pds.shape[0], 1, pds.shape[1])
        if layout == "bshd" and pds.shape[0] == t.shape[2]:
            return t * pds.view(1, 1, pds.shape[0], pds.shape[1])
        return t * pds
    if pds.ndim == 3 and layout == "bhsd" and pds.shape[0] == t.shape[1]:
        return t * pds.unsqueeze(0)
    if pds.ndim == 4:
        if layout == "bhsd" and pds.shape[1] == 1 and pds.shape[2] == t.shape[1]:
            return t * pds.transpose(1, 2)
        if layout == "bshd" and pds.shape[2] == 1 and pds.shape[1] == t.shape[2]:
            return t * pds.transpose(1, 2)
    return t * pds


class _CollapsedToto2VariateLayer(_BaseModule):  # type: ignore[misc]
    """Exact Pass 2 (R2) Singleton Variate Collapse for Toto-2.0 ``SelfAttentionTransformerLayer``.

    Operates directly on ``(B, S, D)`` (intermediate variate layers in multi-group
    models such as 313m) or ``(B, 1, D)`` (terminal variate layer after R1), completely
    eliminating ``(B*S, 1, D)`` reshapes and length-1 SDPA (fixing the ``B >= 1009``
    CUDA ``gridDim`` crash at ``B=4096``).
    """

    def __init__(self, orig_layer: Any, us_mod: Any) -> None:
        super().__init__()
        attn = orig_layer.attn
        dq, dk, dv = (int(s) for s in attn._split_sizes)
        self.Hq = int(attn._Hq)
        self.Hkv = int(attn._Hkv)
        self.head_dim = dq // max(self.Hq, 1)
        self.dq = dq
        self.dv = dv
        self.heads_per_group = int(
            getattr(getattr(attn, "config", None), "heads_per_group", self.Hq // max(self.Hkv, 1))
        )

        self._us = us_mod
        self.attn_tau = getattr(orig_layer, "attn_tau", None)
        self.mlp_tau = getattr(orig_layer, "mlp_tau", None)
        self.norm1 = orig_layer.norm1
        self.norm2 = orig_layer.norm2
        self.v_proj = _slice_linear_rows(attn.in_proj, dq + dk, dq + dk + dv)
        self.v_norm = getattr(attn, "v_norm", None)
        self.out_proj = attn.out_proj
        self.ffn = orig_layer.ffn
        self.attn_resid_dropout = getattr(orig_layer, "attn_resid_dropout", None)
        self.mlp_resid_dropout = getattr(orig_layer, "mlp_resid_dropout", None)

        # Verify exact numerical equivalence against live orig_layer on singleton (1, 1, D).
        dev = attn.in_proj.weight.device
        dtype = attn.in_proj.weight.dtype
        d_in = int(getattr(attn.in_proj, "in_features", attn.in_proj.weight.shape[1]))
        gen = torch.Generator(device="cpu").manual_seed(20261)
        x_probe = torch.randn((1, 1, d_in), generator=gen, dtype=torch.float32).to(
            device=dev, dtype=dtype
        )
        takes_seq, takes_mask, takes_causal = _inspect_layer_call_sig(orig_layer)
        with torch.inference_mode(), tf32_precision_context("fp32"):
            y_ref = None
            last_exc: Exception | None = None
            for seq_cand in (
                None,
                torch.zeros((1,), dtype=torch.long, device=dev),
                torch.zeros((1, 1), dtype=torch.long, device=dev),
            ):
                try:
                    y_ref = _invoke_layer_with_sig(
                        orig_layer,
                        x_probe,
                        seq_cand,
                        None,
                        takes_seq,
                        takes_mask,
                        takes_causal,
                        is_causal_override=False,
                    )
                    break
                except Exception as exc:
                    last_exc = exc
            if y_ref is None:
                raise RuntimeError(f"Unable to probe variate layer: {last_exc}") from last_exc
            y_got = self.forward(x_probe)
            ref_mag = max(float(y_ref.to(torch.float32).abs().max().item()), 1e-6)
            rel_err = float((y_got.to(torch.float32) - y_ref.to(torch.float32)).abs().max().item())
            if not math.isfinite(rel_err) or (rel_err / ref_mag) > 1e-4:
                raise ValueError(
                    f"Collapsed Toto2 variate layer probe mismatch: rel_err={rel_err / ref_mag:.3e}"
                )

    def forward(self, x: Any) -> Any:
        x_in, skip = _us_residual_split(self._us, x, self.attn_tau)
        h = self.norm1(x_in)
        v = self.v_proj(h)
        if self.v_norm is not None:
            B, S, _ = v.shape
            v = self.v_norm(v.view(B, S, self.Hkv, self.head_dim)).reshape(B, S, self.dv)
        if self.Hkv != self.Hq:
            B, S, _ = v.shape
            v = (
                v.view(B, S, self.Hkv, 1, self.head_dim)
                .expand(B, S, self.Hkv, self.heads_per_group, self.head_dim)
                .reshape(B, S, self.dq)
            )
        x_attn = self.out_proj(v)
        if self.attn_resid_dropout is not None:
            x_attn = self.attn_resid_dropout(x_attn)
        x = _us_residual_add(self._us, x_attn, skip, self.attn_tau)

        x_in2, skip2 = _us_residual_split(self._us, x, self.mlp_tau)
        x_ffn = self.ffn(self.norm2(x_in2))
        if self.mlp_resid_dropout is not None:
            x_ffn = self.mlp_resid_dropout(x_ffn)
        return _us_residual_add(self._us, x_ffn, skip2, self.mlp_tau)


class _ContractedToto2TimeLayer(_BaseModule):  # type: ignore[misc]
    """Exact Pass 1 (R1) Anchor-Query Contraction for the final Toto-2.0 time layer.

    Computes ``Q`` ONLY for the final anchor patch token ``h[:, -1:, :]`` and runs
    ``FFN`` ONLY on ``(B, 1, D)``, eliminating ``(S-1)/S`` of query + FFN FLOPs while
    matching unmodified ``SelfAttentionTransformerLayer`` to ``< 1e-4``.
    """

    def __init__(self, orig_layer: Any, us_mod: Any) -> None:
        super().__init__()
        attn = orig_layer.attn
        dq, dk, dv = (int(s) for s in attn._split_sizes)
        self.Hq = int(attn._Hq)
        self.Hkv = int(attn._Hkv)
        self.head_dim = dq // max(self.Hq, 1)
        self.dq = dq
        self.dk = dk
        self.dv = dv
        self.heads_per_group = int(
            getattr(getattr(attn, "config", None), "heads_per_group", self.Hq // max(self.Hkv, 1))
        )

        self._us = us_mod
        self.attn_tau = getattr(orig_layer, "attn_tau", None)
        self.mlp_tau = getattr(orig_layer, "mlp_tau", None)
        self.norm1 = orig_layer.norm1
        self.norm2 = orig_layer.norm2
        self.q_proj = _slice_linear_rows(attn.in_proj, 0, dq)
        self.kv_proj = _slice_linear_rows(attn.in_proj, dq, dq + dk + dv)
        self.out_proj = attn.out_proj
        self.ffn = orig_layer.ffn
        self.q_norm = getattr(attn, "q_norm", None)
        self.k_norm = getattr(attn, "k_norm", None)
        self.qk_proj = getattr(attn, "qk_proj", None)
        self.attn_resid_dropout = getattr(orig_layer, "attn_resid_dropout", None)
        self.mlp_resid_dropout = getattr(orig_layer, "mlp_resid_dropout", None)

        pds = getattr(attn, "_pds", None)
        self._pds_mod: Any = None
        self._pds_scalar: float | None = None
        if isinstance(pds, nn.Module):
            self._pds_mod = pds
            self._pds_tensor: Any = None
        elif isinstance(pds, nn.Parameter):
            self._pds_tensor = pds
        elif isinstance(pds, torch.Tensor):
            self.register_buffer("_pds_tensor", pds, persistent=False)
        elif callable(pds):
            self._pds_mod = pds
            self._pds_tensor = None
        elif isinstance(pds, (int, float)) and not isinstance(pds, bool):
            self._pds_tensor = None
            self._pds_scalar = float(pds)
        else:
            self._pds_tensor = None

        # Immutable static execution configuration calibrated once against live orig_layer.
        mult_attr = getattr(attn, "mult", None)
        self._layout: str = "bhsd"
        self._pds_mode: str = "none"
        self._qk_mode: str = "none" if self.qk_proj is None else "query_kv_ids"
        self._expand_before_qk: bool = False
        self._seq_ids_ndim: int = 1
        self._sdpa_scale: float = (
            float(mult_attr) if mult_attr is not None else (1.0 / math.sqrt(self.head_dim))
        )

        self._calibrate_against_orig_layer(orig_layer, pds, mult_attr)

    def _apply_pds(self, q: Any, k: Any, mode: str, layout: str) -> tuple[Any, Any]:
        if mode == "none":
            return q, k
        if mode == "call_q":
            return self._pds_mod(q), k
        if mode == "call_qk":
            return self._pds_mod(q), self._pds_mod(k)
        if mode == "call_pair":
            return self._pds_mod(q, k)
        if mode == "mul_q":
            if self._pds_tensor is not None:
                return _apply_pds_tensor(q, self._pds_tensor, layout), k
            return q * self._pds_scalar, k
        if mode == "mul_qk":
            if self._pds_tensor is not None:
                return (
                    _apply_pds_tensor(q, self._pds_tensor, layout),
                    _apply_pds_tensor(k, self._pds_tensor, layout),
                )
            return q * self._pds_scalar, k * self._pds_scalar
        if mode == "one_plus_q" and self._pds_tensor is not None:
            return _apply_pds_tensor(q, 1.0 + self._pds_tensor, layout), k
        if mode == "one_plus_qk" and self._pds_tensor is not None:
            sc = 1.0 + self._pds_tensor
            return _apply_pds_tensor(q, sc, layout), _apply_pds_tensor(k, sc, layout)
        if mode == "softplus_q" and self._pds_tensor is not None:
            return _apply_pds_tensor(q, F.softplus(self._pds_tensor), layout), k
        if mode == "softplus_r0_q" and self._pds_tensor is not None:
            scale_t = F.softplus(self._pds_tensor) * (1.0 / math.log(2.0))
            return _apply_pds_tensor(q, scale_t, layout), k
        if mode == "exp_q" and self._pds_tensor is not None:
            return _apply_pds_tensor(q, self._pds_tensor.exp(), layout), k
        return q, k

    def _apply_qk(
        self, q: Any, k: Any, time_ids: Any, qk_mode: str, layout: str
    ) -> tuple[Any, Any]:
        if self.qk_proj is None or qk_mode == "none":
            return q, k
        q_ids = time_ids[-1:] if time_ids.ndim == 1 else time_ids[:, -1:]
        if qk_mode == "query_kv_ids":
            return self.qk_proj(q, k, query_ids=q_ids, kv_ids=time_ids)
        if qk_mode == "q_k_ids":
            return self.qk_proj(q, k, q_ids=q_ids, k_ids=time_ids)
        if qk_mode == "pos_qk_ids":
            return self.qk_proj(q, k, q_ids, time_ids)
        if qk_mode == "seq_ids_pad":
            # Pad Q left to S tokens when a variant qk_proj only takes (q, k, seq_ids).
            if layout == "bhsd":
                S = k.shape[2]
                q_pad = torch.zeros(
                    (q.shape[0], q.shape[1], S, q.shape[3]), dtype=q.dtype, device=q.device
                )
                q_pad[:, :, -1:, :] = q
                q_rot, k_rot = self.qk_proj(q_pad, k, seq_ids=time_ids)
                return q_rot[:, :, -1:, :], k_rot
            S = k.shape[1]
            q_pad = torch.zeros(
                (q.shape[0], S, q.shape[2], q.shape[3]), dtype=q.dtype, device=q.device
            )
            q_pad[:, -1:, :, :] = q
            q_rot, k_rot = self.qk_proj(q_pad, k, seq_ids=time_ids)
            return q_rot[:, -1:, :, :], k_rot
        if qk_mode == "pos_args":
            return self.qk_proj(q, k)
        return q, k

    def _run_attn_with_cfg(
        self,
        h: Any,
        time_ids: Any,
        attn_mask_last: Any,
        *,
        layout: str,
        pds_mode: str,
        qk_mode: str,
        expand_before_qk: bool,
        sdpa_scale: float,
    ) -> Any:
        B, S, _ = h.shape
        q = self.q_proj(h[:, -1:, :])
        kv = self.kv_proj(h)
        k, v = kv.split((self.dk, self.dv), dim=-1)

        if layout == "bhsd":
            q = q.view(B, 1, self.Hq, self.head_dim).transpose(1, 2)
            k = k.view(B, S, self.Hkv, self.head_dim).transpose(1, 2)
            v = v.view(B, S, self.Hkv, self.head_dim).transpose(1, 2)
        else:
            q = q.view(B, 1, self.Hq, self.head_dim)
            k = k.view(B, S, self.Hkv, self.head_dim)
            v = v.view(B, S, self.Hkv, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        pre_pds = pds_mode[5:] if pds_mode.startswith("post_") else pds_mode
        if not pds_mode.startswith("post_"):
            q, k = self._apply_pds(q, k, pre_pds, layout)

        if expand_before_qk and self.Hkv != self.Hq:
            if layout == "bhsd":
                k = (
                    k.unsqueeze(2)
                    .expand(B, self.Hkv, self.heads_per_group, S, self.head_dim)
                    .reshape(B, self.Hq, S, self.head_dim)
                )
            else:
                k = (
                    k.unsqueeze(3)
                    .expand(B, S, self.Hkv, self.heads_per_group, self.head_dim)
                    .reshape(B, S, self.Hq, self.head_dim)
                )

        q, k = self._apply_qk(q, k, time_ids, qk_mode, layout)

        if pds_mode.startswith("post_"):
            q, k = self._apply_pds(q, k, pre_pds, layout)

        if layout == "bshd":
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)

        if self.Hkv != self.Hq:
            if not expand_before_qk:
                k = (
                    k.unsqueeze(2)
                    .expand(B, self.Hkv, self.heads_per_group, S, self.head_dim)
                    .reshape(B, self.Hq, S, self.head_dim)
                )
            v = (
                v.unsqueeze(2)
                .expand(B, self.Hkv, self.heads_per_group, S, self.head_dim)
                .reshape(B, self.Hq, S, self.head_dim)
            )

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask_last,
            dropout_p=0.0,
            is_causal=False,
            scale=sdpa_scale,
        )
        out = out.transpose(1, 2).contiguous().view(B, 1, self.dq)
        return self.out_proj(out)

    def _calibrate_against_orig_layer(self, orig_layer: Any, pds: Any, mult_attr: Any) -> None:
        attn = orig_layer.attn
        dev = attn.in_proj.weight.device
        dtype = attn.in_proj.weight.dtype
        d_in = int(getattr(attn.in_proj, "in_features", attn.in_proj.weight.shape[1]))
        gen = torch.Generator(device="cpu").manual_seed(20262)
        x_probe = torch.randn((1, 4, d_in), generator=gen, dtype=torch.float32).to(
            device=dev, dtype=dtype
        )

        takes_seq_l, takes_mask_l, takes_causal_l = _inspect_layer_call_sig(orig_layer)
        takes_seq_a, takes_mask_a, takes_causal_a = _inspect_layer_call_sig(attn)

        with torch.inference_mode(), tf32_precision_context("fp32"):
            x_in_p, _ = _us_residual_split(self._us, x_probe, self.attn_tau)
            h_probe = self.norm1(x_in_p)

            # Auto-detect 1D (S,) vs 2D (B, S) seq_ids convention on the live layer.
            probe_ids_1d = torch.arange(4, device=dev, dtype=torch.long)
            probe_ids_2d = probe_ids_1d.unsqueeze(0)
            probe_ids = probe_ids_1d
            y_attn_ref = None
            y_layer_ref = None
            last_exc: Exception | None = None
            for seq_ndim, cand_ids in ((1, probe_ids_1d), (2, probe_ids_2d)):
                try:
                    y_a = _invoke_layer_with_sig(
                        attn, h_probe, cand_ids, None, takes_seq_a, takes_mask_a, takes_causal_a
                    )[:, -1:, :]
                    y_l = _invoke_layer_with_sig(
                        orig_layer,
                        x_probe,
                        cand_ids,
                        None,
                        takes_seq_l,
                        takes_mask_l,
                        takes_causal_l,
                    )[:, -1:, :]
                    self._seq_ids_ndim = seq_ndim
                    probe_ids = cand_ids
                    y_attn_ref = y_a
                    y_layer_ref = y_l
                    break
                except Exception as exc:
                    last_exc = exc
            if y_attn_ref is None or y_layer_ref is None:
                raise RuntimeError(f"Unable to probe R1 time layer: {last_exc}") from last_exc

            if pds is None:
                pds_cands: tuple[str, ...] = ("none",)
            elif callable(pds):
                pds_cands = (
                    "call_q",
                    "call_qk",
                    "call_pair",
                    "post_call_q",
                    "post_call_qk",
                    "post_call_pair",
                    "none",
                )
            elif isinstance(pds, torch.Tensor):
                pds_cands = (
                    "mul_q",
                    "mul_qk",
                    "softplus_q",
                    "softplus_r0_q",
                    "exp_q",
                    "one_plus_q",
                    "one_plus_qk",
                    "post_mul_q",
                    "post_mul_qk",
                    "post_softplus_q",
                    "post_softplus_r0_q",
                    "post_exp_q",
                    "none",
                )
            elif isinstance(pds, (int, float)) and not isinstance(pds, bool):
                pds_cands = ("mul_q", "mul_qk", "post_mul_q", "post_mul_qk", "none")
            else:
                pds_cands = ("none", "mul_q")

            qk_cands: tuple[str, ...] = (
                ("none",)
                if self.qk_proj is None
                else ("query_kv_ids", "seq_ids_pad", "q_k_ids", "pos_qk_ids", "pos_args")
            )
            scale_cands: list[float] = []
            if mult_attr is not None:
                scale_cands.append(float(mult_attr))
                scale_cands.append(float(mult_attr) / math.sqrt(self.head_dim))
            default_sc = 1.0 / math.sqrt(self.head_dim)
            if all(abs(default_sc - s) > 1e-12 for s in scale_cands):
                scale_cands.append(default_sc)
            expand_cands = (False, True) if self.Hkv != self.Hq else (False,)

            ref_mag = max(float(y_attn_ref.to(torch.float32).abs().max().item()), 1e-6)
            matched = False
            best_err = float("inf")

            for layout in ("bhsd", "bshd"):
                for qk_mode in qk_cands:
                    for pds_mode in pds_cands:
                        for expand_before_qk in expand_cands:
                            for sdpa_scale in scale_cands:
                                try:
                                    y_try = self._run_attn_with_cfg(
                                        h_probe,
                                        probe_ids,
                                        None,
                                        layout=layout,
                                        pds_mode=pds_mode,
                                        qk_mode=qk_mode,
                                        expand_before_qk=expand_before_qk,
                                        sdpa_scale=sdpa_scale,
                                    )
                                except Exception:
                                    continue
                                err = (
                                    float(
                                        (y_try.to(torch.float32) - y_attn_ref.to(torch.float32))
                                        .abs()
                                        .max()
                                        .item()
                                    )
                                    / ref_mag
                                )
                                if err < best_err:
                                    best_err = err
                                if math.isfinite(err) and err <= 1e-4:
                                    self._layout = layout
                                    self._pds_mode = pds_mode
                                    self._qk_mode = qk_mode
                                    self._expand_before_qk = expand_before_qk
                                    self._sdpa_scale = sdpa_scale
                                    matched = True
                                    break
                            if matched:
                                break
                        if matched:
                            break
                    if matched:
                        break
                if matched:
                    break

            if not matched:
                raise ValueError(
                    f"Contracted Toto2 R1 attention calibration failed "
                    f"(best rel_err={best_err:.3e})"
                )

            # Verify full R1 layer (including residual_split/add + FFN) against orig_layer.
            y_layer_got = self.forward(x_probe, probe_ids, None)
            layer_mag = max(float(y_layer_ref.to(torch.float32).abs().max().item()), 1e-6)
            layer_err = (
                float(
                    (y_layer_got.to(torch.float32) - y_layer_ref.to(torch.float32))
                    .abs()
                    .max()
                    .item()
                )
                / layer_mag
            )
            if not math.isfinite(layer_err) or layer_err > 1e-4:
                raise ValueError(
                    f"Contracted Toto2 R1 layer probe mismatch: rel_err={layer_err:.3e}"
                )

    def forward(self, x: Any, time_ids: Any, attn_mask_last: Any | None = None) -> Any:
        x_in, skip = _us_residual_split(self._us, x, self.attn_tau)
        h = self.norm1(x_in)
        x_attn = self._run_attn_with_cfg(
            h,
            time_ids,
            attn_mask_last,
            layout=self._layout,
            pds_mode=self._pds_mode,
            qk_mode=self._qk_mode,
            expand_before_qk=self._expand_before_qk,
            sdpa_scale=self._sdpa_scale,
        )
        if self.attn_resid_dropout is not None:
            x_attn = self.attn_resid_dropout(x_attn)
        x_last = _us_residual_add(self._us, x_attn, skip[:, -1:, :], self.attn_tau)

        x_in2, skip2 = _us_residual_split(self._us, x_last, self.mlp_tau)
        x_ffn = self.ffn(self.norm2(x_in2))
        if self.mlp_resid_dropout is not None:
            x_ffn = self.mlp_resid_dropout(x_ffn)
        return _us_residual_add(self._us, x_ffn, skip2, self.mlp_tau)


class _CompiledToto2Core(_BaseModule):  # type: ignore[misc]
    """Dedicated ``torch.compile``-friendly direct R1 + R2 Toto-2.0 inference core.

    Runs when ``horizon <= patch_size`` and ``n_var == 1``:
    1. Exact 3D ``scaler(x_clean, target_mask_3d)`` + ``asinh`` + ``_embed_patches``
       with ZERO dummy ``pred_x`` patch (``S = L_ctx // P``).
    2. Prefix layers ``0 .. final_time_idx - 1`` (time layers on ``(B, S, D)`` with fast
       ``is_causal=True`` on clean inputs or exact additive ``0.0 / -inf`` mask when any
       patch is all-missing; intermediate variate layers collapsed on ``(B, S, D)``).
    3. Pass 1 (R1) anchor-query contraction on final time layer ``final_time_idx`` -> ``(B, 1, D)``.
    4. Pass 2 (R2) singleton variate collapse on trailing variate layer(s) on ``(B, 1, D)``
       (zero ``(B*S, 1, D)`` reshape, eliminating ``B=4096`` CUDA ``gridDim`` crash!).
    5. ``out_norm`` -> ``output_head(pred_out, q=None)[..., :1, :]`` -> ``sinh * scale + loc``
       -> branchless non-finite clamp -> monotonic sort -> ``(B, Q, horizon)``.
    """

    def __init__(
        self,
        backbone: Any,
        layers: list[Any],
        arch: Toto2ArchitectureInfo,
        *,
        horizon: int,
        quantiles: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        self.patch_size = int(arch.patch_size)
        self.horizon = int(horizon)
        self.scaler = backbone.scaler
        self.out_norm = backbone.transformer.out_norm
        self.output_head = backbone.output_head
        # Keep non-submodule reference box so named_children() only walks live trunk layers.
        self._backbone_box = [backbone]
        self._embed_patches_fn = backbone._embed_patches

        try:
            embed_params = [
                p
                for p in inspect.signature(backbone._embed_patches).parameters.values()
                if p.name != "self"
                and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            ]
            self._embed_takes_patch_size = len(embed_params) >= 3 or any(
                p.name == "patch_size" for p in embed_params
            )
        except (TypeError, ValueError):
            self._embed_takes_patch_size = True

        try:
            head_params = inspect.signature(self.output_head.forward).parameters
            self._head_takes_q = "q" in head_params
        except (TypeError, ValueError):
            self._head_takes_q = True

        self._head_nop = max(1, int(getattr(self.output_head, "num_output_patches", 1)))
        self._use_custom_q = (
            quantiles is not None
            and tuple(float(v) for v in quantiles) != Toto2CompiledEngine.DEFAULT_QUANTILES
            and self._head_takes_q
        )

        us_mod = _resolve_layer_unit_scaling(layers[arch.final_time_layer_idx])
        final_t = arch.final_time_layer_idx
        var_set = set(arch.variate_layer_indices)

        prefix_mods: list[Any] = []
        prefix_is_var: list[bool] = []
        prefix_sigs: list[tuple[bool, bool, bool]] = []
        for idx in range(final_t):
            if idx in var_set:
                prefix_mods.append(_CollapsedToto2VariateLayer(layers[idx], us_mod))
                prefix_is_var.append(True)
                prefix_sigs.append((False, False, False))
            else:
                prefix_mods.append(layers[idx])
                prefix_is_var.append(False)
                prefix_sigs.append(_inspect_layer_call_sig(layers[idx]))

        self.prefix_layers = nn.ModuleList(prefix_mods)
        self._prefix_is_variate = tuple(prefix_is_var)
        self._prefix_sigs = tuple(prefix_sigs)
        self.r1_layer = _ContractedToto2TimeLayer(layers[final_t], us_mod)
        self._seq_ids_ndim: int = self.r1_layer._seq_ids_ndim
        self.suffix_layers = nn.ModuleList(
            [
                _CollapsedToto2VariateLayer(layers[idx], us_mod)
                for idx in range(final_t + 1, len(layers))
            ]
        )

        # Compile-time scaler anchor verification: confirm loc[:, 0, -1:] matches
        # static_loc[..., pred_start:pred_end] (with 1 masked future patch); if a
        # strictly-causal scaler differs, auto-enable future-patch pad on scaler ONLY.
        dev = self.r1_layer.q_proj.weight.device
        w_dtype = self.r1_layer.q_proj.weight.dtype
        self._scaler_needs_future_pad = False
        self._scaler_out_kind = "pair"  # 'pair' | 'triple' | 'nested_pair'
        self._calibrate_scaler_and_head(dev, w_dtype, quantiles)

    @staticmethod
    def _unpack_scaler_out(scaler_out: Any) -> tuple[Any, Any, str]:
        if isinstance(scaler_out, tuple) and len(scaler_out) >= 3:
            return scaler_out[1], scaler_out[2], "triple"
        if isinstance(scaler_out, tuple) and len(scaler_out) == 2:
            if isinstance(scaler_out[1], tuple) and len(scaler_out[1]) == 2:
                return scaler_out[1][0], scaler_out[1][1], "nested_pair"
            return scaler_out[0], scaler_out[1], "pair"
        raise ValueError("Unexpected return signature from Toto-2.0 scaler")

    def _calibrate_scaler_and_head(
        self,
        dev: Any,
        w_dtype: Any,
        quantiles: Sequence[float] | None,
    ) -> None:
        P = self.patch_size
        t = torch.linspace(0.5, 6.5, 2 * P, dtype=torch.float32, device=dev)
        x_p = (2.0 + torch.sin(t) + 0.5 * t).view(1, 1, 2 * P).to(dtype=w_dtype)
        m_p = torch.ones((1, 1, 2 * P), dtype=torch.bool, device=dev)
        pad_x = torch.zeros((1, 1, P), dtype=w_dtype, device=dev)
        pad_m = torch.zeros((1, 1, P), dtype=torch.bool, device=dev)
        with torch.inference_mode(), tf32_precision_context("fp32"):
            loc_s, scale_s, kind = self._unpack_scaler_out(self.scaler(x_p, m_p))
            self._scaler_out_kind = kind
            try:
                loc_f, scale_f, _ = self._unpack_scaler_out(
                    self.scaler(torch.cat([x_p, pad_x], dim=-1), torch.cat([m_p, pad_m], dim=-1))
                )
                d_loc = float(
                    (loc_s.reshape(1, -1)[:, -1] - loc_f.reshape(1, -1)[:, -1])
                    .to(torch.float32)
                    .abs()
                    .max()
                    .item()
                )
                d_sc = float(
                    (scale_s.reshape(1, -1)[:, -1] - scale_f.reshape(1, -1)[:, -1])
                    .to(torch.float32)
                    .abs()
                    .max()
                    .item()
                )
                if d_loc > 1e-5 or d_sc > 1e-5:
                    self._scaler_needs_future_pad = True
            except Exception:
                self._scaler_needs_future_pad = False

        # Probe output_head 3D vs 4D input convention first (with q=None), then custom quantiles.
        d_model = int(self.r1_layer.q_proj.in_features)
        h_probe = torch.zeros((1, 1, d_model), dtype=w_dtype, device=dev)
        self._head_expects_4d = False
        self.register_buffer("_q_levels", None, persistent=False)
        with torch.inference_mode(), tf32_precision_context("fp32"):
            try:
                p_norm = self.out_norm(h_probe)
                _ = (
                    self.output_head(p_norm, q=None)
                    if self._head_takes_q
                    else self.output_head(p_norm)
                )
            except Exception:
                try:
                    p_norm4 = self.out_norm(h_probe.unsqueeze(1))
                    _ = (
                        self.output_head(p_norm4, q=None)
                        if self._head_takes_q
                        else self.output_head(p_norm4)
                    )
                    self._head_expects_4d = True
                except Exception:
                    pass

            if self._use_custom_q and quantiles is not None:
                q_probe = torch.tensor(list(quantiles), dtype=torch.float32, device=dev)
                try:
                    p_in = self.out_norm(h_probe.unsqueeze(1) if self._head_expects_4d else h_probe)
                    _ = self.output_head(p_in, q=q_probe)
                    self.register_buffer("_q_levels", q_probe, persistent=False)
                except Exception:
                    self._use_custom_q = False

    def _forward_impl(self, x_2d_f32: Any, *, use_missing_mask: bool) -> Any:
        B, L = x_2d_f32.shape
        P = self.patch_size
        if L < P:
            pad = torch.zeros((B, P - L), dtype=x_2d_f32.dtype, device=x_2d_f32.device)
            x_2d_f32 = torch.cat([pad, x_2d_f32], dim=1)
            L = P
        S = L // P
        usable_len = S * P
        if usable_len != L:
            x_2d_f32 = x_2d_f32[:, -usable_len:]
            L = usable_len

        obs_mask_2d = torch.isfinite(x_2d_f32)
        last_true = torch.ones((B, P), dtype=torch.bool, device=x_2d_f32.device)
        obs_mask_2d = (
            torch.cat([obs_mask_2d[:, :-P], last_true], dim=1) if L > P else last_true
        )
        x_clean_2d = torch.where(
            obs_mask_2d,
            torch.nan_to_num(x_2d_f32, nan=0.0, posinf=0.0, neginf=0.0),
            torch.zeros_like(x_2d_f32),
        )
        x_clean = x_clean_2d.unsqueeze(1)  # (B, 1, L)
        target_mask_3d = obs_mask_2d.unsqueeze(1)  # (B, 1, L) bool

        if self._scaler_needs_future_pad:
            pad_x = torch.zeros((B, 1, P), dtype=x_clean.dtype, device=x_clean.device)
            pad_m = torch.zeros((B, 1, P), dtype=torch.bool, device=x_clean.device)
            scaler_out = self.scaler(
                torch.cat([x_clean, pad_x], dim=-1),
                torch.cat([target_mask_3d, pad_m], dim=-1),
            )
            if self._scaler_out_kind == "triple":
                loc_all, scale_all = scaler_out[1], scaler_out[2]
            elif self._scaler_out_kind == "nested_pair":
                loc_all, scale_all = scaler_out[1]
            else:
                loc_all, scale_all = scaler_out[0], scaler_out[1]
            loc_anchor_src = loc_all.reshape(B, -1)[:, -1:]
            scale_anchor_src = scale_all.reshape(B, -1)[:, -1:]
            if loc_all.shape[-1] == S + 1 and S + 1 != L + P:
                loc = loc_all[..., :S]
                scale = scale_all[..., :S]
            else:
                loc = loc_all[..., :L]
                scale = scale_all[..., :L]
        else:
            scaler_out = self.scaler(x_clean, target_mask_3d)
            if self._scaler_out_kind == "triple":
                loc, scale = scaler_out[1], scaler_out[2]
            elif self._scaler_out_kind == "nested_pair":
                loc, scale = scaler_out[1]
            else:
                loc, scale = scaler_out[0], scaler_out[1]
            loc_anchor_src = loc.reshape(B, -1)[:, -1:]
            scale_anchor_src = scale.reshape(B, -1)[:, -1:]

        if loc.shape[-1] == S and S != L:
            loc_full = loc.repeat_interleave(P, dim=-1)
            scale_full = scale.repeat_interleave(P, dim=-1)
        else:
            loc_full = loc
            scale_full = scale

        raw_ctx = (x_clean - loc_full) / scale_full
        scaled_ctx = torch.where(target_mask_3d, raw_ctx, torch.zeros_like(raw_ctx)).asinh()

        if self._embed_takes_patch_size:
            context_x = self._embed_patches_fn(scaled_ctx, target_mask_3d, P)
        else:
            context_x = self._embed_patches_fn(scaled_ctx, target_mask_3d)
        x = context_x.squeeze(1) if context_x.ndim == 4 else context_x  # (B, S, D)

        if self._seq_ids_ndim == 1:
            time_ids = torch.arange(S, device=x.device, dtype=torch.long)
        else:
            time_ids = torch.arange(S, device=x.device, dtype=torch.long).unsqueeze(0).expand(B, S)

        if use_missing_mask:
            ctx_patch_obs = obs_mask_2d.view(B, S, P).any(dim=-1)  # (B, S)
            causal_tril = torch.tril(torch.ones((S, S), dtype=torch.bool, device=x.device))
            eye_s = torch.eye(S, dtype=torch.bool, device=x.device)
            valid = (causal_tril.unsqueeze(0) & ctx_patch_obs.unsqueeze(1)) | eye_s.unsqueeze(0)
            zero_v = torch.zeros((), dtype=x.dtype, device=x.device)
            ninf_v = torch.full((), float("-inf"), dtype=x.dtype, device=x.device)
            attn_mask: Any = torch.where(valid.unsqueeze(1), zero_v, ninf_v)  # (B, 1, S, S)
            attn_mask_last: Any = attn_mask[:, :, -1:, :]  # (B, 1, 1, S)
        else:
            attn_mask = None
            attn_mask_last = None

        for idx, layer in enumerate(self.prefix_layers):
            if self._prefix_is_variate[idx]:
                x = layer(x)
            else:
                takes_seq, takes_mask, takes_causal = self._prefix_sigs[idx]
                x = _invoke_layer_with_sig(
                    layer, x, time_ids, attn_mask, takes_seq, takes_mask, takes_causal
                )

        # Pass 1 (R1) on final time layer -> (B, 1, D).
        x_last = self.r1_layer(x, time_ids, attn_mask_last)

        # Pass 2 (R2) singleton variate collapse on trailing variate layer(s) -> (B, 1, D).
        for layer in self.suffix_layers:
            x_last = layer(x_last)

        pred_out = self.out_norm(x_last.unsqueeze(1) if self._head_expects_4d else x_last)
        q_arg = self._q_levels if self._use_custom_q else None
        raw_q = (
            self.output_head(pred_out, q=q_arg)
            if self._head_takes_q
            else self.output_head(pred_out)
        )
        if raw_q.ndim >= 4:
            block_q = (
                raw_q[..., :: self._head_nop, :]
                if raw_q.shape[-2] >= self._head_nop
                else raw_q[..., :1, :]
            )
        else:
            block_q = raw_q
        q_slice = block_q.reshape(block_q.shape[0], B, -1)[:, :, : self.horizon]

        loc_anchor = loc_anchor_src.reshape(1, B, 1).to(q_slice.dtype)
        scale_anchor = scale_anchor_src.reshape(1, B, 1).to(q_slice.dtype)
        q_qbh = q_slice.sinh() * scale_anchor + loc_anchor
        q_qbh = Toto2CompiledEngine._clamp_nonfinite(q_qbh)
        q_qbh = q_qbh.sort(dim=0).values
        return q_qbh.permute(1, 0, 2).contiguous()

    def forward(self, x_2d_f32: Any) -> Any:
        return self._forward_impl(x_2d_f32, use_missing_mask=False)

    def forward_masked(self, x_2d_f32: Any) -> Any:
        return self._forward_impl(x_2d_f32, use_missing_mask=True)


class Toto2CompiledEngine:
    """Compiled Toto-2.0 engine with Pass 1 (R1), Pass 2 (Variate Collapse), and CUDA Graphs."""

    DEFAULT_QUANTILES: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    _SKIP_PRECISION_NAMES: tuple[str, ...] = (
        "patch_embed",
        "embed",
        "norm",
        "scaler",
        "output_distribution",
        "output_head",
        "quantile_head",
        "head",
        "unembed",
        "orig_attn",
        "qk_proj",
        "_pds",
    )

    def __init__(
        self,
        model: Any,
        *,
        horizon: int = 16,
        context_length: int = 512,
        precision: str = "auto",
        batch_buckets: Sequence[int] | None = None,
        dc_strip: bool = True,
        quantiles: Sequence[float] | None = None,
        use_cuda_graphs: bool = True,
        copy_model: bool = True,
        strict_a10: bool = False,
    ) -> None:
        if torch is None:
            raise RuntimeError("Toto2CompiledEngine requires PyTorch.")

        prec_key = precision.strip().lower()
        canonical_prec = normalize_precision_alias(prec_key)
        if prec_key == "a10-auto" or (strict_a10 and canonical_prec == "auto"):
            precision = "fp16x3"
        else:
            precision = canonical_prec

        self.horizon = int(horizon)
        self.context_length = int(context_length)
        self.precision = precision
        self.strict_a10 = bool(strict_a10 or precision == "fp16x3")
        self.quantiles = tuple(self.DEFAULT_QUANTILES if quantiles is None else quantiles)
        self.dc_stripper = CausalFloat64DCStripper(enabled=dc_strip)

        raw_backbone = unwrap_toto2_backbone(model)
        self.backbone = copy.deepcopy(raw_backbone) if copy_model else raw_backbone
        self.backbone.eval()

        self.arch = inspect_toto2_architecture(self.backbone)
        self.layers = _get_transformer_layers(self.backbone)
        self._forces_last_patch_observed = callable(
            getattr(self.backbone, "_prepare_forecast_inputs", None)
        )

        # Number of anchor output tokens needed for requested horizon.
        self.nq = max(1, math.ceil(self.horizon / self.arch.patch_size))
        self.n_dummy_pred_patches = max(0, self.nq - 1)
        self.dropped_dummy_pred_patch = self.horizon <= self.arch.patch_size

        first_param = next(self.backbone.parameters(), None)
        self.device = first_param.device if first_param is not None else torch.device("cpu")

        self._execution_path = "manual_r1_variate"
        self._direct_core: _CompiledToto2Core | None = None
        self._direct_core_fallback_reason: str | None = None
        self._native_variate_collapsed_layers: list[int] = []
        self.precision_summary: dict[str, Any] = {
            "mode": precision,
            "replaced": 0,
            "escalated_w": 0,
        }

        # Tier 1: Direct R1 + R2 _CompiledToto2Core for real Toto-2.0 (horizon <= patch_size).
        used_direct = False
        if self.horizon <= self.arch.patch_size and _is_direct_toto2_architecture(
            self.backbone, self.layers, self.arch
        ):
            try:
                core = _CompiledToto2Core(
                    self.backbone,
                    self.layers,
                    self.arch,
                    horizon=self.horizon,
                    quantiles=self.quantiles,
                )
                core.eval()
                if (not _has_native_toto2_forecast(raw_backbone)) or self._verify_direct_core(
                    raw_backbone, core
                ):
                    self._execution_path = "direct_r1_variate"
                    self._direct_core = core
                    used_direct = True
                    if precision in ("auto", "scaled-fp16", "fp16x3"):
                        self.precision_summary = replace_linears_with_crest_guarded(
                            self._direct_core,
                            mode=precision,
                            graph_safe=True,
                            skip_names=self._SKIP_PRECISION_NAMES,
                        )
                elif self._direct_core_fallback_reason is None:
                    self._direct_core_fallback_reason = "compile_probe_parity_mismatch"
            except Exception as exc:
                self._direct_core_fallback_reason = f"{type(exc).__name__}: {exc}"

        if not used_direct:
            # Tier 2 vs Tier 3: manual R1+variate for generic backbones, or native variate collapse.
            use_manual = True
            if _has_native_toto2_forecast(raw_backbone):
                use_manual = self._probe_manual_matches_native(raw_backbone)

            if use_manual:
                _prepare_compiled_qkv_projections(self.layers, self.arch)
                if precision in ("auto", "scaled-fp16", "fp16x3"):
                    self.precision_summary = replace_linears_with_crest_guarded(
                        self.backbone,
                        mode=precision,
                        graph_safe=True,
                        skip_names=self._SKIP_PRECISION_NAMES,
                    )
            else:
                self._execution_path = "native_variate_collapse"
                self.backbone = copy.deepcopy(raw_backbone) if copy_model else raw_backbone
                self.backbone.eval()
                self.layers = _get_transformer_layers(self.backbone)
                probe_x = self._build_compile_probe()
                probe_io_map = _capture_variate_attn_probe_io(
                    self.backbone,
                    self.layers,
                    self.arch,
                    probe_x,
                    self.horizon,
                    len(self.quantiles),
                )
                installed = _install_native_singleton_variate_collapse(
                    self.backbone,
                    self.layers,
                    self.arch,
                    probe_io_map=probe_io_map,
                )
                if installed and not self._verify_native_variate_collapse(raw_backbone):
                    # Revert wrappers cleanly if a custom attention signature diverges.
                    for layer, attr_name, wrapper, _idx in installed:
                        setattr(layer, attr_name, wrapper.orig_attn)
                    installed = []
                self._native_variate_collapsed_layers = [idx for _, _, _, idx in installed]

                if precision in ("auto", "scaled-fp16", "fp16x3"):
                    self.precision_summary = replace_linears_with_crest_guarded(
                        self.backbone,
                        mode=precision,
                        graph_safe=True,
                        skip_names=self._SKIP_PRECISION_NAMES,
                    )

        self.dropped_dummy_pred_patch = (
            self._execution_path in ("direct_r1_variate", "manual_r1_variate")
            and self.horizon <= self.arch.patch_size
        )

        has_crest_guarded = int(self.precision_summary.get("replaced", 0)) > 0
        use_inductor = (
            self.device.type == "cuda"
            and use_cuda_graphs
            and (not has_crest_guarded)
            and self.precision in ("auto", "tf32", "fp32")
        )
        # Cap Inductor to B <= 1024 on Toto-2.0 so B=4096 (B*S=65536) routes directly to
        # BucketedCudaGraphRunner without triggering Inductor AssertionError or disabling
        # Inductor for B <= 1024 (inductor_fallback_count stays 0).
        inductor_max_batch = 1024 if use_inductor else None
        self.graph_runner = InductorOrGraphRunner(
            self._forward_e2e_clean,
            sample_shape=(self.context_length,),
            buckets=batch_buckets,
            device=self.device,
            dtype=torch.float32,
            enabled=(use_cuda_graphs and self.device.type == "cuda"),
            use_inductor=use_inductor,
            precision=self.precision,
            inductor_max_batch=inductor_max_batch,
        )
        self._masked_graph_runner: InductorOrGraphRunner | None = (
            InductorOrGraphRunner(
                self._forward_e2e_masked,
                sample_shape=(self.context_length,),
                buckets=batch_buckets,
                device=self.device,
                dtype=torch.float32,
                enabled=(use_cuda_graphs and self.device.type == "cuda"),
                use_inductor=use_inductor,
                precision=self.precision,
                inductor_max_batch=inductor_max_batch,
            )
            if self._execution_path == "direct_r1_variate"
            else None
        )

    def _build_direct_compile_probe(self) -> tuple[Any, Any]:
        p = self.arch.patch_size
        probe_len = max(p * 3, min(self.context_length, p * 4))
        t = torch.linspace(0.0, 8.0, probe_len, dtype=torch.float64, device=self.device)
        base = 50.0 + torch.sin(t) + 0.3 * torch.cos(2.1 * t)
        nan_row = base.clone()
        nan_row[::4] = float("nan")
        nan_row[-1] = base[-1]
        outage_row = base.clone()
        outage_row[:p] = float("nan")
        return torch.stack([base, nan_row], dim=0), outage_row.unsqueeze(0)

    def _verify_direct_core(self, raw_backbone: Any, core: _CompiledToto2Core) -> bool:
        """Verify direct R1+R2 core matches unmodified native ``.forecast`` on clean+NaN+outage.

        Checks TWO distinct contracts:
        1. Exact structural parity (``<= 2e-3``) comparing ``core(x_s)`` vs
           ``raw_backbone.forecast(x_s)`` on the exact same DC-stripped inputs
           (isolating R1 + R2 + dead-``pred_x``-drop + missing-patch masking from
           FP32 shift-cancellation noise).
        2. End-to-end DC-stripped shift-equivariance parity (``<= 0.02``) against
           unstripped ``raw_backbone.forecast(x)`` on both clean+25%-NaN and outage.
        """
        try:
            clean_and_nan, outage = self._build_direct_compile_probe()
            P = self.arch.patch_size
            with torch.inference_mode(), tf32_precision_context("fp32"):
                x_s, dc_cn = _strip_toto2_inputs_f64(
                    self.dc_stripper, clean_and_nan, P, self._forces_last_patch_observed
                )
                ref_s_cn = _call_toto2_native_forecast(
                    raw_backbone,
                    x_s,
                    self.horizon,
                    num_quantiles_hint=len(self.quantiles),
                    has_missing_override=True,
                ).to(torch.float64)
                got_s_cn = core(x_s).to(torch.float64)
                if ref_s_cn.shape != got_s_cn.shape:
                    self._direct_core_fallback_reason = (
                        f"clean_shape_mismatch:{tuple(ref_s_cn.shape)}!={tuple(got_s_cn.shape)}"
                    )
                    return False
                if not (
                    torch.isfinite(ref_s_cn).all().item() and torch.isfinite(got_s_cn).all().item()
                ):
                    self._direct_core_fallback_reason = "clean_nonfinite"
                    return False
                denom_s_cn = max(float(ref_s_cn.std().item()), 1e-4)
                err_s_cn = float((ref_s_cn - got_s_cn).abs().max().item()) / denom_s_cn
                if err_s_cn > 2e-3:
                    self._direct_core_fallback_reason = f"clean_struct_err={err_s_cn:.4e}"
                    return False

                ref_cn = _call_toto2_native_forecast(
                    raw_backbone,
                    clean_and_nan.to(torch.float32),
                    self.horizon,
                    num_quantiles_hint=len(self.quantiles),
                    has_missing_override=True,
                ).to(torch.float64)
                got_cn = self.dc_stripper.restore(got_s_cn, dc_cn).sort(dim=1).values
                denom_cn = max(float(ref_cn.std().item()), 1e-4)
                err_cn = float((ref_cn - got_cn).abs().max().item()) / denom_cn
                if err_cn > 0.02:
                    self._direct_core_fallback_reason = f"clean_dc_equiv_err={err_cn:.4e}"
                    return False

                x_s_o, dc_o = _strip_toto2_inputs_f64(
                    self.dc_stripper, outage, P, self._forces_last_patch_observed
                )
                ref_s_out = _call_toto2_native_forecast(
                    raw_backbone,
                    x_s_o,
                    self.horizon,
                    num_quantiles_hint=len(self.quantiles),
                    has_missing_override=True,
                ).to(torch.float64)
                got_s_out = core.forward_masked(x_s_o).to(torch.float64)
                if ref_s_out.shape != got_s_out.shape:
                    self._direct_core_fallback_reason = (
                        f"outage_shape_mismatch:{tuple(ref_s_out.shape)}!={tuple(got_s_out.shape)}"
                    )
                    return False
                if not (
                    torch.isfinite(ref_s_out).all().item()
                    and torch.isfinite(got_s_out).all().item()
                ):
                    self._direct_core_fallback_reason = "outage_nonfinite"
                    return False
                denom_s_out = max(float(ref_s_out.std().item()), 1e-4)
                err_s_out = float((ref_s_out - got_s_out).abs().max().item()) / denom_s_out
                if err_s_out > 2e-3:
                    self._direct_core_fallback_reason = f"outage_struct_err={err_s_out:.4e}"
                    return False

                ref_out = _call_toto2_native_forecast(
                    raw_backbone,
                    outage.to(torch.float32),
                    self.horizon,
                    num_quantiles_hint=len(self.quantiles),
                    has_missing_override=True,
                ).to(torch.float64)
                got_out = self.dc_stripper.restore(got_s_out, dc_o).sort(dim=1).values
                denom_out = max(float(ref_out.std().item()), 1e-4)
                err_out = float((ref_out - got_out).abs().max().item()) / denom_out
                if err_out > 0.02:
                    self._direct_core_fallback_reason = f"outage_dc_equiv_err={err_out:.4e}"
                    return False
                return True
        except Exception as exc:
            self._direct_core_fallback_reason = f"{type(exc).__name__}: {exc}"
            return False

    def _build_compile_probe(self) -> Any:
        p = self.arch.patch_size
        probe_len = max(p * 2, min(self.context_length, p * 4))
        t = torch.linspace(0.0, 8.0, probe_len, dtype=torch.float32, device=self.device)
        clean = torch.sin(t) + 0.3 * torch.cos(2.1 * t)
        nan_row = clean.clone()
        nan_row[::4] = float("nan")
        nan_row[-1] = clean[-1]
        return torch.stack([clean, nan_row], dim=0)

    def _probe_manual_matches_native(self, raw_backbone: Any) -> bool:
        """Verify at compile time whether manual R1+variate forward matches native .forecast."""
        try:
            probe_x = self._build_compile_probe()
            with torch.inference_mode(), tf32_precision_context("fp32"):
                ref = _call_toto2_native_forecast(
                    raw_backbone,
                    probe_x,
                    self.horizon,
                    num_quantiles_hint=len(self.quantiles),
                    has_missing_override=True,
                )
                got = self._forward_manual_f32(probe_x)
            if ref.shape != got.shape:
                return False
            if not (torch.isfinite(ref).all().item() and torch.isfinite(got).all().item()):
                return False
            denom = max(float(ref.std().item()), 1e-4)
            err = float((ref - got).abs().max().item()) / denom
            return err <= 0.02
        except Exception:
            return False

    def _verify_native_variate_collapse(self, raw_backbone: Any) -> bool:
        """Verify in-place singleton variate collapse matches unmodified native .forecast."""
        try:
            probe_x = self._build_compile_probe()
            with torch.inference_mode(), tf32_precision_context("fp32"):
                ref = _call_toto2_native_forecast(
                    raw_backbone,
                    probe_x,
                    self.horizon,
                    num_quantiles_hint=len(self.quantiles),
                    has_missing_override=True,
                )
                got = _call_toto2_native_forecast(
                    self.backbone,
                    probe_x,
                    self.horizon,
                    num_quantiles_hint=len(self.quantiles),
                    has_missing_override=True,
                )
            if ref.shape != got.shape:
                return False
            if not (torch.isfinite(ref).all().item() and torch.isfinite(got).all().item()):
                return False
            denom = max(float(ref.std().item()), 1e-4)
            err = float((ref - got).abs().max().item()) / denom
            return err <= 1e-3
        except Exception:
            return False

    @staticmethod
    def _clamp_nonfinite(q: Any) -> Any:
        """Branchless CUDA-Graph-safe replacement of non-finite entries (zero host sync)."""
        is_fin = torch.isfinite(q)
        q_zero = torch.where(is_fin, q, torch.zeros_like(q))
        cnt = is_fin.to(q.dtype).sum(dim=0, keepdim=True).clamp(min=1.0)
        fallback = q_zero.sum(dim=0, keepdim=True) / cnt
        return torch.where(is_fin, q, fallback)

    def _embed_context(self, scaled_3d: Any, obs_mask_3d: Any) -> Any:
        """Patch-embed 3D ``(B, 1, L)`` scaled context, dropping dummy pred_x for H <= P."""
        B, _, L = scaled_3d.shape
        P = self.arch.patch_size
        S_ctx = L // P
        usable_len = S_ctx * P
        if usable_len != L:
            scaled_3d = scaled_3d[:, :, -usable_len:]
            obs_mask_3d = obs_mask_3d[:, :, -usable_len:]

        patch_embed = getattr(self.backbone, "patch_embed", None)
        if patch_embed is None:
            raise ValueError("Toto-2.0 backbone missing patch_embed")

        sig = inspect.signature(patch_embed.forward)
        n_params = len([p for p in sig.parameters.values() if p.name != "self"])
        if n_params >= 2:
            tokens = patch_embed(scaled_3d, obs_mask_3d)
        else:
            tokens = patch_embed(scaled_3d)

        # Normalize token tensor shape to (B, S_ctx, D) for univariate n_var == 1.
        if tokens.ndim == 4:
            # (B, 1, S_ctx, D) -> (B, S_ctx, D)
            tokens = tokens.squeeze(1)

        if self.n_dummy_pred_patches > 0:
            dummy = torch.zeros(
                (B, self.n_dummy_pred_patches, tokens.shape[-1]),
                dtype=tokens.dtype,
                device=tokens.device,
            )
            tokens = torch.cat([tokens, dummy], dim=1)
        return tokens

    def _forward_manual_f32(self, x_2d_f32: Any) -> Any:
        """Contracted R1 + Variate Collapse forward on ``(B, L)`` FP32 -> ``(B, Q, H)``."""
        B, L = x_2d_f32.shape
        P = self.arch.patch_size
        if L < P:
            pad = torch.full(
                (B, P - L), float("nan"), dtype=x_2d_f32.dtype, device=x_2d_f32.device
            )
            x_2d_f32 = torch.cat([pad, x_2d_f32], dim=1)
            L = P
        S_ctx = L // P
        usable_len = S_ctx * P
        if usable_len != L:
            x_2d_f32 = x_2d_f32[:, -usable_len:]
            L = usable_len

        obs_mask_2d = torch.isfinite(x_2d_f32)
        x_3d = x_2d_f32.unsqueeze(1)
        mask_3d = obs_mask_2d.unsqueeze(1)

        # Exact 3D scaler evaluation with masked horizon carry-forward matching Toto2Model.
        scaler = getattr(self.backbone, "scaler", None)
        scaled_3d, static_loc, static_scale = _scale_context_with_horizon_carry(
            scaler, x_3d, mask_3d, P
        )

        # Exact patch-level observation mask + all-NaN softmax guard.
        ctx_patch_obs = obs_mask_2d.view(B, S_ctx, P).sum(dim=-1) > 0
        total_patches = S_ctx + self.n_dummy_pred_patches
        valid_mask = _build_toto_time_mask(ctx_patch_obs, total_patches)

        h = self._embed_context(scaled_3d, mask_3d)
        var_set = set(self.arch.variate_layer_indices)
        final_time_idx = self.arch.final_time_layer_idx

        for idx, layer in enumerate(self.layers):
            if idx in var_set:
                # Pass 2: Singleton Variate Collapse (n_var == 1).
                h = _run_singleton_variate_layer(layer, h, self.arch.num_heads)
            elif idx == final_time_idx:
                # Pass 1 (R1): Final time layer computes Q ONLY for last nq anchor tokens.
                h = _run_time_layer(
                    layer,
                    h,
                    valid_mask,
                    self.arch.num_heads,
                    anchor_nq=self.nq,
                )
            else:
                h = _run_time_layer(
                    layer,
                    h,
                    valid_mask,
                    self.arch.num_heads,
                    anchor_nq=None,
                )

        final_norm = getattr(self.backbone, "final_norm", getattr(self.backbone, "norm", None))
        if final_norm is not None:
            h = final_norm(h)

        # Output shape (Q, B, H) -> clamp non-finite -> monotonic sort along dim=0 -> (B, Q, H).
        q_qbh = _decode_toto_quantiles(
            self.backbone,
            h[:, -self.nq :, :],
            static_loc,
            static_scale,
            self.horizon,
            P,
            self.quantiles,
        )
        q_qbh = self._clamp_nonfinite(q_qbh)
        q_qbh = q_qbh.sort(dim=0).values
        return q_qbh.permute(1, 0, 2).contiguous()

    def _forward_stripped_f32(self, x_2d_f32: Any) -> Any:
        """Core graph-capturable forward on DC-stripped ``(B, L)`` FP32 tensor -> ``(B, Q, H)``."""
        if self._execution_path == "direct_r1_variate" and self._direct_core is not None:
            return self._direct_core(x_2d_f32)
        if self._execution_path == "native_variate_collapse":
            return _call_toto2_native_forecast(
                self.backbone,
                x_2d_f32,
                self.horizon,
                num_quantiles_hint=len(self.quantiles),
                has_missing_override=True,
            )
        return self._forward_manual_f32(x_2d_f32)

    def _forward_e2e_clean(self, x_2d: Any) -> Any:
        x_s, dc_f64 = _strip_toto2_inputs_f64(
            self.dc_stripper, x_2d, self.arch.patch_size, self._forces_last_patch_observed
        )
        q_s = self._forward_stripped_f32(x_s)
        # q_s is already clamped-finite & sorted along dim=1; adding finite broadcast dc_f64
        # in FP64 is strictly order-preserving (eliminates a redundant FP64 sort kernel).
        return self.dc_stripper.restore(q_s, dc_f64).contiguous()

    def _forward_e2e_masked(self, x_2d: Any) -> Any:
        x_s, dc_f64 = _strip_toto2_inputs_f64(
            self.dc_stripper, x_2d, self.arch.patch_size, self._forces_last_patch_observed
        )
        assert self._direct_core is not None
        q_s = self._direct_core.forward_masked(x_s)
        return self.dc_stripper.restore(q_s, dc_f64).contiguous()

    def predict_quantiles(
        self,
        x: Any,
        *,
        has_missing_values: bool | None = None,
    ) -> Any:
        """Predict monotonic quantiles ``(B, Q, H)`` in FP64 with causal DC stripping."""
        cpu_has_all_missing_patch: bool | None = None
        if (
            isinstance(x, torch.Tensor)
            and x.device == self.device
            and x.dtype in (torch.float32, torch.float64)
        ):
            x_t = x.unsqueeze(0) if x.ndim == 1 else x
        else:
            # Check patch observation validity on CPU BEFORE H2D transfer to avoid GPU D2H sync.
            if (
                has_missing_values is None
                and self._execution_path == "direct_r1_variate"
                and self._masked_graph_runner is not None
            ):
                if isinstance(x, torch.Tensor) and x.device.type == "cpu":
                    x_cpu = x.unsqueeze(0) if x.ndim == 1 else x
                    if x_cpu.ndim == 2:
                        B_c, L_c = int(x_cpu.shape[0]), int(x_cpu.shape[-1])
                        P_c = self.arch.patch_size
                        S_c = L_c // P_c
                        if S_c > 1 and B_c > 0:
                            prev_c = x_cpu[:, L_c - S_c * P_c : L_c - P_c]
                            cpu_has_all_missing_patch = not bool(
                                torch.isfinite(prev_c)
                                .reshape(B_c, S_c - 1, P_c)
                                .any(dim=-1)
                                .all()
                            )
                        else:
                            cpu_has_all_missing_patch = False
                elif not isinstance(x, torch.Tensor):
                    x_np = np.asarray(x, dtype=np.float64)
                    if x_np.ndim == 1:
                        x_np = x_np[None, :]
                    if x_np.ndim == 2:
                        B_c, L_c = int(x_np.shape[0]), int(x_np.shape[-1])
                        P_c = self.arch.patch_size
                        S_c = L_c // P_c
                        if S_c > 1 and B_c > 0:
                            prev_c = x_np[:, L_c - S_c * P_c : L_c - P_c]
                            cpu_has_all_missing_patch = not bool(
                                np.isfinite(prev_c)
                                .reshape(B_c, S_c - 1, P_c)
                                .any(axis=-1)
                                .all()
                            )
                        else:
                            cpu_has_all_missing_patch = False
            x_t = torch.as_tensor(x, device=self.device, dtype=torch.float64)
            if x_t.ndim == 1:
                x_t = x_t.unsqueeze(0)

        if self._execution_path == "direct_r1_variate" and self._masked_graph_runner is not None:
            if has_missing_values is True:
                return self._masked_graph_runner(x_t)
            if has_missing_values is False:
                return self.graph_runner(x_t)

            B = int(x_t.shape[0])
            L = int(x_t.shape[-1])
            P = self.arch.patch_size
            S = L // P
            if S > 1 and B > 0:
                if cpu_has_all_missing_patch is not None:
                    if cpu_has_all_missing_patch:
                        return self._masked_graph_runner(x_t)
                else:
                    usable = S * P
                    prev_part = x_t[:, L - usable : L - P]
                    ctx_prev_obs = torch.isfinite(prev_part).reshape(B, S - 1, P).any(dim=-1)
                    if not bool(ctx_prev_obs.all().item()):
                        return self._masked_graph_runner(x_t)

        return self.graph_runner(x_t)

    def warmup(self, buckets: Sequence[int] | None = None) -> list[int]:
        """Warm up clean (fast is_causal=True) CUDA/Inductor runner across requested buckets."""
        return self.graph_runner.warmup(buckets=buckets)

    def predict(self, x: Any, *, has_missing_values: bool | None = None) -> Any:
        """Predict point forecast ``(B, H)`` (median quantile)."""
        q = self.predict_quantiles(x, has_missing_values=has_missing_values)
        mid = q.shape[1] // 2
        return q[:, mid, :]

    def report(self) -> dict[str, Any]:
        rep: dict[str, Any] = {
            "pass": "toto2",
            "execution_path": self._execution_path,
            "num_layers": self.arch.num_layers,
            "layer_group_size": self.arch.layer_group_size,
            "num_variate_layers_per_group": self.arch.num_variate_layers_per_group,
            "patch_size": self.arch.patch_size,
            "final_time_layer_r1_idx": self.arch.final_time_layer_idx,
            "collapsed_variate_layers": (
                list(self.arch.variate_layer_indices)
                if self._execution_path in ("direct_r1_variate", "manual_r1_variate")
                else list(self._native_variate_collapsed_layers)
            ),
            "anchor_nq": self.nq,
            "dropped_dummy_pred_patch": self.dropped_dummy_pred_patch,
            "forces_last_patch_observed": self._forces_last_patch_observed,
            "dc_strip": self.dc_stripper.enabled,
            "precision": self.precision_summary,
            "cuda_graphs": self.graph_runner.stats(),
        }
        if self._masked_graph_runner is not None:
            rep["masked_cuda_graphs"] = self._masked_graph_runner.stats()
        if self._direct_core_fallback_reason is not None:
            rep["direct_core_fallback_reason"] = self._direct_core_fallback_reason
        return rep


Toto2Compiler = Toto2CompiledEngine


def compile_toto2(
    model: Any,
    *,
    horizon: int = 16,
    context_length: int = 512,
    precision: str = "auto",
    batch_buckets: Sequence[int] | None = None,
    dc_strip: bool = True,
    quantiles: Sequence[float] | None = None,
    use_cuda_graphs: bool = True,
    strict_a10: bool = False,
) -> Toto2CompiledEngine:
    """Compile a Toto-2.0 model into a ``Toto2CompiledEngine``."""
    return Toto2CompiledEngine(
        model,
        horizon=horizon,
        context_length=context_length,
        precision=precision,
        batch_buckets=batch_buckets,
        dc_strip=dc_strip,
        quantiles=quantiles,
        use_cuda_graphs=use_cuda_graphs,
        strict_a10=strict_a10,
    )


def _run_uncontracted_eager_toto2(
    orig_model: Any,
    x_one: Any,
    horizon: int,
    quantiles: Sequence[float],
    dc_strip: bool,
) -> Any:
    """Run unmodified Toto-2.0 eager reference (uses ``model.forecast`` when available)."""
    if hasattr(orig_model, "predict_quantiles"):
        return orig_model.predict_quantiles(x_one)

    with tf32_precision_context("fp32"):
        stripper = CausalFloat64DCStripper(enabled=dc_strip)

        forecast_owner = None
        if _has_native_toto2_forecast(orig_model):
            forecast_owner = orig_model
        else:
            cand_bb = unwrap_toto2_backbone(orig_model)
            if _has_native_toto2_forecast(cand_bb):
                forecast_owner = cand_bb

        if forecast_owner is not None:
            forces_last = callable(getattr(forecast_owner, "_prepare_forecast_inputs", None))
            if forces_last and dc_strip:
                arch_ref = inspect_toto2_architecture(forecast_owner)
                x_s, dc_f64 = _strip_toto2_inputs_f64(
                    stripper, x_one, arch_ref.patch_size, forces_last_patch_observed=True
                )
            else:
                x_s, dc_f64 = stripper.strip(x_one)
            q_bqh = _call_toto2_native_forecast(
                forecast_owner,
                x_s,
                horizon,
                num_quantiles_hint=len(quantiles),
                has_missing_override=None,
            )
            return stripper.restore(q_bqh, dc_f64).sort(dim=1).values

        x_s, dc_f64 = stripper.strip(x_one)

        backbone = unwrap_toto2_backbone(orig_model)
        arch = inspect_toto2_architecture(backbone)
        layers = _get_transformer_layers(backbone)
        B, L = x_s.shape
        P = arch.patch_size
        if L < P:
            pad = torch.full((B, P - L), float("nan"), dtype=x_s.dtype, device=x_s.device)
            x_s = torch.cat([pad, x_s], dim=1)
            L = P
        S_ctx = L // P
        usable_len = S_ctx * P
        if usable_len != L:
            x_s = x_s[:, -usable_len:]

        obs_mask_2d = torch.isfinite(x_s)
        x_3d = x_s.unsqueeze(1)
        mask_3d = obs_mask_2d.unsqueeze(1)
        scaled_3d, static_loc, static_scale = _scale_context_with_horizon_carry(
            getattr(backbone, "scaler", None), x_3d, mask_3d, P
        )

        nq = max(1, math.ceil(horizon / P))
        n_dummy = max(0, nq - 1)
        ctx_patch_obs = obs_mask_2d.view(B, S_ctx, P).sum(dim=-1) > 0
        valid_mask = _build_toto_time_mask(ctx_patch_obs, S_ctx + n_dummy)

        patch_embed = backbone.patch_embed
        sig = inspect.signature(patch_embed.forward)
        n_params = len([p for p in sig.parameters.values() if p.name != "self"])
        h = patch_embed(scaled_3d, mask_3d) if n_params >= 2 else patch_embed(scaled_3d)
        if h.ndim == 4:
            h = h.squeeze(1)
        if n_dummy > 0:
            h = torch.cat(
                [h, torch.zeros((B, n_dummy, h.shape[-1]), dtype=h.dtype, device=h.device)], dim=1
            )

        # Full uncontracted eager forward across all time and variate layers (explicit SDPA on V=1).
        var_set = set(arch.variate_layer_indices)
        for idx, layer in enumerate(layers):
            if idx in var_set:
                norm1 = _get_norm1(layer)
                attn = _get_attn_module(layer)
                norm2 = _get_norm2(layer)
                mlp = _get_mlp_module(layer)
                res1, branch_in = _unit_residual_split(layer, h)
                hn = norm1(branch_in) if norm1 is not None else branch_in
                n_heads, n_kv_heads, head_dim, d_q, d_kv = _infer_qkv_dims(
                    attn, h.shape[-1], arch.num_heads
                )
                q, k, v = _project_qkv(attn, hn, hn, d_q=d_q, d_kv=d_kv)
                BS = B * h.shape[1]
                qv = q.reshape(BS, 1, n_heads, head_dim).transpose(1, 2)
                kv = k.reshape(BS, 1, n_kv_heads, head_dim).transpose(1, 2)
                vv = v.reshape(BS, 1, n_kv_heads, head_dim).transpose(1, 2)
                if n_kv_heads != n_heads:
                    rep = n_heads // n_kv_heads
                    kv = kv.repeat_interleave(rep, dim=1)
                    vv = vv.repeat_interleave(rep, dim=1)
                ov = F.scaled_dot_product_attention(qv, kv, vv, dropout_p=0.0, is_causal=False)
                ov = ov.transpose(1, 2).reshape(B, h.shape[1], d_q)
                h = _unit_residual_add(layer, res1, _project_out(attn, ov))
                if mlp is not None:
                    res2, mlp_in = _unit_residual_split(layer, h)
                    h = _unit_residual_add(
                        layer, res2, mlp(norm2(mlp_in) if norm2 is not None else mlp_in)
                    )
            else:
                h = _run_time_layer(layer, h, valid_mask, arch.num_heads, anchor_nq=None)

        final_norm = getattr(backbone, "final_norm", getattr(backbone, "norm", None))
        if final_norm is not None:
            h = final_norm(h)
        q_qbh = _decode_toto_quantiles(
            backbone, h[:, -nq:, :], static_loc, static_scale, horizon, P, quantiles
        )
        q_qbh = Toto2CompiledEngine._clamp_nonfinite(q_qbh).sort(dim=0).values
        q_bqh = q_qbh.permute(1, 0, 2).contiguous()
        return stripper.restore(q_bqh, dc_f64).sort(dim=1).values


def verify_parity(
    orig_model: Any,
    compiled_model: Toto2CompiledEngine,
    x_test: Any,
    *,
    tol: float = 0.05,
    nan_fraction: float = 0.25,
    seed: int = 2026,
) -> dict[str, ParityAuditReport]:
    """Verify strict-FP32 parity on both clean and 25% NaN-bearing series."""
    if torch is None:
        raise RuntimeError("verify_parity requires PyTorch.")

    def _ref_fn(x_one: Any) -> Any:
        return _run_uncontracted_eager_toto2(
            orig_model,
            x_one,
            compiled_model.horizon,
            compiled_model.quantiles,
            compiled_model.dc_stripper.enabled,
        )

    validator = StrictParityValidator(tol=tol)
    x_clean = torch.as_tensor(x_test, device=compiled_model.device, dtype=torch.float64)
    clean_report = validator.audit(_ref_fn, compiled_model.predict_quantiles, x_clean)

    # Inject missing values (default 25% NaN-bearing slice) to catch missing-value bugs.
    rng = np.random.default_rng(seed)
    mask_np = rng.random(size=tuple(x_clean.shape)) < nan_fraction
    # Keep at least the final observation finite so DC anchor is well-defined.
    mask_np[..., -1] = False
    x_nan = x_clean.clone()
    x_nan[torch.from_numpy(mask_np).to(device=x_clean.device)] = float("nan")
    nan_report = validator.audit(_ref_fn, compiled_model.predict_quantiles, x_nan)

    return {"clean": clean_report, "nan_25pct": nan_report}
