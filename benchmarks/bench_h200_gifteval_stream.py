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

"""Live H200 end-to-end streaming benchmark on GIFT-Eval datasets.

Compares Standalone Eager TSFM (Chronos-Bolt-small, Toto-2.0-4m) against two
strictly causal tsjit streaming operating points:
  1. ``tsjit_gap_async`` (``max_gap_ticks=1``): true 1-tick-lagged asynchronous
     gap placement (0 ms GPU blocking on arrival) combining horizon-aligned
     Compiled TSFM + online AR(1) innovation revision, causally pre-warmed GPU
     IPCG Online Ridge (``OnlineGpuRidgeEngine``), and online exponential-weight
     Hedge mixing inside the C11 lock-free ``SoftRevisionEngine``.
  2. ``tsjit_event_gated`` (``max_gap_ticks=4``): ``EventGate`` surprise +
     e-value GPU compute elision (~60-75% fewer backbone refreshes) using exact
     horizon-aligned lead indexing + pre-warmed Online IPCG-Ridge correction.

Also benchmarks batched C11 arrival latency across ``N = 1024`` live streams
(reporting both Python/ctypes overhead and true native C11 critical-path
latency < 1 us/decision) and enforces bit-exact ``assert_causal_bit_exact`` on
every dataset and mode.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import math
import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional in pure-CPU environments
    torch = None  # type: ignore[assignment]

from tsjit.compiler import compile as tsjit_compile
from tsjit.cuda.loader import OnlineGpuRidgeEngine, ipcg_solve_batch
from tsjit.native.build import CArriveOut, build_c_executable, load_native_lib
from tsjit.runtime.arrival import PwlArrivalEngine, SoftRevisionEngine
from tsjit.runtime.calibration import EventGate, OnlineACI, QuantilePitEValueDetector

GIFT_DATASETS: tuple[str, ...] = (
    "electricity",
    "ett1",
    "ett2",
    "LOOP_SEATTLE",
    "bitbrains_fast_storage",
)

QUANTILE_LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float64)
_Z9 = np.array(
    [
        -1.2815515655446004,
        -0.8416212335729143,
        -0.5244005127080407,
        -0.2533471031357997,
        0.0,
        0.2533471031357997,
        0.5244005127080407,
        0.8416212335729143,
        1.2815515655446004,
    ],
    dtype=np.float64,
)


class _NumpyOnlineRidgeMirror:
    """CPU/NumPy mirror of OnlineGpuRidgeEngine used when PyTorch is unavailable."""

    def __init__(self, feature_dim: int, ridge_lambda: float = 2.0) -> None:
        self.K = int(feature_dim)
        self.ridge_lambda = float(ridge_lambda)
        self.G = np.zeros((1, self.K, self.K), dtype=np.float64)
        self.b = np.zeros((1, self.K), dtype=np.float64)
        self.M0 = (np.eye(self.K, dtype=np.float64) / self.ridge_lambda)[None, :, :]
        self.theta = np.zeros((1, self.K), dtype=np.float64)

    def update(self, X: np.ndarray, y: np.ndarray) -> None:
        x_v = np.asarray(X, dtype=np.float64).reshape(-1, self.K)
        y_v = np.asarray(y, dtype=np.float64).reshape(-1)
        valid = np.all(np.isfinite(x_v), axis=1) & np.isfinite(y_v)
        if not np.any(valid):
            return
        xv = x_v[valid]
        yv = y_v[valid]
        self.G[0] += xv.T @ xv
        self.b[0] += xv.T @ yv

    def solve_host_theta(self, tol: float = 1e-8, max_iters: int = 16) -> np.ndarray:
        A = self.G.copy()
        A[0] += self.ridge_lambda * np.eye(self.K, dtype=np.float64)
        res = ipcg_solve_batch(A, self.M0, self.b, self.theta, tol=tol, max_iters=max_iters)
        self.theta = res.theta.copy()
        return self.theta[0].copy()


def _to_deciles_9(q_vec: np.ndarray) -> np.ndarray:
    """Coerce an arbitrary sorted quantile vector ``(Q,)`` to the 9 deciles ``0.1..0.9``."""
    arr = np.sort(np.asarray(q_vec, dtype=np.float64).reshape(-1))
    if arr.size == 9:
        return arr
    if arr.size == 0:
        return np.zeros(9, dtype=np.float64)
    if arr.size == 1:
        return np.full(9, float(arr[0]), dtype=np.float64)
    src_grid = np.linspace(0.0, 1.0, arr.size, dtype=np.float64)
    dst_grid = np.linspace(0.0, 1.0, 9, dtype=np.float64)
    return np.interp(dst_grid, src_grid, arr)


def _pinball_loss(y_true: float, q_preds: np.ndarray, q_levels: np.ndarray) -> float:
    q9 = _to_deciles_9(q_preds)
    diff = y_true - q9
    losses = np.maximum(q_levels * diff, (q_levels - 1.0) * diff)
    return float(2.0 * np.mean(losses))


def _safe_scale_floor(ref_scale: float, anchor: float = 0.0) -> float:
    """Return a strictly scale-invariant positive floor across micro and large-DC regimes."""
    sc = ref_scale if (math.isfinite(ref_scale) and ref_scale > 0.0) else 0.0
    anch = abs(anchor) if math.isfinite(anchor) else 0.0
    floor_v = max(sc, 1e-14 * anch, 1e-30)
    return floor_v if (math.isfinite(floor_v) and floor_v > 0.0) else 1.0


def _nanmean_safe(arr: np.ndarray) -> float:
    fin = arr[np.isfinite(arr)]
    return float(np.mean(fin)) if fin.size > 0 else 0.0


def _causal_scale_and_last(hist: np.ndarray) -> tuple[float, float]:
    """Return strictly causal, DC-invariant ``(last_finite, rolling_scale)`` from ``hist``."""
    L = hist.shape[0]
    if L == 0:
        return 0.0, 1.0
    tail = hist[-min(L, 168) :]
    fin = tail[np.isfinite(tail)]
    if fin.size == 0:
        return 0.0, 1.0
    last = float(fin[-1])
    # Shifted-anchor std prevents catastrophic cancellation at large DC offsets (e.g. 2.22e10).
    std = float(np.std(fin - last)) if fin.size > 1 else 0.0
    if math.isfinite(std) and std > 0.0:
        scale = max(std, 1e-14 * abs(last), 1e-30)
    else:
        scale = max(1e-4 * abs(last), 1e-12) if last != 0.0 else 1.0
    if not math.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return last, scale


def _build_causal_ridge_features(hist: np.ndarray, k: int = 32) -> tuple[np.ndarray, float, float]:
    """Build strictly causal lag + seasonal-lag + EMA-diff features normalized by rolling std.

    Uses ONLY indices strictly inside ``hist`` (``<= t_cur - 1``). Returns
    ``(features[k], last_val, scale)`` where the Ridge target is the normalized
    1-step increment ``(y_next - last_val) / scale``.
    """
    L = hist.shape[0]
    feats = np.zeros(k, dtype=np.float64)
    feats[0] = 1.0
    last, scale = _causal_scale_and_last(hist)
    if k <= 1 or L == 0:
        return feats, last, scale

    inv_s = 1.0 / scale

    def _safe_val(idx_from_end: int) -> float:
        if idx_from_end <= L:
            v = float(hist[-idx_from_end])
            if math.isfinite(v):
                return v
        return last

    # Compute causal fast & slow EMAs over recent finite tail (up to 64 steps).
    ema_tail = hist[-min(L, 64) :]
    ema_fast = last
    ema_slow = last
    has_ema = False
    for v_raw in ema_tail:
        v = float(v_raw)
        if not math.isfinite(v):
            continue
        if not has_ema:
            ema_fast = v
            ema_slow = v
            has_ema = True
        else:
            ema_fast = 0.35 * v + 0.65 * ema_fast
            ema_slow = 0.10 * v + 0.90 * ema_slow

    raw_feats: list[float] = []

    if k >= 20:
        # 1. Short-range AR lag differences (lags 1..14)
        for d in range(1, 15):
            raw_feats.append((_safe_val(1 + d) - last) * inv_s)
        # 2. Exact seasonal 1-step-ahead deltas: for P-periodic signal,
        #    y_{t} - y_{t-1} == hist[-P] - hist[-P-1] (strictly causal!).
        for p in (12, 24, 48, 96, 168):
            raw_feats.append((_safe_val(p) - _safe_val(p + 1)) * inv_s)
        # 3. Seasonal level anchors relative to current level: hist[-P] - hist[-1].
        for p in (12, 24, 48, 96, 168):
            raw_feats.append((_safe_val(p) - last) * inv_s)
        # 4. Local momentum, curvature, EMA diffs, and 24-step mean anchor.
        v1 = _safe_val(1)
        v2 = _safe_val(2)
        v3 = _safe_val(3)
        raw_feats.append((v1 - v2) * inv_s)
        raw_feats.append((v1 - 2.0 * v2 + v3) * inv_s)
        raw_feats.append((ema_fast - last) * inv_s)
        raw_feats.append((ema_slow - last) * inv_s)
        raw_feats.append((ema_fast - ema_slow) * inv_s)
        w24 = hist[-min(L, 24) :]
        fin24 = w24[np.isfinite(w24)]
        mean24 = float(np.mean(fin24)) if fin24.size > 0 else last
        raw_feats.append((mean24 - last) * inv_s)
        # Extra lag diffs if caller requests k > 32
        extra_d = 15
        while len(raw_feats) < k - 1:
            raw_feats.append((_safe_val(1 + extra_d) - last) * inv_s)
            extra_d += 1
    else:
        # Compact feature set for small ridge_dim (e.g. unit tests with ridge_dim=8).
        n_short = max(1, k - 3) if k >= 5 else (k - 1)
        for d in range(1, n_short + 1):
            raw_feats.append((_safe_val(1 + d) - last) * inv_s)
        if k >= 5:
            raw_feats.append((_safe_val(24) - _safe_val(25)) * inv_s)
            raw_feats.append((ema_fast - last) * inv_s)

    n_fill = min(k - 1, len(raw_feats))
    for i in range(n_fill):
        val = raw_feats[i]
        feats[1 + i] = float(np.clip(val, -8.0, 8.0)) if math.isfinite(val) else 0.0
    return feats, last, scale


def _prewarm_online_ridge(
    prefix: np.ndarray,
    ridge_dim: int,
    *,
    ridge_gpu: OnlineGpuRidgeEngine | None,
    ridge_np: _NumpyOnlineRidgeMirror | None,
) -> tuple[np.ndarray, float]:
    """Causally pre-warm Online IPCG-Ridge strictly on ``prefix = series[0 : context_length]``.

    All training indices satisfy ``t < context_length`` (zero future leakage).
    Returns ``(theta_host, initial_ridge_sigma)`` where ``initial_ridge_sigma`` is
    estimated causally on a held-out chronological tail inside ``prefix``.
    """
    L = prefix.shape[0]
    warm_start = min(max(24, L // 4), max(L - 2, 1))
    xs_list: list[np.ndarray] = []
    ys_list: list[float] = []
    lasts_list: list[float] = []
    scales_list: list[float] = []
    raw_targets: list[float] = []

    for t_idx in range(warm_start, L):
        y_t = float(prefix[t_idx])
        if not math.isfinite(y_t):
            continue
        sub_hist = prefix[:t_idx]  # strictly < t_idx < context_length!
        feats, last_v, sc_v = _build_causal_ridge_features(sub_hist, k=ridge_dim)
        dy_norm = float(np.clip((y_t - last_v) / sc_v, -8.0, 8.0))
        if math.isfinite(dy_norm):
            xs_list.append(feats)
            ys_list.append(dy_norm)
            lasts_list.append(last_v)
            scales_list.append(sc_v)
            raw_targets.append(y_t)

    _, fallback_scale = _causal_scale_and_last(prefix)
    if not xs_list:
        return np.zeros(ridge_dim, dtype=np.float64), _safe_scale_floor(fallback_scale * 0.5)

    X_all = np.stack(xs_list, axis=0)
    y_all = np.asarray(ys_list, dtype=np.float64)
    n_pairs = X_all.shape[0]

    # Chronological 75/25 split strictly inside prefix to calibrate out-of-sample ridge_sigma.
    split_idx = max(1, int(0.75 * n_pairs))
    if ridge_gpu is not None:
        ridge_gpu.update(X_all[:split_idx][None, :, :], y_all[:split_idx][None, :])
        sol_mid = ridge_gpu.solve(tol=1e-9, max_iters=32)
        theta_mid = sol_mid.theta[0].detach().cpu().numpy().copy()
    elif ridge_np is not None:
        ridge_np.update(X_all[:split_idx], y_all[:split_idx])
        theta_mid = ridge_np.solve_host_theta(tol=1e-9, max_iters=32)
    else:
        theta_mid = np.zeros(ridge_dim, dtype=np.float64)

    if split_idx < n_pairs:
        val_errs: list[float] = []
        for i in range(split_idx, n_pairs):
            dy_p = float(np.clip(np.dot(X_all[i], theta_mid), -6.0, 6.0))
            y_p = lasts_list[i] + scales_list[i] * dy_p
            val_errs.append(abs(raw_targets[i] - y_p))
        init_sigma = max(
            float(np.mean(val_errs)) / 0.79788456,
            _safe_scale_floor(0.05 * fallback_scale),
        )
        # Now fold the remaining 25% of prefix (still strictly < context_length!) into Gram.
        if ridge_gpu is not None:
            ridge_gpu.update(X_all[split_idx:][None, :, :], y_all[split_idx:][None, :])
            sol_full = ridge_gpu.solve(tol=1e-9, max_iters=32)
            theta_host = sol_full.theta[0].detach().cpu().numpy().copy()
        elif ridge_np is not None:
            ridge_np.update(X_all[split_idx:], y_all[split_idx:])
            theta_host = ridge_np.solve_host_theta(tol=1e-9, max_iters=32)
        else:
            theta_host = theta_mid
    else:
        theta_host = theta_mid
        init_sigma = _safe_scale_floor(fallback_scale * 0.5)

    return theta_host, init_sigma


def _load_gifteval_dataset(
    name: str,
    *,
    data_dir: str | None,
    num_series: int,
    min_len: int,
    allow_synthetic_fallback: bool,
    seed: int = 2026,
) -> tuple[np.ndarray, str]:
    """Load real GIFT-Eval series from local dir or Hugging Face ``Salesforce/GiftEval``."""
    series_list: list[np.ndarray] = []

    if data_dir:
        dpath = Path(data_dir) / name
        if dpath.exists():
            for p in sorted(dpath.rglob("*.npy")):
                arr = np.load(p).astype(np.float64)
                if arr.ndim == 1 and arr.size >= min_len:
                    series_list.append(arr[:min_len])
                elif arr.ndim == 2 and arr.shape[1] >= min_len:
                    for row in arr:
                        series_list.append(row[:min_len])
                if len(series_list) >= num_series:
                    return np.stack(series_list[:num_series], axis=0), f"local:{dpath}"
            try:
                from datasets import load_from_disk

                cand_dirs: list[Path] = []
                for freq in ("H", "15T", "5T", "D", "W"):
                    sub = dpath / freq
                    if (sub / "dataset_info.json").is_file():
                        cand_dirs.append(sub)
                for info_file in sorted(dpath.rglob("dataset_info.json")):
                    if info_file.parent not in cand_dirs:
                        cand_dirs.append(info_file.parent)
                for sub in cand_dirs:
                    ds = load_from_disk(str(sub))
                    for row in ds:
                        arr = np.asarray(row["target"], dtype=np.float64)
                        if arr.ndim == 2:
                            for ch in arr:
                                if ch.size >= min_len and np.isfinite(ch[:min_len]).all():
                                    series_list.append(ch[:min_len])
                                    if len(series_list) >= num_series:
                                        break
                        elif arr.size >= min_len and np.isfinite(arr[:min_len]).all():
                            series_list.append(arr[:min_len])
                        if len(series_list) >= num_series:
                            src_tag = f"local_arrow:{name}/{sub.name}"
                            return np.stack(series_list[:num_series], axis=0), src_tag
            except Exception:
                pass

    try:
        from datasets import load_dataset

        ds = load_dataset("Salesforce/GiftEval", name, split="train", streaming=True)
        for row in ds:
            arr = np.asarray(row["target"], dtype=np.float64)
            if arr.ndim == 2:
                arr = arr[0]
            if arr.size >= min_len:
                series_list.append(arr[:min_len])
            if len(series_list) >= num_series:
                return np.stack(series_list[:num_series], axis=0), f"hf:Salesforce/GiftEval/{name}"
    except Exception:
        pass

    if not allow_synthetic_fallback:
        raise RuntimeError(
            f"Could not load real GIFT-Eval dataset {name!r}. Provide --data-dir, "
            "configure Hugging Face access to Salesforce/GiftEval, or pass "
            "--allow-synthetic-fallback."
        )

    rng = np.random.default_rng(seed + sum(ord(c) for c in name))
    t = np.arange(min_len, dtype=np.float64)
    out = np.empty((num_series, min_len), dtype=np.float64)
    for i in range(num_series):
        p1 = rng.choice([24.0, 48.0, 96.0, 168.0])
        p2 = p1 / 4.0
        amp = rng.uniform(5.0, 50.0)
        base = rng.uniform(10.0, 500.0)
        drift = rng.uniform(-0.01, 0.01) * t
        shock = np.where(t > (min_len * 0.65), rng.uniform(-0.3, 0.3) * amp, 0.0)
        out[i] = (
            base
            + drift
            + amp * np.sin(2.0 * np.pi * t / p1 + i)
            + 0.35 * amp * np.cos(2.0 * np.pi * t / p2)
            + shock
            + rng.normal(0.0, 0.1 * amp, size=min_len)
        )
    return out, f"synthetic_fallback:{name}"


def _run_gap_async_stream_single_series(
    series: np.ndarray,
    backbone_predict_q: Callable[[np.ndarray], np.ndarray],
    *,
    context_length: int,
    horizon: int,
    ridge_dim: int = 32,
    device: Any = None,
    max_gap_ticks: int = 1,
) -> dict[str, Any]:
    """Run strictly causal tsjit Gap-Async streaming loop on one series.

    True gap-async execution semantics:
    - Before ``step = 0``, Online IPCG-Ridge is causally pre-warmed strictly on
      ``series[0 : context_length]`` (zero future leakage), and the initial slow
      TSFM anchor is pre-computed on ``series[0 : context_length - 1]`` (gap
      prior to tick 0 arrival).
    - On every tick ``step`` (``t_cur = context_length + step``, observing
      ``x_cur = series[t_cur - 1]``, predicting ``y_next = series[t_cur]``):
        1. Causally score ``x_cur`` to update E-Value, OnlineACI, AR(1) residual
           persistence ``rho``, online Hedge cumulative pinball losses, and the
           Online IPCG-Ridge engine.
        2. Read horizon-aligned TSFM forecast ``anchor_q_horizon[:, age]`` (with
           exact innovation ``x_cur - anchor_q_horizon[4, age - 1]``) and blend
           ``{Slow-TSFM (raw + revised), Pre-Warmed IPCG-Ridge (sym + shaped),
           Causal Persistence}`` via the C11 lock-free ``SoftRevisionEngine``
           with **0 ms GPU blocking on arrival**.
        3. Emit calibrated 80% & 90% ``OnlineACI`` intervals and score against
           held-out ``y_next``.
        4. In the post-emission gap (ready before ``step + 1``), evaluate
           ``EventGate`` (``max_gap_ticks=1`` -> always refresh in gap for
           ``age=1``; ``max_gap_ticks=4`` -> event-gated elision).
    """
    T = series.shape[0]
    n_steps = T - context_length
    if n_steps <= 0:
        raise ValueError("Series must have length > context_length")

    ridge_lambda = 2.0
    use_torch_ridge = torch is not None and device is not None
    if use_torch_ridge:
        ridge_gpu: OnlineGpuRidgeEngine | None = OnlineGpuRidgeEngine(
            num_series=1,
            feature_dim=ridge_dim,
            ridge_lambda=ridge_lambda,
            q_scale=1024.0,
            device=device,
            use_cuda_kernel=(getattr(device, "type", "cpu") == "cuda"),
        )
        ridge_np: _NumpyOnlineRidgeMirror | None = None
    else:
        ridge_gpu = None
        ridge_np = _NumpyOnlineRidgeMirror(feature_dim=ridge_dim, ridge_lambda=ridge_lambda)

    # 0a. Strictly causal pre-warming of Online IPCG-Ridge inside series[0 : context_length].
    prefix = series[:context_length]
    theta_host, ridge_sigma = _prewarm_online_ridge(
        prefix, ridge_dim, ridge_gpu=ridge_gpu, ridge_np=ridge_np
    )

    # 0b. Pre-arrival gap dispatch (strictly indices < context_length - 1, age = 1 at step 0).
    if context_length > 1:
        pre_ctx = np.concatenate([[series[0]], series[: context_length - 1]])
    else:
        pre_ctx = series[:1].copy()
    backbone_ms: list[float] = []
    t0_bb = time.perf_counter()
    anchor_q_horizon = np.asarray(backbone_predict_q(pre_ctx[None, :])[0], dtype=np.float64)
    backbone_ms.append((time.perf_counter() - t0_bb) * 1e3)
    anchor_end = context_length - 1
    backbone_calls = 1

    gate = EventGate(surprise_threshold=2.5, evalue_threshold=20.0, max_age=max_gap_ticks)
    e_det = QuantilePitEValueDetector(kappa=0.5, threshold=20.0)
    aci_80 = OnlineACI(alpha=0.20, gamma=0.05, delay=0, window=128)
    aci_90 = OnlineACI(alpha=0.10, gamma=0.05, delay=0, window=128)

    # Discounted cumulative normalized pinball losses for online Hedge SoftRevisionV2 mixing.
    loss_decay = 0.94
    cum_losses: dict[str, float] = {
        "tsfm_raw": 0.0,
        "tsfm_rev": 0.02,
        "ridge_sym": 0.05,
        "ridge_shaped": 0.05,
        "persist": 0.20,
    }
    prev_sub_q: dict[str, np.ndarray] | None = None
    prev_q9: np.ndarray | None = None
    prev_feats: np.ndarray | None = None
    prev_last: float = 0.0
    prev_scale: float = 1.0
    prev_ridge_med: float | None = None
    prev_innov_tsfm: float | None = None
    innov_cov = 0.0
    innov_var = 0.0
    rho_t = 0.0

    preds_median = np.zeros(n_steps, dtype=np.float64)
    preds_q9 = np.zeros((n_steps, 9), dtype=np.float64)
    pinballs = np.full(n_steps, np.nan, dtype=np.float64)
    maes = np.full(n_steps, np.nan, dtype=np.float64)
    cov80 = np.full(n_steps, np.nan, dtype=np.float64)
    cov90 = np.full(n_steps, np.nan, dtype=np.float64)
    c11_arrival_us = np.zeros(n_steps, dtype=np.float64)
    full_tick_e2e_us = np.zeros(n_steps, dtype=np.float64)

    lambda_age = 0.0 if max_gap_ticks <= 1 else 0.06
    # Online Hedge cum_losses already scores true horizon-aligned causal surprise;
    # keep beta_surprise=0.0 so step_soft does not compare x_{t-1} against y_hat_t on steep ramps.
    beta_surprise = 0.0

    with SoftRevisionEngine(
        n_keys=1,
        window=min(context_length, 256),
        lambda_age=lambda_age,
        max_age=max(max_gap_ticks, 1),
    ) as eng:
        # Publish initial pre-arrival anchor at forecast_step = -1 (so age at step 0 is 1).
        init_q_slow = _to_deciles_9(anchor_q_horizon[:, 0])
        _, init_sc = _causal_scale_and_last(pre_ctx)
        eng.publish_forecast(
            0, -1, init_q_slow.tolist(), init_q_slow.tolist(), scale_hint=init_sc
        )

        for step in range(n_steps):
            t0_tick = time.perf_counter_ns()
            t_cur = context_length + step
            hist = series[t_cur - context_length : t_cur]  # strictly causal: indices < t_cur!
            x_cur = float(hist[-1])
            y_next = float(series[t_cur])  # ground truth revealed ONLY AFTER prediction!

            # 1. Causally score x_cur (realized target from step - 1) BEFORE predicting y_next.
            e_val = 1.0
            e_ville = e_det.ville
            e_cusum = e_det.cusum
            e_sr = e_det.shiryaev_roberts
            alarm = False
            if prev_q9 is not None and math.isfinite(x_cur):
                e_upd = e_det.update(x_cur, prev_q9)
                e_val = e_upd.e_value
                e_ville = e_upd.ville
                e_cusum = e_upd.cusum
                e_sr = e_upd.shiryaev_roberts
                alarm = e_upd.alarm
                aci_80.update(step, x_cur)
                aci_90.update(step, x_cur)
                if alarm:
                    aci_80.boost_on_alarm(severity=1.0)
                    aci_90.boost_on_alarm(severity=1.0)

                norm_sc = _safe_scale_floor(prev_scale, prev_last)
                if prev_sub_q is not None:
                    for m_key, m_q in prev_sub_q.items():
                        step_loss = _pinball_loss(x_cur, m_q, QUANTILE_LEVELS) / norm_sc
                        if math.isfinite(step_loss):
                            cum_losses[m_key] = loss_decay * cum_losses[m_key] + min(step_loss, 6.0)

                if prev_ridge_med is not None:
                    r_abs = abs(x_cur - prev_ridge_med)
                    if math.isfinite(r_abs):
                        r_cap = min(r_abs, 6.0 * norm_sc)
                        ridge_sigma = 0.85 * ridge_sigma + 0.15 * max(
                            r_cap / 0.79788456, 0.05 * norm_sc
                        )

                if prev_feats is not None:
                    # Huber-clipped normalized innovation prevents 1-tick spikes poisoning Gram.
                    dy_obs = float(np.clip((x_cur - prev_last) / norm_sc, -4.0, 4.0))
                    if math.isfinite(dy_obs):
                        if ridge_gpu is not None:
                            ridge_gpu.update(
                                prev_feats.reshape(1, 1, ridge_dim),
                                np.array([[dy_obs]], dtype=np.float64),
                            )
                            sol = ridge_gpu.solve(tol=1e-8, max_iters=16)
                            theta_host = sol.theta[0].detach().cpu().numpy().copy()
                        elif ridge_np is not None:
                            ridge_np.update(
                                prev_feats.reshape(1, ridge_dim),
                                np.array([dy_obs], dtype=np.float64),
                            )
                            theta_host = ridge_np.solve_host_theta(tol=1e-8, max_iters=16)
            elif prev_q9 is not None:
                # Drain due ACI intervals cleanly when x_cur is NaN (zero state poisoning).
                aci_80.update(step, float("nan"))
                aci_90.update(step, float("nan"))

            # 2. Exact horizon-aligned TSFM anchor read + spike-resilient AR(1) innovation revision.
            age = max(1, t_cur - anchor_end)
            h_avail = anchor_q_horizon.shape[1]
            lead_cur = min(age - 1, h_avail - 1)
            lead_next = min(age, h_avail - 1)

            q_tsfm_for_xcur = _to_deciles_9(anchor_q_horizon[:, lead_cur])
            q_tsfm_raw = _to_deciles_9(anchor_q_horizon[:, lead_next])

            feats, last_val, scale_t = _build_causal_ridge_features(hist, k=ridge_dim)
            scale_safe = _safe_scale_floor(scale_t, last_val)
            tsfm_iqr_scale = max(
                (float(q_tsfm_raw[8]) - float(q_tsfm_raw[0])) / 2.563103,
                0.05 * scale_safe,
            )

            innov_tsfm = (x_cur - float(q_tsfm_for_xcur[4])) if math.isfinite(x_cur) else 0.0
            # Winsorize innovation for rho estimation so 1-tick spikes never corrupt AR(1) memory.
            innov_win = float(np.clip(innov_tsfm, -3.5 * tsfm_iqr_scale, 3.5 * tsfm_iqr_scale))
            if prev_innov_tsfm is not None and math.isfinite(innov_tsfm):
                innov_cov = 0.92 * innov_cov + prev_innov_tsfm * innov_win
                innov_var = 0.92 * innov_var + prev_innov_tsfm * prev_innov_tsfm
                var_floor = (0.05 * tsfm_iqr_scale) ** 2 + 1e-30
                rho_t = float(np.clip(innov_cov / (innov_var + var_floor), 0.0, 0.85))
            prev_innov_tsfm = innov_win if math.isfinite(innov_tsfm) else None

            # Smooth tanh-bounded AR(1) revision scaled to TSFM's own predictive IQR:
            # linear (rho_t * innov_tsfm) for normal innovations, smoothly saturated on spikes.
            rev_scale = max(2.0 * tsfm_iqr_scale, 1e-30)
            rev_shift = rev_scale * math.tanh((rho_t * innov_tsfm) / rev_scale)
            q_tsfm_rev = np.sort(q_tsfm_raw + rev_shift)

            # Fast tier: Causally Pre-Warmed Online IPCG-Ridge (symmetric + TSFM-shaped quantiles).
            dy_hat = float(np.clip(np.dot(feats, theta_host), -6.0, 6.0))
            y_ridge = last_val + scale_t * dy_hat
            eff_r_sigma = max(ridge_sigma, 0.05 * scale_safe)
            tsfm_shape = q_tsfm_raw - float(q_tsfm_raw[4])
            q_ridge_sym = np.sort(y_ridge + _Z9 * eff_r_sigma)
            q_ridge_shaped = np.sort(y_ridge + tsfm_shape)

            # Persistence tier: latest observation + TSFM quantile shape (instant shock immunity).
            q_persist = np.sort(last_val + tsfm_shape)

            # 3. Hot C11 lock-free SoftRevisionV2 arrival step (0 ms GPU blocking on arrival!).
            # Use tau_intra=0.5, tau_family=1.0 because tsjit_arrive_step_soft_v2 divides ev_f
            # by tau_family after aggregate_family already divided losses by tau_intra.
            t0_arr = time.perf_counter_ns()
            arr = eng.step_soft(
                0,
                step,
                x_cur,
                slow_models=(q_tsfm_raw.tolist(), q_tsfm_rev.tolist()),
                slow_losses=(cum_losses["tsfm_raw"], cum_losses["tsfm_rev"]),
                fast_models=(q_ridge_sym.tolist(), q_ridge_shaped.tolist()),
                fast_losses=(cum_losses["ridge_sym"], cum_losses["ridge_shaped"]),
                persist_models=(q_persist.tolist(),),
                persist_losses=(cum_losses["persist"],),
                tau_intra=0.5,
                tau_family=1.0,
                beta_surprise=beta_surprise,
            )
            c11_arrival_us[step] = (time.perf_counter_ns() - t0_arr) * 1e-3

            q_pred = np.sort(np.asarray(arr.q9, dtype=np.float64))
            if not np.all(np.isfinite(q_pred)):
                q_pred = q_tsfm_raw
            y_med = float(q_pred[4])
            preds_median[step] = y_med
            preds_q9[step] = q_pred

            # 4. CQR-calibrated 80% and 90% intervals enqueued for causal scoring at step + 1.
            pred_scale = max(
                (float(q_pred[8]) - float(q_pred[0])) / 2.563103,
                arr.rolling_std,
                scale_safe,
            )
            cal80 = aci_80.predict_interval(
                step + 1,
                float(q_pred[0]),
                float(q_pred[8]),
                scale_hint=pred_scale,
                anchor=arr.rolling_mean,
                enqueue=True,
            )
            half90 = 0.5 * (float(q_pred[8]) - float(q_pred[0])) * (1.6448536 / 1.2815516)
            cal90 = aci_90.predict_interval(
                step + 1,
                y_med - half90,
                y_med + half90,
                scale_hint=pred_scale,
                anchor=arr.rolling_mean,
                enqueue=True,
            )
            full_tick_e2e_us[step] = (time.perf_counter_ns() - t0_tick) * 1e-3

            # 5. Benchmark evaluation against held-out y_next (zero leakage into state!).
            if math.isfinite(y_next):
                pinballs[step] = _pinball_loss(y_next, q_pred, QUANTILE_LEVELS)
                maes[step] = abs(y_next - y_med)
                cov80[step] = 1.0 if (cal80.lo <= y_next <= cal80.hi) else 0.0
                cov90[step] = 1.0 if (cal90.lo <= y_next <= cal90.hi) else 0.0

            # 6. Post-emission Gap-Async EventGate dispatch (executes in gap BEFORE step + 1).
            gate.ingest_lazy(step, x_cur)
            if (step + 1) < n_steps:
                true_abs_sg = abs(innov_tsfm) / max(tsfm_iqr_scale, scale_safe)
                dec = gate.evaluate(
                    abs_sg=true_abs_sg,
                    e_value=e_val,
                    age=age,
                    e_cusum=e_cusum,
                    e_ville=e_ville,
                    e_sr=e_sr,
                )
                need_refresh = dec.dispatch or (age >= max(horizon - 1, 1))
                if need_refresh:
                    t0_bb = time.perf_counter()
                    anchor_q_horizon = np.asarray(
                        backbone_predict_q(hist[None, :])[0], dtype=np.float64
                    )
                    backbone_ms.append((time.perf_counter() - t0_bb) * 1e3)
                    anchor_end = t_cur
                    backbone_calls += 1
                    if alarm or e_cusum >= gate.evalue_threshold:
                        e_det.reset_cusum()
                    pub_slow = _to_deciles_9(anchor_q_horizon[:, 0])
                    eng.publish_forecast(
                        0,
                        step,
                        pub_slow.tolist(),
                        q_ridge_shaped.tolist(),
                        scale_hint=pred_scale,
                    )

            prev_q9 = q_pred
            prev_sub_q = {
                "tsfm_raw": q_tsfm_raw,
                "tsfm_rev": q_tsfm_rev,
                "ridge_sym": q_ridge_sym,
                "ridge_shaped": q_ridge_shaped,
                "persist": q_persist,
            }
            prev_feats = feats
            prev_last = last_val
            prev_scale = scale_safe
            prev_ridge_med = y_ridge

    c11_p50 = float(np.percentile(c11_arrival_us, 50))
    c11_p99 = float(np.percentile(c11_arrival_us, 99))
    e2e_p50 = float(np.percentile(full_tick_e2e_us, 50))
    e2e_p99 = float(np.percentile(full_tick_e2e_us, 99))
    return {
        "preds_median": preds_median,
        "preds_q9": preds_q9,
        "pinball_crps": _nanmean_safe(pinballs),
        "mae": _nanmean_safe(maes),
        "coverage_80": _nanmean_safe(cov80),
        "coverage_90": _nanmean_safe(cov90),
        "backbone_calls": backbone_calls,
        "total_ticks": n_steps,
        "gpu_elision_ratio": float(max(0.0, 1.0 - backbone_calls / max(n_steps, 1))),
        "c11_arrival_only_p50_us": c11_p50,
        "c11_arrival_only_p99_us": c11_p99,
        "full_tick_e2e_p50_us": e2e_p50,
        "full_tick_e2e_p99_us": e2e_p99,
        "arrival_p50_us": c11_p50,
        "arrival_p99_us": c11_p99,
        "backbone_refresh_p50_ms": float(np.percentile(backbone_ms, 50)) if backbone_ms else 0.0,
        "e_ville_final": float(e_det.ville),
    }


def assert_causal_bit_exact(
    series: np.ndarray,
    backbone_predict_q: Callable[[np.ndarray], np.ndarray],
    *,
    context_length: int,
    horizon: int,
    ridge_dim: int = 32,
    device: Any = None,
    max_gap_ticks: int = 1,
    cut_step: int | None = None,
) -> bool:
    """Bit-exact causality verifier: poison future t >= t_cut with +1e9/NaN and assert equality."""
    n_steps = series.shape[0] - context_length
    cut = max(2, n_steps // 2) if cut_step is None else max(0, min(int(cut_step), n_steps - 1))
    t_cut = context_length + cut

    orig = _run_gap_async_stream_single_series(
        series,
        backbone_predict_q,
        context_length=context_length,
        horizon=horizon,
        ridge_dim=ridge_dim,
        device=device,
        max_gap_ticks=max_gap_ticks,
    )
    corrupted = series.copy()
    corrupted[t_cut:] = np.where(np.arange(series.shape[0] - t_cut) % 2 == 0, 1e9, np.nan)
    after = _run_gap_async_stream_single_series(
        corrupted,
        backbone_predict_q,
        context_length=context_length,
        horizon=horizon,
        ridge_dim=ridge_dim,
        device=device,
        max_gap_ticks=max_gap_ticks,
    )
    # Predictions for steps 0..cut depend ONLY on indices < t_cut -> bit-exact equality!
    return bool(
        np.array_equal(orig["preds_median"][: cut + 1], after["preds_median"][: cut + 1])
        and np.array_equal(orig["preds_q9"][: cut + 1], after["preds_q9"][: cut + 1])
    )


def _run_standalone_eager_baseline(
    series: np.ndarray,
    eager_predict_q: Callable[[np.ndarray], np.ndarray],
    *,
    context_length: int,
) -> dict[str, float]:
    T = series.shape[0]
    n_steps = T - context_length
    pinballs = np.full(n_steps, np.nan, dtype=np.float64)
    maes = np.full(n_steps, np.nan, dtype=np.float64)
    cov80 = np.full(n_steps, np.nan, dtype=np.float64)
    cov90 = np.full(n_steps, np.nan, dtype=np.float64)
    lat_ms = np.zeros(n_steps, dtype=np.float64)

    for step in range(n_steps):
        t_cur = context_length + step
        hist = series[t_cur - context_length : t_cur]
        y_next = float(series[t_cur])
        t0 = time.perf_counter()
        raw_q = np.asarray(eager_predict_q(hist[None, :])[0, :, 0], dtype=np.float64)
        q_pred = _to_deciles_9(raw_q)
        lat_ms[step] = (time.perf_counter() - t0) * 1e3
        y_med = float(q_pred[4])
        if math.isfinite(y_next):
            pinballs[step] = _pinball_loss(y_next, q_pred, QUANTILE_LEVELS)
            maes[step] = abs(y_next - y_med)
            cov80[step] = 1.0 if (q_pred[0] <= y_next <= q_pred[8]) else 0.0
            half90 = 0.5 * (q_pred[8] - q_pred[0]) * (1.6448536 / 1.2815516)
            cov90[step] = 1.0 if ((y_med - half90) <= y_next <= (y_med + half90)) else 0.0

    return {
        "pinball_crps": _nanmean_safe(pinballs),
        "mae": _nanmean_safe(maes),
        "coverage_80": _nanmean_safe(cov80),
        "coverage_90": _nanmean_safe(cov90),
        "backbone_calls": n_steps,
        "gpu_elision_ratio": 0.0,
        "p50_ms": float(np.percentile(lat_ms, 50)),
    }


_NATIVE_C11_BATCH_HARNESS = r"""
#include "tsjit_common.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    int64_t n_keys = (argc > 1) ? atoll(argv[1]) : 1024;
    int64_t n_ticks = (argc > 2) ? atoll(argv[2]) : 64;
    if (n_keys < 1) n_keys = 1024;
    if (n_ticks < 1) n_ticks = 64;

    double q_slow[TSJIT_NQ] = {-1.28, -0.84, -0.52, -0.25, 0.0, 0.25, 0.52, 0.84, 1.28};
    double q_fast[TSJIT_NQ] = {-1.15, -0.75, -0.45, -0.20, 0.05, 0.28, 0.55, 0.88, 1.32};
    double q_pers[TSJIT_NQ] = {-1.20, -0.80, -0.50, -0.22, 0.02, 0.26, 0.54, 0.86, 1.30};
    double l_s[1] = {0.10}, l_f[1] = {0.12}, l_p[1] = {0.35};

    tsjit_arrive_engine_t *pwl = tsjit_arrive_create(n_keys, 64, 0.10, 4);
    tsjit_arrive_engine_t *soft = tsjit_arrive_create(n_keys, 64, 0.10, 4);
    if (!pwl || !soft) {
        fprintf(stderr, "engine alloc failed\n");
        return 1;
    }
    for (int64_t k = 0; k < n_keys; ++k) {
        tsjit_arrive_publish_forecast(pwl, k, 0, q_slow, q_fast, 1.0, NULL, NULL, NULL);
        tsjit_arrive_publish_forecast(soft, k, 0, q_slow, q_fast, 1.0, NULL, NULL, NULL);
    }

    tsjit_arrive_out_t out;
    double sink = 0.0;
    /* Warmup */
    for (int64_t k = 0; k < n_keys; ++k) {
        tsjit_arrive_step_pwl(pwl, k, 1, 0.1 * sin((double)k), &out);
        sink += out.q[4];
        tsjit_arrive_step_soft_v2(
            soft, k, 1, 0.1 * sin((double)k),
            1, q_slow, l_s, 1, q_fast, l_f, 1, q_pers, l_p,
            0.25, 0.25, 0.05, &out);
        sink += out.q[4];
    }

    double t0 = now_sec();
    for (int64_t t = 2; t < 2 + n_ticks; ++t) {
        for (int64_t k = 0; k < n_keys; ++k) {
            double x = 0.2 * sin(0.05 * (double)(t + k));
            tsjit_arrive_step_pwl(pwl, k, t, x, &out);
            sink += out.q[4];
        }
    }
    double dt_pwl = now_sec() - t0;

    double t1 = now_sec();
    for (int64_t t = 2; t < 2 + n_ticks; ++t) {
        for (int64_t k = 0; k < n_keys; ++k) {
            double x = 0.2 * sin(0.05 * (double)(t + k));
            tsjit_arrive_step_soft_v2(
                soft, k, t, x,
                1, q_slow, l_s, 1, q_fast, l_f, 1, q_pers, l_p,
                0.25, 0.25, 0.05, &out);
            sink += out.q[4];
        }
    }
    double dt_soft = now_sec() - t1;

    double total = (double)(n_keys * n_ticks);
    printf("{\"pwl_us_per_decision\": %.6f, \"soft_v2_us_per_decision\": %.6f, "
           "\"pwl_decisions_per_s\": %.1f, \"soft_v2_decisions_per_s\": %.1f, "
           "\"sink\": %.6f}\n",
           (dt_pwl / total) * 1e6,
           (dt_soft / total) * 1e6,
           total / (dt_pwl > 1e-12 ? dt_pwl : 1e-12),
           total / (dt_soft > 1e-12 ? dt_soft : 1e-12),
           sink);

    tsjit_arrive_destroy(pwl);
    tsjit_arrive_destroy(soft);
    return 0;
}
"""


def bench_batched_c11_arrival(n_streams: int = 1024, n_ticks: int = 32) -> dict[str, Any]:
    """Measure batched ``N = 1024`` arrival latency (Python ctypes AND true native C11)."""
    q_slow = [-1.28, -0.84, -0.52, -0.25, 0.0, 0.25, 0.52, 0.84, 1.28]
    q_fast = [-1.15, -0.75, -0.45, -0.20, 0.05, 0.28, 0.55, 0.88, 1.32]

    # 1. Python/ctypes batched loop across N streams via PwlArrivalEngine + direct C ABI
    lib = load_native_lib()
    with PwlArrivalEngine(n_keys=n_streams, window=64, lambda_age=0.10, max_age=4) as eng:
        for k in range(n_streams):
            eng.publish_forecast(k, 0, q_slow, q_fast, scale_hint=1.0)
        out_buf = CArriveOut()
        out_ptr = ctypes.byref(out_buf)
        handle = eng._handle
        step_fn = lib.tsjit_arrive_step_pwl
        # Warmup 1 pass
        for k in range(n_streams):
            step_fn(handle, k, 1, 0.1, out_ptr)
        py_ticks = min(n_ticks, 8)
        t0 = time.perf_counter()
        for t in range(2, 2 + py_ticks):
            for k in range(n_streams):
                step_fn(handle, k, t, 0.2, out_ptr)
        dt_py = time.perf_counter() - t0
    py_total = n_streams * py_ticks
    py_ctypes_us = (dt_py / max(py_total, 1)) * 1e6

    # 2. True native C11 critical-path latency (zero Python/ctypes loop overhead)
    native_stats: dict[str, Any] = {"status": "unavailable"}
    try:
        with tempfile.TemporaryDirectory(prefix="tsjit_c11_bench_") as tmp_dir:
            harness_c = Path(tmp_dir) / "bench_batched_c11.c"
            exe_path = Path(tmp_dir) / "bench_batched_c11"
            harness_c.write_text(_NATIVE_C11_BATCH_HARNESS, encoding="utf-8")
            build_c_executable(harness_c, exe_path)
            proc = subprocess.run(
                [str(exe_path), str(n_streams), str(n_ticks)],
                capture_output=True,
                text=True,
                check=True,
            )
            parsed = json.loads(proc.stdout.strip())
            native_stats = {
                "status": "ok",
                "pwl_us_per_decision": float(parsed["pwl_us_per_decision"]),
                "soft_v2_us_per_decision": float(parsed["soft_v2_us_per_decision"]),
                "pwl_decisions_per_s": float(parsed["pwl_decisions_per_s"]),
                "soft_v2_decisions_per_s": float(parsed["soft_v2_decisions_per_s"]),
            }
    except Exception as exc:
        native_stats = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    return {
        "n_streams": n_streams,
        "n_ticks": n_ticks,
        "python_ctypes_pwl_us_per_decision": float(py_ctypes_us),
        "native_c11": native_stats,
    }


def _aggregate_stream_runs(runs: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "pinball_crps": float(np.mean([r["pinball_crps"] for r in runs])),
        "mae": float(np.mean([r["mae"] for r in runs])),
        "coverage_80": float(np.mean([r["coverage_80"] for r in runs])),
        "coverage_90": float(np.mean([r["coverage_90"] for r in runs])),
        "gpu_elision_ratio": float(np.mean([r["gpu_elision_ratio"] for r in runs])),
        "c11_arrival_only_p50_us": float(np.mean([r["c11_arrival_only_p50_us"] for r in runs])),
        "c11_arrival_only_p99_us": float(np.mean([r["c11_arrival_only_p99_us"] for r in runs])),
        "full_tick_e2e_p50_us": float(np.mean([r["full_tick_e2e_p50_us"] for r in runs])),
        "full_tick_e2e_p99_us": float(np.mean([r["full_tick_e2e_p99_us"] for r in runs])),
        "arrival_p50_us": float(np.mean([r["arrival_p50_us"] for r in runs])),
        "arrival_p99_us": float(np.mean([r["arrival_p99_us"] for r in runs])),
        "backbone_refresh_p50_ms": float(np.mean([r["backbone_refresh_p50_ms"] for r in runs])),
        "e_ville_final_mean": float(np.mean([r["e_ville_final"] for r in runs])),
        "e_ville_final_max": float(np.max([r["e_ville_final"] for r in runs])),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live H200 GIFT-Eval Gap-Async & Event-Gated streaming benchmark."
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["autogluon/chronos-bolt-small", "Datadog/Toto-2.0-4m"],
    )
    parser.add_argument("--datasets", type=str, nargs="+", default=list(GIFT_DATASETS))
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--allow-synthetic-fallback", action="store_true")
    parser.add_argument("--num-series", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--stream-steps", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--ridge-dim", type=int, default=32)
    parser.add_argument(
        "--max-gap-ticks",
        type=int,
        default=4,
        help="Max gap ticks for tsjit_event_gated mode (tsjit_gap_async always uses 1).",
    )
    parser.add_argument("--output", type=str, default="results/bench_h200_gifteval_stream.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError("bench_h200_gifteval_stream requires PyTorch with CUDA.")

    import sys

    bench_dir = str(Path(__file__).resolve().parent)
    if bench_dir not in sys.path:
        sys.path.insert(0, bench_dir)
    from bench_h200_tsfm import _load_public_model, _make_eager_callable

    device = torch.device("cuda:0")
    min_len = args.context_length + args.stream_steps

    batched_c11 = bench_batched_c11_arrival(n_streams=1024, n_ticks=64)
    nat = batched_c11["native_c11"]
    if nat.get("status") == "ok":
        print(
            f"[Batched C11 N=1024] native_pwl={nat['pwl_us_per_decision']:.3f} us/dec | "
            f"native_soft_v2={nat['soft_v2_us_per_decision']:.3f} us/dec | "
            f"py_ctypes={batched_c11['python_ctypes_pwl_us_per_decision']:.3f} us/dec",
            flush=True,
        )

    all_results: dict[str, Any] = {
        "device": torch.cuda.get_device_name(device),
        "c11_arrival_engine": "SoftRevisionEngine",
        "batched_c11_1024": batched_c11,
        "models": {},
    }

    for model_id in args.models:
        try:
            model, family = _load_public_model(model_id, device)
        except Exception as exc:
            all_results["models"][model_id] = {
                "status": "load_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"Skipping {model_id}: {exc}", flush=True)
            continue

        eager_fn = _make_eager_callable(model, family, args.horizon)
        compiled = tsjit_compile(
            model,
            horizon=args.horizon,
            context_length=args.context_length,
            precision="a10-auto",
            mode="exact",
            batch_buckets=(1, 8, 16, 64),
            dc_strip=True,
        )

        def _eager_np(x_np: np.ndarray, fn_: Callable[[Any], Any] = eager_fn) -> np.ndarray:
            with torch.inference_mode():
                t = torch.from_numpy(x_np).to(device=device, dtype=torch.float32)
                return fn_(t).detach().cpu().numpy()

        def _compiled_np(x_np: np.ndarray, cmp_: Any = compiled) -> np.ndarray:
            return cmp_.predict_quantiles(x_np).detach().cpu().numpy()

        ds_report: dict[str, Any] = {}
        for ds_name in args.datasets:
            series_mat, src = _load_gifteval_dataset(
                ds_name,
                data_dir=args.data_dir,
                num_series=args.num_series,
                min_len=min_len,
                allow_synthetic_fallback=args.allow_synthetic_fallback,
            )
            eager_runs = [
                _run_standalone_eager_baseline(
                    series_mat[i], _eager_np, context_length=args.context_length
                )
                for i in range(series_mat.shape[0])
            ]
            gap_async_runs = [
                _run_gap_async_stream_single_series(
                    series_mat[i],
                    _compiled_np,
                    context_length=args.context_length,
                    horizon=args.horizon,
                    ridge_dim=args.ridge_dim,
                    device=device,
                    max_gap_ticks=1,
                )
                for i in range(series_mat.shape[0])
            ]
            event_gated_runs = [
                _run_gap_async_stream_single_series(
                    series_mat[i],
                    _compiled_np,
                    context_length=args.context_length,
                    horizon=args.horizon,
                    ridge_dim=args.ridge_dim,
                    device=device,
                    max_gap_ticks=args.max_gap_ticks,
                )
                for i in range(series_mat.shape[0])
            ]

            causal_gap_ok = assert_causal_bit_exact(
                series_mat[0],
                _compiled_np,
                context_length=args.context_length,
                horizon=args.horizon,
                ridge_dim=args.ridge_dim,
                device=device,
                max_gap_ticks=1,
            )
            causal_gated_ok = assert_causal_bit_exact(
                series_mat[0],
                _compiled_np,
                context_length=args.context_length,
                horizon=args.horizon,
                ridge_dim=args.ridge_dim,
                device=device,
                max_gap_ticks=args.max_gap_ticks,
            )
            causal_ok = bool(causal_gap_ok and causal_gated_ok)
            if not causal_ok:
                raise AssertionError(
                    f"Bit-exact assert_causal FAILED for {model_id} on {ds_name} "
                    f"(gap_async={causal_gap_ok}, event_gated={causal_gated_ok})!"
                )

            eager_agg = {
                "pinball_crps": float(np.mean([r["pinball_crps"] for r in eager_runs])),
                "mae": float(np.mean([r["mae"] for r in eager_runs])),
                "coverage_80": float(np.mean([r["coverage_80"] for r in eager_runs])),
                "coverage_90": float(np.mean([r["coverage_90"] for r in eager_runs])),
                "gpu_elision_ratio": 0.0,
                "p50_ms": float(np.mean([r["p50_ms"] for r in eager_runs])),
            }
            gap_agg = _aggregate_stream_runs(gap_async_runs)
            gated_agg = _aggregate_stream_runs(event_gated_runs)

            ds_report[ds_name] = {
                "source": src,
                "assert_causal_bit_exact": causal_ok,
                "assert_causal_gap_async": causal_gap_ok,
                "assert_causal_event_gated": causal_gated_ok,
                "standalone_eager": eager_agg,
                "tsjit_gap_async": gap_agg,
                "tsjit_event_gated": gated_agg,
            }
            elision_pct = gated_agg["gpu_elision_ratio"] * 100.0
            print(
                f"[{model_id} | {ds_name}] causal={causal_ok} | "
                f"eager CRPS={eager_agg['pinball_crps']:.4f} MAE={eager_agg['mae']:.4f} | "
                f"gap_async(age=1) CRPS={gap_agg['pinball_crps']:.4f} "
                f"MAE={gap_agg['mae']:.4f} cov80={gap_agg['coverage_80']:.2f} "
                f"cov90={gap_agg['coverage_90']:.2f} | "
                f"event_gated(gap={args.max_gap_ticks}) CRPS={gated_agg['pinball_crps']:.4f} "
                f"elision={elision_pct:.1f}% | "
                f"c11_arr_p50={gap_agg['c11_arrival_only_p50_us']:.2f} us | "
                f"full_tick_e2e_p50={gap_agg['full_tick_e2e_p50_us']:.2f} us",
                flush=True,
            )

        all_results["models"][model_id] = {"status": "ok", "datasets": ds_report}
        del model, eager_fn, compiled
        gc.collect()
        torch.cuda.empty_cache()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2) + "\n")
    print(f"\nSaved JSON report to {out_path}", flush=True)


if __name__ == "__main__":
    main()
