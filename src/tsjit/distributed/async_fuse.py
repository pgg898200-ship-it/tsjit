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

"""Distributed cross-series fusion (AsyncFuse & PipeFuse) and O(1) stream coordination.

Implements:
- ``AsyncFuseMonoid``: exact FP32 additive monoid ``m_b = [a_b * phi_b, a_b, sqrt(a_b)]``
  with T3 design-effect leave-one-out shrinkage:
    n_eff = (Q')^2 / A'
    A_eff = A' / (1 + max(n_eff - 1, 0) * rho)
    gamma = A_eff / (A_eff + kappa)
- ``SeriesLaggedLOOTracker``: series-ID-keyed lagged LOO table tracking
  ``(series_id, group_id)`` across cycles so dynamic batch joins, drops,
  reorders, and group changes never subtract another series's monoid entry.
- ``AsyncFuseRunner``: 3-graph ping-pong overlapped NCCL runner using 2 ping-pong
  buffer slots, a high-priority side ``comm_stream``, and single concatenated
  FP32 ``all_reduce(SUM)`` (~11-22 KB) overlapped behind compute.
- ``PipeFuseRunner``: exact zero-lag double-buffered panel split (G >= 2)
  achieving <= 1.2e-7 parity with central cross-series fusion.
- ``DistributedStreamCoordinator`` (``distributed_stream``): O(1) < 2 KB async
  NCCL monoid sync for cross-GPU conformal quantiles and swarm e-values.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - optional in pure-CPU environments
    torch = None  # type: ignore[assignment]
    dist = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Pure-NumPy exact reference for CPU / zero-torch verification
# ---------------------------------------------------------------------------


def async_fuse_monoid_numpy(
    h: np.ndarray,
    phi: np.ndarray,
    a: np.ndarray,
    group_ids: np.ndarray,
    num_groups: int,
    *,
    rho: float = 0.25,
    kappa: float = 1.0,
    global_monoid: np.ndarray | None = None,
    own_override: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure-NumPy exact FP32 additive monoid + T3 design-effect LOO fusion.

    Returns ``(out, local_monoid_table, per_series_monoid)``.
    """
    h_f = np.asarray(h, dtype=np.float32)
    phi_f = np.asarray(phi, dtype=np.float32)
    a_f = np.maximum(np.asarray(a, dtype=np.float32), np.float32(1e-6))
    gids = np.asarray(group_ids, dtype=np.int64)
    n_groups = int(num_groups)
    if n_groups < 1:
        raise ValueError(f"num_groups must be >= 1, got {n_groups}")

    B, D_f = phi_f.shape
    if gids.ndim != 1 or gids.shape[0] != B:
        raise ValueError(f"Expected 1D group_ids of length {B}, got {gids.shape}")
    if B > 0 and (np.any(gids < 0) or np.any(gids >= n_groups)):
        raise ValueError(
            f"group_ids out of bounds for num_groups={n_groups}: "
            f"min={int(np.min(gids))}, max={int(np.max(gids))}"
        )

    m_rows = np.empty((B, D_f + 2), dtype=np.float32)
    m_rows[:, :D_f] = a_f[:, None] * phi_f
    m_rows[:, D_f] = a_f
    m_rows[:, D_f + 1] = np.sqrt(a_f)

    local_table = np.zeros((n_groups, D_f + 2), dtype=np.float32)
    np.add.at(local_table, gids, m_rows)

    tot_table = (
        local_table if global_monoid is None else np.asarray(global_monoid, dtype=np.float32)
    )
    own_sub = m_rows if own_override is None else np.asarray(own_override, dtype=np.float32)

    peer = tot_table[gids] - own_sub
    s_phi_loo = peer[:, :D_f]
    a_prime = np.maximum(peer[:, D_f], np.float32(0.0))
    q_prime = np.maximum(peer[:, D_f + 1], np.float32(0.0))

    has_peers = a_prime > np.float32(1e-12)
    safe_a = np.where(has_peers, a_prime, np.float32(1.0))
    mu_loo = np.where(has_peers[:, None], s_phi_loo / safe_a[:, None], np.float32(0.0))
    n_eff = np.where(has_peers, (q_prime * q_prime) / safe_a, np.float32(0.0))
    a_eff = a_prime / (
        np.float32(1.0) + np.maximum(n_eff - np.float32(1.0), np.float32(0.0)) * np.float32(rho)
    )
    gamma = np.where(has_peers, a_eff / (a_eff + np.float32(kappa)), np.float32(0.0))

    out = h_f + gamma[:, None] * mu_loo
    return out, local_table, m_rows


class SeriesLaggedLOOTracker:
    """Series-ID-keyed lagged LOO tracker eliminating dynamic-batching subtraction hazards.

    Tracks ``(series_id, group_id) -> monoid_row`` from cycle ``t-1``. On cycle ``t``,
    a series subtracts its prior monoid contribution ONLY if the exact same
    ``series_id`` was present in the exact same ``group_id`` on cycle ``t-1``.
    Newly joined or regrouped series cleanly receive ``own_prev = 0``.

    Shard ownership contract:
    - Each ``SeriesLaggedLOOTracker`` owns the lagged history of a single rank's local
      shard, and ``series_ids`` MUST be unique within each cycle (duplicates are rejected
      with ``ValueError`` instead of causing silent ``searchsorted`` collisions).
    - When ``num_groups`` is configured, ``group_ids`` are validated to lie strictly in
      ``[0, num_groups)``.
    - When series migrate across ranks during shard rebalancing, call
      ``reset_for_migration(migrated_series_ids)`` (or ``reset_for_migration()`` for a
      full rebalance) so migrated series cleanly start with ``own_prev = 0`` on their
      new rank rather than reusing stale local state.
    """

    def __init__(
        self,
        monoid_dim: int,
        *,
        num_groups: int | None = None,
        device: Any = None,
    ) -> None:
        self.monoid_dim = int(monoid_dim)
        if num_groups is not None and int(num_groups) < 1:
            raise ValueError(f"num_groups must be >= 1, got {num_groups}")
        self.num_groups = None if num_groups is None else int(num_groups)
        self.device = device
        self._prev_sids: Any = None
        self._prev_gids: Any = None
        self._prev_rows: Any = None

    def reset(self) -> None:
        self._prev_sids = None
        self._prev_gids = None
        self._prev_rows = None

    def reset_for_migration(self, migrated_series_ids: Any | None = None) -> None:
        """Clear lagged LOO history for all series (``None``) or specific migrated IDs."""
        if migrated_series_ids is None or self._prev_sids is None:
            self.reset()
            return
        if torch is not None and isinstance(self._prev_sids, torch.Tensor):
            mig = torch.as_tensor(
                migrated_series_ids, dtype=torch.int64, device=self._prev_sids.device
            ).reshape(-1)
            if mig.numel() == 0 or self._prev_sids.numel() == 0:
                return
            keep = ~torch.isin(self._prev_sids, mig)
            self._prev_sids = self._prev_sids[keep].contiguous()
            self._prev_gids = self._prev_gids[keep].contiguous()
            self._prev_rows = self._prev_rows[keep].contiguous()
            return

        mig_np = np.asarray(migrated_series_ids, dtype=np.int64).reshape(-1)
        if mig_np.size == 0 or len(self._prev_sids) == 0:
            return
        keep_np = ~np.isin(self._prev_sids, mig_np)
        self._prev_sids = self._prev_sids[keep_np].copy()
        self._prev_gids = self._prev_gids[keep_np].copy()
        self._prev_rows = self._prev_rows[keep_np].copy()

    def _validate_torch_ids(
        self, sids: Any, gids: Any, *, check_unique: bool = True
    ) -> None:
        if sids.ndim != 1 or gids.ndim != 1 or sids.shape[0] != gids.shape[0]:
            raise ValueError(
                f"Expected matching 1D series_ids and group_ids, got "
                f"series_ids={tuple(sids.shape)}, group_ids={tuple(gids.shape)}"
            )
        B = int(sids.shape[0])
        if B == 0:
            return
        if self.num_groups is not None:
            out_of_bounds = bool(((gids < 0) | (gids >= self.num_groups)).any().item())
            if out_of_bounds:
                raise ValueError(
                    f"group_ids out of bounds for num_groups={self.num_groups}: "
                    f"min={int(gids.min().item())}, max={int(gids.max().item())}"
                )
        if check_unique and B > 1:
            sorted_sids = torch.sort(sids).values
            if bool((sorted_sids[1:] == sorted_sids[:-1]).any().item()):
                raise ValueError(
                    "Duplicate series_ids detected in SeriesLaggedLOOTracker; "
                    "series IDs must be unique within a single-rank shard cycle."
                )

    def _validate_numpy_ids(
        self, sids_np: np.ndarray, gids_np: np.ndarray, *, check_unique: bool = True
    ) -> None:
        if sids_np.ndim != 1 or gids_np.ndim != 1 or sids_np.shape[0] != gids_np.shape[0]:
            raise ValueError(
                f"Expected matching 1D series_ids and group_ids, got "
                f"series_ids={sids_np.shape}, group_ids={gids_np.shape}"
            )
        B = sids_np.shape[0]
        if B == 0:
            return
        if self.num_groups is not None:
            if np.any(gids_np < 0) or np.any(gids_np >= self.num_groups):
                raise ValueError(
                    f"group_ids out of bounds for num_groups={self.num_groups}: "
                    f"min={int(np.min(gids_np))}, max={int(np.max(gids_np))}"
                )
        if check_unique and B > 1:
            sorted_sids = np.sort(sids_np)
            if np.any(sorted_sids[1:] == sorted_sids[:-1]):
                raise ValueError(
                    "Duplicate series_ids detected in SeriesLaggedLOOTracker; "
                    "series IDs must be unique within a single-rank shard cycle."
                )

    def lookup_own_prev(
        self,
        series_ids: Any,
        group_ids: Any,
        *,
        validate: bool = True,
    ) -> Any:
        """Return lagged ``own_prev`` rows aligned to ``(series_ids, group_ids)``."""
        if torch is not None and isinstance(series_ids, torch.Tensor):
            dev = series_ids.device
            sids = series_ids.to(dtype=torch.int64, device=dev)
            gids = torch.as_tensor(group_ids, dtype=torch.int64, device=dev)
            if validate:
                self._validate_torch_ids(sids, gids, check_unique=True)
            elif sids.ndim != 1 or gids.ndim != 1 or sids.shape[0] != gids.shape[0]:
                raise ValueError(
                    f"Expected matching 1D series_ids and group_ids, got "
                    f"series_ids={tuple(sids.shape)}, group_ids={tuple(gids.shape)}"
                )
            B = int(sids.shape[0])
            if self._prev_sids is None or self._prev_sids.numel() == 0 or B == 0:
                return torch.zeros((B, self.monoid_dim), dtype=torch.float32, device=dev)

            prev_s = self._prev_sids.to(device=dev)
            prev_g = self._prev_gids.to(device=dev)
            prev_r = self._prev_rows.to(device=dev)

            pos = torch.searchsorted(prev_s, sids)
            prev_n = prev_s.shape[0]
            in_bounds = pos < prev_n
            safe_pos = pos.clamp(max=prev_n - 1)
            matched = in_bounds & (prev_s[safe_pos] == sids) & (prev_g[safe_pos] == gids)
            gathered = prev_r[safe_pos]
            return torch.where(matched.unsqueeze(-1), gathered, torch.zeros_like(gathered))

        sids_np = np.asarray(series_ids, dtype=np.int64)
        gids_np = np.asarray(group_ids, dtype=np.int64)
        if validate:
            self._validate_numpy_ids(sids_np, gids_np, check_unique=True)
        B = sids_np.shape[0]
        if self._prev_sids is None or len(self._prev_sids) == 0 or B == 0:
            return np.zeros((B, self.monoid_dim), dtype=np.float32)

        pos = np.searchsorted(self._prev_sids, sids_np)
        prev_n = len(self._prev_sids)
        in_bounds = pos < prev_n
        safe_pos = np.minimum(pos, prev_n - 1)
        matched = (
            in_bounds
            & (self._prev_sids[safe_pos] == sids_np)
            & (self._prev_gids[safe_pos] == gids_np)
        )
        gathered = self._prev_rows[safe_pos]
        return np.where(matched[:, None], gathered, np.float32(0.0)).astype(np.float32)

    def update(
        self,
        series_ids: Any,
        group_ids: Any,
        monoid_rows: Any,
        *,
        validate: bool = True,
    ) -> None:
        """Record sorted ``(series_ids, group_ids, monoid_rows)`` snapshot for cycle ``t+1``."""
        if torch is not None and isinstance(series_ids, torch.Tensor):
            dev = series_ids.device
            sids = series_ids.detach().to(dtype=torch.int64, device=dev)
            gids = torch.as_tensor(group_ids, dtype=torch.int64, device=dev).detach()
            rows = torch.as_tensor(monoid_rows, dtype=torch.float32, device=dev).detach()
            if validate:
                self._validate_torch_ids(sids, gids, check_unique=False)
            elif sids.ndim != 1 or gids.ndim != 1 or sids.shape[0] != gids.shape[0]:
                raise ValueError(
                    f"Expected matching 1D series_ids and group_ids, got "
                    f"series_ids={tuple(sids.shape)}, group_ids={tuple(gids.shape)}"
                )
            if rows.ndim != 2 or rows.shape != (sids.shape[0], self.monoid_dim):
                raise ValueError(
                    f"Expected monoid_rows shape ({sids.shape[0]}, {self.monoid_dim}), "
                    f"got {tuple(rows.shape)}"
                )
            order = torch.argsort(sids)
            sorted_sids = sids[order].contiguous()
            if validate and sorted_sids.numel() > 1:
                if bool((sorted_sids[1:] == sorted_sids[:-1]).any().item()):
                    raise ValueError(
                        "Duplicate series_ids detected in SeriesLaggedLOOTracker.update; "
                        "series IDs must be unique within a single-rank shard cycle."
                    )
            self._prev_sids = sorted_sids
            self._prev_gids = gids[order].contiguous()
            self._prev_rows = rows[order].contiguous()
            return

        sids_np = np.asarray(series_ids, dtype=np.int64)
        gids_np = np.asarray(group_ids, dtype=np.int64)
        rows_np = np.asarray(monoid_rows, dtype=np.float32)
        if validate:
            self._validate_numpy_ids(sids_np, gids_np, check_unique=False)
        elif sids_np.ndim != 1 or gids_np.ndim != 1 or sids_np.shape[0] != gids_np.shape[0]:
            raise ValueError(
                f"Expected matching 1D series_ids and group_ids, got "
                f"series_ids={sids_np.shape}, group_ids={gids_np.shape}"
            )
        if rows_np.ndim != 2 or rows_np.shape != (sids_np.shape[0], self.monoid_dim):
            raise ValueError(
                f"Expected monoid_rows shape ({sids_np.shape[0]}, {self.monoid_dim}), "
                f"got {rows_np.shape}"
            )
        order = np.argsort(sids_np)
        sorted_sids_np = sids_np[order].copy()
        if (
            validate
            and sorted_sids_np.size > 1
            and np.any(sorted_sids_np[1:] == sorted_sids_np[:-1])
        ):
            raise ValueError(
                "Duplicate series_ids detected in SeriesLaggedLOOTracker.update; "
                "series IDs must be unique within a single-rank shard cycle."
            )
        self._prev_sids = sorted_sids_np
        self._prev_gids = gids_np[order].copy()
        self._prev_rows = rows_np[order].copy()


# ---------------------------------------------------------------------------
# PyTorch AsyncFuseMonoid, 3-Graph Ping-Pong AsyncFuseRunner, PipeFuseRunner
# ---------------------------------------------------------------------------

_BaseModule = nn.Module if nn is not None else object


class AsyncFuseMonoid(_BaseModule):  # type: ignore[misc]
    """Drop-in PyTorch cross-series fusion layer using an exact FP32 additive monoid.

    Monoid element per series ``b``:
        ``m_b = [a_b * phi_b, a_b, sqrt(a_b)]``  (width ``D_f + 2`` in FP32).
    """

    def __init__(
        self,
        embed_dim: int,
        fuse_dim: int | None = None,
        *,
        num_groups: int = 32,
        rho: float = 0.25,
        kappa: float = 1.0,
    ) -> None:
        if torch is None:
            raise RuntimeError("AsyncFuseMonoid requires PyTorch.")
        super().__init__()
        if int(num_groups) < 1:
            raise ValueError(f"num_groups must be >= 1, got {num_groups}")
        self.embed_dim = int(embed_dim)
        self.fuse_dim = int(embed_dim if fuse_dim is None else fuse_dim)
        self.num_groups = int(num_groups)
        self.monoid_dim = self.fuse_dim + 2
        self.rho = float(rho)
        self.kappa = float(kappa)

        self.gate_proj = nn.Linear(self.embed_dim, 1)
        self.value_proj = nn.Linear(self.embed_dim, self.fuse_dim)
        self.out_proj = nn.Linear(self.fuse_dim, self.embed_dim, bias=False)

    @property
    def payload_bytes(self) -> int:
        """Exact FP32 AllReduce buffer size in bytes."""
        return self.num_groups * self.monoid_dim * 4

    def compute_monoid_rows(self, h: Any) -> Any:
        """Compute per-series FP32 monoid rows ``m_b = [a_b * phi_b, a_b, sqrt(a_b)]``."""
        h_f32 = h.to(torch.float32)
        a = F.softplus(self.gate_proj(h_f32)).squeeze(-1).clamp(min=1e-6, max=1e4)
        phi = self.value_proj(h_f32)
        a_phi = a.unsqueeze(-1) * phi
        sqrt_a = torch.sqrt(a).unsqueeze(-1)
        return torch.cat([a_phi, a.unsqueeze(-1), sqrt_a], dim=-1)

    def scatter_local_monoid(
        self, m_rows: Any, group_ids: Any, *, out_buf: Any | None = None
    ) -> Any:
        """Accumulate local per-group FP32 additive monoid table ``(G, D_f + 2)``."""
        if out_buf is None:
            tot = torch.zeros(
                (self.num_groups, self.monoid_dim),
                dtype=torch.float32,
                device=m_rows.device,
            )
        else:
            tot = out_buf
            tot.zero_()
        gids = group_ids.to(dtype=torch.int64, device=m_rows.device)
        tot.index_add_(0, gids, m_rows.to(torch.float32))
        return tot

    def apply_loo_fusion(
        self,
        h: Any,
        group_ids: Any,
        global_monoid: Any,
        own_monoid: Any,
    ) -> Any:
        """Apply leave-one-out T3 design-effect shrinkage and residual fusion."""
        gids = group_ids.to(dtype=torch.int64, device=h.device)
        peer = global_monoid.index_select(0, gids) - own_monoid.to(torch.float32)
        s_phi_loo = peer[:, : self.fuse_dim]
        a_prime = peer[:, self.fuse_dim].clamp(min=0.0)
        q_prime = peer[:, self.fuse_dim + 1].clamp(min=0.0)

        has_peers = a_prime > 1e-12
        safe_a = torch.where(has_peers, a_prime, torch.ones_like(a_prime))
        mu_loo = torch.where(
            has_peers.unsqueeze(-1), s_phi_loo / safe_a.unsqueeze(-1), torch.zeros_like(s_phi_loo)
        )
        n_eff = torch.where(has_peers, (q_prime * q_prime) / safe_a, torch.zeros_like(a_prime))
        a_eff = a_prime / (1.0 + (n_eff - 1.0).clamp(min=0.0) * self.rho)
        gamma = torch.where(has_peers, a_eff / (a_eff + self.kappa), torch.zeros_like(a_prime))

        delta = gamma.unsqueeze(-1) * self.out_proj(mu_loo)
        return h + delta.to(h.dtype)

    def forward(
        self,
        h: Any,
        group_ids: Any,
        *,
        global_monoid: Any | None = None,
        own_monoid: Any | None = None,
    ) -> tuple[Any, Any, Any]:
        """Central or externally-reduced exact forward pass.

        Returns ``(h_fused, local_monoid_table, m_rows)``.
        """
        is_cap_fn = getattr(torch.cuda, "is_current_stream_capturing", None)
        capturing = bool(h.is_cuda and callable(is_cap_fn) and is_cap_fn())
        if not capturing and group_ids.numel() > 0:
            g_chk = group_ids.to(dtype=torch.int64)
            if bool(((g_chk < 0) | (g_chk >= self.num_groups)).any().item()):
                raise ValueError(f"group_ids out of bounds for num_groups={self.num_groups}")
        m_rows = self.compute_monoid_rows(h)
        local_tot = self.scatter_local_monoid(m_rows, group_ids)
        tot = local_tot if global_monoid is None else global_monoid
        own = m_rows if own_monoid is None else own_monoid
        out = self.apply_loo_fusion(h, group_ids, tot, own)
        return out, local_tot, m_rows


def _dist_active(process_group: Any = None) -> bool:
    return bool(
        dist is not None
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size(process_group) > 1
    )


class AsyncFuseRunner:
    """3-Graph Ping-Pong Overlapped NCCL ``AsyncFuse`` Runner.

    Uses 2 ping-pong buffer slots and a high-priority side ``comm_stream``
    (``priority=-1``) to overlap a single concatenated FP32 ``all_reduce(SUM)``
    (~11-22 KB) behind ``tail(t-1) + pre(t)``, with series-ID-keyed lagged LOO.
    """

    def __init__(
        self,
        pre_fn: Callable[[Any], Any],
        monoid: AsyncFuseMonoid,
        tail_fn: Callable[[Any], Any],
        *,
        batch_size: int,
        input_shape: tuple[int, ...],
        device: Any = None,
        process_group: Any = None,
        use_cuda_graphs: bool = True,
    ) -> None:
        if torch is None:
            raise RuntimeError("AsyncFuseRunner requires PyTorch.")
        self.pre_fn = pre_fn
        self.monoid = monoid
        self.tail_fn = tail_fn
        self.batch_size = int(batch_size)
        self.input_shape = tuple(int(s) for s in input_shape)
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.process_group = process_group
        self.use_cuda_graphs = bool(use_cuda_graphs and self.device.type == "cuda")

        self.tracker = SeriesLaggedLOOTracker(
            monoid.monoid_dim, num_groups=monoid.num_groups, device=self.device
        )
        self._step = 0
        self._captured = False
        self._last_exec_mode: str | None = None
        self._pool: Any = None
        self._comm_stream: Any = None
        self._ev_pre_done: list[Any] = []
        self._ev_comm_done: list[Any] = []
        self._ev_comm_pending: list[bool] = [False, False]
        self._inflight_work: list[Any] = [None, None]

        # Ping-pong static buffers (2 slots)
        self._x_bufs: list[Any] = []
        self._gids_bufs: list[Any] = []
        self._own_prev_bufs: list[Any] = []
        self._h_bufs: list[Any] = []
        self._curr_m_bufs: list[Any] = []
        self._tot_bufs: list[Any] = []
        self._h_fused_bufs: list[Any] = []
        self._y_bufs: list[Any] = []

        # 3 captured CUDA Graphs per slot: pre, fuse, tail
        self._g_pre: list[Any] = []
        self._g_fuse: list[Any] = []
        self._g_tail: list[Any] = []

        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                self._comm_stream = torch.cuda.Stream(device=self.device, priority=-1)
                self._ev_pre_done = [torch.cuda.Event(), torch.cuda.Event()]
                self._ev_comm_done = [torch.cuda.Event(), torch.cuda.Event()]
                self._tot_bufs = [
                    torch.zeros(
                        (monoid.num_groups, monoid.monoid_dim),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    for _ in range(2)
                ]
        else:
            self._tot_bufs = [
                torch.zeros(
                    (monoid.num_groups, monoid.monoid_dim), dtype=torch.float32, device=self.device
                )
                for _ in range(2)
            ]

    def _drain_slot_comm(self, slot: int, compute_stream: Any | None = None) -> None:
        """Wait for any in-flight NCCL work and pending CUDA comm event on ``slot``."""
        if self._inflight_work[slot] is not None:
            self._inflight_work[slot].wait()
            self._inflight_work[slot] = None
        if self.device.type == "cuda" and self._ev_comm_pending[slot]:
            stream = (
                compute_stream
                if compute_stream is not None
                else torch.cuda.current_stream(self.device)
            )
            stream.wait_event(self._ev_comm_done[slot])
            self._ev_comm_pending[slot] = False

    def _clear_comm_events(self) -> None:
        """Drain all slots and recreate fresh comm events (safe across graphed <-> eager switch)."""
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                compute_stream = torch.cuda.current_stream(self.device)
                for slot in range(2):
                    self._drain_slot_comm(slot, compute_stream)
                self._ev_pre_done = [torch.cuda.Event(), torch.cuda.Event()]
                self._ev_comm_done = [torch.cuda.Event(), torch.cuda.Event()]
                self._ev_comm_pending = [False, False]
        else:
            for slot in range(2):
                self._drain_slot_comm(slot)

    def flush(self) -> None:
        """Drain all pending asynchronous NCCL all-reduce handles across both ping-pong slots."""
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                compute_stream = torch.cuda.current_stream(self.device)
                for slot in range(2):
                    self._drain_slot_comm(slot, compute_stream)
        else:
            for slot in range(2):
                self._drain_slot_comm(slot)

    def reset_state(self) -> None:
        """Reset ping-pong monoid buffers, CUDA comm events, and series-ID LOO tracker."""
        self.tracker.reset()
        self._step = 0
        self._last_exec_mode = None
        self._clear_comm_events()
        for slot in range(2):
            self._tot_bufs[slot].zero_()

    def reset_for_migration(self, migrated_series_ids: Any | None = None) -> None:
        """Handle shard rebalancing / series migration across ranks cleanly."""
        if migrated_series_ids is None:
            self.reset_state()
        else:
            self.tracker.reset_for_migration(migrated_series_ids)

    def capture_graphs(self) -> None:
        """Capture the 3 ping-pong CUDA Graphs (pre, fuse, tail) inside ``torch.cuda.device``."""
        if not self.use_cuda_graphs or self._captured:
            return

        with torch.cuda.device(self.device), torch.inference_mode():
            self._pool = torch.cuda.graph_pool_handle()
            side = torch.cuda.Stream(device=self.device)
            side.wait_stream(torch.cuda.current_stream(self.device))

            self._x_bufs = []
            self._gids_bufs = []
            self._own_prev_bufs = []
            self._h_bufs = []
            self._curr_m_bufs = []
            self._h_fused_bufs = []
            self._y_bufs = []
            self._g_pre = []
            self._g_fuse = []
            self._g_tail = []

            for _slot in range(2):
                x_b = torch.sin(
                    torch.linspace(
                        0.1,
                        5.0,
                        self.batch_size * int(np.prod(self.input_shape)),
                        device=self.device,
                    )
                ).reshape(self.batch_size, *self.input_shape)
                g_b = (
                    torch.arange(self.batch_size, dtype=torch.int64, device=self.device)
                    % self.monoid.num_groups
                )
                own_b = torch.zeros(
                    (self.batch_size, self.monoid.monoid_dim),
                    dtype=torch.float32,
                    device=self.device,
                )

                self._x_bufs.append(x_b)
                self._gids_bufs.append(g_b)
                self._own_prev_bufs.append(own_b)

            # Warmup 3 iterations on side stream before capture.
            with torch.cuda.stream(side):
                for _ in range(3):
                    for slot in range(2):
                        h_tmp = self.pre_fn(self._x_bufs[slot])
                        m_tmp = self.monoid.compute_monoid_rows(h_tmp)
                        self.monoid.scatter_local_monoid(
                            m_tmp, self._gids_bufs[slot], out_buf=self._tot_bufs[slot]
                        )
                        hf_tmp = self.monoid.apply_loo_fusion(
                            h_tmp,
                            self._gids_bufs[slot],
                            self._tot_bufs[slot ^ 1],
                            self._own_prev_bufs[slot],
                        )
                        _ = self.tail_fn(hf_tmp)
            torch.cuda.current_stream(self.device).wait_stream(side)
            torch.cuda.synchronize(self.device)

            # Capture 3 graphs per ping-pong slot sharing one pool handle.
            for slot in range(2):
                prev_slot = slot ^ 1

                g_pre = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g_pre, pool=self._pool, stream=side):
                    h_out = self.pre_fn(self._x_bufs[slot])
                    m_out = self.monoid.compute_monoid_rows(h_out)
                    self.monoid.scatter_local_monoid(
                        m_out, self._gids_bufs[slot], out_buf=self._tot_bufs[slot]
                    )
                self._h_bufs.append(h_out)
                self._curr_m_bufs.append(m_out)
                self._g_pre.append(g_pre)

                g_fuse = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g_fuse, pool=self._pool, stream=side):
                    hf_out = self.monoid.apply_loo_fusion(
                        self._h_bufs[slot],
                        self._gids_bufs[slot],
                        self._tot_bufs[prev_slot],
                        self._own_prev_bufs[slot],
                    )
                self._h_fused_bufs.append(hf_out)
                self._g_fuse.append(g_fuse)

                g_tail = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g_tail, pool=self._pool, stream=side):
                    y_out = self.tail_fn(self._h_fused_bufs[slot])
                self._y_bufs.append(y_out)
                self._g_tail.append(g_tail)

            for slot in range(2):
                self._tot_bufs[slot].zero_()
            self._captured = True

    def step(
        self,
        x: Any,
        group_ids: Any,
        series_ids: Any | None = None,
        *,
        validate_ids: bool = True,
    ) -> Any:
        """Execute one overlapped AsyncFuse cycle with series-ID-keyed lagged LOO."""
        B = int(x.shape[0])
        gids_dev = group_ids.to(dtype=torch.int64, device=self.device, non_blocking=True)
        if series_ids is None:
            sids_dev = torch.arange(B, dtype=torch.int64, device=self.device)
        else:
            sids_dev = series_ids.to(dtype=torch.int64, device=self.device, non_blocking=True)

        slot = self._step & 1
        prev_slot = slot ^ 1
        has_prev = self._step > 0
        # Validate once on entry (unique series_ids & group_ids in [0, num_groups));
        # skip duplicate validation in update() since sids_dev/gids_dev are unchanged.
        own_prev = self.tracker.lookup_own_prev(sids_dev, gids_dev, validate=validate_ids)

        exec_mode = "graphed" if (self.use_cuda_graphs and B == self.batch_size) else "eager"
        if self._last_exec_mode is not None and self._last_exec_mode != exec_mode:
            # Drain and reset comm events cleanly when transitioning graphed <-> eager.
            self._clear_comm_events()
        self._last_exec_mode = exec_mode

        if exec_mode == "graphed":
            if not self._captured:
                self.capture_graphs()

            with torch.cuda.device(self.device), torch.inference_mode():
                compute_stream = torch.cuda.current_stream(self.device)
                # Ensure prior all_reduce on this slot finished before overwriting tot_bufs[slot].
                self._drain_slot_comm(slot, compute_stream)

                self._x_bufs[slot].copy_(x)
                self._gids_bufs[slot].copy_(gids_dev)
                self._own_prev_bufs[slot].copy_(own_prev)

                # Graph 1: prefix backbone + local monoid scatter-add into tot_bufs[slot].
                self._g_pre[slot].replay()
                self._ev_pre_done[slot].record(compute_stream)

                # Launch async single-buffer FP32 AllReduce on high-priority side comm_stream.
                with torch.cuda.stream(self._comm_stream):
                    self._comm_stream.wait_event(self._ev_pre_done[slot])
                    if _dist_active(self.process_group):
                        self._inflight_work[slot] = dist.all_reduce(
                            self._tot_bufs[slot],
                            op=dist.ReduceOp.SUM,
                            group=self.process_group,
                            async_op=True,
                        )
                    self._ev_comm_done[slot].record(self._comm_stream)
                    self._ev_comm_pending[slot] = True

                # Wait for prev_slot's global monoid (already overlapped!) and run Graphs 2 + 3.
                if has_prev:
                    self._drain_slot_comm(prev_slot, compute_stream)
                else:
                    self._tot_bufs[prev_slot].zero_()

                self._g_fuse[slot].replay()
                self._g_tail[slot].replay()

                self.tracker.update(sids_dev, gids_dev, self._curr_m_bufs[slot], validate=False)
                self._step += 1
                return self._y_bufs[slot].clone()

        # Eager overlapped fallback (arbitrary dynamic batch size or CPU).
        ctx_dev = torch.cuda.device(self.device) if self.device.type == "cuda" else None
        if ctx_dev is not None:
            ctx_dev.__enter__()
        try:
            with torch.inference_mode():
                compute_stream = (
                    torch.cuda.current_stream(self.device) if self.device.type == "cuda" else None
                )
                x_dev = (
                    x if x.device == self.device else x.to(device=self.device, non_blocking=True)
                )
                h = self.pre_fn(x_dev)
                curr_m = self.monoid.compute_monoid_rows(h)
                self._drain_slot_comm(slot, compute_stream)
                self.monoid.scatter_local_monoid(curr_m, gids_dev, out_buf=self._tot_bufs[slot])

                if self.device.type == "cuda" and self._comm_stream is not None:
                    self._ev_pre_done[slot].record(compute_stream)
                    with torch.cuda.stream(self._comm_stream):
                        self._comm_stream.wait_event(self._ev_pre_done[slot])
                        if _dist_active(self.process_group):
                            self._inflight_work[slot] = dist.all_reduce(
                                self._tot_bufs[slot],
                                op=dist.ReduceOp.SUM,
                                group=self.process_group,
                                async_op=True,
                            )
                        self._ev_comm_done[slot].record(self._comm_stream)
                        self._ev_comm_pending[slot] = True
                elif _dist_active(self.process_group):
                    self._inflight_work[slot] = dist.all_reduce(
                        self._tot_bufs[slot],
                        op=dist.ReduceOp.SUM,
                        group=self.process_group,
                        async_op=True,
                    )

                if has_prev:
                    self._drain_slot_comm(prev_slot, compute_stream)
                    global_prev = self._tot_bufs[prev_slot]
                else:
                    global_prev = torch.zeros_like(self._tot_bufs[prev_slot])

                h_fused = self.monoid.apply_loo_fusion(h, gids_dev, global_prev, own_prev)
                y = self.tail_fn(h_fused)
                self.tracker.update(sids_dev, gids_dev, curr_m, validate=False)
                self._step += 1
                return y
        finally:
            if ctx_dev is not None:
                ctx_dev.__exit__(None, None, None)


class PipeFuseRunner:
    """Exact zero-lag double-buffered panel-split fusion runner (``G >= 2``).

    Splits groups into two panels ``A`` (``g < G // 2``) and ``B`` (``g >= G // 2``),
    overlapping ``all_reduce(Panel A)`` with ``pre(Panel B)`` and ``all_reduce(Panel B)``
    with ``fuse_tail(Panel A)`` for exact <= 1.2e-7 central parity at zero lag.
    """

    def __init__(
        self,
        pre_fn: Callable[[Any], Any],
        monoid: AsyncFuseMonoid,
        tail_fn: Callable[[Any], Any],
        *,
        device: Any = None,
        process_group: Any = None,
    ) -> None:
        if torch is None:
            raise RuntimeError("PipeFuseRunner requires PyTorch.")
        if monoid.num_groups < 2:
            raise ValueError("PipeFuseRunner requires num_groups >= 2 for panel splitting.")
        self.pre_fn = pre_fn
        self.monoid = monoid
        self.tail_fn = tail_fn
        self.split_group = monoid.num_groups // 2
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.process_group = process_group
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                self._comm_stream = torch.cuda.Stream(device=self.device, priority=-1)
        else:
            self._comm_stream = None

    def _run_panel_pre(
        self, x_p: Any, g_p: Any, local_offset: int, panel_groups: int
    ) -> tuple[Any, Any, Any]:
        h_p = (
            self.pre_fn(x_p)
            if x_p.shape[0] > 0
            else torch.empty((0, self.monoid.embed_dim), dtype=torch.float32, device=self.device)
        )
        m_p = (
            self.monoid.compute_monoid_rows(h_p)
            if x_p.shape[0] > 0
            else torch.empty((0, self.monoid.monoid_dim), dtype=torch.float32, device=self.device)
        )
        tot_p = torch.zeros(
            (panel_groups, self.monoid.monoid_dim), dtype=torch.float32, device=self.device
        )
        if x_p.shape[0] > 0:
            tot_p.index_add_(0, g_p - local_offset, m_p.to(torch.float32))
        return h_p, m_p, tot_p

    def step(self, x: Any, group_ids: Any) -> Any:
        """Execute one zero-lag double-buffered PipeFuse step."""
        ctx_dev = torch.cuda.device(self.device) if self.device.type == "cuda" else None
        if ctx_dev is not None:
            ctx_dev.__enter__()
        try:
            with torch.inference_mode():
                gids = group_ids.to(dtype=torch.int64, device=self.device)
                if gids.numel() > 0 and bool(
                    ((gids < 0) | (gids >= self.monoid.num_groups)).any().item()
                ):
                    raise ValueError(
                        f"group_ids out of bounds for num_groups={self.monoid.num_groups}"
                    )
                mask_a = gids < self.split_group
                idx_a = torch.nonzero(mask_a, as_tuple=False).squeeze(-1)
                idx_b = torch.nonzero(~mask_a, as_tuple=False).squeeze(-1)

                g_a_count = self.split_group
                g_b_count = self.monoid.num_groups - self.split_group

                # 1. Pre(Panel A)
                h_a, m_a, tot_a = self._run_panel_pre(
                    x.index_select(0, idx_a), gids.index_select(0, idx_a), 0, g_a_count
                )

                # 2. Launch async AllReduce(Panel A) on side comm_stream
                work_a = None
                ev_a_done = None
                if _dist_active(self.process_group):
                    if self._comm_stream is not None:
                        ev_a = torch.cuda.Event()
                        ev_a.record(torch.cuda.current_stream(self.device))
                        with torch.cuda.stream(self._comm_stream):
                            self._comm_stream.wait_event(ev_a)
                            work_a = dist.all_reduce(
                                tot_a, op=dist.ReduceOp.SUM, group=self.process_group, async_op=True
                            )
                            ev_a_done = torch.cuda.Event()
                            ev_a_done.record(self._comm_stream)
                    else:
                        work_a = dist.all_reduce(
                            tot_a, op=dist.ReduceOp.SUM, group=self.process_group, async_op=True
                        )

                # 3. Overlapped Pre(Panel B)
                h_b, m_b, tot_b = self._run_panel_pre(
                    x.index_select(0, idx_b),
                    gids.index_select(0, idx_b),
                    self.split_group,
                    g_b_count,
                )

                # 4. Launch async AllReduce(Panel B)
                work_b = None
                ev_b_done = None
                if _dist_active(self.process_group):
                    if self._comm_stream is not None:
                        ev_b = torch.cuda.Event()
                        ev_b.record(torch.cuda.current_stream(self.device))
                        with torch.cuda.stream(self._comm_stream):
                            self._comm_stream.wait_event(ev_b)
                            work_b = dist.all_reduce(
                                tot_b, op=dist.ReduceOp.SUM, group=self.process_group, async_op=True
                            )
                            ev_b_done = torch.cuda.Event()
                            ev_b_done.record(self._comm_stream)
                    else:
                        work_b = dist.all_reduce(
                            tot_b, op=dist.ReduceOp.SUM, group=self.process_group, async_op=True
                        )

                # 5. Finish Panel A fusion + tail while Panel B reduces (wait ONLY Panel A!)
                if work_a is not None:
                    work_a.wait()
                if ev_a_done is not None:
                    torch.cuda.current_stream(self.device).wait_event(ev_a_done)

                y_a = None
                if idx_a.numel() > 0:
                    hf_a = self.monoid.apply_loo_fusion(
                        h_a, gids.index_select(0, idx_a), tot_a, m_a
                    )
                    y_a = self.tail_fn(hf_a)

                # 6. Finish Panel B fusion + tail
                if work_b is not None:
                    work_b.wait()
                if ev_b_done is not None:
                    torch.cuda.current_stream(self.device).wait_event(ev_b_done)

                y_b = None
                if idx_b.numel() > 0:
                    hf_b = self.monoid.apply_loo_fusion(
                        h_b, gids.index_select(0, idx_b) - self.split_group, tot_b, m_b
                    )
                    y_b = self.tail_fn(hf_b)

                # 7. Reassemble in original batch order (handles empty local rank cleanly!)
                ref_y = y_a if y_a is not None else y_b
                if ref_y is None:
                    empty_h = torch.empty(
                        (0, self.monoid.embed_dim), dtype=x.dtype, device=self.device
                    )
                    return self.tail_fn(empty_h)
                out = torch.empty(
                    (x.shape[0], *ref_y.shape[1:]), dtype=ref_y.dtype, device=self.device
                )
                if y_a is not None:
                    out.index_copy_(0, idx_a, y_a)
                if y_b is not None:
                    out.index_copy_(0, idx_b, y_b)
                return out
        finally:
            if ctx_dev is not None:
                ctx_dev.__exit__(None, None, None)


@dataclass(frozen=True)
class StreamCoordinationState:
    """Global O(1) cross-GPU conformal quantile & swarm e-value summary."""

    miscov_rate_80: np.ndarray
    miscov_rate_90: np.ndarray
    total_ticks: np.ndarray
    swarm_mean_log_e: np.ndarray
    swarm_wealth: np.ndarray
    payload_bytes: int


class DistributedStreamCoordinator:
    """O(1) ``< 2 KB`` async NCCL monoid coordinator for conformal quantiles & swarm e-values.

    Maintains a compact FP32 additive monoid of 6 scalars per group:
        ``[miscov_80, miscov_90, tick_count, sum_log_e, sum_e_wealth, sum_sq_err]``
    enforcing total wire payload strictly ``< 2048`` bytes across all ranks.
    """

    _FIELDS_PER_GROUP = 6
    MAX_PAYLOAD_BYTES = 2048

    def __init__(
        self,
        num_groups: int = 16,
        *,
        device: Any = None,
        process_group: Any = None,
    ) -> None:
        self.num_groups = int(num_groups)
        self.payload_bytes = self.num_groups * self._FIELDS_PER_GROUP * 4
        if self.payload_bytes >= self.MAX_PAYLOAD_BYTES:
            max_g = self.MAX_PAYLOAD_BYTES // (self._FIELDS_PER_GROUP * 4) - 1
            raise ValueError(
                f"DistributedStreamCoordinator payload {self.payload_bytes} B must be < "
                f"{self.MAX_PAYLOAD_BYTES} B (max {max_g} groups)."
            )
        self.process_group = process_group
        self.device = (
            torch.device(
                device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
            )
            if torch is not None
            else None
        )
        if torch is not None and self.device is not None and self.device.type == "cuda":
            with torch.cuda.device(self.device):
                self._comm_stream = torch.cuda.Stream(device=self.device, priority=-1)
        else:
            self._comm_stream = None
        self._pending_work: Any = None
        self._last_buf: Any = (
            torch.zeros(
                (self.num_groups, self._FIELDS_PER_GROUP), dtype=torch.float32, device=self.device
            )
            if torch is not None
            else np.zeros((self.num_groups, self._FIELDS_PER_GROUP), dtype=np.float32)
        )

    def submit_async(
        self,
        miscov_80: Any,
        miscov_90: Any,
        tick_count: Any,
        sum_log_e: Any,
        sum_e_wealth: Any,
        sum_sq_err: Any,
    ) -> None:
        """Pack local conformal + swarm e-value monoid (< 2 KB) and launch async AllReduce."""
        if torch is not None and self.device is not None:
            ctx_dev = torch.cuda.device(self.device) if self.device.type == "cuda" else None
            if ctx_dev is not None:
                ctx_dev.__enter__()
            try:
                if self._pending_work is not None:
                    self._pending_work.wait()
                    self._pending_work = None
                stacked = torch.stack(
                    [
                        torch.as_tensor(miscov_80, dtype=torch.float32, device=self.device),
                        torch.as_tensor(miscov_90, dtype=torch.float32, device=self.device),
                        torch.as_tensor(tick_count, dtype=torch.float32, device=self.device),
                        torch.as_tensor(sum_log_e, dtype=torch.float32, device=self.device),
                        torch.as_tensor(sum_e_wealth, dtype=torch.float32, device=self.device),
                        torch.as_tensor(sum_sq_err, dtype=torch.float32, device=self.device),
                    ],
                    dim=-1,
                ).contiguous()
                self._last_buf.copy_(stacked)
                if _dist_active(self.process_group):
                    if self._comm_stream is not None:
                        ev_copy = torch.cuda.Event()
                        ev_copy.record(torch.cuda.current_stream(self.device))
                        with torch.cuda.stream(self._comm_stream):
                            self._comm_stream.wait_event(ev_copy)
                            self._pending_work = dist.all_reduce(
                                self._last_buf,
                                op=dist.ReduceOp.SUM,
                                group=self.process_group,
                                async_op=True,
                            )
                    else:
                        self._pending_work = dist.all_reduce(
                            self._last_buf,
                            op=dist.ReduceOp.SUM,
                            group=self.process_group,
                            async_op=True,
                        )
            finally:
                if ctx_dev is not None:
                    ctx_dev.__exit__(None, None, None)
            return

        self._last_buf = np.stack(
            [
                np.asarray(miscov_80, dtype=np.float32),
                np.asarray(miscov_90, dtype=np.float32),
                np.asarray(tick_count, dtype=np.float32),
                np.asarray(sum_log_e, dtype=np.float32),
                np.asarray(sum_e_wealth, dtype=np.float32),
                np.asarray(sum_sq_err, dtype=np.float32),
            ],
            axis=-1,
        )

    def poll_state(self) -> StreamCoordinationState:
        """Finalize pending async sync and return global conformal & e-value statistics."""
        if torch is not None and self.device is not None:
            ctx_dev = torch.cuda.device(self.device) if self.device.type == "cuda" else None
            if ctx_dev is not None:
                ctx_dev.__enter__()
            try:
                if self._pending_work is not None:
                    self._pending_work.wait()
                    self._pending_work = None
                arr = self._last_buf.detach().cpu().numpy()
            finally:
                if ctx_dev is not None:
                    ctx_dev.__exit__(None, None, None)
        else:
            arr = np.asarray(self._last_buf, dtype=np.float32)
        counts = np.maximum(arr[:, 2], 1.0)
        return StreamCoordinationState(
            miscov_rate_80=arr[:, 0] / counts,
            miscov_rate_90=arr[:, 1] / counts,
            total_ticks=arr[:, 2],
            swarm_mean_log_e=arr[:, 3] / counts,
            swarm_wealth=arr[:, 4],
            payload_bytes=self.payload_bytes,
        )


distributed_stream = DistributedStreamCoordinator
