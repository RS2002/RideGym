"""Assignment-Net acting path: structured encoding -> Q-matrix -> matching.

Mirrors :mod:`iddqn.inference` but uses :class:`AssignmentNet`, whose forward
produces the whole [N, M] Q-matrix in one matrix product. The Q-matrix builder
:func:`assignment_q_matrix` is shared by acting and by the trainer's target
step (no_grad), so the two never drift.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

from iddqn.features import FeatureEncoder
from iddqn.assignment_net import AssignmentNet
from iddqn.matching import match_drivers_to_orders, NEG_INF
from iddqn.replay import StateView
from iddqn.encode import encode_state
from iddqn.exploration import QNoiseExplorer
from benchmark.spatial import GridIndex

Coord = Tuple[float, float]


def assignment_q_matrix(
    net: AssignmentNet, sv: StateView, device: str
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Compute ``(q_real [N,M], q_dummy [N], num_pairs)`` under ``net`` (no_grad).

    Single source of truth for the no-grad Assignment-Net Q-matrix used by
    acting and by the trainer's target step. Illegal pairs are reset to
    NEG_INF after the dense matrix product so matching ignores them.
    """
    n = sv.n_drivers
    m = sv.n_orders
    with torch.no_grad():
        non_seq = torch.from_numpy(sv.driver_non_seq).float().to(device)
        seq = torch.from_numpy(sv.driver_seq).float().to(device)
        mask = torch.from_numpy(sv.driver_mask).float().to(device)
        order_feats = torch.from_numpy(sv.order_feats).float().to(device)
        q_real_t, q_dummy_t = net.q_matrix(non_seq, seq, mask, order_feats)
        q_real = q_real_t.cpu().numpy().astype(np.float64)
        q_dummy = q_dummy_t.cpu().numpy().astype(np.float64)
    if m > 0:
        # Mask out illegal (capacity/candidacy) pairs.
        q_real = np.where(sv.legal_mask, q_real, NEG_INF)
        num_pairs = int(sv.legal_mask.sum())
    else:
        q_real = np.full((n, 0), NEG_INF, dtype=np.float64)
        num_pairs = 0
    return q_real, q_dummy, num_pairs


class AssignmentActor:
    """Selects actions for all drivers via Assignment-Net + bipartite matching."""

    def __init__(
        self,
        qnet: AssignmentNet,
        encoder: FeatureEncoder,
        area: Coord,
        network_speed: float,
        k_nearest: int = 20,
        use_knn: bool = False,
        device: str = "cpu",
        explorer: Optional[QNoiseExplorer] = None,
        pickup_distance_threshold: Optional[float] = None,
        distance_fn: Optional[Callable[[Coord, Coord], float]] = None,
        coord_to_km: Tuple[float, float] = (1.0, 1.0),
        network_distance_is_metres: bool = False,
        allow_idle: bool = True,
    ):
        self.qnet = qnet
        self.encoder = encoder
        self.area = area
        self.k_nearest = int(k_nearest)
        self.use_knn = bool(use_knn)
        self.device = device
        self.cell_size = max(network_speed, 1e-6)
        self._index = GridIndex(area, self.cell_size)
        self.explorer = explorer
        self.pickup_distance_threshold = pickup_distance_threshold
        self.distance_fn = distance_fn
        self.coord_to_km = coord_to_km
        self.network_distance_is_metres = network_distance_is_metres
        # Whether a driver may actively pick the no-order (dummy) action when a
        # legal order is available (False -> idling is only a passive fallback).
        self.allow_idle = bool(allow_idle)

    def act(
        self,
        observations: Dict[int, Dict],
        explore_step: Optional[int] = None,
    ):
        """Select actions for all drivers.

        Returns ``(actions, state, action_order_feats, action_is_dummy, debug)``
        where the last two are the per-driver current-Q inputs stored in replay.
        """
        if not observations:
            return {}, None, None, None, {"num_orders": 0, "num_pairs": 0}

        state, driver_ids, order_ids = encode_state(
            observations,
            self.encoder,
            index=self._index,
            k_nearest=self.k_nearest,
            use_knn=self.use_knn,
            structured=True,
            pickup_distance_threshold=self.pickup_distance_threshold,
            distance_fn=self.distance_fn,
            coord_to_km=self.coord_to_km,
            network_distance_is_metres=self.network_distance_is_metres,
        )
        n = state.n_drivers
        m = state.n_orders

        q_real, q_dummy, num_pairs = assignment_q_matrix(
            self.qnet, state, self.device
        )

        if explore_step is not None and self.explorer is not None:
            q_real, q_dummy = self.explorer.perturb(
                q_real, q_dummy, state.legal_mask, explore_step
            )

        chosen_col, _ = match_drivers_to_orders(
            q_real, q_dummy, state.legal_mask, allow_idle=self.allow_idle
        )

        actions: Dict[int, Dict] = {}
        order_dim = state.dummy_feat.shape[0]
        action_order_feats = np.zeros((n, order_dim), dtype=np.float32)
        action_is_dummy = np.zeros((n,), dtype=bool)
        for i, did in enumerate(driver_ids):
            c = chosen_col[i]
            if c >= 0:
                actions[did] = {"orders": [order_ids[c]]}
                action_order_feats[i] = state.order_feats[c]
            else:
                actions[did] = {"orders": []}
                action_is_dummy[i] = True

        return (
            actions,
            state,
            action_order_feats,
            action_is_dummy,
            {"num_orders": m, "num_pairs": int(num_pairs)},
        )
