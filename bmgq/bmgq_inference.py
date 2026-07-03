"""BMG-Q acting path: encode -> GAT Q-matrix -> Hungarian -> actions.

Mirrors :class:`iddqn.inference.IDDQNActor`, but the Q-matrix comes from the
GATQNet (one attention pass over the driver set, no fixed-point loop). The
shared :func:`gat_q_matrix` builds the dense (q_real, q_dummy) under no_grad and
is reused by the trainer's target step so acting and training never drift.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from iddqn.features import FeatureEncoder
from iddqn.encode import encode_state
from iddqn.exploration import QNoiseExplorer
from iddqn.matching import match_drivers_to_orders, NEG_INF
from iddqn.replay import StateView
from benchmark.spatial import GridIndex

from mfddqn.mean_field import build_neighbour_lists
from bmgq.gat_qnet import GATQNet, build_neighbour_tensors

Coord = Tuple[float, float]
Area = Tuple[float, float, float, float]


def gat_q_matrix(
    net: GATQNet,
    sv: StateView,
    neighbours: List[np.ndarray],
    k: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Dense (q_real [N,M], q_dummy [N], num_pairs) under no_grad.

    Runs the GAT once, scores only legal pairs, and scatters them into a dense
    matrix (illegal entries stay NEG_INF). Single source of truth for acting and
    the target step's no-grad Q-matrix.
    """
    n, m = sv.n_drivers, sv.n_orders
    q_real = np.full((n, m), NEG_INF, dtype=np.float64)

    idx_np, mask_np = build_neighbour_tensors(neighbours, k)
    drv = torch.from_numpy(sv.driver_feats).float().to(device)
    nb_idx = torch.from_numpy(idx_np).to(device)
    nb_mask = torch.from_numpy(mask_np).to(device)
    dummy = torch.from_numpy(sv.dummy_feat).float().to(device)
    if m > 0:
        ords = torch.from_numpy(sv.order_feats).float().to(device)
        rows_np, cols_np = np.nonzero(sv.legal_mask)
        rows = torch.from_numpy(rows_np).long().to(device)
        cols = torch.from_numpy(cols_np).long().to(device)
    else:
        ords = torch.zeros((0, sv.dummy_feat.shape[0]), device=device)
        rows = torch.zeros(0, dtype=torch.long, device=device)
        cols = torch.zeros(0, dtype=torch.long, device=device)

    with torch.no_grad():
        _enriched, q_legal, q_dummy_t = net(
            drv, nb_idx, nb_mask, ords, dummy, rows, cols
        )

    num_pairs = int(rows.numel())
    if num_pairs > 0:
        q_real[rows_np, cols_np] = q_legal.cpu().numpy()
    q_dummy = q_dummy_t.cpu().numpy().astype(np.float64)
    return q_real, q_dummy, num_pairs


class BMGQActor:
    """Selects actions for all drivers via the GAT Q-net + Hungarian matching."""

    def __init__(
        self,
        qnet: GATQNet,
        encoder: FeatureEncoder,
        area: Area,
        network_speed: float,
        neighbours_k: int = 30,
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
        self.neighbours_k = int(neighbours_k)
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

        Returns (actions, state, neighbours, chosen_col, debug). ``chosen_col``
        is the per-driver order column (-1 = dummy) for replay.
        """
        if not observations:
            return {}, None, None, None, {"num_orders": 0, "num_pairs": 0}

        state, driver_ids, order_ids = encode_state(
            observations,
            self.encoder,
            index=self._index,
            k_nearest=self.k_nearest,
            use_knn=self.use_knn,
            pickup_distance_threshold=self.pickup_distance_threshold,
            distance_fn=self.distance_fn,
            coord_to_km=self.coord_to_km,
            network_distance_is_metres=self.network_distance_is_metres,
        )
        m = state.n_orders

        driver_locs = [observations[d]["self"]["location"] for d in driver_ids]
        neighbours = build_neighbour_lists(
            driver_locs, self._index, self.neighbours_k
        )

        q_real, q_dummy, num_pairs = gat_q_matrix(
            self.qnet, state, neighbours, self.neighbours_k, self.device
        )

        if explore_step is not None and self.explorer is not None:
            q_real, q_dummy = self.explorer.perturb(
                q_real, q_dummy, state.legal_mask, explore_step
            )

        chosen_col, _ = match_drivers_to_orders(
            q_real, q_dummy, state.legal_mask, allow_idle=self.allow_idle
        )

        actions: Dict[int, Dict] = {}
        for i, did in enumerate(driver_ids):
            c = chosen_col[i]
            actions[did] = {"orders": [order_ids[c]]} if c >= 0 else {"orders": []}

        return (
            actions,
            state,
            neighbours,
            chosen_col,
            {"num_orders": m, "num_pairs": int(num_pairs)},
        )