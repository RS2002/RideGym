"""IDDQN acting path: state encoding -> Q-matrix -> matching -> actions.

Used for data collection (with exploration) and evaluation (greedy). State
encoding is delegated to :func:`iddqn.encode.encode_state` and the Q-matrix
computation (:func:`q_matrix_for_state`) is shared with the trainer's target
step, so acting and training never drift in how they featurise or score a state.

Exploration is provided by a :class:`iddqn.exploration.QNoiseExplorer`.

Gradient note: :func:`q_matrix_for_state` runs under ``no_grad`` and is for
action selection / target evaluation only. The trainer computes the gradient
carrying current-Q ``Q_online(s_i, a_i)`` SEPARATELY by calling the online net
directly on stored action pair features -- it must NOT go through this function.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch

from iddqn.features import FeatureEncoder
from iddqn.qnet import PairQNet
from iddqn.matching import match_drivers_to_orders, NEG_INF
from iddqn.replay import StateView
from iddqn.encode import encode_state
from iddqn.exploration import QNoiseExplorer
from benchmark.spatial import GridIndex

Coord = Tuple[float, float]


def q_matrix_for_state(
    net: PairQNet, sv: StateView, device: str
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Compute ``(q_real [N,M], q_dummy [N], num_pairs)`` for a state under ``net``.

    Only legal pairs are scored in one batched forward; illegal entries stay at
    NEG_INF. The dummy column uses the encoder-produced ``sv.dummy_feat``. Runs
    under ``no_grad`` -- single source of truth for the no-grad Q-matrix used by
    acting and by the trainer's target step.
    """
    n = sv.n_drivers
    m = sv.n_orders
    q_real = np.full((n, m), NEG_INF, dtype=np.float64)
    with torch.no_grad():
        num_pairs = 0
        if m > 0:
            rows, cols = np.nonzero(sv.legal_mask)
            num_pairs = len(rows)
            if num_pairs > 0:
                pair = np.concatenate(
                    [sv.driver_feats[rows], sv.order_feats[cols]], axis=1
                )
                pt = torch.from_numpy(pair).float().to(device)
                q_real[rows, cols] = net(pt).cpu().numpy()
        dummy_tiled = np.tile(sv.dummy_feat, (n, 1))
        dpair = np.concatenate([sv.driver_feats, dummy_tiled], axis=1)
        dpt = torch.from_numpy(dpair).float().to(device)
        q_dummy = net(dpt).cpu().numpy().astype(np.float64)
    return q_real, q_dummy, num_pairs


class IDDQNActor:
    """Selects actions for all drivers via the shared Q-net + bipartite matching."""

    def __init__(
        self,
        qnet: PairQNet,
        encoder: FeatureEncoder,
        area: Coord,
        network_speed: float,
        k_nearest: int = 20,
        use_knn: bool = False,
        device: str = "cpu",
        explorer: Optional[QNoiseExplorer] = None,
    ):
        """
        Parameters
        ----------
        use_knn:
            Whether to prune the candidate (driver, order) set to each order's k
            nearest drivers before matching. Defaults to ``False`` (dense,
            fully-connected matching). This flag is threaded down from the
            trainer's top-level config; the actor does not decide it itself. The
            spatial index is only consulted when this is ``True``.
        k_nearest:
            k for the nearest pruning; ignored when ``use_knn`` is ``False``.
        """
        self.qnet = qnet
        self.encoder = encoder
        self.area = area
        self.k_nearest = int(k_nearest)
        self.use_knn = bool(use_knn)
        self.device = device
        self.cell_size = max(network_speed, 1e-6)
        self._index = GridIndex(area, self.cell_size)
        self.explorer = explorer

    def act(
        self,
        observations: Dict[int, Dict],
        explore_step: Optional[int] = None,
    ):
        """Select actions for all drivers.

        Parameters
        ----------
        observations:
            Per-driver observation dict from the env.
        explore_step:
            If not ``None`` and an explorer is set, exploration noise at this
            global step is applied (behaviour policy). If ``None``, the greedy
            matching is used (evaluation).

        Returns
        -------
        actions:
            ``{driver_id: {"orders": [order_id]}}`` (empty list = no order).
        state:
            Encoded :class:`StateView` (for replay).
        action_pair_feats:
            ``[N, pair_dim]`` assembled pair features of the chosen actions
            (for replay's index-free current-Q).
        debug:
            Sizes for timing/inspection.
        """
        if not observations:
            return {}, None, None, {"num_orders": 0, "num_pairs": 0}

        state, driver_ids, order_ids = encode_state(
            observations,
            self.encoder,
            index=self._index,
            k_nearest=self.k_nearest,
            use_knn=self.use_knn,
        )
        n = state.n_drivers
        m = state.n_orders

        q_real, q_dummy, num_pairs = q_matrix_for_state(
            self.qnet, state, self.device
        )

        if explore_step is not None and self.explorer is not None:
            q_real, q_dummy = self.explorer.perturb(
                q_real, q_dummy, state.legal_mask, explore_step
            )

        chosen_col, _ = match_drivers_to_orders(q_real, q_dummy, state.legal_mask)

        actions: Dict[int, Dict] = {}
        pair_dim = state.driver_feats.shape[1] + state.dummy_feat.shape[0]
        action_pair_feats = np.empty((n, pair_dim), dtype=np.float32)
        for i, did in enumerate(driver_ids):
            c = chosen_col[i]
            if c >= 0:
                actions[did] = {"orders": [order_ids[c]]}
                action_pair_feats[i] = np.concatenate(
                    [state.driver_feats[i], state.order_feats[c]]
                )
            else:
                actions[did] = {"orders": []}
                action_pair_feats[i] = np.concatenate(
                    [state.driver_feats[i], state.dummy_feat]
                )

        return (
            actions,
            state,
            action_pair_feats,
            {"num_orders": m, "num_pairs": int(num_pairs)},
        )