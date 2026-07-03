"""MeanField DDQN acting path: encode -> mean-field solve -> actions.

Mirrors :class:`iddqn.inference.IDDQNActor`: it encodes the observation into a
:class:`StateView` (reusing :func:`iddqn.encode.encode_state`), then -- instead
of a single Q-matrix + matching -- runs the mean-field fixed-point loop
(:func:`mfddqn.mean_field.mean_field_solve`) which itself solves the Hungarian
matching every iteration. The final conflict-free assignment is returned as env
actions, together with the converged mean field and assembled action triples for
replay.

The dense (fully-connected, capacity-feasible) legality mask is used exactly as
in IDDQN's default (``use_knn=False``); the spatial index here serves the
separate purpose of the top-K mean-field neighbourhood, not candidate pruning.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np

from iddqn.features import FeatureEncoder
from iddqn.encode import encode_state
from iddqn.exploration import QNoiseExplorer
from benchmark.spatial import GridIndex

from mfddqn.mf_qnet import MeanFieldPairQNet
from mfddqn.mean_field import (
    MeanFieldConfig,
    mean_field_solve,
    mean_field_solve_simplified,
    build_neighbour_lists,
)

Coord = Tuple[float, float]
Area = Tuple[float, float, float, float]


class MFDDQNActor:
    """Selects actions for all drivers via the mean-field Q-net + Hungarian loop."""

    def __init__(
        self,
        qnet: MeanFieldPairQNet,
        encoder: FeatureEncoder,
        area: Area,
        network_speed: float,
        mf_cfg: MeanFieldConfig,
        k_nearest: int = 20,
        use_knn: bool = False,
        device: str = "cpu",
        explorer: Optional[QNoiseExplorer] = None,
        pickup_distance_threshold: Optional[float] = None,
        distance_fn: Optional[Callable[[Coord, Coord], float]] = None,
        coord_to_km: Tuple[float, float] = (1.0, 1.0),
        network_distance_is_metres: bool = False,
    ):
        """
        Parameters
        ----------
        qnet:
            The shared :class:`MeanFieldPairQNet`.
        encoder:
            Feature encoder (shared with IDDQN's FeatureEncoder).
        area:
            True coordinate range of the scenario (graph bounds for osmnx/nyc).
        network_speed:
            Per-step travel distance, used to size the GridIndex cell.
        mf_cfg:
            Mean-field config (neighbours_k, iters).
        k_nearest / use_knn:
            Candidate-pruning controls threaded from the trainer, kept for parity
            with IDDQN. Default ``use_knn=False`` -> dense matching. The spatial
            index is shared between optional candidate pruning and the mandatory
            top-K neighbour computation.
        explorer:
            Optional exploration noise applied on the final mean-field iteration.
        """
        self.qnet = qnet
        self.encoder = encoder
        self.area = area
        self.mf_cfg = mf_cfg
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

    def act(
        self,
        observations: Dict[int, Dict],
        explore_step: Optional[int] = None,
        a_bar_in: Optional[np.ndarray] = None,
    ):
        """Select actions for all drivers via the mean-field fixed-point loop.

        Returns
        -------
        actions:
            ``{driver_id: {"orders": [order_id]}}`` (empty list = no order).
        state:
            Encoded :class:`StateView` (for replay).
        neighbours:
            Per-driver top-K neighbour row-index arrays (for replay).
        action_triple_feats:
            ``[N, pair_dim + mean_field_dim]`` assembled triples of the chosen
            actions (for the index-free current-Q).
        debug:
            Sizes for timing/inspection.
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
        n = state.n_drivers
        m = state.n_orders

        # Top-K spatial neighbour lists (computed once; reused every iteration).
        driver_locs = [
            observations[d]["self"]["location"] for d in driver_ids
        ]
        neighbours = build_neighbour_lists(
            driver_locs, self._index, self.mf_cfg.neighbours_k
        )

        if self.mf_cfg.simplified:
            # Use the previous step's mean field (zeros at the first step) and a
            # single Q-matrix + Hungarian; a_bar_out is carried to the next step.
            d = state.dummy_feat.shape[0]
            if a_bar_in is None:
                a_bar_in = np.zeros((n, d), dtype=np.float32)
            chosen_col, a_bar, a_bar_out, q_real, q_dummy, num_pairs = (
                mean_field_solve_simplified(
                    self.qnet,
                    state,
                    neighbours,
                    a_bar_in,
                    self.mf_cfg,
                    self.device,
                    explorer=self.explorer,
                    explore_step=explore_step,
                )
            )
        else:
            chosen_col, a_bar, q_real, q_dummy, num_pairs = mean_field_solve(
                self.qnet,
                state,
                neighbours,
                self.mf_cfg,
                self.device,
                explorer=self.explorer,
                explore_step=explore_step,
            )
            a_bar_out = None

        # Assemble env actions + the action triples for replay.
        actions: Dict[int, Dict] = {}
        mf_dim = state.dummy_feat.shape[0]
        pair_dim = state.driver_feats.shape[1] + mf_dim
        triple_dim = pair_dim + mf_dim
        action_triple_feats = np.empty((n, triple_dim), dtype=np.float32)
        for i, did in enumerate(driver_ids):
            c = chosen_col[i]
            if c >= 0:
                actions[did] = {"orders": [order_ids[c]]}
                order_part = state.order_feats[c]
            else:
                actions[did] = {"orders": []}
                order_part = state.dummy_feat
            action_triple_feats[i] = np.concatenate(
                [state.driver_feats[i], order_part, a_bar[i]]
            )

        return (
            actions,
            state,
            neighbours,
            action_triple_feats,
            a_bar_out,
            {"num_orders": m, "num_pairs": int(num_pairs)},
        )