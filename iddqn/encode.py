"""Shared state encoding: observations -> StateView.

Both the acting path (inference / data collection) and the training target
computation must turn an environment observation into the same encoded form (a
:class:`StateView`: driver features, order features, legality mask, dummy
feature). Centralising it here guarantees a single source of truth, so acting
and training can never drift apart in how they featurise a state.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from iddqn.features import FeatureEncoder
from iddqn.matching import build_legal_mask
from iddqn.replay import StateView
from benchmark.spatial import GridIndex

Coord = Tuple[float, float]


def _eucl_sq(a: Coord, b: Coord) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def encode_state(
    observations: Dict[int, Dict],
    encoder: FeatureEncoder,
    index: Optional[GridIndex] = None,
    k_nearest: int = 20,
    use_knn: bool = False,
    structured: bool = False,
    pickup_distance_threshold: Optional[float] = None,
    distance_fn: Optional[Callable[[Coord, Coord], float]] = None,
    coord_to_km: Tuple[float, float] = (1.0, 1.0),
    network_distance_is_metres: bool = False,
) -> Tuple[StateView, List[int], List[int]]:
    """Encode an observation into a :class:`StateView`.

    Candidate set
    -------------
    By default (``use_knn=False``) NO spatial candidate pruning is applied: every
    driver may match any order that fits its capacity (a fully-connected legality
    mask). This is the recommended setting -- the bipartite matching is solved
    densely and is fast enough at scale, and pruning the candidates to each
    driver's k nearest orders structurally removes most of the action space the
    RL agent is trying to learn over (it can only ever pick a locally-biased
    subset, which was found to cripple learning). Pruning helps a pure
    nearest-distance heuristic but hurts a policy that must reason globally.

    Set ``use_knn=True`` (and pass a built ``index``) to restore the k-nearest
    candidate pruning, e.g. for very large scenarios where the dense matching
    becomes a bottleneck. When enabled, ``index`` must be a :class:`GridIndex`
    and ``k_nearest`` controls how many nearest drivers each order proposes.

    Returns
    -------
    state:
        The encoded :class:`StateView`.
    driver_ids:
        Row order of drivers (maps row index -> env driver id).
    order_ids:
        Column order of orders (maps column index -> env order id).
    """
    if use_knn and index is None:
        raise ValueError("use_knn=True requires a GridIndex `index`.")
    driver_ids = list(observations.keys())
    n = len(driver_ids)
    any_obs = observations[driver_ids[0]]
    pending = any_obs["pending_orders"]
    time = any_obs["time"]

    drv_feats = np.stack(
        [encoder.encode_driver(observations[d]["self"], time) for d in driver_ids]
    ).astype(np.float32)

    order_ids = [o["order_id"] for o in pending]
    m = len(order_ids)
    if m > 0:
        ord_feats = np.stack([encoder.encode_order(o) for o in pending]).astype(
            np.float32
        )
        order_party = np.array(
            [o["num_passengers"] for o in pending], dtype=np.int64
        )
    else:
        ord_feats = np.zeros((0, encoder.cfg.order_dim), dtype=np.float32)
        order_party = np.zeros((0,), dtype=np.int64)

    cap = encoder.cfg.max_capacity
    free_cap = np.array(
        [cap - observations[d]["self"]["committed_passengers"] for d in driver_ids],
        dtype=np.int64,
    )

    # Candidate columns per driver. With k-NN pruning disabled (the default),
    # `candidate_cols` stays None so build_legal_mask applies ONLY the capacity
    # constraint -> a fully-connected (capacity-feasible) legality mask.
    candidate_cols: Optional[Dict[int, List[int]]] = None
    if use_knn and m > 0:
        candidate_cols = {i: [] for i in range(n)}
        driver_loc = {
            i: observations[d]["self"]["location"]
            for i, d in enumerate(driver_ids)
        }
        index.build(driver_loc)
        for j, o in enumerate(pending):
            nearest = index.nearest(o["origin"], k_nearest, distance_fn=_eucl_sq)
            for _, row_i in nearest:
                candidate_cols[row_i].append(j)

    legal_mask = build_legal_mask(
        driver_ids=list(range(n)),
        order_party=order_party,
        driver_free_cap=free_cap,
        candidate_cols=candidate_cols,
    )

    # Pickup-distance gate: a (driver, order) pair is only legal if the pickup
    # distance (order origin -> driver location) is within the threshold. This
    # is ANDed onto the capacity/candidacy mask, so an order with no driver in
    # range simply stays pending. ``distance_fn`` defaults to Euclidean when not
    # supplied (the abstract scenarios' metric); callers measuring under a road
    # network pass ``network.distance`` for the true pickup distance.
    if pickup_distance_threshold is not None and m > 0:
        driver_locs = np.array(
            [observations[d]["self"]["location"] for d in driver_ids],
            dtype=np.float64,
        )  # [N, 2]
        order_origins = np.array(
            [o["origin"] for o in pending], dtype=np.float64
        )  # [M, 2]
        if distance_fn is None:
            # Vectorised straight-line distance in KILOMETRES. Coordinate
            # deltas are scaled to km by coord_to_km (= (1, 1) for the
            # abstract km scenarios; = (111*cos(lat0), 111) for (lon, lat)
            # graph scenarios, a lat-linear correction adequate at city
            # scale). Compared on squared distance to skip the sqrt; the
            # threshold is already in km.
            kx, ky = coord_to_km
            diff = driver_locs[:, None, :] - order_origins[None, :, :]  # [N,M,2]
            diff[:, :, 0] *= kx
            diff[:, :, 1] *= ky
            d2 = np.einsum("ijk,ijk->ij", diff, diff)  # [N, M] squared km
            dist_ok = d2 <= (pickup_distance_threshold ** 2)
        else:
            # Exact road-network distance: one shortest-path lookup per pair.
            # SLOW on graph scenarios; only used when the caller explicitly asks
            # for the network metric. Threshold is in km; OSMnx returns metres,
            # so scale the threshold up to metres when needed.
            thr = pickup_distance_threshold * (
                1000.0 if network_distance_is_metres else 1.0
            )
            dist_ok = np.zeros((n, m), dtype=bool)
            for i in range(n):
                dloc = tuple(driver_locs[i])
                for j in range(m):
                    dist_ok[i, j] = (
                        distance_fn(tuple(order_origins[j]), dloc) <= thr
                    )
        legal_mask = legal_mask & dist_ok

    drv_non_seq = drv_seq = drv_mask = None
    if structured:
        _, drv_non_seq, drv_seq, drv_mask = encoder.encode_drivers_struct(
            {d: observations[d] for d in driver_ids}, time
        )

    state = StateView(
        driver_feats=drv_feats,
        order_feats=ord_feats,
        legal_mask=legal_mask,
        free_cap=free_cap,
        dummy_feat=encoder.dummy_order(),
        driver_non_seq=drv_non_seq,
        driver_seq=drv_seq,
        driver_mask=drv_mask,
    )
    return state, driver_ids, order_ids