"""Mean-field fixed-point solver shared by acting and the training target.

This is the single source of truth for turning a state into a conflict-free
mean-field assignment, used by BOTH the acting path (inference / data
collection) and the trainer's next-state target step -- so the two can never
drift in how they compute the mean field or the matching (mirroring the role of
:func:`iddqn.inference.q_matrix_for_state`).

The mean field and the matching are mutually dependent: a driver's chosen action
depends on the mean action of its neighbours, which depends on those neighbours'
chosen actions. We resolve this circular dependency with a short fixed-point
loop (Option B, Hungarian inside every iteration), repeated
``mean_field_iters`` times:

    a_bar_i = 0                         # initial neighbour mean for every driver
    repeat mean_field_iters times:
        Q_real[i, j] = Q([drv_i, ord_j, a_bar_i])     # mean-field-conditioned
        Q_dummy[i]   = Q([drv_i, dummy, a_bar_i])
        chosen = hungarian(Q_real, Q_dummy, legal_mask)   # conflict-free
        a_bar_i = mean( action_embed(chosen_j) for j in top-K neighbours of i )

The action embedding of a driver is the order feature vector it was assigned, or
the dummy feature vector when it took no order. Building ``a_bar`` from the
conflict-free Hungarian assignment (not a greedy argmax) keeps the mean field
consistent with what the drivers will actually do.

The top-K neighbour structure is purely spatial (the K nearest drivers by
location, via the shared :class:`benchmark.spatial.GridIndex`) and is computed
ONCE per state, then reused across every iteration -- driver locations do not
change within a single decision step. At training time the neighbour lists are
carried on the snapshot so the target step can rebuild the same mean field
without the original observations.

All Q-matrix computation here runs under ``no_grad``; it is for action selection
and target evaluation only. The trainer computes the gradient-carrying current-Q
separately by calling the online net directly on the stored action triples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from iddqn.matching import match_drivers_to_orders, NEG_INF
from iddqn.replay import StateView
from mfddqn.mf_qnet import MeanFieldPairQNet

Coord = Tuple[float, float]


@dataclass
class MeanFieldConfig:
    """Mean-field hyper-parameters.

    Attributes
    ----------
    neighbours_k:
        Number of nearest spatial neighbours whose action embeddings are
        averaged into each driver's mean field ``a_bar`` (top-K). The paper uses
        a local neighbourhood; we use K = 30.
    iters:
        Number of fixed-point iterations (compute Q-matrix -> Hungarian ->
        rebuild a_bar). Mean field converges fast; 2 is a good default.
    """

    neighbours_k: int = 30
    iters: int = 2
    # Simplified variant: instead of solving the within-step fixed point, use
    # the mean field carried over from the PREVIOUS step (all-zeros at the first
    # step) and do a single Q-matrix + Hungarian (``iters`` is ignored). The
    # caller threads the previous step's a_bar in and gets this step's a_bar
    # out, persisting it to the next step.
    simplified: bool = False
    # Whether a driver may ACTIVELY take the no-order (dummy) action when a
    # legal order is available. False -> idling is only a passive fallback (a
    # real legal order is assigned whenever available), preventing the policy
    # from learning to refuse demand. Threaded into every Hungarian matching
    # the solver runs, and mirrored in the agent's Q-target selection.
    allow_idle: bool = True


def _eucl_sq(a: Coord, b: Coord) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def build_neighbour_lists(
    driver_locs: List[Coord], index, k: int
) -> List[np.ndarray]:
    """Return, for each driver row, the row indices of its K nearest neighbours.

    Spatial only (nearest by location), excluding the driver itself. Computed
    once per state via the shared :class:`GridIndex`; the result is reused across
    every fixed-point iteration and stored on the snapshot for the target step.

    Parameters
    ----------
    driver_locs:
        Row-ordered driver locations ``[(x, y), ...]`` (row i == matrix row i).
    index:
        A :class:`benchmark.spatial.GridIndex` to (re)build over the drivers.
    k:
        Neighbourhood size.
    """
    n = len(driver_locs)
    loc_map = {i: driver_locs[i] for i in range(n)}
    index.build(loc_map)
    neighbours: List[np.ndarray] = []
    for i in range(n):
        # Query k+1 because the driver itself is in the index; drop self below.
        found = index.nearest(driver_locs[i], k + 1, distance_fn=_eucl_sq)
        rows = [row_i for _d, row_i in found if row_i != i][:k]
        neighbours.append(np.asarray(rows, dtype=np.int64))
    return neighbours


def _mean_field_q_matrix(
    net: MeanFieldPairQNet,
    sv: StateView,
    a_bar: np.ndarray,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Compute ``(q_real [N,M], q_dummy [N], num_pairs)`` given the mean field.

    Only legal pairs are scored (illegal entries stay at NEG_INF). The triple fed
    to the net is ``[driver_feat, order_feat, a_bar_i]`` for real pairs and
    ``[driver_feat, dummy_feat, a_bar_i]`` for the dummy column. Runs under
    ``no_grad``.
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
                triple = np.concatenate(
                    [
                        sv.driver_feats[rows],
                        sv.order_feats[cols],
                        a_bar[rows],
                    ],
                    axis=1,
                )
                pt = torch.from_numpy(triple).float().to(device)
                q_real[rows, cols] = net(pt).cpu().numpy()
        dummy_tiled = np.tile(sv.dummy_feat, (n, 1))
        dtriple = np.concatenate(
            [sv.driver_feats, dummy_tiled, a_bar], axis=1
        )
        dpt = torch.from_numpy(dtriple).float().to(device)
        q_dummy = net(dpt).cpu().numpy().astype(np.float64)
    return q_real, q_dummy, num_pairs


def _action_embeddings(
    sv: StateView, chosen_col: np.ndarray
) -> np.ndarray:
    """Per-driver action embedding ``[N, order_dim]`` from a chosen-column array.

    Driver i's embedding is its assigned order's feature vector, or the dummy
    feature vector when it took no order (``chosen_col[i] == -1``).
    """
    n = sv.n_drivers
    embeds = np.empty((n, sv.dummy_feat.shape[0]), dtype=np.float32)
    for i in range(n):
        c = chosen_col[i]
        embeds[i] = sv.order_feats[c] if c >= 0 else sv.dummy_feat
    return embeds


def _update_mean_field(
    action_embeds: np.ndarray, neighbours: List[np.ndarray], dummy_feat: np.ndarray
) -> np.ndarray:
    """Average each driver's neighbours' action embeddings into ``a_bar [N, D]``.

    A driver with no neighbours (isolated) gets the dummy feature vector as its
    mean field, which is the natural 'no neighbour activity' baseline.
    """
    n = action_embeds.shape[0]
    a_bar = np.empty((n, action_embeds.shape[1]), dtype=np.float32)
    for i in range(n):
        idx = neighbours[i]
        if idx.size:
            a_bar[i] = action_embeds[idx].mean(axis=0)
        else:
            a_bar[i] = dummy_feat
    return a_bar


def mean_field_solve(
    net: MeanFieldPairQNet,
    sv: StateView,
    neighbours: List[np.ndarray],
    cfg: MeanFieldConfig,
    device: str,
    explorer=None,
    explore_step: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Run the mean-field fixed-point loop; return the converged solution.

    Returns
    -------
    chosen_col:
        ``[N]`` final conflict-free assignment (order column per driver, or -1).
    a_bar:
        ``[N, order_dim]`` converged per-driver mean field used to PRODUCE that
        final assignment (i.e. the input the final Q-matrix was scored with).
        Carried to replay so current-Q is reproducible.
    q_real:
        ``[N, M]`` final mean-field Q-matrix (post-exploration if exploring).
    q_dummy:
        ``[N]`` final dummy Q-values (post-exploration if exploring).
    num_pairs:
        Number of legal pairs scored (for debug/timing).

    Exploration (per the agreed design) is applied ONLY on the final iteration's
    Q-matrix: the mean field is allowed to converge on clean (greedy) Q-values,
    then noise is added to the last Q-matrix that yields the stored action. The
    target step calls this with ``explorer=None`` (greedy) -- Double-DQN greedy
    target selection.
    """
    n = sv.n_drivers
    d = sv.dummy_feat.shape[0]
    # Initial mean field: zeros (no assumed neighbour activity yet).
    a_bar = np.zeros((n, d), dtype=np.float32)

    chosen_col = np.full(n, -1, dtype=np.int64)
    q_real = np.full((n, sv.n_orders), NEG_INF, dtype=np.float64)
    q_dummy = np.zeros(n, dtype=np.float64)
    num_pairs = 0

    iters = max(1, int(cfg.iters))
    for it in range(iters):
        q_real, q_dummy, num_pairs = _mean_field_q_matrix(
            net, sv, a_bar, device
        )
        # Exploration only on the FINAL iteration (clean convergence first).
        is_final = it == iters - 1
        if (
            is_final
            and explorer is not None
            and explore_step is not None
        ):
            q_real, q_dummy = explorer.perturb(
                q_real, q_dummy, sv.legal_mask, explore_step
            )
        chosen_col, _ = match_drivers_to_orders(
            q_real, q_dummy, sv.legal_mask, allow_idle=cfg.allow_idle
        )
        if is_final:
            break
        # Rebuild the mean field from the new conflict-free assignment for the
        # NEXT iteration. (a_bar at this point is the input that produced the
        # assignment of the *next* iteration; the value returned alongside the
        # final chosen_col is the one used to score the final Q-matrix.)
        action_embeds = _action_embeddings(sv, chosen_col)
        a_bar = _update_mean_field(action_embeds, neighbours, sv.dummy_feat)

    return chosen_col, a_bar, q_real, q_dummy, num_pairs


def mean_field_solve_simplified(
    net: MeanFieldPairQNet,
    sv: StateView,
    neighbours: List[np.ndarray],
    a_bar_in: np.ndarray,
    cfg: MeanFieldConfig,
    device: str,
    explorer=None,
    explore_step: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Simplified mean field: use last step's ``a_bar_in`` (no within-step loop).

    A single Q-matrix is scored under the carried-over ``a_bar_in`` and a single
    Hungarian gives the conflict-free assignment; ``a_bar_out`` is then rebuilt
    from that assignment and the top-K neighbours, to be carried to the NEXT
    step. ``iters`` is ignored here.

    Returns
    -------
    chosen_col:
        ``[N]`` conflict-free assignment (order column per driver, or -1).
    a_bar_in:
        ``[N, D]`` the mean field actually used to score Q (echoed back so the
        caller can store it for current-Q reproducibility).
    a_bar_out:
        ``[N, D]`` mean field rebuilt from this step's assignment, to carry to
        the next step (and used as the next-state mean field at train time).
    q_real:
        ``[N, M]`` Q-matrix (post-exploration if exploring).
    q_dummy:
        ``[N]`` dummy Q-values (post-exploration if exploring).
    num_pairs:
        Number of legal pairs scored.
    """
    q_real, q_dummy, num_pairs = _mean_field_q_matrix(
        net, sv, a_bar_in, device
    )
    if explorer is not None and explore_step is not None:
        q_real, q_dummy = explorer.perturb(
            q_real, q_dummy, sv.legal_mask, explore_step
        )
    chosen_col, _ = match_drivers_to_orders(
        q_real, q_dummy, sv.legal_mask, allow_idle=cfg.allow_idle
    )
    action_embeds = _action_embeddings(sv, chosen_col)
    a_bar_out = _update_mean_field(action_embeds, neighbours, sv.dummy_feat)
    return chosen_col, a_bar_in, a_bar_out, q_real, q_dummy, num_pairs