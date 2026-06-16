"""Bipartite matching of drivers to orders over a Q-value matrix.

Given per-(driver, order) Q-values plus a per-driver 'dummy order' Q-value (the
value of taking no order), this builds an augmented assignment matrix and solves
for the matching that maximises total Q under the constraints:

* each driver takes at most one order (or the dummy = no order);
* each real order is taken by at most one driver;
* a full driver (no free capacity for an order) may only take the dummy.

The augmented matrix is ``[N, M + N]``:

* left block ``[N, M]``  : real-order Q, with illegal (full / non-candidate)
  entries set to ``-INF`` so the solver never picks them;
* right block ``[N, N]`` : a diagonal of per-driver dummy Q-values, off-diagonal
  ``-INF``. Each driver thus owns a private dummy column, so 'take no order' is
  always feasible and drivers never compete for it.

Because there are ``M + N >= N`` columns, every driver is always matched to some
column (at worst its own finite-valued dummy), so the matching is always
feasible -- no driver is ever left without an action.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

NEG_INF = -1e9


def match_drivers_to_orders(
    q_real: np.ndarray,
    q_dummy: np.ndarray,
    legal_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the augmented assignment problem.

    Parameters
    ----------
    q_real:
        ``[N, M]`` Q-values for (driver, real-order) pairs.
    q_dummy:
        ``[N]`` Q-value for each driver's dummy (take-no-order) action.
    legal_mask:
        ``[N, M]`` boolean; ``True`` where (driver, order) is a legal,
        in-candidate pair. Illegal entries are forced to ``-INF``.

    Returns
    -------
    chosen_order_col:
        ``[N]`` int array. For driver ``i``, the matched real-order column index
        in ``[0, M)``, or ``-1`` if the driver took its dummy (no order).
    chosen_q:
        ``[N]`` float array of the Q-value of each driver's chosen action
        (the real-pair Q, or the dummy Q). Used directly for TD targets.
    """
    from scipy.optimize import linear_sum_assignment

    n, m = q_real.shape
    aug = np.full((n, m + n), NEG_INF, dtype=float)

    # Left block: legal real-order Q-values.
    if m > 0:
        aug[:, :m] = np.where(legal_mask, q_real, NEG_INF)

    # Right block: per-driver private dummy column (diagonal).
    rows = np.arange(n)
    aug[rows, m + rows] = q_dummy

    # Maximise total Q  ->  minimise -Q.
    row_ind, col_ind = linear_sum_assignment(-aug)

    chosen_order_col = np.full(n, -1, dtype=np.int64)
    chosen_q = np.empty(n, dtype=np.float64)
    for r, c in zip(row_ind, col_ind):
        if c < m:
            # Invariant: the optimal matching can never place a driver on an
            # -INF real-order column, because re-routing that driver to its own
            # finite-valued private dummy column is always feasible and strictly
            # increases the total -- contradicting optimality. So reaching a
            # -INF real column here means an upstream bug (dummy set to -INF,
            # malformed mask, dimension mismatch); fail loudly rather than
            # silently masking it as 'no order'.
            assert aug[r, c] > NEG_INF / 2, (
                f"Driver row {r} was matched to illegal (-INF) order column {c}; "
                f"this violates the matching feasibility invariant and indicates "
                f"a bug in the Q-values, legality mask, or matrix construction."
            )
            chosen_order_col[r] = c
            chosen_q[r] = aug[r, c]
        else:
            # Driver took its private dummy column: no order this step.
            chosen_order_col[r] = -1
            chosen_q[r] = q_dummy[r]
    return chosen_order_col, chosen_q


def build_legal_mask(
    driver_ids: List[int],
    order_party: np.ndarray,
    driver_free_cap: np.ndarray,
    candidate_cols: Optional[Dict[int, List[int]]] = None,
) -> np.ndarray:
    """Construct the ``[N, M]`` legality mask.

    A (driver, order) pair is legal iff the driver has enough free capacity for
    the order's party size and (if candidate pruning is used) the order is among
    the driver's candidate columns.

    Parameters
    ----------
    driver_ids:
        Row order of drivers (length N).
    order_party:
        ``[M]`` party size per order column.
    driver_free_cap:
        ``[N]`` true free capacity per driver (capacity - committed).
    candidate_cols:
        Optional ``{row_index: [legal order column indices]}`` from k-NN
        pruning. If ``None``, only the capacity constraint is applied.
    """
    n = len(driver_ids)
    m = order_party.shape[0]
    mask = np.zeros((n, m), dtype=bool)
    if m == 0:
        return mask

    # Capacity feasibility: party <= free capacity (broadcast).
    cap_ok = order_party[None, :] <= driver_free_cap[:, None]

    if candidate_cols is None:
        return cap_ok

    for i in range(n):
        cols = candidate_cols.get(i)
        if not cols:
            continue
        mask[i, cols] = True
    return mask & cap_ok