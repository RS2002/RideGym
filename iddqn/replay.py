"""Experience replay storing whole-step snapshots.

The TD target in this method is special: the next-state action of every driver
must be obtained by running a *global bipartite matching* over the next-state
Q-matrix (not an independent per-driver max), to avoid the over-estimation that
conflicting greedy actions would cause. Reconstructing that matching at training
time requires the full next-state context -- all drivers' features, all pending
orders' features, the candidate structure, and the legality mask.

Therefore a replay unit is an entire environment step, not a single (s, a, r,
s') tuple. Each :class:`StepSnapshot` bundles, for one decision step:

* current-state features for all drivers and all pending orders + candidacy;
* the action every driver actually took (chosen order column, or -1 = dummy);
* the per-driver reward;
* the next-state features / candidacy (to rebuild the s' matching);
* the done flag.

Features are stored as compact numpy arrays so memory stays manageable even with
1000 drivers and hundreds of orders per step.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class StateView:
    """Encoded state needed to score and match all drivers at one step.

    Attributes
    ----------
    driver_feats:
        ``[N, driver_dim]`` driver feature matrix.
    order_feats:
        ``[M, order_dim]`` order feature matrix (M may be 0).
    legal_mask:
        ``[N, M]`` boolean legality mask (capacity + candidacy).
    free_cap:
        ``[N]`` true free capacity per driver (for reference/debug).
    dummy_feat:
        ``[order_dim]`` the dummy (no-order) order feature vector from the
        encoder. Carried as data so consumers never re-derive the dummy
        representation (single source of truth = encoder), and so the dummy
        width is available even when ``order_feats`` is empty (M == 0).
    """

    driver_feats: np.ndarray
    order_feats: np.ndarray
    legal_mask: np.ndarray
    free_cap: np.ndarray
    dummy_feat: np.ndarray
    # Structured driver inputs for Assignment-Net (None for the flat MLP path):
    #   driver_non_seq [N, non_seq_dim], driver_seq [N, L, tok_dim],
    #   driver_mask [N, L].
    driver_non_seq: Optional[np.ndarray] = None
    driver_seq: Optional[np.ndarray] = None
    driver_mask: Optional[np.ndarray] = None

    @property
    def n_drivers(self) -> int:
        return self.driver_feats.shape[0]

    @property
    def n_orders(self) -> int:
        return self.order_feats.shape[0]


@dataclass
class StepSnapshot:
    """One full decision step's transition for all drivers.

    Attributes
    ----------
    state:
        Current-state :class:`StateView`.
    action_pair_feats:
        ``[N, pair_dim]`` the *already-assembled* pair feature vector of the
        action each driver actually took: ``driver_feat ++ order_feat`` for a
        taken order, or ``driver_feat ++ dummy_order_feat`` for no-order. Storing
        the assembled pair (rather than a column index into ``state.order_feats``)
        makes the current-Q computation ``Q(s_i, a_i)`` self-contained and
        immune to the otherwise-fragile rule that a column index is only valid
        within its own StateView. (The next-state target still rebuilds the s'
        matching from ``next_state`` -- that is intrinsic to the method.)
    rewards:
        ``[N]`` float; per-driver reward for this step.
    next_state:
        Next-state :class:`StateView` (for rebuilding the s' matching).
    done:
        Episode-termination flag (shared across drivers).
    """

    state: StateView
    action_pair_feats: np.ndarray
    rewards: np.ndarray
    next_state: StateView
    done: bool
    # Assignment-Net current-Q inputs (None for the flat MLP path). Per driver
    # the chosen real order's feature vector (irrelevant where dummy) and a
    # boolean flag marking the no-order (dummy) action, which is scored through
    # the net's learnable dummy embedding rather than any stored vector.
    action_order_feats: Optional[np.ndarray] = None
    action_is_dummy: Optional[np.ndarray] = None


class ReplayBuffer:
    """Fixed-capacity ring buffer of :class:`StepSnapshot`."""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self._buf: List[Optional[StepSnapshot]] = []
        self._pos = 0

    def __len__(self) -> int:
        return len(self._buf)

    def push(self, snapshot: StepSnapshot) -> None:
        if len(self._buf) < self.capacity:
            self._buf.append(snapshot)
        else:
            self._buf[self._pos] = snapshot
        self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size: int) -> List[StepSnapshot]:
        """Sample a list of step snapshots (variable shapes -> list, not tensor)."""
        return random.sample(self._buf, batch_size)

    def can_sample(self, batch_size: int) -> bool:
        return len(self._buf) >= batch_size