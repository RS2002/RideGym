"""Replay snapshot for BMG-Q (whole-step, GAT-aware).

Because the GAT couples all drivers, the gradient current-Q cannot be stored as
flat per-pair rows (as MFDDQN does). Instead each snapshot keeps the full
:class:`StateView` plus the neighbour lists for both states, so the trainer can
re-run the GAT with grad on the current state (then gather the chosen pairs) and
rebuild the next-state Q-matrix + Hungarian for the Double-DQN target.

Reuses :class:`iddqn.replay.StateView` and :class:`iddqn.replay.ReplayBuffer`
unchanged; only the per-step record differs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from iddqn.replay import StateView


@dataclass
class BMGStepSnapshot:
    """One decision step's transition for all drivers.

    Attributes
    ----------
    state / next_state:
        Current / next :class:`StateView`.
    state_neighbours / next_state_neighbours:
        Per-driver top-K neighbour row-index arrays for each state.
    chosen_col:
        ``[N]`` int; the order column each driver took (-1 = dummy / no order).
        Used to gather current-Q after the grad GAT pass.
    rewards:
        ``[N]`` per-driver reward for this step.
    done:
        Episode-termination flag (shared across drivers).
    """

    state: StateView
    state_neighbours: List[np.ndarray]
    chosen_col: np.ndarray
    rewards: np.ndarray
    next_state: StateView
    next_state_neighbours: List[np.ndarray]
    done: bool