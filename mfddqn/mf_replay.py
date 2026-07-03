"""Replay snapshot for MeanField DDQN (whole-step, mean-field aware).

Identical in role to :class:`iddqn.replay.StepSnapshot`, but carries the extra
state needed to reconstruct the mean-field target on the next state:

* ``next_state_neighbours`` (FULL variant only) -- the per-driver top-K spatial
  neighbour row-index lists for the next state, computed once at collection time
  (driver locations do not change within a step) so the trainer can re-run the
  mean-field fixed-point loop on the next state WITHOUT the original
  observations. The current state's neighbours are NOT stored: current-Q uses
  the pre-assembled ``action_triple_feats`` in both variants, so they are never
  re-read. The SIMPLIFIED variant stores ``a_bar_out`` instead and needs no
  next-state neighbours.
* ``action_triple_feats`` -- the already-assembled ``[N, pair_dim +
  mean_field_dim]`` triple of the action each driver actually took, i.e.
  ``driver_feat ++ order_feat ++ a_bar_i`` (or the dummy variant for no-order),
  where ``a_bar_i`` is the CONVERGED mean field that produced the action. Storing
  the assembled triple makes the gradient-carrying current-Q ``Q(s_i, a_i,
  a_bar_i)`` self-contained, exactly as IDDQN stores ``action_pair_feats``.

The :class:`~iddqn.replay.StateView` (driver/order features, legality mask,
dummy feature) and the :class:`~iddqn.replay.ReplayBuffer` are reused unchanged;
only the per-step snapshot differs, so the next-state mean-field matching can be
rebuilt at training time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from iddqn.replay import StateView


@dataclass
class MFStepSnapshot:
    """One full decision step's transition for all drivers, mean-field aware.

    Attributes
    ----------
    state:
        Current-state :class:`StateView`.
    action_triple_feats:
        ``[N, pair_dim + mean_field_dim]`` assembled triple of the action each
        driver took (driver ++ order/dummy ++ converged a_bar). Used for the
        index-free, gradient-carrying current-Q.
    rewards:
        ``[N]`` per-driver reward for this step.
    next_state:
        Next-state :class:`StateView` (to rebuild the s' mean-field matching).
    done:
        Episode-termination flag (shared across drivers).
    """

    state: StateView
    action_triple_feats: np.ndarray
    rewards: np.ndarray
    next_state: StateView
    # Full variant only: per-driver top-K neighbours for the NEXT state, so the
    # target step can re-run the within-step loop. None in the simplified
    # variant (which reuses ``a_bar_out`` directly -> no loop -> saves storage).
    next_state_neighbours: Optional[List[np.ndarray]] = None
    done: bool = False
    # Simplified-variant only (None in the full variant): the mean field this
    # step PRODUCED (this step's assignment + top-K neighbours). It is the next
    # state's carried-over mean field, so the target step scores
    # Q(s', a', a_bar_out) under it with no within-step loop.
    a_bar_out: Optional[np.ndarray] = None