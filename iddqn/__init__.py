"""IDDQN: Independent Double DQN with bipartite-matching action selection.

A shared MLP scores every (driver, order) pair's Q-value; a special *dummy
order* yields the Q-value of not taking any order. Each step, a bipartite
matching over the Q-matrix assigns at most one order per driver and at most one
driver per order (full drivers may only pick the dummy). TD targets are computed
by running the same matching on the next state, avoiding the over-estimation
that independent per-driver greedy max would cause under action conflicts.
"""

from iddqn.features import FeatureConfig, FeatureEncoder
from iddqn.qnet import PairQNet
from iddqn.matching import match_drivers_to_orders, build_legal_mask

__all__ = [
    "FeatureConfig",
    "FeatureEncoder",
    "PairQNet",
    "match_drivers_to_orders",
    "build_legal_mask",
]