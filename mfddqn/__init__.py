"""MeanField DDQN: Double DQN with a mean-field neighbour term in the Q-value.

Faithful to *Efficient Ridesharing Order Dispatching with Mean Field Multi-Agent
Reinforcement Learning* (Li et al., 2019, https://arxiv.org/abs/1901.11454),
adapted to this simulator's dynamic (driver, order) action space.

The only change from the IDDQN baseline is **how each (driver, order) Q-value is
computed**: every Q-value is additionally conditioned on a *mean action field*
``a_bar_i`` -- the average action embedding of driver ``i``'s top-K spatial
neighbours. This turns the intractable joint-action dependence into a two-body
(driver + averaged-neighbour) interaction that scales to thousands of drivers.

Everything else is identical to IDDQN: the same dense legality mask, the same
Hungarian bipartite matching for the final conflict-free assignment, the same
Double-DQN target / soft target update / replay / exploration, and the same
``BenchmarkConfig`` and training/eval/logging loop.

Because the mean field and the matching are mutually dependent (a driver's
action depends on its neighbours' mean action, which depends on their actions),
each decision step runs a short **fixed-point loop**: compute the mean-field
Q-matrix, solve the Hungarian matching, rebuild every driver's neighbour mean
from the new conflict-free assignment, and repeat ``mean_field_iters`` times.
"""

from mfddqn.mf_qnet import MeanFieldPairQNet
from mfddqn.mean_field import MeanFieldConfig, mean_field_solve

__all__ = [
    "MeanFieldPairQNet",
    "MeanFieldConfig",
    "mean_field_solve",
]