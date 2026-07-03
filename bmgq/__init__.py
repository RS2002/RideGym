"""BMG-Q: Localized Bipartite Match Graph Attention Q-Learning.

Faithful to *BMG-Q: Localized Bipartite Match Graph Attention Q-Learning for
Ride-Pooling Order Dispatch* (Hu et al., 2025, https://arxiv.org/abs/2501.13448),
adapted to this simulator's dynamic (driver, order) action space.

The distinguishing idea is the **Graph Attention Double Deep Q-Network
(GATDDQN)**: before a driver is paired with an order, its state embedding is
enriched by a multi-head graph-attention pass over its top-K nearest
driver-neighbours (and itself). Each neighbour receives a *learned, distinct*
attention weight (GAT-style additive scoring), so -- unlike the mean-field
baseline which averages neighbours' actions -- the model captures heterogeneous,
per-neighbour interdependence among vehicles. No explicit graph object is built:
the neighbourhood is expressed as a masked attention over each driver's
neighbour-index list.

The attention is over driver **states** (known up-front within a decision step),
so there is NO fixed-point loop: each step computes the attention-enriched
driver embeddings once, scores every (enriched-driver, order) pair, and produces
the final conflict-free assignment with the SAME Hungarian bipartite matching as
IDDQN / MFDDQN. Everything else -- the dense legality mask, Double-DQN target,
soft target update, replay, ``QNoiseExplorer`` exploration, and the
``BenchmarkConfig`` / training / eval / logging loop -- is identical to the
other benchmarks, so results are directly comparable.
"""

from bmgq.gat import MultiHeadGAT
from bmgq.gat_qnet import GATQNet

__all__ = [
    "MultiHeadGAT",
    "GATQNet",
]