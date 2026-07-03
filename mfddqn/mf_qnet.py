"""Shared MLP Q-network scoring (driver, order, mean-field) triples.

Identical in spirit to :class:`iddqn.qnet.PairQNet`, but the input is augmented
with a **mean action field** block ``a_bar`` -- the average action embedding of
the driver's spatial neighbours. The network therefore maps

    [driver_features, order_features, mean_field] -> scalar Q

The mean-field block has the SAME width as an order feature vector, because an
action's embedding is exactly its order feature vector (the dummy / no-order
action uses the dummy order feature vector). A single shared network scores
every triple, so the model is independent of the number of drivers / orders and
handles the variable per-step order count, exactly like the IDDQN pair net.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class MeanFieldPairQNet(nn.Module):
    """MLP mapping a (driver, order, mean-field) feature vector to a scalar Q.

    Parameters
    ----------
    pair_dim:
        Width of the concatenated ``[driver_features, order_features]`` vector
        (the same ``FeatureConfig.pair_dim`` IDDQN uses).
    mean_field_dim:
        Width of the appended mean-field block ``a_bar``. Equals the order
        feature width (``FeatureConfig.order_dim``), since an action embedding is
        an order feature vector.
    hidden:
        Hidden layer widths (kept identical to IDDQN's default for parity).
    """

    def __init__(
        self,
        pair_dim: int,
        mean_field_dim: int,
        hidden: Sequence[int] = (128, 128),
    ):
        super().__init__()
        self.pair_dim = int(pair_dim)
        self.mean_field_dim = int(mean_field_dim)
        self.input_dim = self.pair_dim + self.mean_field_dim
        layers = []
        in_dim = self.input_dim
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, triples: torch.Tensor) -> torch.Tensor:
        """Score a batch of ``[*, input_dim]`` vectors; squeezes the trailing dim."""
        return self.net(triples).squeeze(-1)