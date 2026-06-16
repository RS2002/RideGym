"""Shared MLP Q-network scoring (driver, order) pairs.

A single shared network maps a concatenated [driver_features, order_features]
vector to a scalar Q-value. The same weights score every pair, so the model is
independent of the number of drivers/orders and naturally handles the variable
order count each step. The dummy order (no-order action) is just another order
feature vector, so its Q-value comes from the same network.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class PairQNet(nn.Module):
    """MLP mapping a (driver, order) pair feature vector to a scalar Q-value."""

    def __init__(self, pair_dim: int, hidden: Sequence[int] = (128, 128)):
        super().__init__()
        layers = []
        in_dim = pair_dim
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, pairs: torch.Tensor) -> torch.Tensor:
        """Score a batch of pair vectors; returns Q with the trailing dim squeezed."""
        return self.net(pairs).squeeze(-1)