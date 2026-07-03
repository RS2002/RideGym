"""Shared Q-network scoring (driver, order) pairs.

A single shared network maps a concatenated ``[driver_features, order_features]``
vector to a scalar Q-value. The same weights score every pair, so the model is
independent of the number of drivers/orders and naturally handles the variable
order count each step. The dummy order (no-order action) is just another order
feature vector, so its Q-value comes from the same network.

Two architectures share one interface (input = the pre-concatenated pair vector,
output = scalar Q), selected at construction:

* **Two-tower (default when ``driver_dim`` is given)**: the driver half and the
  order half of the input are encoded SEPARATELY by their own MLP towers into
  embeddings, then FUSED (concatenated) and passed through a small head MLP to
  the Q-value. Encoding each side independently before fusing lets the network
  learn a reusable driver representation and a reusable order representation,
  which tends to generalise better than forcing an early raw-feature mix.
* **Single-tower (``driver_dim=None``)**: the original behaviour -- the whole
  concatenated vector goes straight through one MLP. Kept for backward
  compatibility (existing checkpoints / callers that pass only ``pair_dim``).

In both cases ``forward`` receives the SAME pre-assembled
``[driver_feat ++ order_feat]`` vector the callers already build (driver first,
order second -- the fixed concatenation order used everywhere upstream), so no
caller needs to change how it assembles pairs.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn


def _mlp(in_dim: int, hidden: Sequence[int], out_dim: int) -> nn.Sequential:
    """Build an MLP ``in_dim -> hidden... -> out_dim`` with ReLU between layers."""
    layers = []
    d = in_dim
    for h in hidden:
        layers.append(nn.Linear(d, h))
        layers.append(nn.ReLU())
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class PairQNet(nn.Module):
    """Score a (driver, order) pair feature vector to a scalar Q-value.

    Parameters
    ----------
    pair_dim:
        Width of the concatenated ``[driver_feat, order_feat]`` input vector.
    hidden:
        Hidden layer widths. In two-tower mode these size BOTH per-side towers
        and the fusion head; in single-tower mode they size the one MLP.
    driver_dim:
        Width of the driver half of the input (the split point: the first
        ``driver_dim`` entries are the driver features, the remaining
        ``pair_dim - driver_dim`` are the order/dummy features). When given,
        the network runs in TWO-TOWER mode (encode each side, then fuse). When
        ``None`` (default), it runs in the original SINGLE-TOWER mode.
    embed_dim:
        Per-tower output embedding width in two-tower mode (ignored in
        single-tower mode). The fusion head sees ``2 * embed_dim`` inputs.
    """

    def __init__(
        self,
        pair_dim: int,
        hidden: Sequence[int] = (128, 128),
        driver_dim: Optional[int] = None,
        embed_dim: int = 64,
    ):
        super().__init__()
        self.pair_dim = int(pair_dim)
        self.two_tower = driver_dim is not None

        if not self.two_tower:
            # Original single-tower MLP over the whole concatenated vector.
            self.driver_dim = None
            self.net = _mlp(self.pair_dim, hidden, 1)
            return

        # Two-tower: separate driver / order encoders, then a fusion head.
        self.driver_dim = int(driver_dim)
        if not (0 < self.driver_dim < self.pair_dim):
            raise ValueError(
                f"driver_dim ({driver_dim}) must be in (0, pair_dim) "
                f"= (0, {self.pair_dim}) so the input splits into a non-empty "
                f"driver half and order half."
            )
        self.order_dim = self.pair_dim - self.driver_dim
        self.embed_dim = int(embed_dim)

        # Each tower encodes its side into an ``embed_dim`` representation.
        self.driver_tower = _mlp(self.driver_dim, hidden, self.embed_dim)
        self.order_tower = _mlp(self.order_dim, hidden, self.embed_dim)
        # Fusion head: concatenate the two embeddings -> scalar Q.
        self.head = _mlp(2 * self.embed_dim, hidden, 1)

    def forward(self, pairs: torch.Tensor) -> torch.Tensor:
        """Score a batch of pair vectors; returns Q with the trailing dim squeezed.

        ``pairs`` is ``[*, pair_dim]`` with the driver features first and the
        order (or dummy) features second -- the fixed layout every caller builds.
        """
        if not self.two_tower:
            return self.net(pairs).squeeze(-1)

        # Split the pre-concatenated vector back into its driver / order halves,
        # encode each independently, then fuse.
        drv = pairs[..., : self.driver_dim]
        ordr = pairs[..., self.driver_dim :]
        drv_emb = self.driver_tower(drv)
        ord_emb = self.order_tower(ordr)
        fused = torch.cat([drv_emb, ord_emb], dim=-1)
        return self.head(fused).squeeze(-1)