"""GATDDQN Q-network: graph-attention driver embedding + per-pair scoring.

The GAT enriches each driver with its neighbours, so forward needs the whole
driver set (not flat pair rows). It returns the legal-pair and dummy Q-values;
the caller scatters them into a dense [N, M] matrix and runs the Hungarian
matching, exactly as IDDQN. One forward is shared by acting, target, and the
gradient current-Q (differing only by no_grad), so the paths never drift.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from bmgq.gat import MultiHeadGAT


def build_neighbour_tensors(
    neighbours: List[np.ndarray], k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Pad neighbour lists into [N, K+1] index + bool mask (self in column 0)."""
    n = len(neighbours)
    k1 = k + 1
    idx = np.zeros((n, k1), dtype=np.int64)
    mask = np.zeros((n, k1), dtype=bool)
    for i in range(n):
        idx[i, 0] = i  # self
        mask[i, 0] = True
        nb = neighbours[i]
        if nb.size:
            c = min(nb.size, k)
            idx[i, 1 : 1 + c] = nb[:c]
            mask[i, 1 : 1 + c] = True
    return idx, mask


class GATQNet(nn.Module):
    """Stacked GAT driver embedding + shared MLP scoring [enriched_driver, order]."""

    def __init__(
        self,
        driver_dim: int,
        order_dim: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        hidden: Sequence[int] = (128, 128),
        gat_layers: int = 1,
    ):
        super().__init__()
        self.driver_dim = int(driver_dim)
        self.order_dim = int(order_dim)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)

        layers = []
        # in_dim = self.driver_dim
        in_dim = self.embed_dim
        for _ in range(max(1, int(gat_layers))):
            layers.append(
                MultiHeadGAT(in_dim, self.embed_dim, self.num_heads, residual=False)
            )
            in_dim = self.embed_dim
        self.gat = nn.ModuleList(layers)
        self.gat_act = nn.ELU()

        self.drive_emb = nn.Sequential(
            nn.Linear(self.driver_dim, self.embed_dim),
            # nn.ReLU(),
            # nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.order_emb = nn.Sequential(
            nn.Linear(self.order_dim, self.embed_dim),
            # nn.ReLU(),
            # nn.Linear(self.embed_dim, self.embed_dim),
        )

        mlp = []
        d = self.embed_dim * 3
        for hsz in hidden:
            mlp += [nn.Linear(d, hsz), nn.ReLU()]
            d = hsz
        mlp.append(nn.Linear(d, 1))
        self.pair_mlp = nn.Sequential(*mlp)

    def embed_drivers(self, driver_feats, neighbour_idx, neighbour_mask):
        """Run the stacked GAT -> enriched driver embeddings [N, embed_dim]."""
        h = driver_feats
        for i, layer in enumerate(self.gat):
            h = layer(h, neighbour_idx, neighbour_mask)
            if i < len(self.gat) - 1:
                h = self.gat_act(h)
        h = torch.concat([driver_feats, h], dim=-1)
        return h

    def score_pairs(self, enriched_driver, order):
        """Score [*, embed_dim + order_dim] pairs -> [*] Q-values."""
        return self.pair_mlp(torch.cat([enriched_driver, order], dim=-1)).squeeze(-1)

    def forward(
        self,
        driver_feats: torch.Tensor,
        neighbour_idx: torch.Tensor,
        neighbour_mask: torch.Tensor,
        order_feats: torch.Tensor,
        dummy_feat: torch.Tensor,
        legal_rows: torch.Tensor,
        legal_cols: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (enriched [N,embed], q_legal [P], q_dummy [N]).

        Only the P legal pairs (legal_rows/cols from nonzero(legal_mask)) are
        scored; the caller scatters them back into the dense [N, M] matrix.
        """

        driver_feats = self.drive_emb(driver_feats)
        order_feats = self.order_emb(order_feats)
        dummy_feat = self.order_emb(dummy_feat)

        enriched = self.embed_drivers(driver_feats, neighbour_idx, neighbour_mask)
        n = enriched.shape[0]

        if legal_rows.numel() > 0:
            q_legal = self.score_pairs(
                enriched[legal_rows], order_feats[legal_cols]
            )
        else:
            q_legal = enriched.new_zeros((0,))

        dummy_tiled = dummy_feat.unsqueeze(0).expand(n, -1)
        q_dummy = self.score_pairs(enriched, dummy_tiled)
        return enriched, q_legal, q_dummy
