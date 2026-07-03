"""Multi-head GAT-style additive attention over a driver's local neighbourhood.

Each driver attends to itself + its top-K nearest driver-neighbours with learned
per-neighbour weights (Velickovic et al.). The neighbourhood is a padded
[N, K+1] index matrix + bool mask, so the whole fleet runs as batched masked
attention with no explicit graph. Per head:

    z_j = W x_j;  e_ij = LeakyReLU(a_src.z_i + a_dst.z_j);
    alpha = softmax_j(e_ij over Nb(i));  head_i = sum_j alpha_ij z_j

Heads are concatenated (+ optional projection) with a residual add of the input.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG_INF = -1e9


class MultiHeadGAT(nn.Module):
    """Multi-head GAT over per-driver local neighbourhoods.

    Parameters
    ----------
    in_dim:
        Input driver-feature width.
    out_dim:
        Output embedding width (after concatenating heads and projecting).
    num_heads:
        Number of attention heads.
    head_dim:
        Per-head projection width. If ``None``, uses ``out_dim // num_heads`` so
        the concatenated heads already have width ``out_dim`` (no extra output
        projection needed); otherwise an output projection maps
        ``num_heads * head_dim -> out_dim``.
    leaky_slope:
        Negative slope of the LeakyReLU in the attention score.
    residual:
        If ``True``, add a (projected if needed) residual of the input to the
        output, keeping the driver's own features prominent.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        head_dim: int = None,
        leaky_slope: float = 0.2,
        residual: bool = True,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.num_heads = int(num_heads)
        if head_dim is None:
            if self.out_dim % self.num_heads != 0:
                raise ValueError(
                    f"out_dim ({self.out_dim}) must be divisible by num_heads "
                    f"({self.num_heads}) when head_dim is None."
                )
            self.head_dim = self.out_dim // self.num_heads
            self._proj_out = None
        else:
            self.head_dim = int(head_dim)
            self._proj_out = nn.Linear(
                self.num_heads * self.head_dim, self.out_dim
            )
        self.leaky_slope = float(leaky_slope)
        self.residual = bool(residual)

        # Per-head linear projection W_h, packed as one [in_dim, H*head_dim] map.
        self.W = nn.Linear(self.in_dim, self.num_heads * self.head_dim, bias=False)
        # GAT additive-attention vectors, factorised into source / destination
        # halves: a_src, a_dst each [num_heads, head_dim].
        self.a_src = nn.Parameter(torch.empty(self.num_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.empty(self.num_heads, self.head_dim))
        # Residual projection (only if the widths differ).
        if self.residual and self.in_dim != self.out_dim:
            self._proj_res = nn.Linear(self.in_dim, self.out_dim, bias=False)
        else:
            self._proj_res = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        if self._proj_out is not None:
            nn.init.xavier_uniform_(self._proj_out.weight)
            nn.init.zeros_(self._proj_out.bias)
        if self._proj_res is not None:
            nn.init.xavier_uniform_(self._proj_res.weight)

    def forward(
        self,
        x: torch.Tensor,
        neighbour_idx: torch.Tensor,
        neighbour_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Enrich each driver embedding via local multi-head graph attention.

        Parameters
        ----------
        x:
            ``[N, in_dim]`` driver features (row i == driver i).
        neighbour_idx:
            ``[N, K1]`` int64 neighbour row indices per driver, where
            ``K1 = 1 + K`` (self in column 0, then the top-K neighbours). Padded
            slots may hold any valid index (e.g. 0); they are masked out.
        neighbour_mask:
            ``[N, K1]`` boolean; ``True`` for real entries (self + actual
            neighbours), ``False`` for padding. Column 0 (self) is always True.

        Returns
        -------
        ``[N, out_dim]`` neighbourhood-enriched driver embeddings.
        """
        n = x.shape[0]
        h, dh = self.num_heads, self.head_dim

        # Project: z [N, H, head_dim].
        z = self.W(x).view(n, h, dh)

        # Source term per driver i: (a_src . z_i)  -> [N, H].
        src_score = (z * self.a_src.unsqueeze(0)).sum(dim=-1)  # [N, H]

        # Destination term per node j: (a_dst . z_j) -> [N, H].
        dst_score_all = (z * self.a_dst.unsqueeze(0)).sum(dim=-1)  # [N, H]

        # Gather each driver's neighbours via advanced indexing. neighbour_idx
        # is [N, K1]; indexing an [N, ...] tensor with it prepends the [N, K1]
        # axes, giving the per-neighbour tensors directly (no manual expand).
        #   dst_score : [N, K1, H]   - neighbour destination scores
        #   z_neigh   : [N, K1, H, head_dim] - neighbour projected values
        dst_score = dst_score_all[neighbour_idx]          # [N, K1, H]
        z_neigh = z[neighbour_idx]                         # [N, K1, H, head_dim]

        # e_ij = LeakyReLU(src_i + dst_j) -> [N, K1, H].
        e = F.leaky_relu(
            src_score.unsqueeze(1) + dst_score, negative_slope=self.leaky_slope
        )

        # Mask padding before softmax (mask broadcast over heads).
        mask = neighbour_mask.unsqueeze(-1)  # [N, K1, 1]
        e = e.masked_fill(~mask, NEG_INF)
        alpha = torch.softmax(e, dim=1)  # over the K1 neighbour axis -> [N, K1, H]

        # Weighted sum over neighbours: [N, H, head_dim].
        head_out = (alpha.unsqueeze(-1) * z_neigh).sum(dim=1)

        # Concatenate heads -> [N, H*head_dim].
        out = head_out.reshape(n, h * dh)
        if self._proj_out is not None:
            out = self._proj_out(out)

        if self.residual:
            res = x if self._proj_res is None else self._proj_res(x)
            out = out + res
        return out