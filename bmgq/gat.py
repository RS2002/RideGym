"""Multi-head masked self-attention over a driver's local neighbourhood.

Each driver is treated as a token that attends to itself + its top-K nearest
driver-neighbours via standard scaled dot-product (QKV) self-attention with a
neighbourhood mask -- a simpler, easier-to-train alternative to the additive
GAT attention (Velickovic et al.) used previously. The neighbourhood is a padded
``[N, K+1]`` index matrix + bool mask, so the whole fleet runs as batched masked
attention with no explicit graph. Per head:

    q_i = W_q x_i;  k_j = W_k x_j;  v_j = W_v x_j
    e_ij = (q_i . k_j) / sqrt(head_dim)
    alpha = softmax_j(e_ij over Nb(i));  head_i = sum_j alpha_ij v_j

Heads are concatenated (+ optional output projection) with a residual add of the
input, keeping the driver's own features prominent. The public interface
(``forward(x, neighbour_idx, neighbour_mask) -> [N, out_dim]``) is unchanged, so
callers (``bmgq.gat_qnet.GATQNet``) need no modification.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

NEG_INF = -1e9


class MultiHeadGAT(nn.Module):
    """Multi-head masked dot-product self-attention over local neighbourhoods.

    Despite the retained class name (kept for import compatibility), this is a
    standard Transformer-style multi-head self-attention restricted to each
    driver's neighbourhood, NOT the additive GAT attention. Each driver token
    attends to itself and its top-K spatial neighbours.

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
    residual:
        If ``True``, add a (projected if needed) residual of the input to the
        output, keeping the driver's own features prominent.
    attn_dropout:
        Dropout probability applied to the attention weights (0 disables it).
    leaky_slope:
        Accepted but UNUSED; retained so existing call sites that still pass it
        (from the previous additive-GAT signature) do not break.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        head_dim: int = None,
        residual: bool = True,
        attn_dropout: float = 0.0,
        leaky_slope: float = 0.2,
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
        self.residual = bool(residual)
        self._scale = 1.0 / math.sqrt(self.head_dim)

        # Query / Key / Value projections, each packed as one
        # [in_dim, H*head_dim] map (per-head slices are contiguous).
        self.W_q = nn.Linear(self.in_dim, self.num_heads * self.head_dim, bias=False)
        self.W_k = nn.Linear(self.in_dim, self.num_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(self.in_dim, self.num_heads * self.head_dim, bias=False)

        self.attn_dropout = nn.Dropout(float(attn_dropout))

        # Residual projection (only if the widths differ).
        if self.residual and self.in_dim != self.out_dim:
            self._proj_res = nn.Linear(self.in_dim, self.out_dim, bias=False)
        else:
            self._proj_res = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
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
        """Enrich each driver embedding via local multi-head self-attention.

        Parameters
        ----------
        x:
            ``[N, in_dim]`` driver features (row i == driver i), one token each.
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

        # Project to per-head queries / keys / values: each [N, H, head_dim].
        q = self.W_q(x).view(n, h, dh)
        k = self.W_k(x).view(n, h, dh)
        v = self.W_v(x).view(n, h, dh)

        # Gather each driver's neighbour keys / values via advanced indexing.
        # neighbour_idx is [N, K1]; indexing an [N, ...] tensor with it prepends
        # the [N, K1] axes, giving the per-neighbour tensors directly.
        #   k_neigh, v_neigh : [N, K1, H, head_dim]
        k_neigh = k[neighbour_idx]
        v_neigh = v[neighbour_idx]

        # Scaled dot-product scores between each driver's query and its
        # neighbours' keys, summed over head_dim -> [N, K1, H]. q_i is broadcast
        # over the K1 neighbour axis via q.unsqueeze(1) [N, 1, H, dh].
        e = (q.unsqueeze(1) * k_neigh).sum(dim=-1) * self._scale  # [N, K1, H]

        # Mask padding before softmax (self is column 0, always kept). Mask is
        # broadcast over heads.
        mask = neighbour_mask.unsqueeze(-1)  # [N, K1, 1]
        e = e.masked_fill(~mask, NEG_INF)
        alpha = torch.softmax(e, dim=1)  # over the K1 neighbour axis -> [N, K1, H]
        alpha = self.attn_dropout(alpha)

        # Weighted sum over neighbour values: [N, H, head_dim].
        head_out = (alpha.unsqueeze(-1) * v_neigh).sum(dim=1)

        # Concatenate heads -> [N, H*head_dim].
        out = head_out.reshape(n, h * dh)
        if self._proj_out is not None:
            out = self._proj_out(out)

        if self.residual:
            res = x if self._proj_res is None else self._proj_res(x)
            out = out + res
        return out