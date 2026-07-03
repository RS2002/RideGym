"""Assignment-Net Q-network: produces the whole driver x order Q-matrix at once.

Unlike :class:`iddqn.qnet.PairQNet` (which scores one concatenated pair to a
scalar), Assignment-Net encodes drivers and orders *independently* into an
h-dim space and forms Q via a matrix product, so one forward yields the full
[N, M] matrix. The dummy ("no order") action is a learnable order embedding.

Driver encoder: non-sequence scalars (MLP) + en-route order sequence
(Transformer, no positional encoding) fused into one h-dim vector. Every input
first passes an ARL gate ``y = x * MLP(x)`` (linear, no activation) that
re-weights features; the sequence shares one ARL across positions.

Order encoder: ARL gate + MLP into the same h-dim space.

Multiply net: driver and order embeddings each pass a small MLP; the order
embedding is softmax-normalised over the feature dimension (non-negative,
sums to 1) so it cannot interfere destructively with the driver embedding, then
``Q = driver(N,h) @ order(M,h)^T`` gives the [N, M] matrix.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def _mlp(in_dim: int, hidden: Sequence[int], out_dim: int) -> nn.Sequential:
    layers = []
    d = in_dim
    for h in hidden:
        layers.append(nn.Linear(d, h))
        layers.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class ARL(nn.Module):
    """Adaptive re-weighting layer: ``y = x * MLP(x)`` (linear gate, no act)."""

    def __init__(self, dim: int, hidden: int = 32):
        super().__init__()
        # MLP outputs a per-feature importance weight (same width as x). No
        # output activation by design (sigmoid hurt performance empirically).
        self.gate = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class DriverEncoder(nn.Module):
    """Encode a driver (non-seq scalars + en-route order seq) to h dims."""

    def __init__(
        self,
        non_seq_dim: int,
        seq_token_dim: int,
        embed_dim: int,
        tf_heads: int = 2,
        tf_ff: int | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        # ARL gates: one for non-seq, one shared across all sequence positions.
        self.non_seq_arl = ARL(non_seq_dim)
        self.seq_arl = ARL(seq_token_dim)
        # Non-sequence branch -> embed_dim.
        self.non_seq_mlp = _mlp(non_seq_dim, (embed_dim,), embed_dim)
        # Sequence branch: project tokens to embed_dim, small Transformer
        # encoder WITHOUT positional encoding (order of en-route stops is not a
        # positional signal we want to bake in), then masked mean-pool.
        self.seq_proj = nn.Linear(seq_token_dim, embed_dim)
        # Learnable [CLS] token prepended to every sequence (BERT-style). It is
        # ALWAYS visible (never masked), so every row has at least one valid
        # position -- this removes the all-padded-row NaN edge case entirely,
        # and its output is the pooled sequence representation.
        self.cls_token = nn.Parameter(torch.randn(embed_dim) * 0.02)
        ff = tf_ff if tf_ff is not None else 2 * embed_dim
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=tf_heads,
            dim_feedforward=ff,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        # Fuse non-seq + pooled-seq into the final h-dim driver embedding.
        self.fuse = _mlp(2 * embed_dim, (embed_dim,), embed_dim)

    def forward(
        self, non_seq: torch.Tensor, seq: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """``non_seq [N, D]``, ``seq [N, L, T]``, ``mask [N, L]`` -> ``[N, h]``."""
        n = non_seq.shape[0]
        ns = self.non_seq_mlp(self.non_seq_arl(non_seq))

        # Apply the shared ARL to every sequence position, then project.
        seq_g = self.seq_arl(seq)  # [N, L, T]
        tok = self.seq_proj(seq_g)  # [N, L, h]
        # Prepend the [CLS] token (column 0) to every sequence.
        cls = self.cls_token.expand(n, 1, -1)  # [N, 1, h]
        tok = torch.cat([cls, tok], dim=1)  # [N, L+1, h]
        # Padding mask (True = IGNORE). CLS (column 0) is always visible; the
        # real-token columns use 1-mask. No row can be fully padded now, so the
        # attention softmax is always well-defined.
        cls_keep = torch.ones(n, 1, dtype=mask.dtype, device=tok.device)
        valid = torch.cat([cls_keep, mask], dim=1)  # [N, L+1], 1=valid
        pad = valid <= 0.5  # [N, L+1] padding mask (True = IGNORE)
        enc = self.transformer(tok, src_key_padding_mask=pad)  # [N, L+1, h]
        # Mean-pool over all valid positions INCLUDING the CLS token (column 0,
        # always valid -> denominator >= 1, no division by zero).
        v = valid.unsqueeze(-1)  # [N, L+1, 1]
        pooled = (enc * v).sum(dim=1) / v.sum(dim=1).clamp_min(1e-6)  # [N, h]

        return self.fuse(torch.cat([ns, pooled], dim=1))


class OrderEncoder(nn.Module):
    """Encode an order (non-seq scalars) to h dims via ARL + MLP."""

    def __init__(self, order_dim: int, embed_dim: int):
        super().__init__()
        self.arl = ARL(order_dim)
        self.mlp = _mlp(order_dim, (embed_dim,), embed_dim)

    def forward(self, order_feats: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.arl(order_feats))


class AssignmentNet(nn.Module):
    """Full Assignment-Net producing the (driver, order) Q-matrix in one pass."""

    def __init__(
        self,
        non_seq_dim: int,
        seq_token_dim: int,
        order_dim: int,
        embed_dim: int = 64,
        tf_heads: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.driver_enc = DriverEncoder(
            non_seq_dim, seq_token_dim, embed_dim, tf_heads=tf_heads
        )
        self.order_enc = OrderEncoder(order_dim, embed_dim)
        # Post-encoder MLPs before the multiply.
        self.driver_mul = _mlp(embed_dim, (embed_dim,), embed_dim)
        self.order_mul = _mlp(embed_dim, (embed_dim,), embed_dim)
        # Magnitude / direction split for the driver side. The direction is
        # L2-normalised (norm pinned to 1 -> blocks the norm-explosion that made
        # training diverge after a few epochs); a PER-DRIVER scalar head carries
        # the magnitude, so Q's scale is still decided by the driver state
        # (chiefly the time step), while the convex (sum-to-1) order side only
        # sets the direction. softplus keeps the magnitude non-negative.
        self.driver_mag_head = nn.Linear(embed_dim, 1)
        # Learnable dummy ("no order") embedding, scored like any other order.
        self.dummy_embed = nn.Parameter(torch.randn(embed_dim) * 0.01)
        # self.dummy_embed = nn.Parameter(torch.zeros(order_dim,), requires_grad=False)

    def encode_drivers(
        self, non_seq: torch.Tensor, seq: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Driver embeddings ready for the multiply: ``[N, h]``.

        Output = magnitude * unit-direction. The unit direction (L2-normalised)
        keeps the embedding norm at 1; the per-driver softplus magnitude carries
        the Q scale. Their product is returned so the downstream matrix product
        ``d @ o^T`` is unchanged.
        """
        e = self.driver_mul(self.driver_enc(non_seq, seq, mask))  # [N, h]
        # direction = e / e.norm(dim=-1, keepdim=True).clamp_min(1e-6)  # [N, h]
        # mag = nn.functional.softplus(self.driver_mag_head(e))  # [N, 1] >= 0
        # return mag * direction
        return e

    def _order_side(self, order_feats: torch.Tensor) -> torch.Tensor:
        """Order embeddings with feature-dim softmax: ``[M, h]``."""
        z = self.order_mul(self.order_enc(order_feats))
        # # Softmax over the feature dim -> non-negative, sums to 1.
        # return torch.softmax(z, dim=-1)
        z = z * z
        z = z / z.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return z
    
    def _dummy_side(self, dummy_embed: torch.Tensor) -> torch.Tensor:
        """Dummy embedding with feature-dim softmax: ``[1, h]``."""
        z = dummy_embed
        # return torch.softmax(z, dim=-1)
        z = z * z
        z = z / z.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return z

    def q_matrix(
        self,
        non_seq: torch.Tensor,
        seq: torch.Tensor,
        mask: torch.Tensor,
        order_feats: torch.Tensor,
    ):
        """Return ``(q_real [N, M], q_dummy [N])``.

        ``order_feats`` are the real orders only; the dummy column is computed
        from the learnable dummy embedding through the same order pathway.
        """
        d = self.encode_drivers(non_seq, seq, mask)  # [N, h]
        if order_feats.shape[0] > 0:
            o = self._order_side(order_feats)  # [M, h]
            q_real = d @ o.t()  # [N, M]
        else:
            q_real = d.new_zeros((d.shape[0], 0))

        # dummy_embed = self.dummy_embed * self.dummy_embed
        # dummy_embed = dummy_embed / dummy_embed.sum(dim=-1, keepdim=True).clamp_min(1e-6)# [1, h]
        # dummy_embed = self._order_side(self.dummy_embed)  # [1, h]
        dummy_embed = self._dummy_side(self.dummy_embed.unsqueeze(0))  # [1, h]
        q_dummy = (d @ dummy_embed.t()).squeeze(-1)  # [N]
        return q_real, q_dummy
