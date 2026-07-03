"""MeanField DDQN agent: Double DQN with a mean-field-conditioned Q + matching.

Mirrors :class:`iddqn.agent.IDDQNAgent` exactly, with one difference: every
Q-value is conditioned on the per-driver mean action field ``a_bar``, and the
next-state action is selected by re-running the mean-field fixed-point loop
(:func:`mfddqn.mean_field.mean_field_solve`) -- which solves the Hungarian
matching every iteration -- rather than a single matching over a plain Q-matrix.

Double DQN semantics are unchanged:

    a' = mean-field-matching(ONLINE, s')          # selection (greedy, no noise)
    y_i = r_i + gamma * Q_target(s'_i, a'_i, a_bar'_i) * (1 - done)

where ``a_bar'`` is the converged mean field the ONLINE net's fixed-point loop
produced on the next state. The target net is then evaluated on that SAME
selected action and mean field (mean field held fixed across the online/target
swap, so the only Double-DQN decoupling is selection-vs-evaluation, exactly as
the pair-Q IDDQN does for its dummy/real columns).

* The no-grad mean-field Q-matrices are computed by the shared
  :func:`mean_field_solve` (same code as the acting path).
* The gradient-carrying current Q ``Q_online(s_i, a_i, a_bar_i)`` is computed
  SEPARATELY, with grad, by calling the online net directly on the stored
  assembled action triples (it must NOT go through the no_grad solver).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn

from iddqn.matching import NEG_INF

from mfddqn.mf_qnet import MeanFieldPairQNet
from mfddqn.mf_replay import MFStepSnapshot
from iddqn.matching import match_drivers_to_orders
from mfddqn.mean_field import (
    MeanFieldConfig,
    mean_field_solve,
    _mean_field_q_matrix,
)


@dataclass
class MFDDQNConfig:
    """Training hyper-parameters (identical to IDDQNConfig for parity).

    See :class:`iddqn.agent.IDDQNConfig` for the target-update rationale; the
    soft (Polyak) update with ``tau > 0`` is recommended for the same reason
    (only ~60 updates per episode make a slow hard sync starve the bootstrap).
    """

    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 8
    tau: float = 0.01
    target_sync_every: int = 20
    grad_clip: float = 10.0
    device: str = "cpu"


class MFDDQNAgent:
    """Holds the online/target mean-field nets and performs gradient updates."""

    def __init__(
        self,
        pair_dim: int,
        mean_field_dim: int,
        cfg: MFDDQNConfig,
        mf_cfg: MeanFieldConfig,
        qnet: MeanFieldPairQNet = None,
    ):
        self.cfg = cfg
        self.mf_cfg = mf_cfg
        self.device = cfg.device
        self.online = (
            qnet or MeanFieldPairQNet(pair_dim, mean_field_dim)
        ).to(self.device)
        self.target = copy.deepcopy(self.online).to(self.device)
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.optim = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        self._updates = 0

    def update(self, batch: List[MFStepSnapshot]) -> float:
        """One gradient step on a batch of whole-step snapshots; returns loss."""
        device = self.device
        gamma = self.cfg.gamma

        # --- Current Q: Q_online(s_i, a_i, a_bar_i), WITH grad, index-free. ---
        # Direct online-net call on stored assembled action triples (driver ++
        # order/dummy ++ converged a_bar). Must NOT use the no_grad solver.
        cur_triples = np.concatenate(
            [snap.action_triple_feats for snap in batch], axis=0
        )
        cur_pt = torch.from_numpy(cur_triples).float().to(device)
        q_cur = self.online(cur_pt)  # [sum_N], grad-tracked

        # --- Targets: y_i = r_i + gamma * Q_target(s'_i, a'_i, a_bar'_i). ---
        # a' and a_bar' from the ONLINE-net mean-field fixed-point loop on s'
        # (greedy: no exploration). The TARGET net then evaluates that selected
        # action under the same converged mean field. All no-grad.
        targets = []
        for snap in batch:
            r = snap.rewards.astype(np.float64)
            ns = snap.next_state
            n = ns.n_drivers
            if snap.done or n == 0:
                targets.append(r)
                continue

            if self.mf_cfg.simplified:
                # Simplified: the next state's mean field is the a_bar this step
                # PRODUCED (carried over, no within-step loop). Online selects
                # via a single Hungarian over Q(s', ., a_bar_out); target then
                # evaluates the same action under the same a_bar_out.
                a_bar = snap.a_bar_out
                qr_on, qd_on, _ = _mean_field_q_matrix(
                    self.online, ns, a_bar, device
                )
                chosen_col, _ = match_drivers_to_orders(
                    qr_on, qd_on, ns.legal_mask,
                    allow_idle=self.mf_cfg.allow_idle,
                )
            else:
                # Full: online runs the fixed-point loop -> conflict-free a'
                # and the converged mean field a_bar' that produced it.
                chosen_col, a_bar, _qr_on, _qd_on, _ = mean_field_solve(
                    self.online,
                    ns,
                    snap.next_state_neighbours,
                    self.mf_cfg,
                    device,
                    explorer=None,
                    explore_step=None,
                )

            # Target evaluation: score the SAME state under the converged a_bar'
            # with the TARGET net, then read off the online-selected action.
            q_real_tg, q_dummy_tg, _ = _mean_field_q_matrix(
                self.target, ns, a_bar, device
            )
            next_q = np.empty(n, dtype=np.float64)
            for i in range(n):
                c = chosen_col[i]
                next_q[i] = q_real_tg[i, c] if c >= 0 else q_dummy_tg[i]
            targets.append(r + gamma * next_q)

        y = torch.from_numpy(np.concatenate(targets, axis=0)).float().to(device)

        # --- Loss + optimise. ---
        loss = nn.functional.smooth_l1_loss(q_cur, y)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.grad_clip)
        self.optim.step()

        self._updates += 1
        self._sync_target()

        return float(loss.item())

    def _sync_target(self) -> None:
        """Polyak soft update if ``tau>0``, else hard sync (mirrors IDDQN)."""
        tau = self.cfg.tau
        if tau and tau > 0.0:
            with torch.no_grad():
                for tp, op in zip(
                    self.target.parameters(), self.online.parameters()
                ):
                    tp.mul_(1.0 - tau).add_(tau * op)
            for tb, ob in zip(self.target.buffers(), self.online.buffers()):
                tb.copy_(ob)
        elif self._updates % self.cfg.target_sync_every == 0:
            self.target.load_state_dict(self.online.state_dict())