"""BMG-Q agent: Graph Attention Double DQN (GATDDQN).

Mirrors :class:`iddqn.agent.IDDQNAgent`. The only differences stem from the GAT
coupling all drivers:

* Current-Q (with grad): re-run the GAT on the stored StateView, then gather
  each driver's chosen (driver, order) Q from the dense matrix. Cannot use a
  pre-stored flat pair vector, because the enriched driver embedding depends on
  the (current) parameters and the neighbours' raw features.
* Target (Double DQN): the ONLINE net's GAT Q-matrix + Hungarian select a' on
  s'; the TARGET net's GAT Q-matrix evaluates that a'. Both no-grad via the
  shared :func:`gat_q_matrix`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn

from iddqn.matching import match_drivers_to_orders

from bmgq.gat_qnet import GATQNet, build_neighbour_tensors
from bmgq.bmgq_replay import BMGStepSnapshot
from bmgq.bmgq_inference import gat_q_matrix


@dataclass
class BMGQConfig:
    """Training hyper-parameters (identical to IDDQNConfig for parity)."""

    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 8
    tau: float = 0.01
    target_sync_every: int = 20
    grad_clip: float = 10.0
    # Whether a driver may ACTIVELY take the no-order (dummy) action in the
    # Q-target's next-state action selection. Must match the actor's
    # ``allow_idle`` (False -> idling is only a passive fallback).
    allow_idle: bool = True
    device: str = "cpu"


class BMGQAgent:
    """Holds the online/target GAT nets and performs gradient updates."""

    def __init__(
        self,
        driver_dim: int,
        order_dim: int,
        neighbours_k: int,
        cfg: BMGQConfig,
        qnet: GATQNet = None,
    ):
        self.cfg = cfg
        self.neighbours_k = int(neighbours_k)
        self.device = cfg.device
        self.online = (qnet or GATQNet(driver_dim, order_dim)).to(self.device)
        self.target = copy.deepcopy(self.online).to(self.device)
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.optim = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        self._updates = 0

    # ----------------------------------------------------------- current-Q
    def _current_q(self, snap: BMGStepSnapshot) -> torch.Tensor:
        """Grad-tracked Q(s_i, a_i) for every driver in one snapshot -> [N]."""
        sv = snap.state
        n, m = sv.n_drivers, sv.n_orders
        device = self.device

        idx_np, mask_np = build_neighbour_tensors(
            snap.state_neighbours, self.neighbours_k
        )
        drv = torch.from_numpy(sv.driver_feats).float().to(device)
        nb_idx = torch.from_numpy(idx_np).to(device)
        nb_mask = torch.from_numpy(mask_np).to(device)
        dummy = torch.from_numpy(sv.dummy_feat).float().to(device)

        if m > 0:
            ords = torch.from_numpy(sv.order_feats).float().to(device)
            rows_np, cols_np = np.nonzero(sv.legal_mask)
            rows = torch.from_numpy(rows_np).long().to(device)
            cols = torch.from_numpy(cols_np).long().to(device)
        else:
            ords = torch.zeros((0, sv.dummy_feat.shape[0]), device=device)
            rows = torch.zeros(0, dtype=torch.long, device=device)
            cols = torch.zeros(0, dtype=torch.long, device=device)

        _enriched, q_legal, q_dummy = self.online(
            drv, nb_idx, nb_mask, ords, dummy, rows, cols
        )

        # Map each legal (row, col) to its position in the q_legal vector so we
        # can gather the chosen pair per driver.
        pair_pos = {}
        if m > 0:
            for p, (ri, ci) in enumerate(zip(rows_np.tolist(), cols_np.tolist())):
                pair_pos[(ri, ci)] = p

        out = q_dummy.new_empty(n)
        for i in range(n):
            c = int(snap.chosen_col[i])
            if c < 0:
                out[i] = q_dummy[i]
            else:
                out[i] = q_legal[pair_pos[(i, c)]]
        return out

    # ----------------------------------------------------------- target
    def _target_q(self, snap: BMGStepSnapshot) -> np.ndarray:
        """Double-DQN next-state value per driver -> [N] (no grad)."""
        r = snap.rewards.astype(np.float64)
        ns = snap.next_state
        n = ns.n_drivers
        if snap.done or n == 0:
            return r

        # Online selection: GAT Q-matrix + Hungarian on s'.
        q_real_on, q_dummy_on, _ = gat_q_matrix(
            self.online, ns, snap.next_state_neighbours,
            self.neighbours_k, self.device,
        )
        chosen_col, _ = match_drivers_to_orders(
            q_real_on, q_dummy_on, ns.legal_mask,
            allow_idle=self.cfg.allow_idle,
        )

        # Target evaluation of the online-selected action.
        q_real_tg, q_dummy_tg, _ = gat_q_matrix(
            self.target, ns, snap.next_state_neighbours,
            self.neighbours_k, self.device,
        )
        next_q = np.empty(n, dtype=np.float64)
        for i in range(n):
            c = chosen_col[i]
            next_q[i] = q_real_tg[i, c] if c >= 0 else q_dummy_tg[i]
        return r + self.cfg.gamma * next_q

    # ----------------------------------------------------------- update
    def update(self, batch: List[BMGStepSnapshot]) -> float:
        """One gradient step on a batch of whole-step snapshots; returns loss."""
        device = self.device

        q_cur = torch.cat([self._current_q(snap) for snap in batch], dim=0)
        targets = [self._target_q(snap) for snap in batch]
        y = torch.from_numpy(np.concatenate(targets, axis=0)).float().to(device)

        loss = nn.functional.smooth_l1_loss(q_cur, y)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.grad_clip)
        self.optim.step()

        self._updates += 1
        self._sync_target()
        return float(loss.item())

    def _sync_target(self) -> None:
        """Polyak soft update if tau>0, else hard sync (mirrors IDDQN)."""
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