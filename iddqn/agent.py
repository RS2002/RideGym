"""IDDQN agent: Double DQN with bipartite-matching action selection.

Key design (as agreed):

* A shared :class:`PairQNet` scores every (driver, order) pair and the per-driver
  dummy (no-order) action.
* The next-state target selects actions by solving the augmented bipartite
  matching over the Q-matrix, NOT by independent per-driver max -- removing the
  over-estimation that conflicting greedy actions would cause.
* Double DQN semantics: the ONLINE network selects the next action (matching on
  the next state), the TARGET network evaluates it:

      y_i = r_i + gamma * Q_target(s'_i, a'_i) * (1 - done)

* The no-grad Q-matrix is computed by the single shared
  :func:`iddqn.inference.q_matrix_for_state` (same code as the acting path).
* The current Q, ``Q_online(s_i, a_i)``, is computed SEPARATELY and WITH grad by
  calling the online net directly on the stored assembled action pair features.
  It deliberately does NOT go through ``q_matrix_for_state`` (which is no_grad).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn

from iddqn.qnet import PairQNet
from iddqn.assignment_net import AssignmentNet
from iddqn.replay import StepSnapshot
from iddqn.matching import match_drivers_to_orders
from iddqn.inference import q_matrix_for_state
from iddqn.assignment_inference import assignment_q_matrix


@dataclass
class IDDQNConfig:
    """Training hyper-parameters.

    Target-network update: choose ONE of two modes.

    * Soft (Polyak) update (``tau > 0``, the default): every gradient step the
      target is nudged toward the online net by ``target <- (1-tau)*target +
      tau*online``. This keeps the target close to the online net at all times,
      which is essential here -- with only ~60 updates per episode a slow hard
      sync leaves the bootstrap target frozen at its random-init value for
      several episodes, so Q never bootstraps up to the discounted-return scale
      and the policy collapses to 'take no order'. ``tau=0.01`` was verified to
      let next-state Q bootstrap correctly.
    * Hard sync (set ``tau = 0`` to disable Polyak): copy the online weights
      into the target every ``target_sync_every`` gradient steps. If you use
      this, keep ``target_sync_every`` small (e.g. 20), NOT 200.
    """

    gamma: float = 0.99
    lr: float = 1e-3
    # Whether a driver may ACTIVELY take the no-order (dummy) action in the
    # next-state action selection of the Q-target. Must match the actor's
    # ``allow_idle`` so target and behaviour share the same idling rule:
    # False -> idling is only a passive fallback (a real legal order is
    # assigned whenever available), preventing the policy from learning to
    # refuse demand.
    allow_idle: bool = True
    batch_size: int = 8           # number of whole-step snapshots per update
    # Soft-update coefficient. > 0 -> Polyak every step (recommended). 0 ->
    # fall back to hard sync every ``target_sync_every`` steps.
    tau: float = 0.01
    target_sync_every: int = 20   # hard-sync interval, used only when tau == 0
    grad_clip: float = 10.0
    device: str = "cpu"


class IDDQNAgent:
    """Holds the online/target networks and performs gradient updates."""

    def __init__(self, pair_dim: int, cfg: IDDQNConfig, qnet: PairQNet = None):
        self.cfg = cfg
        self.device = cfg.device
        self.online = (qnet or PairQNet(pair_dim)).to(self.device)
        self.target = copy.deepcopy(self.online).to(self.device)
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.optim = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        self._updates = 0

    def update(self, batch: List[StepSnapshot]) -> float:
        """One gradient step on a batch of whole-step snapshots; returns loss."""
        device = self.device
        gamma = self.cfg.gamma

        # --- Current Q: Q_online(s_i, a_i), WITH grad, index-free. ---
        # Direct online-net call on stored assembled action pairs. Must NOT use
        # the no_grad q_matrix_for_state.
        cur_pairs = np.concatenate(
            [snap.action_pair_feats for snap in batch], axis=0
        )
        cur_pt = torch.from_numpy(cur_pairs).float().to(device)
        q_cur = self.online(cur_pt)  # [sum_N], grad-tracked

        # --- Targets: y_i = r_i + gamma * Q_target(s'_i, a'_i) * (1 - done). ---
        # a' from ONLINE-net matching on the next state (Double DQN selection),
        # evaluated by the TARGET net. All no-grad via q_matrix_for_state.
        targets = []
        for snap in batch:
            r = snap.rewards.astype(np.float64)
            ns = snap.next_state
            n = ns.n_drivers
            if snap.done or n == 0:
                targets.append(r)
                continue

            q_real_on, q_dummy_on, _ = q_matrix_for_state(self.online, ns, device)
            chosen_col, _ = match_drivers_to_orders(
                q_real_on, q_dummy_on, ns.legal_mask,
                allow_idle=self.cfg.allow_idle,
            )

            q_real_tg, q_dummy_tg, _ = q_matrix_for_state(self.target, ns, device)
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
        """Update the target net: Polyak soft update if ``tau>0``, else hard sync."""
        tau = self.cfg.tau
        if tau and tau > 0.0:
            with torch.no_grad():
                for tp, op in zip(
                    self.target.parameters(), self.online.parameters()
                ):
                    tp.mul_(1.0 - tau).add_(tau * op)
            # Buffers (e.g. BatchNorm stats) — copy directly; PairQNet has none
            # currently, but this keeps the contract correct if layers change.
            for tb, ob in zip(self.target.buffers(), self.online.buffers()):
                tb.copy_(ob)
        elif self._updates % self.cfg.target_sync_every == 0:
            self.target.load_state_dict(self.online.state_dict())


class AssignmentAgent:
    """IDDQN agent variant using :class:`AssignmentNet` (matrix-product Q).

    Identical training scheme to :class:`IDDQNAgent` (Double DQN with
    bipartite-matching action selection, soft/hard target sync). The ONLY
    difference is the Q-function: drivers and orders are encoded independently
    and Q is their inner product. Current-Q for the chosen action is therefore
    ``driver_embed_i . order_side(a_i)`` -- still per-driver independent, so it
    is recomputed WITH grad from the stored structured driver inputs and the
    chosen order's features (dummy actions use the net's learnable dummy
    embedding).

    Alignment invariant: both ``q_cur`` and the target ``y`` are concatenated
    over the SAME batch list, and within each snapshot over the SAME driver row
    index ``i`` (current uses ``snap.state`` row i, target uses
    ``snap.next_state`` row i and ``snap.rewards[i]``). So element k of
    ``q_cur`` and of ``y`` always refer to the same (snapshot, driver). The
    order column order is irrelevant: current-Q never rebuilds the matrix, and
    the target reads ``q_real_tg[i, chosen_col[i]]`` by row.
    """

    def __init__(self, net: AssignmentNet, cfg: IDDQNConfig):
        self.cfg = cfg
        self.device = cfg.device
        self.online = net.to(self.device)
        self.target = copy.deepcopy(self.online).to(self.device)
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.optim = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        self._updates = 0

    def _current_q(self, batch: List[StepSnapshot]) -> torch.Tensor:
        """``Q_online(s_i, a_i)`` for every driver in the batch, WITH grad."""
        device = self.device
        qs = []
        for snap in batch:
            sv = snap.state
            n = sv.n_drivers
            if n == 0:
                continue
            non_seq = torch.from_numpy(sv.driver_non_seq).float().to(device)
            seq = torch.from_numpy(sv.driver_seq).float().to(device)
            mask = torch.from_numpy(sv.driver_mask).float().to(device)
            d = self.online.encode_drivers(non_seq, seq, mask)  # [n, h]

            ord_feats = torch.from_numpy(
                snap.action_order_feats
            ).float().to(device)
            o_real = self.online._order_side(ord_feats)  # [n, h]
            dummy = self.online._dummy_side(
                self.online.dummy_embed.unsqueeze(0)
            )  # [1, h]
            is_dummy = (
                torch.from_numpy(snap.action_is_dummy.astype(np.float32))
                .to(device)
                .unsqueeze(-1)
            )  # [n, 1]
            o = o_real * (1.0 - is_dummy) + dummy * is_dummy  # [n, h]
            qs.append((d * o).sum(dim=1))  # [n]
        return torch.cat(qs, dim=0)

    def update(self, batch: List[StepSnapshot]) -> float:
        """One gradient step on a batch of whole-step snapshots; returns loss."""
        device = self.device
        gamma = self.cfg.gamma

        q_cur = self._current_q(batch)  # [sum_N], grad-tracked

        targets = []
        for snap in batch:
            r = snap.rewards.astype(np.float64)
            ns = snap.next_state
            n = ns.n_drivers
            if snap.done or n == 0:
                targets.append(r)
                continue

            q_real_on, q_dummy_on, _ = assignment_q_matrix(
                self.online, ns, device
            )
            chosen_col, _ = match_drivers_to_orders(
                q_real_on, q_dummy_on, ns.legal_mask,
                allow_idle=self.cfg.allow_idle,
            )

            q_real_tg, q_dummy_tg, _ = assignment_q_matrix(
                self.target, ns, device
            )
            next_q = np.empty(n, dtype=np.float64)
            for i in range(n):
                c = chosen_col[i]
                next_q[i] = q_real_tg[i, c] if c >= 0 else q_dummy_tg[i]
            targets.append(r + gamma * next_q)

        y = torch.from_numpy(np.concatenate(targets, axis=0)).float().to(device)

        loss = nn.functional.smooth_l1_loss(q_cur, y)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.grad_clip)
        self.optim.step()

        self._updates += 1
        self._sync_target()
        return float(loss.item())

    _sync_target = IDDQNAgent._sync_target