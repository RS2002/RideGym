"""Baseline policies for demos and benchmarking.

These are intentionally simple but *conflict-aware*: because the environment
raises :class:`ConflictError` when two drivers bid on the same order, a naive
independent random policy would frequently crash. The provided policies use a
lightweight central coordination pass to guarantee each pending order is bid on
by at most one driver, which is exactly the kind of coordination the spec
expects upstream policies to perform.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


class RandomConflictFreePolicy:
    """Assign each pending order to at most one random capable driver.

    Idle drivers that receive no order may randomly relocate. The policy reads
    the per-driver observation dict produced by :class:`RidePoolEnv` and emits a
    matching ``{driver_id: action}`` dict.
    """

    def __init__(self, relocate_prob: float = 0.3, seed=None):
        self.relocate_prob = relocate_prob
        self._rng = np.random.default_rng(seed)

    def act(self, observations: Dict[int, Dict]) -> Dict[int, Dict]:
        if not observations:
            return {}

        # Any driver's obs carries the shared pending list and relocation grid.
        any_obs = next(iter(observations.values()))
        pending = list(any_obs["pending_orders"])
        num_reloc_pts = len(any_obs["relocation_points"])

        # Track remaining free capacity per driver this step.
        free_cap: Dict[int, int] = {}
        for did, obs in observations.items():
            s = obs["self"]
            free_cap[did] = s["capacity"] - s["onboard_passengers"]

        bids: Dict[int, List[int]] = {did: [] for did in observations}
        driver_ids = list(observations.keys())

        # One bidder per order: shuffle orders, pick a random capable driver.
        order_perm = self._rng.permutation(len(pending))
        for idx in order_perm:
            order = pending[idx]
            party = order["num_passengers"]
            capable = [d for d in driver_ids if free_cap[d] >= party]
            if not capable:
                continue
            chosen = capable[int(self._rng.integers(len(capable)))]
            bids[chosen].append(order["order_id"])
            free_cap[chosen] -= party

        actions: Dict[int, Dict] = {}
        for did, obs in observations.items():
            if bids[did]:
                actions[did] = {"orders": bids[did]}
                continue
            # No bid: idle drivers may relocate.
            if (
                obs["self"]["status"] == "idle"
                and num_reloc_pts > 0
                and self._rng.random() < self.relocate_prob
            ):
                idx = int(self._rng.integers(num_reloc_pts))
                actions[did] = {"relocate": idx}
            else:
                actions[did] = {"orders": []}
        return actions