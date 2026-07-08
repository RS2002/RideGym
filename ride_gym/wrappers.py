"""Centralised control wrapper.

Wraps the decentralised :class:`RidePoolEnv` into a single-agent view for global
optimisation research. A central policy emits all drivers' actions at once; the
wrapper still relies on the base environment's strict conflict detection, so a
central policy that produces conflicting bids will likewise raise
:class:`ConflictError`.

Reward is reported at two levels:

* ``global_reward`` -- aggregate (sum or mean) of per-driver rewards, the
  single-agent optimisation target.
* ``individual_rewards`` -- the per-driver reward dict, returned in ``info`` for
  fairness / revenue-distribution analysis.
"""

from __future__ import annotations

from typing import Dict

from ride_gym.env import RidePoolEnv


class CentralizedWrapper:
    """Single-agent view over the multi-agent ride-pooling environment."""

    def __init__(self, env: RidePoolEnv, aggregate: str = "sum"):
        """Wrap ``env``.

        Parameters
        ----------
        env:
            The base decentralised environment.
        aggregate:
            How to combine per-driver rewards into ``global_reward``: ``"sum"``
            or ``"mean"``.
        """
        if aggregate not in ("sum", "mean"):
            raise ValueError("aggregate must be 'sum' or 'mean'")
        self.env = env
        self.aggregate = aggregate

    def reset(self, seed=None):
        """Reset and return ``(observations, info)`` (full per-driver obs dict)."""
        return self.env.reset(seed=seed)

    def _aggregate(self, rewards: Dict[int, float]) -> float:
        if not rewards:
            return 0.0
        total = sum(rewards.values())
        return total if self.aggregate == "sum" else total / len(rewards)

    def step(self, joint_action: Dict[int, Dict]):
        """Apply a joint action dict ``{driver_id: action}`` for all drivers.

        Returns ``(observations, global_reward, done, info)``. ``info`` carries
        ``individual_rewards`` (the per-driver reward dict) plus the base info.
        """
        obs, rewards, dones, info = self.env.step(joint_action)
        global_reward = self._aggregate(rewards)
        done = dones.get("__all__", all(dones.values()))
        info = {**info, "individual_rewards": rewards}
        return obs, global_reward, done, info

    def __getattr__(self, name):
        # Transparently expose base-env attributes (area, drivers, etc.).
        return getattr(self.env, name)