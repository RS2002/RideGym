"""Reward functions.

The environment computes a per-step ``info`` event log for every driver (orders
assigned, pickups/drop-offs completed, idle/empty movement, etc.). A
:class:`RewardFunction` maps that event log into a scalar per-driver reward.

Users may fully override the reward by passing a custom callable / subclass to
the environment. The default is a sparse, multi-component shaping reward.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict


class RewardFunction(ABC):
    """Abstract per-driver reward.

    Implementations receive the rich per-driver event dict assembled by the
    environment for the current step and must return a float reward.
    """

    @abstractmethod
    def __call__(self, driver_id: int, event: Dict) -> float:
        """Return the scalar reward for ``driver_id`` given its step ``event``.

        The ``event`` dict contains (keys always present, values may be empty):

        * ``assigned_orders``      : list of order ids newly assigned this step.
        * ``assigned_solo_times``  : dict order_id -> solo service time (min),
                                     i.e. direct pickup->dropoff travel time.
        * ``completed_orders``     : list of order ids dropped off this step.
        * ``picked_up_orders``     : list of order ids picked up this step.
        * ``distance_moved``       : coordinate units travelled this step.
        * ``time_moved``           : minutes spent moving this step.
        * ``is_empty_move``        : bool, moved with zero onboard passengers.
        * ``is_idle_wait``         : bool, stayed put with no tasks.
        * ``extra_detour_time``    : minutes of pooling-induced detour (0 if the
                                     environment cannot attribute it).
        """
        raise NotImplementedError


class DefaultRewardFunction(RewardFunction):
    """Default sparse, multi-component shaping reward.

    Components (all configurable via constructor coefficients):

    * **assignment bonus** -- fixed positive reward per newly assigned order,
      encouraging drivers to serve demand. This is the immediate credit for the
      *acting* driver's *current* decision, so it is clean for credit
      assignment.
    * **service-time penalty** -- proportional to each newly assigned order's
      solo (direct) service time, discouraging acceptance of very far orders.
    * **detour penalty** -- proportional to pooling-induced extra travel time.
    * **empty-move penalty** -- small negative reward for moving while empty.
    * **idle penalty** -- small negative reward for waiting in place.

    Note: there is deliberately NO completion bonus. A drop-off is the
    consequence of an assignment decision made many steps earlier, so rewarding
    it on the delivery step mis-attributes the credit to whatever (usually
    no-op) action the driver happened to take then, which interferes with
    learning. The long-run value of serving an order is instead propagated back
    to the assignment action through the TD bootstrap.
    """

    def __init__(
        self,
        assignment_bonus: float = 1.0,
        service_time_coef: float = 0.01,
        detour_coef: float = 0.05,
        empty_move_penalty: float = 0.02,
        idle_penalty: float = 0.01,
    ):
        self.assignment_bonus = assignment_bonus
        self.service_time_coef = service_time_coef
        self.detour_coef = detour_coef
        self.empty_move_penalty = empty_move_penalty
        self.idle_penalty = idle_penalty

    def __call__(self, driver_id: int, event: Dict) -> float:
        reward = 0.0

        assigned = event.get("assigned_orders", [])
        reward += self.assignment_bonus * len(assigned)

        solo_times = event.get("assigned_solo_times", {})
        for oid in assigned:
            reward -= self.service_time_coef * solo_times.get(oid, 0.0)

        reward -= self.detour_coef * event.get("extra_detour_time", 0.0)

        if event.get("is_empty_move", False):
            reward -= self.empty_move_penalty
        if event.get("is_idle_wait", False):
            reward -= self.idle_penalty

        return reward