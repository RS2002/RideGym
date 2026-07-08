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
        * ``assigned_party_sizes`` : dict order_id -> passenger (party) count
                                     of each newly assigned order.
        * ``assigned_service_times`` : dict order_id -> predicted END-TO-END
                                     service time (min) for each newly assigned
                                     order: from the user's request to the
                                     planned drop-off on the re-optimised route
                                     (platform/dispatch wait + remaining pickup
                                     wait + in-vehicle ride).
        * ``completed_orders``     : list of order ids dropped off this step.
        * ``picked_up_orders``     : list of order ids picked up this step.
        * ``distance_moved``       : coordinate units travelled this step.
        * ``time_moved``           : minutes spent moving this step.
        * ``is_empty_move``        : bool, moved with zero onboard passengers.
        * ``is_idle_wait``         : bool, stayed put with no tasks.
        * ``extra_detour_time``    : SIGNED minutes of re-routing impact on the
                                     driver's already-committed en-route orders
                                     (sum over them of new predicted drop-off
                                     time minus old). Positive = the new orders
                                     delayed existing deliveries; negative = the
                                     re-optimisation actually sped them up. 0 if
                                     the environment cannot attribute it.
        """
        raise NotImplementedError


class DefaultRewardFunction(RewardFunction):
    """Default sparse, multi-component shaping reward.

    Components (all configurable via constructor coefficients):

    * **assignment bonus** -- fixed positive reward per newly assigned order,
      encouraging drivers to serve demand. This is the immediate credit for the
      *acting* driver's *current* decision, so it is clean for credit
      assignment.
    * **revenue bonus** -- positive reward proportional to each newly assigned
      order's solo (direct pickup->dropoff) service time *times its passenger
      count*, modelling the fare the platform collects for that trip (ride-
      pooling is typically charged per passenger, so longer trips and larger
      parties earn more).
    * **service-time penalty** -- proportional to each newly assigned order's
      predicted END-TO-END service time (request -> planned drop-off, including
      the platform/dispatch wait, the remaining pickup wait and the in-vehicle
      ride), discouraging acceptance of orders that will take long to fulfil.
    * **detour penalty** -- proportional to the SIGNED re-routing impact on the
      driver's already-committed en-route orders (later deliveries are
      penalised; an earlier re-optimised delivery is rewarded via a negative
      penalty).
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
        revenue_coef: float = 0.1,
        service_time_coef: float = 0.01,
        detour_coef: float = 0.05,
        empty_move_penalty: float = 0.02,
        idle_penalty: float = 0.01,
    ):
        self.assignment_bonus = assignment_bonus
        # Fare the platform collects per order, modelled as proportional to the
        # order's solo (direct) service time. Longer trips pay more.
        self.revenue_coef = revenue_coef
        self.service_time_coef = service_time_coef
        self.detour_coef = detour_coef
        self.empty_move_penalty = empty_move_penalty
        self.idle_penalty = idle_penalty

    def __call__(self, driver_id: int, event: Dict) -> float:
        reward = 0.0

        assigned = event.get("assigned_orders", [])
        reward += self.assignment_bonus * len(assigned)

        # Revenue bonus: fare collected per order, proportional to its solo
        # (direct pickup->dropoff) service time AND its passenger count -- the
        # platform earns more on longer trips and on larger parties, since
        # ride-pooling fares are typically charged per passenger.
        solo_times = event.get("assigned_solo_times", {})
        party_sizes = event.get("assigned_party_sizes", {})
        for oid in assigned:
            party = party_sizes.get(oid, 1)
            reward += self.revenue_coef * solo_times.get(oid, 0.0) * party

        # Service-time penalty: predicted end-to-end fulfilment time per newly
        # assigned order (request -> planned drop-off), set by the env.
        service_times = event.get("assigned_service_times", {})
        for oid in assigned:
            reward -= self.service_time_coef * (service_times.get(oid, 0.0) - solo_times.get(oid, 0.0)) # the solo time could be absorbed into the revenue bonus, so we only penalise the extra wait/ride time.

        # Detour penalty: SIGNED re-routing impact on en-route orders (the env
        # already sums new-minus-old drop-off times across them, so a faster
        # re-plan yields a negative value that rewards the driver).
        detour_time = event.get("extra_detour_time", 0.0)
        reward -= self.detour_coef * detour_time

        if event.get("is_empty_move", False):
            reward -= self.empty_move_penalty
        if event.get("is_idle_wait", False):
            reward -= self.idle_penalty
        return reward