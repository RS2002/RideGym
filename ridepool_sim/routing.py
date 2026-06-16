"""Task-point sequencing (multi-order route planning).

When a driver holds several pooled orders, the environment must decide the
order in which to visit all pickup/drop-off stops. The hard constraint is
*pickup-before-dropoff* for every order. Within that constraint we minimise the
total travel distance.

The job is owned by the simulator (per the project spec) but exposed through the
:class:`RoutesPlanner` interface so users can later inject their own algorithm
(e.g. a constrained TSP solver).

The default :class:`GreedyInsertionPlanner` re-optimises from scratch with a
nearest-feasible-stop heuristic: at each step it greedily visits the closest
stop that is currently *feasible* (a drop-off is feasible only after its pickup
has been visited). This is fast and adequate for typical pool sizes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from ridepool_sim.entities import TaskPoint
from ridepool_sim.road_network import RoadNetwork

Coord = Tuple[float, float]


class RoutesPlanner(ABC):
    """Abstract task-point sequencer."""

    @abstractmethod
    def plan(
        self,
        start: Coord,
        task_points: List[TaskPoint],
        network: RoadNetwork,
    ) -> List[TaskPoint]:
        """Return a reordered task-point list satisfying precedence.

        Parameters
        ----------
        start:
            The driver's current location.
        task_points:
            Unordered remaining stops. May contain pickups whose dropoff is
            also present, or lone dropoffs (order already picked up).
        network:
            Road network used to measure distances between stops.
        """
        raise NotImplementedError


class GreedyInsertionPlanner(RoutesPlanner):
    """Nearest-feasible-stop heuristic with precedence enforcement."""

    def plan(
        self,
        start: Coord,
        task_points: List[TaskPoint],
        network: RoadNetwork,
    ) -> List[TaskPoint]:
        ordered, _times = self.plan_with_times(start, task_points, network)
        return ordered

    def plan_with_times(
        self,
        start: Coord,
        task_points: List[TaskPoint],
        network: RoadNetwork,
    ) -> Tuple[List[TaskPoint], "Dict[int, Dict[str, float]]"]:
        """Plan a route AND report each stop's arrival time, for free.

        Identical greedy sequencing as :meth:`plan`, but additionally returns
        ``{order_id: {"pickup": t, "dropoff": t}}`` giving the cumulative travel
        time (minutes from ``start``) at which each scheduled stop is reached.

        These arrival times are obtained at ZERO extra cost: the greedy loop
        already computes ``network.distance(current, candidate)`` to pick the
        nearest feasible stop, so the distance of the *selected* leg is simply
        accumulated as we go rather than thrown away and recomputed by a second
        pass over the finished route. This lets the environment attribute
        pooling detour without an additional route traversal.
        """
        remaining = list(task_points)
        # Orders for which a pickup stop is still pending in this plan.
        pending_pickup = {
            tp.order_id for tp in remaining if tp.kind == "pickup"
        }
        ordered: List[TaskPoint] = []
        times: Dict[int, Dict[str, float]] = {}
        current = start
        elapsed = 0.0

        while remaining:
            feasible = [
                tp
                for tp in remaining
                if tp.kind == "pickup" or tp.order_id not in pending_pickup
            ]
            # A dropoff is feasible only once its pickup has been scheduled.
            if not feasible:
                # Should never happen with well-formed input, but guard anyway:
                # fall back to any pickup to make progress.
                feasible = [tp for tp in remaining if tp.kind == "pickup"]

            # Distance to every feasible candidate is computed here to choose
            # the nearest; capture the winner's distance so the selected leg is
            # accounted for without a second pass.
            best = None
            best_dist = float("inf")
            for tp in feasible:
                d = network.distance(current, tp.location)
                if d < best_dist:
                    best_dist = d
                    best = tp
            nxt = best
            elapsed += network.travel_time(best_dist)
            ordered.append(nxt)
            remaining.remove(nxt)
            current = nxt.location
            times.setdefault(nxt.order_id, {})[nxt.kind] = elapsed
            if nxt.kind == "pickup":
                pending_pickup.discard(nxt.order_id)

        return ordered, times

    def plan_distance(
        self,
        start: Coord,
        ordered: List[TaskPoint],
        network: RoadNetwork,
    ) -> float:
        """Total travel distance of an already-ordered task list from ``start``."""
        total = 0.0
        current = start
        for tp in ordered:
            total += network.distance(current, tp.location)
            current = tp.location
        return total