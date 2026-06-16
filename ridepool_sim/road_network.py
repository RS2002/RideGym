"""Road network abstraction and lightweight default implementations.

The :class:`RoadNetwork` interface decouples path/distance computation from the
rest of the environment, enabling a drop-in OSMnx-based implementation later
without touching core logic. Two fast defaults are provided:

* :class:`EuclideanNetwork` - straight-line distance.
* :class:`ManhattanNetwork` - grid (L1) distance.

All methods return distances in coordinate units and travel times in minutes,
derived from a configurable constant ``speed`` (coordinate units per minute).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

Coord = Tuple[float, float]


@dataclass
class PathResult:
    """Result of a shortest-path query."""

    distance: float          # coordinate units
    travel_time: float       # minutes
    nodes: List[Coord]       # ordered coordinate sequence origin..destination


class RoadNetwork(ABC):
    """Abstract road network interface.

    Subclasses must implement :meth:`shortest_path`. The default ``speed`` is
    shared by all implementations so travel time is consistently derived from
    distance. Override :meth:`travel_time` if a network has edge-specific
    speeds (e.g. a real OSMnx graph).
    """

    def __init__(self, speed: float = 1.0):
        if speed <= 0:
            raise ValueError("speed must be positive")
        self.speed = float(speed)

    @abstractmethod
    def shortest_path(self, origin: Coord, destination: Coord) -> PathResult:
        """Return distance, travel time and node sequence between two points."""
        raise NotImplementedError

    def distance(self, origin: Coord, destination: Coord) -> float:
        """Distance only, between two points.

        The default delegates to :meth:`shortest_path`, but subclasses with a
        cheap closed-form metric should override this to avoid building a full
        :class:`PathResult` (node list + travel time) on the hot path; dispatch
        matching calls this millions of times per episode.
        """
        return self.shortest_path(origin, destination).distance

    def travel_time(self, distance: float) -> float:
        """Convert a distance into minutes using the network speed."""
        return distance / self.speed


class EuclideanNetwork(RoadNetwork):
    """Straight-line distance network. Fast, suitable for large-scale RL."""

    def distance(self, origin: Coord, destination: Coord) -> float:
        # Hot-path: skip PathResult construction.
        dx = destination[0] - origin[0]
        dy = destination[1] - origin[1]
        return math.sqrt(dx * dx + dy * dy)

    def shortest_path(self, origin: Coord, destination: Coord) -> PathResult:
        dist = math.dist(origin, destination)
        return PathResult(
            distance=dist,
            travel_time=self.travel_time(dist),
            nodes=[tuple(origin), tuple(destination)],
        )


class ManhattanNetwork(RoadNetwork):
    """Grid (L1) distance network.

    The returned node sequence routes first along x then along y, giving an
    L-shaped path consistent with the Manhattan distance metric.
    """

    def distance(self, origin: Coord, destination: Coord) -> float:
        # Hot-path: skip PathResult construction.
        return abs(destination[0] - origin[0]) + abs(destination[1] - origin[1])

    def shortest_path(self, origin: Coord, destination: Coord) -> PathResult:
        (ox, oy), (dx, dy) = origin, destination
        dist = abs(dx - ox) + abs(dy - oy)
        corner = (dx, oy)  # travel along x first, then y
        nodes = [tuple(origin), corner, tuple(destination)]
        # Collapse degenerate corner if movement is purely along one axis.
        nodes = [p for i, p in enumerate(nodes) if i == 0 or p != nodes[i - 1]]
        return PathResult(
            distance=dist,
            travel_time=self.travel_time(dist),
            nodes=nodes,
        )