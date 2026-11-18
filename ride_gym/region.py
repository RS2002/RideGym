"""Opt-in road-network size and geographic coverage checks.

This module never downloads a graph. Bounding boxes use ``(xmin, ymin, xmax,
ymax)``. Geographic helpers require longitude/latitude coordinates in degrees;
they report distances in kilometres or metres as stated by each function.
"""

from __future__ import annotations

import math
from numbers import Integral

import numpy as np

_EARTH_RADIUS_KM = 6371.0088


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _finite(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _probability(value, name):
    result = _finite(value, name)
    if not 0 <= result <= 1:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _points(points, *, geographic=False):
    try:
        rows = list(points)
        if not rows:
            raise ValueError("At least one point is required")
        parsed = []
        for row in rows:
            if isinstance(row, (str, bytes)) or len(row) != 2:
                raise ValueError("Each point must contain two coordinates")
            x, y = (_finite(value, "coordinate") for value in row)
            if geographic and not (-180 <= x <= 180 and -90 <= y <= 90):
                raise ValueError("Geographic coordinates must be (longitude, latitude) in degrees")
            parsed.append((x, y))
    except TypeError as exc:
        raise ValueError("Points must be an iterable of coordinate pairs") from exc
    return parsed


def estimate_matrix_bytes(num_nodes):
    """Return ``num_nodes**2 * 12`` bytes for float64/int32 dense matrices.

    This covers the distance and predecessor arrays only, excluding the graph,
    temporary path-search structures and cache-loading copies.
    """
    n = _integer(num_nodes, "num_nodes")
    return n * n * 12


class NodeBudget:
    """An explicit byte ceiling for the two dense all-pairs routing arrays."""

    def __init__(self, max_bytes):
        self.max_bytes = _integer(max_bytes, "max_bytes")

    def max_nodes(self):
        """Largest integer node count whose two arrays fit the budget."""
        return math.isqrt(self.max_bytes // 12)

    def check(self, num_nodes, what="network"):
        """Raise ``MemoryError`` before matrix allocation if the budget is exceeded."""
        needed = estimate_matrix_bytes(num_nodes)
        if needed > self.max_bytes:
            raise MemoryError(
                f"{what} needs {needed:,} bytes for its {num_nodes:,}-node routing "
                f"matrices; budget is {self.max_bytes:,} bytes "
                f"({self.max_nodes():,} nodes maximum)"
            )


def bbox_of_points(points, pad_frac=0.02):
    """Return a finite bounding box in the input coordinate units.

    Each axis expands by ``pad_frac`` times its span at both ends. A degenerate
    axis keeps zero span; this helper does not invent a unit-dependent padding.
    ``pad_frac=0`` returns the exact extrema.
    """
    points = _points(points)
    pad = _finite(pad_frac, "pad_frac")
    if pad < 0:
        raise ValueError("pad_frac must be nonnegative")
    xs, ys = zip(*points)
    left, right, bottom, top = min(xs), max(xs), min(ys), max(ys)
    if pad == 0:
        return left, bottom, right, top
    dx, dy = (right - left) * pad, (top - bottom) * pad
    result = (left - dx, bottom - dy, right + dx, top + dy)
    if not all(math.isfinite(value) for value in result):
        raise ValueError("Padded bounding box is not finite")
    return result


def _geographic_box(bbox):
    try:
        if isinstance(bbox, (str, bytes)) or len(bbox) != 4:
            raise ValueError("bbox must contain xmin, ymin, xmax, ymax")
        left, bottom, right, top = (_finite(value, "bbox coordinate") for value in bbox)
    except TypeError as exc:
        raise ValueError("bbox must contain xmin, ymin, xmax, ymax") from exc
    if not (-180 <= left <= right <= 180 and -90 <= bottom <= top <= 90):
        raise ValueError("bbox must be ordered longitude/latitude bounds in degrees")
    return left, bottom, right, top


def bbox_center_radius(lonlat_bbox):
    """Return ``(center_lat, center_lon, enclosing_radius_m)`` on a sphere.

    Input order is ``(lon_min, lat_min, lon_max, lat_max)`` in degrees. The
    radius reaches the farthest point of that geographic rectangle. Bounds
    crossing the antimeridian must first be represented as separate boxes.
    """
    left, bottom, right, top = _geographic_box(lonlat_bbox)
    center_lat, center_lon = (bottom + top) / 2, (left + right) / 2
    latitude = math.radians(center_lat)
    half_longitude = math.radians((right - left) / 2)
    # The cosine of angular distance is A*sin(phi) + B*cos(phi).
    # Its minimum over the latitude interval occurs at an endpoint or a
    # stationary point. This also handles wide rectangles near the poles.
    a = math.sin(latitude)
    b = math.cos(latitude) * math.cos(half_longitude)
    low, high = math.radians(bottom), math.radians(top)
    candidates = [low, high]
    stationary = math.atan2(a, b)
    for offset in (-math.pi, 0.0, math.pi):
        candidate = stationary + offset
        if low <= candidate <= high:
            candidates.append(candidate)
    minimum = min(a * math.sin(phi) + b * math.cos(phi) for phi in candidates)
    angular = math.acos(max(-1.0, min(1.0, minimum)))
    return center_lat, center_lon, _EARTH_RADIUS_KM * 1000 * angular


def check_coverage(network, points, max_snap_km=1.0, min_share=0.95, sample=2000):
    """Measure geographic point-to-nearest-node coverage without raising on a miss.

    ``network.node_coords`` and ``points`` must be longitude/latitude degrees.
    Distances use a spherical great-circle calculation, not the network's
    coordinate-unit or path-distance metric. At most ``sample`` evenly spaced
    input points are checked, deterministically and without consuming an RNG.
    Empty or malformed input raises; use :func:`require_coverage` to reject an
    otherwise valid report below the required share.
    """
    maximum = _finite(max_snap_km, "max_snap_km")
    if maximum < 0:
        raise ValueError("max_snap_km must be nonnegative")
    minimum = _probability(min_share, "min_share")
    count = _integer(sample, "sample")
    if count == 0:
        raise ValueError("sample must be a positive integer")
    points = _points(points, geographic=True)
    nodes = np.asarray(_points(network.node_coords, geographic=True), dtype=float)
    selected = np.linspace(0, len(points) - 1, min(count, len(points)), dtype=int)
    node_lon, node_lat = np.radians(nodes[:, 0]), np.radians(nodes[:, 1])
    distances = []
    for index in selected:
        lon, lat = map(math.radians, points[index])
        haversine = (
            np.sin((node_lat - lat) / 2) ** 2
            + math.cos(lat) * np.cos(node_lat) * np.sin((node_lon - lon) / 2) ** 2
        )
        angular = 2 * math.asin(math.sqrt(float(np.clip(haversine.min(), 0.0, 1.0))))
        distances.append(_EARTH_RADIUS_KM * angular)
    distances.sort()
    within = sum(distance <= maximum for distance in distances)
    share = within / len(distances)
    return {
        "checked": len(distances),
        "within": within,
        "share": share,
        "worst_km": distances[-1],
        "p95_km": float(np.quantile(distances, 0.95)),
        "max_snap_km": maximum,
        "min_share": minimum,
        "passed": share >= minimum,
    }


def require_coverage(report, min_share=0.95):
    """Raise ``ValueError`` unless a nonempty coverage report reaches ``min_share``."""
    minimum = _probability(min_share, "min_share")
    try:
        checked = _integer(report["checked"], "checked")
        within = _integer(report["within"], "within")
        share = _probability(report["share"], "share")
    except (KeyError, TypeError) as exc:
        raise ValueError("Invalid coverage report") from exc
    if checked == 0 or within > checked or not math.isclose(share, within / checked):
        raise ValueError("Coverage report counts are inconsistent or empty")
    if share < minimum:
        raise ValueError(
            f"Only {share:.1%} of {checked} checked points meet the snap-distance "
            f"threshold; at least {minimum:.1%} is required"
        )
