"""Reusable spatial index for fast nearest-driver queries.

Dispatch baselines repeatedly need 'the nearest available drivers to an order'.
Computing this against all drivers is O(orders * drivers) per step, which is
prohibitive at benchmark scale (1000 drivers, hundreds of pending orders).

This module provides a uniform :class:`GridIndex` that buckets driver locations
into square cells. A nearest query inspects the order's cell and expands ring by
ring outward, gathering candidates until enough have been found, so each query
typically touches only a handful of cells instead of every driver. The index is
cheap to rebuild each step (O(drivers)), which suits the fact that drivers move
every step.

The index is metric-agnostic for *bucketing* (it buckets on raw coordinates),
but ranks candidates using a supplied distance function, so it works for both
Manhattan and Euclidean networks and can later wrap a real road-network metric.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

Coord = Tuple[float, float]
Area = Tuple[float, float, float, float]


class GridIndex:
    """Uniform-grid bucket index over a rectangular service area.

    Parameters
    ----------
    area:
        Service-area bounds ``(xmin, ymin, xmax, ymax)``.
    cell_size:
        Side length of each square cell in coordinate units. A good default is
        roughly the per-step travel distance, so the nearest available driver to
        an order is usually within a few rings.
    """

    def __init__(self, area: Area, cell_size: float):
        if cell_size <= 0:
            raise ValueError("cell_size must be positive")
        self.area = area
        self.cell_size = float(cell_size)
        xmin, ymin, xmax, ymax = area
        self._xmin = xmin
        self._ymin = ymin
        self._ncols = max(1, int((xmax - xmin) / self.cell_size) + 1)
        self._nrows = max(1, int((ymax - ymin) / self.cell_size) + 1)
        # cell (col, row) -> list of (item_id, coord)
        self._cells: Dict[Tuple[int, int], List[Tuple[int, Coord]]] = {}

    # ----------------------------------------------------------- build
    def _cell_of(self, coord: Coord) -> Tuple[int, int]:
        col = int((coord[0] - self._xmin) / self.cell_size)
        row = int((coord[1] - self._ymin) / self.cell_size)
        # Clamp to valid range (points on the upper border, or any drift).
        col = min(max(col, 0), self._ncols - 1)
        row = min(max(row, 0), self._nrows - 1)
        return col, row

    def build(self, items: Dict[int, Coord]) -> None:
        """(Re)build the index from ``{item_id: coord}`` (e.g. driver locations)."""
        self._cells = {}
        for item_id, coord in items.items():
            key = self._cell_of(coord)
            self._cells.setdefault(key, []).append((item_id, coord))

    # ----------------------------------------------------------- query
    def nearest(
        self,
        query: Coord,
        k: int,
        distance_fn: Callable[[Coord, Coord], float],
        candidate_filter: Callable[[int], bool] = None,
    ) -> List[Tuple[float, int]]:
        """Return up to ``k`` nearest items as ``[(distance, item_id), ...]``.

        Expands outward ring by ring from the query's cell, counting only
        candidates that pass ``candidate_filter`` toward ``k`` (so filtered-out
        items, e.g. full drivers, do not prematurely stop the search). Once at
        least ``k`` *passing* candidates are collected it expands **one extra
        ring** before stopping to cover the cell-boundary case, then sorts and
        truncates. This is not a strict global nearest guarantee but is an
        accurate, fast approximation suitable for the nearest-distance baseline.

        Parameters
        ----------
        query:
            The query coordinate (e.g. an order's pickup location).
        k:
            Number of nearest items to return.
        distance_fn:
            Ranking metric ``(a, b) -> distance`` (network distance).
        candidate_filter:
            Optional predicate on ``item_id``; only items returning ``True`` are
            considered (e.g. drivers with free capacity).
        """
        if k <= 0:
            return []
        qcol, qrow = self._cell_of(query)
        collected: List[Tuple[float, int]] = []
        max_ring = max(self._ncols, self._nrows)
        extra_ring_budget = 1
        ring = 0
        while ring <= max_ring:
            ring_items = self._ring_items(qcol, qrow, ring)
            for item_id, coord in ring_items:
                if candidate_filter is not None and not candidate_filter(item_id):
                    continue
                collected.append((distance_fn(query, coord), item_id))

            if len(collected) >= k:
                # Expand one more ring to be safe near cell boundaries, then stop.
                if extra_ring_budget == 0:
                    break
                extra_ring_budget -= 1
            ring += 1

        collected.sort(key=lambda t: t[0])
        return collected[:k]

    def _ring_items(
        self, ccol: int, crow: int, ring: int
    ) -> List[Tuple[int, Coord]]:
        """All items in the square ring at Chebyshev radius ``ring`` from centre."""
        if ring == 0:
            return self._cells.get((ccol, crow), [])
        items: List[Tuple[int, Coord]] = []
        col_lo, col_hi = ccol - ring, ccol + ring
        row_lo, row_hi = crow - ring, crow + ring
        for col in range(col_lo, col_hi + 1):
            for row in range(row_lo, row_hi + 1):
                # Only the perimeter of the square (the ring), not the interior.
                on_perimeter = (
                    col == col_lo or col == col_hi or row == row_lo or row == row_hi
                )
                if not on_perimeter:
                    continue
                if 0 <= col < self._ncols and 0 <= row < self._nrows:
                    cell = self._cells.get((col, row))
                    if cell:
                        items.extend(cell)
        return items