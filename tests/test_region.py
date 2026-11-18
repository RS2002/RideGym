"""Synthetic memory-budget and geographic coverage checks."""

from __future__ import annotations

import importlib.util
import inspect
import math
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ride_gym.region import (
    NodeBudget,
    bbox_center_radius,
    bbox_of_points,
    check_coverage,
    estimate_matrix_bytes,
    require_coverage,
)


class RegionTests(unittest.TestCase):
    def test_matrix_estimate_and_exact_budget_boundaries(self):
        self.assertEqual(estimate_matrix_bytes(0), 0)
        self.assertEqual(estimate_matrix_bytes(5), 300)
        budget = NodeBudget(300)
        self.assertEqual(budget.max_nodes(), 5)
        budget.check(5)
        with self.assertRaises(MemoryError):
            budget.check(6)
        self.assertEqual(NodeBudget(0).max_nodes(), 0)
        NodeBudget(0).check(0)

    def test_matrix_counts_and_budgets_are_strict(self):
        for value in (True, -1, 2.5, float("inf"), "4"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    estimate_matrix_bytes(value)
                with self.assertRaises(ValueError):
                    NodeBudget(value)

    def test_bbox_padding_and_degenerate_coordinates(self):
        np.testing.assert_allclose(bbox_of_points([(1, 2), (3, 6)], 0.1), (0.8, 1.6, 3.2, 6.4))
        self.assertEqual(bbox_of_points([(2, 4)], 0), (2, 4, 2, 4))
        self.assertEqual(bbox_of_points([(2, 4)]), (2, 4, 2, 4))
        self.assertEqual(bbox_of_points([(-1e308, 0), (1e308, 0)], 0), (-1e308, 0, 1e308, 0))
        for points in ([], [(float("nan"), 1)], [(0,)], ["12"], [(False, 1)]):
            with self.subTest(points=points), self.assertRaises(ValueError):
                bbox_of_points(points)
        for padding in (-1, float("inf"), True):
            with self.assertRaises(ValueError):
                bbox_of_points([(0, 0)], padding)

    def test_geographic_radius_has_explicit_metres_and_coordinate_order(self):
        lat, lon, radius = bbox_center_radius((-1, 0, 1, 0))
        self.assertEqual((lat, lon), (0, 0))
        self.assertAlmostEqual(radius, math.radians(1) * 6371008.8, places=5)
        self.assertEqual(bbox_center_radius((0, 0, 0, 0)), (0, 0, 0))
        # A wide equatorial rectangle includes the antipode, not only corners.
        self.assertAlmostEqual(
            bbox_center_radius((-180, -90, 180, 90))[2], math.pi * 6371008.8, places=5
        )

    def test_invalid_geographic_boxes_are_rejected(self):
        for box in (
            (1, 0, -1, 1),
            (-181, 0, 0, 0),
            (0, -91, 1, 0),
            (0, 0, 1),
            (0, 0, float("inf"), 1),
        ):
            with self.subTest(box=box), self.assertRaises(ValueError):
                bbox_center_radius(box)

    def test_coverage_reports_geographic_snap_distances(self):
        network = SimpleNamespace(node_coords=np.array([(0, 0), (0.01, 0)]))
        report = check_coverage(
            network, [(0, 0), (0.005, 0), (1, 1)], max_snap_km=0.75, min_share=0.6
        )
        self.assertEqual((report["checked"], report["within"]), (3, 2))
        self.assertAlmostEqual(report["share"], 2 / 3)
        self.assertTrue(report["passed"])
        self.assertGreater(report["worst_km"], 100)
        require_coverage(report, min_share=0.6)
        with self.assertRaises(ValueError):
            require_coverage(report)

    def test_coverage_handles_longitude_wrap_and_deterministic_sampling(self):
        network = SimpleNamespace(node_coords=[(179.99, 0)])
        report = check_coverage(network, [(-179.99, 0)], max_snap_km=3)
        self.assertTrue(report["passed"])
        self.assertAlmostEqual(report["worst_km"], 2.2239016, places=5)
        network = SimpleNamespace(node_coords=[(0, 0)])
        points = [(i / 1000, 0) for i in range(20)]
        first = check_coverage(network, points, sample=4)
        second = check_coverage(network, points, sample=4)
        self.assertEqual(first, second)
        self.assertEqual(first["checked"], 4)

    def test_empty_and_invalid_coverage_inputs_do_not_claim_success(self):
        network = SimpleNamespace(node_coords=[(0, 0)])
        for points in ([], [(181, 0)], [(0, -91)], [(0, float("nan"))]):
            with self.subTest(points=points), self.assertRaises(ValueError):
                check_coverage(network, points)
        for kwargs in (
            {"sample": 0},
            {"sample": -1},
            {"sample": True},
            {"min_share": 1.1},
            {"max_snap_km": -1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                check_coverage(network, [(0, 0)], **kwargs)
        with self.assertRaises(ValueError):
            check_coverage(SimpleNamespace(node_coords=[]), [(0, 0)])
        with self.assertRaises(ValueError):
            require_coverage({"checked": 0, "within": 0, "share": 1})
        with self.assertRaises(ValueError):
            require_coverage({"checked": 1, "within": 2, "share": 1})


@unittest.skipUnless(importlib.util.find_spec("networkx"), "networkx extra is not installed")
class OptionalGraphBudgetTests(unittest.TestCase):
    def make_graph(self, folder):
        import networkx as nx

        graph = nx.MultiDiGraph()
        graph.add_node(0, x=0.0, y=0.0)
        graph.add_node(1, x=0.001, y=0.0)
        graph.add_edge(0, 1, length=100.0)
        graph.add_edge(1, 0, length=130.0)
        path = Path(folder) / "synthetic.gpickle"
        with path.open("wb") as stream:
            pickle.dump(graph, stream)
        return str(path)

    def test_explicit_budget_is_checked_before_loading_or_allocating_matrices(self):
        from ride_gym.osmnx_network import OSMnxNetwork

        with tempfile.TemporaryDirectory() as folder:
            path = self.make_graph(folder)
            with patch.object(OSMnxNetwork, "_load_or_build_matrices") as allocate:
                with self.assertRaises(MemoryError):
                    OSMnxNetwork(path, cache_matrices=False, max_matrix_bytes=47)
                allocate.assert_not_called()

    def test_default_and_sufficient_budget_have_identical_routing(self):
        from ride_gym.osmnx_network import OSMnxNetwork

        with tempfile.TemporaryDirectory() as folder:
            path = self.make_graph(folder)
            original = OSMnxNetwork(path, 60, True, False, None)
            limited = OSMnxNetwork(path, cache_matrices=False, max_matrix_bytes=48)
            np.testing.assert_array_equal(original._dist_m, limited._dist_m)
            np.testing.assert_array_equal(original._pred, limited._pred)
            self.assertEqual(original.node_path(0, 1), limited.node_path(0, 1))
        parameter = inspect.signature(OSMnxNetwork).parameters["max_matrix_bytes"]
        self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIsNone(parameter.default)


if __name__ == "__main__":
    unittest.main()
