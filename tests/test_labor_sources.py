import csv
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

import numpy as np

from ride_gym.entities import Driver, Order
from ride_gym.enums import DriverStatus, OrderStatus
from ride_gym.labor_source import (
    AlwaysOnShiftPolicy,
    CsvLaborSource,
    ExplicitLaborSource,
    LegacyLaborSource,
    ShiftWindow,
    WindowShiftPolicy,
)
from ride_gym.road_network import EuclideanNetwork


class LaborSourceTests(unittest.TestCase):
    def setUp(self):
        self.network = EuclideanNetwork()
        self.area = (0.0, 0.0, 10.0, 10.0)

    def test_original_entity_positional_arguments_retain_meaning(self):
        order = Order(7, (0, 0), (1, 1), 3, 2, OrderStatus.ASSIGNED, 4, 5, 6, 7, 8)
        self.assertEqual(
            (order.status, order.assigned_driver, order.eta), (OrderStatus.ASSIGNED, 4, 8)
        )
        self.assertIsNone(order.ready_time)
        driver = Driver(
            4, (1, 2), 3, 2, DriverStatus.TO_DROPOFF, 1, [7], [], [], None, 8, 9, 10, 11
        )
        self.assertEqual(
            (driver.node_idx, driver.edge_from, driver.edge_to, driver.edge_pos_m), (8, 9, 10, 11)
        )
        self.assertTrue(driver.on_duty)
        self.assertEqual(
            [field.name for field in fields(Driver)][:14],
            [
                "driver_id",
                "location",
                "capacity",
                "speed",
                "status",
                "onboard_passengers",
                "assigned_orders",
                "task_points",
                "route_nodes",
                "relocation_target",
                "node_idx",
                "edge_from",
                "edge_to",
                "edge_pos_m",
            ],
        )

    def test_explicit_fleet_uses_native_units_and_fresh_drivers(self):
        source = ExplicitLaborSource(
            [
                {"x": 2, "y": 3, "capacity": 2, "speed": 0.5, "vehicle_type": "cycle"},
                {"lon": 4, "lat": 5, "pickup_service_time": 1.5},
            ],
            default_capacity=3,
            default_speed=2,
        )
        first = source.build(2, None, self.area, self.network)
        first[0].assigned_orders.append(99)
        second = source.build(2, None, self.area, self.network)
        self.assertEqual(second[0].location, (2, 3))
        self.assertEqual(
            (second[0].capacity, second[0].speed, second[0].vehicle_type), (2, 0.5, "cycle")
        )
        self.assertEqual(
            (second[1].capacity, second[1].speed, second[1].pickup_service_time), (3, 2, 1.5)
        )
        self.assertEqual(second[0].assigned_orders, [])
        with self.assertRaises(ValueError):
            source.build(3, None, self.area, self.network)

    def test_invalid_optional_inputs_fail_explicitly(self):
        for changes in (
            {"x": float("nan")},
            {"capacity": -1},
            {"capacity": 1.5},
            {"speed": 0},
            {"speed": float("inf")},
            {"pickup_service_time": -1},
            {"dropoff_service_time": float("nan")},
            {"shift_start": -1},
            {"shift_end": 0},
            {"shift_start": 5, "shift_end": 5},
            {"vehicle_type": 2},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ExplicitLaborSource([{"x": 0, "y": 0, **changes}])

    def test_windows_validate_and_normalize(self):
        source = ExplicitLaborSource(
            [
                {
                    "x": 0,
                    "y": 0,
                    "availability_windows": [(20, 30), {"start": 0, "end": 10, "location": [2, 3]}],
                }
            ]
        )
        driver = source.build(1, None, self.area, self.network)[0]
        self.assertEqual(
            driver.availability_windows, (ShiftWindow(0, 10, (2, 3)), ShiftWindow(20, 30))
        )
        for windows in ([(0, 10), (9, 20)], [(0, 10), (0, 10)], [(10, 10)], [(0, float("inf"))]):
            with self.subTest(windows=windows), self.assertRaises(ValueError):
                ExplicitLaborSource([{"x": 0, "y": 0, "availability_windows": windows}])
        with self.assertRaises(ValueError):
            ExplicitLaborSource(
                [{"x": 0, "y": 0, "shift_start": 0, "availability_windows": [(0, 10)]}]
            )

    def test_shift_stops_new_work_but_drains_commitments(self):
        driver = Driver(0, (0, 0), shift_start=2, shift_end=5)
        policy = WindowShiftPolicy()
        policy.update([driver], 0)
        self.assertFalse(policy.is_available(driver))
        self.assertFalse(driver.departed)
        policy.update([driver], 2)
        self.assertTrue(policy.is_available(driver))
        driver.assigned_orders.append(1)
        policy.update([driver], 5)
        self.assertFalse(driver.on_duty)
        self.assertFalse(driver.departed)
        self.assertEqual(driver.assigned_orders, [1])
        driver.assigned_orders.clear()
        policy.update([driver], 6)
        self.assertTrue(driver.departed)

    def test_later_window_reenters_and_applies_location_only_when_idle(self):
        driver = Driver(
            0, (0, 0), availability_windows=(ShiftWindow(2, 5, (1, 2)), ShiftWindow(10, 15, (8, 9)))
        )
        policy = WindowShiftPolicy()
        policy.update([driver], 0)
        self.assertEqual(driver.location, (0, 0))
        policy.update([driver], 2)
        self.assertEqual(driver.location, (1, 2))
        policy.update([driver], 5)
        self.assertTrue(driver.departed)
        driver.assigned_orders = [1]
        driver.location = (4, 4)
        policy.update([driver], 10)
        self.assertTrue(policy.is_available(driver))
        self.assertEqual(driver.location, (4, 4))
        driver.assigned_orders.clear()
        policy.update([driver], 11)
        self.assertEqual(driver.location, (4, 4))

    def test_entry_location_resets_graph_motion_state(self):
        driver = Driver(
            0,
            (0, 0),
            node_idx=1,
            edge_from=1,
            edge_to=2,
            edge_pos_m=3,
            route_nodes=[(1, 1)],
            relocation_target=(2, 2),
            availability_windows=(ShiftWindow(0, 5, (4, 5)),),
        )
        WindowShiftPolicy().update([driver], 0)
        self.assertEqual(driver.location, (4, 5))
        self.assertEqual((driver.node_idx, driver.edge_from, driver.edge_to), (None, None, None))
        self.assertEqual(
            (driver.edge_pos_m, driver.route_nodes, driver.relocation_target), (0, [], None)
        )

    def test_restored_active_window_does_not_apply_entry_location_again(self):
        driver = Driver(
            0, (3, 4), availability_windows=(ShiftWindow(0, 10, (0, 0)),), active_shift_window=0
        )
        WindowShiftPolicy().update([driver], 5)
        self.assertEqual(driver.location, (3, 4))

    def test_service_in_progress_prevents_entry_teleport_and_departure(self):
        driver = Driver(
            0, (3, 4), availability_windows=(ShiftWindow(2, 5, (0, 0)),), service_until=7
        )
        policy = WindowShiftPolicy()
        policy.update([driver], 2)
        self.assertEqual(driver.location, (3, 4))
        policy.update([driver], 5)
        self.assertFalse(driver.departed)
        driver.service_until = None
        policy.update([driver], 7)
        self.assertTrue(driver.departed)

    def test_pending_invitation_prevents_entry_teleport_and_departure(self):
        driver = Driver(
            0, (3, 4), availability_windows=(ShiftWindow(2, 5, (0, 0)),), awaiting_response=True
        )
        policy = WindowShiftPolicy()
        policy.update([driver], 2)
        self.assertEqual(driver.location, (3, 4))
        policy.update([driver], 5)
        self.assertFalse(driver.departed)
        driver.awaiting_response = False
        policy.update([driver], 7)
        self.assertTrue(driver.departed)

    def test_csv_reads_public_schema_and_window_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fleet.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream, ["x", "y", "speed", "capacity", "availability_windows"]
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "x": 1,
                        "y": 2,
                        "speed": "",
                        "capacity": 3,
                        "availability_windows": '[{"start": 2, "end": 5}]',
                    }
                )
            driver = CsvLaborSource(path).build(1, None, self.area, self.network)[0]
        self.assertEqual((driver.location, driver.capacity, driver.speed), ((1, 2), 3, None))
        self.assertEqual(driver.availability_windows, (ShiftWindow(2, 5),))

    def test_legacy_source_matches_original_rng_consumption(self):
        rng = np.random.default_rng(12)
        expected = [(float(rng.uniform(0, 10)), float(rng.uniform(0, 10))) for _ in range(3)]
        source = LegacyLaborSource(capacity=3, speeds=[None, 2, None])
        actual_rng = np.random.default_rng(12)
        drivers = source.build(3, actual_rng, self.area, self.network)
        self.assertEqual([driver.location for driver in drivers], expected)
        self.assertEqual(actual_rng.random(), rng.random())
        self.assertEqual([driver.capacity for driver in drivers], [3, 3, 3])

    def test_always_on_and_unscheduled_drivers(self):
        driver = Driver(0, (0, 0), on_duty=False, departed=True)
        AlwaysOnShiftPolicy().update([driver], 2)
        self.assertTrue(driver.on_duty)
        self.assertFalse(driver.departed)
        WindowShiftPolicy().update([driver], 100)
        self.assertTrue(driver.on_duty)


if __name__ == "__main__":
    unittest.main()
