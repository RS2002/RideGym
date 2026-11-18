import builtins
import csv
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from ride_gym.data_tools.csv_io import read_order_csv
from ride_gym.enums import OrderStatus
from ride_gym.order_generator import CsvOrderGenerator, DataFrameOrderGenerator


class CsvOrderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "orders.csv"
        self.basic = {"origin_x": 0, "origin_y": 1, "dest_x": 2, "dest_y": 3, "request_time": 2}

    def write(self, rows, fields=None):
        if fields is None:
            fields = list(dict.fromkeys(key for row in rows for key in row))
        with self.path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_relative_minutes_defaults_sort_and_fresh_lifecycle(self):
        self.write([self.basic, {**self.basic, "request_time": 1}])
        generator = CsvOrderGenerator(self.path)
        first = generator.generate()
        self.assertEqual([order.order_id for order in first], [1, 0])
        self.assertEqual([order.request_time for order in first], [1, 2])
        self.assertEqual(first[0].num_passengers, 1)
        self.assertIsNone(first[0].ready_time)
        first[0].status = OrderStatus.COMPLETED
        self.assertEqual(generator.generate()[0].status, OrderStatus.PENDING)

    def test_explicit_ids_are_exact_and_duplicates_rejected(self):
        large = 9007199254740993
        self.write([{**self.basic, "order_id": large}, {**self.basic, "order_id": large + 1}])
        self.assertEqual(
            [order.order_id for order in read_order_csv(self.path)], [large, large + 1]
        )
        self.write([{**self.basic, "order_id": 5}, {**self.basic, "order_id": 5}])
        with self.assertRaisesRegex(ValueError, "duplicate order_id"):
            read_order_csv(self.path)

    def test_optional_readiness_and_party(self):
        self.write(
            [
                {**self.basic, "ready_time": "", "num_passengers": ""},
                {**self.basic, "ready_time": 5, "num_passengers": 2},
            ]
        )
        orders = read_order_csv(self.path)
        self.assertEqual(
            [(order.ready_time, order.num_passengers) for order in orders], [(None, 1), (5, 2)]
        )
        self.write([{**self.basic, "ready_time": 1}])
        self.assertEqual(read_order_csv(self.path)[0].ready_time, 1)

    def test_custom_columns_and_disabled_optional_fields(self):
        row = {
            "ox": 0,
            "oy": 1,
            "dx": 2,
            "dy": 3,
            "at": 4,
            "ready_time": "ignored",
            "unused": "ignored",
        }
        self.write([row])
        columns = {
            "origin_x": "ox",
            "origin_y": "oy",
            "dest_x": "dx",
            "dest_y": "dy",
            "request_time": "at",
            "ready_time": None,
        }
        order = CsvOrderGenerator(self.path, columns).generate()[0]
        self.assertEqual((order.origin, order.destination, order.request_time), ((0, 1), (2, 3), 4))
        self.assertIsNone(order.ready_time)

    def test_csv_loading_does_not_import_pandas(self):
        self.write([self.basic])
        original = builtins.__import__

        def no_pandas(name, *args, **kwargs):
            if name == "pandas" or name.startswith("pandas."):
                raise ModuleNotFoundError("pandas is not installed")
            return original(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=no_pandas):
            order = CsvOrderGenerator(self.path).generate()[0]
        self.assertEqual(order.request_time, 2)

    def test_explicit_aware_time_origin_converts_both_times(self):
        self.write(
            [
                {
                    **self.basic,
                    "request_time": "2030-01-02T10:02:00+02:00",
                    "ready_time": "2030-01-02T10:05:30+02:00",
                }
            ]
        )
        origin = datetime(2030, 1, 2, 10, tzinfo=timezone(timedelta(hours=2)))
        order = read_order_csv(self.path, time_origin=origin)[0]
        self.assertEqual((order.request_time, order.ready_time), (2, 5.5))
        self.assertEqual(read_order_csv(self.path, time_origin=origin.isoformat())[0], order)

    def test_utc_z_time_and_negative_relative_offsets_are_explicit(self):
        self.write([{**self.basic, "request_time": "2030-01-02T09:59:00Z"}])
        order = read_order_csv(self.path, time_origin="2030-01-02T10:00:00Z")[0]
        self.assertEqual(order.request_time, -1)

    def test_timestamps_require_consistent_timezone_and_explicit_origin(self):
        for origin, request in (
            (None, "2030-01-02T10:00:00Z"),
            ("2030-01-02T10:00:00", "2030-01-02T10:00:00"),
            ("2030-01-02T10:00:00Z", "2030-01-02T10:00:00"),
            ("2030-01-02T10:00:00Z", "2030-01-02T10:00:00+02:00"),
            ("2030-01-02T10:00:00Z", 1234567890000),
        ):
            self.write([{**self.basic, "request_time": request}])
            with self.subTest(origin=origin, request=request), self.assertRaises(ValueError):
                read_order_csv(self.path, time_origin=origin)

    def test_finite_numbers_positive_integer_parties_and_integer_ids(self):
        for change in (
            {"origin_x": "nan"},
            {"dest_y": "inf"},
            {"request_time": "nan"},
            {"ready_time": "inf"},
            {"num_passengers": 0},
            {"num_passengers": -1},
            {"num_passengers": 1.2},
            {"order_id": ""},
            {"order_id": 1.2},
        ):
            self.write([{**self.basic, **change}])
            with self.subTest(change=change), self.assertRaises(ValueError):
                read_order_csv(self.path)

    def test_headers_and_row_shape_are_validated(self):
        self.path.write_text("request_time,origin_x,origin_x\n0,1,2\n")
        with self.assertRaisesRegex(ValueError, "duplicate column"):
            read_order_csv(self.path)
        self.write([{"request_time": 0}])
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            read_order_csv(self.path)
        self.path.write_text("origin_x,origin_y,dest_x,dest_y,request_time\n0,1,2,3,4,5\n")
        with self.assertRaisesRegex(ValueError, "more cells"):
            read_order_csv(self.path)

    def test_dataframe_preserves_default_ids_and_accepts_optional_ready_time(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas is optional")
        frame = pd.DataFrame(
            [
                {**self.basic, "request_time": 2, "order_id": 99, "ready_time": float("nan")},
                {**self.basic, "request_time": 1, "order_id": 98, "ready_time": 3},
            ]
        )
        orders = DataFrameOrderGenerator(frame).generate()
        self.assertEqual([order.order_id for order in orders], [0, 1])
        self.assertEqual([order.ready_time for order in orders], [None, None])
        orders = DataFrameOrderGenerator(frame, columns={"ready_time": "ready_time"}).generate()
        self.assertEqual([order.ready_time for order in orders], [3, None])
        orders = DataFrameOrderGenerator(frame, columns={"order_id": "order_id"}).generate()
        self.assertEqual([order.order_id for order in orders], [98, 99])

    def test_dataframe_explicit_large_integer_ids_are_not_cast_through_row_floats(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas is optional")
        frame = pd.DataFrame(
            [
                {**self.basic, "order_id": 9007199254740993},
                {**self.basic, "order_id": 9007199254740994},
            ]
        )
        generator = DataFrameOrderGenerator(frame, columns={"order_id": "order_id"})
        self.assertEqual(
            [order.order_id for order in generator.generate()], [9007199254740993, 9007199254740994]
        )

    def test_dataframe_legacy_extra_columns_and_party_conversion_are_unchanged(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas is optional")
        frame = pd.DataFrame(
            [
                {
                    **self.basic,
                    "ready_time": "unrelated text",
                    "order_id": "unrelated label",
                    "num_passengers": 1.5,
                }
            ]
        )
        order = DataFrameOrderGenerator(frame).generate()[0]
        self.assertEqual(order.order_id, 0)
        self.assertIsNone(order.ready_time)
        self.assertEqual(order.num_passengers, 1)
        with self.assertRaises(ValueError):
            DataFrameOrderGenerator(frame, {"ready_time": "ready_time"}).generate()

    def test_dataframe_custom_readiness_and_invalid_explicit_ids(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas is optional")
        frame = pd.DataFrame([{**self.basic, "available_at": 4}])
        order = DataFrameOrderGenerator(frame, {"ready_time": "available_at"}).generate()[0]
        self.assertEqual(order.ready_time, 4)
        duplicate = pd.DataFrame([{**self.basic, "id": 5}, {**self.basic, "id": 5}])
        with self.assertRaisesRegex(ValueError, "duplicate order_id"):
            DataFrameOrderGenerator(duplicate, {"order_id": "id"}).generate()
        with self.assertRaisesRegex(ValueError, "absent"):
            DataFrameOrderGenerator(frame, {"order_id": "missing"}).generate()


if __name__ == "__main__":
    unittest.main()
