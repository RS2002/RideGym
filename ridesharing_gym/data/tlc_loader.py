"""
TLC Trip Record Data loader supporting CSV and Parquet formats with LocationID mode and date/hour filtering.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple
from ..core.order import Order
from .base_loader import DataLoader
from .zone_loader import load_default_zone_coords


class TLCDataLoader(DataLoader):
    """
    Load orders from TLC Trip Record files.

    Supports two modes:
    1. Use location ID (PULocationID/DOLocationID) with a mapping to coordinates.
    2. Use direct longitude/latitude columns.

    If use_location_id=True and zone_coords is None, default zones from package are loaded.

    Args:
        file_path: Path to the CSV or Parquet file.
        use_location_id: If True, use location ID columns; otherwise use lat/lon columns.
        zone_coords: Optional dict mapping LocationID to (lat, lon). If None and use_location_id=True,
                     default zones are loaded automatically.
        date_filter: Optional date string (e.g., "2025-11-15") to filter orders for a specific day.
                     If None, all orders are used.
        hour_range: Optional tuple (start_hour, end_hour) to filter orders within a hour range (inclusive start, exclusive end).
                    e.g., (8,20) for 8:00 to 19:59.
        request_time_column: Column name for pickup time.
        passenger_count_column: Column name for passenger count.
        pickup_location_id_column: Column name for pickup location ID (if use_location_id=True).
        dropoff_location_id_column: Column name for dropoff location ID (if use_location_id=True).
        pickup_lon_column: Column name for pickup longitude (if use_location_id=False).
        pickup_lat_column: Column name for pickup latitude (if use_location_id=False).
        dropoff_lon_column: Column name for dropoff longitude (if use_location_id=False).
        dropoff_lat_column: Column name for dropoff latitude (if use_location_id=False).
        cancel_time_delta: Optional fixed duration after request when order cancels (seconds).
        start_time_shift: Shift all times by this amount (seconds) to start from 0.
                          If None, auto-shift to make earliest request time 0.
        skip_nan_passenger: If True, skip rows with NaN passenger count. If False, replace NaN with 1.
    """

    def __init__(
        self,
        file_path: str,
        use_location_id: bool = True,
        zone_coords: Optional[Dict[int, Tuple[float, float]]] = None,
        date_filter: Optional[str] = None,
        hour_range: Optional[Tuple[int, int]] = None,
        request_time_column: str = 'tpep_pickup_datetime',
        passenger_count_column: str = 'passenger_count',
        pickup_location_id_column: str = 'PULocationID',
        dropoff_location_id_column: str = 'DOLocationID',
        pickup_lon_column: str = 'pickup_longitude',
        pickup_lat_column: str = 'pickup_latitude',
        dropoff_lon_column: str = 'dropoff_longitude',
        dropoff_lat_column: str = 'dropoff_latitude',
        cancel_time_delta: Optional[float] = None,
        start_time_shift: Optional[float] = None,
        skip_nan_passenger: bool = True
    ):
        self.file_path = file_path
        self.use_location_id = use_location_id
        self.date_filter = date_filter
        self.hour_range = hour_range
        self.request_col = request_time_column
        self.passenger_col = passenger_count_column
        self.pickup_loc_id_col = pickup_location_id_column
        self.dropoff_loc_id_col = dropoff_location_id_column
        self.pickup_lon_col = pickup_lon_column
        self.pickup_lat_col = pickup_lat_column
        self.dropoff_lon_col = dropoff_lon_column
        self.dropoff_lat_col = dropoff_lat_column
        self.cancel_time_delta = cancel_time_delta
        self.start_time_shift = start_time_shift
        self.skip_nan_passenger = skip_nan_passenger

        if use_location_id:
            if zone_coords is None:
                self.zone_coords = load_default_zone_coords()
                print(f"Automatically loaded {len(self.zone_coords)} zones.")
            else:
                self.zone_coords = zone_coords
        else:
            self.zone_coords = None

        # Load data
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.csv':
            self.df = pd.read_csv(file_path)
        elif ext == '.parquet':
            self.df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        # Convert pickup time to datetime
        self.df['pickup_datetime'] = pd.to_datetime(self.df[self.request_col], errors='coerce')
        self.df = self.df.dropna(subset=['pickup_datetime'])

        # Apply date filter if specified
        if date_filter is not None:
            filter_date = pd.to_datetime(date_filter).date()
            self.df = self.df[self.df['pickup_datetime'].dt.date == filter_date].copy()
            if len(self.df) == 0:
                raise ValueError(f"No orders found for date {date_filter}")

        # Apply hour filter if specified
        if hour_range is not None:
            start_hour, end_hour = hour_range
            self.df = self.df[
                (self.df['pickup_datetime'].dt.hour >= start_hour) &
                (self.df['pickup_datetime'].dt.hour < end_hour)
            ].copy()
            if len(self.df) == 0:
                raise ValueError(f"No orders found in hour range {hour_range}")

        # Handle NaN passenger counts
        if skip_nan_passenger:
            self.df = self.df.dropna(subset=[self.passenger_col])
        else:
            self.df[self.passenger_col] = self.df[self.passenger_col].fillna(1)

        # Filter out rows with passenger_count <= 0
        self.df = self.df[self.df[self.passenger_col] > 0]
        self.df[self.passenger_col] = self.df[self.passenger_col].astype(int)

        # Convert datetime to Unix timestamp (seconds)
        epoch = pd.Timestamp("1970-01-01")
        self.df['timestamp'] = (self.df['pickup_datetime'] - epoch) // pd.Timedelta('1s')
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)

        # Determine time shift
        if start_time_shift is None:
            self.start_time_shift = float(self.df['timestamp'].min())
        else:
            self.start_time_shift = start_time_shift

        self.df['shifted_time'] = self.df['timestamp'] - self.start_time_shift
        self._total_duration = float(self.df['shifted_time'].max())

        # Debug print
        print(f"DataLoader: shifted_time min={self.df['shifted_time'].min():.0f}, "
              f"max={self.df['shifted_time'].max():.0f}, total_duration={self._total_duration:.0f}")

    def _get_coordinates(self, row) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Return (pickup_location, dropoff_location) as (lat, lon)."""
        if self.use_location_id:
            pu_id = row[self.pickup_loc_id_col]
            do_id = row[self.dropoff_loc_id_col]
            if pu_id not in self.zone_coords or do_id not in self.zone_coords:
                raise KeyError(f"Location ID {pu_id} or {do_id} not found")
            return self.zone_coords[pu_id], self.zone_coords[do_id]
        else:
            pu = (float(row[self.pickup_lat_col]), float(row[self.pickup_lon_col]))
            do = (float(row[self.dropoff_lat_col]), float(row[self.dropoff_lon_col]))
            return pu, do

    def load_orders(self, start_time: float, end_time: float) -> List[Order]:
        """Load orders with request time in [start_time, end_time)."""
        mask = (self.df['shifted_time'] >= start_time) & (self.df['shifted_time'] < end_time)
        subset = self.df[mask]

        orders = []
        for idx, row in subset.iterrows():
            cancel = None
            if self.cancel_time_delta is not None:
                cancel = row['shifted_time'] + self.cancel_time_delta

            try:
                pu, do = self._get_coordinates(row)
            except KeyError:
                continue  # skip orders with missing zone mapping

            order = Order(
                order_id=int(idx),
                pickup_location=pu,
                dropoff_location=do,
                request_time=float(row['shifted_time']),
                passenger_count=int(row[self.passenger_col]),
                cancel_time=cancel
            )
            orders.append(order)

        return orders

    def get_total_duration(self) -> float:
        return self._total_duration