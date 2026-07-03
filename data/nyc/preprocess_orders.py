"""Turn the giant FHVHV parquet into a small, simulation-ready order file.

The raw ``fhvhv_tripdata_2026-04.parquet`` holds ~21 million trips for the whole
month and the whole city, located by *taxi zone* id. The simulation needs a
small set of orders, located by ``(lon, lat)``, with request times measured in
minutes from the start of the episode. This script bridges that gap by streaming
through the parquet in batches (it never loads all 21 M rows at once) and
applying, in order:

1. Time window  -- keep only trips whose ``request_datetime`` falls in
   ``[start, end)`` (default: 2026-04-01 08:00-09:00, the morning peak).
2. Region filter -- keep only trips whose pickup AND drop-off zones both lie
   inside region A's bounding box (so endpoints are on the cached Manhattan
   network).
3. Sampling      -- optionally down-sample the surviving trips by a fraction
   ``sample_rate`` (default 1.0 = keep all) with a fixed seed for reproducibility.

Each surviving trip becomes one order row with the columns expected by
:class:`~ridepool_sim.order_generator.DataFrameOrderGenerator`:

    origin_x, origin_y   -- pickup zone centroid (lon, lat)
    dest_x,   dest_y     -- drop-off zone centroid (lon, lat)
    request_time         -- minutes from ``start`` (float)
    num_passengers       -- 1 (FHVHV records carry no party size; see note)

Note on party size: the FHVHV schema has no passenger-count column, only a
shared-ride request flag. We set ``num_passengers = 1`` for every order (each
request is one party); pooling still happens via the planner when several
single-passenger orders are matched to the same driver.

Run::

    python -m data.nyc.preprocess_orders
    python -m data.nyc.preprocess_orders --start "2026-04-01 18:00" --end "2026-04-01 19:00" --sample-rate 0.1
"""

from __future__ import annotations

import argparse
import os
from typing import Set

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data.nyc.zone_centroids import load_zone_centroids
from data.nyc.build_nyc_network import REGION_A_BBOX

_HERE = os.path.dirname(__file__)
DEFAULT_PARQUET = os.path.join(
    _HERE, "..", "..", "dataset", "fhvhv_tripdata_2026-04.parquet"
)
DEFAULT_OUT = os.path.join(_HERE, "orders.parquet")

# Only the columns we actually need (fewer columns = faster, lighter batches).
_READ_COLS = ["request_datetime", "PULocationID", "DOLocationID"]


def _zones_in_bbox(centroids: dict, bbox: tuple) -> Set[int]:
    """Set of zone ids whose centroid lies inside ``bbox``."""
    lon_min, lat_min, lon_max, lat_max = bbox
    return {
        zid
        for zid, (lon, lat) in centroids.items()
        if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max
    }


def preprocess_orders(
    parquet_path: str = DEFAULT_PARQUET,
    out_path: str = DEFAULT_OUT,
    start: str = "2026-04-01 08:00",
    end: str = "2026-04-01 09:00",
    bbox: tuple = REGION_A_BBOX,
    sample_rate: float = 1.0,
    seed: int = 0,
    batch_size: int = 300_000,
) -> str:
    """Stream-filter the raw parquet into a small order file. Returns out_path."""
    if not (0.0 < sample_rate <= 1.0):
        raise ValueError(f"sample_rate must be in (0, 1], got {sample_rate}")

    centroids = load_zone_centroids()
    zone_set = _zones_in_bbox(centroids, bbox)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    rng = np.random.default_rng(seed)

    # Zone-id -> centroid component lookups for vectorised pandas .map().
    cx = {zid: lonlat[0] for zid, lonlat in centroids.items()}
    cy = {zid: lonlat[1] for zid, lonlat in centroids.items()}

    kept_frames = []
    total_in_window = 0
    f = pq.ParquetFile(parquet_path)
    for b in f.iter_batches(batch_size=batch_size, columns=_READ_COLS):
        t = pa.Table.from_batches([b]).to_pandas()
        # The file is time-sorted; once a batch begins at/after the window end we
        # can stop streaming entirely (big speed-up on the 21 M-row file).
        if t["request_datetime"].iloc[0] >= end_ts:
            break
        if t["request_datetime"].iloc[-1] < start_ts:
            continue

        mask = (
            (t["request_datetime"] >= start_ts)
            & (t["request_datetime"] < end_ts)
            & (t["PULocationID"].isin(zone_set))
            & (t["DOLocationID"].isin(zone_set))
        )
        sub = t[mask]
        if sub.empty:
            continue
        total_in_window += len(sub)

        if sample_rate < 1.0:
            keep = rng.random(len(sub)) < sample_rate
            sub = sub[keep]
            if sub.empty:
                continue

        out = pd.DataFrame(
            {
                "origin_x": sub["PULocationID"].map(cx).astype(float).values,
                "origin_y": sub["PULocationID"].map(cy).astype(float).values,
                "dest_x": sub["DOLocationID"].map(cx).astype(float).values,
                "dest_y": sub["DOLocationID"].map(cy).astype(float).values,
                "request_time": (
                    (sub["request_datetime"] - start_ts).dt.total_seconds()
                    / 60.0
                ).values,
                "num_passengers": 1,
            }
        )
        kept_frames.append(out)

    if not kept_frames:
        raise RuntimeError(
            "no orders matched the window/region filter; check --start/--end "
            "and that the bbox overlaps populated zones."
        )

    orders = pd.concat(kept_frames, ignore_index=True)
    orders = orders.sort_values("request_time").reset_index(drop=True)
    orders.to_parquet(out_path, index=False)

    horizon = (end_ts - start_ts).total_seconds() / 60.0
    print(
        f"saved {out_path}\n"
        f"  window {start}..{end} ({horizon:.0f} min), bbox zones={len(zone_set)}\n"
        f"  in-window/in-region trips={total_in_window}, "
        f"sampled (rate={sample_rate})={len(orders)}\n"
        f"  request_time range: {orders.request_time.min():.2f}"
        f"..{orders.request_time.max():.2f} min"
    )
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(
        description="Filter the FHVHV parquet into a small order file."
    )
    p.add_argument("--parquet", default=DEFAULT_PARQUET)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--start", default="2026-04-01 08:00")
    p.add_argument("--end", default="2026-04-01 09:00")
    p.add_argument(
        "--sample-rate",
        type=float,
        default=1.0,
        help="fraction of in-window/in-region trips to keep (default 1.0=all).",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    preprocess_orders(
        parquet_path=args.parquet,
        out_path=args.out,
        start=args.start,
        end=args.end,
        sample_rate=args.sample_rate,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()