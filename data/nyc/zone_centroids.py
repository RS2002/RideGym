"""Build a NYC taxi-zone -> centroid coordinate mapping (run once).

The FHVHV trip records locate every pickup/drop-off by a *taxi zone* id
(``PULocationID`` / ``DOLocationID``), not by latitude/longitude. The simulation
and the real road network, however, work in continuous ``(lon, lat)``
coordinates. This module bridges the two by reducing each taxi-zone polygon to a
single representative point -- its centroid -- and reprojecting that point to
WGS84 longitude/latitude.

Why the centroid (and a caveat)
-------------------------------
A zone is an area, so any trip starting/ending in it is approximated by the
zone's centroid. This is the standard, lightweight way to turn zone-level NYC
data into point demand; it loses intra-zone detail but is exact at zone
granularity, which is all the raw data carries anyway.

Projection note
---------------
The shapefile ships in EPSG:2263 (NY State Plane, US feet). Centroids MUST be
computed in this *projected* CRS (geometric centroid of a planar polygon); doing
it in lon/lat would be geometrically wrong (geopandas even warns). We therefore
compute the centroid first, THEN reproject the resulting points to EPSG:4326.

Output
------
A CSV ``data/nyc/zone_centroids.csv`` with columns
``LocationID, zone, borough, lon, lat`` and a convenience loader
:func:`load_zone_centroids` returning a ``{location_id: (lon, lat)}`` dict.

Run::

    python -m data.nyc.zone_centroids
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import geopandas as gpd
import pandas as pd

_HERE = os.path.dirname(__file__)
DEFAULT_SHP = os.path.join(
    _HERE, "..", "..", "dataset", "taxi_zones", "taxi_zones.shp"
)
DEFAULT_OUT = os.path.join(_HERE, "zone_centroids.csv")


def build_zone_centroids(
    shp_path: str = DEFAULT_SHP, out_path: str = DEFAULT_OUT
) -> str:
    """Compute zone centroids in lon/lat and write them to ``out_path``."""
    g = gpd.read_file(shp_path)
    # Centroid in the projected CRS (correct), then reproject the POINTS to 4326.
    cent_proj = g.geometry.centroid
    cent_ll = gpd.GeoSeries(cent_proj, crs=g.crs).to_crs(4326)
    out = pd.DataFrame(
        {
            "LocationID": g["LocationID"].astype(int).values,
            "zone": g["zone"].values,
            "borough": g["borough"].values,
            "lon": cent_ll.x.values,
            "lat": cent_ll.y.values,
        }
    )
    # A few zones (e.g. islands split into multiple polygons) can repeat the
    # same LocationID; keep the first (largest is first in the source order).
    out = out.drop_duplicates(subset="LocationID", keep="first")
    out.to_csv(out_path, index=False)
    print(
        f"saved {out_path}: {len(out)} zones, "
        f"lon {out.lon.min():.4f}..{out.lon.max():.4f}, "
        f"lat {out.lat.min():.4f}..{out.lat.max():.4f}"
    )
    return out_path


def load_zone_centroids(
    path: str = DEFAULT_OUT,
) -> Dict[int, Tuple[float, float]]:
    """Load the centroid CSV into a ``{LocationID: (lon, lat)}`` dict."""
    df = pd.read_csv(path)
    return {
        int(r.LocationID): (float(r.lon), float(r.lat))
        for r in df.itertuples(index=False)
    }


if __name__ == "__main__":
    build_zone_centroids()