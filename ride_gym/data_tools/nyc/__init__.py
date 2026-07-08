"""NYC FHVHV scenario data-preparation tools (code only, no bundled data).

Modules
-------
build_nyc_network:
    Download & cache the Manhattan region-A drive network (OSMnx).
zone_centroids:
    Reduce taxi-zone polygons to ``(lon, lat)`` centroids (geopandas).
preprocess_orders:
    Stream-filter the raw FHVHV parquet into a small, simulation-ready order
    file located by centroid coordinates.
build_splits:
    Slice the raw parquet into many train/val/test time-window order files
    plus a manifest, for the multi-window order generator.

Run any of them as a module, e.g. ``python -m ride_gym.data_tools.nyc.build_splits``.
All outputs default to ``./data/nyc/`` relative to the current working directory.
"""
