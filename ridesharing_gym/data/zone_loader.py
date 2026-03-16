"""
Zone coordinate loader from packaged shapefile.
"""

import pickle
from typing import Dict, Tuple, Optional
from importlib import resources

try:
    import geopandas as gpd
    from pyproj import CRS, Transformer
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

_zone_coords_cache = None


def load_default_zone_coords(
    borough_filter: Optional[str] = None,
    use_cache: bool = True
) -> Dict[int, Tuple[float, float]]:
    """
    Load zone coordinates from the packaged taxi_zones shapefile.

    Args:
        borough_filter: If provided (e.g., 'Manhattan'), only include zones from that borough.
        use_cache: If True, cache the result to avoid recomputation.

    Returns:
        Dictionary mapping LocationID to (lat, lon).
    """
    global _zone_coords_cache
    if use_cache and _zone_coords_cache is not None:
        return _zone_coords_cache

    if not GEOPANDAS_AVAILABLE:
        raise ImportError("geopandas is required. Install with: pip install geopandas pyproj")

    with resources.path('ridesharing_gym.taxi_zones', 'taxi_zones.shp') as shp_path:
        gdf = gpd.read_file(shp_path)

    prj_path = shp_path.parent / 'taxi_zones.prj'
    with open(prj_path, 'r') as f:
        prj_content = f.read()
    input_crs = CRS.from_wkt(prj_content)
    output_crs = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(input_crs, output_crs, always_xy=True)

    if borough_filter is not None:
        gdf = gdf[gdf['borough'] == borough_filter]

    zone_coords = {}
    for _, row in gdf.iterrows():
        location_id = row['LocationID']
        centroid = row['geometry'].centroid
        lon, lat = transformer.transform(centroid.x, centroid.y)
        zone_coords[int(location_id)] = (lat, lon)

    if use_cache:
        _zone_coords_cache = zone_coords

    return zone_coords