"""Download and cache the Manhattan (region A) drive network (run once).

Region A is the midtown / lower-Manhattan bounding box that contains the taxi
zones used by the NYC FHVHV scenario:

    (lon_min, lat_min, lon_max, lat_max) = (-74.02, 40.70, -73.93, 40.80)

This fetches the drivable street network inside that box from OpenStreetMap via
OSMnx, annotates edges with speeds / travel times, prunes to the largest
strongly-connected component (so the all-pairs distance matrix is finite), and
pickles the result -- exactly like :mod:`ride_gym.data_tools.build_network`, but
bounded by a box rather than a centre+radius so the network lines up with the
chosen zones.

The cached graph is consumed by :class:`~ride_gym.osmnx_network.OSMnxNetwork`
(``graph_path=...``), the same backend used for the Guomao network.

Run (once, online)::

    python -m ride_gym.data_tools.nyc.build_nyc_network

By default the graph is written to ``./data/nyc/manhattan.gpickle`` (relative to
the current working directory). Requires the ``osmnx`` / ``networkx`` extras
(``pip install ride_gym[data]``).

NOTE: Manhattan is far larger than the Guomao region, so the all-pairs distance
+ predecessor matrices built by OSMnxNetwork scale with N**2 (N = node count).
For a few-thousand-node network that is hundreds of MB and a one-off build of a
few seconds; keep the box from growing without re-checking memory.
"""

from __future__ import annotations

import argparse
import os
import pickle

# Region A bounding box: (lon_min, lat_min, lon_max, lat_max).
REGION_A_BBOX = (-74.02, 40.70, -73.93, 40.80)


def default_out_path() -> str:
    """Default output path: ``./data/nyc/manhattan.gpickle`` under the cwd."""
    return os.path.join(os.getcwd(), "data", "nyc", "manhattan.gpickle")


def build_nyc_network(
    bbox: tuple = REGION_A_BBOX,
    out_path: str | None = None,
) -> str:
    """Download, annotate, prune and cache the drive network inside ``bbox``.

    Parameters
    ----------
    bbox:
        ``(lon_min, lat_min, lon_max, lat_max)`` of the region to fetch.
    out_path:
        Destination pickle path (a networkx MultiDiGraph, same format as the
        Guomao network so :class:`OSMnxNetwork` loads it unchanged). ``None`` ->
        :func:`default_out_path`.
    """
    # Lazy import so importing this module for its constants / default path does
    # not require the heavy OSMnx / networkx stack.
    import networkx as nx
    import osmnx as ox

    out_path = out_path or default_out_path()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    lon_min, lat_min, lon_max, lat_max = bbox
    print(f"downloading drive network for bbox={bbox} ...")

    # OSMnx >= 2.0 takes the bbox as (left, bottom, right, top) = (W, S, E, N),
    # i.e. (lon_min, lat_min, lon_max, lat_max), matching our convention.
    g = ox.graph_from_bbox(bbox=bbox, network_type="drive")

    # Same edge annotation as the Guomao builder: infer per-edge speeds then
    # derive travel times. The simulation ignores per-segment speed (constant
    # driver speed), but keeping the attributes makes the graph format identical.
    g = ox.add_edge_speeds(g)
    g = ox.add_edge_travel_times(g)

    # Largest strongly-connected component: every node must reach every other
    # node for the dense all-pairs distance matrix to be finite.
    if not nx.is_strongly_connected(g):
        largest = max(nx.strongly_connected_components(g), key=len)
        before = g.number_of_nodes()
        g = g.subgraph(largest).copy()
        print(
            f"  pruned to largest strongly-connected component: "
            f"{before} -> {g.number_of_nodes()} nodes"
        )

    with open(out_path, "wb") as f:
        pickle.dump(g, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"saved: {out_path}\n"
        f"  nodes={g.number_of_nodes()} edges={g.number_of_edges()}"
    )
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(
        description="Download & cache the Manhattan region-A drive network."
    )
    p.add_argument("--lon-min", type=float, default=REGION_A_BBOX[0])
    p.add_argument("--lat-min", type=float, default=REGION_A_BBOX[1])
    p.add_argument("--lon-max", type=float, default=REGION_A_BBOX[2])
    p.add_argument("--lat-max", type=float, default=REGION_A_BBOX[3])
    p.add_argument("--out", type=str, default=None,
                   help="output path (default: ./data/nyc/manhattan.gpickle).")
    args = p.parse_args()
    build_nyc_network(
        (args.lon_min, args.lat_min, args.lon_max, args.lat_max), args.out
    )


if __name__ == "__main__":
    main()
