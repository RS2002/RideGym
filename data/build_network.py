"""Download and cache a real road network for the simulation (run once).

Fetches a drivable street network around a centre point from OpenStreetMap via
OSMnx, annotates each edge with a travel time (derived from inferred/maximum
speeds), and saves it to a local ``.graphml`` file. The simulation then loads
this cached graph instead of re-downloading every run, so episodes are fast and
fully reproducible.

Run (once, online):

    python -m data.build_network

Re-run only to change the region or refresh the data. The default region is
Beijing Guomao CBD within a 2 km radius (~900 nodes), which gives a small,
realistic, fully-precomputable network (see OSMnxNetwork).
"""

from __future__ import annotations

import argparse
import os
import pickle

import networkx as nx
import osmnx as ox

# Default region: (lat, lon) of Beijing Guomao CBD and a 2 km radius.
DEFAULT_CENTER = (39.9087, 116.4570)
DEFAULT_RADIUS_M = 2000
# Stored as a pickled networkx graph rather than GraphML: the GraphML writer in
# networkx 3.1 references the removed numpy ``np.float_`` symbol and crashes on
# numpy >= 2.0. A pickle round-trips the annotated MultiDiGraph faithfully and
# loads faster.
DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "guomao.gpickle")


def build_network(
    center: tuple = DEFAULT_CENTER,
    radius_m: int = DEFAULT_RADIUS_M,
    out_path: str = DEFAULT_OUT,
) -> str:
    """Download, annotate, and save a drivable network. Returns the file path.

    Parameters
    ----------
    center:
        ``(lat, lon)`` centre of the region (OSMnx point convention).
    radius_m:
        Radius in metres of the drivable network to fetch around ``center``.
    out_path:
        Destination ``.graphml`` file.
    """
    print(f"downloading drive network: center={center} radius={radius_m} m ...")
    g = ox.graph_from_point(center, dist=radius_m, network_type="drive")

    # Annotate edges with travel times. add_edge_speeds infers a speed (km/h)
    # for every edge from its OSM 'maxspeed' tag, falling back to type-based
    # defaults; add_edge_travel_times then derives 'travel_time' (seconds) from
    # edge length and speed. These power the network's realistic time matrix.
    g = ox.add_edge_speeds(g)
    g = ox.add_edge_travel_times(g)

    # Keep the largest strongly-connected component so EVERY node can reach
    # every other node -- a hard requirement for a dense all-pairs distance
    # matrix (otherwise some pairs are unreachable / infinite).
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
    p = argparse.ArgumentParser(description="Download & cache a road network.")
    p.add_argument("--lat", type=float, default=DEFAULT_CENTER[0])
    p.add_argument("--lon", type=float, default=DEFAULT_CENTER[1])
    p.add_argument("--radius", type=int, default=DEFAULT_RADIUS_M)
    p.add_argument("--out", type=str, default=DEFAULT_OUT)
    args = p.parse_args()
    build_network((args.lat, args.lon), args.radius, args.out)


if __name__ == "__main__":
    main()