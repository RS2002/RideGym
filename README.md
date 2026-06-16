# ridepool-sim

A multi-agent **ride-pooling & dispatching** simulation environment for
transportation gig-market research. It provides a Gym-like (but **not**
Gym-dependent) interface, a strict and reproducible event loop, pluggable road
networks / order sources / reward functions, a set of dispatch baselines
(nearest-distance, Hungarian) and an Independent Double-DQN (IDDQN) learned
dispatcher. The design targets low-cost extension to food delivery, dynamic
pricing, multimodal transport, EV charging, multi-platform competition, and
mixed passenger/freight.

The environment runs in two coordinate regimes:

* **Abstract coordinates** (kilometres) on a Euclidean / Manhattan metric, for
  fast synthetic experiments.
* **Real road networks** (geographic `(lon, lat)`) backed by an OpenStreetMap
  graph, with O(1) shortest-path distances from precomputed all-pairs matrices.
  This regime can be driven by either synthetic demand or **real historical
  trip data** (e.g. the NYC FHVHV ride-hailing dataset).

---

## Table of contents

1. [Installation](#installation)
2. [Quick start](#quick-start)
3. [Repository layout](#repository-layout)
4. [Environment interface](#environment-interface)
5. [Core mechanics](#core-mechanics)
6. [The benchmark scenario](#the-benchmark-scenario)
7. [Running baselines](#running-baselines)
8. [Training the IDDQN dispatcher](#training-the-iddqn-dispatcher)
9. [Using your own data and run region](#using-your-own-data-and-run-region)
10. [Tests](#tests)

---

## Installation

```bash
pip install -e .            # core (numpy only)
pip install -e .[data]      # + pandas (DataFrame / file order sources)
pip install -e .[osmnx]     # + osmnx (real road networks)
pip install -e .[dev]       # + pytest
```

Real-network and learned-dispatch features additionally need `networkx`,
`scipy`, `pyarrow`, `geopandas`, `matplotlib` and `torch`. Install the ones you
need for the workflow you intend to run (see below).

---

## Quick start

A minimal abstract-coordinate episode with a conflict-free random policy:

```python
from ridepool_sim import RidePoolEnv, RandomOrderGenerator, ManhattanNetwork
from ridepool_sim.policies import RandomConflictFreePolicy

env = RidePoolEnv(
    area=(0, 0, 50, 50),
    num_drivers=8,
    driver_capacity=4,
    dt=1.0,
    horizon=120.0,
    order_timeout=12.0,
    road_network=ManhattanNetwork(speed=2.0),
    order_generator=RandomOrderGenerator((0, 0, 50, 50), 120.0, 120, arrival="poisson"),
    seed=0,
)

policy = RandomConflictFreePolicy(seed=0)
obs, info = env.reset(seed=0)
while True:
    obs, rewards, dones, info = env.step(policy.act(obs))
    if dones["__all__"]:
        break
```

Run the bundled demo (decentralised + centralised):

```bash
python -m examples.demo_random
```

---

## Repository layout

| Path | Responsibility |
|------|----------------|
| `ridepool_sim/` | Core simulation package (see [modules](#modules)). |
| `benchmark/` | Standard benchmark config, env factory, baselines, episode runner, recorder, comparison driver. |
| `iddqn/` | Independent Double-DQN learned dispatcher: features, Q-net, replay, exploration, bipartite matching, trainer, log plotting. |
| `data/` | Road-network builders + cached graphs. `data/build_network.py` builds an arbitrary region; `data/nyc/` builds the NYC scenario assets. |
| `dataset/` | Raw external inputs (NYC FHVHV parquet + taxi-zone shapefile). Download separately. |
| `examples/` | Runnable demos. |
| `tests/` | Unit tests. |

### Modules

| Module | Responsibility |
|--------|----------------|
| `enums.py` | Driver / order lifecycle states |
| `exceptions.py` | `InvalidActionError`, `ConflictError` |
| `entities.py` | `Order`, `Driver`, `TaskPoint` |
| `road_network.py` | `RoadNetwork` interface + Euclidean / Manhattan defaults |
| `osmnx_network.py` | Real OSM road network with O(1) distances and disk-cached all-pairs matrices |
| `routing.py` | `RoutesPlanner` + greedy precedence-aware sequencer |
| `order_generator.py` | Random / OSM-random / DataFrame / NYC-file order sources |
| `rewards.py` | `RewardFunction` + default multi-component reward |
| `env.py` | `RidePoolEnv` (decentralised core) |
| `wrappers.py` | `CentralizedWrapper` (single-agent view) |
| `policies.py` | Conflict-free baseline policy |

---

## Environment interface

### Decentralised multi-agent (base env)

* `reset(seed=None) -> (observations, info)`
* `step(actions) -> (observations, rewards, dones, info)`

Keyed by `driver_id`. `dones` includes an `"__all__"` flag.

**Action schema** (plain dict per driver; the two keys are mutually exclusive):

| Key | Meaning |
|-----|---------|
| `{"orders": [id, ...]}` | Bid on a set of pending order ids (may be empty = hold). |
| `{"relocate": idx \| (x, y)}` | Relocate to a preset point index or a coordinate. |

**Observation** per driver (a plain dict):

| Key | Contents |
|-----|----------|
| `self` | This driver's full private state (id, location, status, capacity, onboard, assigned orders). |
| `all_drivers` | Public state `{driver_id: {location, status, onboard_passengers}}` of **every** driver, *including this one*. |
| `pending_orders` | Shared snapshot of pending orders (id, origin, destination, party size, waiting time). |
| `time` | Current simulation time (minutes). |
| `relocation_points` | The relocation grid (an immutable tuple). |

> **Performance / safety note:** the `all_drivers`, `pending_orders` and
> `relocation_points` structures are **shared by reference** across all drivers'
> observations (built once per step, not per driver) to avoid an
> O(num_drivers²) cost. They must be treated as **read-only**.

### Centralised single-agent (wrapper)

```python
from ridepool_sim import CentralizedWrapper
env = CentralizedWrapper(RidePoolEnv(...), aggregate="sum")
obs, reward, done, info = env.step(joint_action)
# info["individual_rewards"] -> per-driver reward dict
```

---

## Core mechanics

Per-step event flow (order is intentional and enforced):

1. **Cancel timed-out** pending orders **before any action handling**.
2. Drivers act on the observation from the previous step.
3. **Validate** actions, run **conflict detection**, then assign / relocate.
4. **Physical movement** one time step along each driver's planned route.
5. **Order state updates** on pickup / drop-off arrival.
6. **Advance clock**, inject newly-arrived orders.
7. **Rewards** from the per-step event log.
8. **Termination** at the horizon.

Key guarantees:

* **Mutual-exclusion** of bid vs relocate is strictly enforced (`InvalidActionError`).
* **Conflict = exception**: if two drivers bid the same order, `ConflictError`
  is raised. The env **never auto-arbitrates**; upstream policies must coordinate.
* **Pickup-before-dropoff** precedence is enforced by the routing planner.
* **Capacity accounting** includes onboard + assigned-but-not-picked-up.
* **Greedy capacity drop**: if a bid set exceeds capacity, the env drops the
  largest-party orders until it fits (it does not raise); dropped orders stay
  pending.
* **Race tolerance**: bidding an order that was auto-cancelled between
  observation and step is silently ignored; bidding an *unknown* id is a hard
  error.

---

## The benchmark scenario

`benchmark/config.py` centralises a single, reproducible benchmark scenario in
the `BenchmarkConfig` dataclass. The defaults model a one-hour urban window:

| Field | Default | Meaning |
|-------|---------|---------|
| `network_kind` | `"nyc"` | `"euclidean"` / `"manhattan"` (abstract) or `"osmnx"` / `"nyc"` (real road network). |
| `num_drivers` | `1000` | Number of driver agents. |
| `num_orders` | `15000` | Total synthetic orders (ignored in `"nyc"` mode, where demand is read from the data file). |
| `horizon` | `60.0` | Episode length in minutes. |
| `dt` | `1.0` | Decision interval in minutes (60 steps). |
| `speed_kmh` | `60.0` | Constant driver speed. |
| `driver_capacity` | `3` | Per-driver passenger capacity. |
| `order_timeout` | `5.0` | Minutes a pending order may wait before auto-cancellation. |
| `arrival` | `"uniform"` | Synthetic temporal demand (`uniform` / `poisson` / `peak`). |
| `osmnx_graph_path` | `data/guomao.gpickle` | Cached graph for `"osmnx"` mode. |
| `nyc_graph_path` | `data/nyc/manhattan.gpickle` | Cached graph for `"nyc"` mode. |
| `nyc_order_path` | `data/nyc/orders.parquet` | Preprocessed real demand file for `"nyc"` mode. |
| `nyc_order_limit` | `None` | Optional cap on loaded NYC orders. |

Build an env from a config with `make_benchmark_env(cfg)`. For real-network
modes the service area is taken from the **graph's geographic bounds**, not the
abstract `area` rectangle. This is handled automatically by `resolved_area()`
and the env factory, and it is critical for correct spatial indexing and feature
normalisation.

---

## Running baselines

Compare the bundled dispatch baselines on the *same* scenario (same seed, same
drivers / orders) and print a side-by-side KPI table:

```bash
python -m benchmark.compare
```

This runs `NearestDistanceDispatch` and `HungarianDispatch`, writes detailed
per-step / per-order / per-driver records under `results/<name>/`, and prints
key metrics (service rate, complete rate, wait / ride / detour times, empty
distance ratio, driver utilisation, wall time). To change the scenario, edit the
`BenchmarkConfig` defaults or pass your own config into `main(cfg=...)`.

---

## Training the IDDQN dispatcher

```bash
python -m iddqn.train_iddqn
```

The trainer (`iddqn/train_iddqn.py`) builds the benchmark env, collects episodes
with Q-magnitude-scaled annealed exploration, trains a shared pairwise Q-network
via a Double-DQN bipartite-matching target, periodically runs a greedy
evaluation episode, and compares it against the nearest-distance and Hungarian
baselines on identical scenarios. All hyper-parameters live in `TrainConfig`
(including a nested `BenchmarkConfig`); logs and checkpoints are written under
`iddqn/runs/<timestamp>/`.

Plot the resulting curves:

```bash
python -m iddqn.plot_logs iddqn/runs/<run_name>
python -m iddqn.plot_logs iddqn/runs/<run_name> --no-show   # headless
```

---

## Using your own data and run region

This is the most important workflow for applying the simulator to a new city or
a real demand dataset. There are **two independent choices**:

1. **The road network (the run region)** -- which streets the drivers move on,
   e.g. Manhattan, Beijing Guomao, or Beijing Daxing.
2. **The demand (the orders)** -- either synthetic demand sampled on the
   network, or **real historical trips** loaded from a file.

### A. Defining a custom run region

A run region is just a cached OpenStreetMap drive network (a pickled
`networkx.MultiDiGraph`). `OSMnxNetwork` consumes it and precomputes O(1)
all-pairs distances (cached to disk as `<graph>.matrices.npz`).

**Option 1 -- centre + radius (any city/district).** Use the general builder.
For example, **Beijing Daxing**:

```bash
python -m data.build_network --lat 39.7267 --lon 116.3389 --radius 3000 \
    --out data/daxing.gpickle
```

Then point the config at it:

```python
from benchmark.config import BenchmarkConfig
cfg = BenchmarkConfig(network_kind="osmnx", osmnx_graph_path="data/daxing.gpickle")
```

**Option 2 -- bounding box (used by the NYC scenario).** Use the bbox builder.
The defaults fetch midtown/lower **Manhattan**:

```bash
python -m data.nyc.build_nyc_network
# or a custom box (lon_min lat_min lon_max lat_max):
python -m data.nyc.build_nyc_network \
    --lon-min -74.02 --lat-min 40.70 --lon-max -73.93 --lat-max 40.80 \
    --out data/nyc/manhattan.gpickle
```

Both builders fetch the drivable network from OpenStreetMap (one-off, online),
annotate edge speeds/times, prune to the largest strongly-connected component
(so the all-pairs distance matrix is finite), and pickle the result.

> **Sizing note.** The all-pairs distance + predecessor matrices scale as `N^2`
> in the node count `N`. A few-thousand-node region is hundreds of MB and a
> one-off build of tens of seconds (then reloaded sub-second from the cache).
> Keep the region from growing without re-checking memory.

With a graph in hand you can drive it with **synthetic demand** immediately
(`network_kind="osmnx"`): orders are sampled on real nodes, guaranteeing every
endpoint is on the network and reachable. No external dataset is required for
this path.

### B. Using real external demand (the NYC FHVHV example)

The NYC pipeline turns the raw FHVHV ride-hailing dataset (trips located by
*taxi zone* id) into a small, simulation-ready order file located by
`(lon, lat)`, snapped onto the Manhattan network. It is a three-step,
run-once-per-scenario pipeline.

**Step 0 -- obtain the raw inputs** (downloaded separately):

* `dataset/fhvhv_tripdata_2026-04.parquet` -- the raw FHVHV trip records.
* `dataset/taxi_zones/taxi_zones.shp` (+ sidecar files) -- the taxi-zone shapefile.

**Step 1 -- zone centroids.** Reduce each taxi-zone polygon to a representative
`(lon, lat)` point (computed in the projected CRS, then reprojected to WGS84):

```bash
python -m data.nyc.zone_centroids
# -> data/nyc/zone_centroids.csv
```

**Step 2 -- the run region** (already covered in part A):

```bash
python -m data.nyc.build_nyc_network
# -> data/nyc/manhattan.gpickle (+ .matrices.npz on first use)
```

**Step 3 -- preprocess the orders** (this is where you choose the time window,
region box and sampling rate). The script streams the ~21M-row parquet in
batches (it never loads it all at once), filtering by time window + region +
sample rate, and converts zone ids to centroid coordinates and request times to
minutes-from-start:

```bash
# default: 2026-04-01 morning peak (08:00-09:00), keep 100% of in-window trips
python -m data.nyc.preprocess_orders

# a different window and a 10% sample for a quicker run
python -m data.nyc.preprocess_orders \
    --start "2026-04-01 18:00" --end "2026-04-01 19:00" \
    --sample-rate 0.1 --out data/nyc/orders_evening.parquet
```

`--start` / `--end` set the episode window (the horizon is their difference in
minutes), `--sample-rate` thins the surviving trips reproducibly, and `--seed`
fixes that sampling. The output columns are
`origin_x, origin_y, dest_x, dest_y, request_time, num_passengers`.

**Step 4 -- run it.** Point the benchmark at the assets and run any baseline or
the trainer. `network_kind="nyc"` is already the default:

```python
from benchmark.config import BenchmarkConfig
cfg = BenchmarkConfig(
    network_kind="nyc",
    nyc_graph_path="data/nyc/manhattan.gpickle",
    nyc_order_path="data/nyc/orders.parquet",
    horizon=60.0,            # match your --start/--end window length
)
```

```bash
python -m benchmark.compare        # baselines on the NYC scenario
python -m iddqn.train_iddqn        # train IDDQN on the NYC scenario
```

In `"nyc"` mode demand is **deterministic** -- the trips, their times and party
sizes all come from the file, so every episode replays the same real demand
(`num_orders` and `arrival` are ignored). Each endpoint is snapped to its
nearest network node, so the chosen run region (Step 2) and the order region
(`preprocess_orders` bbox) must overlap.

### C. Bringing your own arbitrary dataset

If your data is not NYC-shaped, you have two clean extension points:

* **`DataFrameOrderGenerator`** (`ridepool_sim/order_generator.py`) consumes any
  pandas DataFrame with columns `origin_x, origin_y, dest_x, dest_y,
  request_time` (and optional `num_passengers`). Map your columns via its
  `columns` argument. Use this for abstract-coordinate demand.
* For real-network demand, mirror `NYCOrderGenerator`: snap each endpoint onto
  the network with `network.snap(...)` / `network.node_coord(...)` so every
  origin/destination lands on a reachable node, exactly as the graph-mode
  movement model requires.

The general recipe for **any city** is therefore: (1) build a graph for the
region, (2) produce an order file/DataFrame whose coordinates fall inside that
region, (3) load it through a generator that snaps onto the graph, (4) set the
matching `network_kind`, paths and `horizon` in `BenchmarkConfig`.

---

## Tests

```bash
python -m pytest -q
```