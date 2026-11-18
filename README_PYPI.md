# RideGym

RideGym is a multi-agent simulator for ride pooling, order dispatching and vehicle relocation. It provides a Gym-like interface without requiring Gym: a centralized dispatcher supplies per-vehicle actions, and the environment handles capacity, routes, movement, order lifecycles and rewards.

The PyPI distribution contains the `ride_gym` simulator package. Research benchmarks, training algorithms and experiment outputs are maintained separately in the [GitHub repository](https://github.com/RS2002/RideGym).

## Installation

Requires Python 3.9 or later. The core dependency is NumPy.

```bash
pip install --upgrade ride-gym
```

Install optional dependencies for the features you use:

```bash
pip install 'ride-gym[osmnx]'  # graph road networks
pip install 'ride-gym[data]'   # table and geographic data preparation
pip install 'ride-gym[viz]'    # rendering and plots
pip install 'ride-gym[all]'    # all optional dependencies
```

## Quickstart

This example uses one synthetic order and only the core dependency:

```python
from ride_gym import Order, OrderGenerator, RidePoolEnv, EuclideanNetwork


class SyntheticOrders(OrderGenerator):
    def generate(self):
        # Each reset receives fresh order lifecycle objects.
        return [Order(0, (0.25, 0.0), (0.75, 0.0), request_time=0.0)]


env = RidePoolEnv(
    area=(0.0, 0.0, 1.0, 1.0),
    num_drivers=1,
    driver_capacity=1,
    dt=0.25,
    horizon=3.0,
    order_timeout=None,
    order_generator=SyntheticOrders(),
    road_network=EuclideanNetwork(speed=1.0),
)
observations, info = env.reset(seed=0)

while True:
    pending = observations[0]["pending_orders"]
    actions = {0: {"orders": [pending[0]["order_id"]]}} if pending else {}
    observations, rewards, dones, info = env.step(actions)
    if dones["__all__"]:
        break

print(env.orders[0].status.value)  # completed
```

Observations, rewards and termination flags are keyed by vehicle ID. The shared `info["events"]` dictionary reports assignments, movement and lifecycle events. The dispatcher must avoid bidding on the same order from multiple vehicles; conflicting bids raise `ConflictError`.

Each vehicle can bid on pending orders with `{"orders": [order_id, ...]}` or relocate while idle with `{"relocate": index_or_coordinate}`. These actions are mutually exclusive. An empty action performs no new assignment or relocation.

## Optional simulation features

The options below are available from version 0.2.0. Existing positional constructor arguments remain supported. Fleet, availability, acceptance, response timing, readiness and service constraints are opt-in; omitting them retains immediate, unconditional assignment and the default observation fields.

Times are minutes from episode start, except response-policy durations, which are seconds. Distances follow the selected network, and speeds are network distance units per minute. A vehicle-type label does not select a speed or capacity automatically.

### Reporting granularity

`statistics_granularity="decision"` is the default: pickup and drop-off timestamps use the start of the decision step in which the event is processed, and step-level distance classification uses occupancy at the start of the step.

```python
from ride_gym import RidePoolEnv

env = RidePoolEnv(num_drivers=1, statistics_granularity="continuous")
```

Continuous reporting records arrival and service-completion times within the step. Events additionally contain `empty_distance_moved` and `occupied_distance_moved`, accumulated per traveled segment; their sum equals `distance_moved`.

For identical actions, the reporting option does not change movement, decision frequency, service waits, assignment, ETA, random draws or the built-in reward. In both modes, `is_empty_move` retains its original meaning: the vehicle started the step empty and moved. The default empty-move penalty remains fixed per step when that flag is true. A custom reward may use the continuous distance fields.

### Vehicle settings and availability

Use `driver_capacities` and `driver_speeds` for per-vehicle capacity and speed. To also configure positions, service times and online intervals, provide an explicit fleet:

```python
from ride_gym import RidePoolEnv, ExplicitLaborSource, ShiftWindow

fleet = ExplicitLaborSource([
    {
        "location": (0.0, 0.0),
        "vehicle_type": "bicycle",
        "capacity": 2,
        "speed": 0.8,
        "pickup_service_time": 0.25,
        "dropoff_service_time": 0.1,
        "availability_windows": [ShiftWindow(0.0, 5.0), ShiftWindow(7.0, 10.0)],
    },
    {
        "location": (1.0, 1.0),
        "vehicle_type": "car",
        "capacity": 3,
        "speed": 1.2,
        "shift_start": 1.0,
        "shift_end": 10.0,
    },
])
env = RidePoolEnv(num_drivers=2, labor_source=fleet, horizon=10.0)
```

Driver IDs are row indices, and fleet size must equal `num_drivers`. Online intervals are `[start, end)` and cannot overlap. At shift end, a vehicle stops receiving new offers but finishes existing commitments, including invitations already awaiting a reply. Offline vehicles stay in the observation and reward dictionaries. Dispatchers can check `observations[driver_id]["self"]["available"]`.

`CsvLaborSource` accepts the same settings in a UTF-8 CSV. Entry locations for later intervals are applied only while the vehicle is free of commitments; a busy vehicle continues its current route.

### Acceptance and response timing

```python
from ride_gym import (
    RidePoolEnv,
    AcceptanceProbabilities,
    PerDriverAcceptancePolicy,
    FixedResponseTimePolicy,
)

env = RidePoolEnv(
    num_drivers=1,
    dt=0.25,
    acceptance_policy=PerDriverAcceptancePolicy({
        0: AcceptanceProbabilities(general=0.85, single=0.9, bundle=0.75),
    }),
    acceptance_seed=11,
    response_time_policy=FixedResponseTimePolicy(seconds=12.0),
    response_seed=23,
)
```

A single offer contains one new order; a bundle contains two or more new orders. Type-specific probabilities override the general probability. Acceptance applies to the entire offered package. Rejected orders remain pending, and assignment rewards are emitted only on successful assignment.

Without a response policy, acceptance is immediate. With one, the response takes effect at the strictly next global decision boundary after its response time. Even a configured zero-second response waits for the next boundary. While awaiting a reply, the vehicle continues existing work; the offered orders are reserved, and it cannot receive another invitation. At resolution, the complete package is checked again before assignment.

`UniformResponseTimePolicy` and `BetaResponseTimePolicy` provide random durations. Acceptance and response timing use separate RNG streams, independent of the environment RNG.

### Readiness, service and order files

```python
from ride_gym import Order

order = Order(0, (0.0, 0.0), (1.0, 0.0), request_time=0.0, ready_time=1.5)
```

The vehicle may be assigned before `ready_time`, but it waits at the pickup until the order is ready. Pickup service starts after readiness; drop-off service takes time before completion. A readiness time before the request or episode start means the order is already ready. Service durations default to zero. Invitation contexts report travel-only route times.

`CsvOrderGenerator` reads orders using the standard library. Required columns are `origin_x`, `origin_y`, `dest_x`, `dest_y` and `request_time`. Optional columns are `order_id`, `num_passengers` and `ready_time`. Times default to relative minutes. Timezone-aware ISO timestamps require an explicit `time_origin`; column names can be mapped with `columns`.

`DataFrameOrderGenerator` is available with the data extra. Existing DataFrame input ignores extra readiness and order-ID columns unless they are explicitly enabled through its `columns` mapping. `validate_order_coords=True` enables warnings for order endpoints outside the service area.

### Snapshots and warm-up

Snapshots capture the current episode's orders, vehicle routes and graph-edge progress, service waits, pending invitations and simulator-owned RNG states. They can be retained in memory:

```python
from ride_gym import RidePoolEnv

env = RidePoolEnv(num_drivers=1, horizon=3.0)
observations, info = env.reset(seed=7)
state = env.capture_initial_state()
env.step({})
env.apply_initial_state(state)
assert env.time == state["time"]
```

Restore requires matching simulation configuration, including reporting granularity. `initial_state=state` can also be supplied when constructing a matching environment. The `save_snapshot` and `load_snapshot` functions support JSON files. An external dispatcher's state, arbitrary planner/reward/shift-policy state, and generator state for future resets must be saved separately.

`ride_gym.warmup.warmup(env, policy, steps)` runs an ordinary prefix of the same episode and returns observations, info and a snapshot. It does not reset the clock or add an evaluation filter.

## Routing and extension interfaces

`EuclideanNetwork` and `ManhattanNetwork` provide abstract routing. With the graph extra, `OSMnxNetwork` loads a supplied graph. A vehicle mid-edge finishes its active directed edge before rerouting; predictions include the remaining edge time and then plan from its endpoint. Optional matrix-memory limits and geographic coverage helpers are available through `OSMnxNetwork` and `ride_gym.region`.

Custom `RoutesPlanner` subclasses need only implement `plan(start, task_points, network)`. The inherited `plan_with_times()` computes cumulative arrival times; optimized implementations can override it. Routes must contain every supplied stop once and respect pickup-before-drop-off precedence. Assignment preparation and validation finish before the current batch is committed, so a planning failure cannot leave that batch partially assigned.

Custom order generators, road networks, rewards and centralized wrappers are also supported. See the [repository documentation](https://github.com/RS2002/RideGym#readme) for the full interfaces.

## Version 0.2.0 compatibility notes

- Optional fleet, schedule, acceptance, response, service, reporting and snapshot features are available in the PyPI package.
- Decision-step reporting remains the default, and existing positional arguments retain their meaning.
- Mid-edge route prediction now follows the same directed-edge state as movement.
- Predicted times and reward time baselines now use each vehicle's effective speed. The default revenue proxy also uses direct-trip time, so heterogeneous-speed reward and ETA values can differ from version 0.1.0.
- A custom planner implementing only `plan()` is supported, with validation before assignment commit.

## Authors and license

Zijian Zhao, Yulong Hu and Sen Li, The Hong Kong University of Science and Technology.

Released under the MIT License. The license text is included in both distributions.
