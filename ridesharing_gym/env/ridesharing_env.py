"""
Main Gym environment for ride-sharing MARL.
"""

import gym
from gym import spaces
import numpy as np
from typing import List, Dict, Tuple, Optional, Any

from ..core.driver import Driver, DriverStatus
from ..core.order import Order
from ..core.waypoint import WaypointType
from .config import EnvConfig
from .action_schemas import create_action_space, validate_action


class RideSharingEnv(gym.Env):
    """
    Multi-agent ride-sharing environment.

    The environment simulates a fleet of drivers over discrete time steps.
    At each step, drivers choose actions (assign orders or reposition),
    the environment updates driver positions, processes pickups/dropoffs,
    and generates new orders from a data source.

    The observation is a global state dictionary containing all drivers and pending orders.
    Action space is a Dict for each driver: order IDs and reposition region.
    Reward is computed per driver by the configured RewardFunction.

    Args:
        config: EnvConfig instance.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, config: EnvConfig):
        super().__init__()
        self.config = config

        if config.seed is not None:
            np.random.seed(config.seed)

        self.data_loader = config.data_loader
        self.total_duration = config.total_duration if config.total_duration is not None else self.data_loader.get_total_duration()

        self.current_time = 0.0
        self.drivers: List[Driver] = []
        self.pending_orders: List[Order] = []
        self.all_orders: List[Order] = []
        self.completed_orders: List[Order] = []

        for i in range(config.num_drivers):
            driver = Driver(
                driver_id=i,
                capacity=config.driver_capacities[i],
                current_location=(0.0, 0.0),
                speed=config.driver_speed
            )
            self.drivers.append(driver)

        self.action_space = create_action_space(
            num_regions=len(config.region_centers),
            max_capacity=max(config.driver_capacities)
        )

        self.observation_space = spaces.Dict({
            "timestamp": spaces.Box(low=0, high=np.inf, shape=()),
            "drivers": spaces.Sequence(spaces.Dict({})),
            "pending_orders": spaces.Sequence(spaces.Dict({}))
        })

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Dict[str, Any]:
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.current_time = 0.0
        self.pending_orders = []
        self.all_orders = []
        self.completed_orders = []

        for i, driver in enumerate(self.drivers):
            if self.config.region_centers:
                idx = self.np_random.integers(0, len(self.config.region_centers))
                driver.current_location = self.config.region_centers[idx]
            else:
                driver.current_location = (0.0, 0.0)
            driver.route = []
            driver.enroute_orders = []
            driver.occupied_capacity = 0
            driver.total_distance_driven = 0.0
            driver._sync_status()

        self.pending_orders = self.data_loader.load_orders(0, self.config.step_duration)
        self.all_orders.extend(self.pending_orders)
        return self._get_obs()

    def step(self, actions: Dict[int, Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[int, float], bool, Dict]:
        """
        Execute one simulation step.

        Args:
            actions: Dictionary mapping driver_id to action dict.

        Returns:
            observation: Global state after step.
            rewards: Dictionary mapping driver_id to reward.
            terminated: Whether simulation has ended.
            info: Additional info dict.
        """
        # 1. Collect actions from idle drivers
        idle_drivers = [d for d in self.drivers if d.status == DriverStatus.IDLE]
        driver_actions = {d.driver_id: actions[d.driver_id] for d in idle_drivers if d.driver_id in actions}

        # 2. Validate actions atomically
        valid_actions = {}
        invalid_reasons = {}

        requested_orders = set()
        duplicate_orders = set()
        for act in driver_actions.values():
            order_ids = [oid for oid in act["order_ids"] if oid != -1]
            for oid in order_ids:
                if oid in requested_orders:
                    duplicate_orders.add(oid)
                requested_orders.add(oid)

        for driver_id, act in driver_actions.items():
            driver = self._get_driver(driver_id)
            order_ids = [oid for oid in act["order_ids"] if oid != -1]

            if any(oid in duplicate_orders for oid in order_ids):
                if self.config.strict_action_check:
                    raise ValueError(f"Invalid action for driver {driver_id}: duplicate order")
                invalid_reasons[driver_id] = "duplicate_order"
                continue

            valid, reason = validate_action(
                act,
                driver,
                self.pending_orders,
                len(self.config.region_centers),
                self.config.region_centers,
                self.config.distance_calc,
                self.config.max_pickup_distance,
                self.config.max_reposition_distance
            )
            if valid:
                valid_actions[driver_id] = act
            else:
                if self.config.strict_action_check:
                    raise ValueError(f"Invalid action for driver {driver_id}: {reason}")
                invalid_reasons[driver_id] = reason

        # 3. Apply all valid actions simultaneously
        events = {}
        assigned_orders = []

        for driver_id, act in valid_actions.items():
            driver = self._get_driver(driver_id)
            order_ids = [oid for oid in act["order_ids"] if oid != -1]

            if order_ids:
                selected = []
                for oid in order_ids:
                    order = next((o for o in self.pending_orders if o.order_id == oid), None)
                    if order is None:
                        continue
                    selected.append(order)

                if selected:
                    success, new_route, _ = self.config.route_planner.plan(
                        driver.current_location,
                        driver.route,
                        selected
                    )
                    if not success:
                        if self.config.strict_action_check:
                            raise RuntimeError(f"Route planning failed for driver {driver_id}")
                        continue

                    passenger_counts = [o.passenger_count for o in selected]
                    order_id_list = [o.order_id for o in selected]
                    driver.assign_orders(
                        order_id_list,
                        passenger_counts,
                        new_route,
                        self.config.distance_calc  # 新增参数
                    )
                    for order in selected:
                        order.assign(driver.driver_id, self.current_time)
                        self.pending_orders.remove(order)
                        assigned_orders.append(order)
                else:
                    # Reposition action
                    region = act["reposition_region"]
                    target = self.config.region_centers[region]
                    driver.set_reposition(target, self.config.distance_calc)


            else:
                region = act["reposition_region"]
                target = self.config.region_centers[region]
                driver.set_reposition(target)

        # 4. Move all drivers
        for driver in self.drivers:
            ev = driver.move(self.config.step_duration, self.config.distance_calc)
            if ev:
                events[driver.driver_id] = ev

        # 5. Process movement events
        for driver_id, ev_list in events.items():
            driver = self._get_driver(driver_id)
            for ev in ev_list:
                if len(ev) == 4:
                    ev_type, order_id, _, time_offset = ev
                else:
                    ev_type, order_id, _ = ev
                    time_offset = self.config.step_duration
                event_abs_time = self.current_time + time_offset
                if ev_type == 'pickup':
                    order = self._get_order(order_id)
                    order.pickup_time = event_abs_time
                elif ev_type == 'dropoff':
                    order = self._get_order(order_id)
                    order.dropoff_time = event_abs_time
                    driver.complete_dropoff(order_id, order.passenger_count)
                    self.completed_orders.append(order)


        # 6. Advance time
        self.current_time += self.config.step_duration

        # 7. Cancel expired orders (after time advance)
        self._cancel_expired_orders()

        # 8. Load new orders for the next step
        new_orders = self.data_loader.load_orders(self.current_time, self.current_time + self.config.step_duration)
        self.pending_orders.extend(new_orders)
        self.all_orders.extend(new_orders)

        # 9. Compute rewards
        rewards = self.config.reward_fn.compute(
            self.drivers,
            self.all_orders,
            self.current_time,
            events=events
        )

        if not self.config.strict_action_check and self.config.invalid_action_penalty != 0.0:
            for driver_id in invalid_reasons:
                rewards[driver_id] += self.config.invalid_action_penalty

        terminated = self.current_time >= self.total_duration
        return self._get_obs(), rewards, terminated, {}

    def render(self, mode='human'):
        print(f"Time: {self.current_time:.1f}")
        print(f"Pending orders: {len(self.pending_orders)}")
        print(f"Completed orders: {len(self.completed_orders)}")
        idle = sum(1 for d in self.drivers if d.status == DriverStatus.IDLE)
        enroute = sum(1 for d in self.drivers if d.status == DriverStatus.ENROUTE)
        reposition = sum(1 for d in self.drivers if d.status == DriverStatus.REPOSITIONING)
        print(f"Drivers: idle={idle}, enroute={enroute}, reposition={reposition}")
        print("-" * 40)

    def _get_obs(self) -> Dict[str, Any]:
        return {
            "timestamp": self.current_time,
            "drivers": [d.to_dict() for d in self.drivers],
            "pending_orders": [o.to_dict() for o in self.pending_orders]
        }

    def _get_driver(self, driver_id: int) -> Driver:
        return next(d for d in self.drivers if d.driver_id == driver_id)

    def _get_order(self, order_id: int) -> Order:
        for o in self.all_orders:
            if o.order_id == order_id:
                return o
        raise ValueError(f"Order {order_id} not found")

    def _cancel_expired_orders(self):
        if self.config.order_cancel_time is None:
            return
        to_remove = []
        for order in self.pending_orders:
            if order.request_time + self.config.order_cancel_time <= self.current_time:
                order.cancel(self.current_time)
                to_remove.append(order)
        for order in to_remove:
            self.pending_orders.remove(order)