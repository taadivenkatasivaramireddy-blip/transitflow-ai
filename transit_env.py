"""
TransitEnv — Core Environment Implementation
Takshashila University Campus Bus Fleet Management

Real-world scenario:
  A single bus serves 8 stops across the Takshashila University campus.
  The agent controls the bus dispatcher. Each episode simulates one
  morning route (7:30 AM – 9:00 AM, 20 time steps of ~4.5 min each).

  The agent must maximise passenger throughput while minimising delays,
  managing fuel, and handling dynamic passenger counts.
"""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional, Tuple

from models import (
    Action,
    ActionSpace,
    BusState,
    Observation,
    StepResult,
    StopState,
)

# ── Route Definition ──────────────────────────────────────────────────────────

CAMPUS_ROUTE: List[Dict[str, Any]] = [
    {"stop_id": 0, "name": "Main Gate",      "base_passengers": 8,  "scheduled": 0,  "distance_km": 0.0},
    {"stop_id": 1, "name": "Admin Block",    "base_passengers": 5,  "scheduled": 2,  "distance_km": 0.4},
    {"stop_id": 2, "name": "Central Library","base_passengers": 7,  "scheduled": 4,  "distance_km": 0.6},
    {"stop_id": 3, "name": "Hostel A",       "base_passengers": 14, "scheduled": 6,  "distance_km": 0.5},
    {"stop_id": 4, "name": "Hostel B",       "base_passengers": 9,  "scheduled": 8,  "distance_km": 0.3},
    {"stop_id": 5, "name": "Sports Complex", "base_passengers": 4,  "scheduled": 10, "distance_km": 0.7},
    {"stop_id": 6, "name": "Medical Block",  "base_passengers": 6,  "scheduled": 13, "distance_km": 0.5},
    {"stop_id": 7, "name": "Canteen Hub",    "base_passengers": 11, "scheduled": 15, "distance_km": 0.4},
]

MAX_STEPS = 20
BUS_CAPACITY = 40
INITIAL_FUEL = 100.0
FUEL_PER_MOVE = 7.0
FUEL_PER_EXPRESS = 15.0
FUEL_PER_SECOND_BUS = 30.0
MAX_WAIT_PENALTY = 2  # steps agent can wait before penalty kicks in


class TransitEnv:
    """
    OpenEnv-compliant environment for Takshashila University bus fleet management.

    API
    ---
    env = TransitEnv(seed=42)
    obs = env.reset()
    result = env.step(action)   # action: int or Action enum
    obs = env.state()

    All return values are Pydantic models with full type safety.
    """

    # ── Metadata ──────────────────────────────────────────────────────────────

    ENV_ID = "TakshashilaTransit-v1"
    VERSION = "1.0.0"
    MAX_STEPS = MAX_STEPS
    action_space = ActionSpace()

    def __init__(self, seed: int = 42, difficulty: str = "medium"):
        """
        Parameters
        ----------
        seed : int
            RNG seed for reproducibility.
        difficulty : str
            "easy" | "medium" | "hard"
            Controls passenger variance and event probability.
        """
        assert difficulty in ("easy", "medium", "hard"), \
            "difficulty must be 'easy', 'medium', or 'hard'"
        self.seed = seed
        self.difficulty = difficulty
        self._rng = random.Random(seed)
        self._obs: Optional[Observation] = None
        self._episode_count = 0
        self._wait_count = 0  # consecutive waits at current stop

        # Difficulty multipliers
        self._variance = {"easy": 0.0, "medium": 0.3, "hard": 0.6}[difficulty]
        self._event_prob = {"easy": 0.0, "medium": 0.1, "hard": 0.2}[difficulty]

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> Observation:
        """
        Reset environment to initial state.

        Returns
        -------
        Observation
            Initial observation with all stops unvisited.
        """
        if seed is not None:
            self.seed = seed
            self._rng = random.Random(seed)

        self._episode_count += 1
        self._wait_count = 0

        # Build stops with optional passenger variance
        stops = []
        for s in CAMPUS_ROUTE:
            variance = int(self._rng.gauss(0, s["base_passengers"] * self._variance)) \
                if self._variance > 0 else 0
            passengers = max(0, s["base_passengers"] + variance)
            stops.append(StopState(
                stop_id=s["stop_id"],
                name=s["name"],
                waiting_passengers=passengers,
                scheduled_arrival_step=s["scheduled"],
                distance_km=s["distance_km"],
            ))

        bus = BusState(
            current_stop_id=0,
            next_stop_id=1,
            passengers_onboard=0,
            capacity=BUS_CAPACITY,
            fuel_level=INITIAL_FUEL,
            speed_kmh=30.0,
            total_delay_steps=0,
        )

        self._obs = Observation(
            step=0,
            max_steps=MAX_STEPS,
            bus=bus,
            stops=stops,
            route_progress=0.0,
            done=False,
            episode_reward_so_far=0.0,
        )
        return copy.deepcopy(self._obs)

    def step(self, action: int | Action) -> StepResult:
        """
        Execute one action in the environment.

        Parameters
        ----------
        action : int or Action
            One of the 6 discrete actions (0–5).

        Returns
        -------
        StepResult
            Contains (observation, reward, done, info).
        """
        if self._obs is None:
            raise RuntimeError("Call reset() before step().")
        if self._obs.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        action = Action(int(action))
        obs = self._obs
        reward = 0.0
        info: Dict[str, Any] = {"action": action.name, "events": []}

        # ── Step counter ──
        obs.step += 1

        # ── Process action ────────────────────────────────────────────────────

        cur_id = obs.bus.current_stop_id
        cur_stop = obs.stops[cur_id]

        if action == Action.MOVE_TO_NEXT_STOP:
            reward, info = self._do_move(obs, reward, info, skip=0)
            self._wait_count = 0

        elif action == Action.WAIT_AT_STOP:
            self._wait_count += 1
            if self._wait_count <= MAX_WAIT_PENALTY:
                reward += 1.0  # small reward for waiting if passengers still boarding
                info["events"].append("Waited at stop")
            else:
                reward -= 2.0  # penalise excessive waiting
                info["events"].append("Excessive wait — penalty applied")

        elif action == Action.PICKUP_PASSENGERS:
            reward, info = self._do_pickup(obs, cur_stop, reward, info)
            self._wait_count = 0

        elif action == Action.SKIP_STOP:
            reward, info = self._do_skip(obs, reward, info)
            self._wait_count = 0

        elif action == Action.REROUTE_EXPRESS:
            reward, info = self._do_move(obs, reward, info, skip=2)
            self._wait_count = 0

        elif action == Action.DISPATCH_SECOND_BUS:
            reward, info = self._do_dispatch(obs, reward, info)

        # ── Per-step costs ────────────────────────────────────────────────────
        reward -= 0.5  # base time cost every step

        # ── Random events (medium/hard) ───────────────────────────────────────
        if self._rng.random() < self._event_prob:
            reward, info = self._random_event(obs, reward, info)

        # ── Fuel depletion check ──────────────────────────────────────────────
        if obs.bus.fuel_level <= 0:
            reward -= 20.0
            obs.done = True
            info["events"].append("BUS OUT OF FUEL — episode terminated")

        # ── Episode end conditions ────────────────────────────────────────────
        all_visited = all(s.visited for s in obs.stops)
        if all_visited:
            # Bonus for completing full route
            completion_bonus = 20.0
            # On-time bonus
            schedule_bonus = obs.on_time_stops * 3.0
            reward += completion_bonus + schedule_bonus
            obs.done = True
            info["events"].append(
                f"Route complete! +{completion_bonus} completion +{schedule_bonus:.1f} schedule bonus"
            )

        if obs.step >= MAX_STEPS and not obs.done:
            # Partial credit for progress made
            reward += obs.route_progress * 10.0
            obs.done = True
            info["events"].append("Max steps reached — partial credit awarded")

        # ── Update cumulative reward ──────────────────────────────────────────
        obs.episode_reward_so_far += reward

        result = StepResult(
            observation=copy.deepcopy(obs),
            reward=round(reward, 3),
            done=obs.done,
            info=info,
        )
        return result

    def state(self) -> Observation:
        """
        Return current observation without advancing the environment.

        Returns
        -------
        Observation
            Current state (deep copy).
        """
        if self._obs is None:
            raise RuntimeError("Call reset() before state().")
        return copy.deepcopy(self._obs)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _do_move(
        self,
        obs: Observation,
        reward: float,
        info: dict,
        skip: int = 0,
    ) -> Tuple[float, dict]:
        """Move bus forward by (1 + skip) stops."""
        cur_id = obs.bus.current_stop_id
        n_stops = len(obs.stops)
        next_id = min(cur_id + 1 + skip, n_stops - 1)

        if next_id == cur_id:
            reward -= 1.0
            info["events"].append("Already at last stop")
            return reward, info

        # Fuel cost
        fuel_cost = FUEL_PER_MOVE + (FUEL_PER_EXPRESS - FUEL_PER_MOVE) * (skip > 0)
        obs.bus.fuel_level = max(0.0, obs.bus.fuel_level - fuel_cost)

        obs.bus.current_stop_id = next_id
        obs.bus.next_stop_id = next_id + 1 if next_id < n_stops - 1 else None
        obs.bus.is_moving = True

        # Arrival analysis
        arrived_stop = obs.stops[next_id]
        delay = obs.step - arrived_stop.scheduled_arrival_step
        arrived_stop.actual_arrival_step = obs.step
        arrived_stop.visited = True

        if delay <= 1:
            reward += 5.0
            obs.on_time_stops += 1
            info["events"].append(f"Arrived at {arrived_stop.name} ON TIME")
        elif delay <= 3:
            reward += 2.0 - delay * 0.5
            obs.late_stops += 1
            info["events"].append(f"Arrived at {arrived_stop.name} slightly late (Δ{delay})")
        else:
            reward -= min(delay, 8)
            obs.late_stops += 1
            obs.bus.total_delay_steps += delay
            info["events"].append(f"Arrived at {arrived_stop.name} LATE (Δ{delay}) — penalty")

        # Route progress
        obs.route_progress = next_id / (n_stops - 1)

        return reward, info

    def _do_pickup(
        self,
        obs: Observation,
        cur_stop: StopState,
        reward: float,
        info: dict,
    ) -> Tuple[float, dict]:
        """Pick up waiting passengers at current stop."""
        waiting = cur_stop.waiting_passengers
        if waiting == 0:
            reward -= 1.0
            info["events"].append("No passengers to pick up")
            return reward, info

        space = obs.bus.capacity - obs.bus.passengers_onboard
        picked = min(waiting, space)
        missed = waiting - picked

        cur_stop.waiting_passengers = 0
        obs.bus.passengers_onboard += picked
        obs.total_passengers_served += picked
        obs.total_passengers_missed += missed

        # Reward: per passenger picked up
        reward += picked * 1.5

        # Penalty for overflow
        if missed > 0:
            reward -= missed * 0.5
            info["events"].append(
                f"Picked up {picked} passengers. OVERFLOW: {missed} missed (bus full)"
            )
        else:
            info["events"].append(f"Picked up all {picked} passengers at {cur_stop.name}")

        return reward, info

    def _do_skip(
        self,
        obs: Observation,
        reward: float,
        info: dict,
    ) -> Tuple[float, dict]:
        """Skip the next stop."""
        cur_id = obs.bus.current_stop_id
        n_stops = len(obs.stops)
        skip_id = min(cur_id + 1, n_stops - 1)
        skip_stop = obs.stops[skip_id]

        if skip_stop.waiting_passengers > 0:
            # Penalise heavily for skipping passengers
            penalty = skip_stop.waiting_passengers * 2.0
            reward -= penalty
            obs.total_passengers_missed += skip_stop.waiting_passengers
            obs.missed_stops += 1
            skip_stop.waiting_passengers = 0
            skip_stop.visited = True
            info["events"].append(
                f"SKIPPED {skip_stop.name} — {skip_stop.waiting_passengers} passengers abandoned! "
                f"Penalty: -{penalty:.1f}"
            )
        else:
            reward += 1.0  # valid skip saves time
            skip_stop.visited = True
            info["events"].append(f"Skipped {skip_stop.name} (empty stop)")

        # Advance bus past skipped stop
        obs.bus.current_stop_id = skip_id
        obs.bus.fuel_level = max(0.0, obs.bus.fuel_level - FUEL_PER_MOVE)
        obs.route_progress = skip_id / (n_stops - 1)

        return reward, info

    def _do_dispatch(
        self,
        obs: Observation,
        reward: float,
        info: dict,
    ) -> Tuple[float, dict]:
        """Dispatch a second bus to handle overflow."""
        if obs.bus.second_bus_dispatched:
            reward -= 5.0
            info["events"].append("Second bus already dispatched — cannot dispatch again")
            return reward, info

        if obs.bus.fuel_level < FUEL_PER_SECOND_BUS:
            reward -= 3.0
            info["events"].append("Insufficient fuel to dispatch second bus")
            return reward, info

        obs.bus.fuel_level -= FUEL_PER_SECOND_BUS
        obs.bus.second_bus_dispatched = True

        # Second bus serves all remaining stops simultaneously
        overflow_served = 0
        for stop in obs.stops:
            if not stop.visited and stop.waiting_passengers > 0:
                overflow_served += stop.waiting_passengers // 2
                stop.waiting_passengers = stop.waiting_passengers // 2

        reward += overflow_served * 0.8
        info["events"].append(
            f"Second bus dispatched — pre-serves ~{overflow_served} overflow passengers"
        )
        return reward, info

    def _random_event(
        self,
        obs: Observation,
        reward: float,
        info: dict,
    ) -> Tuple[float, dict]:
        """Inject a random real-world event."""
        events = [
            ("traffic_jam",   -3.0, "Traffic jam ahead — +2 delay steps"),
            ("rain_surge",     0.0, "Rain surge: +5 passengers added at next stop"),
            ("fuel_leak",     -5.0, "Fuel leak detected — -10 fuel"),
            ("gate_closed",   -2.0, "Campus gate temporarily closed — wait required"),
        ]
        name, r, msg = self._rng.choice(events)
        reward += r

        if name == "rain_surge" and obs.bus.current_stop_id < len(obs.stops) - 1:
            next_id = obs.bus.current_stop_id + 1
            obs.stops[next_id].waiting_passengers += 5
        elif name == "fuel_leak":
            obs.bus.fuel_level = max(0.0, obs.bus.fuel_level - 10.0)

        info["events"].append(f"[RANDOM EVENT] {msg}")
        return reward, info

    # ── Utilities ─────────────────────────────────────────────────────────────

    def render(self) -> str:
        """Return a simple text rendering of current state."""
        if self._obs is None:
            return "Environment not initialized. Call reset()."
        obs = self._obs
        lines = [
            f"╔══ {self.ENV_ID} | Step {obs.step}/{MAX_STEPS} | Episode Reward: {obs.episode_reward_so_far:.1f} ══╗",
            f"  Bus @ Stop {obs.bus.current_stop_id}: {obs.stops[obs.bus.current_stop_id].name}",
            f"  Passengers: {obs.bus.passengers_onboard}/{obs.bus.capacity}  |  Fuel: {obs.bus.fuel_level:.0f}%",
            f"  On-time: {obs.on_time_stops}  Late: {obs.late_stops}  Missed: {obs.missed_stops}",
            f"  Route Progress: {'█' * int(obs.route_progress * 20)}{'░' * (20 - int(obs.route_progress * 20))} {obs.route_progress:.0%}",
            "",
            "  STOPS:",
        ]
        for stop in obs.stops:
            status = "✓" if stop.visited else f"{stop.waiting_passengers}👤"
            arrow = " ← BUS" if stop.stop_id == obs.bus.current_stop_id else ""
            lines.append(f"    [{stop.stop_id}] {stop.name:<20} {status}{arrow}")
        lines.append("╚══════════════════════════════════════════════════════╝")
        return "\n".join(lines)

    def seed_info(self) -> Dict[str, Any]:
        return {"seed": self.seed, "difficulty": self.difficulty, "env_id": self.ENV_ID}
