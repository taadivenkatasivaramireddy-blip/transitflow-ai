"""
Task 3 — HARD: Optimal Fleet Management (Score 0.0 – 1.0)

Objective:
  Achieve optimal campus transit service under hard difficulty with:
  - Dynamic passenger surges (random events)
  - Fuel scarcity (starts at 70%)
  - Strict schedule: ≥ 6/8 stops on time (≤1 step late)
  - Zero missed stops with waiting passengers
  - Overflow management (second bus dispatch decision)
  - Finish with positive cumulative reward

Grader (multi-objective):
  score = schedule_score       * 0.25
        + zero_miss_score      * 0.20
        + overflow_score       * 0.15
        + reward_efficiency    * 0.20
        + passenger_density    * 0.20

Baseline random agent score: ~0.08
Baseline greedy agent score: ~0.38
Optimal agent target:         ~0.75+
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

from transit_env import TransitEnv
from models import Action, Observation
from transit_env import CAMPUS_ROUTE

TOTAL_BASE_PASSENGERS = sum(s["base_passengers"] for s in CAMPUS_ROUTE)
HARD_FUEL_START = 70.0  # Override in env init
STRICT_ONTIME_MIN = 6
MAX_EPISODE_REWARD = 120.0  # Theoretical maximum


@dataclass
class Task3Result:
    score: float
    schedule_score: float
    zero_miss_score: float
    overflow_score: float
    reward_efficiency: float
    passenger_density_score: float
    cumulative_reward: float
    on_time_stops: int
    missed_stops: int
    passengers_served: int
    breakdown: Dict[str, float]


class HardTransitEnv(TransitEnv):
    """Task3 variant: reduced starting fuel."""

    def reset(self, seed=None):
        obs = super().reset(seed=seed)
        # Override starting fuel to make it harder
        self._obs.bus.fuel_level = HARD_FUEL_START
        obs.bus.fuel_level = HARD_FUEL_START
        return obs


def grade(env: TransitEnv, trajectory: List[Tuple[int, Observation]]) -> Task3Result:
    final_obs = trajectory[-1][1]

    # ── Component 1: Strict schedule (0.0 – 0.25) ────────────────────────────
    on_time = final_obs.on_time_stops
    schedule_rate = min(on_time / STRICT_ONTIME_MIN, 1.0)
    schedule_score = schedule_rate * 0.25

    # ── Component 2: Zero missed stops (0.0 – 0.20) ──────────────────────────
    missed = final_obs.missed_stops
    if missed == 0:
        zero_miss = 0.20
    elif missed == 1:
        zero_miss = 0.10
    else:
        zero_miss = max(0.0, 0.20 - missed * 0.05)

    # ── Component 3: Overflow management (0.0 – 0.15) ────────────────────────
    # Agent should dispatch second bus if > 30 passengers missed
    total_missed_passengers = final_obs.total_passengers_missed
    dispatched = final_obs.bus.second_bus_dispatched

    if total_missed_passengers <= 5:
        overflow_score = 0.15  # excellent — no overflow needed
    elif dispatched and total_missed_passengers <= 15:
        overflow_score = 0.12  # dispatched appropriately
    elif dispatched:
        overflow_score = 0.07  # dispatched but still high overflow
    else:
        overflow_score = max(0.0, 0.15 - total_missed_passengers * 0.01)

    # ── Component 4: Reward efficiency (0.0 – 0.20) ──────────────────────────
    cumulative_reward = final_obs.episode_reward_so_far
    reward_efficiency = max(0.0, min(cumulative_reward / MAX_EPISODE_REWARD, 1.0))
    reward_score = reward_efficiency * 0.20

    # ── Component 5: Passenger density (0.0 – 0.20) ──────────────────────────
    # Reward high utilisation: passengers_served / (steps * capacity)
    served = final_obs.total_passengers_served
    steps = max(final_obs.step, 1)
    density = served / (steps * 5)  # normalised throughput rate
    density = min(density, 1.0)
    passenger_density_score = density * 0.20

    total_score = round(
        schedule_score + zero_miss + overflow_score + reward_score + passenger_density_score, 4
    )
    total_score = max(0.0, min(1.0, total_score))

    breakdown = {
        "schedule":         round(schedule_score, 4),
        "zero_miss":        round(zero_miss, 4),
        "overflow_mgmt":    round(overflow_score, 4),
        "reward_efficiency":round(reward_score, 4),
        "passenger_density":round(passenger_density_score, 4),
    }

    return Task3Result(
        score=total_score,
        schedule_score=schedule_score,
        zero_miss_score=zero_miss,
        overflow_score=overflow_score,
        reward_efficiency=reward_efficiency,
        passenger_density_score=passenger_density_score,
        cumulative_reward=round(cumulative_reward, 2),
        on_time_stops=on_time,
        missed_stops=missed,
        passengers_served=served,
        breakdown=breakdown,
    )


def run_task(agent_fn, seed: int = 42) -> Task3Result:
    env = HardTransitEnv(seed=seed, difficulty="hard")
    obs = env.reset()
    trajectory = []
    done = False

    while not done:
        action = agent_fn(obs)
        result = env.step(action)
        trajectory.append((action, result.observation))
        obs = result.observation
        done = result.done

    return grade(env, trajectory)
