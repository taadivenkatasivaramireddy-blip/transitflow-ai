"""
Task 2 — MEDIUM: Full Route Under Budget (Score 0.0 – 1.0)

Objective:
  Complete the full 8-stop route within 20 steps while:
  - Keeping fuel above 20% at all times
  - Serving ≥ 70% of total waiting passengers
  - Arriving at ≥ 5 of 8 stops on schedule (≤1 step late)

Grader:
  score = route_completion_score * 0.35
        + passenger_throughput_score * 0.35
        + fuel_efficiency_score * 0.15
        + schedule_score * 0.15

Baseline random agent score: ~0.15
Baseline greedy agent score: ~0.55
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

from transit_env import TransitEnv
from transit_env.models import Action, Observation
from transit_env.env import CAMPUS_ROUTE

TOTAL_BASE_PASSENGERS = sum(s["base_passengers"] for s in CAMPUS_ROUTE)
MIN_PASSENGER_RATE = 0.70
MIN_ONTIME_STOPS = 5
FUEL_DANGER_THRESHOLD = 20.0
MAX_STEPS = 20


@dataclass
class Task2Result:
    score: float
    route_completion: float
    passenger_throughput: float
    fuel_efficiency: float
    schedule_adherence: float
    total_passengers_served: int
    final_fuel: float
    breakdown: Dict[str, float]


def grade(env: TransitEnv, trajectory: list[tuple[int, Observation]]) -> Task2Result:
    final_obs = trajectory[-1][1]

    # ── Component 1: Route completion (0.0 – 0.35) ───────────────────────────
    stops_visited = sum(1 for s in final_obs.stops if s.visited)
    route_completion = stops_visited / len(final_obs.stops)
    route_score = route_completion * 0.35

    # ── Component 2: Passenger throughput (0.0 – 0.35) ───────────────────────
    throughput = final_obs.total_passengers_served / TOTAL_BASE_PASSENGERS
    throughput = min(throughput, 1.0)
    # Threshold bonus: agent gets full marks only if ≥70% served
    if throughput >= MIN_PASSENGER_RATE:
        passenger_score = 0.35
    else:
        passenger_score = (throughput / MIN_PASSENGER_RATE) * 0.25  # partial credit

    # ── Component 3: Fuel efficiency (0.0 – 0.15) ────────────────────────────
    final_fuel = final_obs.bus.fuel_level
    # Check if agent ever went into danger zone
    min_fuel_seen = min(
        (obs.bus.fuel_level for _, obs in trajectory),
        default=final_fuel
    )
    if min_fuel_seen >= FUEL_DANGER_THRESHOLD:
        fuel_score = 0.15  # never went into danger
    else:
        # Partial credit based on how far above 0
        fuel_score = max(0.0, (min_fuel_seen / FUEL_DANGER_THRESHOLD)) * 0.10

    # ── Component 4: Schedule adherence (0.0 – 0.15) ─────────────────────────
    on_time = final_obs.on_time_stops
    schedule_rate = min(on_time / MIN_ONTIME_STOPS, 1.0)
    schedule_score = schedule_rate * 0.15

    total_score = round(route_score + passenger_score + fuel_score + schedule_score, 4)
    total_score = max(0.0, min(1.0, total_score))

    breakdown = {
        "route_completion":    round(route_score, 4),
        "passenger_throughput": round(passenger_score, 4),
        "fuel_efficiency":     round(fuel_score, 4),
        "schedule_adherence":  round(schedule_score, 4),
    }

    return Task2Result(
        score=total_score,
        route_completion=round(route_completion, 4),
        passenger_throughput=round(throughput, 4),
        fuel_efficiency=round(final_fuel, 2),
        schedule_adherence=round(on_time / 8, 4),
        total_passengers_served=final_obs.total_passengers_served,
        final_fuel=final_fuel,
        breakdown=breakdown,
    )


def run_task(agent_fn, seed: int = 42) -> Task2Result:
    env = TransitEnv(seed=seed, difficulty="medium")
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
