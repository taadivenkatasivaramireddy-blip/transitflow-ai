"""
Task 1 — EASY: Morning Pickup (Score 0.0 – 1.0)

Objective:
  Successfully pick up passengers from the first 3 stops
  (Main Gate, Admin Block, Central Library) within 8 steps,
  arriving at each stop no more than 2 steps late.

Grader:
  score = (stops_served / 3) * 0.5
        + (passengers_collected / total_possible) * 0.3
        + schedule_adherence_bonus * 0.2

Baseline random agent score: ~0.25
Baseline greedy agent score: ~0.80
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any

from transit_env import TransitEnv
from transit_env.models import Action, Observation


@dataclass
class Task1Result:
    score: float
    stops_served: int
    passengers_collected: int
    total_possible_passengers: int
    schedule_adherence: float
    steps_taken: int
    breakdown: Dict[str, float]


TARGET_STOPS = {0, 1, 2}  # Main Gate, Admin Block, Central Library
MAX_STEPS_ALLOWED = 8
MAX_LATE_STEPS = 2


def grade(env: TransitEnv, trajectory: list[tuple[int, Observation]]) -> Task1Result:
    """
    Grade a completed trajectory for Task 1.

    Parameters
    ----------
    env : TransitEnv
        Environment after episode completion.
    trajectory : list of (action, observation) tuples
        Full trajectory from the episode.

    Returns
    -------
    Task1Result with score in [0.0, 1.0]
    """
    final_obs = trajectory[-1][1]
    stops = final_obs.stops

    # ── Component 1: Stop coverage (0.0 – 0.5) ──────────────────────────────
    stops_served = sum(1 for s in stops if s.stop_id in TARGET_STOPS and s.visited)
    stop_score = (stops_served / len(TARGET_STOPS)) * 0.5

    # ── Component 2: Passenger collection (0.0 – 0.3) ───────────────────────
    total_possible = sum(
        s.waiting_passengers + (
            # add back collected passengers if stop was visited
            0  # passengers already cleared from stop
        )
        for s in stops if s.stop_id in TARGET_STOPS
    )
    # Re-derive from original env passenger counts
    from transit_env.env import CAMPUS_ROUTE
    original_passengers = sum(
        CAMPUS_ROUTE[sid]["base_passengers"] for sid in TARGET_STOPS
    )
    passengers_collected = final_obs.total_passengers_served
    passenger_score = min(passengers_collected / max(original_passengers, 1), 1.0) * 0.3

    # ── Component 3: Schedule adherence (0.0 – 0.2) ─────────────────────────
    on_time = 0
    visited_target = 0
    for s in stops:
        if s.stop_id in TARGET_STOPS and s.visited and s.actual_arrival_step is not None:
            visited_target += 1
            delay = s.actual_arrival_step - s.scheduled_arrival_step
            if delay <= MAX_LATE_STEPS:
                on_time += 1

    adherence = on_time / max(visited_target, 1) if visited_target > 0 else 0.0
    schedule_score = adherence * 0.2

    total_score = round(stop_score + passenger_score + schedule_score, 4)
    total_score = max(0.0, min(1.0, total_score))

    breakdown = {
        "stop_coverage":        round(stop_score, 4),
        "passenger_collection": round(passenger_score, 4),
        "schedule_adherence":   round(schedule_score, 4),
    }

    return Task1Result(
        score=total_score,
        stops_served=stops_served,
        passengers_collected=passengers_collected,
        total_possible_passengers=original_passengers,
        schedule_adherence=adherence,
        steps_taken=final_obs.step,
        breakdown=breakdown,
    )


def run_task(agent_fn, seed: int = 42) -> Task1Result:
    """
    Run Task 1 with a given agent function.

    Parameters
    ----------
    agent_fn : callable(observation) -> int
        Agent that maps observation to action.
    seed : int
        RNG seed.

    Returns
    -------
    Task1Result
    """
    env = TransitEnv(seed=seed, difficulty="easy")
    obs = env.reset()
    trajectory = []
    done = False

    while not done and obs.step < MAX_STEPS_ALLOWED:
        action = agent_fn(obs)
        result = env.step(action)
        trajectory.append((action, result.observation))
        obs = result.observation
        done = result.done

    return grade(env, trajectory)
