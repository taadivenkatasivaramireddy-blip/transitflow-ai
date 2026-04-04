"""
Baseline Inference Script — Takshashila Transit OpenEnv
=======================================================

Three baseline agents with reproducible scores across all 3 tasks.

Usage
-----
    python baselines/run_baselines.py
    python baselines/run_baselines.py --task 2 --agent greedy --seed 42

Agents
------
  random  — Uniform random action selection
  greedy  — Rule-based: always move → pickup cycle
  optimal — Heuristic: schedule-aware with fuel management
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transit_env.models import Action, Observation

# ── Agent Definitions ─────────────────────────────────────────────────────────


def random_agent(obs: Observation, seed: int = 0) -> int:
    """Uniform random action — lower bound baseline."""
    rng = random.Random(obs.step + seed)
    return rng.randint(0, 5)


def greedy_agent(obs: Observation) -> int:
    """
    Greedy agent: simple move-pickup cycle.
    - If current stop has unvisited passengers → PICKUP
    - Else → MOVE_TO_NEXT_STOP
    """
    cur_id = obs.bus.current_stop_id
    cur_stop = obs.stops[cur_id]

    if cur_stop.waiting_passengers > 0 and cur_id not in [
        s.stop_id for s in obs.stops if s.visited and s.stop_id == cur_id
    ]:
        return Action.PICKUP_PASSENGERS

    return Action.MOVE_TO_NEXT_STOP


def optimal_agent(obs: Observation) -> int:
    """
    Heuristic optimal agent with schedule awareness and fuel management.

    Policy:
    1. If current stop has passengers and not yet picked up → PICKUP
    2. If next stop is empty and fuel is high → SKIP_STOP
    3. If running very low fuel → REROUTE_EXPRESS to finish faster
    4. If bus near capacity and passengers remain → DISPATCH_SECOND_BUS
    5. Default → MOVE_TO_NEXT_STOP
    """
    cur_id = obs.bus.current_stop_id
    cur_stop = obs.stops[cur_id]
    bus = obs.bus
    n_stops = len(obs.stops)

    # 1. Pick up if passengers waiting here and stop not fully cleared
    if cur_stop.waiting_passengers > 0:
        return Action.PICKUP_PASSENGERS

    # 2. Dispatch second bus if overflow risk is high and fuel allows
    total_waiting = sum(s.waiting_passengers for s in obs.stops if not s.visited)
    remaining_capacity = bus.capacity - bus.passengers_onboard
    if (total_waiting > remaining_capacity * 1.5
            and not bus.second_bus_dispatched
            and bus.fuel_level > 50):
        return Action.DISPATCH_SECOND_BUS

    # 3. Skip if next stop is empty and we're behind schedule
    if cur_id + 1 < n_stops:
        next_stop = obs.stops[cur_id + 1]
        if next_stop.waiting_passengers == 0 and not next_stop.visited:
            return Action.SKIP_STOP

    # 4. Express if fuel is good and we're behind schedule
    steps_behind = obs.bus.total_delay_steps
    if steps_behind > 3 and bus.fuel_level > 40 and cur_id + 2 < n_stops:
        return Action.REROUTE_EXPRESS

    # 5. Default: move forward
    return Action.MOVE_TO_NEXT_STOP


# ── Reproducible Scoring ──────────────────────────────────────────────────────


SEEDS = [42, 123, 777, 2024, 99]


def run_all_baselines(seeds=SEEDS, verbose=True) -> dict:
    """
    Run all 3 agents × 3 tasks × N seeds and return mean scores.
    Fully reproducible given fixed seeds.
    """
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from tasks.task1_easy import run_task as run_t1
    from tasks.task2_medium import run_task as run_t2
    from tasks.task3_hard import run_task as run_t3

    agents = {
        "random":  lambda obs: random_agent(obs),
        "greedy":  greedy_agent,
        "optimal": optimal_agent,
    }

    tasks = {
        "task1_easy":   run_t1,
        "task2_medium": run_t2,
        "task3_hard":   run_t3,
    }

    results = {}

    for agent_name, agent_fn in agents.items():
        results[agent_name] = {}
        for task_name, task_fn in tasks.items():
            scores = []
            for seed in seeds:
                try:
                    result = task_fn(agent_fn, seed=seed)
                    scores.append(result.score)
                except Exception as e:
                    if verbose:
                        print(f"  ERROR: {agent_name}/{task_name}/seed={seed}: {e}")
                    scores.append(0.0)

            mean_score = round(sum(scores) / len(scores), 4)
            results[agent_name][task_name] = {
                "mean_score": mean_score,
                "scores":     [round(s, 4) for s in scores],
                "min":        round(min(scores), 4),
                "max":        round(max(scores), 4),
            }

            if verbose:
                print(f"  [{agent_name:7s}] {task_name:15s}  mean={mean_score:.4f}  "
                      f"min={min(scores):.4f}  max={max(scores):.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run Takshashila Transit baselines")
    parser.add_argument("--task", type=int, choices=[1, 2, 3], default=None,
                        help="Run single task (1=easy, 2=medium, 3=hard)")
    parser.add_argument("--agent", choices=["random", "greedy", "optimal"], default=None,
                        help="Run single agent")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    print("\n" + "═" * 60)
    print("  Takshashila Transit OpenEnv — Baseline Inference")
    print("═" * 60 + "\n")

    if args.task is None and args.agent is None:
        # Full benchmark
        print("Running full benchmark (3 agents × 3 tasks × 5 seeds)...\n")
        results = run_all_baselines(verbose=True)

        print("\n" + "═" * 60)
        print("  SUMMARY TABLE")
        print("═" * 60)
        print(f"  {'Agent':10s}  {'Task1 (Easy)':15s}  {'Task2 (Med)':13s}  {'Task3 (Hard)':12s}")
        print("  " + "-" * 56)
        for agent, tasks in results.items():
            t1 = tasks["task1_easy"]["mean_score"]
            t2 = tasks["task2_medium"]["mean_score"]
            t3 = tasks["task3_hard"]["mean_score"]
            print(f"  {agent:10s}  {t1:<15.4f}  {t2:<13.4f}  {t3:.4f}")
        print()

    else:
        # Single run
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from tasks.task1_easy import run_task as run_t1
        from tasks.task2_medium import run_task as run_t2
        from tasks.task3_hard import run_task as run_t3

        agent_map = {"random": random_agent, "greedy": greedy_agent, "optimal": optimal_agent}
        task_map = {1: run_t1, 2: run_t2, 3: run_t3}

        agent_fn = agent_map.get(args.agent or "optimal")
        task_fn = task_map.get(args.task or 1)

        result = task_fn(agent_fn, seed=args.seed)
        print(f"  Score: {result.score:.4f}")
        print(f"  Breakdown: {result.breakdown}")
        results = {"score": result.score, "breakdown": result.breakdown}

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved to {args.output}")


if __name__ == "__main__":
    main()
