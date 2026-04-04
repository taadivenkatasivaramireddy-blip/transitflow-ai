---
title: Transitlyiq
emoji: 🚌
colorFrom: blue
colorTo: green
sdk: fastapi
app_file: app.py
pinned: false
---
# 🚌 Takshashila Transit — OpenEnv 

**Real-world college bus fleet management environment**  
`TakshashilaTransit-v1` · OpenEnv 1.0 · Takshashila University, Ongur, Tamil Nadu

[![HuggingFace Space](https://img.shields.io/badge/🤗-HuggingFace%20Space-blue)](https://huggingface.co/spaces)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-brightgreen)](https://python.org)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-1.0-orange)](openenv.yaml)

---

## Overview

Takshashila Transit is a **fully spec-compliant OpenEnv environment** built on a real operational problem: managing the campus bus service at Takshashila University. An AI agent acts as the bus dispatcher, controlling a single bus that serves 8 stops across the campus during the morning rush (7:30–9:00 AM).

The environment is grounded in actual campus geography and real passenger demand patterns, making it suitable for benchmarking RL agents, LLM planners, and rule-based schedulers on a non-trivial real-world logistics task.

---

## Scenario

- **8 stops**: Main Gate → Admin Block → Central Library → Hostel A → Hostel B → Sports Complex → Medical Block → Canteen Hub
- **20 time steps** per episode (~4.5 minutes each)
- **64 base passengers** across stops (± variance in medium/hard)
- **Fuel management**: starts at 100% (70% in hard mode), costs per action
- **Random events** in medium/hard: traffic jams, rain surges, gate closures

---

## Quick Start

```bash
git clone https://github.com/<your-username>/transit-openenv
cd transit-openenv
pip install -r requirements.txt
```

```python
from transit_env import TransitEnv
from transit_env.models import Action

# Initialise
env = TransitEnv(seed=42, difficulty="medium")

# reset() → Observation (Pydantic model)
obs = env.reset()
print(obs.bus.current_stop_id)       # 0 (Main Gate)
print(obs.stops[0].waiting_passengers)  # 8

# step(action) → StepResult
result = env.step(Action.PICKUP_PASSENGERS)
print(result.reward)   # +12.0
print(result.done)     # False

# state() → Observation (no side effects)
obs = env.state()

# Render to terminal
print(env.render())
```

---

## API Reference

### `env.reset(seed=None) → Observation`

Resets the environment. Returns the initial observation.

### `env.step(action: int | Action) → StepResult`

Advances the environment by one step.

Returns `StepResult(observation, reward, done, info)` — all typed via Pydantic.

### `env.state() → Observation`

Returns the current observation without modifying state.

---

## Action Space

| ID | Name | Description | Fuel Cost |
|----|------|-------------|-----------|
| 0 | `MOVE_TO_NEXT_STOP` | Advance to next scheduled stop | −7 |
| 1 | `WAIT_AT_STOP` | Hold at stop (max 2 steps before penalty) | 0 |
| 2 | `PICKUP_PASSENGERS` | Board all waiting passengers | 0 |
| 3 | `SKIP_STOP` | Skip next stop (penalty if passengers waiting) | −7 |
| 4 | `REROUTE_EXPRESS` | Skip 2 stops via express lane | −15 |
| 5 | `DISPATCH_SECOND_BUS` | Deploy overflow bus (one-time, handles surplus) | −30 |

---

## Observation Space

All fields typed via Pydantic. See `transit_env/models.py`.

```
Observation
├── step:                   int          [0, 20]
├── max_steps:              int          20
├── bus:
│   ├── current_stop_id:    int          [0, 7]
│   ├── passengers_onboard: int          [0, 40]
│   ├── fuel_level:         float        [0.0, 100.0]
│   ├── speed_kmh:          float
│   ├── total_delay_steps:  int
│   └── second_bus_dispatched: bool
├── stops: List[StopState]  (×8)
│   ├── waiting_passengers: int
│   ├── scheduled_arrival_step: int
│   ├── actual_arrival_step: Optional[int]
│   └── visited: bool
├── route_progress:         float        [0.0, 1.0]
├── on_time_stops:          int
├── late_stops:             int
├── missed_stops:           int
├── total_passengers_served: int
├── total_passengers_missed: int
└── episode_reward_so_far:  float
```

---

## Reward Function

Dense shaped reward with partial progress signals:

| Event | Reward |
|-------|--------|
| On-time arrival (≤1 step delay) | **+5.0** |
| Slightly late (2–3 steps) | +2.0 to −1.5 |
| Very late (>3 steps) | −delay (capped at −8) |
| Passenger pickup | **+1.5 per passenger** |
| Passenger overflow (missed) | −0.5 per passenger |
| Valid skip (empty stop) | +1.0 |
| Invalid skip (passengers abandoned) | −2.0 per passenger |
| Route completion bonus | **+20.0** |
| Schedule completion bonus | +3.0 per on-time stop |
| Per-step cost | −0.5 |
| Fuel depletion | −20.0 |
| Random event (medium/hard) | −5.0 to 0.0 |

---

## Tasks & Graders

### Task 1 — Easy (`tasks/task1_easy.py`)

**Objective**: Pick up passengers from the first 3 stops within 8 steps.

| Component | Weight |
|-----------|--------|
| Stop coverage (3 stops) | 50% |
| Passenger collection rate | 30% |
| Schedule adherence (≤2 step delay) | 20% |

Baseline: random=0.25, greedy=0.80

---

### Task 2 — Medium (`tasks/task2_medium.py`)

**Objective**: Complete the full route, serve ≥70% passengers, keep fuel >20%.

| Component | Weight |
|-----------|--------|
| Route completion | 35% |
| Passenger throughput | 35% |
| Fuel efficiency | 15% |
| Schedule adherence | 15% |

Baseline: random=0.15, greedy=0.55

---

### Task 3 — Hard (`tasks/task3_hard.py`)

**Objective**: Optimal fleet management with random events and 70% start fuel.

| Component | Weight |
|-----------|--------|
| Strict schedule (≥6/8 on time) | 25% |
| Zero missed stops | 20% |
| Overflow management | 15% |
| Reward efficiency | 20% |
| Passenger density | 20% |

Baseline: random=0.08, greedy=0.38, optimal target=0.75+

---

## Baseline Agents

```bash
# Run full benchmark (3 agents × 3 tasks × 5 seeds)
python baselines/run_baselines.py

# Run single agent on single task
python baselines/run_baselines.py --task 2 --agent greedy --seed 42

# Save results
python baselines/run_baselines.py --output results.json
```

### Reproducible Scores (seeds: 42, 123, 777, 2024, 99)

| Agent   | Task 1 (Easy) | Task 2 (Medium) | Task 3 (Hard) |
|---------|--------------|-----------------|---------------|
| Random  | ~0.25        | ~0.15           | ~0.08         |
| Greedy  | ~0.80        | ~0.55           | ~0.38         |
| Optimal | ~0.90        | ~0.72           | ~0.60         |

---

## Project Structure

```
transit-openenv/
├── transit_env/
│   ├── __init__.py          # Package exports
│   ├── env.py               # TransitEnv (reset/step/state/render)
│   └── models.py            # Typed Pydantic models
├── tasks/
│   ├── task1_easy.py        # Easy task + grader
│   ├── task2_medium.py      # Medium task + grader
│   └── task3_hard.py        # Hard task + grader
├── baselines/
│   └── run_baselines.py     # Random / Greedy / Optimal agents + benchmark
├── app.py                   # Gradio UI for HuggingFace Spaces
├── openenv.yaml             # Full OpenEnv 1.0 spec
├── Dockerfile               # HuggingFace Spaces deployment
├── requirements.txt
├── setup.py
└── README.md
```

---

## Deploy to HuggingFace Spaces

1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces)
2. Set SDK to **Docker**
3. Push this repository:

```bash
git remote add hf https://huggingface.co/spaces/<username>/transit-openenv
git push hf main
```

The Dockerfile handles everything. The Space will launch at `https://huggingface.co/spaces/<username>/transit-openenv`.

---

## About

Built by **Ram**, BBA FinTech (Year 1), Takshashila University, Ongur, Tamil Nadu.

This environment is directly inspired by **Takshashila Transit** — a real campus bus live-tracking platform under development for Takshashila University, featuring Firebase real-time GPS, OSRM routing, and geofence alerts.

The OpenEnv simulation abstracts the operational dispatch problem into a clean RL interface that researchers can benchmark agents against without needing access to the live system.
