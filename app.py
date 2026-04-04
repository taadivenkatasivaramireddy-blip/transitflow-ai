"""
app.py — Gradio interface for Takshashila Transit OpenEnv
Deployed on HuggingFace Spaces.
"""

import sys
import json
import gradio as gr
import pandas as pd

from transit_env import TransitEnv
from transit_env.models import Action
from tasks.task1_easy import run_task as run_t1
from tasks.task2_medium import run_task as run_t2
from tasks.task3_hard import run_task as run_t3, HardTransitEnv
from baselines.run_baselines import random_agent, greedy_agent, optimal_agent

# ── Global state ──────────────────────────────────────────────────────────────

_env: TransitEnv | None = None
_trajectory = []
_current_obs = None

ACTION_NAMES = [a.name for a in Action]


def init_env(difficulty: str, seed: int):
    global _env, _trajectory, _current_obs
    _env = TransitEnv(seed=int(seed), difficulty=difficulty)
    _current_obs = _env.reset()
    _trajectory = []
    return _env.render(), get_state_table(), "✅ Environment reset. Ready to step."


def do_step(action_name: str):
    global _current_obs, _trajectory
    if _env is None:
        return "❌ Call reset() first.", None, ""
    if _current_obs.done:
        return _env.render(), get_state_table(), "⚠️ Episode done. Reset to play again."

    action_id = ACTION_NAMES.index(action_name)
    result = _env.step(action_id)
    _trajectory.append((action_id, result.observation))
    _current_obs = result.observation

    events = " | ".join(result.info.get("events", []))
    status = f"Step {_current_obs.step} | Reward: {result.reward:+.2f} | {events}"
    if result.done:
        status += f"\n🏁 Episode complete! Total reward: {_current_obs.episode_reward_so_far:.2f}"

    return _env.render(), get_state_table(), status


def get_state_table():
    if _env is None or _current_obs is None:
        return pd.DataFrame()
    stops = _current_obs.stops
    rows = []
    for s in stops:
        status = "✓ Visited" if s.visited else f"{s.waiting_passengers} waiting"
        current = "🚌 HERE" if s.stop_id == _current_obs.bus.current_stop_id else ""
        rows.append({
            "Stop": s.name,
            "Status": status,
            "Sched.": s.scheduled_arrival_step,
            "Actual": s.actual_arrival_step or "—",
            "Bus": current,
        })
    return pd.DataFrame(rows)


def run_benchmark(agent_name: str, seed: int):
    agent_map = {"Random": random_agent, "Greedy": greedy_agent, "Optimal": optimal_agent}
    agent_fn = agent_map[agent_name]

    rows = []
    for task_fn, task_label, diff in [
        (run_t1, "Task 1 — Easy",   "easy"),
        (run_t2, "Task 2 — Medium", "medium"),
        (run_t3, "Task 3 — Hard",   "hard"),
    ]:
        result = task_fn(agent_fn, seed=int(seed))
        breakdown_str = " | ".join(f"{k}: {v:.3f}" for k, v in result.breakdown.items())
        rows.append({
            "Task":      task_label,
            "Score":     f"{result.score:.4f}",
            "Breakdown": breakdown_str,
        })

    return pd.DataFrame(rows)


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="Takshashila Transit OpenEnv",
    theme=gr.themes.Base(primary_hue="emerald"),
) as demo:

    gr.Markdown("""
# 🚌 Takshashila Transit — OpenEnv
**Real-world college bus fleet management environment**  
Takshashila University, Ongur, Tamil Nadu · `TakshashilaTransit-v1`

> An AI agent controls a campus bus dispatcher, serving 8 stops in a morning route.
> API: `env.reset()` → `env.step(action)` → `env.state()`
""")

    with gr.Tabs():

        # ── Tab 1: Interactive Play ───────────────────────────────────────────
        with gr.Tab("🎮 Interactive Environment"):
            with gr.Row():
                diff_dropdown = gr.Dropdown(
                    ["easy", "medium", "hard"], value="medium", label="Difficulty"
                )
                seed_input = gr.Number(value=42, label="Seed", precision=0)
                reset_btn = gr.Button("↺ reset()", variant="primary")

            status_box = gr.Textbox(label="Step Info", lines=2)

            with gr.Row():
                render_box = gr.Textbox(label="env.render()", lines=16, max_lines=20)
                state_table = gr.Dataframe(label="Bus Stops State", wrap=True)

            action_dropdown = gr.Dropdown(
                ACTION_NAMES, value="MOVE_TO_NEXT_STOP", label="Action"
            )
            step_btn = gr.Button("▶ step(action)", variant="secondary")

            reset_btn.click(init_env, [diff_dropdown, seed_input],
                            [render_box, state_table, status_box])
            step_btn.click(do_step, [action_dropdown],
                           [render_box, state_table, status_box])

        # ── Tab 2: Benchmark ──────────────────────────────────────────────────
        with gr.Tab("📊 Baseline Benchmark"):
            gr.Markdown("Run a baseline agent across all 3 tasks and see scores.")
            with gr.Row():
                agent_dropdown = gr.Dropdown(
                    ["Random", "Greedy", "Optimal"], value="Optimal", label="Agent"
                )
                bench_seed = gr.Number(value=42, label="Seed", precision=0)
                bench_btn = gr.Button("Run Benchmark", variant="primary")

            bench_table = gr.Dataframe(label="Results (score 0.0 – 1.0)", wrap=True)
            bench_btn.click(run_benchmark, [agent_dropdown, bench_seed], bench_table)

        # ── Tab 3: API Reference ──────────────────────────────────────────────
        with gr.Tab("📖 API Reference"):
            gr.Markdown("""
## Environment API

```python
from transit_env import TransitEnv
from transit_env.models import Action

env = TransitEnv(seed=42, difficulty="medium")

# reset() → Observation
obs = env.reset()

# step(action) → StepResult(observation, reward, done, info)
result = env.step(Action.MOVE_TO_NEXT_STOP)
print(result.reward, result.done)

# state() → Observation (no side effects)
obs = env.state()
```

## Action Space

| ID | Name | Description | Fuel Cost |
|----|------|-------------|-----------|
| 0  | MOVE_TO_NEXT_STOP | Advance to next stop | 7 |
| 1  | WAIT_AT_STOP | Hold at current stop | 0 |
| 2  | PICKUP_PASSENGERS | Board waiting passengers | 0 |
| 3  | SKIP_STOP | Skip next stop | 7 |
| 4  | REROUTE_EXPRESS | Skip 2 stops fast | 15 |
| 5  | DISPATCH_SECOND_BUS | Deploy overflow bus | 30 |

## Reward Signal

| Event | Reward |
|-------|--------|
| On-time arrival (≤1 step) | +5.0 |
| Passenger pickup | +1.5 per passenger |
| Very late arrival | −delay (max −8) |
| Route completion | +20.0 |
| Fuel depletion | −20.0 |
| Per step cost | −0.5 |

## Tasks

| Task | Difficulty | Score Target | Baseline (greedy) |
|------|------------|-------------|-------------------|
| task1_easy | easy | 0.80+ | 0.80 |
| task2_medium | medium | 0.65+ | 0.55 |
| task3_hard | hard | 0.75+ | 0.38 |
""")

    gr.Markdown("""
---
Built by **Ram** · BBA FinTech · Takshashila University  
[GitHub](https://github.com) · [openenv.yaml](openenv.yaml)
""")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
