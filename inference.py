import os
import textwrap
from typing import List, Optional
from openai import OpenAI

from transit_env import TransitEnv
from models import Action

# --- Mandatory Hackathon Variables ---
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "sk-dummy-key"
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK = "TakshashilaTransit-v1"
MAX_STEPS = 20
SUCCESS_SCORE_TARGET = 15.0  

# Define the 3 mandatory tasks the Judge is looking for
TASKS = [
    {"name": "task1_easy", "difficulty": "easy"},
    {"name": "task2_medium", "difficulty": "medium"},
    {"name": "task3_hard", "difficulty": "hard"}
]

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are the AI dispatcher for the campus bus system.
    Maximize passenger pickup, maintain schedule adherence, and manage fuel.
    
    Choose ONE action ID from the following options:
    0 = MOVE_TO_NEXT_STOP
    1 = WAIT_AT_STOP
    2 = PICKUP_PASSENGERS
    3 = SKIP_STOP
    4 = REROUTE_EXPRESS
    5 = DISPATCH_SECOND_BUS

    Reply with EXACTLY ONE NUMBER between 0 and 5. Do not add any text.
    """
).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def get_model_action(client: OpenAI, step: int, obs) -> int:
    try:
        bus = getattr(obs, 'bus', None)
        fuel = getattr(bus, 'fuel', "N/A") if bus else "N/A"
        passengers = getattr(bus, 'passengers', "N/A") if bus else "N/A"

        user_prompt = f"Step: {step} | Fuel: {fuel} | Passengers: {passengers}\nWhich action ID (0-5) will you take?"
        
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=10,
        )
        text = (completion.choices[0].message.content or "").strip()
        action_num = int(''.join(filter(str.isdigit, text)) or 0)
        return max(0, min(5, action_num))
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return 0

def run_single_task(client: OpenAI, task_name: str, difficulty: str) -> None:
    """Runs a single episode for the specified task and difficulty."""
    env = TransitEnv(seed=42, difficulty=difficulty)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        raw_reset = env.reset()
        obs = getattr(raw_reset, 'observation', raw_reset)
        if isinstance(raw_reset, tuple):
            obs = raw_reset[0]
        
        for step in range(1, MAX_STEPS + 1):
            action_id = get_model_action(client, step, obs)
            
            try:
                action_enum = list(Action)[action_id]
            except IndexError:
                action_enum = list(Action)[0]
                
            error = None
            reward = 0.0
            done = False
            
            try:
                result = env.step(action_enum)
                obs = getattr(result, 'observation', result)
                reward = float(getattr(result, 'reward', 0.0))
                done = bool(getattr(result, 'done', False))
                
                if isinstance(result, tuple):
                    obs = result[0]
                    reward = float(result[1])
                    done = bool(result[2])
                    
            except Exception as e:
                error = str(e)
                reward = -1.0
                done = True
            
            rewards.append(reward)
            steps_taken = step
            
            log_step(step=step, action=action_enum.name, reward=reward, done=done, error=error)

            if done:
                break

        total_reward = sum(rewards)
        raw_score = total_reward / SUCCESS_SCORE_TARGET
        
        # The clamp that fixed the out-of-range error!
        score = max(0.01, min(0.99, raw_score))
        success = score >= 0.1

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # Loop through all 3 tasks in a single execution to satisfy the Judge
    for task_info in TASKS:
        run_single_task(client, task_info["name"], task_info["difficulty"])

if __name__ == "__main__":
    main()
