import os
import textwrap
from typing import List, Optional
from openai import OpenAI

from transit_env import TransitEnv
from models import Action

# --- Mandatory Hackathon Variables ---
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
"sk-dummy-key-to-stop-crash"
TASK_NAME = "task1_easy"
BENCHMARK = "TakshashilaTransit-v1"
MAX_STEPS = 20
SUCCESS_SCORE_TARGET = 15.0  # Adjust based on your max possible score

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
        # Fallbacks to prevent script crashing if the object structure shifts
        fuel = getattr(obs.bus, 'fuel', "N/A") if hasattr(obs, 'bus') else "N/A"
        passengers = getattr(obs.bus, 'passengers', "N/A") if hasattr(obs, 'bus') else "N/A"

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
        # Bulletproof extraction: grabs the first digit found
        action_num = int(''.join(filter(str.isdigit, text)) or 0)
        return max(0, min(5, action_num))
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return 0

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) "sk-dummy-key-to-bypass-hf-error"
    env = TransitEnv(seed=42, difficulty="medium")

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs, info = env.reset()
        
        for step in range(1, MAX_STEPS + 1):
            action_id = get_model_action(client, step, obs)
            
            try:
                action_enum = list(Action)[action_id]
                obs, reward, terminated, truncated, info = env.step(action_enum)
                error = None
            except Exception as e:
                action_enum = Action.MOVE_TO_NEXT_STOP # Failsafe
                reward = -1.0
                terminated = True
                truncated = False
                error = str(e)
            
            done = terminated or truncated
            rewards.append(float(reward))
            steps_taken = step
            
            log_step(step=step, action=action_enum.name, reward=reward, done=done, error=error)

            if done:
                break

        # Mandatory Hackathon score normalization [0.0 to 1.0]
        total_reward = sum(rewards)
        score = max(0.0, min(1.0, total_reward / SUCCESS_SCORE_TARGET))
        success = score >= 0.1  # Matches the sample script's threshold

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    main()
