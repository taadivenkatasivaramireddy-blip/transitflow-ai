import sys
import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

# Inject the parent directory into the path so it can find your transit logic
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transit_env import TransitEnv

app = FastAPI()

# Safely initialize the environment
try:
    env = TransitEnv(seed=42, difficulty="medium")
except Exception as e:
    print(f"Init Error: {e}")
    env = TransitEnv()

class ActionRequest(BaseModel):
    action: int

@app.get("/")
async def health():
    return {"status": "ready"}

@app.post("/reset")
async def reset():
    try:
        obs, info = env.reset()
        # Force JSON compliance
        if isinstance(obs, np.ndarray):
            obs = obs.tolist()
        elif hasattr(obs, '__dict__'):
            obs = obs.__dict__
        return {"observation": obs, "info": info}
    except Exception as e:
        return {"error": str(e)}

@app.post("/step")
async def step(request: ActionRequest):
    try:
        obs, reward, terminated, truncated, info = env.step(request.action)
        if isinstance(obs, np.ndarray):
            obs = obs.tolist()
        elif hasattr(obs, '__dict__'):
            obs = obs.__dict__
        return {
            "observation": obs,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": info
        }
    except Exception as e:
        return {"error": str(e)}

def main():
    # The unbreakable engine to keep Hugging Face alive
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
