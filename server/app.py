import sys
import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transit_env import TransitEnv

app = FastAPI()

try:
    env = TransitEnv(seed=42, difficulty="medium")
except Exception as e:
    print(f"Init Error: {e}")
    env = TransitEnv()

class ActionRequest(BaseModel):
    action: int

def serialize_obs(obs):
    """Recursively convert custom objects to standard dicts for JSON safety"""
    if hasattr(obs, '__dict__'):
        return {k: serialize_obs(v) for k, v in obs.__dict__.items()}
    elif isinstance(obs, list):
        return [serialize_obs(i) for i in obs]
    return obs

@app.get("/")
async def health():
    return {"status": "ready"}

@app.post("/reset")
async def reset():
    try:
        # Capture the single observation
        obs = env.reset()
        safe_obs = serialize_obs(obs)
        return {"observation": safe_obs, "info": {}}
    except Exception as e:
        return {"error": str(e)}

@app.post("/step")
async def step(request: ActionRequest):
    try:
        # Capture the single result object
        result = env.step(request.action)
        
        obs = getattr(result, 'observation', result)
        reward = float(getattr(result, 'reward', 0.0))
        done = bool(getattr(result, 'done', False))
        info = getattr(result, 'info', {})
        
        safe_obs = serialize_obs(obs)
        
        return {
            "observation": safe_obs,
            "reward": reward,
            "terminated": done,
            "truncated": False,
            "info": info
        }
    except Exception as e:
        return {"error": str(e)}

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
