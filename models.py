"""
Typed data models for Takshashila Transit OpenEnv.
All observations and actions are strictly typed via Pydantic.
"""

from __future__ import annotations
from enum import IntEnum
from typing import List, Optional
from pydantic import BaseModel, Field


class Action(IntEnum):
    """
    Discrete action space for the bus agent.

    Each action represents a real operational decision a bus driver/dispatcher makes.
    """
    MOVE_TO_NEXT_STOP   = 0  # Advance bus to the next scheduled stop
    WAIT_AT_STOP        = 1  # Hold at current stop (for late passengers)
    PICKUP_PASSENGERS   = 2  # Open doors and pick up waiting passengers
    SKIP_STOP           = 3  # Skip next stop (used when no passengers are waiting)
    REROUTE_EXPRESS     = 4  # Take express lane, skip 2 stops, saves time, costs fuel
    DISPATCH_SECOND_BUS = 5  # Dispatch a secondary bus for overflow passengers


class ActionSpace(BaseModel):
    """Describes the full action space."""
    n: int = 6
    actions: List[str] = [
        "MOVE_TO_NEXT_STOP",
        "WAIT_AT_STOP",
        "PICKUP_PASSENGERS",
        "SKIP_STOP",
        "REROUTE_EXPRESS",
        "DISPATCH_SECOND_BUS",
    ]
    descriptions: List[str] = [
        "Advance to the next stop on the route",
        "Hold at current stop up to 2 extra steps",
        "Pick up all waiting passengers at current stop",
        "Skip the next stop (only valid if 0 passengers waiting there)",
        "Express route: skip 2 stops, -15 fuel, saves 2 time steps",
        "Dispatch overflow bus: -30 fuel, handles excess passengers",
    ]


class StopState(BaseModel):
    """State of a single bus stop."""
    stop_id: int
    name: str
    waiting_passengers: int = Field(ge=0, description="Passengers waiting at this stop")
    scheduled_arrival_step: int = Field(description="Planned arrival step in episode")
    actual_arrival_step: Optional[int] = Field(default=None, description="Actual arrival step (None if not yet visited)")
    visited: bool = False
    distance_km: float = Field(description="Distance from previous stop in km")


class BusState(BaseModel):
    """State of the primary bus."""
    current_stop_id: int = Field(description="Stop index the bus is currently at")
    next_stop_id: Optional[int] = Field(default=None, description="Next stop in route")
    passengers_onboard: int = Field(ge=0, description="Current passenger count")
    capacity: int = Field(default=40, description="Max passenger capacity")
    fuel_level: float = Field(ge=0.0, le=100.0, description="Fuel percentage")
    speed_kmh: float = Field(description="Current speed in km/h")
    total_delay_steps: int = Field(default=0, description="Cumulative delay (steps behind schedule)")
    is_moving: bool = False
    second_bus_dispatched: bool = False


class Observation(BaseModel):
    """
    Full observation returned by state() and step().
    This is what the agent sees at every timestep.
    """
    # Time
    step: int = Field(description="Current step in episode (0-indexed)")
    max_steps: int = Field(description="Maximum steps allowed in this episode")

    # Bus
    bus: BusState

    # Route
    stops: List[StopState]
    route_progress: float = Field(ge=0.0, le=1.0, description="Fraction of route completed")
    
    # Schedule adherence
    on_time_stops: int = Field(default=0, description="Stops visited within 1 step of schedule")
    late_stops: int = Field(default=0, description="Stops arrived late")
    missed_stops: int = Field(default=0, description="Stops skipped with waiting passengers")

    # Passengers
    total_passengers_served: int = Field(default=0)
    total_passengers_missed: int = Field(default=0)

    # Episode info
    done: bool = False
    episode_reward_so_far: float = 0.0


class StepResult(BaseModel):
    """Result of calling step(action)."""
    observation: Observation
    reward: float = Field(description="Reward for this step")
    done: bool = Field(description="Whether the episode has ended")
    info: dict = Field(default_factory=dict, description="Debug/diagnostic info")

    class Config:
        arbitrary_types_allowed = True
