from collections.abc import Callable
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, Field

EventType = Literal[
    "run_started",
    "iteration_started",
    "planner_completed",
    "browser_completed",
    "critique_completed",
    "run_failed",
    "run_finished",
]


class OrchestratorEvent(BaseModel):
    event_type: EventType
    message: str = ""
    iteration: int | None = None
    current_step: str = ""
    plan: str = ""
    final_response: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


EventHandler: TypeAlias = Callable[[OrchestratorEvent], None]
