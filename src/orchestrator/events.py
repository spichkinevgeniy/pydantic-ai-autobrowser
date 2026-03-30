from collections.abc import Callable
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, Field

EventType = Literal[
    "run_started",
    "iteration_started",
    "planner_completed",
    "browser_running",
    "browser_completed",
    "critique_completed",
    "run_paused",
    "run_resumed",
    "human_input_requested",
    "human_input_received",
    "human_manual_action_requested",
    "human_manual_action_confirmed",
    "security_approval_requested",
    "security_approval_received",
    "security_action_rejected",
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
