from dataclasses import dataclass, field
from typing import Any

from src.types import HumanActionRequest, HumanActionResponse
from src.utils.msg_parser import AgentConversationHandler, ConversationStorage
from src.utils.screenshot import ScreenshotHelper


@dataclass
class OrchestratorState:
    terminate: bool = False
    iteration_counter: int = 0
    consecutive_step_errors: int = 0
    max_step_errors: int = 3
    plan: str = ""
    current_step: str = ""
    planner_prompt: str = ""
    waiting_for_user: bool = False
    pending_human_request: HumanActionRequest | None = None
    pending_human_response: HumanActionResponse | None = None
    message_histories: dict[str, list[Any]] = field(
        default_factory=lambda: {
            "planner": [],
            "browser": [],
            "critique": [],
        }
    )
    conversation_handler: AgentConversationHandler = field(default_factory=AgentConversationHandler)
    conversation_storage: ConversationStorage = field(default_factory=ConversationStorage)
    screenshot_helper: ScreenshotHelper = field(default_factory=ScreenshotHelper)

    def reset_for_run(self, user_query: str) -> None:
        self.terminate = False
        self.iteration_counter = 0
        self.consecutive_step_errors = 0
        self.plan = ""
        self.current_step = ""
        self.planner_prompt = f"User Query: {user_query}\nFeedback: None"
        self.waiting_for_user = False
        self.pending_human_request = None
        self.pending_human_response = None
        self.message_histories = {
            "planner": [],
            "browser": [],
            "critique": [],
        }
        self.conversation_handler = AgentConversationHandler()
        self.conversation_storage.reset_file()
