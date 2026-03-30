from collections.abc import Callable
from typing import Literal, TypeAlias

from pydantic import BaseModel

HumanActionKind = Literal[
    "login",
    "password",
    "phone",
    "otp",
    "captcha",
    "manual_browser_action",
]
HumanResponseMode = Literal["provide_value", "manual_confirmation"]
HumanActionResponseType = Literal["provide_value", "manual_done", "abort"]
BrowserStepStatus = Literal["completed", "blocked_for_human"]


class HumanActionRequest(BaseModel):
    kind: HumanActionKind
    instruction: str
    prompt: str = ""
    response_mode: HumanResponseMode
    sensitive: bool = False


class HumanActionResponse(BaseModel):
    action: HumanActionResponseType
    value: str = ""


class BrowserStepResult(BaseModel):
    status: BrowserStepStatus
    summary: str
    answer: str = ""
    human_action: HumanActionRequest | None = None


class OrchestratorRunResult(BaseModel):
    user_query: str
    plan: str
    next_step: str
    final_response: str = ""


HumanInputHandler: TypeAlias = Callable[[HumanActionRequest], HumanActionResponse]
