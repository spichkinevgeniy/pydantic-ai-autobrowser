from pydantic import BaseModel


class OrchestratorRunResult(BaseModel):
    user_query: str
    plan: str
    next_step: str
    final_response: str = ""
