from src.orchestrator.engine import Orchestrator
from src.orchestrator.events import EventHandler
from src.types import OrchestratorRunResult


async def run_orchestration(
    user_query: str,
    on_event: EventHandler | None = None,
) -> OrchestratorRunResult:
    orchestrator = Orchestrator()
    return await orchestrator.run(user_query, on_event=on_event)
