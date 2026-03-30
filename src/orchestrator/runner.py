from src.orchestrator.engine import Orchestrator
from src.orchestrator.events import EventHandler
from src.types import HumanInputHandler, OrchestratorRunResult


async def run_orchestration(
    user_query: str,
    on_event: EventHandler | None = None,
    human_input_handler: HumanInputHandler | None = None,
) -> OrchestratorRunResult:
    orchestrator = Orchestrator()
    return await orchestrator.run(
        user_query,
        on_event=on_event,
        human_input_handler=human_input_handler,
    )
