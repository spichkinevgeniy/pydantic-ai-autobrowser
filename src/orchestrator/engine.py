import logging
from typing import Any

from src.agents.browser_agent import get_playwright_mcp_server
from src.orchestrator.events import EventHandler, EventType, OrchestratorEvent
from src.orchestrator.state import OrchestratorState
from src.orchestrator.workflow import run_workflow
from src.types import OrchestratorRunResult
from src.utils.image_analysis import ImageAnalyzer


class Orchestrator:
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self._started = False
        self._browser_server_entered = False
        self.state = OrchestratorState()
        self._event_handler: EventHandler | None = None
        ImageAnalyzer.clear_history()

    async def run(
        self,
        user_query: str,
        on_event: EventHandler | None = None,
    ) -> OrchestratorRunResult:
        self._event_handler = on_event
        self.state.reset_for_run(user_query)
        ImageAnalyzer.clear_history()
        await self.start()
        self.emit_event(
            "run_started",
            message="Orchestration started",
            data={"user_query": user_query},
        )

        try:
            return await run_workflow(self, user_query)
        except Exception as exc:
            self.logger.exception("Orchestrator failed during execution")
            self.emit_event("run_failed", message=str(exc))
            raise
        finally:
            await self.shutdown()
            self._event_handler = None

    def emit_event(
        self,
        event_type: EventType,
        *,
        message: str = "",
        iteration: int | None = None,
        current_step: str = "",
        plan: str = "",
        final_response: str = "",
        data: dict[str, Any] | None = None,
    ) -> None:
        if self._event_handler is None:
            return

        event = OrchestratorEvent(
            event_type=event_type,
            message=message,
            iteration=iteration,
            current_step=current_step,
            plan=plan,
            final_response=final_response,
            data=data or {},
        )
        self._event_handler(event)

    async def start(self) -> None:
        if self._started:
            self.logger.debug("Start skipped: orchestrator already running")
            return

        await get_playwright_mcp_server().__aenter__()
        self._browser_server_entered = True
        self._started = True
        self.logger.info("Orchestrator started")

    async def shutdown(self) -> None:
        if not self._started:
            return

        self.logger.info("Orchestrator shutdown started")
        try:
            await self.cleanup()
        finally:
            if self._browser_server_entered:
                await get_playwright_mcp_server().__aexit__(None, None, None)
                self._browser_server_entered = False
            self._started = False
            self.logger.info("Orchestrator shutdown completed")

    async def cleanup(self) -> None:
        self.logger.info("Cleanup completed")

    async def wait_for_exit(self) -> None:
        self.logger.debug("wait_for_exit called; no background workers registered")
