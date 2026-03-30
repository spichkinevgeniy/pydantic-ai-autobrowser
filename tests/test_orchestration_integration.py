import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orchestrator.runner import run_orchestration
from src.types import BrowserStepResult, HumanActionRequest, HumanActionResponse
from src.utils.msg_parser import ConversationStorage


class FakeAgentResult:
    def __init__(self, output: object) -> None:
        self.output = output
        self._all_messages: list[object] = []

    def new_messages(self) -> list[object]:
        return []

    def all_messages(self) -> list[object]:
        return list(self._all_messages)


class FakePlaywrightServer:
    def __init__(self) -> None:
        self.is_running = False
        self.enter_calls = 0
        self.exit_calls = 0

    async def __aenter__(self) -> "FakePlaywrightServer":
        self.is_running = True
        self.enter_calls += 1
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.is_running = False
        self.exit_calls += 1


def test_run_orchestration_handles_manual_human_action(monkeypatch, tmp_path: Path) -> None:
    import asyncio

    import src.config as config_module
    import src.orchestrator.engine as engine_module
    import src.orchestrator.state as state_module
    import src.orchestrator.workflow as workflow_module

    fake_server = FakePlaywrightServer()
    browser_calls: list[HumanActionResponse | None] = []
    events: list[str] = []
    original_reset_for_run = state_module.OrchestratorState.reset_for_run

    async def fake_create_plan(user_query: str, message_history=None) -> FakeAgentResult:
        return FakeAgentResult(
            SimpleNamespace(
                plan="1. Open the login page\n2. Wait for manual login\n3. Inspect the dashboard",
                next_step="Open the login page and request manual login if authentication is required",
            )
        )

    async def fake_run_browser_step(
        current_step: str,
        human_response: HumanActionResponse | None = None,
        message_history=None,
    ) -> FakeAgentResult:
        browser_calls.append(human_response)

        if human_response is None:
            return FakeAgentResult(
                BrowserStepResult(
                    status="blocked_for_human",
                    summary="Authentication is required before the task can continue.",
                    human_action=HumanActionRequest(
                        kind="login",
                        instruction="Complete login in the browser, then continue.",
                        response_mode="manual_confirmation",
                    ),
                )
            )

        return FakeAgentResult(
            BrowserStepResult(
                status="completed",
                summary="The dashboard is visible after manual login.",
            )
        )

    async def fake_run_critique(
        current_step: str,
        orignal_plan: str,
        tool_response: str,
        ss_analysis: str = "",
        message_history=None,
    ) -> FakeAgentResult:
        return FakeAgentResult(
            SimpleNamespace(
                feedback="The dashboard is open and the task can be finished.",
                terminate=True,
                final_response="Пользователь вошел в систему, дашборд открыт.",
            )
        )

    def fake_reset_for_run(self, user_query: str) -> None:
        original_reset_for_run(self, user_query)
        self.conversation_storage = ConversationStorage(tmp_path)

    def on_event(event) -> None:
        events.append(event.event_type)

    def human_input_handler(request: HumanActionRequest) -> HumanActionResponse:
        assert request.kind == "login"
        assert request.response_mode == "manual_confirmation"
        return HumanActionResponse(action="manual_done")

    monkeypatch.setattr(config_module.settings, "ENABLE_SCREENSHOTS", False)
    monkeypatch.setattr(config_module.settings, "ENABLE_SS_ANALYSIS", False)
    monkeypatch.setattr(engine_module, "get_playwright_mcp_server", lambda: fake_server)
    monkeypatch.setattr(workflow_module, "get_current_browser_url", fake_current_browser_url)
    monkeypatch.setattr(workflow_module, "create_plan", fake_create_plan)
    monkeypatch.setattr(workflow_module, "run_browser_step", fake_run_browser_step)
    monkeypatch.setattr(workflow_module, "run_critique", fake_run_critique)
    monkeypatch.setattr(state_module.OrchestratorState, "reset_for_run", fake_reset_for_run)

    result = asyncio.run(
        run_orchestration(
            "Открой мой дашборд после логина",
            on_event=on_event,
            human_input_handler=human_input_handler,
        )
    )

    assert result.final_response == "Пользователь вошел в систему, дашборд открыт."
    assert result.next_step == (
        "Open the login page and request manual login if authentication is required"
    )
    assert browser_calls == [None, HumanActionResponse(action="manual_done", value="")]
    assert fake_server.enter_calls == 1
    assert fake_server.exit_calls == 1
    assert events == [
        "run_started",
        "iteration_started",
        "planner_completed",
        "run_paused",
        "human_manual_action_requested",
        "human_manual_action_confirmed",
        "run_resumed",
        "browser_completed",
        "critique_completed",
        "run_finished",
    ]
    assert any(tmp_path.glob("task_conversation_*.json"))


async def fake_current_browser_url() -> str:
    return "about:blank"
