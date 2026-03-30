import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypedDict

from src.agents.browser_agent import run_browser_step
from src.agents.critique_agent import run_critique
from src.agents.planner_agent import create_plan
from src.config import settings
from src.orchestrator.events import EventType
from src.orchestrator.helpers import (
    build_critique_tool_response,
    ensure_tool_response_sequence,
    extract_tool_interactions,
    filter_dom_messages,
    filter_tool_interactions_for_critique,
    format_payload,
    run_with_transient_retry,
    strip_snapshot_refs,
)
from src.types import (
    BrowserStepResult,
    HumanActionRequest,
    HumanActionResponse,
    OrchestratorRunResult,
)
from src.utils.image_analysis import ImageAnalyzer

if TYPE_CHECKING:
    from pydantic_ai.run import AgentRunResult
    from src.orchestrator.state import OrchestratorState

logger = logging.getLogger(__name__)
BROWSER_PROGRESS_HEARTBEAT_SECONDS = 5.0
REFUSAL_MARKERS = (
    "i am unable to",
    "i cannot",
    "cannot be performed",
    "unable to access",
    "security and privacy restrictions",
    "privacy restrictions",
)


class OrchestratorRuntime(Protocol):
    state: "OrchestratorState"

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
    ) -> None: ...

    def request_human_action(self, request: HumanActionRequest) -> HumanActionResponse: ...


class BrowserStageSuccess(TypedDict):
    browser_output: "AgentRunResult[BrowserStepResult]"
    browser_summary: str
    tool_interactions_str: str
    ss_analysis: str


class BrowserStageAborted(TypedDict):
    aborted_result: OrchestratorRunResult


def looks_like_refusal_text(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    return any(marker in normalized for marker in REFUSAL_MARKERS)


def build_human_assisted_replan_feedback(user_query: str, refusal_step: str) -> str:
    return (
        f"The planner produced a refusal-like next step instead of an executable browser action.\n"
        f"User query: {user_query}\n"
        f"Rejected next step: {refusal_step}\n\n"
        "Replan into a human-assisted browser workflow. If the task involves the user's own account, "
        "open the relevant website, allow the user to complete login or verification manually in the "
        "browser, then continue with atomic browser actions. Do not output another refusal unless the "
        "task is impossible even after manual user assistance."
    )


async def run_browser_step_with_progress(
    orchestrator: OrchestratorRuntime,
    *,
    current_step: str,
    message_history: list[Any],
    human_response: HumanActionResponse | None = None,
) -> "AgentRunResult[BrowserStepResult]":
    browser_task = asyncio.create_task(
        run_with_transient_retry(
            "browser",
            lambda: run_browser_step(
                current_step=current_step,
                human_response=human_response,
                message_history=message_history,
            ),
        )
    )
    heartbeat_count = 0

    while True:
        done, _ = await asyncio.wait(
            {browser_task},
            timeout=BROWSER_PROGRESS_HEARTBEAT_SECONDS,
        )
        if browser_task in done:
            return await browser_task

        heartbeat_count += 1
        elapsed_seconds = heartbeat_count * int(BROWSER_PROGRESS_HEARTBEAT_SECONDS)
        orchestrator.emit_event(
            "browser_running",
            message=f"Browser agent is still working ({elapsed_seconds}s)",
            current_step=current_step,
            data={"elapsed_seconds": elapsed_seconds},
        )


def should_persist_browser_artifacts(human_response: HumanActionResponse | None) -> bool:
    return not (
        human_response is not None
        and human_response.action == "provide_value"
        and bool(human_response.value)
    )


def build_user_abort_result(user_query: str, plan: str, current_step: str) -> OrchestratorRunResult:
    return OrchestratorRunResult(
        user_query=user_query,
        plan=plan,
        next_step=current_step,
        final_response="Execution aborted by the user during a required human action.",
    )


def request_human_action(
    orchestrator: OrchestratorRuntime,
    request: HumanActionRequest,
    *,
    iteration: int,
    current_step: str,
) -> HumanActionResponse:
    state = orchestrator.state
    state.waiting_for_user = True
    state.pending_human_request = request

    orchestrator.emit_event(
        "run_paused",
        message="Waiting for required human action",
        iteration=iteration,
        current_step=current_step,
        data={"kind": request.kind},
    )
    request_event_type = (
        "human_input_requested"
        if request.response_mode == "provide_value"
        else "human_manual_action_requested"
    )
    orchestrator.emit_event(
        request_event_type,
        message=request.instruction,
        iteration=iteration,
        current_step=current_step,
        data=request.model_dump(mode="json"),
    )

    response = orchestrator.request_human_action(request)
    state.pending_human_response = response

    if response.action == "provide_value":
        orchestrator.emit_event(
            "human_input_received",
            message="Human provided the requested value",
            iteration=iteration,
            current_step=current_step,
            data={"kind": request.kind, "sensitive": request.sensitive},
        )
    elif response.action == "manual_done":
        orchestrator.emit_event(
            "human_manual_action_confirmed",
            message="Human confirmed the manual browser action is complete",
            iteration=iteration,
            current_step=current_step,
            data={"kind": request.kind},
        )

    state.waiting_for_user = False
    state.pending_human_request = None
    orchestrator.emit_event(
        "run_resumed",
        message="Resuming execution after human action",
        iteration=iteration,
        current_step=current_step,
    )
    return response


async def execute_browser_stage(
    orchestrator: OrchestratorRuntime,
    *,
    user_query: str,
) -> BrowserStageSuccess | BrowserStageAborted:
    state = orchestrator.state
    tool_interactions_str: str | None = None
    browser_summary = ""
    ss_analysis = ""
    pre_action_ss: Path | None = None
    post_action_ss: Path | None = None
    active_human_response: HumanActionResponse | None = None

    browser_history = filter_dom_messages(state.message_histories["browser"])
    pre_action_ss = await state.screenshot_helper.capture(
        "pre",
        state.iteration_counter,
        full_page=settings.SCREENSHOT_FULL_PAGE,
    )

    while True:
        browser_output = await run_browser_step_with_progress(
            orchestrator,
            current_step=state.current_step,
            message_history=browser_history,
            human_response=active_human_response,
        )
        browser_summary = strip_snapshot_refs(browser_output.output.summary).strip()

        persist_browser_artifacts = should_persist_browser_artifacts(active_human_response)
        browser_new_messages = browser_output.new_messages()

        if persist_browser_artifacts:
            state.conversation_handler.add_browser_nav_message(browser_output)
            state.message_histories["browser"].extend(browser_new_messages)
            tool_interactions_str = extract_tool_interactions(browser_new_messages)
            all_messages = browser_output.all_messages()
            logger.info(
                "All messages from browser agent (%s messages):\n%s",
                len(all_messages),
                format_payload(all_messages),
            )
            logger.info(
                "Tool interactions from browser agent:\n%s",
                tool_interactions_str or "No tool interactions recorded.",
            )
        else:
            tool_interactions_str = ""
            logger.info(
                "Sensitive human input was used; raw browser history and tool interactions were not persisted"
            )

        if (
            browser_output.output.status == "blocked_for_human"
            and browser_output.output.human_action is not None
        ):
            active_human_response = request_human_action(
                orchestrator,
                browser_output.output.human_action,
                iteration=state.iteration_counter,
                current_step=state.current_step,
            )
            if active_human_response.action == "abort":
                state.terminate = True
                result = build_user_abort_result(user_query, state.plan, state.current_step)
                orchestrator.emit_event(
                    "run_finished",
                    message="Run aborted by the user",
                    iteration=state.iteration_counter,
                    current_step=state.current_step,
                    plan=state.plan,
                    final_response=result.final_response,
                )
                return BrowserStageAborted(
                    aborted_result=result,
                )

            browser_history = filter_dom_messages(state.message_histories["browser"])
            continue

        if browser_output.output.status == "blocked_for_human":
            raise RuntimeError(
                "Browser agent returned blocked_for_human without a human_action request."
            )
        break

    state.pending_human_response = None
    post_action_ss = await state.screenshot_helper.capture(
        "post",
        state.iteration_counter,
        full_page=settings.SCREENSHOT_FULL_PAGE,
    )
    logger.info(
        "Screenshot paths for iteration %s: pre=%s post=%s",
        state.iteration_counter,
        pre_action_ss,
        post_action_ss,
    )

    if pre_action_ss and post_action_ss:
        analyzer = ImageAnalyzer(
            image1_path=pre_action_ss,
            image2_path=post_action_ss,
            next_step=state.current_step,
        )
        ss_analysis = await analyzer.analyze_images()
        if ss_analysis:
            logger.info("Screenshot analysis:\n%s", ss_analysis)
        else:
            logger.info("Screenshot analysis skipped or returned empty output")
    else:
        logger.info("Screenshot analysis skipped because one or both screenshots are missing")

    orchestrator.emit_event(
        "browser_completed",
        message="Browser step executed",
        iteration=state.iteration_counter,
        current_step=state.current_step,
        data={
            "browser_summary": browser_summary,
            "ss_analysis": strip_snapshot_refs(ss_analysis),
            "pre_screenshot": str(pre_action_ss) if pre_action_ss else "",
            "post_screenshot": str(post_action_ss) if post_action_ss else "",
        },
    )
    return BrowserStageSuccess(
        browser_output=browser_output,
        browser_summary=browser_summary,
        tool_interactions_str=tool_interactions_str or "",
        ss_analysis=ss_analysis,
    )


async def run_workflow(
    orchestrator: OrchestratorRuntime,
    user_query: str,
) -> OrchestratorRunResult:
    state = orchestrator.state

    while not state.terminate:
        state.iteration_counter += 1
        orchestrator.emit_event(
            "iteration_started",
            message=f"Starting iteration {state.iteration_counter}",
            iteration=state.iteration_counter,
        )
        logger.info("Orchestrator iteration started: %s", state.iteration_counter)

        try:
            validated_history = ensure_tool_response_sequence(state.message_histories["planner"])
            logger.info(
                "Validated planner history (%s messages):\n%s",
                len(validated_history),
                format_payload(validated_history),
            )

            planner_output = await run_with_transient_retry(
                "planner",
                lambda: create_plan(
                    state.planner_prompt,
                    message_history=validated_history,
                ),
            )
            planner_new_messages = planner_output.new_messages()
            state.message_histories["planner"].extend(planner_new_messages)
            state.plan = planner_output.output.plan
            state.current_step = planner_output.output.next_step

            logger.info("Planner returned plan")
            logger.info("Current step: %s", state.current_step)
            if state.iteration_counter == 1:
                logger.info("Initial full plan:\n%s", state.plan)

            if looks_like_refusal_text(state.current_step):
                logger.warning(
                    "Planner returned refusal-like next step. Requesting human-assisted replan."
                )
                state.planner_prompt = build_human_assisted_replan_feedback(
                    user_query,
                    state.current_step,
                )
                continue

            orchestrator.emit_event(
                "planner_completed",
                message="Planner produced the next step",
                iteration=state.iteration_counter,
                current_step=state.current_step,
                plan=state.plan,
            )

            browser_stage = await execute_browser_stage(
                orchestrator,
                user_query=user_query,
            )
            if "aborted_result" in browser_stage:
                return browser_stage["aborted_result"]

            filtered_interactions = filter_tool_interactions_for_critique(
                browser_stage["tool_interactions_str"]
            )
            critique_tool_response = build_critique_tool_response(
                filtered_interactions,
                browser_stage["browser_summary"],
            )

            critique_response = await run_with_transient_retry(
                "critique",
                lambda: run_critique(
                    current_step=state.current_step,
                    orignal_plan=state.plan,
                    tool_response=critique_tool_response,
                    ss_analysis=browser_stage["ss_analysis"],
                    message_history=state.message_histories["critique"],
                ),
            )
            state.conversation_handler.add_critique_message(critique_response)

            critique_new_messages = critique_response.new_messages()
            state.message_histories["critique"].extend(critique_new_messages)

            openai_messages = state.conversation_handler.get_conversation_history()
            saved_path = state.conversation_storage.save_conversation(
                openai_messages,
                prefix="task",
            )
            logger.info("Conversation appended to: %s", saved_path)
            logger.info("Critique feedback:\n%s", critique_response.output.feedback)
            logger.info("Critique terminate=%s", critique_response.output.terminate)
            sanitized_feedback = strip_snapshot_refs(critique_response.output.feedback)

            orchestrator.emit_event(
                "critique_completed",
                message="Critique evaluated the current step",
                iteration=state.iteration_counter,
                current_step=state.current_step,
                data={
                    "terminate": critique_response.output.terminate,
                    "feedback": sanitized_feedback,
                },
            )

            if critique_response.output.terminate:
                state.terminate = True
                result = OrchestratorRunResult(
                    user_query=user_query,
                    plan=state.plan,
                    next_step=state.current_step,
                    final_response=critique_response.output.final_response,
                )
                orchestrator.emit_event(
                    "run_finished",
                    message="Run completed",
                    iteration=state.iteration_counter,
                    current_step=state.current_step,
                    plan=state.plan,
                    final_response=result.final_response,
                )
                return result

            state.planner_prompt = (
                f"User Query: {user_query}\n"
                f"Previous Plan:\n{state.plan}\n\n"
                f"Feedback:\n{sanitized_feedback}"
            )
            state.consecutive_step_errors = 0
        except Exception as step_error:
            state.consecutive_step_errors += 1
            error_message = f"Error in execution step {state.iteration_counter}: {step_error}"
            logger.exception(error_message)

            if state.consecutive_step_errors >= state.max_step_errors:
                state.terminate = True
                final_response = (
                    "Execution stopped due to repeated internal errors. "
                    f"Last error: {step_error}"
                )
                orchestrator.emit_event(
                    "run_failed",
                    message=error_message,
                    iteration=state.iteration_counter,
                    current_step=state.current_step,
                )
                result = OrchestratorRunResult(
                    user_query=user_query,
                    plan=state.plan,
                    next_step=state.current_step,
                    final_response=final_response,
                )
                orchestrator.emit_event(
                    "run_finished",
                    message="Run stopped after repeated internal errors",
                    iteration=state.iteration_counter,
                    current_step=state.current_step,
                    plan=state.plan,
                    final_response=result.final_response,
                )
                return result

    raise RuntimeError("Orchestrator loop ended without result")
