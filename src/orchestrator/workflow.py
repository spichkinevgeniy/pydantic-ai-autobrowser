import logging
from pathlib import Path

from src.agents.browser_agent import run_browser_step
from src.agents.critique_agent import run_critique
from src.agents.planner_agent import create_plan
from src.config import settings
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
from src.types import OrchestratorRunResult
from src.utils.image_analysis import ImageAnalyzer

logger = logging.getLogger(__name__)


async def run_workflow(orchestrator, user_query: str) -> OrchestratorRunResult:
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

            orchestrator.emit_event(
                "planner_completed",
                message="Planner produced the next step",
                iteration=state.iteration_counter,
                current_step=state.current_step,
                plan=state.plan,
            )

            tool_interactions_str: str | None = None
            ss_analysis = ""
            pre_action_ss: Path | None = None
            post_action_ss: Path | None = None

            browser_history = filter_dom_messages(state.message_histories["browser"])
            pre_action_ss = await state.screenshot_helper.capture(
                "pre",
                state.iteration_counter,
                full_page=settings.SCREENSHOT_FULL_PAGE,
            )
            browser_output = await run_with_transient_retry(
                "browser",
                lambda: run_browser_step(
                    current_step=state.current_step,
                    message_history=browser_history,
                ),
            )
            post_action_ss = await state.screenshot_helper.capture(
                "post",
                state.iteration_counter,
                full_page=settings.SCREENSHOT_FULL_PAGE,
            )

            state.conversation_handler.add_browser_nav_message(browser_output)

            browser_new_messages = browser_output.new_messages()
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
                logger.info(
                    "Screenshot analysis skipped because one or both screenshots are missing"
                )

            orchestrator.emit_event(
                "browser_completed",
                message="Browser step executed",
                iteration=state.iteration_counter,
                current_step=state.current_step,
                data={
                    "pre_screenshot": str(pre_action_ss) if pre_action_ss else "",
                    "post_screenshot": str(post_action_ss) if post_action_ss else "",
                },
            )

            filtered_interactions = filter_tool_interactions_for_critique(tool_interactions_str)
            critique_tool_response = build_critique_tool_response(
                filtered_interactions,
                str(browser_output.output),
            )

            critique_response = await run_with_transient_retry(
                "critique",
                lambda: run_critique(
                    current_step=state.current_step,
                    orignal_plan=state.plan,
                    tool_response=critique_tool_response,
                    ss_analysis=ss_analysis,
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

            orchestrator.emit_event(
                "critique_completed",
                message="Critique evaluated the current step",
                iteration=state.iteration_counter,
                current_step=state.current_step,
                data={"terminate": critique_response.output.terminate},
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

            sanitized_feedback = strip_snapshot_refs(critique_response.output.feedback)
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
