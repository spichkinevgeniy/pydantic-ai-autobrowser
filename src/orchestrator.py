import json
import logging
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from pprint import pformat
from typing import Any

from pydantic_ai.messages import ModelRequest, ToolReturnPart
from src.agents.browser_agent import get_playwright_mcp_server, run_browser_step
from src.agents.critique_agent import run_critique
from src.agents.planner_agent import create_plan
from src.config import settings
from src.services.image_analyzer import ImageAnalyzer
from src.services.screenshot_service import ScreenshotService
from src.types import OrchestratorRunResult
from src.utils.msg_parser import AgentConversationHandler, ConversationStorage


def ensure_tool_response_sequence(messages: Sequence[Any]) -> list[Any]:
    """Ensures that every tool call has a corresponding tool response."""
    tool_calls: dict[str, bool] = {}
    result: list[Any] = []

    for msg in messages:
        if isinstance(msg, dict) and "tool_calls" in msg.get("parts", [{}])[0]:
            for tool_call in msg["parts"][0]["tool_calls"]:
                tool_calls[tool_call["tool_call_id"]] = False
            result.append(msg)
        elif isinstance(msg, dict) and "tool_return" in msg.get("parts", [{}])[0]:
            tool_call_id = msg["parts"][0].get("tool_call_id")
            if tool_call_id in tool_calls:
                tool_calls[tool_call_id] = True
            result.append(msg)
        else:
            result.append(msg)

    missing_responses = [
        tool_call_id for tool_call_id, has_response in tool_calls.items() if not has_response
    ]
    if missing_responses:
        raise ValueError(f"Missing tool responses for: {missing_responses}")

    return result


def extract_tool_interactions(messages: Sequence[Any]) -> str:
    """Extract tool calls and matching responses from browser agent messages."""
    tool_interactions: dict[str, dict[str, Any]] = {}
    pending_call_ids: list[str] = []

    for msg in messages:
        if getattr(msg, "kind", None) == "response":
            for part in msg.parts:
                if getattr(part, "part_kind", "") == "tool-call":
                    raw_args = getattr(part, "args", {})
                    if hasattr(raw_args, "args_json"):
                        args_value = raw_args.args_json
                    elif isinstance(raw_args, dict):
                        args_value = json.dumps(raw_args, ensure_ascii=False)
                    else:
                        args_value = str(raw_args)

                    tool_interactions[part.tool_call_id] = {
                        "call": {
                            "tool_name": part.tool_name,
                            "args": args_value,
                        },
                        "response": None,
                    }
                    pending_call_ids.append(part.tool_call_id)
        elif getattr(msg, "kind", None) == "request":
            for part in msg.parts:
                if getattr(part, "part_kind", "") == "tool-return":
                    content = serialize_content(part.content)
                    if part.tool_call_id in tool_interactions:
                        tool_interactions[part.tool_call_id]["response"] = {
                            "content": content,
                        }
                        continue

                    # Some MCP tool responses may arrive without a matching id in the
                    # incremental message slice. Fall back to the latest unresolved call.
                    unresolved_call_id = next(
                        (
                            tool_call_id
                            for tool_call_id in reversed(pending_call_ids)
                            if tool_interactions[tool_call_id]["response"] is None
                        ),
                        None,
                    )
                    if unresolved_call_id is not None:
                        tool_interactions[unresolved_call_id]["response"] = {
                            "content": content,
                        }

    interactions_str = ""
    for interaction in tool_interactions.values():
        call = interaction["call"]
        response = interaction["response"]
        interactions_str += f"Tool Call: {call['tool_name']}\n"
        interactions_str += f"Arguments: {call['args']}\n"
        if response:
            interactions_str += f"Response: {response['content']}\n"
        interactions_str += "---\n"

    return interactions_str


def serialize_content(content: Any) -> str:
    """Serialize tool content for logs and critique without losing non-string payloads."""
    if isinstance(content, str):
        return content
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False, indent=2)
    return str(content)


def build_critique_tool_response(
    tool_interactions_str: str | None,
    browser_summary: str,
) -> str:
    """Build critique input from tool interactions and the browser agent summary."""
    sections: list[str] = []
    if tool_interactions_str:
        sections.append("Tool interactions:\n" + tool_interactions_str.rstrip())
    if browser_summary:
        sections.append("Browser summary:\n" + browser_summary.strip())
    return "\n\n".join(sections)


def filter_tool_interactions_for_critique(tool_interactions_str: str | None) -> str:
    """Compress large Playwright MCP responses before sending them to critique."""
    if not tool_interactions_str:
        return ""

    dom_like_tools = {
        "playwright_browser_navigate",
        "playwright_browser_snapshot",
    }
    interactions = tool_interactions_str.split("---\n")
    filtered_interactions: list[str] = []

    for interaction in interactions:
        if not interaction.strip():
            continue

        lines = interaction.splitlines()
        tool_call_line = next(
            (line for line in lines if line.startswith("Tool Call: ")),
            "",
        )
        tool_name = tool_call_line.removeprefix("Tool Call: ").strip()
        has_snapshot = "### Snapshot" in interaction

        if tool_name in dom_like_tools or has_snapshot:
            filtered_lines: list[str] = []
            for line in lines:
                if line.startswith("Tool Call:") or line.startswith("Arguments:"):
                    filtered_lines.append(line)
                elif line.startswith("Response:"):
                    filtered_lines.append(f"Response: {tool_name or 'tool'} completed successfully")
                    break

            filtered_interactions.append("\n".join(filtered_lines))
        else:
            filtered_interactions.append(interaction.rstrip())

    return "---\n".join(filtered_interactions) + ("---\n" if filtered_interactions else "")


def format_payload(payload: Any) -> str:
    """Formats payloads for detailed debug logging."""
    return pformat(payload, width=120, compact=False, sort_dicts=False)


def filter_dom_messages(
    messages: Sequence[Any],
    dom_tool_names: set[str] | None = None,
) -> list[Any]:
    """Replace large DOM-like tool responses in message history with a placeholder."""
    dom_tool_names = dom_tool_names or {
        "playwright_browser_navigate",
        "playwright_browser_snapshot",
    }
    filtered_messages: list[Any] = []

    for msg in messages:
        if not isinstance(msg, ModelRequest):
            filtered_messages.append(msg)
            continue

        new_parts = []
        changed = False

        for part in msg.parts:
            should_compress = False
            if isinstance(part, ToolReturnPart):
                if part.tool_name in dom_tool_names:
                    should_compress = True
                elif isinstance(part.content, str) and "### Snapshot" in part.content:
                    should_compress = True

            if should_compress:
                new_parts.append(
                    ToolReturnPart(
                        tool_name=part.tool_name,
                        content=f"{part.tool_name} completed successfully",
                        tool_call_id=part.tool_call_id,
                        timestamp=part.timestamp,
                        metadata=part.metadata,
                        outcome=part.outcome,
                    )
                )
                changed = True
            else:
                new_parts.append(part)

        if changed:
            filtered_messages.append(replace(msg, parts=new_parts))
        else:
            filtered_messages.append(msg)

    return filtered_messages


class Orchestrator:
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self._started = False
        self._browser_server_entered = False
        self.terminate = False
        self.iteration_counter = 0
        self.max_step_errors = 3
        self.message_histories = {
            "planner": [],
            "browser": [],
            "critique": [],
        }
        self.conversation_handler = AgentConversationHandler()
        self.conversation_storage = ConversationStorage()
        self.screenshot_service = ScreenshotService()
        ImageAnalyzer.clear_history()

    async def run(self, user_query: str) -> OrchestratorRunResult:
        await self.start()

        try:
            self.logger.info("User query received")
            self.terminate = False
            self.iteration_counter = 0
            consecutive_step_errors = 0
            planner_prompt = f"User Query: {user_query}\nFeedback: None"

            while not self.terminate:
                self.iteration_counter += 1
                self.logger.info(
                    "Orchestrator iteration started: %s",
                    self.iteration_counter,
                )
                plan = ""
                current_step = ""

                try:
                    try:
                        validated_history = ensure_tool_response_sequence(
                            self.message_histories["planner"]
                        )
                        self.logger.info(
                            "Validated planner history (%s messages):\n%s",
                            len(validated_history),
                            format_payload(validated_history),
                        )

                        planner_output = await create_plan(
                            planner_prompt,
                            message_history=validated_history,
                        )

                        planner_new_messages = planner_output.new_messages()
                        self.message_histories["planner"].extend(planner_new_messages)

                        plan = planner_output.output.plan
                        current_step = planner_output.output.next_step

                        self.logger.info("Planner returned plan")
                        self.logger.info("Current step: %s", current_step)

                        if self.iteration_counter == 1:
                            self.logger.info("Initial full plan:\n%s", plan)
                    except Exception:
                        self.logger.exception(
                            "Planner error on iteration %s",
                            self.iteration_counter,
                        )
                        raise

                    tool_interactions_str: str | None = None
                    ss_analysis = ""
                    pre_action_ss: Path | None = None
                    post_action_ss: Path | None = None
                    try:
                        browser_history = filter_dom_messages(self.message_histories["browser"])
                        pre_action_ss = await self.screenshot_service.capture(
                            "pre",
                            self.iteration_counter,
                            full_page=settings.SCREENSHOT_FULL_PAGE,
                        )
                        browser_output = await run_browser_step(
                            current_step=current_step,
                            message_history=browser_history,
                        )
                        post_action_ss = await self.screenshot_service.capture(
                            "post",
                            self.iteration_counter,
                            full_page=settings.SCREENSHOT_FULL_PAGE,
                        )

                        self.conversation_handler.add_browser_nav_message(browser_output)

                        new_messages = browser_output.new_messages()
                        self.message_histories["browser"].extend(new_messages)
                        tool_interactions_str = extract_tool_interactions(new_messages)
                        all_messages = browser_output.all_messages()
                        self.logger.info(
                            "All messages from browser agent (%s messages):\n%s",
                            len(all_messages),
                            format_payload(all_messages),
                        )
                        self.logger.info(
                            "Tool interactions from browser agent:\n%s",
                            tool_interactions_str or "No tool interactions recorded.",
                        )
                        self.logger.info(
                            "Screenshot paths for iteration %s: pre=%s post=%s",
                            self.iteration_counter,
                            pre_action_ss,
                            post_action_ss,
                        )

                        if pre_action_ss and post_action_ss:
                            analyzer = ImageAnalyzer(
                                image1_path=pre_action_ss,
                                image2_path=post_action_ss,
                                next_step=current_step,
                            )
                            ss_analysis = await analyzer.analyze_images()
                            if ss_analysis:
                                self.logger.info("Screenshot analysis:\n%s", ss_analysis)
                            else:
                                self.logger.info(
                                    "Screenshot analysis skipped or returned empty output"
                                )
                        else:
                            self.logger.info(
                                "Screenshot analysis skipped because one or both screenshots are missing"
                            )
                    except Exception:
                        self.logger.exception(
                            "Browser agent error on iteration %s",
                            self.iteration_counter,
                        )
                        raise

                    try:
                        filtered_interactions = filter_tool_interactions_for_critique(
                            tool_interactions_str
                        )
                        critique_history = self.message_histories["critique"]
                        critique_tool_response = build_critique_tool_response(
                            filtered_interactions,
                            str(browser_output.output),
                        )

                        critique_response = await run_critique(
                            current_step=current_step,
                            orignal_plan=plan,
                            tool_response=critique_tool_response,
                            ss_analysis=ss_analysis,
                            message_history=critique_history,
                        )
                        self.conversation_handler.add_critique_message(critique_response)

                        critique_new_messages = critique_response.new_messages()
                        self.message_histories["critique"].extend(critique_new_messages)

                        openai_messages = self.conversation_handler.get_conversation_history()
                        saved_path = self.conversation_storage.save_conversation(
                            openai_messages,
                            prefix="task",
                        )
                        self.logger.info("Conversation appended to: %s", saved_path)

                        self.logger.info(
                            "Critique feedback:\n%s",
                            critique_response.output.feedback,
                        )
                        self.logger.info(
                            "Critique terminate=%s",
                            critique_response.output.terminate,
                        )

                        if critique_response.output.terminate:
                            self.terminate = True
                            return OrchestratorRunResult(
                                user_query=user_query,
                                plan=plan,
                                next_step=current_step,
                                final_response=critique_response.output.final_response,
                            )

                        planner_prompt = (
                            f"User Query: {user_query}\n"
                            f"Previous Plan:\n{plan}\n\n"
                            f"Feedback:\n{critique_response.output.feedback}"
                        )
                        consecutive_step_errors = 0
                    except Exception:
                        self.logger.exception(
                            "Critique agent error on iteration %s",
                            self.iteration_counter,
                        )
                        raise
                except Exception as step_error:
                    consecutive_step_errors += 1
                    error_msg = f"Error in execution step {self.iteration_counter}: {step_error}"
                    self.logger.exception(error_msg)
                    if consecutive_step_errors >= self.max_step_errors:
                        self.terminate = True
                        final_response = (
                            "Execution stopped due to repeated internal errors. "
                            f"Last error: {step_error}"
                        )
                        return OrchestratorRunResult(
                            user_query=user_query,
                            plan=plan,
                            next_step=current_step,
                            final_response=final_response,
                        )
                    continue

            raise RuntimeError("Orchestrator loop ended without result")
        except Exception:
            self.logger.exception("Orchestrator failed during execution")
            raise
        finally:
            await self.shutdown()

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
