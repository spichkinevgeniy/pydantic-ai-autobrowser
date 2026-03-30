import logging
import json
from pprint import pformat
from collections.abc import Sequence
from typing import Any

from pydantic_ai.messages import ModelRequest, ToolReturnPart

from src.agents.browser_agent import run_browser_step
from src.agents.planner_agent import create_plan
from src.types import OrchestratorRunResult
from src.utils.msg_parser import AgentConversationHandler


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

    missing_responses = [tool_call_id for tool_call_id, has_response in tool_calls.items() if not has_response]
    if missing_responses:
        raise ValueError(f"Missing tool responses for: {missing_responses}")

    return result


def extract_tool_interactions(messages: Sequence[Any]) -> str:
    """Extract tool calls and matching responses from browser agent messages."""
    tool_interactions: dict[str, dict[str, Any]] = {}

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
        elif getattr(msg, "kind", None) == "request":
            for part in msg.parts:
                if getattr(part, "part_kind", "") == "tool-return":
                    if part.tool_call_id in tool_interactions:
                        tool_interactions[part.tool_call_id]["response"] = {
                            "content": part.content,
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


def format_payload(payload: Any) -> str:
    """Formats payloads for detailed debug logging."""
    return pformat(payload, width=120, compact=False, sort_dicts=False)


def filter_dom_messages(
    messages: Sequence[Any],
    dom_tool_names: set[str] | None = None,
) -> list[Any]:
    """Replace large DOM tool responses in message history with a placeholder."""
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
            filtered_messages.append(msg.model_copy(update={"parts": new_parts}))
        else:
            filtered_messages.append(msg)

    return filtered_messages


class Orchestrator:
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self._started = False
        self.terminate = False
        self.iteration_counter = 0
        self.message_histories = {
            "planner": [],
            "browser": [],
        }
        self.conversation_handler = AgentConversationHandler()

    async def run(self, user_query: str) -> OrchestratorRunResult:
        await self.start()

        try:
            self.logger.info("Получен пользовательский запрос")
            self.terminate = False
            self.iteration_counter = 0
            planner_prompt = f"User Query: {user_query}"

            while not self.terminate:
                self.iteration_counter += 1
                self.logger.info("Запущена итерация оркестратора: %s", self.iteration_counter)

                # Planner execution
                try:
                    validated_history = ensure_tool_response_sequence(self.message_histories["planner"])
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
                    self.logger.info(
                        "Conversation handler history before append (%s messages):\n%s",
                        len(self.conversation_handler.conversation_history),
                        format_payload(self.conversation_handler.conversation_history),
                    )
                    self.conversation_handler.add_planner_message(planner_output)
                    self.logger.info(
                        "Conversation handler history after append (%s messages):\n%s",
                        len(self.conversation_handler.conversation_history),
                        format_payload(self.conversation_handler.conversation_history),
                    )

                    self.message_histories["planner"].extend(planner_new_messages)

                    plan = planner_output.output.plan
                    current_step = planner_output.output.next_step

                    self.logger.info("План от планировщика получен")
                    self.logger.info("Текущий шаг: %s", current_step)

                    if self.iteration_counter == 1:
                        self.logger.info("Полный план на первой итерации:\n%s", plan)
                except Exception:
                    self.logger.exception(
                        "Произошла ошибка планировщика на итерации %s",
                        self.iteration_counter,
                    )
                    raise
                # Browser Execution
                try:
                    browser_history = filter_dom_messages(self.message_histories["browser"])
                    browser_output = await run_browser_step(
                        current_step=current_step,
                        message_history=browser_history,
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
                        "Browser agent response:\n%s",
                        browser_output.output,
                    )
                except Exception:
                    self.logger.exception(
                        "Произошла ошибка браузерного агента на итерации %s",
                        self.iteration_counter,
                    )
                    raise

            raise RuntimeError("Оркестратор завершил цикл без результата")
        except Exception:
            self.logger.exception("Во время работы оркестратора произошла ошибка")
            raise
        finally:
            await self.shutdown()

    async def start(self) -> None:
        if self._started:
            self.logger.debug("Запуск пропущен: оркестратор уже работает")
            return

        self._started = True
        self.logger.info("Оркестратор запущен")

    async def shutdown(self) -> None:
        if not self._started:
            return

        self.logger.info("Начато завершение работы оркестратора")
        await self.cleanup()
        self._started = False
        self.logger.info("Завершение работы оркестратора выполнено")

    async def cleanup(self) -> None:
        self.logger.info("Очистка завершена")

    async def wait_for_exit(self) -> None:
        self.logger.debug("Ожидание завершения вызвано, фоновые воркеры не зарегистрированы")
