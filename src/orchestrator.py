import logging
from pprint import pformat
from collections.abc import Sequence
from typing import Any

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


def format_payload(payload: Any) -> str:
    """Formats payloads for detailed debug logging."""
    return pformat(payload, width=120, compact=False, sort_dicts=False)


class Orchestrator:
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self._started = False
        self.terminate = False
        self.iteration_counter = 0
        self.message_histories = {
            "planner": [],
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
