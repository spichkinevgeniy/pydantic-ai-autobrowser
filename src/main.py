import asyncio
import logging
import sys

from src.logging_setup import configure_logging
from src.orchestrator import OrchestratorEvent, run_orchestration


def configure_stdio() -> None:
    """Force UTF-8 for console IO so prompts and logs stay readable on Windows."""
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


def print_event(event: OrchestratorEvent) -> None:
    if event.event_type == "run_started":
        print("Запуск orchestration...")
    elif event.event_type == "iteration_started" and event.iteration is not None:
        print(f"\n[Итерация {event.iteration}]")
    elif event.event_type == "planner_completed" and event.current_step:
        print(f"Planner: {event.current_step}")
    elif event.event_type == "browser_completed":
        print("Browser: шаг выполнен")
    elif event.event_type == "critique_completed":
        print("Critique: шаг оценен")
    elif event.event_type == "run_failed" and event.message:
        print(f"Ошибка выполнения: {event.message}")


async def async_main() -> int:
    configure_stdio()
    log_file = configure_logging()
    logger = logging.getLogger("main")
    logger.info("Приложение запущено")
    logger.info("Логи записываются в %s", log_file)

    user_query = input("Введите запрос на автоматизацию браузера: ").strip()
    if not user_query:
        logger.warning("Из консоли получен пустой запрос")
        print("Пустой запрос. Выполнять нечего.")
        return 1

    try:
        result = await run_orchestration(user_query, on_event=print_event)
    except Exception:
        logger.exception("Во время обработки запроса произошла ошибка")
        print("Не удалось обработать запрос. Подробности смотри в logs/orchestrator.log.")
        return 1

    print("\nПлан:")
    print(result.plan)
    if result.final_response:
        print("\nФинальный ответ:")
        print(result.final_response)
    else:
        print("\nСледующий шаг:")
        print(result.next_step)

    logger.info("Приложение завершилось успешно")
    return 0


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
