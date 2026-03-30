import asyncio
import logging
import sys
import warnings

from src.logging_setup import configure_logging


def configure_stdio() -> None:
    """Force UTF-8 for console IO so prompts and logs stay readable on Windows."""
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


def configure_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r".*_UnionGenericAlias.*deprecated.*",
        category=DeprecationWarning,
        module=r"google\.genai\.types",
    )


def parse_initial_query(argv: list[str]) -> str:
    if len(argv) <= 1:
        return ""
    return " ".join(argv[1:]).strip()


async def async_main() -> int:
    configure_stdio()
    configure_warnings()
    from src.orchestrator import run_orchestration
    from src.ui import ConsoleProgressUI

    log_file = configure_logging(console=False)
    ui = ConsoleProgressUI(log_file)
    initial_query = parse_initial_query(sys.argv)
    logger = logging.getLogger("main")
    user_query = initial_query or input("Введите запрос на автоматизацию браузера: ").strip()
    if not user_query:
        logger.warning("Из консоли получен пустой запрос")
        print("Пустой запрос. Выполнять нечего.")
        return 1

    try:
        result = await run_orchestration(user_query, on_event=ui.handle_event)
    except Exception:
        logger.exception("Во время обработки запроса произошла ошибка")
        print("\nНе удалось обработать запрос.")
        print(f"Трассировка: {log_file}")
        return 1

    ui.render_result(result)
    return 0


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
