import logging
from pathlib import Path

import logfire
from logfire.integrations.logging import LogfireLoggingHandler
from src.config import settings

ROOT_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT_DIR / "logs"
LOG_FILE = LOG_DIR / "orchestrator.log"


def configure_logging(level: int = logging.INFO, *, console: bool = True) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    if any(
        getattr(handler, "_auto_browser_demo_handler", False) for handler in root_logger.handlers
    ):
        root_logger.setLevel(level)
        return LOG_FILE

    logfire.configure(
        send_to_logfire="if-token-present",
        token=settings.LOGFIRE_TOKEN,
        service_name=settings.LOGFIRE_SERVICE_NAME,
        console=False,
    )
    logfire.instrument_pydantic_ai()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler._auto_browser_demo_handler = True  # type: ignore[attr-defined]

    logfire_handler = LogfireLoggingHandler(level=level)
    logfire_handler._auto_browser_demo_handler = True  # type: ignore[attr-defined]

    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)
    if console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler._auto_browser_demo_handler = True  # type: ignore[attr-defined]
        root_logger.addHandler(stream_handler)
    root_logger.addHandler(logfire_handler)

    return LOG_FILE
