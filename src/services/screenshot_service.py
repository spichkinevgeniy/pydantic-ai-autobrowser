import logging
from pathlib import Path

from src.agents.browser_agent import get_playwright_mcp_server
from src.config import ROOT_DIR, settings

logger = logging.getLogger(__name__)


class ScreenshotService:
    def __init__(self, screenshot_dir: Path | None = None) -> None:
        self.screenshot_dir = (screenshot_dir or settings.SCREENSHOT_DIR).resolve()
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

    async def capture(
        self,
        label: str,
        iteration: int,
        *,
        full_page: bool = False,
    ) -> Path | None:
        """Capture a screenshot from the current Playwright MCP session."""
        if not settings.ENABLE_SCREENSHOTS:
            logger.info("Screenshot capture disabled in settings")
            return None

        server = get_playwright_mcp_server()
        if not server.is_running:
            logger.warning("Skipping screenshot capture because Playwright MCP server is not running")
            return None

        safe_label = label.lower().replace(" ", "_")
        output_path = self.screenshot_dir / f"iter_{iteration:03d}_{safe_label}.png"
        filename = self._to_server_relative_path(output_path)

        try:
            result = await server.direct_call_tool(
                "browser_take_screenshot",
                {
                    "type": "png",
                    "filename": filename,
                    "fullPage": full_page,
                },
            )
        except Exception:
            logger.exception("Failed to capture %s screenshot for iteration %s", label, iteration)
            return None

        if output_path.exists():
            logger.info(
                "Saved %s screenshot for iteration %s to %s",
                label,
                iteration,
                output_path,
            )
            logger.debug("Screenshot tool result: %s", result)
            return output_path

        logger.warning(
            "Screenshot tool completed but file was not found for %s iteration %s: %s",
            label,
            iteration,
            output_path,
        )
        logger.debug("Screenshot tool result: %s", result)
        return None

    def _to_server_relative_path(self, output_path: Path) -> str:
        try:
            return output_path.relative_to(ROOT_DIR).as_posix()
        except ValueError:
            return output_path.as_posix()
