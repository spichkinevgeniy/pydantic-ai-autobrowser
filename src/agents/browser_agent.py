import logging
from collections.abc import Sequence
from functools import lru_cache

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.run import AgentRunResult
from pydantic_ai.settings import ModelSettings
from src.config import settings

BROWSER_SYS_PROMPT = """
<agent_role>
    You are a browser execution agent in a loop: Planner -> Browser Agent[You] -> Critique.
    The planner gives you one current step. Execute that step using the available Playwright MCP tools.
</agent_role>

<core_rules>
    <rule>Execute exactly one meaningful browser action or tool sequence needed for the current step.</rule>
    <rule>Prefer Playwright MCP tools over free-text reasoning.</rule>
    <rule>Use page snapshots / DOM inspection tools before clicking or typing when element identity is unclear.</rule>
    <rule>Do not invent selectors, element ids, or page state. Use only information returned by tools.</rule>
    <rule>If the current step is ambiguous or blocked, explain the blocker briefly.</rule>
    <rule>Do not repeat the same failing action in a loop.</rule>
</core_rules>

<navigation_rules>
    <rule>If the step contains a direct URL, navigate to it directly.</rule>
    <rule>If a search is needed, use browser tools to search and then continue from the result.</rule>
    <rule>Preserve browser state between steps when possible.</rule>
</navigation_rules>

<output_rules>
    <rule>Return a short summary of the action taken.</rule>
    <rule>If the step was completed, state the result briefly.</rule>
    <rule>If the step could not be completed, state why.</rule>
    <rule>Do not include internal selectors or low-level ids in the final response unless explicitly asked.</rule>
</output_rules>
"""


class BrowserStepDeps(BaseModel):
    current_step: str


provider = GoogleProvider(api_key=settings.GOOGLE_API_KEY)
model = GoogleModel(settings.MODEL_NAME, provider=provider)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_playwright_mcp_server() -> MCPServerStdio:
    return MCPServerStdio(
        "npx",
        args=["-y", "@playwright/mcp@latest"],
        tool_prefix="playwright",
        timeout=30,
        read_timeout=300,
    )


@lru_cache(maxsize=1)
def get_browser_agent() -> Agent:
    return Agent(
        model=model,
        system_prompt=BROWSER_SYS_PROMPT,
        deps_type=BrowserStepDeps,
        name="Browser Agent",
        retries=3,
        model_settings=ModelSettings(
            temperature=0.2,
            timeout=settings.MODEL_TIMEOUT_SECONDS,
        ),
        toolsets=[get_playwright_mcp_server()],
    )


async def run_browser_step(
    current_step: str,
    message_history: Sequence[ModelMessage] | None = None,
) -> AgentRunResult[str]:
    logger.info("Browser agent started step execution")
    logger.info("Browser current step: %s", current_step)

    result = await get_browser_agent().run(
        user_prompt=current_step,
        deps=BrowserStepDeps(current_step=current_step),
        message_history=message_history,
    )

    logger.info("Browser agent completed step execution")
    logger.info("Browser agent output: %s", result.output)
    return result


browser_agent = get_browser_agent()
