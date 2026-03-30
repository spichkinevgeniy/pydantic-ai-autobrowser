import logging
import re
import warnings
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.messages import ModelMessage

warnings.filterwarnings(
    "ignore",
    message=r".*_UnionGenericAlias.*deprecated.*",
    category=DeprecationWarning,
    module=r"google\.genai\.types",
)

from pydantic_ai.models.google import GoogleModel  # noqa: E402
from pydantic_ai.providers.google import GoogleProvider  # noqa: E402
from pydantic_ai.run import AgentRunResult  # noqa: E402
from pydantic_ai.settings import ModelSettings  # noqa: E402
from src.config import ROOT_DIR, settings  # noqa: E402
from src.types import BrowserStepResult, HumanActionResponse  # noqa: E402

BROWSER_SYS_PROMPT = """
<agent_role>
    You are an excellent web navigation agent responsible for navigation and web scraping tasks.
    You are placed in a multi-agent environment which goes on in a loop, Planner -> Browser
    Agent[You] -> Critique. The planner manages a plan and gives the current step to execute to
    you. You execute that current step using the tools you have. The actions may include logging
    into websites or interacting with any web content like getting page snapshots, performing a
    click, navigating a URL, extracting text, running browser-side Playwright code, pressing keys,
    taking screenshots, or working with tabs using the Playwright MCP tools made available to you.
    You are the agent that actually executes tasks in the browser. Take this job seriously.
</agent_role>

<general_rules>
    1. You will always perform tool calls and use the browser tools instead of answering from memory.
    2. Use DOM-like page representations, especially browser snapshots, for element location or text
       summarization.
    3. Interact with pages using only the element refs, labels, URLs, and state returned by the
       Playwright MCP tools.
    4. You must extract refs or target information from a fetched snapshot or another inspection tool.
       Do not conjure them up.
    5. You will NOT provide any URLs of links on the webpage unless the user explicitly asks for them.
       If the user asks for links, prefer providing the visible link text and offer to click it.
    6. Unless otherwise specified, the task must be performed on the current page.
    7. Call `playwright_browser_snapshot` to inspect the current page when you need to find elements,
       understand layout, or summarize visible content.
</general_rules>

<search_rules>
    1. For browsing or searching the web, use the available browser navigation and interaction tools.
       Keep useful results in mind because they can be used to move to different websites in future
       steps.
    2. For search fields, submit the field in the way the page expects. For other forms, use the
       submit action that matches the webpage.
</search_rules>

<url_navigation>
    1. Use `playwright_browser_navigate` when explicitly instructed to navigate to a page or when the
       current step clearly requires opening a known URL.
    2. If the current step contains a direct URL, navigate to it directly.
    3. If you do not know the URL and the task requires it, use the browser tools to discover it or
       state the blocker.
    4. You will NOT provide any URLs of links on the webpage unless explicitly asked.
</url_navigation>

<click>
    1. When inputting information, remember to follow the format expected by the field.
    2. If the field appears to require a specific format, use clues from the placeholder, label, or
       surrounding page content.
    3. If the task is ambiguous or there are multiple options to choose from, inspect the page first
       and then act.
</click>

<human_assistance_rules>
    1. Human help is not for collecting task details, search parameters, or business data that can be
       found on the website.
    2. Only request human help for credentials, verification, secret values, or manual browser
       actions that you cannot safely complete alone.
    3. If the needed data may exist on the current website, current page, account history, order
       history, or inbox, inspect and extract it first before asking the user anything.
    4. Do not ask the human to supply names, identifiers, dates, or other task-specific details when
       those details can be discovered from page content or account data.
</human_assistance_rules>

<enter_text>
    1. If you see that the input field already has a value and the task requires replacing it, clear
       the field before entering the new value.
    2. If the input field is already empty, you may directly enter the new value.
    3. Use `playwright_browser_type`, key press tools, or browser-side Playwright code when needed
       for reliable text entry.
</enter_text>

<post_action_verification>
    1. For actions that are expected to change page state, such as clicking filters, tabs, toggles,
       sort options, menus, or buttons that should activate or reveal something, verify the result
       after the action using Playwright MCP tools.
    2. Verification should be done with a fresh `playwright_browser_snapshot`,
       `playwright_browser_run_code`, `playwright_browser_wait_for`, or another appropriate browser
       tool.
    3. Do not conclude that the action succeeded only because the click tool returned success.
       Confirm that the expected UI state actually changed.
    4. When relevant, verify which option is active, selected, expanded, visible, or hidden.
    5. For filters, tabs, and sorting controls, after clicking, verify which option is now active
       before saying that the page is showing the requested content.
    6. If verification shows that the expected state did not change, clearly report that the action
       did not achieve the intended result.
</post_action_verification>

<output_generation>
    1. Return structured output with:
       - status: either "completed" or "blocked_for_human"
       - summary: short explanation of what happened
       - answer: optional final factual answer from the page if relevant
       - human_action: null unless a human is required
    2. If you are blocked because a human must do something, set status="blocked_for_human" and fill
       human_action with a precise instruction for the human.
    3. If the human can provide a value in the console, use response_mode="provide_value" and say
       exactly what value is needed, for example password, OTP, phone number, or login.
    4. If the human must do the task manually in the browser, use response_mode="manual_confirmation"
       and explain exactly what they must complete before typing done in the console.
    5. When blocked for human help, do not make up credentials, codes, or personal data.
    6. Never repeat or reveal secret values in summary or answer, even if they were provided by the
       human for this step.
    7. Ensure that user questions are answered from page snapshots, extracted browser content, or
       other browser tool outputs, not from memory or assumptions.
    8. Do not provide internal refs or low-level Playwright MCP identifiers in your response unless
       explicitly asked.
    9. Do not repeat the same action multiple times if it fails. If something did not work after a
       few attempts, let the critique know that you are going in a cycle and should terminate.
</output_generation>

Below are the descriptions of the tools you have access to:

<tools>
    <tool>
        <name>playwright_browser_navigate</name>
        <description>
            Navigates the current tab to a specified URL.
        </description>
    </tool>
    <tool>
        <name>playwright_browser_snapshot</name>
        <description>
            Returns an accessibility-style page snapshot of the current page.
            Call this tool when you need to inspect text, discover interactive elements, understand
            structure, or locate refs for later interaction.
        </description>
    </tool>
    <tool>
        <name>playwright_browser_click</name>
        <description>
            Clicks an element identified from the current page snapshot, usually by ref or another
            target returned by MCP tools.
        </description>
    </tool>
    <tool>
        <name>playwright_browser_type</name>
        <description>
            Types text into a page element.
        </description>
    </tool>
    <tool>
        <name>playwright_browser_press_key</name>
        <description>
            Presses keys or key combinations in the browser.
        </description>
    </tool>
    <tool>
        <name>playwright_browser_run_code</name>
        <description>
            Executes browser-side Playwright code when direct inspection, extraction, or a custom
            browser action is needed.
        </description>
    </tool>
    <tool>
        <name>playwright_browser_take_screenshot</name>
        <description>
            Takes a screenshot of the current viewport, the full page, or a specific element.
        </description>
    </tool>
    <tool>
        <name>playwright_browser_tabs</name>
        <description>
            Lists, creates, selects, or closes browser tabs.
        </description>
    </tool>
    <tool>
        <name>playwright_browser_wait_for</name>
        <description>
            Waits for page state changes, elements, or timing conditions when the page needs time to
            update after an action.
        </description>
    </tool>
</tools>
"""


class BrowserStepDeps(BaseModel):
    current_step: str


provider = GoogleProvider(api_key=settings.GOOGLE_API_KEY)
model = GoogleModel(settings.MODEL_NAME, provider=provider)

logger = logging.getLogger(__name__)

BROWSER_AGENT_RETRIES = 1
BROWSER_AGENT_TIMEOUT_SECONDS = 45.0
CURRENT_TAB_URL_PATTERN = re.compile(r"\(current\)\s.*\((?P<url>[^)]+)\)")
PAGE_URL_PATTERN = re.compile(r"^- Page URL:\s*(?P<url>.+)$", re.MULTILINE)


def get_playwright_user_data_dir() -> Path:
    user_data_dir = settings.PLAYWRIGHT_USER_DATA_DIR.resolve()
    user_data_dir.mkdir(parents=True, exist_ok=True)
    return user_data_dir


@lru_cache(maxsize=1)
def get_playwright_mcp_server() -> MCPServerStdio:
    return MCPServerStdio(
        "npx",
        args=[
            "-y",
            "@playwright/mcp@latest",
            "--user-data-dir",
            str(get_playwright_user_data_dir()),
        ],
        cwd=ROOT_DIR,
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
        retries=BROWSER_AGENT_RETRIES,
        model_settings=ModelSettings(
            temperature=0.2,
            timeout=BROWSER_AGENT_TIMEOUT_SECONDS,
        ),
        output_type=BrowserStepResult,
        toolsets=[get_playwright_mcp_server()],
    )


def _extract_current_url(raw_result: object) -> str:
    result_text = str(raw_result)
    current_tab_match = CURRENT_TAB_URL_PATTERN.search(result_text)
    if current_tab_match:
        return current_tab_match.group("url").strip()

    page_url_match = PAGE_URL_PATTERN.search(result_text)
    if page_url_match:
        return page_url_match.group("url").strip()

    return ""


async def get_current_browser_url() -> str:
    server = get_playwright_mcp_server()
    if not server.is_running:
        return ""

    try:
        tabs_result = await server.direct_call_tool("browser_tabs", {"action": "list"})
        current_url = _extract_current_url(tabs_result)
        if current_url:
            return current_url
    except Exception:
        logger.exception("Failed to read current browser URL from browser_tabs")

    try:
        snapshot_result = await server.direct_call_tool("browser_snapshot", {})
        current_url = _extract_current_url(snapshot_result)
        if current_url:
            return current_url
    except Exception:
        logger.exception("Failed to read current browser URL from browser_snapshot")

    return ""


async def run_browser_step(
    current_step: str,
    human_response: HumanActionResponse | None = None,
    message_history: Sequence[ModelMessage] | None = None,
) -> AgentRunResult[BrowserStepResult]:
    logger.info("Browser agent started step execution")
    logger.info("Browser current step: %s", current_step)
    user_prompt = current_step

    if human_response is not None:
        if human_response.action == "provide_value":
            user_prompt = (
                f"{current_step}\n\n"
                "Human assistance context:\n"
                "The human provided the requested value for this step. Use it now, but do not repeat "
                "the value in your summary or answer.\n"
                f"Provided value: {human_response.value}"
            )
        elif human_response.action == "manual_done":
            user_prompt = (
                f"{current_step}\n\n"
                "Human assistance context:\n"
                "The human completed the required manual action in the browser. Inspect the current "
                "page state, verify what changed, and continue from there."
            )

    result = await get_browser_agent().run(
        user_prompt=user_prompt,
        deps=BrowserStepDeps(current_step=current_step),
        message_history=message_history,
    )

    logger.info("Browser agent completed step execution")
    logger.info("Browser agent status: %s", result.output.status)
    logger.info("Browser agent summary: %s", result.output.summary)
    return result


browser_agent = get_browser_agent()
