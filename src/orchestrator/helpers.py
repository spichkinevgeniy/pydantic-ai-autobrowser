import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace
from pprint import pformat
from typing import Any, TypeVar

from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelRequest, ToolReturnPart
from src.config import settings

T = TypeVar("T")


def ensure_tool_response_sequence(messages: Sequence[Any]) -> list[Any]:
    """Ensure that every tool call has a corresponding tool response."""
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


def strip_snapshot_refs(text: str | None) -> str:
    """Remove volatile Playwright snapshot refs before forwarding text across steps."""
    if not text:
        return ""

    sanitized = re.sub(r"`?ref\s*=\s*[^`\s,.;:)\]]+`?", "[snapshot-ref]", text)
    sanitized = re.sub(r"\[ref=[^\]]+\]", "[ref]", sanitized)
    return sanitized


def build_critique_tool_response(
    tool_interactions_str: str | None,
    browser_summary: str,
) -> str:
    """Build critique input from tool interactions and the browser agent summary."""
    sections: list[str] = []
    if tool_interactions_str:
        sections.append("Tool interactions:\n" + strip_snapshot_refs(tool_interactions_str).rstrip())
    if browser_summary:
        sections.append("Browser summary:\n" + strip_snapshot_refs(browser_summary).strip())
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
    """Format payloads for detailed debug logging."""
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


def is_transient_model_error(error: Exception) -> bool:
    """Return True for model/API overload and timeout style errors."""
    if isinstance(error, ModelHTTPError):
        return error.status_code in {429, 503, 504}

    error_text = str(error)
    transient_markers = (
        "429",
        "503",
        "504",
        "UNAVAILABLE",
        "DEADLINE_EXCEEDED",
        "RESOURCE_EXHAUSTED",
        "high demand",
    )
    return any(marker in error_text for marker in transient_markers)


async def run_with_transient_retry(
    label: str,
    operation: Callable[[], Awaitable[T]],
) -> T:
    """Retry transient model failures with exponential backoff."""
    max_attempts = max(1, settings.TRANSIENT_RETRY_ATTEMPTS)
    base_delay = max(0.0, settings.TRANSIENT_RETRY_BASE_DELAY_SECONDS)
    logger = logging.getLogger("TransientRetry")

    for attempt in range(1, max_attempts + 1):
        try:
            return await operation()
        except Exception as exc:
            is_last_attempt = attempt >= max_attempts
            if not is_transient_model_error(exc) or is_last_attempt:
                raise

            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "%s transient model error on attempt %s/%s: %s. Retrying in %.1fs",
                label,
                attempt,
                max_attempts,
                exc,
                delay,
            )
            await asyncio.sleep(delay)

    raise RuntimeError(f"{label} retry loop exited unexpectedly")
