import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orchestrator.helpers import (
    filter_tool_interactions_for_critique,
    strip_snapshot_refs,
)
from src.orchestrator.workflow import (
    build_planner_prompt,
    detect_security_approval_request,
    looks_like_refusal_text,
)


def test_looks_like_refusal_text_detects_known_refusal_markers() -> None:
    assert looks_like_refusal_text("I cannot access your private inbox due to privacy restrictions.")
    assert looks_like_refusal_text("This task cannot be performed from the current page.")


def test_looks_like_refusal_text_ignores_regular_browser_steps() -> None:
    assert not looks_like_refusal_text("Open Gmail login page.")
    assert not looks_like_refusal_text("Click the search box and enter RTX 3060 Ti.")


def test_build_planner_prompt_appends_current_url() -> None:
    prompt = build_planner_prompt(
        base_prompt="User Query: Find flights\nFeedback: None",
        current_url="https://example.com/search",
    )

    assert prompt == (
        "User Query: Find flights\n"
        "Feedback: None\n"
        "Current URL: https://example.com/search"
    )


def test_build_planner_prompt_replaces_existing_current_url_line() -> None:
    prompt = build_planner_prompt(
        base_prompt=(
            "User Query: Find flights\n"
            "Feedback: Retry with a safer step\n"
            "Current URL: https://old.example.com/page"
        ),
        current_url="https://new.example.com/page",
    )

    assert prompt == (
        "User Query: Find flights\n"
        "Feedback: Retry with a safer step\n"
        "Current URL: https://new.example.com/page"
    )


def test_build_planner_prompt_uses_about_blank_when_url_missing() -> None:
    prompt = build_planner_prompt(
        base_prompt="User Query: Open dashboard\nFeedback: None",
        current_url="",
    )

    assert prompt.endswith("Current URL: about:blank")


def test_detect_security_approval_request_for_payment_step() -> None:
    request = detect_security_approval_request("Click the checkout button to complete purchase")

    assert request is not None
    assert request.kind == "security_approval"
    assert request.response_mode == "approval_confirmation"
    assert "charge money" in request.reason


def test_detect_security_approval_request_for_delete_step() -> None:
    request = detect_security_approval_request("Delete the selected email thread permanently")

    assert request is not None
    assert request.kind == "security_approval"
    assert request.response_mode == "approval_confirmation"
    assert "permanently remove data" in request.reason


def test_detect_security_approval_request_returns_none_for_safe_step() -> None:
    assert detect_security_approval_request("Open the profile settings page") is None


def test_filter_tool_interactions_for_critique_compresses_dom_like_tools() -> None:
    tool_interactions = (
        "Tool Call: playwright_browser_snapshot\n"
        "Arguments: {}\n"
        "Response: ### Snapshot\n"
        "Page URL: https://example.com\n"
        "---\n"
        "Tool Call: playwright_browser_click\n"
        'Arguments: {"element":"Login"}\n'
        "Response: Clicked successfully\n"
        "---\n"
    )

    filtered = filter_tool_interactions_for_critique(tool_interactions)

    assert "Response: playwright_browser_snapshot completed successfully" in filtered
    assert "Page URL: https://example.com" not in filtered
    assert "Response: Clicked successfully" in filtered


def test_filter_tool_interactions_for_critique_compresses_snapshot_like_response_even_for_other_tool() -> None:
    tool_interactions = (
        "Tool Call: playwright_browser_run_code\n"
        'Arguments: {"script":"return document.body.innerText"}\n'
        "Response: ### Snapshot\n"
        "Some large DOM payload\n"
        "---\n"
    )

    filtered = filter_tool_interactions_for_critique(tool_interactions)

    assert filtered == (
        "Tool Call: playwright_browser_run_code\n"
        'Arguments: {"script":"return document.body.innerText"}\n'
        "Response: playwright_browser_run_code completed successfully---\n"
    )


def test_strip_snapshot_refs_replaces_ref_markers() -> None:
    text = (
        "Click button ref=abc123 and inspect [ref=node-7]. "
        "Then verify `ref = xyz-9` in the summary."
    )

    assert strip_snapshot_refs(text) == (
        "Click button [snapshot-ref] and inspect [[snapshot-ref]]. "
        "Then verify [snapshot-ref] in the summary."
    )


def test_strip_snapshot_refs_returns_empty_string_for_none() -> None:
    assert strip_snapshot_refs(None) == ""
