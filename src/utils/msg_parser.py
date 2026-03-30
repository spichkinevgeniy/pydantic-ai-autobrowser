import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from openai.types.chat import ChatCompletionMessageParam
from src.config import ROOT_DIR

CONVERSATION_LOG_DIR = ROOT_DIR / "logs" / "conversations"


class AgentConversationHandler:
    """Handles conversion and storage of agent conversations in OpenAI format."""

    def __init__(self) -> None:
        self.conversation_history: list[ChatCompletionMessageParam] = []

    def add_browser_nav_message(self, browser_response: Any) -> None:
        """Convert and store browser navigation agent messages"""
        messages = self._extract_from_model_request(browser_response)
        self.conversation_history.extend(messages)

    def add_critique_message(self, critique_response: Any) -> None:
        """Convert and store critique agent messages"""
        if hasattr(critique_response, "output"):
            data = critique_response.output
            feedback = str(getattr(data, "feedback", ""))
            final_response = str(getattr(data, "final_response", ""))
        elif hasattr(critique_response, "data"):
            data = critique_response.data
            feedback = str(getattr(data, "feedback", ""))
            final_response = str(getattr(data, "final_response", ""))
        else:
            feedback = ""
            final_response = ""

        content = json.dumps({"feedback": feedback, "final_response": final_response})

        critique_message = {"role": "assistant", "content": content, "name": "critique_agent"}
        self.conversation_history.append(critique_message)

    def get_conversation_history(self) -> list[ChatCompletionMessageParam]:
        """Get the full conversation history in OpenAI format"""
        return self.conversation_history

    def _extract_tool_call(self, response_part: Any) -> dict[str, Any]:
        """Extract tool call information from a response part."""
        tool_call_id = getattr(response_part, "tool_call_id", str(uuid.uuid4()))
        tool_name = getattr(response_part, "tool_name", "")
        args = {}

        if hasattr(response_part, "args") and hasattr(response_part.args, "args_json"):
            try:
                args = json.loads(response_part.args.args_json)
            except json.JSONDecodeError:
                args = {"raw_args": response_part.args.args_json}

        return {
            "id": tool_call_id,
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(args),
            },
        }

    def _format_content(self, content: Any) -> str:
        """Format content into a string, handling various input types."""
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            return json.dumps(content, indent=2)
        if content is None:
            return ""

        try:
            return json.dumps(content, indent=2)
        except Exception:
            return str(content)

    def _extract_from_model_request(self, response: Any) -> list[dict[str, Any]]:
        """Extract all message components from a model response."""
        messages = []

        if not hasattr(response, "_all_messages"):
            return messages

        for msg in response._all_messages:
            if not hasattr(msg, "parts"):
                continue

            for part in msg.parts:
                part_kind = getattr(part, "part_kind", "")

                if part_kind == "tool-call":
                    messages.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [self._extract_tool_call(part)],
                        }
                    )
                elif part_kind == "tool-return":
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": getattr(part, "tool_call_id", ""),
                            "name": getattr(part, "tool_name", ""),
                            "content": self._format_content(getattr(part, "content", "")),
                        }
                    )
                elif part_kind == "text":
                    messages.append(
                        {
                            "role": "assistant",
                            "content": getattr(part, "content", ""),
                            "name": "browser_nav_agent",
                        }
                    )

        return messages

    def add_planner_message(self, planner_response: Any) -> None:
        """Convert and store planner agent messages."""
        plan = ""
        next_step = ""

        if hasattr(planner_response, "output"):
            data = planner_response.output
            plan = str(getattr(data, "plan", ""))
            next_step = str(getattr(data, "next_step", ""))
        elif hasattr(planner_response, "data"):
            data = planner_response.data
            plan = str(getattr(data, "plan", ""))
            next_step = str(getattr(data, "next_step", ""))

        tool_call_id = str(uuid.uuid4())

        assistant_message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": "planner_agent",
                        "arguments": json.dumps(
                            {
                                "plan": plan,
                                "next_step": next_step,
                            }
                        ),
                    },
                }
            ],
        }
        self.conversation_history.append(assistant_message)

        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": "planner_agent",
            "content": json.dumps(
                {
                    "plan": plan,
                    "next_step": next_step,
                }
            ),
        }
        self.conversation_history.append(tool_message)


class ConversationStorage:
    def __init__(self, storage_dir: str | Path | None = None):
        """
        Initialize conversation storage with configurable directory.
        If no storage_dir is provided, conversations are written to logs/conversations.
        """

        self.storage_dir = Path(storage_dir) if storage_dir else CONVERSATION_LOG_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.current_filepath: Path | None = None

    def _get_filepath(self, prefix: str = "") -> Path:
        """Get the filepath for the conversation, creating it if it doesn't exist."""
        if self.current_filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = (
                f"{prefix}_conversation_{timestamp}.json"
                if prefix
                else f"conversation_{timestamp}.json"
            )
            self.current_filepath = self.storage_dir / filename
        return self.current_filepath

    def _read_existing_messages(self, filepath: Path) -> list[dict[str, Any]]:
        """Read existing messages from the file if it exists."""
        try:
            if filepath.exists():
                with filepath.open("r", encoding="utf-8") as f:
                    return json.load(f)
        except json.JSONDecodeError:
            return []
        return []

    def _serialize_message(self, msg: Any) -> dict[str, Any]:
        """Convert message-like objects into a JSON-serializable dictionary."""
        if isinstance(msg, dict):
            return dict(msg)

        if hasattr(msg, "model_dump"):
            return msg.model_dump(mode="json")

        serialized: dict[str, Any] = {
            "role": getattr(msg, "role", None),
            "content": getattr(msg, "content", None),
        }

        for field_name in ("name", "tool_calls", "tool_call_id"):
            field_value = getattr(msg, field_name, None)
            if field_value is not None:
                serialized[field_name] = field_value

        return serialized

    def save_conversation(
        self,
        messages: list[ChatCompletionMessageParam],
        prefix: str = "",
    ) -> str:
        """
        Append conversation messages to a single JSON file.

        Args:
            messages: List of conversation messages
            prefix: Optional prefix for the filename

        Returns:
            str: Path to the saved file
        """
        filepath = self._get_filepath(prefix)
        serializable_messages = [self._serialize_message(msg) for msg in messages]
        existing_messages = self._read_existing_messages(filepath)

        last_message_index = len(existing_messages)
        new_messages = serializable_messages[last_message_index:]
        updated_messages = existing_messages + new_messages

        with filepath.open("w", encoding="utf-8") as f:
            json.dump(updated_messages, f, indent=2, ensure_ascii=False)

        return str(filepath)

    def reset_file(self) -> None:
        """Reset the current file path to create a new file for a new conversation."""
        self.current_filepath = None
