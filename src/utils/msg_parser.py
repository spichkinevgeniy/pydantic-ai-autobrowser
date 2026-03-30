import json
import uuid
from typing import Any, Dict, List

from openai.types.chat import ChatCompletionMessageParam


class AgentConversationHandler:
    """Handles conversion and storage of agent conversations in OpenAI format."""

    def __init__(self) -> None:
        self.conversation_history: list[ChatCompletionMessageParam] = []

    def _extract_tool_call(self, response_part: Any) -> Dict[str, Any]:
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

    def _extract_from_model_request(self, response: Any) -> List[Dict[str, Any]]:
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
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [self._extract_tool_call(part)],
                    })
                elif part_kind == "tool-return":
                    messages.append({
                        "role": "tool",
                        "tool_call_id": getattr(part, "tool_call_id", ""),
                        "name": getattr(part, "tool_name", ""),
                        "content": self._format_content(getattr(part, "content", "")),
                    })
                elif part_kind == "text":
                    messages.append({
                        "role": "assistant",
                        "content": getattr(part, "content", ""),
                        "name": "browser_nav_agent",
                    })

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
            "tool_calls": [{
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": "planner_agent",
                    "arguments": json.dumps({
                        "plan": plan,
                        "next_step": next_step,
                    }),
                },
            }],
        }
        self.conversation_history.append(assistant_message)

        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": "planner_agent",
            "content": json.dumps({
                "plan": plan,
                "next_step": next_step,
            }),
        }
        self.conversation_history.append(tool_message)
