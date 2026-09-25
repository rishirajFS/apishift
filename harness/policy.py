"""AsyncPolicy backed by an Inspect AI model (any provider, e.g. vLLM on Modal)."""

from __future__ import annotations

import json
from typing import Any

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentReasoning,
    Model,
)
from inspect_ai.tool import ToolCall as InspectToolCall
from inspect_ai.tool import ToolInfo, ToolParams

from apishift.agent.types import AssistantTurn, Message, ToolCall

# Sent to the environment when the model's arguments were not valid JSON, so it
# gets the same malformed_arguments error a real API would return.
UNPARSEABLE = "\x00unparseable-arguments"
MALFORMED_TOOL = "malformed_tool_call"


def _loads(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def to_inspect_messages(messages: list[Message]) -> list[ChatMessage]:
    out: list[ChatMessage] = []
    for m in messages:
        role = m["role"]
        if role == "system":
            out.append(ChatMessageSystem(content=m["content"]))
        elif role == "user":
            out.append(ChatMessageUser(content=m["content"]))
        elif role == "assistant":
            calls = [
                InspectToolCall(id=tc["id"], function=tc["function"]["name"],
                                arguments=_loads(tc["function"]["arguments"]), type="function")
                for tc in m.get("tool_calls", [])
            ]
            out.append(ChatMessageAssistant(content=m["content"] or "", tool_calls=calls or None))
        elif role == "tool":
            out.append(ChatMessageTool(content=m["content"], tool_call_id=m["tool_call_id"], function=m["name"]))
        else:
            raise ValueError(f"unknown role {role}")
    return out


def to_tool_info(tool: dict[str, Any]) -> ToolInfo:
    fn = tool["function"]
    return ToolInfo(name=fn["name"], description=fn["description"],
                    parameters=ToolParams.model_validate(fn["parameters"]))


class InspectPolicy:
    def __init__(self, model: Model, name: str) -> None:
        self.model = model
        self.name = name

    async def aact(self, messages: list[Message], tools: list[dict[str, Any]]) -> AssistantTurn:
        output = await self.model.generate(
            to_inspect_messages(messages), tools=[to_tool_info(t) for t in tools], tool_choice="auto"
        )
        msg = output.message
        calls = tuple(
            ToolCall(
                id=tc.id,
                name=tc.function,
                arguments=UNPARSEABLE if tc.parse_error else json.dumps(tc.arguments, sort_keys=True),
            )
            for tc in (msg.tool_calls or [])
        )
        text = msg.text or ""
        if not calls and "<tool_call>" in text:
            # The server's tool parser rejected a malformed call and returned it as text.
            # Count it as an invalid call (malformed_arguments) instead of a final answer.
            calls = (ToolCall(id=f"malformed_{len(messages)}", name=MALFORMED_TOOL, arguments=UNPARSEABLE),)
        reasoning = "\n".join(c.reasoning for c in msg.content if isinstance(c, ContentReasoning)) \
            if isinstance(msg.content, list) else ""
        return AssistantTurn(content=text, tool_calls=calls, reasoning=reasoning)
