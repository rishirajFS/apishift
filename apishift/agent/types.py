"""Policy interface and OpenAI-format message types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

Message = dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string, exactly as the model emitted it


@dataclass(frozen=True)
class AssistantTurn:
    content: str
    tool_calls: tuple[ToolCall, ...] = ()

    def to_message(self) -> Message:
        msg: Message = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.name, "arguments": tc.arguments}}
                for tc in self.tool_calls
            ]
        return msg


class Policy(Protocol):
    name: str

    def act(self, messages: list[Message], tools: list[dict[str, Any]]) -> AssistantTurn: ...


class AsyncPolicy(Protocol):
    name: str

    async def aact(self, messages: list[Message], tools: list[dict[str, Any]]) -> AssistantTurn: ...
