"""LLM client protocol and stub implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from agent_loop.turns import Role, ToolCall


@dataclass(frozen=True)
class Message:
    """A message in the LLM conversation format."""

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None  # tool name for tool role messages

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str = "", tool_calls: list[ToolCall] | None = None) -> Message:
        return cls(role=Role.ASSISTANT, content=content, tool_calls=tool_calls)

    @classmethod
    def tool(cls, tool_call_id: str, content: str, name: str = "") -> Message:
        return cls(role=Role.TOOL, content=content, tool_call_id=tool_call_id, name=name)


@dataclass(frozen=True)
class CompletionRequest:
    """Request to the LLM."""

    messages: list[Message]
    model: str = ""
    tools: list[dict[str, Any]] = field(default_factory=list)
    system: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    provider_options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompletionResponse:
    """Response from the LLM."""

    message: Message
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""
    stop_reason: str = "end_turn"

    @property
    def text(self) -> str:
        return self.message.content

    @property
    def tool_calls(self) -> list[ToolCall]:
        return self.message.tool_calls or []


class Client(Protocol):
    """Protocol for LLM clients."""

    def complete(self, request: CompletionRequest) -> CompletionResponse: ...


class StubClient:
    """Test stub that returns predefined responses in sequence.

    When all responses are exhausted, cycles the last one.
    If no responses provided, returns a default text-only response.
    """

    def __init__(self, responses: list[CompletionResponse] | None = None) -> None:
        self._responses = responses or [
            CompletionResponse(message=Message.assistant("Hello! How can I help?"))
        ]
        self._index = 0
        self._requests: list[CompletionRequest] = []

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        self._requests.append(request)
        if self._index < len(self._responses):
            response = self._responses[self._index]
            self._index += 1
        else:
            response = self._responses[-1]
        return response

    @property
    def call_count(self) -> int:
        return len(self._requests)

    @property
    def requests(self) -> list[CompletionRequest]:
        return list(self._requests)
