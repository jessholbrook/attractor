"""Multimodal message type."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from unified_llm.types.enums import ContentKind, Role
from unified_llm.types.content import ContentPart, ToolCallData, ToolResultData


@dataclass(frozen=True)
class Message:
    """A single message in a conversation."""

    role: Role
    content: tuple[ContentPart, ...] = ()
    name: str | None = None
    tool_call_id: str | None = None

    # --- Factory classmethods ---

    @classmethod
    def system(cls, text: str) -> Message:
        """Create a system message."""
        return cls(
            role=Role.SYSTEM,
            content=(ContentPart.of_text(text),),
        )

    @classmethod
    def user(cls, text: str) -> Message:
        """Create a user message."""
        return cls(
            role=Role.USER,
            content=(ContentPart.of_text(text),),
        )

    @classmethod
    def assistant(
        cls,
        text: str = "",
        tool_calls: Sequence[ToolCallData] | None = None,
    ) -> Message:
        """Create an assistant message with optional tool calls."""
        parts: list[ContentPart] = []
        if text:
            parts.append(ContentPart.of_text(text))
        if tool_calls:
            for tc in tool_calls:
                parts.append(
                    ContentPart.of_tool_call(
                        id=tc.id, name=tc.name, arguments=tc.arguments,
                    )
                )
        return cls(role=Role.ASSISTANT, content=tuple(parts))

    @classmethod
    def developer(cls, text: str) -> Message:
        """Create a developer message."""
        return cls(
            role=Role.DEVELOPER,
            content=(ContentPart.of_text(text),),
        )

    @classmethod
    def tool_result(
        cls,
        tool_call_id: str,
        content: str,
        is_error: bool = False,
    ) -> Message:
        """Create a tool result message."""
        return cls(
            role=Role.TOOL,
            content=(
                ContentPart.of_tool_result(
                    tool_call_id=tool_call_id,
                    content=content,
                    is_error=is_error,
                ),
            ),
            tool_call_id=tool_call_id,
        )

    # --- Properties ---

    @property
    def text(self) -> str:
        """Concatenate all TEXT content part texts. Returns '' if none."""
        parts = [
            p.text for p in self.content
            if p.kind == ContentKind.TEXT and p.text is not None
        ]
        return "".join(parts)
