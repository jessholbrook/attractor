"""High-level streaming generation functions."""
from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from typing import Any

from unified_llm.client import Client, get_default_client
from unified_llm.types.config import AbortSignal, TimeoutConfig
from unified_llm.types.enums import StreamEventType
from unified_llm.types.messages import Message
from unified_llm.types.request import Request, ResponseFormat
from unified_llm.types.streaming import StreamEvent
from unified_llm.types.tools import Tool, ToolChoice


def stream(
    model: str,
    *,
    prompt: str | None = None,
    messages: Sequence[Message] | None = None,
    system: str | None = None,
    tools: Sequence[Tool] | None = None,
    tool_choice: ToolChoice | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    stop_sequences: Sequence[str] | None = None,
    reasoning_effort: str | None = None,
    provider: str | None = None,
    provider_options: dict[str, Any] | None = None,
    abort_signal: AbortSignal | None = None,
    client: Client | None = None,
) -> Iterator[StreamEvent]:
    """Stream generation results.

    Yields :class:`StreamEvent` instances as they arrive from the
    provider.  Either *prompt* or *messages* must be provided, but not
    both.
    """
    # Validate inputs
    if prompt is not None and messages is not None:
        raise ValueError("Provide either 'prompt' or 'messages', not both")
    if prompt is None and messages is None:
        raise ValueError("Either 'prompt' or 'messages' must be provided")

    # Build messages
    msg_list: list[Message] = []
    if system:
        msg_list.append(Message.system(system))
    if prompt is not None:
        msg_list.append(Message.user(prompt))
    else:
        assert messages is not None
        msg_list.extend(messages)

    c = client or get_default_client()

    request = Request(
        model=model,
        messages=tuple(msg_list),
        provider=provider,
        tools=tuple(tools) if tools else None,
        tool_choice=tool_choice,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop_sequences=tuple(stop_sequences) if stop_sequences else None,
        reasoning_effort=reasoning_effort,
        provider_options=provider_options,
    )

    yield from c.stream(request)


def stream_object(
    model: str,
    *,
    prompt: str | None = None,
    messages: Sequence[Message] | None = None,
    system: str | None = None,
    schema: dict[str, Any],
    temperature: float | None = None,
    max_tokens: int | None = None,
    provider: str | None = None,
    provider_options: dict[str, Any] | None = None,
    client: Client | None = None,
) -> Iterator[StreamEvent]:
    """Stream structured output generation.

    Injects the JSON schema into the system prompt so the model is
    guided to produce conforming output.  Yields the same
    :class:`StreamEvent` stream as :func:`stream`.
    """
    schema_instruction = (
        f"Respond with valid JSON matching this schema: {json.dumps(schema)}"
    )
    effective_system = (
        f"{system}\n\n{schema_instruction}" if system else schema_instruction
    )

    yield from stream(
        model=model,
        prompt=prompt,
        messages=messages,
        system=effective_system,
        temperature=temperature,
        max_tokens=max_tokens,
        provider=provider,
        provider_options=provider_options,
        client=client,
    )
