"""Tests for unified_llm.stream — high-level streaming generation."""
from __future__ import annotations

import json
from typing import Any

import pytest

from unified_llm.adapter import StubAdapter
from unified_llm.client import Client
from unified_llm.stream import stream, stream_object
from unified_llm.types.enums import FinishReason, Role, StreamEventType
from unified_llm.types.messages import Message
from unified_llm.types.response import FinishReasonInfo, Usage
from unified_llm.types.streaming import StreamEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text_events(text: str = "Hello") -> list[StreamEvent]:
    """Create a minimal text-streaming event sequence."""
    return [
        StreamEvent(type=StreamEventType.STREAM_START),
        StreamEvent(type=StreamEventType.TEXT_START, text_id="t0"),
        StreamEvent(type=StreamEventType.TEXT_DELTA, delta=text, text_id="t0"),
        StreamEvent(type=StreamEventType.TEXT_END, text_id="t0"),
        StreamEvent(
            type=StreamEventType.FINISH,
            finish_reason=FinishReasonInfo(reason=FinishReason.STOP),
            usage=Usage(input_tokens=5, output_tokens=3, total_tokens=8),
        ),
    ]


def _stub_stream_client(
    events: list[list[StreamEvent]],
) -> tuple[Client, StubAdapter]:
    """Build a Client backed by a StubAdapter with stream events."""
    adapter = StubAdapter(stream_events=events)
    client = Client(providers={"stub": adapter}, default_provider="stub")
    return client, adapter


# ===================================================================
# TestStream
# ===================================================================


class TestStream:
    """Tests for the stream() function."""

    def test_basic_text_streaming(self) -> None:
        """stream() yields events from the stub adapter."""
        events = _text_events("World")
        client, _ = _stub_stream_client([events])

        collected = list(stream("test-model", prompt="Hi", client=client))

        assert len(collected) == len(events)
        deltas = [e.delta for e in collected if e.type == StreamEventType.TEXT_DELTA]
        assert deltas == ["World"]

    def test_system_message_prepended(self) -> None:
        """System message is included in the request."""
        events = _text_events()
        client, adapter = _stub_stream_client([events])

        list(stream("test-model", prompt="Hi", system="Be kind", client=client))

        req = adapter.requests[0]
        assert req.messages[0].role == Role.SYSTEM
        assert req.messages[0].text == "Be kind"
        assert req.messages[1].role == Role.USER

    def test_with_prompt(self) -> None:
        """A prompt string is turned into a user message."""
        events = _text_events()
        client, adapter = _stub_stream_client([events])

        list(stream("test-model", prompt="Hello there", client=client))

        req = adapter.requests[0]
        user_msgs = [m for m in req.messages if m.role == Role.USER]
        assert len(user_msgs) == 1
        assert user_msgs[0].text == "Hello there"

    def test_with_messages(self) -> None:
        """A list of messages is used directly."""
        events = _text_events()
        client, adapter = _stub_stream_client([events])
        msgs = [Message.user("First"), Message.assistant("Reply"), Message.user("Second")]

        list(stream("test-model", messages=msgs, client=client))

        req = adapter.requests[0]
        assert len(req.messages) == 3
        assert req.messages[0].text == "First"
        assert req.messages[2].text == "Second"

    def test_both_prompt_and_messages_raises(self) -> None:
        """Providing both raises ValueError."""
        with pytest.raises(ValueError, match="not both"):
            events = _text_events()
            client, _ = _stub_stream_client([events])
            list(stream(
                "test-model",
                prompt="Hi",
                messages=[Message.user("Hello")],
                client=client,
            ))

    def test_neither_prompt_nor_messages_raises(self) -> None:
        """Providing neither raises ValueError."""
        with pytest.raises(ValueError, match="must be provided"):
            events = _text_events()
            client, _ = _stub_stream_client([events])
            list(stream("test-model", client=client))

    def test_provider_passed_through(self) -> None:
        """Provider name is forwarded in the request."""
        events = _text_events()
        adapter = StubAdapter(stream_events=[events])
        client = Client(providers={"custom": adapter}, default_provider="custom")

        list(stream("test-model", prompt="Hi", provider="custom", client=client))

        assert adapter.requests[0].provider == "custom"

    def test_custom_client(self) -> None:
        """A custom client is used."""
        events = _text_events("Custom")
        client, adapter = _stub_stream_client([events])

        collected = list(stream("test-model", prompt="Hi", client=client))

        assert len(adapter.requests) == 1
        deltas = [e.delta for e in collected if e.delta]
        assert "Custom" in deltas

    def test_temperature_forwarded(self) -> None:
        """Temperature is passed to the request."""
        events = _text_events()
        client, adapter = _stub_stream_client([events])

        list(stream("test-model", prompt="Hi", temperature=0.5, client=client))

        assert adapter.requests[0].temperature == 0.5

    def test_finish_event_present(self) -> None:
        """The FINISH event carries finish_reason and usage."""
        events = _text_events()
        client, _ = _stub_stream_client([events])

        collected = list(stream("test-model", prompt="Hi", client=client))

        finish_events = [e for e in collected if e.type == StreamEventType.FINISH]
        assert len(finish_events) == 1
        assert finish_events[0].finish_reason.reason == FinishReason.STOP
        assert finish_events[0].usage.total_tokens == 8


# ===================================================================
# TestStreamObject
# ===================================================================


class TestStreamObject:
    """Tests for the stream_object() function."""

    def test_yields_events(self) -> None:
        """stream_object() yields the same events as stream()."""
        events = _text_events('{"name": "Alice"}')
        client, _ = _stub_stream_client([events])

        collected = list(stream_object(
            "test-model",
            prompt="Generate person",
            schema={"type": "object", "properties": {"name": {"type": "string"}}},
            client=client,
        ))

        assert len(collected) == len(events)

    def test_system_prompt_includes_schema(self) -> None:
        """Schema instruction is injected into the system prompt."""
        events = _text_events('{}')
        client, adapter = _stub_stream_client([events])

        schema = {"type": "object"}
        list(stream_object(
            "test-model",
            prompt="Generate",
            schema=schema,
            client=client,
        ))

        req = adapter.requests[0]
        system_msgs = [m for m in req.messages if m.role == Role.SYSTEM]
        assert len(system_msgs) == 1
        assert "json" in system_msgs[0].text.lower() or "JSON" in system_msgs[0].text
        assert json.dumps(schema) in system_msgs[0].text

    def test_custom_system_preserved(self) -> None:
        """User's system prompt is preserved alongside schema instruction."""
        events = _text_events('{}')
        client, adapter = _stub_stream_client([events])

        list(stream_object(
            "test-model",
            prompt="Generate",
            system="You are helpful.",
            schema={"type": "object"},
            client=client,
        ))

        req = adapter.requests[0]
        system_msgs = [m for m in req.messages if m.role == Role.SYSTEM]
        assert len(system_msgs) == 1
        system_text = system_msgs[0].text
        assert "You are helpful." in system_text
        assert "JSON" in system_text or "json" in system_text

    def test_stream_object_no_prompt_no_messages_raises(self) -> None:
        """stream_object() with no prompt or messages raises ValueError."""
        events = _text_events()
        client, _ = _stub_stream_client([events])

        with pytest.raises(ValueError, match="must be provided"):
            list(stream_object(
                "test-model",
                schema={"type": "object"},
                client=client,
            ))

    def test_stream_object_with_messages(self) -> None:
        """stream_object() works with messages instead of prompt."""
        events = _text_events('{"result": 42}')
        client, adapter = _stub_stream_client([events])

        collected = list(stream_object(
            "test-model",
            messages=[Message.user("Compute 6*7")],
            schema={"type": "object", "properties": {"result": {"type": "number"}}},
            client=client,
        ))

        assert len(collected) == len(events)
        req = adapter.requests[0]
        # System message + user message
        assert req.messages[0].role == Role.SYSTEM  # schema injection
        assert req.messages[1].role == Role.USER
