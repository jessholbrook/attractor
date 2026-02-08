"""Tests for the agent loop event system."""
from __future__ import annotations

import pytest

from agent_loop.events import (
    AssistantTextDeltaEvent,
    AssistantTextEndEvent,
    AssistantTextStartEvent,
    ErrorEvent,
    EventEmitter,
    EventKind,
    LoopDetectionEvent,
    SessionEndEvent,
    SessionStartEvent,
    SteeringInjectedEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolOutputDeltaEvent,
    TurnLimitEvent,
    UserInputEvent,
)


# --- EventEmitter tests ---


class TestEventEmitterSubscribeAndEmit:
    def test_subscribe_and_emit_dispatches_to_typed_callback(self) -> None:
        emitter = EventEmitter()
        received: list = []
        emitter.subscribe(SessionStartEvent, received.append)

        event = SessionStartEvent(session_id="abc")
        emitter.emit(event)

        assert received == [event]

    def test_on_all_receives_every_event_type(self) -> None:
        emitter = EventEmitter()
        received: list = []
        emitter.on_all(received.append)

        e1 = SessionStartEvent(session_id="s1")
        e2 = UserInputEvent(content="hello")
        e3 = ErrorEvent(error="boom")
        emitter.emit(e1)
        emitter.emit(e2)
        emitter.emit(e3)

        assert received == [e1, e2, e3]

    def test_multiple_listeners_for_same_type_all_called(self) -> None:
        emitter = EventEmitter()
        first: list = []
        second: list = []
        emitter.subscribe(UserInputEvent, first.append)
        emitter.subscribe(UserInputEvent, second.append)

        event = UserInputEvent(content="hi")
        emitter.emit(event)

        assert first == [event]
        assert second == [event]

    def test_emit_with_no_listeners_does_not_error(self) -> None:
        emitter = EventEmitter()
        # Should not raise
        emitter.emit(SessionStartEvent(session_id="x"))

    def test_listeners_called_in_registration_order(self) -> None:
        emitter = EventEmitter()
        order: list[int] = []
        emitter.subscribe(ErrorEvent, lambda _: order.append(1))
        emitter.subscribe(ErrorEvent, lambda _: order.append(2))
        emitter.subscribe(ErrorEvent, lambda _: order.append(3))

        emitter.emit(ErrorEvent(error="test"))

        assert order == [1, 2, 3]

    def test_global_listener_called_before_typed_listener(self) -> None:
        emitter = EventEmitter()
        order: list[str] = []
        emitter.subscribe(SessionStartEvent, lambda _: order.append("typed"))
        emitter.on_all(lambda _: order.append("global"))

        emitter.emit(SessionStartEvent(session_id="s"))

        assert order == ["global", "typed"]


# --- EventKind tests ---


class TestEventKind:
    def test_event_kind_has_all_13_members(self) -> None:
        assert len(EventKind) == 13

    def test_event_kind_values(self) -> None:
        expected = {
            "session_start",
            "session_end",
            "user_input",
            "assistant_text_start",
            "assistant_text_delta",
            "assistant_text_end",
            "tool_call_start",
            "tool_output_delta",
            "tool_call_end",
            "steering_injected",
            "turn_limit",
            "loop_detection",
            "error",
        }
        assert {e.value for e in EventKind} == expected


# --- Event dataclass tests ---


class TestSessionStartEvent:
    def test_construction_and_immutability(self) -> None:
        event = SessionStartEvent(session_id="abc-123")
        assert event.session_id == "abc-123"
        with pytest.raises(AttributeError):
            event.session_id = "changed"  # type: ignore[misc]


class TestSessionEndEvent:
    def test_default_reason(self) -> None:
        event = SessionEndEvent(session_id="s1")
        assert event.session_id == "s1"
        assert event.reason == "completed"

    def test_custom_reason(self) -> None:
        event = SessionEndEvent(session_id="s1", reason="cancelled")
        assert event.reason == "cancelled"


class TestUserInputEvent:
    def test_construction(self) -> None:
        event = UserInputEvent(content="hello world")
        assert event.content == "hello world"


class TestAssistantTextStartEvent:
    def test_is_empty_frozen_dataclass(self) -> None:
        event = AssistantTextStartEvent()
        # No fields to check, just verify it's frozen
        with pytest.raises(AttributeError):
            event.x = 1  # type: ignore[attr-defined]


class TestAssistantTextDeltaEvent:
    def test_has_text_field(self) -> None:
        event = AssistantTextDeltaEvent(text="chunk")
        assert event.text == "chunk"


class TestAssistantTextEndEvent:
    def test_has_full_text_and_reasoning(self) -> None:
        event = AssistantTextEndEvent(full_text="complete response", reasoning="thought")
        assert event.full_text == "complete response"
        assert event.reasoning == "thought"

    def test_reasoning_defaults_to_empty(self) -> None:
        event = AssistantTextEndEvent(full_text="done")
        assert event.reasoning == ""


class TestToolCallStartEvent:
    def test_with_arguments(self) -> None:
        args = {"path": "/tmp/file.txt", "content": "data"}
        event = ToolCallStartEvent(
            tool_call_id="tc-1", tool_name="write_file", arguments=args
        )
        assert event.tool_call_id == "tc-1"
        assert event.tool_name == "write_file"
        assert event.arguments == args

    def test_arguments_default_to_empty_dict(self) -> None:
        event = ToolCallStartEvent(tool_call_id="tc-2", tool_name="read_file")
        assert event.arguments == {}


class TestToolCallEndEvent:
    def test_with_is_error_flag(self) -> None:
        event = ToolCallEndEvent(
            tool_call_id="tc-1",
            tool_name="bash",
            output="exit code 1",
            is_error=True,
        )
        assert event.tool_call_id == "tc-1"
        assert event.tool_name == "bash"
        assert event.output == "exit code 1"
        assert event.is_error is True

    def test_is_error_defaults_to_false(self) -> None:
        event = ToolCallEndEvent(
            tool_call_id="tc-2", tool_name="bash", output="success"
        )
        assert event.is_error is False


class TestTurnLimitEvent:
    def test_construction(self) -> None:
        event = TurnLimitEvent(turns_used=10, max_turns=15)
        assert event.turns_used == 10
        assert event.max_turns == 15


class TestLoopDetectionEvent:
    def test_construction(self) -> None:
        event = LoopDetectionEvent(message="detected loop", pattern="A->B->A")
        assert event.message == "detected loop"
        assert event.pattern == "A->B->A"

    def test_pattern_defaults_to_empty(self) -> None:
        event = LoopDetectionEvent(message="loop")
        assert event.pattern == ""


class TestErrorEvent:
    def test_with_recoverable_flag(self) -> None:
        event = ErrorEvent(error="rate limit", recoverable=True)
        assert event.error == "rate limit"
        assert event.recoverable is True

    def test_non_recoverable(self) -> None:
        event = ErrorEvent(error="fatal", recoverable=False)
        assert event.recoverable is False

    def test_recoverable_defaults_to_true(self) -> None:
        event = ErrorEvent(error="oops")
        assert event.recoverable is True
