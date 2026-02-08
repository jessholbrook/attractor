"""Tests for the core agentic loop Session."""

import pytest

from agent_loop.client import CompletionResponse, Message, StubClient
from agent_loop.environment.stub import StubExecutionEnvironment
from agent_loop.events import (
    AssistantTextEndEvent,
    EventEmitter,
    LoopDetectionEvent,
    SessionEndEvent,
    SteeringInjectedEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    TurnLimitEvent,
    UserInputEvent,
)
from agent_loop.providers.profile import StubProfile
from agent_loop.session import Session
from agent_loop.session_config import SessionConfig, SessionState
from agent_loop.tools.registry import RegisteredTool, ToolDefinition, ToolRegistry
from agent_loop.turns import (
    AssistantTurn,
    SteeringTurn,
    ToolCall,
    ToolResultsTurn,
    UserTurn,
)


# --- Helpers ---


def _make_text_response(text: str) -> CompletionResponse:
    return CompletionResponse(message=Message.assistant(text))


def _make_tool_response(tool_name: str, tool_id: str = "tc_1", args: dict | None = None) -> CompletionResponse:
    tc = ToolCall(id=tool_id, name=tool_name, arguments=args or {})
    return CompletionResponse(message=Message.assistant("", tool_calls=[tc]))


def _make_session(
    responses: list[CompletionResponse] | None = None,
    tools: dict[str, str] | None = None,
    config: SessionConfig | None = None,
) -> tuple[Session, list]:
    """Create a Session with stub client/env/profile, and an event collector."""
    client = StubClient(responses=responses)
    env = StubExecutionEnvironment()

    registry = ToolRegistry()
    if tools:
        for name, output in tools.items():
            registry.register(RegisteredTool(
                definition=ToolDefinition(name=name, description=f"Test {name}"),
                executor=lambda args, env, _out=output: _out,
            ))

    profile = StubProfile(registry=registry)
    emitter = EventEmitter()
    events: list = []
    emitter.on_all(lambda e: events.append(e))

    session = Session(
        llm_client=client,
        provider_profile=profile,
        execution_env=env,
        config=config,
        event_emitter=emitter,
    )
    return session, events


# --- Construction ---


class TestSessionConstruction:
    def test_default_state_is_idle(self):
        session, _ = _make_session()
        assert session.state == SessionState.IDLE

    def test_generates_uuid_id(self):
        session, _ = _make_session()
        assert len(session.id) == 36  # UUID format

    def test_custom_session_id(self):
        client = StubClient()
        env = StubExecutionEnvironment()
        profile = StubProfile()
        session = Session(client, profile, env, session_id="custom-123")
        assert session.id == "custom-123"

    def test_empty_history(self):
        session, _ = _make_session()
        assert session.history == []

    def test_default_config(self):
        session, _ = _make_session()
        assert session.config.max_tool_rounds_per_input == 200


# --- process_input: text-only response ---


class TestSessionProcessInput:
    def test_simple_text_response(self):
        session, _ = _make_session(responses=[_make_text_response("Hello!")])
        result = session.process_input("Hi")
        assert result.content == "Hello!"

    def test_appends_user_turn(self):
        session, _ = _make_session(responses=[_make_text_response("ok")])
        session.process_input("Hello")
        assert isinstance(session.history[0], UserTurn)
        assert session.history[0].content == "Hello"

    def test_appends_assistant_turn(self):
        session, _ = _make_session(responses=[_make_text_response("reply")])
        session.process_input("Hello")
        assert isinstance(session.history[1], AssistantTurn)
        assert session.history[1].content == "reply"

    def test_returns_to_idle_state(self):
        session, _ = _make_session(responses=[_make_text_response("done")])
        session.process_input("Hello")
        assert session.state == SessionState.IDLE

    def test_raises_when_closed(self):
        session, _ = _make_session()
        session.close()
        with pytest.raises(RuntimeError, match="closed"):
            session.process_input("Hello")


# --- Events ---


class TestSessionEvents:
    def test_emits_user_input_event(self):
        session, events = _make_session(responses=[_make_text_response("ok")])
        session.process_input("Hello")
        assert any(isinstance(e, UserInputEvent) and e.content == "Hello" for e in events)

    def test_emits_assistant_text_end_event(self):
        session, events = _make_session(responses=[_make_text_response("reply")])
        session.process_input("Hello")
        assert any(isinstance(e, AssistantTextEndEvent) and e.full_text == "reply" for e in events)


# --- Tool execution ---


class TestSessionToolExecution:
    def test_single_tool_call_executed(self):
        responses = [
            _make_tool_response("read_file"),
            _make_text_response("Done reading."),
        ]
        session, events = _make_session(
            responses=responses,
            tools={"read_file": "file content here"},
        )
        result = session.process_input("Read main.py")
        assert result.content == "Done reading."

    def test_tool_result_appended_to_history(self):
        responses = [
            _make_tool_response("read_file"),
            _make_text_response("Done."),
        ]
        session, _ = _make_session(
            responses=responses,
            tools={"read_file": "content"},
        )
        session.process_input("Read it")
        tool_turns = [t for t in session.history if isinstance(t, ToolResultsTurn)]
        assert len(tool_turns) == 1
        assert tool_turns[0].results[0].output == "content"

    def test_tool_call_events_emitted(self):
        responses = [
            _make_tool_response("shell"),
            _make_text_response("Done."),
        ]
        session, events = _make_session(
            responses=responses,
            tools={"shell": "output"},
        )
        session.process_input("Run it")
        assert any(isinstance(e, ToolCallStartEvent) for e in events)
        assert any(isinstance(e, ToolCallEndEvent) for e in events)

    def test_multi_round_tool_usage(self):
        responses = [
            _make_tool_response("read_file", "tc_1"),
            _make_tool_response("shell", "tc_2"),
            _make_text_response("All done."),
        ]
        session, _ = _make_session(
            responses=responses,
            tools={"read_file": "code", "shell": "ok"},
        )
        result = session.process_input("Read and run")
        assert result.content == "All done."
        tool_turns = [t for t in session.history if isinstance(t, ToolResultsTurn)]
        assert len(tool_turns) == 2


# --- Tool errors ---


class TestSessionToolErrors:
    def test_unknown_tool_returns_error(self):
        responses = [
            _make_tool_response("nonexistent_tool"),
            _make_text_response("Hmm."),
        ]
        session, _ = _make_session(responses=responses, tools={})
        session.process_input("Try it")
        tool_turns = [t for t in session.history if isinstance(t, ToolResultsTurn)]
        assert tool_turns[0].results[0].is_error is True
        assert "Unknown tool" in tool_turns[0].results[0].output

    def test_tool_exception_returns_error(self):
        registry = ToolRegistry()
        registry.register(RegisteredTool(
            definition=ToolDefinition(name="bad_tool", description="Fails"),
            executor=lambda args, env: (_ for _ in ()).throw(ValueError("boom")),
        ))
        client = StubClient(responses=[
            _make_tool_response("bad_tool"),
            _make_text_response("Recovered."),
        ])
        env = StubExecutionEnvironment()
        profile = StubProfile(registry=registry)
        session = Session(client, profile, env)
        session.process_input("Do it")
        tool_turns = [t for t in session.history if isinstance(t, ToolResultsTurn)]
        assert tool_turns[0].results[0].is_error is True
        assert "boom" in tool_turns[0].results[0].output


# --- Turn limits ---


class TestSessionTurnLimits:
    def test_max_tool_rounds_stops_processing(self):
        # Always return tool calls - should stop at round limit
        responses = [_make_tool_response("shell", f"tc_{i}") for i in range(10)]
        responses.append(_make_text_response("never reached"))
        session, events = _make_session(
            responses=responses,
            tools={"shell": "ok"},
            config=SessionConfig(max_tool_rounds_per_input=3),
        )
        session.process_input("Go")
        limit_events = [e for e in events if isinstance(e, TurnLimitEvent)]
        assert len(limit_events) == 1

    def test_unlimited_turns_when_zero(self):
        # With max_turns=0, no turn limit check fires
        responses = [
            _make_tool_response("shell", "tc_1"),
            _make_text_response("Done."),
        ]
        session, events = _make_session(
            responses=responses,
            tools={"shell": "ok"},
            config=SessionConfig(max_turns=0),
        )
        session.process_input("Go")
        limit_events = [e for e in events if isinstance(e, TurnLimitEvent)]
        assert len(limit_events) == 0


# --- Steering ---


class TestSessionSteering:
    def test_steer_adds_to_history(self):
        session, events = _make_session(responses=[
            _make_tool_response("shell"),
            _make_text_response("Ok."),
        ], tools={"shell": "ok"})
        session.steer("Focus on tests only")
        session.process_input("Do stuff")
        steering_turns = [t for t in session.history if isinstance(t, SteeringTurn)]
        assert len(steering_turns) >= 1
        assert any(t.content == "Focus on tests only" for t in steering_turns)

    def test_steering_event_emitted(self):
        session, events = _make_session(responses=[_make_text_response("ok")])
        session.steer("Redirect")
        session.process_input("Go")
        assert any(isinstance(e, SteeringInjectedEvent) and e.content == "Redirect" for e in events)

    def test_multiple_steering_messages(self):
        session, _ = _make_session(responses=[_make_text_response("ok")])
        session.steer("First")
        session.steer("Second")
        session.process_input("Go")
        steering_turns = [t for t in session.history if isinstance(t, SteeringTurn)]
        assert len(steering_turns) == 2
        assert steering_turns[0].content == "First"
        assert steering_turns[1].content == "Second"


# --- Follow-up ---


class TestSessionFollowUp:
    def test_follow_up_processed_after_completion(self):
        session, _ = _make_session(responses=[
            _make_text_response("First done."),
            _make_text_response("Follow-up done."),
        ])
        session.follow_up("Now do this")
        session.process_input("Start")
        user_turns = [t for t in session.history if isinstance(t, UserTurn)]
        assert len(user_turns) == 2
        assert user_turns[1].content == "Now do this"


# --- Abort ---


class TestSessionAbort:
    def test_abort_stops_processing(self):
        responses = [_make_tool_response("shell", f"tc_{i}") for i in range(10)]
        session, _ = _make_session(
            responses=responses,
            tools={"shell": "ok"},
        )
        # Abort immediately - should stop after first check
        session.abort()
        session.process_input("Go")
        # Should have very few turns since abort was set before processing
        tool_turns = [t for t in session.history if isinstance(t, ToolResultsTurn)]
        assert len(tool_turns) == 0


# --- Close ---


class TestSessionClose:
    def test_close_sets_state(self):
        session, _ = _make_session()
        session.close()
        assert session.state == SessionState.CLOSED

    def test_close_emits_event(self):
        session, events = _make_session()
        session.close()
        assert any(isinstance(e, SessionEndEvent) for e in events)


# --- Loop detection ---


class TestSessionLoopDetection:
    def test_loop_detected(self):
        # Return the same tool call 12 times (window=10)
        responses = [_make_tool_response("shell", f"tc_{i}", {"cmd": "echo same"}) for i in range(12)]
        responses.append(_make_text_response("Gave up."))
        session, events = _make_session(
            responses=responses,
            tools={"shell": "same output"},
            config=SessionConfig(loop_detection_window=10, max_tool_rounds_per_input=15),
        )
        session.process_input("Loop forever")
        loop_events = [e for e in events if isinstance(e, LoopDetectionEvent)]
        assert len(loop_events) >= 1

    def test_loop_detection_disabled(self):
        responses = [_make_tool_response("shell", f"tc_{i}", {"cmd": "echo same"}) for i in range(12)]
        responses.append(_make_text_response("Done."))
        session, events = _make_session(
            responses=responses,
            tools={"shell": "ok"},
            config=SessionConfig(enable_loop_detection=False, max_tool_rounds_per_input=15),
        )
        session.process_input("Go")
        loop_events = [e for e in events if isinstance(e, LoopDetectionEvent)]
        assert len(loop_events) == 0


# --- History conversion ---


class TestSessionHistoryConversion:
    def test_user_turn_becomes_user_message(self):
        session, _ = _make_session(responses=[_make_text_response("ok")])
        session.process_input("Hello")
        messages = session._convert_history_to_messages()
        assert messages[0].role.value == "user"
        assert messages[0].content == "Hello"

    def test_steering_turn_becomes_user_message(self):
        session, _ = _make_session(responses=[_make_text_response("ok")])
        session.steer("Focus")
        session.process_input("Go")
        messages = session._convert_history_to_messages()
        user_msgs = [m for m in messages if m.role.value == "user"]
        assert any(m.content == "Focus" for m in user_msgs)

    def test_tool_results_become_tool_messages(self):
        responses = [
            _make_tool_response("shell"),
            _make_text_response("Done."),
        ]
        session, _ = _make_session(responses=responses, tools={"shell": "output"})
        session.process_input("Run")
        messages = session._convert_history_to_messages()
        tool_msgs = [m for m in messages if m.role.value == "tool"]
        assert len(tool_msgs) == 1


# --- Multiple sequential inputs ---


class TestSessionMultipleInputs:
    def test_two_sequential_inputs(self):
        session, _ = _make_session(responses=[
            _make_text_response("First reply."),
            _make_text_response("Second reply."),
        ])
        r1 = session.process_input("First")
        r2 = session.process_input("Second")
        assert r1.content == "First reply."
        assert r2.content == "Second reply."
        assert len(session.history) == 4  # 2 user + 2 assistant
