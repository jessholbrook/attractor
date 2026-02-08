"""Tests for agent loop turn types."""

from __future__ import annotations

import dataclasses

import pytest

from agent_loop.turns import (
    AssistantTurn,
    Role,
    SteeringTurn,
    SystemTurn,
    ToolCall,
    ToolResult,
    ToolResultsTurn,
    Turn,
    UserTurn,
)


# ---------------------------------------------------------------------------
# Role enum
# ---------------------------------------------------------------------------


class TestRole:
    def test_role_values(self):
        assert Role.SYSTEM.value == "system"
        assert Role.USER.value == "user"
        assert Role.ASSISTANT.value == "assistant"
        assert Role.TOOL.value == "tool"


# ---------------------------------------------------------------------------
# ToolCall
# ---------------------------------------------------------------------------


class TestToolCall:
    def test_construction_and_field_access(self):
        tc = ToolCall(id="call_1", name="read_file", arguments={"path": "/tmp/x"})
        assert tc.id == "call_1"
        assert tc.name == "read_file"
        assert tc.arguments == {"path": "/tmp/x"}

    def test_empty_arguments_default(self):
        tc = ToolCall(id="call_2", name="list_dir")
        assert tc.arguments == {}

    def test_frozen_immutability(self):
        tc = ToolCall(id="call_3", name="search")
        with pytest.raises(dataclasses.FrozenInstanceError):
            tc.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------


class TestToolResult:
    def test_success_result(self):
        tr = ToolResult(tool_call_id="call_1", output="file contents here")
        assert tr.tool_call_id == "call_1"
        assert tr.output == "file contents here"
        assert tr.is_error is False

    def test_error_result(self):
        tr = ToolResult(tool_call_id="call_2", output="not found", is_error=True)
        assert tr.is_error is True

    def test_frozen_immutability(self):
        tr = ToolResult(tool_call_id="call_3")
        with pytest.raises(dataclasses.FrozenInstanceError):
            tr.output = "nope"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# UserTurn
# ---------------------------------------------------------------------------


class TestUserTurn:
    def test_construction(self):
        turn = UserTurn(content="Hello")
        assert turn.content == "Hello"

    def test_frozen_immutability(self):
        turn = UserTurn(content="Hello")
        with pytest.raises(dataclasses.FrozenInstanceError):
            turn.content = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AssistantTurn
# ---------------------------------------------------------------------------


class TestAssistantTurn:
    def test_defaults(self):
        turn = AssistantTurn()
        assert turn.content == ""
        assert turn.tool_calls == []
        assert turn.reasoning == ""
        assert turn.usage == {}
        assert turn.response_id == ""

    def test_with_tool_calls(self):
        tc = ToolCall(id="call_1", name="bash", arguments={"cmd": "ls"})
        turn = AssistantTurn(content="Running command", tool_calls=[tc])
        assert len(turn.tool_calls) == 1
        assert turn.tool_calls[0].name == "bash"

    def test_with_usage_tracking(self):
        turn = AssistantTurn(
            content="answer",
            usage={"input_tokens": 100, "output_tokens": 50},
            response_id="resp_abc",
        )
        assert turn.usage["input_tokens"] == 100
        assert turn.usage["output_tokens"] == 50
        assert turn.response_id == "resp_abc"

    def test_frozen_immutability(self):
        turn = AssistantTurn(content="answer")
        with pytest.raises(dataclasses.FrozenInstanceError):
            turn.content = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ToolResultsTurn
# ---------------------------------------------------------------------------


class TestToolResultsTurn:
    def test_empty_results_default(self):
        turn = ToolResultsTurn()
        assert turn.results == []

    def test_multiple_results(self):
        r1 = ToolResult(tool_call_id="c1", output="ok")
        r2 = ToolResult(tool_call_id="c2", output="fail", is_error=True)
        turn = ToolResultsTurn(results=[r1, r2])
        assert len(turn.results) == 2
        assert turn.results[0].is_error is False
        assert turn.results[1].is_error is True


# ---------------------------------------------------------------------------
# SystemTurn
# ---------------------------------------------------------------------------


class TestSystemTurn:
    def test_construction(self):
        turn = SystemTurn(content="You are a coding assistant.")
        assert turn.content == "You are a coding assistant."

    def test_frozen_immutability(self):
        turn = SystemTurn(content="system prompt")
        with pytest.raises(dataclasses.FrozenInstanceError):
            turn.content = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SteeringTurn
# ---------------------------------------------------------------------------


class TestSteeringTurn:
    def test_default_source_is_host(self):
        turn = SteeringTurn(content="focus on tests")
        assert turn.source == "host"

    def test_custom_source(self):
        turn = SteeringTurn(content="pause", source="supervisor")
        assert turn.source == "supervisor"

    def test_frozen_immutability(self):
        turn = SteeringTurn(content="steer")
        with pytest.raises(dataclasses.FrozenInstanceError):
            turn.source = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Turn type union
# ---------------------------------------------------------------------------


class TestTurnTypeUnion:
    @pytest.mark.parametrize(
        "turn",
        [
            UserTurn(content="hi"),
            AssistantTurn(content="hello"),
            ToolResultsTurn(),
            SystemTurn(content="system"),
            SteeringTurn(content="steer"),
        ],
        ids=["user", "assistant", "tool_results", "system", "steering"],
    )
    def test_each_turn_type_is_valid_turn(self, turn: Turn):
        assert isinstance(turn, (UserTurn, AssistantTurn, ToolResultsTurn, SystemTurn, SteeringTurn))
