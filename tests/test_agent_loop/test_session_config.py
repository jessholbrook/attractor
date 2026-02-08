"""Tests for session configuration and state."""

import dataclasses

import pytest

from agent_loop.session_config import SessionConfig, SessionState


class TestSessionState:
    def test_all_states_exist(self):
        assert len(SessionState) == 4

    def test_state_values(self):
        assert SessionState.IDLE.value == "idle"
        assert SessionState.PROCESSING.value == "processing"
        assert SessionState.AWAITING_INPUT.value == "awaiting_input"
        assert SessionState.CLOSED.value == "closed"


class TestSessionConfig:
    def test_defaults(self):
        c = SessionConfig()
        assert c.max_turns == 0
        assert c.max_tool_rounds_per_input == 200
        assert c.default_command_timeout_ms == 10_000
        assert c.max_command_timeout_ms == 600_000
        assert c.reasoning_effort is None
        assert c.tool_output_limits == {}
        assert c.enable_loop_detection is True
        assert c.loop_detection_window == 10
        assert c.max_subagent_depth == 1

    def test_custom_values(self):
        c = SessionConfig(max_turns=50, reasoning_effort="high", max_subagent_depth=0)
        assert c.max_turns == 50
        assert c.reasoning_effort == "high"
        assert c.max_subagent_depth == 0

    def test_frozen_immutability(self):
        c = SessionConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            c.max_turns = 10  # type: ignore[misc]

    def test_max_turns_zero_means_unlimited(self):
        c = SessionConfig()
        assert c.max_turns == 0  # convention: 0 = unlimited
