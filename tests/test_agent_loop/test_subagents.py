"""Tests for subagent types and tool factories."""

import dataclasses

import pytest

from agent_loop.client import CompletionResponse, Message, StubClient
from agent_loop.environment.stub import StubExecutionEnvironment
from agent_loop.providers.profile import StubProfile
from agent_loop.session import Session
from agent_loop.session_config import SessionConfig
from agent_loop.subagents import (
    SubAgentHandle,
    SubAgentResult,
    make_subagent_tools,
)


# --- Data types ---


class TestSubAgentResult:
    def test_construction(self):
        r = SubAgentResult(output="done", success=True, turns_used=5)
        assert r.output == "done"
        assert r.success is True
        assert r.turns_used == 5

    def test_frozen_immutability(self):
        r = SubAgentResult(output="x", success=True, turns_used=1)
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.output = "y"  # type: ignore[misc]


class TestSubAgentHandle:
    def test_construction(self):
        h = SubAgentHandle(id="abc", session=None)
        assert h.id == "abc"
        assert h.status == "running"
        assert h.result is None


# --- make_subagent_tools ---


def _make_parent_session(depth: int = 0, max_depth: int = 1) -> Session:
    client = StubClient(responses=[
        CompletionResponse(message=Message.assistant("child response")),
    ])
    env = StubExecutionEnvironment()
    profile = StubProfile()
    return Session(
        llm_client=client,
        provider_profile=profile,
        execution_env=env,
        config=SessionConfig(max_subagent_depth=max_depth),
        depth=depth,
    )


class TestMakeSubagentTools:
    def test_returns_four_tools(self):
        parent = _make_parent_session()
        tools = make_subagent_tools(parent)
        assert len(tools) == 4

    def test_tool_names(self):
        parent = _make_parent_session()
        tools = make_subagent_tools(parent)
        names = {t.definition.name for t in tools}
        assert names == {"spawn_agent", "send_input", "wait", "close_agent"}

    def test_returns_empty_when_depth_exceeded(self):
        parent = _make_parent_session(depth=1, max_depth=1)
        tools = make_subagent_tools(parent)
        assert tools == []

    def test_returns_empty_when_depth_zero(self):
        parent = _make_parent_session(depth=0, max_depth=0)
        tools = make_subagent_tools(parent)
        assert tools == []


class TestSpawnAgentExecutor:
    def test_creates_and_runs_child(self):
        parent = _make_parent_session()
        tools = make_subagent_tools(parent)
        spawn = next(t for t in tools if t.definition.name == "spawn_agent")
        result = spawn.executor({"task": "write hello.py"}, None)
        assert "completed" in result.lower() or "child response" in result.lower()
        assert len(parent._subagents) == 1

    def test_child_has_incremented_depth(self):
        parent = _make_parent_session(depth=0, max_depth=2)
        tools = make_subagent_tools(parent)
        spawn = next(t for t in tools if t.definition.name == "spawn_agent")
        spawn.executor({"task": "test"}, None)
        child_handle = list(parent._subagents.values())[0]
        assert child_handle.session.depth == 1


class TestCloseAgentExecutor:
    def test_closes_child(self):
        parent = _make_parent_session()
        tools = make_subagent_tools(parent)
        spawn = next(t for t in tools if t.definition.name == "spawn_agent")
        close = next(t for t in tools if t.definition.name == "close_agent")

        spawn.executor({"task": "test"}, None)
        agent_id = list(parent._subagents.keys())[0]
        result = close.executor({"agent_id": agent_id}, None)
        assert "closed" in result.lower()


class TestWaitExecutor:
    def test_returns_completed_result(self):
        parent = _make_parent_session()
        tools = make_subagent_tools(parent)
        spawn = next(t for t in tools if t.definition.name == "spawn_agent")
        wait = next(t for t in tools if t.definition.name == "wait")

        spawn.executor({"task": "test"}, None)
        agent_id = list(parent._subagents.keys())[0]
        result = wait.executor({"agent_id": agent_id}, None)
        assert "completed" in result.lower()

    def test_unknown_agent(self):
        parent = _make_parent_session()
        tools = make_subagent_tools(parent)
        wait = next(t for t in tools if t.definition.name == "wait")
        result = wait.executor({"agent_id": "nonexistent"}, None)
        assert "Unknown" in result


class TestSubagentDepthLimiting:
    def test_child_cannot_spawn_at_max_depth(self):
        """A child at depth 1 with max_depth 1 cannot create subagent tools."""
        parent = _make_parent_session(depth=0, max_depth=1)
        # The child would be at depth 1
        child_client = StubClient()
        child = Session(
            llm_client=child_client,
            provider_profile=parent.provider_profile,
            execution_env=parent.execution_env,
            config=SessionConfig(max_subagent_depth=1),
            depth=1,
        )
        tools = make_subagent_tools(child)
        assert tools == []
