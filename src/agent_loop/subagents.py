"""Subagent types and tool factories."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_loop.session_config import SessionConfig
from agent_loop.tools.registry import RegisteredTool, ToolDefinition


@dataclass(frozen=True)
class SubAgentResult:
    """Result from a completed subagent."""

    output: str
    success: bool
    turns_used: int


@dataclass
class SubAgentHandle:
    """Handle to a running or completed subagent."""

    id: str
    session: Any  # Session (avoid circular import)
    status: str = "running"  # "running", "completed", "failed"
    result: SubAgentResult | None = None


# --- Tool definitions ---

SPAWN_AGENT_DEFINITION = ToolDefinition(
    name="spawn_agent",
    description="Spawn a subagent to handle a scoped task autonomously.",
    parameters={
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "Natural language task description"},
            "working_dir": {"type": "string", "description": "Subdirectory to scope the agent to"},
            "model": {"type": "string", "description": "Model override"},
            "max_turns": {"type": "integer", "description": "Turn limit (default: 50)"},
        },
        "required": ["task"],
    },
)

SEND_INPUT_DEFINITION = ToolDefinition(
    name="send_input",
    description="Send a message to a running subagent.",
    parameters={
        "type": "object",
        "properties": {
            "agent_id": {"type": "string", "description": "Subagent ID"},
            "message": {"type": "string", "description": "Message to send"},
        },
        "required": ["agent_id", "message"],
    },
)

WAIT_DEFINITION = ToolDefinition(
    name="wait",
    description="Wait for a subagent to complete and return its result.",
    parameters={
        "type": "object",
        "properties": {
            "agent_id": {"type": "string", "description": "Subagent ID"},
        },
        "required": ["agent_id"],
    },
)

CLOSE_AGENT_DEFINITION = ToolDefinition(
    name="close_agent",
    description="Terminate a subagent.",
    parameters={
        "type": "object",
        "properties": {
            "agent_id": {"type": "string", "description": "Subagent ID"},
        },
        "required": ["agent_id"],
    },
)


def make_subagent_tools(parent_session: Any) -> list[RegisteredTool]:
    """Create subagent tool definitions + executors bound to a parent session.

    Returns empty list if depth >= max_subagent_depth (preventing infinite nesting).
    """
    from agent_loop.session import Session

    if parent_session.depth >= parent_session.config.max_subagent_depth:
        return []

    def spawn_executor(arguments: dict[str, Any], env: Any) -> str:
        task = arguments["task"]
        max_turns = arguments.get("max_turns", 50)

        child = Session(
            llm_client=parent_session.llm_client,
            provider_profile=parent_session.provider_profile,
            execution_env=parent_session.execution_env,
            config=SessionConfig(max_turns=max_turns),
            depth=parent_session.depth + 1,
        )

        handle = SubAgentHandle(id=child.id, session=child)
        parent_session._subagents[handle.id] = handle

        try:
            result_turn = child.process_input(task)
            handle.status = "completed"
            handle.result = SubAgentResult(
                output=result_turn.content,
                success=True,
                turns_used=len([t for t in child.history if hasattr(t, "tool_calls")]),
            )
            return f"Agent {handle.id} completed. Output:\n{result_turn.content}"
        except Exception as e:
            handle.status = "failed"
            handle.result = SubAgentResult(output=str(e), success=False, turns_used=0)
            return f"Agent {handle.id} failed: {e}"

    def send_input_executor(arguments: dict[str, Any], env: Any) -> str:
        agent_id = arguments["agent_id"]
        handle = parent_session._subagents.get(agent_id)
        if handle is None:
            return f"Unknown agent: {agent_id}"
        if handle.status != "running":
            return f"Agent {agent_id} is {handle.status}, cannot send input"
        result_turn = handle.session.process_input(arguments["message"])
        return f"Agent {agent_id} responded:\n{result_turn.content}"

    def wait_executor(arguments: dict[str, Any], env: Any) -> str:
        agent_id = arguments["agent_id"]
        handle = parent_session._subagents.get(agent_id)
        if handle is None:
            return f"Unknown agent: {agent_id}"
        if handle.result:
            return (
                f"Agent {agent_id} {handle.status}. "
                f"Output:\n{handle.result.output}\n"
                f"Turns used: {handle.result.turns_used}"
            )
        return f"Agent {agent_id} is still {handle.status}"

    def close_executor(arguments: dict[str, Any], env: Any) -> str:
        agent_id = arguments["agent_id"]
        handle = parent_session._subagents.get(agent_id)
        if handle is None:
            return f"Unknown agent: {agent_id}"
        handle.session.close()
        handle.status = "closed"
        return f"Agent {agent_id} closed"

    return [
        RegisteredTool(definition=SPAWN_AGENT_DEFINITION, executor=spawn_executor),
        RegisteredTool(definition=SEND_INPUT_DEFINITION, executor=send_input_executor),
        RegisteredTool(definition=WAIT_DEFINITION, executor=wait_executor),
        RegisteredTool(definition=CLOSE_AGENT_DEFINITION, executor=close_executor),
    ]
