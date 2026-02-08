"""Agent Loop: a language-agnostic coding agent loop library."""

from agent_loop.client import Client, CompletionRequest, CompletionResponse, Message, StubClient
from agent_loop.environment.types import ExecResult, ExecutionEnvironment, GrepOptions
from agent_loop.events import EventEmitter
from agent_loop.providers.profile import ProviderProfile
from agent_loop.session import Session
from agent_loop.session_config import SessionConfig, SessionState
from agent_loop.tools.registry import RegisteredTool, ToolDefinition, ToolRegistry
from agent_loop.turns import (
    AssistantTurn,
    SteeringTurn,
    ToolCall,
    ToolResult,
    ToolResultsTurn,
    Turn,
    UserTurn,
)

__all__ = [
    # Core orchestrator
    "Session",
    "SessionConfig",
    "SessionState",
    # LLM client
    "Client",
    "CompletionRequest",
    "CompletionResponse",
    "Message",
    "StubClient",
    # Tools
    "ToolDefinition",
    "RegisteredTool",
    "ToolRegistry",
    # Environment
    "ExecutionEnvironment",
    "ExecResult",
    "GrepOptions",
    # Events
    "EventEmitter",
    # Providers
    "ProviderProfile",
    # Turn types
    "Turn",
    "UserTurn",
    "AssistantTurn",
    "ToolCall",
    "ToolResult",
    "ToolResultsTurn",
    "SteeringTurn",
]
