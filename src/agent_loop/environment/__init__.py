"""Execution environment abstraction and implementations."""

from agent_loop.environment.local import EnvVarPolicy, LocalExecutionEnvironment
from agent_loop.environment.stub import StubExecutionEnvironment
from agent_loop.environment.types import DirEntry, ExecResult, ExecutionEnvironment, GrepOptions

__all__ = [
    "DirEntry",
    "EnvVarPolicy",
    "ExecResult",
    "ExecutionEnvironment",
    "GrepOptions",
    "LocalExecutionEnvironment",
    "StubExecutionEnvironment",
]
