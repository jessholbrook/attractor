"""Node handlers for the Attractor pipeline engine."""

from attractor.handlers.base import Handler
from attractor.handlers.codergen import CodergenBackend, CodergenHandler, StubBackend
from attractor.handlers.conditional import ConditionalHandler
from attractor.handlers.exit_handler import ExitHandler
from attractor.handlers.fan_in import FanInHandler
from attractor.handlers.parallel import ParallelHandler
from attractor.handlers.stack_manager import StackManagerHandler
from attractor.handlers.start import StartHandler
from attractor.handlers.tool import ToolHandler
from attractor.handlers.wait_human import Interviewer, WaitHumanHandler

__all__ = [
    "Handler",
    "StartHandler",
    "ExitHandler",
    "ConditionalHandler",
    "CodergenHandler",
    "CodergenBackend",
    "StubBackend",
    "WaitHumanHandler",
    "Interviewer",
    "ParallelHandler",
    "FanInHandler",
    "ToolHandler",
    "StackManagerHandler",
    "create_default_registry",
]


def create_default_registry(
    *,
    codergen_backend: CodergenBackend | None = None,
    interviewer: Interviewer | None = None,
    tool_registry: dict | None = None,
) -> "HandlerRegistry":
    """Create a HandlerRegistry with all default handlers registered.

    Args:
        codergen_backend: Backend for code generation. Uses StubBackend if None.
        interviewer: Interviewer for human interaction nodes. Uses a no-op if None.
        tool_registry: Dict mapping tool names to callables. Empty dict if None.

    Returns:
        A fully configured HandlerRegistry.
    """
    from attractor.engine.engine import HandlerRegistry

    registry = HandlerRegistry()

    # Start / Exit
    registry.register("start", StartHandler())
    registry.register("exit", ExitHandler())

    # Conditional
    registry.register("conditional", ConditionalHandler())

    # Codergen
    backend = codergen_backend or StubBackend()
    registry.register("codergen", CodergenHandler(backend))

    # Wait human
    if interviewer is not None:
        registry.register("wait.human", WaitHumanHandler(interviewer))

    # Parallel
    registry.register("parallel", ParallelHandler(registry))

    # Fan-in
    registry.register("parallel.fan_in", FanInHandler())

    # Tool
    tools = tool_registry or {}
    registry.register("tool", ToolHandler(tools))

    # Stack manager loop
    registry.register("stack.manager_loop", StackManagerHandler(registry))

    return registry
