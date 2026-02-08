"""CodergenHandler: delegates code/text generation to a pluggable backend."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from attractor.model.context import Context
from attractor.model.graph import Graph, Node
from attractor.model.outcome import Outcome, Status


class CodergenBackend(Protocol):
    """Protocol for code generation backends."""

    def generate(
        self,
        prompt: str,
        context: dict,
        *,
        model: str = "",
        fidelity: str = "",
        reasoning_effort: str = "high",
    ) -> str: ...


class StubBackend:
    """A stub backend that returns a predictable response for testing."""

    def generate(
        self,
        prompt: str,
        context: dict,
        *,
        model: str = "",
        fidelity: str = "",
        reasoning_effort: str = "high",
    ) -> str:
        return f"stub response: {prompt[:50]}"


class CodergenHandler:
    """Handler for codergen (box) nodes.

    Calls the backend's generate() method with the node's prompt (or label),
    a snapshot of the context, and optional model/fidelity/reasoning_effort.
    Stores the response in context_updates under '{node.id}.response'.
    Returns FAIL if the backend raises an exception.
    """

    def __init__(self, backend: CodergenBackend) -> None:
        self._backend = backend

    def execute(self, node: Node, context: Context, graph: Graph, logs_root: Path) -> Outcome:
        prompt = node.prompt or node.label
        snapshot = context.snapshot()

        try:
            response = self._backend.generate(
                prompt,
                snapshot,
                model=node.llm_model,
                fidelity=node.fidelity,
                reasoning_effort=node.reasoning_effort,
            )
        except Exception as exc:
            return Outcome(
                status=Status.FAIL,
                failure_reason=f"CodergenBackend error: {exc}",
            )

        return Outcome(
            status=Status.SUCCESS,
            context_updates={f"{node.id}.response": response},
        )
