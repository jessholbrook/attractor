from __future__ import annotations

from unified_llm import Client as UnifiedClient
from unified_llm import generate


class StubWolverineBackend:
    """Stub backend that returns canned responses for testing."""

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        self._responses = responses or {}
        self._calls: list[str] = []

    def generate(
        self,
        prompt: str,
        context: dict,
        *,
        model: str = "",
        fidelity: str = "",
        reasoning_effort: str = "high",
    ) -> str:
        self._calls.append(prompt)
        for key, response in self._responses.items():
            if key.lower() in prompt.lower():
                return response
        return "Stub response"


class WolverineBackend:
    """CodergenBackend that delegates to unified_llm for real LLM calls."""

    def __init__(
        self,
        client: UnifiedClient,
        default_model: str = "claude-sonnet-4-20250514",
    ) -> None:
        self._client = client
        self._default_model = default_model

    def generate(
        self,
        prompt: str,
        context: dict,
        *,
        model: str | None = None,
        fidelity: str | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        """Generate a response using unified_llm."""
        model = model or self._default_model
        result = generate(
            model,
            prompt=prompt,
            client=self._client,
            temperature=0.0,
        )
        return result.text
