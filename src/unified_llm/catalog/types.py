"""Model catalog types."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelInfo:
    """Static metadata about a model."""

    id: str
    """API identifier (e.g., "claude-opus-4-6")."""

    provider: str
    """Provider key: "openai", "anthropic", or "gemini"."""

    display_name: str
    """Human-readable name."""

    context_window: int
    """Max total tokens."""

    max_output: int | None = None
    """Max output tokens."""

    supports_tools: bool = True
    """Whether the model supports tool/function calling."""

    supports_vision: bool = False
    """Whether the model supports image inputs."""

    supports_reasoning: bool = False
    """Whether the model supports extended reasoning."""

    input_cost_per_million: float | None = None
    """USD per 1M input tokens."""

    output_cost_per_million: float | None = None
    """USD per 1M output tokens."""

    aliases: tuple[str, ...] = ()
    """Shorthand names that resolve to this model."""
