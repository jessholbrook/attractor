"""Model catalog for looking up model capabilities and pricing."""
from __future__ import annotations

from unified_llm.catalog.types import ModelInfo
from unified_llm.catalog._data import MODELS


def get_model_info(model_id: str) -> ModelInfo | None:
    """Look up a model by exact ID or alias.

    First tries an exact match on ``ModelInfo.id``, then checks aliases.
    Returns ``None`` if no match is found.
    """
    for model in MODELS:
        if model.id == model_id:
            return model
    for model in MODELS:
        if model_id in model.aliases:
            return model
    return None


def list_models(provider: str | None = None) -> list[ModelInfo]:
    """Return models, optionally filtered by provider.

    Models are returned in definition order (flagship first per provider).
    """
    if provider is None:
        return list(MODELS)
    return [m for m in MODELS if m.provider == provider]


def get_latest_model(
    provider: str, capability: str | None = None
) -> ModelInfo | None:
    """Return the first (flagship/latest) model for a provider.

    Optionally filter by capability: ``"reasoning"``, ``"vision"``, or
    ``"tools"``.  Returns ``None`` if no model matches.
    """
    candidates = [m for m in MODELS if m.provider == provider]
    if capability == "reasoning":
        candidates = [m for m in candidates if m.supports_reasoning]
    elif capability == "vision":
        candidates = [m for m in candidates if m.supports_vision]
    elif capability == "tools":
        candidates = [m for m in candidates if m.supports_tools]
    return candidates[0] if candidates else None


__all__ = [
    "ModelInfo",
    "MODELS",
    "get_model_info",
    "list_models",
    "get_latest_model",
]
