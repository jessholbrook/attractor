"""Tests for the model catalog."""
from __future__ import annotations

import dataclasses

import pytest

from unified_llm.catalog import get_model_info, get_latest_model, list_models, ModelInfo


# ---------------------------------------------------------------------------
# TestModelInfo
# ---------------------------------------------------------------------------


class TestModelInfo:
    def test_construction(self) -> None:
        info = ModelInfo(
            id="test-model",
            provider="test",
            display_name="Test Model",
            context_window=4096,
        )
        assert info.id == "test-model"
        assert info.provider == "test"
        assert info.display_name == "Test Model"
        assert info.context_window == 4096

    def test_defaults(self) -> None:
        info = ModelInfo(
            id="m", provider="p", display_name="M", context_window=1000
        )
        assert info.max_output is None
        assert info.supports_tools is True
        assert info.supports_vision is False
        assert info.supports_reasoning is False
        assert info.input_cost_per_million is None
        assert info.output_cost_per_million is None
        assert info.aliases == ()

    def test_frozen(self) -> None:
        info = ModelInfo(
            id="m", provider="p", display_name="M", context_window=1000
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            info.id = "other"  # type: ignore[misc]

    def test_aliases_are_tuple(self) -> None:
        info = ModelInfo(
            id="m",
            provider="p",
            display_name="M",
            context_window=1000,
            aliases=("a", "b"),
        )
        assert isinstance(info.aliases, tuple)
        assert info.aliases == ("a", "b")


# ---------------------------------------------------------------------------
# TestGetModelInfo
# ---------------------------------------------------------------------------


class TestGetModelInfo:
    def test_exact_id_match(self) -> None:
        result = get_model_info("claude-opus-4-6")
        assert result is not None
        assert result.id == "claude-opus-4-6"
        assert result.provider == "anthropic"

    def test_alias_match(self) -> None:
        result = get_model_info("opus")
        assert result is not None
        assert result.id == "claude-opus-4-6"

    def test_alias_match_sonnet(self) -> None:
        result = get_model_info("sonnet")
        assert result is not None
        assert result.id == "claude-sonnet-4-5-20250929"

    def test_alias_match_4o(self) -> None:
        result = get_model_info("4o")
        assert result is not None
        assert result.id == "gpt-4o"

    def test_alias_match_gemini_pro(self) -> None:
        result = get_model_info("gemini-pro")
        assert result is not None
        assert result.id == "gemini-2.5-pro"

    def test_unknown_returns_none(self) -> None:
        assert get_model_info("nonexistent-model") is None

    def test_case_sensitive(self) -> None:
        # IDs are case-sensitive; "Claude-Opus-4-6" should not match
        assert get_model_info("Claude-Opus-4-6") is None

    def test_exact_id_takes_priority_over_alias(self) -> None:
        # Ensure that if a model's ID is queried, it's returned even if
        # another model happens to have it as an alias (not the case in our
        # data, but the logic should prefer exact ID match).
        result = get_model_info("gpt-4o")
        assert result is not None
        assert result.id == "gpt-4o"


# ---------------------------------------------------------------------------
# TestListModels
# ---------------------------------------------------------------------------


class TestListModels:
    def test_all_models_non_empty(self) -> None:
        models = list_models()
        assert len(models) > 0

    def test_all_models_count(self) -> None:
        models = list_models()
        assert len(models) == 11

    def test_filter_anthropic(self) -> None:
        models = list_models(provider="anthropic")
        assert len(models) == 3
        assert all(m.provider == "anthropic" for m in models)

    def test_filter_openai(self) -> None:
        models = list_models(provider="openai")
        assert len(models) == 5
        assert all(m.provider == "openai" for m in models)

    def test_filter_gemini(self) -> None:
        models = list_models(provider="gemini")
        assert len(models) == 3
        assert all(m.provider == "gemini" for m in models)

    def test_unknown_provider_returns_empty(self) -> None:
        models = list_models(provider="unknown")
        assert models == []

    def test_returns_list(self) -> None:
        models = list_models()
        assert isinstance(models, list)


# ---------------------------------------------------------------------------
# TestGetLatestModel
# ---------------------------------------------------------------------------


class TestGetLatestModel:
    def test_anthropic_flagship(self) -> None:
        model = get_latest_model("anthropic")
        assert model is not None
        assert model.id == "claude-opus-4-6"

    def test_openai_flagship(self) -> None:
        model = get_latest_model("openai")
        assert model is not None
        assert model.id == "gpt-4o"

    def test_gemini_flagship(self) -> None:
        model = get_latest_model("gemini")
        assert model is not None
        assert model.id == "gemini-2.5-pro"

    def test_reasoning_capability_anthropic(self) -> None:
        model = get_latest_model("anthropic", capability="reasoning")
        assert model is not None
        assert model.supports_reasoning is True
        assert model.id == "claude-opus-4-6"

    def test_reasoning_capability_openai(self) -> None:
        model = get_latest_model("openai", capability="reasoning")
        assert model is not None
        assert model.supports_reasoning is True
        # First OpenAI model with reasoning is o3 (gpt-4o and gpt-4o-mini
        # don't support reasoning)
        assert model.id == "o3"

    def test_vision_capability(self) -> None:
        model = get_latest_model("gemini", capability="vision")
        assert model is not None
        assert model.supports_vision is True

    def test_tools_capability(self) -> None:
        model = get_latest_model("openai", capability="tools")
        assert model is not None
        assert model.supports_tools is True

    def test_unknown_provider_returns_none(self) -> None:
        assert get_latest_model("unknown") is None
