from __future__ import annotations

from unified_llm import (
    Client,
    FinishReason,
    FinishReasonInfo,
    Message,
    Response,
    StubAdapter,
    Usage,
)

from wolverine.pipeline.backend import StubWolverineBackend, WolverineBackend


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def make_stub_client(text: str) -> Client:
    """Create a Client with a StubAdapter that returns the given text."""
    response = Response(
        message=Message.assistant(text),
        finish_reason=FinishReasonInfo(reason=FinishReason.STOP),
        usage=Usage(input_tokens=10, output_tokens=10),
    )
    adapter = StubAdapter(responses=[response])
    return Client(providers={"stub": adapter}, default_provider="stub")


# ---------------------------------------------------------------------------
# StubWolverineBackend
# ---------------------------------------------------------------------------


class TestStubWolverineBackend:
    def test_returns_default_response(self) -> None:
        backend = StubWolverineBackend()
        result = backend.generate("hello", {})
        assert result == "Stub response"

    def test_returns_canned_response_matching_key(self) -> None:
        backend = StubWolverineBackend(responses={"classify": "classified!"})
        result = backend.generate("Please classify this signal", {})
        assert result == "classified!"

    def test_key_matching_is_case_insensitive(self) -> None:
        backend = StubWolverineBackend(responses={"CLASSIFY": "found"})
        result = backend.generate("classify this", {})
        assert result == "found"

    def test_tracks_calls(self) -> None:
        backend = StubWolverineBackend()
        backend.generate("first prompt", {})
        backend.generate("second prompt", {})
        assert backend._calls == ["first prompt", "second prompt"]

    def test_returns_default_when_no_key_matches(self) -> None:
        backend = StubWolverineBackend(responses={"diagnose": "diagnosed"})
        result = backend.generate("something else", {})
        assert result == "Stub response"


# ---------------------------------------------------------------------------
# WolverineBackend
# ---------------------------------------------------------------------------


class TestWolverineBackend:
    def test_constructs_with_defaults(self) -> None:
        client = make_stub_client("hello")
        backend = WolverineBackend(client)
        assert backend._default_model == "claude-sonnet-4-20250514"

    def test_constructs_with_custom_model(self) -> None:
        client = make_stub_client("hello")
        backend = WolverineBackend(client, default_model="gpt-4o")
        assert backend._default_model == "gpt-4o"

    def test_generate_returns_text(self) -> None:
        client = make_stub_client("LLM response here")
        backend = WolverineBackend(client)
        result = backend.generate("test prompt", {})
        assert result == "LLM response here"

    def test_generate_calls_unified_llm(self) -> None:
        client = make_stub_client("response")
        backend = WolverineBackend(client)
        backend.generate("my prompt", {"key": "value"})
        # Verify the StubAdapter received the request
        adapter = client.providers["stub"]
        assert len(adapter.requests) == 1
        request = adapter.requests[0]
        assert request.model == "claude-sonnet-4-20250514"

    def test_generate_uses_custom_model_when_provided(self) -> None:
        client = make_stub_client("response")
        backend = WolverineBackend(client)
        backend.generate("prompt", {}, model="custom-model")
        adapter = client.providers["stub"]
        request = adapter.requests[0]
        assert request.model == "custom-model"

    def test_generate_uses_default_model_when_model_is_none(self) -> None:
        client = make_stub_client("response")
        backend = WolverineBackend(client, default_model="my-default")
        backend.generate("prompt", {}, model=None)
        adapter = client.providers["stub"]
        request = adapter.requests[0]
        assert request.model == "my-default"

    def test_generate_sets_temperature_zero(self) -> None:
        client = make_stub_client("response")
        backend = WolverineBackend(client)
        backend.generate("prompt", {})
        adapter = client.providers["stub"]
        request = adapter.requests[0]
        assert request.temperature == 0.0

    def test_multiple_calls_work(self) -> None:
        client = make_stub_client("response")
        backend = WolverineBackend(client)
        r1 = backend.generate("first", {})
        r2 = backend.generate("second", {})
        assert r1 == "response"
        assert r2 == "response"
        adapter = client.providers["stub"]
        assert len(adapter.requests) == 2
