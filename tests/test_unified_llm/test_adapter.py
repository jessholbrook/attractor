"""Tests for the adapter protocol and implementations."""
from __future__ import annotations

from collections.abc import Iterator

import pytest

from unified_llm.adapter import BaseAdapter, ProviderAdapter, StubAdapter
from unified_llm.types.enums import FinishReason, Role, StreamEventType
from unified_llm.types.request import Request
from unified_llm.types.response import FinishReasonInfo, Response, Usage
from unified_llm.types.streaming import StreamEvent
from unified_llm.types.messages import Message


# ---------------------------------------------------------------------------
# ProviderAdapter Protocol
# ---------------------------------------------------------------------------


class _MinimalAdapter:
    """Minimal class that satisfies ProviderAdapter."""

    @property
    def name(self) -> str:
        return "minimal"

    def complete(self, request: Request) -> Response:
        return Response()

    def stream(self, request: Request) -> Iterator[StreamEvent]:
        yield StreamEvent(type=StreamEventType.FINISH)


def test_minimal_adapter_satisfies_protocol() -> None:
    adapter = _MinimalAdapter()
    assert isinstance(adapter, ProviderAdapter)


def test_stub_adapter_satisfies_protocol() -> None:
    adapter = StubAdapter()
    assert isinstance(adapter, ProviderAdapter)


def test_base_adapter_satisfies_protocol() -> None:
    adapter = BaseAdapter()
    assert isinstance(adapter, ProviderAdapter)


def test_plain_object_does_not_satisfy_protocol() -> None:
    assert not isinstance(object(), ProviderAdapter)


# ---------------------------------------------------------------------------
# BaseAdapter defaults
# ---------------------------------------------------------------------------


def test_base_adapter_name_raises() -> None:
    adapter = BaseAdapter()
    with pytest.raises(NotImplementedError):
        _ = adapter.name


def test_base_adapter_complete_raises() -> None:
    adapter = BaseAdapter()
    with pytest.raises(NotImplementedError):
        adapter.complete(Request(model="test"))


def test_base_adapter_stream_raises() -> None:
    adapter = BaseAdapter()
    with pytest.raises(NotImplementedError):
        adapter.stream(Request(model="test"))


def test_base_adapter_close_noop() -> None:
    adapter = BaseAdapter()
    adapter.close()  # Should not raise


def test_base_adapter_initialize_noop() -> None:
    adapter = BaseAdapter()
    adapter.initialize()  # Should not raise


def test_base_adapter_supports_tool_choice_default() -> None:
    adapter = BaseAdapter()
    assert adapter.supports_tool_choice("auto") is True
    assert adapter.supports_tool_choice("required") is True
    assert adapter.supports_tool_choice("none") is True


# ---------------------------------------------------------------------------
# StubAdapter
# ---------------------------------------------------------------------------


def test_stub_adapter_default_name() -> None:
    adapter = StubAdapter()
    assert adapter.name == "stub"


def test_stub_adapter_custom_name() -> None:
    adapter = StubAdapter(name="test-provider")
    assert adapter.name == "test-provider"


def test_stub_adapter_complete_returns_responses() -> None:
    r1 = Response(id="r1", model="m1")
    r2 = Response(id="r2", model="m2")
    adapter = StubAdapter(responses=[r1, r2])

    req = Request(model="test")
    assert adapter.complete(req).id == "r1"
    assert adapter.complete(req).id == "r2"
    # Cycles back
    assert adapter.complete(req).id == "r1"


def test_stub_adapter_complete_empty_returns_default() -> None:
    adapter = StubAdapter()
    resp = adapter.complete(Request(model="test"))
    assert isinstance(resp, Response)


def test_stub_adapter_stream_yields_events() -> None:
    events = [
        StreamEvent(type=StreamEventType.TEXT_DELTA, delta="hello"),
        StreamEvent(type=StreamEventType.FINISH),
    ]
    adapter = StubAdapter(stream_events=[events])

    result = list(adapter.stream(Request(model="test")))
    assert len(result) == 2
    assert result[0].delta == "hello"


def test_stub_adapter_stream_empty_yields_nothing() -> None:
    adapter = StubAdapter()
    result = list(adapter.stream(Request(model="test")))
    assert result == []


def test_stub_adapter_tracks_requests() -> None:
    adapter = StubAdapter()
    req1 = Request(model="m1")
    req2 = Request(model="m2")
    adapter.complete(req1)
    list(adapter.stream(req2))  # consume the iterator
    assert len(adapter.requests) == 2
    assert adapter.requests[0].model == "m1"
    assert adapter.requests[1].model == "m2"
