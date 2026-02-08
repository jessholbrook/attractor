"""Tests for the HTTP client wrapper."""
from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from unified_llm._http import HttpClient, HttpResponse
from unified_llm.errors import (
    NetworkError,
    ProviderError,
    RequestTimeoutError,
)
from unified_llm.types.config import AdapterTimeout


# ---------------------------------------------------------------------------
# HttpResponse dataclass
# ---------------------------------------------------------------------------


def test_http_response_construction() -> None:
    resp = HttpResponse(status_code=200, body={"ok": True}, headers={"x-id": "1"})
    assert resp.status_code == 200
    assert resp.body == {"ok": True}
    assert resp.headers == {"x-id": "1"}
    assert resp.raw_text == ""


def test_http_response_with_raw_text() -> None:
    resp = HttpResponse(
        status_code=200, body={}, headers={}, raw_text='{"ok":true}'
    )
    assert resp.raw_text == '{"ok":true}'


def test_http_response_is_frozen() -> None:
    resp = HttpResponse(status_code=200, body={}, headers={})
    with pytest.raises(AttributeError):
        resp.status_code = 400  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_transport(
    status_code: int = 200,
    json_body: dict[str, Any] | None = None,
    text: str | None = None,
) -> httpx.MockTransport:
    """Create a mock transport that returns a fixed response."""
    body = json.dumps(json_body or {}).encode() if text is None else text.encode()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=status_code,
            content=body,
            headers={"content-type": "application/json"},
        )

    return httpx.MockTransport(handler)


def _client_with_transport(transport: httpx.MockTransport) -> HttpClient:
    """Create an HttpClient and inject a mock transport."""
    client = HttpClient(base_url="https://api.test.com", headers={"Authorization": "Bearer test"})
    # Replace the internal httpx client with one using our transport
    client._client = httpx.Client(
        base_url="https://api.test.com",
        headers={"Authorization": "Bearer test"},
        transport=transport,
    )
    return client


# ---------------------------------------------------------------------------
# POST success
# ---------------------------------------------------------------------------


def test_post_success() -> None:
    transport = _make_transport(200, {"result": "ok"})
    client = _client_with_transport(transport)
    resp = client.post("/v1/chat", json={"model": "test"})
    assert resp.status_code == 200
    assert resp.body == {"result": "ok"}
    client.close()


def test_post_with_extra_headers() -> None:
    captured_headers: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_headers.update(dict(request.headers))
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(handler)
    client = _client_with_transport(transport)
    client.post("/v1/chat", json={}, extra_headers={"X-Custom": "value"})
    assert captured_headers.get("x-custom") == "value"
    client.close()


# ---------------------------------------------------------------------------
# POST error status codes
# ---------------------------------------------------------------------------


def test_post_error_status() -> None:
    transport = _make_transport(500, {"error": {"message": "Internal Server Error"}})
    client = _client_with_transport(transport)
    with pytest.raises(ProviderError):
        client.post("/v1/chat", json={})
    client.close()


def test_post_404_raises_provider_error() -> None:
    transport = _make_transport(404, {"error": {"message": "Not Found"}})
    client = _client_with_transport(transport)
    with pytest.raises(ProviderError):
        client.post("/v1/chat", json={})
    client.close()


# ---------------------------------------------------------------------------
# Timeout and network errors
# ---------------------------------------------------------------------------


def test_timeout_raises_request_timeout_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("Connection timed out")

    transport = httpx.MockTransport(handler)
    client = _client_with_transport(transport)
    with pytest.raises(RequestTimeoutError):
        client.post("/v1/chat", json={})
    client.close()


def test_connect_error_raises_network_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    transport = httpx.MockTransport(handler)
    client = _client_with_transport(transport)
    with pytest.raises(NetworkError):
        client.post("/v1/chat", json={})
    client.close()


# ---------------------------------------------------------------------------
# Timeout config
# ---------------------------------------------------------------------------


def test_timeout_configuration() -> None:
    timeout = AdapterTimeout(connect=10.0, request=120.0, stream_read=60.0)
    client = HttpClient(
        base_url="https://api.test.com",
        headers={},
        timeout=timeout,
    )
    assert client._client.timeout.connect == 10.0
    assert client._client.timeout.read == 120.0
    client.close()


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def test_post_stream_yields_lines() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = b"data: hello\n\ndata: world\n\n"
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    transport = httpx.MockTransport(handler)
    client = _client_with_transport(transport)
    lines = list(client.post_stream("/v1/chat", json={}))
    assert len(lines) > 0
    client.close()


def test_post_stream_error_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            content=b'{"error": {"message": "fail"}}',
            headers={"content-type": "application/json"},
        )

    transport = httpx.MockTransport(handler)
    client = _client_with_transport(transport)
    with pytest.raises(ProviderError):
        list(client.post_stream("/v1/chat", json={}))
    client.close()
