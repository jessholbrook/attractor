"""HTTP client wrapper around httpx."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import httpx

from unified_llm.errors import NetworkError, RequestTimeoutError, error_from_status_code
from unified_llm.types.config import AdapterTimeout


@dataclass(frozen=True)
class HttpResponse:
    """Parsed HTTP response."""

    status_code: int
    body: dict[str, Any]
    headers: dict[str, str]
    raw_text: str = ""


class HttpClient:
    """Thin wrapper around :mod:`httpx` that maps errors into unified_llm exceptions."""

    def __init__(
        self,
        base_url: str,
        headers: dict[str, str],
        timeout: AdapterTimeout | None = None,
    ) -> None:
        t = timeout or AdapterTimeout()
        self._client = httpx.Client(
            base_url=base_url,
            headers=headers,
            timeout=httpx.Timeout(
                connect=t.connect,
                read=t.request,
                write=t.request,
                pool=t.connect,
            ),
        )

    def post(
        self,
        path: str,
        json: dict[str, Any],
        extra_headers: dict[str, str] | None = None,
    ) -> HttpResponse:
        """Send a POST request and return the parsed response.

        Raises a unified_llm error on non-2xx status or transport failure.
        """
        try:
            resp = self._client.post(path, json=json, headers=extra_headers or {})
        except httpx.TimeoutException as exc:
            raise RequestTimeoutError(str(exc), cause=exc) from exc
        except httpx.ConnectError as exc:
            raise NetworkError(str(exc), cause=exc) from exc

        raw_text = resp.text
        hdrs = dict(resp.headers)

        if resp.status_code >= 300:
            try:
                body = resp.json()
            except Exception:
                body = {}
            msg = body.get("error", {}).get("message", raw_text) if isinstance(body.get("error"), dict) else raw_text
            raise error_from_status_code(resp.status_code, msg, raw=body)

        try:
            body = resp.json()
        except Exception:
            body = {}

        return HttpResponse(
            status_code=resp.status_code,
            body=body,
            headers=hdrs,
            raw_text=raw_text,
        )

    def post_stream(
        self,
        path: str,
        json: dict[str, Any],
        extra_headers: dict[str, str] | None = None,
    ) -> Iterator[str]:
        """Send a streaming POST request and yield raw text lines."""
        try:
            with self._client.stream(
                "POST", path, json=json, headers=extra_headers or {}
            ) as resp:
                if resp.status_code >= 300:
                    raw_text = resp.read().decode("utf-8", errors="replace")
                    try:
                        body = resp.json()
                    except Exception:
                        body = {}
                    msg = (
                        body.get("error", {}).get("message", raw_text)
                        if isinstance(body.get("error"), dict)
                        else raw_text
                    )
                    raise error_from_status_code(resp.status_code, msg, raw=body)

                for line in resp.iter_lines():
                    yield line
        except httpx.TimeoutException as exc:
            raise RequestTimeoutError(str(exc), cause=exc) from exc
        except httpx.ConnectError as exc:
            raise NetworkError(str(exc), cause=exc) from exc

    def close(self) -> None:
        """Close the underlying httpx client."""
        self._client.close()
