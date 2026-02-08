"""Core client with provider routing and middleware."""
from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any, Callable

from unified_llm.adapter import ProviderAdapter
from unified_llm.errors import ConfigurationError
from unified_llm.types.request import Request
from unified_llm.types.response import Response
from unified_llm.types.streaming import StreamEvent

Middleware = Callable[[Request, Callable[[Request], Response]], Response]
StreamMiddleware = Callable[
    [Request, Callable[[Request], Iterator[StreamEvent]]], Iterator[StreamEvent]
]


class Client:
    """Provider-agnostic LLM client with middleware support."""

    def __init__(
        self,
        providers: dict[str, ProviderAdapter] | None = None,
        default_provider: str | None = None,
        middleware: list[Middleware] | None = None,
    ) -> None:
        self._providers: dict[str, ProviderAdapter] = dict(providers) if providers else {}
        self._default_provider = default_provider
        self._middleware = list(middleware) if middleware else []

    @classmethod
    def from_env(
        cls,
        *,
        middleware: list[Middleware] | None = None,
    ) -> Client:
        """Create client from environment variables.

        Checks for ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY.
        Registers adapters for keys that are found.
        First found becomes default_provider.
        """
        providers: dict[str, ProviderAdapter] = {}
        default: str | None = None

        # Lazy import providers to avoid import errors if httpx not configured
        if os.environ.get("ANTHROPIC_API_KEY"):
            from unified_llm.providers.anthropic import AnthropicAdapter

            providers["anthropic"] = AnthropicAdapter(
                api_key=os.environ["ANTHROPIC_API_KEY"],
                base_url=os.environ.get(
                    "ANTHROPIC_BASE_URL", "https://api.anthropic.com"
                ),
            )
            if default is None:
                default = "anthropic"

        if os.environ.get("OPENAI_API_KEY"):
            from unified_llm.providers.openai import OpenAIAdapter

            providers["openai"] = OpenAIAdapter(
                api_key=os.environ["OPENAI_API_KEY"],
                base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com"),
            )
            if default is None:
                default = "openai"

        if os.environ.get("GEMINI_API_KEY"):
            from unified_llm.providers.gemini import GeminiAdapter

            providers["gemini"] = GeminiAdapter(
                api_key=os.environ["GEMINI_API_KEY"],
                base_url=os.environ.get(
                    "GEMINI_BASE_URL",
                    "https://generativelanguage.googleapis.com",
                ),
            )
            if default is None:
                default = "gemini"

        return cls(providers=providers, default_provider=default, middleware=middleware)

    def register_provider(self, name: str, adapter: ProviderAdapter) -> None:
        """Register a provider adapter. Sets as default if no default exists."""
        self._providers[name] = adapter
        if self._default_provider is None:
            self._default_provider = name

    def complete(self, request: Request) -> Response:
        """Send a completion request through the middleware chain."""
        adapter = self._resolve_provider(request)

        def handler(req: Request) -> Response:
            return adapter.complete(req)

        chain = handler
        for mw in reversed(self._middleware):
            prev_chain = chain
            chain = lambda req, _prev=prev_chain, _mw=mw: _mw(req, _prev)

        return chain(request)

    def stream(self, request: Request) -> Iterator[StreamEvent]:
        """Send a streaming request (bypasses middleware)."""
        adapter = self._resolve_provider(request)
        return adapter.stream(request)

    def close(self) -> None:
        """Close all provider adapters."""
        for adapter in self._providers.values():
            if hasattr(adapter, "close"):
                adapter.close()

    @property
    def providers(self) -> dict[str, ProviderAdapter]:
        """Return a copy of the registered providers."""
        return dict(self._providers)

    @property
    def default_provider(self) -> str | None:
        """Return the name of the default provider."""
        return self._default_provider

    def _resolve_provider(self, request: Request) -> ProviderAdapter:
        """Resolve the provider adapter from the request or default."""
        provider_name = request.provider or self._default_provider
        if provider_name is None:
            raise ConfigurationError(
                "No provider specified and no default provider set"
            )
        adapter = self._providers.get(provider_name)
        if adapter is None:
            raise ConfigurationError(f"Unknown provider: {provider_name}")
        return adapter


# ---------------------------------------------------------------------------
# Module-level default client
# ---------------------------------------------------------------------------

_default_client: Client | None = None


def set_default_client(client: Client) -> None:
    """Set the module-level default client."""
    global _default_client
    _default_client = client


def get_default_client() -> Client:
    """Get the module-level default client, creating from env if needed."""
    global _default_client
    if _default_client is None:
        _default_client = Client.from_env()
    return _default_client
