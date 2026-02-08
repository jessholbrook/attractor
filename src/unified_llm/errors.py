"""Error hierarchy for the unified LLM client."""
from __future__ import annotations

from typing import Any


class SDKError(Exception):
    """Base error for all unified_llm errors."""

    def __init__(self, message: str, *, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.cause = cause


class ProviderError(SDKError):
    """Error originating from a provider API."""

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        status_code: int | None = None,
        error_code: str | None = None,
        retryable: bool = False,
        retry_after: float | None = None,
        raw: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message, cause=cause)
        self.provider = provider
        self.status_code = status_code
        self.error_code = error_code
        self.retryable = retryable
        self.retry_after = retry_after
        self.raw = raw


# ---------------------------------------------------------------------------
# Specific provider errors
# ---------------------------------------------------------------------------


class AuthenticationError(ProviderError):
    """Authentication failed (e.g. invalid API key)."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("retryable", False)
        super().__init__(message, **kwargs)


class AccessDeniedError(ProviderError):
    """Access denied (e.g. insufficient permissions)."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("retryable", False)
        super().__init__(message, **kwargs)


class NotFoundError(ProviderError):
    """Resource not found (e.g. unknown model)."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("retryable", False)
        super().__init__(message, **kwargs)


class InvalidRequestError(ProviderError):
    """The request was malformed or invalid."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("retryable", False)
        super().__init__(message, **kwargs)


class RateLimitError(ProviderError):
    """Rate limit exceeded."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("retryable", True)
        super().__init__(message, **kwargs)


class ServerError(ProviderError):
    """Server-side error from the provider."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("retryable", True)
        super().__init__(message, **kwargs)


class ContentFilterError(ProviderError):
    """Content was blocked by a safety filter."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("retryable", False)
        super().__init__(message, **kwargs)


class ContextLengthError(ProviderError):
    """Input or output exceeded the model's context window."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("retryable", False)
        super().__init__(message, **kwargs)


class QuotaExceededError(ProviderError):
    """Account quota has been exceeded."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("retryable", False)
        super().__init__(message, **kwargs)


# ---------------------------------------------------------------------------
# Non-provider errors
# ---------------------------------------------------------------------------


class RequestTimeoutError(SDKError):
    """A request timed out."""


class AbortError(SDKError):
    """The operation was aborted by the caller."""


class NetworkError(SDKError):
    """A network-level error occurred."""


class StreamError(SDKError):
    """An error occurred while processing a stream."""


class InvalidToolCallError(SDKError):
    """A tool call from the model was invalid."""


class NoObjectGeneratedError(SDKError):
    """The model did not produce the expected structured object."""


class ConfigurationError(SDKError):
    """Invalid SDK configuration."""


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def error_from_status_code(
    status_code: int,
    message: str,
    *,
    provider: str = "",
    error_code: str | None = None,
    raw: dict[str, Any] | None = None,
    retry_after: float | None = None,
) -> ProviderError:
    """Map HTTP status code to the appropriate error type."""
    common = dict(
        provider=provider,
        status_code=status_code,
        error_code=error_code,
        raw=raw,
        retry_after=retry_after,
    )

    if status_code in (400, 422):
        return InvalidRequestError(message, **common)
    if status_code == 401:
        return AuthenticationError(message, **common)
    if status_code == 403:
        return AccessDeniedError(message, **common)
    if status_code == 404:
        return NotFoundError(message, **common)
    if status_code == 408:
        return ProviderError(message, retryable=True, **common)
    if status_code == 413:
        return ContextLengthError(message, **common)
    if status_code == 429:
        return RateLimitError(message, **common)
    if 500 <= status_code <= 599:
        return ServerError(message, **common)

    # Unknown status codes are retryable by default
    return ProviderError(message, retryable=True, **common)
