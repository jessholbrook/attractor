"""Tests for unified_llm.errors."""
from __future__ import annotations

import pytest

from unified_llm.errors import (
    AbortError,
    AccessDeniedError,
    AuthenticationError,
    ConfigurationError,
    ContentFilterError,
    ContextLengthError,
    InvalidRequestError,
    InvalidToolCallError,
    NetworkError,
    NoObjectGeneratedError,
    NotFoundError,
    ProviderError,
    QuotaExceededError,
    RateLimitError,
    RequestTimeoutError,
    SDKError,
    ServerError,
    StreamError,
    error_from_status_code,
)


# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------


class TestSDKError:
    def test_is_exception(self) -> None:
        assert issubclass(SDKError, Exception)

    def test_message(self) -> None:
        err = SDKError("boom")
        assert str(err) == "boom"

    def test_cause_default_none(self) -> None:
        err = SDKError("boom")
        assert err.cause is None

    def test_cause_set(self) -> None:
        orig = ValueError("original")
        err = SDKError("wrapped", cause=orig)
        assert err.cause is orig


class TestProviderErrorFields:
    def test_is_sdk_error(self) -> None:
        assert issubclass(ProviderError, SDKError)

    def test_defaults(self) -> None:
        err = ProviderError("fail")
        assert err.provider == ""
        assert err.status_code is None
        assert err.error_code is None
        assert err.retryable is False
        assert err.retry_after is None
        assert err.raw is None
        assert err.cause is None

    def test_all_fields(self) -> None:
        cause = RuntimeError("inner")
        err = ProviderError(
            "fail",
            provider="openai",
            status_code=500,
            error_code="server_error",
            retryable=True,
            retry_after=1.5,
            raw={"error": "oops"},
            cause=cause,
        )
        assert err.provider == "openai"
        assert err.status_code == 500
        assert err.error_code == "server_error"
        assert err.retryable is True
        assert err.retry_after == 1.5
        assert err.raw == {"error": "oops"}
        assert err.cause is cause


# ---------------------------------------------------------------------------
# Specific provider errors: hierarchy + retryable defaults
# ---------------------------------------------------------------------------


class TestAuthenticationError:
    def test_isinstance_provider(self) -> None:
        err = AuthenticationError("bad key")
        assert isinstance(err, ProviderError)
        assert isinstance(err, SDKError)

    def test_retryable_default(self) -> None:
        assert AuthenticationError("bad key").retryable is False

    def test_retryable_override(self) -> None:
        assert AuthenticationError("bad key", retryable=True).retryable is True


class TestAccessDeniedError:
    def test_retryable_default(self) -> None:
        assert AccessDeniedError("nope").retryable is False

    def test_isinstance(self) -> None:
        assert isinstance(AccessDeniedError("nope"), ProviderError)


class TestNotFoundError:
    def test_retryable_default(self) -> None:
        assert NotFoundError("gone").retryable is False

    def test_isinstance(self) -> None:
        assert isinstance(NotFoundError("gone"), ProviderError)


class TestInvalidRequestError:
    def test_retryable_default(self) -> None:
        assert InvalidRequestError("bad").retryable is False


class TestRateLimitError:
    def test_retryable_default(self) -> None:
        assert RateLimitError("slow down").retryable is True

    def test_retryable_override(self) -> None:
        assert RateLimitError("slow down", retryable=False).retryable is False


class TestServerError:
    def test_retryable_default(self) -> None:
        assert ServerError("500").retryable is True


class TestContentFilterError:
    def test_retryable_default(self) -> None:
        assert ContentFilterError("blocked").retryable is False

    def test_isinstance(self) -> None:
        assert isinstance(ContentFilterError("blocked"), ProviderError)


class TestContextLengthError:
    def test_retryable_default(self) -> None:
        assert ContextLengthError("too long").retryable is False


class TestQuotaExceededError:
    def test_retryable_default(self) -> None:
        assert QuotaExceededError("no quota").retryable is False

    def test_isinstance(self) -> None:
        assert isinstance(QuotaExceededError("no quota"), ProviderError)


# ---------------------------------------------------------------------------
# Non-provider errors
# ---------------------------------------------------------------------------


class TestNonProviderErrors:
    @pytest.mark.parametrize(
        "cls",
        [
            RequestTimeoutError,
            AbortError,
            NetworkError,
            StreamError,
            InvalidToolCallError,
            NoObjectGeneratedError,
            ConfigurationError,
        ],
    )
    def test_isinstance_sdk_error(self, cls: type) -> None:
        err = cls("oops")
        assert isinstance(err, SDKError)
        assert not isinstance(err, ProviderError)

    def test_request_timeout_message(self) -> None:
        err = RequestTimeoutError("timed out")
        assert str(err) == "timed out"

    def test_abort_error_cause(self) -> None:
        cause = RuntimeError("cancelled")
        err = AbortError("aborted", cause=cause)
        assert err.cause is cause


# ---------------------------------------------------------------------------
# error_from_status_code helper
# ---------------------------------------------------------------------------


class TestErrorFromStatusCode:
    def test_400(self) -> None:
        err = error_from_status_code(400, "bad request")
        assert isinstance(err, InvalidRequestError)
        assert err.status_code == 400

    def test_422(self) -> None:
        err = error_from_status_code(422, "unprocessable")
        assert isinstance(err, InvalidRequestError)

    def test_401(self) -> None:
        err = error_from_status_code(401, "unauthorized")
        assert isinstance(err, AuthenticationError)

    def test_403(self) -> None:
        err = error_from_status_code(403, "forbidden")
        assert isinstance(err, AccessDeniedError)

    def test_404(self) -> None:
        err = error_from_status_code(404, "not found")
        assert isinstance(err, NotFoundError)

    def test_408(self) -> None:
        err = error_from_status_code(408, "timeout")
        assert isinstance(err, ProviderError)
        assert err.retryable is True
        # 408 should NOT be a RequestTimeoutError (that's non-provider)
        assert not isinstance(err, AuthenticationError)

    def test_413(self) -> None:
        err = error_from_status_code(413, "too large")
        assert isinstance(err, ContextLengthError)

    def test_429(self) -> None:
        err = error_from_status_code(429, "rate limited")
        assert isinstance(err, RateLimitError)
        assert err.retryable is True

    def test_500(self) -> None:
        err = error_from_status_code(500, "internal")
        assert isinstance(err, ServerError)
        assert err.retryable is True

    def test_502(self) -> None:
        err = error_from_status_code(502, "bad gateway")
        assert isinstance(err, ServerError)

    def test_503(self) -> None:
        err = error_from_status_code(503, "unavailable")
        assert isinstance(err, ServerError)

    def test_599(self) -> None:
        err = error_from_status_code(599, "unknown server")
        assert isinstance(err, ServerError)

    def test_unknown_code(self) -> None:
        err = error_from_status_code(418, "teapot")
        assert isinstance(err, ProviderError)
        assert err.retryable is True

    def test_provider_kwarg(self) -> None:
        err = error_from_status_code(500, "fail", provider="anthropic")
        assert err.provider == "anthropic"

    def test_retry_after_kwarg(self) -> None:
        err = error_from_status_code(429, "wait", retry_after=30.0)
        assert err.retry_after == 30.0

    def test_raw_kwarg(self) -> None:
        raw = {"error": {"type": "invalid_request"}}
        err = error_from_status_code(400, "bad", raw=raw)
        assert err.raw is raw

    def test_error_code_kwarg(self) -> None:
        err = error_from_status_code(500, "fail", error_code="internal_error")
        assert err.error_code == "internal_error"
