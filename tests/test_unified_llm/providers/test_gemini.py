"""Tests for the Gemini provider adapter."""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from unified_llm.errors import (
    AuthenticationError,
    InvalidRequestError,
    NotFoundError,
    RateLimitError,
    ServerError,
)
from unified_llm.providers.gemini import GeminiAdapter
from unified_llm.types.content import ContentPart, ImageData, ToolCallData
from unified_llm.types.enums import (
    ContentKind,
    FinishReason,
    Role,
    StreamEventType,
    ToolChoiceMode,
)
from unified_llm.types.messages import Message
from unified_llm.types.request import Request
from unified_llm.types.response import FinishReasonInfo, Response, Usage
from unified_llm.types.streaming import StreamEvent
from unified_llm.types.tools import Tool, ToolCall, ToolChoice


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(handler) -> GeminiAdapter:
    """Create an adapter wired to a mock transport."""
    adapter = GeminiAdapter(api_key="test-key")
    adapter._client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://generativelanguage.googleapis.com",
        headers={"content-type": "application/json"},
        timeout=httpx.Timeout(120.0),
    )
    return adapter


def _simple_response(**overrides: Any) -> dict[str, Any]:
    base = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Hello!"}],
                    "role": "model",
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "totalTokenCount": 15,
        },
    }
    base.update(overrides)
    return base


def _sse_lines(data_objects: list[dict[str, Any]]) -> str:
    """Build SSE text from a list of data objects (Gemini uses data-only SSE)."""
    lines: list[str] = []
    for data in data_objects:
        lines.append(f"data: {json.dumps(data)}")
        lines.append("")
    return "\n".join(lines)


# ===========================================================================
# Request Translation Tests
# ===========================================================================


class TestGeminiRequestTranslation:

    def test_system_instruction(self):
        """System messages should become system_instruction."""
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="gemini-2.0-flash",
            messages=(
                Message.system("You are helpful."),
                Message.user("Hi"),
            ),
        )
        adapter.complete(req)

        body = captured["body"]
        assert "system_instruction" in body
        assert body["system_instruction"]["parts"][0]["text"] == "You are helpful."

    def test_developer_merged_with_system(self):
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="gemini-2.0-flash",
            messages=(
                Message.system("System."),
                Message.developer("Developer."),
                Message.user("Hi"),
            ),
        )
        adapter.complete(req)

        parts = captured["body"]["system_instruction"]["parts"]
        texts = [p["text"] for p in parts]
        assert "System." in texts
        assert "Developer." in texts

    def test_user_message_format(self):
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="gemini-2.0-flash",
            messages=(Message.user("Hello"),),
        )
        adapter.complete(req)

        contents = captured["body"]["contents"]
        assert len(contents) == 1
        assert contents[0]["role"] == "user"
        assert contents[0]["parts"][0]["text"] == "Hello"

    def test_assistant_message_role_model(self):
        """Assistant messages should use role=model in Gemini."""
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="gemini-2.0-flash",
            messages=(
                Message.user("Hi"),
                Message.assistant("Hello"),
                Message.user("Thanks"),
            ),
        )
        adapter.complete(req)

        contents = captured["body"]["contents"]
        model_msgs = [c for c in contents if c["role"] == "model"]
        assert len(model_msgs) == 1
        assert model_msgs[0]["parts"][0]["text"] == "Hello"

    def test_tool_result_format(self):
        """Tool results should become functionResponse parts."""
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="gemini-2.0-flash",
            messages=(
                Message.user("Search"),
                Message.assistant(
                    tool_calls=[
                        ToolCallData(id="tc_1", name="search", arguments={"q": "test"})
                    ]
                ),
                Message.tool_result("tc_1", "Found results"),
            ),
        )
        adapter.complete(req)

        contents = captured["body"]["contents"]
        # Tool result should be in a user-role content
        tool_msg = [c for c in contents if c["role"] == "user"][-1]
        fr = tool_msg["parts"][0].get("functionResponse")
        assert fr is not None
        assert fr["response"] == {"result": "Found results"}

    def test_image_inline_data(self):
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="gemini-2.0-flash",
            messages=(
                Message(
                    role=Role.USER,
                    content=(
                        ContentPart.image_base64(
                            data=b"imgdata", media_type="image/png"
                        ),
                    ),
                ),
            ),
        )
        adapter.complete(req)

        parts = captured["body"]["contents"][0]["parts"]
        inline = parts[0].get("inlineData")
        assert inline is not None
        assert inline["mimeType"] == "image/png"

    def test_image_url_to_file_data(self):
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="gemini-2.0-flash",
            messages=(
                Message(
                    role=Role.USER,
                    content=(ContentPart.image_url("https://example.com/img.jpg"),),
                ),
            ),
        )
        adapter.complete(req)

        parts = captured["body"]["contents"][0]["parts"]
        file_data = parts[0].get("fileData")
        assert file_data is not None
        assert file_data["fileUri"] == "https://example.com/img.jpg"

    def test_tool_call_to_function_call(self):
        """Tool calls in assistant messages should become functionCall parts."""
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="gemini-2.0-flash",
            messages=(
                Message.user("Do something"),
                Message.assistant(
                    tool_calls=[
                        ToolCallData(id="tc_1", name="fn", arguments={"x": 1})
                    ]
                ),
                Message.tool_result("tc_1", "done"),
            ),
        )
        adapter.complete(req)

        contents = captured["body"]["contents"]
        model_msg = [c for c in contents if c["role"] == "model"][0]
        fc_part = [p for p in model_msg["parts"] if "functionCall" in p][0]
        assert fc_part["functionCall"]["name"] == "fn"
        assert fc_part["functionCall"]["args"] == {"x": 1}

    def test_function_declarations(self):
        """Tools should be formatted as functionDeclarations."""
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        tool = Tool(
            name="search",
            description="Search the web",
            parameters={"type": "object", "properties": {"q": {"type": "string"}}},
        )
        req = Request(
            model="gemini-2.0-flash",
            messages=(Message.user("Search"),),
            tools=(tool,),
        )
        adapter.complete(req)

        tools = captured["body"]["tools"]
        assert len(tools) == 1
        decls = tools[0]["functionDeclarations"]
        assert len(decls) == 1
        assert decls[0]["name"] == "search"
        assert decls[0]["description"] == "Search the web"

    def test_tool_choice_auto(self):
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="gemini-2.0-flash",
            messages=(Message.user("Hi"),),
            tools=(Tool(name="t", description="d"),),
            tool_choice=ToolChoice(mode=ToolChoiceMode.AUTO),
        )
        adapter.complete(req)
        tc = captured["body"]["tool_config"]["function_calling_config"]
        assert tc["mode"] == "AUTO"

    def test_tool_choice_none(self):
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="gemini-2.0-flash",
            messages=(Message.user("Hi"),),
            tools=(Tool(name="t", description="d"),),
            tool_choice=ToolChoice(mode=ToolChoiceMode.NONE),
        )
        adapter.complete(req)
        tc = captured["body"]["tool_config"]["function_calling_config"]
        assert tc["mode"] == "NONE"
        assert "tools" not in captured["body"]

    def test_tool_choice_required(self):
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="gemini-2.0-flash",
            messages=(Message.user("Hi"),),
            tools=(Tool(name="t", description="d"),),
            tool_choice=ToolChoice(mode=ToolChoiceMode.REQUIRED),
        )
        adapter.complete(req)
        tc = captured["body"]["tool_config"]["function_calling_config"]
        assert tc["mode"] == "ANY"

    def test_tool_choice_named(self):
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="gemini-2.0-flash",
            messages=(Message.user("Hi"),),
            tools=(Tool(name="search", description="d"),),
            tool_choice=ToolChoice(mode=ToolChoiceMode.NAMED, tool_name="search"),
        )
        adapter.complete(req)
        tc = captured["body"]["tool_config"]["function_calling_config"]
        assert tc["mode"] == "ANY"
        assert tc["allowed_function_names"] == ["search"]

    def test_generation_config(self):
        """Temperature, topP, maxOutputTokens, stopSequences should go to generationConfig."""
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="gemini-2.0-flash",
            messages=(Message.user("Hi"),),
            temperature=0.7,
            top_p=0.9,
            max_tokens=1024,
            stop_sequences=("STOP",),
        )
        adapter.complete(req)

        gc = captured["body"]["generationConfig"]
        assert gc["temperature"] == 0.7
        assert gc["topP"] == 0.9
        assert gc["maxOutputTokens"] == 1024
        assert gc["stopSequences"] == ["STOP"]

    def test_api_key_in_query_params(self):
        """API key should be sent as a query parameter, not a header."""
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["headers"] = dict(request.headers)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        adapter.complete(
            Request(model="gemini-2.0-flash", messages=(Message.user("Hi"),))
        )
        assert "key=test-key" in captured["url"]
        # Should NOT be in headers
        assert "x-api-key" not in captured["headers"]

    def test_endpoint_path(self):
        """Should use /v1beta/models/{model}:generateContent."""
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        adapter.complete(
            Request(model="gemini-2.0-flash", messages=(Message.user("Hi"),))
        )
        assert "/v1beta/models/gemini-2.0-flash:generateContent" in captured["url"]

    def test_consecutive_same_role_merged(self):
        """Consecutive same-role messages should be merged."""
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="gemini-2.0-flash",
            messages=(
                Message.user("First"),
                Message.user("Second"),
                Message.assistant("Reply"),
            ),
        )
        adapter.complete(req)

        contents = captured["body"]["contents"]
        assert len(contents) == 2
        assert contents[0]["role"] == "user"
        assert len(contents[0]["parts"]) == 2

    def test_provider_options_merged(self):
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="gemini-2.0-flash",
            messages=(Message.user("Hi"),),
            provider_options={"gemini": {"safetySettings": []}},
        )
        adapter.complete(req)
        assert "safetySettings" in captured["body"]


# ===========================================================================
# Response Parsing Tests
# ===========================================================================


class TestGeminiResponseParsing:

    def test_simple_text_response(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        response = adapter.complete(
            Request(model="gemini-2.0-flash", messages=(Message.user("Hi"),))
        )
        assert response.text == "Hello!"
        assert response.finish_reason.reason == FinishReason.STOP
        assert response.provider == "gemini"

    def test_function_call_response(self):
        raw = _simple_response(
            candidates=[
                {
                    "content": {
                        "parts": [
                            {"functionCall": {"name": "get_weather", "args": {"city": "NYC"}}}
                        ],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ]
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=raw)

        adapter = _make_adapter(handler)
        response = adapter.complete(
            Request(model="gemini-2.0-flash", messages=(Message.user("Weather"),))
        )
        assert response.finish_reason.reason == FinishReason.TOOL_CALLS
        assert len(response.tool_calls) == 1
        tc = response.tool_calls[0]
        assert tc.name == "get_weather"
        assert tc.arguments == {"city": "NYC"}
        # Should have a synthetic ID
        assert tc.id.startswith("call_")

    def test_synthetic_tool_call_id(self):
        """Gemini doesn't provide tool call IDs, so they should be synthesized."""
        raw = _simple_response(
            candidates=[
                {
                    "content": {
                        "parts": [
                            {"functionCall": {"name": "fn1", "args": {}}},
                            {"functionCall": {"name": "fn2", "args": {}}},
                        ],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ]
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=raw)

        adapter = _make_adapter(handler)
        response = adapter.complete(
            Request(model="gemini-2.0-flash", messages=(Message.user("Multi"),))
        )
        ids = [tc.id for tc in response.tool_calls]
        assert len(ids) == 2
        # IDs should be unique
        assert ids[0] != ids[1]
        assert all(id_.startswith("call_") for id_ in ids)

    def test_finish_reason_stop(self):
        raw = _simple_response()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=raw)

        adapter = _make_adapter(handler)
        response = adapter.complete(
            Request(model="gemini-2.0-flash", messages=(Message.user("Hi"),))
        )
        assert response.finish_reason.reason == FinishReason.STOP
        assert response.finish_reason.raw == "STOP"

    def test_finish_reason_max_tokens(self):
        raw = _simple_response(
            candidates=[
                {
                    "content": {"parts": [{"text": "Truncated"}], "role": "model"},
                    "finishReason": "MAX_TOKENS",
                }
            ]
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=raw)

        adapter = _make_adapter(handler)
        response = adapter.complete(
            Request(model="gemini-2.0-flash", messages=(Message.user("Hi"),))
        )
        assert response.finish_reason.reason == FinishReason.LENGTH

    def test_finish_reason_safety(self):
        raw = _simple_response(
            candidates=[
                {
                    "content": {"parts": [{"text": ""}], "role": "model"},
                    "finishReason": "SAFETY",
                }
            ]
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=raw)

        adapter = _make_adapter(handler)
        response = adapter.complete(
            Request(model="gemini-2.0-flash", messages=(Message.user("Hi"),))
        )
        assert response.finish_reason.reason == FinishReason.CONTENT_FILTER

    def test_function_call_overrides_finish_reason(self):
        """If function calls present but finish reason is STOP, should be TOOL_CALLS."""
        raw = _simple_response(
            candidates=[
                {
                    "content": {
                        "parts": [
                            {"functionCall": {"name": "fn", "args": {}}}
                        ],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ]
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=raw)

        adapter = _make_adapter(handler)
        response = adapter.complete(
            Request(model="gemini-2.0-flash", messages=(Message.user("Call"),))
        )
        assert response.finish_reason.reason == FinishReason.TOOL_CALLS

    def test_usage_extraction(self):
        raw = _simple_response(
            usageMetadata={
                "promptTokenCount": 100,
                "candidatesTokenCount": 50,
                "totalTokenCount": 150,
                "cachedContentTokenCount": 20,
                "thoughtsTokenCount": 10,
            }
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=raw)

        adapter = _make_adapter(handler)
        response = adapter.complete(
            Request(model="gemini-2.0-flash", messages=(Message.user("Hi"),))
        )
        assert response.usage.input_tokens == 100
        assert response.usage.output_tokens == 50
        assert response.usage.total_tokens == 150
        assert response.usage.cache_read_tokens == 20
        assert response.usage.reasoning_tokens == 10

    def test_basic_usage(self):
        raw = _simple_response()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=raw)

        adapter = _make_adapter(handler)
        response = adapter.complete(
            Request(model="gemini-2.0-flash", messages=(Message.user("Hi"),))
        )
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 5
        assert response.usage.total_tokens == 15

    def test_mixed_text_and_function_call(self):
        raw = _simple_response(
            candidates=[
                {
                    "content": {
                        "parts": [
                            {"text": "Let me search."},
                            {"functionCall": {"name": "search", "args": {"q": "test"}}},
                        ],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ]
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=raw)

        adapter = _make_adapter(handler)
        response = adapter.complete(
            Request(model="gemini-2.0-flash", messages=(Message.user("Search"),))
        )
        assert response.text == "Let me search."
        assert len(response.tool_calls) == 1

    def test_empty_candidates(self):
        raw = {"candidates": [], "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 0, "totalTokenCount": 5}}

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=raw)

        adapter = _make_adapter(handler)
        response = adapter.complete(
            Request(model="gemini-2.0-flash", messages=(Message.user("Hi"),))
        )
        assert response.text == ""


# ===========================================================================
# Error Handling Tests
# ===========================================================================


class TestGeminiErrorHandling:

    def test_401_authentication_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                401,
                json={"error": {"code": 401, "message": "API key not valid", "status": "UNAUTHENTICATED"}},
            )

        adapter = _make_adapter(handler)
        with pytest.raises(AuthenticationError) as exc_info:
            adapter.complete(Request(model="m", messages=(Message.user("Hi"),)))
        assert exc_info.value.provider == "gemini"

    def test_429_rate_limit_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                429,
                json={"error": {"code": 429, "message": "Resource exhausted", "status": "RESOURCE_EXHAUSTED"}},
            )

        adapter = _make_adapter(handler)
        with pytest.raises(RateLimitError):
            adapter.complete(Request(model="m", messages=(Message.user("Hi"),)))

    def test_500_server_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                500,
                json={"error": {"code": 500, "message": "Internal error", "status": "INTERNAL"}},
            )

        adapter = _make_adapter(handler)
        with pytest.raises(ServerError):
            adapter.complete(Request(model="m", messages=(Message.user("Hi"),)))

    def test_400_invalid_request(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400,
                json={"error": {"code": 400, "message": "Invalid argument", "status": "INVALID_ARGUMENT"}},
            )

        adapter = _make_adapter(handler)
        with pytest.raises(InvalidRequestError):
            adapter.complete(Request(model="m", messages=(Message.user("Hi"),)))

    def test_404_not_found(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                404,
                json={"error": {"code": 404, "message": "Model not found", "status": "NOT_FOUND"}},
            )

        adapter = _make_adapter(handler)
        with pytest.raises(NotFoundError):
            adapter.complete(Request(model="m", messages=(Message.user("Hi"),)))

    def test_timeout_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("Timed out")

        adapter = _make_adapter(handler)
        from unified_llm.errors import RequestTimeoutError
        with pytest.raises(RequestTimeoutError):
            adapter.complete(Request(model="m", messages=(Message.user("Hi"),)))

    def test_network_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused")

        adapter = _make_adapter(handler)
        from unified_llm.errors import NetworkError
        with pytest.raises(NetworkError):
            adapter.complete(Request(model="m", messages=(Message.user("Hi"),)))

    def test_error_with_plain_text_body(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Internal Server Error")

        adapter = _make_adapter(handler)
        with pytest.raises(ServerError):
            adapter.complete(Request(model="m", messages=(Message.user("Hi"),)))


# ===========================================================================
# Streaming Tests
# ===========================================================================


class TestGeminiStreaming:

    def test_text_streaming(self):
        sse_text = _sse_lines([
            {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "Hello"}], "role": "model"},
                    }
                ],
            },
            {
                "candidates": [
                    {
                        "content": {"parts": [{"text": " world"}], "role": "model"},
                    }
                ],
            },
            {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "!"}], "role": "model"},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 5,
                    "totalTokenCount": 15,
                },
            },
        ])

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=sse_text.encode(),
                headers={"content-type": "text/event-stream"},
            )

        adapter = _make_adapter(handler)
        events = list(adapter.stream(
            Request(model="gemini-2.0-flash", messages=(Message.user("Hi"),))
        ))

        event_types = [e.type for e in events]
        assert StreamEventType.STREAM_START in event_types
        assert StreamEventType.TEXT_DELTA in event_types
        assert StreamEventType.FINISH in event_types

        text_deltas = [e.delta for e in events if e.type == StreamEventType.TEXT_DELTA]
        assert "".join(text_deltas) == "Hello world!"

    def test_function_call_streaming(self):
        """Gemini sends function calls as complete objects in single chunks."""
        sse_text = _sse_lines([
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"functionCall": {"name": "get_weather", "args": {"city": "NYC"}}}
                            ],
                            "role": "model",
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 5,
                    "totalTokenCount": 15,
                },
            },
        ])

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=sse_text.encode(),
                headers={"content-type": "text/event-stream"},
            )

        adapter = _make_adapter(handler)
        events = list(adapter.stream(
            Request(model="gemini-2.0-flash", messages=(Message.user("Weather"),))
        ))

        event_types = [e.type for e in events]
        assert StreamEventType.TOOL_CALL_START in event_types
        assert StreamEventType.TOOL_CALL_END in event_types

        tc_end = [e for e in events if e.type == StreamEventType.TOOL_CALL_END][0]
        assert tc_end.tool_call.name == "get_weather"
        assert tc_end.tool_call.arguments == {"city": "NYC"}

    def test_finish_event_has_usage(self):
        sse_text = _sse_lines([
            {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "Done"}], "role": "model"},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 20,
                    "candidatesTokenCount": 10,
                    "totalTokenCount": 30,
                },
            },
        ])

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=sse_text.encode(),
                headers={"content-type": "text/event-stream"},
            )

        adapter = _make_adapter(handler)
        events = list(adapter.stream(
            Request(model="gemini-2.0-flash", messages=(Message.user("Hi"),))
        ))

        finish = [e for e in events if e.type == StreamEventType.FINISH][0]
        assert finish.usage.input_tokens == 20
        assert finish.usage.output_tokens == 10

    def test_stream_endpoint_path(self):
        """Should use streamGenerateContent with alt=sse."""
        captured: dict[str, Any] = {}

        sse_text = _sse_lines([
            {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "OK"}], "role": "model"},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 1, "totalTokenCount": 6},
            },
        ])

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            return httpx.Response(
                200,
                content=sse_text.encode(),
                headers={"content-type": "text/event-stream"},
            )

        adapter = _make_adapter(handler)
        list(adapter.stream(
            Request(model="gemini-2.0-flash", messages=(Message.user("Hi"),))
        ))
        assert "streamGenerateContent" in captured["url"]
        assert "alt=sse" in captured["url"]
        assert "key=test-key" in captured["url"]

    def test_stream_http_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                429,
                json={"error": {"code": 429, "message": "Rate limited"}},
            )

        adapter = _make_adapter(handler)
        with pytest.raises(RateLimitError):
            list(adapter.stream(
                Request(model="m", messages=(Message.user("Hi"),))
            ))

    def test_stream_multiple_chunks(self):
        """Multiple chunks should all produce TEXT_DELTA events."""
        sse_text = _sse_lines([
            {"candidates": [{"content": {"parts": [{"text": "A"}], "role": "model"}}]},
            {"candidates": [{"content": {"parts": [{"text": "B"}], "role": "model"}}]},
            {"candidates": [{"content": {"parts": [{"text": "C"}], "role": "model"}}]},
            {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "D"}], "role": "model"},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 4, "totalTokenCount": 9},
            },
        ])

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=sse_text.encode(),
                headers={"content-type": "text/event-stream"},
            )

        adapter = _make_adapter(handler)
        events = list(adapter.stream(
            Request(model="gemini-2.0-flash", messages=(Message.user("Hi"),))
        ))

        text_deltas = [e.delta for e in events if e.type == StreamEventType.TEXT_DELTA]
        assert text_deltas == ["A", "B", "C", "D"]

    def test_finish_event_contains_response(self):
        sse_text = _sse_lines([
            {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "OK"}], "role": "model"},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 1, "totalTokenCount": 6},
            },
        ])

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=sse_text.encode(),
                headers={"content-type": "text/event-stream"},
            )

        adapter = _make_adapter(handler)
        events = list(adapter.stream(
            Request(model="gemini-2.0-flash", messages=(Message.user("Hi"),))
        ))

        finish = [e for e in events if e.type == StreamEventType.FINISH][0]
        assert finish.response is not None
        assert finish.response.provider == "gemini"

    def test_stream_start_emitted_once(self):
        sse_text = _sse_lines([
            {"candidates": [{"content": {"parts": [{"text": "A"}], "role": "model"}}]},
            {"candidates": [{"content": {"parts": [{"text": "B"}], "role": "model"}}]},
            {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "C"}], "role": "model"},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3, "totalTokenCount": 8},
            },
        ])

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=sse_text.encode(),
                headers={"content-type": "text/event-stream"},
            )

        adapter = _make_adapter(handler)
        events = list(adapter.stream(
            Request(model="gemini-2.0-flash", messages=(Message.user("Hi"),))
        ))

        stream_starts = [e for e in events if e.type == StreamEventType.STREAM_START]
        assert len(stream_starts) == 1


# ===========================================================================
# Adapter Properties
# ===========================================================================


class TestGeminiAdapterProperties:

    def test_name(self):
        adapter = GeminiAdapter(api_key="test")
        assert adapter.name == "gemini"

    def test_close(self):
        adapter = GeminiAdapter(api_key="test")
        adapter.close()  # Should not raise

    def test_default_base_url(self):
        adapter = GeminiAdapter(api_key="test")
        assert "generativelanguage.googleapis.com" in adapter._base_url

    def test_custom_base_url(self):
        adapter = GeminiAdapter(api_key="test", base_url="https://custom.api.com")
        assert adapter._base_url == "https://custom.api.com"
