"""Tests for the OpenAI provider adapter (Responses API)."""
from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from unified_llm.errors import (
    AuthenticationError,
    InvalidRequestError,
    NotFoundError,
    RateLimitError,
    ServerError,
)
from unified_llm.providers.openai import OpenAIAdapter
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

def _make_adapter(handler, **kwargs) -> OpenAIAdapter:
    """Create an adapter wired to a mock transport."""
    adapter = OpenAIAdapter(api_key="test-key", **kwargs)
    adapter._client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://api.openai.com",
        headers={
            "authorization": "Bearer test-key",
            "content-type": "application/json",
        },
        timeout=httpx.Timeout(120.0),
    )
    return adapter


def _simple_response(**overrides: Any) -> dict[str, Any]:
    base = {
        "id": "resp_123",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello!"}],
            }
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }
    base.update(overrides)
    return base


def _sse_lines(events: list[tuple[str, dict[str, Any]]]) -> str:
    lines: list[str] = []
    for event_type, data in events:
        lines.append(f"event: {event_type}")
        lines.append(f"data: {json.dumps(data)}")
        lines.append("")
    return "\n".join(lines)


# ===========================================================================
# Request Translation Tests
# ===========================================================================


class TestOpenAIRequestTranslation:

    def test_system_message_to_instructions(self):
        """System messages should be concatenated into the instructions field."""
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="gpt-4o",
            messages=(
                Message.system("You are helpful."),
                Message.system("Be concise."),
                Message.user("Hi"),
            ),
        )
        adapter.complete(req)

        body = captured["body"]
        assert "instructions" in body
        assert "You are helpful." in body["instructions"]
        assert "Be concise." in body["instructions"]
        # System messages should not appear in input
        for item in body["input"]:
            if item.get("type") == "message":
                assert item.get("role") != "system"

    def test_developer_messages_to_instructions(self):
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="gpt-4o",
            messages=(
                Message.developer("Dev instruction."),
                Message.user("Hi"),
            ),
        )
        adapter.complete(req)
        assert "Dev instruction." in captured["body"]["instructions"]

    def test_user_message_format(self):
        """User messages should use input_text content type."""
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="gpt-4o",
            messages=(Message.user("Hello"),),
        )
        adapter.complete(req)

        input_items = captured["body"]["input"]
        assert len(input_items) == 1
        assert input_items[0]["type"] == "message"
        assert input_items[0]["role"] == "user"
        assert input_items[0]["content"][0]["type"] == "input_text"
        assert input_items[0]["content"][0]["text"] == "Hello"

    def test_assistant_message_format(self):
        """Assistant text messages should use output_text content type."""
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="gpt-4o",
            messages=(
                Message.user("Hi"),
                Message.assistant("Hello back"),
                Message.user("Thanks"),
            ),
        )
        adapter.complete(req)

        input_items = captured["body"]["input"]
        assistant_msgs = [i for i in input_items if i.get("role") == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0]["content"][0]["type"] == "output_text"
        assert assistant_msgs[0]["content"][0]["text"] == "Hello back"

    def test_tool_call_becomes_function_call(self):
        """Tool calls should become function_call items."""
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="gpt-4o",
            messages=(
                Message.user("Search"),
                Message.assistant(
                    tool_calls=[
                        ToolCallData(
                            id="call_1",
                            name="search",
                            arguments={"q": "test"},
                        )
                    ]
                ),
                Message.tool_result("call_1", "results"),
            ),
        )
        adapter.complete(req)

        input_items = captured["body"]["input"]
        fc_items = [i for i in input_items if i.get("type") == "function_call"]
        assert len(fc_items) == 1
        assert fc_items[0]["id"] == "call_1"
        assert fc_items[0]["name"] == "search"
        assert fc_items[0]["arguments"] == '{"q": "test"}'

    def test_tool_result_becomes_function_call_output(self):
        """Tool results should become function_call_output items."""
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="gpt-4o",
            messages=(
                Message.user("Search"),
                Message.assistant(
                    tool_calls=[
                        ToolCallData(id="call_1", name="search", arguments={"q": "test"})
                    ]
                ),
                Message.tool_result("call_1", "Found 3 results"),
            ),
        )
        adapter.complete(req)

        input_items = captured["body"]["input"]
        fco_items = [i for i in input_items if i.get("type") == "function_call_output"]
        assert len(fco_items) == 1
        assert fco_items[0]["call_id"] == "call_1"
        assert fco_items[0]["output"] == "Found 3 results"

    def test_image_url_content(self):
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="gpt-4o",
            messages=(
                Message(
                    role=Role.USER,
                    content=(ContentPart.image_url("https://example.com/img.png"),),
                ),
            ),
        )
        adapter.complete(req)

        content = captured["body"]["input"][0]["content"]
        assert content[0]["type"] == "input_image"
        assert content[0]["image_url"] == "https://example.com/img.png"

    def test_tool_definition_format(self):
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
            model="gpt-4o",
            messages=(Message.user("Search"),),
            tools=(tool,),
        )
        adapter.complete(req)

        tools = captured["body"]["tools"]
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["name"] == "search"
        assert tools[0]["parameters"] == tool.parameters

    def test_tool_choice_auto(self):
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="gpt-4o",
            messages=(Message.user("Hi"),),
            tools=(Tool(name="t", description="d"),),
            tool_choice=ToolChoice(mode=ToolChoiceMode.AUTO),
        )
        adapter.complete(req)
        assert captured["body"]["tool_choice"] == "auto"

    def test_tool_choice_none(self):
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="gpt-4o",
            messages=(Message.user("Hi"),),
            tools=(Tool(name="t", description="d"),),
            tool_choice=ToolChoice(mode=ToolChoiceMode.NONE),
        )
        adapter.complete(req)
        assert captured["body"]["tool_choice"] == "none"

    def test_tool_choice_required(self):
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="gpt-4o",
            messages=(Message.user("Hi"),),
            tools=(Tool(name="t", description="d"),),
            tool_choice=ToolChoice(mode=ToolChoiceMode.REQUIRED),
        )
        adapter.complete(req)
        assert captured["body"]["tool_choice"] == "required"

    def test_tool_choice_named(self):
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="gpt-4o",
            messages=(Message.user("Hi"),),
            tools=(Tool(name="search", description="d"),),
            tool_choice=ToolChoice(mode=ToolChoiceMode.NAMED, tool_name="search"),
        )
        adapter.complete(req)
        assert captured["body"]["tool_choice"] == {"type": "function", "name": "search"}

    def test_max_tokens_mapped_to_max_output_tokens(self):
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="gpt-4o",
            messages=(Message.user("Hi"),),
            max_tokens=2048,
        )
        adapter.complete(req)
        assert captured["body"]["max_output_tokens"] == 2048

    def test_reasoning_effort(self):
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="o3",
            messages=(Message.user("Think hard"),),
            reasoning_effort="high",
        )
        adapter.complete(req)
        assert captured["body"]["reasoning"] == {"effort": "high"}

    def test_provider_options_merged(self):
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="gpt-4o",
            messages=(Message.user("Hi"),),
            provider_options={"openai": {"store": True}},
        )
        adapter.complete(req)
        assert captured["body"]["store"] is True

    def test_endpoint_is_v1_responses(self):
        """Requests should go to /v1/responses."""
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        adapter.complete(Request(model="gpt-4o", messages=(Message.user("Hi"),)))
        assert "/v1/responses" in captured["url"]


# ===========================================================================
# Response Parsing Tests
# ===========================================================================


class TestOpenAIResponseParsing:

    def test_simple_text_response(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        response = adapter.complete(
            Request(model="gpt-4o", messages=(Message.user("Hi"),))
        )
        assert response.text == "Hello!"
        assert response.finish_reason.reason == FinishReason.STOP
        assert response.provider == "openai"
        assert response.id == "resp_123"

    def test_function_call_response(self):
        raw = _simple_response(
            output=[
                {
                    "type": "function_call",
                    "id": "call_1",
                    "name": "get_weather",
                    "arguments": '{"city": "NYC"}',
                }
            ]
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=raw)

        adapter = _make_adapter(handler)
        response = adapter.complete(
            Request(model="gpt-4o", messages=(Message.user("Weather"),))
        )
        assert response.finish_reason.reason == FinishReason.TOOL_CALLS
        assert len(response.tool_calls) == 1
        tc = response.tool_calls[0]
        assert tc.name == "get_weather"
        assert tc.arguments == {"city": "NYC"}

    def test_mixed_output(self):
        """Response with both text and function call."""
        raw = _simple_response(
            output=[
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Let me search."}],
                },
                {
                    "type": "function_call",
                    "id": "call_1",
                    "name": "search",
                    "arguments": '{"q": "test"}',
                },
            ]
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=raw)

        adapter = _make_adapter(handler)
        response = adapter.complete(
            Request(model="gpt-4o", messages=(Message.user("Search"),))
        )
        assert response.text == "Let me search."
        assert len(response.tool_calls) == 1
        assert response.finish_reason.reason == FinishReason.TOOL_CALLS

    def test_incomplete_status(self):
        raw = _simple_response(status="incomplete")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=raw)

        adapter = _make_adapter(handler)
        response = adapter.complete(
            Request(model="gpt-4o", messages=(Message.user("Hi"),))
        )
        assert response.finish_reason.reason == FinishReason.LENGTH

    def test_usage_with_details(self):
        raw = _simple_response(
            usage={
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "output_tokens_details": {"reasoning_tokens": 20},
                "input_tokens_details": {"cached_tokens": 30},
            }
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=raw)

        adapter = _make_adapter(handler)
        response = adapter.complete(
            Request(model="gpt-4o", messages=(Message.user("Hi"),))
        )
        assert response.usage.input_tokens == 100
        assert response.usage.output_tokens == 50
        assert response.usage.total_tokens == 150
        assert response.usage.reasoning_tokens == 20
        assert response.usage.cache_read_tokens == 30

    def test_basic_usage(self):
        raw = _simple_response(
            usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=raw)

        adapter = _make_adapter(handler)
        response = adapter.complete(
            Request(model="gpt-4o", messages=(Message.user("Hi"),))
        )
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 5
        assert response.usage.total_tokens == 15

    def test_empty_output(self):
        raw = _simple_response(output=[])

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=raw)

        adapter = _make_adapter(handler)
        response = adapter.complete(
            Request(model="gpt-4o", messages=(Message.user("Hi"),))
        )
        assert response.text == ""

    def test_multiple_function_calls(self):
        raw = _simple_response(
            output=[
                {"type": "function_call", "id": "c1", "name": "fn1", "arguments": "{}"},
                {"type": "function_call", "id": "c2", "name": "fn2", "arguments": '{"x":1}'},
            ]
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=raw)

        adapter = _make_adapter(handler)
        response = adapter.complete(
            Request(model="gpt-4o", messages=(Message.user("Multi"),))
        )
        assert len(response.tool_calls) == 2
        assert response.tool_calls[0].name == "fn1"
        assert response.tool_calls[1].name == "fn2"

    def test_invalid_json_arguments_preserved(self):
        """Invalid JSON in function call arguments should be preserved as-is."""
        raw = _simple_response(
            output=[
                {
                    "type": "function_call",
                    "id": "c1",
                    "name": "fn",
                    "arguments": "not valid json",
                }
            ]
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=raw)

        adapter = _make_adapter(handler)
        response = adapter.complete(
            Request(model="gpt-4o", messages=(Message.user("Bad"),))
        )
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].arguments == "not valid json"


# ===========================================================================
# Error Handling Tests
# ===========================================================================


class TestOpenAIErrorHandling:

    def test_401_authentication_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                401,
                json={"error": {"message": "Invalid API key", "type": "invalid_api_key"}},
            )

        adapter = _make_adapter(handler)
        with pytest.raises(AuthenticationError) as exc_info:
            adapter.complete(Request(model="m", messages=(Message.user("Hi"),)))
        assert exc_info.value.status_code == 401
        assert exc_info.value.provider == "openai"

    def test_429_rate_limit_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                429,
                json={"error": {"message": "Rate limit exceeded", "type": "rate_limit"}},
                headers={"retry-after": "60"},
            )

        adapter = _make_adapter(handler)
        with pytest.raises(RateLimitError) as exc_info:
            adapter.complete(Request(model="m", messages=(Message.user("Hi"),)))
        assert exc_info.value.retryable is True
        assert exc_info.value.retry_after == 60.0

    def test_500_server_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                500,
                json={"error": {"message": "Server error", "type": "server_error"}},
            )

        adapter = _make_adapter(handler)
        with pytest.raises(ServerError):
            adapter.complete(Request(model="m", messages=(Message.user("Hi"),)))

    def test_400_invalid_request(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400,
                json={"error": {"message": "Invalid model", "type": "invalid_request"}},
            )

        adapter = _make_adapter(handler)
        with pytest.raises(InvalidRequestError):
            adapter.complete(Request(model="m", messages=(Message.user("Hi"),)))

    def test_404_not_found(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                404,
                json={"error": {"message": "Not found", "type": "not_found"}},
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
        """Should handle error responses without JSON body."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Internal Server Error")

        adapter = _make_adapter(handler)
        with pytest.raises(ServerError):
            adapter.complete(Request(model="m", messages=(Message.user("Hi"),)))


# ===========================================================================
# Streaming Tests
# ===========================================================================


class TestOpenAIStreaming:

    def test_text_streaming(self):
        sse_text = _sse_lines([
            ("response.created", {"type": "response.created", "response": {"id": "resp_1"}}),
            ("response.output_item.added", {
                "type": "response.output_item.added",
                "item": {"type": "message", "role": "assistant"},
            }),
            ("response.output_text.delta", {
                "type": "response.output_text.delta",
                "delta": "Hello",
            }),
            ("response.output_text.delta", {
                "type": "response.output_text.delta",
                "delta": " world",
            }),
            ("response.output_item.done", {
                "type": "response.output_item.done",
                "item": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hello world"}]},
            }),
            ("response.completed", {
                "type": "response.completed",
                "response": _simple_response(
                    output=[{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hello world"}]}]
                ),
            }),
        ])

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=sse_text.encode(),
                headers={"content-type": "text/event-stream"},
            )

        adapter = _make_adapter(handler)
        events = list(adapter.stream(
            Request(model="gpt-4o", messages=(Message.user("Hi"),))
        ))

        event_types = [e.type for e in events]
        assert StreamEventType.STREAM_START in event_types
        assert StreamEventType.TEXT_DELTA in event_types
        assert StreamEventType.TEXT_END in event_types
        assert StreamEventType.FINISH in event_types

        text_deltas = [e.delta for e in events if e.type == StreamEventType.TEXT_DELTA]
        assert "".join(text_deltas) == "Hello world"

    def test_tool_call_streaming(self):
        sse_text = _sse_lines([
            ("response.created", {"type": "response.created", "response": {"id": "resp_2"}}),
            ("response.output_item.added", {
                "type": "response.output_item.added",
                "item": {"type": "function_call", "id": "call_1", "name": "get_weather"},
            }),
            ("response.function_call_arguments.delta", {
                "type": "response.function_call_arguments.delta",
                "delta": '{"city":',
            }),
            ("response.function_call_arguments.delta", {
                "type": "response.function_call_arguments.delta",
                "delta": '"NYC"}',
            }),
            ("response.output_item.done", {
                "type": "response.output_item.done",
                "item": {"type": "function_call", "id": "call_1", "name": "get_weather", "arguments": '{"city":"NYC"}'},
            }),
            ("response.completed", {
                "type": "response.completed",
                "response": _simple_response(
                    output=[{"type": "function_call", "id": "call_1", "name": "get_weather", "arguments": '{"city":"NYC"}'}]
                ),
            }),
        ])

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=sse_text.encode(),
                headers={"content-type": "text/event-stream"},
            )

        adapter = _make_adapter(handler)
        events = list(adapter.stream(
            Request(model="gpt-4o", messages=(Message.user("Weather"),))
        ))

        event_types = [e.type for e in events]
        assert StreamEventType.TOOL_CALL_START in event_types
        assert StreamEventType.TOOL_CALL_DELTA in event_types
        assert StreamEventType.TOOL_CALL_END in event_types

        tc_end = [e for e in events if e.type == StreamEventType.TOOL_CALL_END][0]
        assert tc_end.tool_call.name == "get_weather"
        assert tc_end.tool_call.arguments == {"city": "NYC"}

    def test_finish_event_has_usage(self):
        sse_text = _sse_lines([
            ("response.created", {"type": "response.created", "response": {"id": "resp_3"}}),
            ("response.completed", {
                "type": "response.completed",
                "response": _simple_response(
                    usage={"input_tokens": 20, "output_tokens": 10, "total_tokens": 30}
                ),
            }),
        ])

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=sse_text.encode(),
                headers={"content-type": "text/event-stream"},
            )

        adapter = _make_adapter(handler)
        events = list(adapter.stream(
            Request(model="gpt-4o", messages=(Message.user("Hi"),))
        ))

        finish = [e for e in events if e.type == StreamEventType.FINISH][0]
        assert finish.usage.input_tokens == 20
        assert finish.usage.output_tokens == 10

    def test_stream_request_body_has_stream_true(self):
        captured: dict[str, Any] = {}

        sse_text = _sse_lines([
            ("response.completed", {
                "type": "response.completed",
                "response": _simple_response(),
            }),
        ])

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                content=sse_text.encode(),
                headers={"content-type": "text/event-stream"},
            )

        adapter = _make_adapter(handler)
        list(adapter.stream(
            Request(model="gpt-4o", messages=(Message.user("Hi"),))
        ))
        assert captured["body"]["stream"] is True

    def test_stream_error_event(self):
        sse_text = _sse_lines([
            ("error", {
                "type": "error",
                "error": {"message": "Something went wrong", "code": "server_error"},
            }),
        ])

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=sse_text.encode(),
                headers={"content-type": "text/event-stream"},
            )

        adapter = _make_adapter(handler)
        events = list(adapter.stream(
            Request(model="gpt-4o", messages=(Message.user("Hi"),))
        ))

        error_events = [e for e in events if e.type == StreamEventType.ERROR]
        assert len(error_events) == 1

    def test_stream_http_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                500,
                json={"error": {"message": "Server error"}},
            )

        adapter = _make_adapter(handler)
        with pytest.raises(ServerError):
            list(adapter.stream(
                Request(model="m", messages=(Message.user("Hi"),))
            ))

    def test_finish_event_contains_response(self):
        sse_text = _sse_lines([
            ("response.completed", {
                "type": "response.completed",
                "response": _simple_response(id="resp_final", model="gpt-4o"),
            }),
        ])

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=sse_text.encode(),
                headers={"content-type": "text/event-stream"},
            )

        adapter = _make_adapter(handler)
        events = list(adapter.stream(
            Request(model="gpt-4o", messages=(Message.user("Hi"),))
        ))

        finish = [e for e in events if e.type == StreamEventType.FINISH][0]
        assert finish.response is not None
        assert finish.response.id == "resp_final"
        assert finish.response.provider == "openai"


# ===========================================================================
# Adapter Properties
# ===========================================================================


class TestOpenAIAdapterProperties:

    def test_name(self):
        adapter = OpenAIAdapter(api_key="test")
        assert adapter.name == "openai"

    def test_close(self):
        adapter = OpenAIAdapter(api_key="test")
        adapter.close()  # Should not raise

    def test_org_id_header(self):
        """Organization ID should be set as a header."""
        captured_headers: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured_headers.update(dict(request.headers))
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler, org_id="org-123")
        # Re-create client with org_id header for the mock
        adapter._client = httpx.Client(
            transport=httpx.MockTransport(handler),
            base_url="https://api.openai.com",
            headers={
                "authorization": "Bearer test-key",
                "content-type": "application/json",
                "openai-organization": "org-123",
            },
            timeout=httpx.Timeout(120.0),
        )
        adapter.complete(Request(model="gpt-4o", messages=(Message.user("Hi"),)))
        assert captured_headers.get("openai-organization") == "org-123"
