"""Tests for the Anthropic provider adapter."""
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
from unified_llm.providers.anthropic import AnthropicAdapter
from unified_llm.types.content import (
    CacheControl,
    ContentPart,
    ImageData,
    ThinkingData,
    ToolCallData,
    ToolResultData,
)
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

def _make_adapter(handler) -> AnthropicAdapter:
    """Create an adapter wired to a mock transport."""
    adapter = AnthropicAdapter(api_key="test-key")
    adapter._client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://api.anthropic.com",
        headers={
            "x-api-key": "test-key",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        timeout=httpx.Timeout(120.0),
    )
    return adapter


def _simple_response(**overrides: Any) -> dict[str, Any]:
    base = {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello!"}],
        "model": "claude-opus-4-6",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    base.update(overrides)
    return base


def _sse_lines(events: list[tuple[str, dict[str, Any]]]) -> str:
    """Build SSE text from a list of (event_type, data) tuples."""
    lines: list[str] = []
    for event_type, data in events:
        lines.append(f"event: {event_type}")
        lines.append(f"data: {json.dumps(data)}")
        lines.append("")
    return "\n".join(lines)


# ===========================================================================
# Request Translation Tests
# ===========================================================================


class TestAnthropicRequestTranslation:
    """Tests for _build_request_body."""

    def test_system_message_extraction(self):
        """System messages should be extracted into the system parameter."""
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="claude-opus-4-6",
            messages=(
                Message.system("You are helpful."),
                Message.user("Hi"),
            ),
        )
        adapter.complete(req)

        body = captured["body"]
        assert "system" in body
        assert body["system"][0]["text"] == "You are helpful."
        # Verify system block has cache_control
        assert body["system"][-1]["cache_control"] == {"type": "ephemeral"}
        # System messages should not appear in messages
        for msg in body["messages"]:
            assert msg["role"] != "system"

    def test_developer_messages_merge_with_system(self):
        """Developer messages should be merged into the system parameter."""
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="claude-opus-4-6",
            messages=(
                Message.system("System instruction."),
                Message.developer("Developer instruction."),
                Message.user("Hi"),
            ),
        )
        adapter.complete(req)

        body = captured["body"]
        system_texts = [b["text"] for b in body["system"]]
        assert "System instruction." in system_texts
        assert "Developer instruction." in system_texts

    def test_message_alternation_merges_consecutive_roles(self):
        """Consecutive same-role messages should be merged."""
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="claude-opus-4-6",
            messages=(
                Message.user("First"),
                Message.user("Second"),
                Message(role=Role.ASSISTANT, content=(ContentPart.of_text("Reply"),)),
            ),
        )
        adapter.complete(req)

        body = captured["body"]
        # Should be two messages after merging
        assert len(body["messages"]) == 2
        assert body["messages"][0]["role"] == "user"
        # Content should have both parts
        assert len(body["messages"][0]["content"]) == 2

    def test_text_content_translation(self):
        """TEXT content parts should translate to text blocks."""
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="claude-opus-4-6",
            messages=(Message.user("Hello"),),
        )
        adapter.complete(req)

        content = captured["body"]["messages"][0]["content"]
        assert content[0] == {"type": "text", "text": "Hello", "cache_control": {"type": "ephemeral"}}

    def test_image_url_content_translation(self):
        """IMAGE content with URL should translate to image URL block."""
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="claude-opus-4-6",
            messages=(
                Message(
                    role=Role.USER,
                    content=(ContentPart.image_url("https://example.com/img.png"),),
                ),
            ),
        )
        adapter.complete(req)

        content = captured["body"]["messages"][0]["content"]
        img_block = content[0]
        assert img_block["type"] == "image"
        assert img_block["source"]["type"] == "url"
        assert img_block["source"]["url"] == "https://example.com/img.png"

    def test_image_base64_content_translation(self):
        """IMAGE content with base64 data should translate to base64 block."""
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="claude-opus-4-6",
            messages=(
                Message(
                    role=Role.USER,
                    content=(
                        ContentPart.image_base64(
                            data=b"fakedata", media_type="image/png"
                        ),
                    ),
                ),
            ),
        )
        adapter.complete(req)

        content = captured["body"]["messages"][0]["content"]
        img_block = content[0]
        assert img_block["type"] == "image"
        assert img_block["source"]["type"] == "base64"
        assert img_block["source"]["media_type"] == "image/png"

    def test_tool_call_content_translation(self):
        """TOOL_CALL content should translate to tool_use blocks."""
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="claude-opus-4-6",
            messages=(
                Message.user("Use a tool"),
                Message.assistant(
                    tool_calls=[
                        ToolCallData(id="tc_1", name="get_weather", arguments={"city": "NYC"})
                    ]
                ),
                Message.tool_result("tc_1", "Sunny"),
            ),
        )
        adapter.complete(req)

        body = captured["body"]
        # Find assistant message
        assistant_msg = [m for m in body["messages"] if m["role"] == "assistant"][0]
        tool_use_block = [
            b for b in assistant_msg["content"] if b.get("type") == "tool_use"
        ][0]
        assert tool_use_block["id"] == "tc_1"
        assert tool_use_block["name"] == "get_weather"
        assert tool_use_block["input"] == {"city": "NYC"}

    def test_tool_definition_format(self):
        """Tools should be formatted with name, description, input_schema."""
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
            model="claude-opus-4-6",
            messages=(Message.user("Search"),),
            tools=(tool,),
        )
        adapter.complete(req)

        tools = captured["body"]["tools"]
        assert len(tools) == 1
        assert tools[0]["name"] == "search"
        assert tools[0]["description"] == "Search the web"
        assert tools[0]["input_schema"] == tool.parameters

    def test_tool_choice_auto(self):
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="claude-opus-4-6",
            messages=(Message.user("Hi"),),
            tools=(Tool(name="t", description="d"),),
            tool_choice=ToolChoice(mode=ToolChoiceMode.AUTO),
        )
        adapter.complete(req)
        assert captured["body"]["tool_choice"] == {"type": "auto"}

    def test_tool_choice_required(self):
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="claude-opus-4-6",
            messages=(Message.user("Hi"),),
            tools=(Tool(name="t", description="d"),),
            tool_choice=ToolChoice(mode=ToolChoiceMode.REQUIRED),
        )
        adapter.complete(req)
        assert captured["body"]["tool_choice"] == {"type": "any"}

    def test_tool_choice_named(self):
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="claude-opus-4-6",
            messages=(Message.user("Hi"),),
            tools=(Tool(name="search", description="d"),),
            tool_choice=ToolChoice(mode=ToolChoiceMode.NAMED, tool_name="search"),
        )
        adapter.complete(req)
        assert captured["body"]["tool_choice"] == {"type": "tool", "name": "search"}

    def test_tool_choice_none_omits_tools(self):
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="claude-opus-4-6",
            messages=(Message.user("Hi"),),
            tools=(Tool(name="t", description="d"),),
            tool_choice=ToolChoice(mode=ToolChoiceMode.NONE),
        )
        adapter.complete(req)
        assert "tools" not in captured["body"]
        assert "tool_choice" not in captured["body"]

    def test_max_tokens_default(self):
        """Default max_tokens should be used if not specified."""
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="claude-opus-4-6",
            messages=(Message.user("Hi"),),
        )
        adapter.complete(req)
        assert captured["body"]["max_tokens"] == 4096

    def test_max_tokens_explicit(self):
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="claude-opus-4-6",
            messages=(Message.user("Hi"),),
            max_tokens=1024,
        )
        adapter.complete(req)
        assert captured["body"]["max_tokens"] == 1024

    def test_cache_control_injection(self):
        """Cache control should be injected on last system and last user content blocks."""
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="claude-opus-4-6",
            messages=(
                Message.system("System text"),
                Message.user("User text"),
            ),
        )
        adapter.complete(req)

        body = captured["body"]
        # Last system block
        assert body["system"][-1]["cache_control"] == {"type": "ephemeral"}
        # Last user message's last content block
        last_user_msg = [m for m in body["messages"] if m["role"] == "user"][-1]
        assert last_user_msg["content"][-1]["cache_control"] == {"type": "ephemeral"}

    def test_beta_headers(self):
        """Beta headers from provider_options should be passed as anthropic-beta header."""
        captured_headers: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured_headers.update(dict(request.headers))
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="claude-opus-4-6",
            messages=(Message.user("Hi"),),
            provider_options={
                "anthropic": {"beta_headers": ["extended-thinking-2025-01-24"]}
            },
        )
        adapter.complete(req)
        assert "anthropic-beta" in captured_headers
        assert "extended-thinking-2025-01-24" in captured_headers["anthropic-beta"]

    def test_provider_options_merged(self):
        """Provider options under 'anthropic' key should be merged into body."""
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="claude-opus-4-6",
            messages=(Message.user("Hi"),),
            provider_options={
                "anthropic": {
                    "thinking": {"type": "enabled", "budget_tokens": 10000}
                }
            },
        )
        adapter.complete(req)
        assert captured["body"]["thinking"] == {
            "type": "enabled",
            "budget_tokens": 10000,
        }

    def test_tool_result_in_user_message(self):
        """TOOL messages should become tool_result blocks inside user messages."""
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="claude-opus-4-6",
            messages=(
                Message.user("Do something"),
                Message.assistant(
                    tool_calls=[ToolCallData(id="tc_1", name="fn", arguments={})]
                ),
                Message.tool_result("tc_1", "result"),
            ),
        )
        adapter.complete(req)

        body = captured["body"]
        # Tool result should be in a user message
        tool_msg = body["messages"][2]
        assert tool_msg["role"] == "user"
        assert tool_msg["content"][0]["type"] == "tool_result"
        assert tool_msg["content"][0]["tool_use_id"] == "tc_1"

    def test_thinking_content_translation(self):
        """THINKING content should translate to thinking blocks."""
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(
            model="claude-opus-4-6",
            messages=(
                Message.user("Think"),
                Message(
                    role=Role.ASSISTANT,
                    content=(
                        ContentPart.of_thinking("My reasoning", signature="sig123"),
                        ContentPart.of_text("My answer"),
                    ),
                ),
                Message.user("Follow up"),
            ),
        )
        adapter.complete(req)

        body = captured["body"]
        assistant_msg = [m for m in body["messages"] if m["role"] == "assistant"][0]
        thinking_block = [
            b for b in assistant_msg["content"] if b.get("type") == "thinking"
        ][0]
        assert thinking_block["thinking"] == "My reasoning"
        assert thinking_block["signature"] == "sig123"


# ===========================================================================
# Response Parsing Tests
# ===========================================================================


class TestAnthropicResponseParsing:

    def test_simple_text_response(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_simple_response())

        adapter = _make_adapter(handler)
        req = Request(model="claude-opus-4-6", messages=(Message.user("Hi"),))
        response = adapter.complete(req)

        assert response.text == "Hello!"
        assert response.finish_reason.reason == FinishReason.STOP
        assert response.id == "msg_123"
        assert response.model == "claude-opus-4-6"
        assert response.provider == "anthropic"

    def test_tool_use_response(self):
        raw = _simple_response(
            content=[
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "get_weather",
                    "input": {"city": "NYC"},
                }
            ],
            stop_reason="tool_use",
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=raw)

        adapter = _make_adapter(handler)
        req = Request(model="claude-opus-4-6", messages=(Message.user("Weather"),))
        response = adapter.complete(req)

        assert response.finish_reason.reason == FinishReason.TOOL_CALLS
        assert len(response.tool_calls) == 1
        tc = response.tool_calls[0]
        assert tc.id == "tu_1"
        assert tc.name == "get_weather"
        assert tc.arguments == {"city": "NYC"}

    def test_thinking_blocks_in_response(self):
        raw = _simple_response(
            content=[
                {"type": "thinking", "thinking": "Let me think...", "signature": "sig1"},
                {"type": "text", "text": "Here is my answer."},
            ]
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=raw)

        adapter = _make_adapter(handler)
        req = Request(model="claude-opus-4-6", messages=(Message.user("Think"),))
        response = adapter.complete(req)

        assert response.reasoning == "Let me think..."
        assert response.text == "Here is my answer."

    def test_redacted_thinking_in_response(self):
        raw = _simple_response(
            content=[
                {"type": "redacted_thinking", "data": "redacted_data"},
                {"type": "text", "text": "Answer."},
            ]
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=raw)

        adapter = _make_adapter(handler)
        req = Request(model="claude-opus-4-6", messages=(Message.user("Think"),))
        response = adapter.complete(req)

        assert response.text == "Answer."
        # Should have a redacted thinking part
        redacted = [
            p for p in response.message.content
            if p.kind == ContentKind.REDACTED_THINKING
        ]
        assert len(redacted) == 1

    def test_usage_extraction(self):
        raw = _simple_response(
            usage={
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_input_tokens": 20,
                "cache_creation_input_tokens": 10,
            }
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=raw)

        adapter = _make_adapter(handler)
        req = Request(model="claude-opus-4-6", messages=(Message.user("Hi"),))
        response = adapter.complete(req)

        assert response.usage.input_tokens == 100
        assert response.usage.output_tokens == 50
        assert response.usage.total_tokens == 150
        assert response.usage.cache_read_tokens == 20
        assert response.usage.cache_write_tokens == 10

    def test_finish_reason_end_turn(self):
        raw = _simple_response(stop_reason="end_turn")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=raw)

        adapter = _make_adapter(handler)
        response = adapter.complete(
            Request(model="m", messages=(Message.user("Hi"),))
        )
        assert response.finish_reason.reason == FinishReason.STOP
        assert response.finish_reason.raw == "end_turn"

    def test_finish_reason_max_tokens(self):
        raw = _simple_response(stop_reason="max_tokens")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=raw)

        adapter = _make_adapter(handler)
        response = adapter.complete(
            Request(model="m", messages=(Message.user("Hi"),))
        )
        assert response.finish_reason.reason == FinishReason.LENGTH

    def test_finish_reason_stop_sequence(self):
        raw = _simple_response(stop_reason="stop_sequence")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=raw)

        adapter = _make_adapter(handler)
        response = adapter.complete(
            Request(model="m", messages=(Message.user("Hi"),))
        )
        assert response.finish_reason.reason == FinishReason.STOP

    def test_finish_reason_tool_use(self):
        raw = _simple_response(
            stop_reason="tool_use",
            content=[{"type": "tool_use", "id": "t1", "name": "f", "input": {}}],
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=raw)

        adapter = _make_adapter(handler)
        response = adapter.complete(
            Request(model="m", messages=(Message.user("Hi"),))
        )
        assert response.finish_reason.reason == FinishReason.TOOL_CALLS

    def test_mixed_content_response(self):
        """Response with text and tool_use blocks."""
        raw = _simple_response(
            content=[
                {"type": "text", "text": "Let me search."},
                {"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "test"}},
            ],
            stop_reason="tool_use",
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=raw)

        adapter = _make_adapter(handler)
        response = adapter.complete(
            Request(model="m", messages=(Message.user("Search"),))
        )
        assert response.text == "Let me search."
        assert len(response.tool_calls) == 1


# ===========================================================================
# Error Handling Tests
# ===========================================================================


class TestAnthropicErrorHandling:

    def test_401_authentication_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                401,
                json={"type": "error", "error": {"type": "authentication_error", "message": "Invalid API key"}},
            )

        adapter = _make_adapter(handler)
        with pytest.raises(AuthenticationError) as exc_info:
            adapter.complete(Request(model="m", messages=(Message.user("Hi"),)))
        assert exc_info.value.status_code == 401
        assert exc_info.value.provider == "anthropic"

    def test_429_rate_limit_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                429,
                json={"type": "error", "error": {"type": "rate_limit_error", "message": "Rate limit exceeded"}},
                headers={"retry-after": "30"},
            )

        adapter = _make_adapter(handler)
        with pytest.raises(RateLimitError) as exc_info:
            adapter.complete(Request(model="m", messages=(Message.user("Hi"),)))
        assert exc_info.value.retryable is True
        assert exc_info.value.retry_after == 30.0

    def test_500_server_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                500,
                json={"type": "error", "error": {"type": "api_error", "message": "Internal server error"}},
            )

        adapter = _make_adapter(handler)
        with pytest.raises(ServerError) as exc_info:
            adapter.complete(Request(model="m", messages=(Message.user("Hi"),)))
        assert exc_info.value.retryable is True

    def test_400_invalid_request_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400,
                json={"type": "error", "error": {"type": "invalid_request_error", "message": "Bad request"}},
            )

        adapter = _make_adapter(handler)
        with pytest.raises(InvalidRequestError):
            adapter.complete(Request(model="m", messages=(Message.user("Hi"),)))

    def test_404_not_found_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                404,
                json={"type": "error", "error": {"type": "not_found_error", "message": "Model not found"}},
            )

        adapter = _make_adapter(handler)
        with pytest.raises(NotFoundError):
            adapter.complete(Request(model="m", messages=(Message.user("Hi"),)))

    def test_timeout_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("Connection timed out")

        adapter = _make_adapter(handler)
        from unified_llm.errors import RequestTimeoutError as RTE
        with pytest.raises(RTE):
            adapter.complete(Request(model="m", messages=(Message.user("Hi"),)))

    def test_network_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused")

        adapter = _make_adapter(handler)
        from unified_llm.errors import NetworkError as NE
        with pytest.raises(NE):
            adapter.complete(Request(model="m", messages=(Message.user("Hi"),)))

    def test_error_code_preserved(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                401,
                json={"type": "error", "error": {"type": "authentication_error", "message": "Bad key"}},
            )

        adapter = _make_adapter(handler)
        with pytest.raises(AuthenticationError) as exc_info:
            adapter.complete(Request(model="m", messages=(Message.user("Hi"),)))
        assert exc_info.value.error_code == "authentication_error"


# ===========================================================================
# Streaming Tests
# ===========================================================================


class TestAnthropicStreaming:

    def test_text_streaming(self):
        """Simple text streaming should produce TEXT_START, TEXT_DELTA, TEXT_END, FINISH."""
        sse_text = _sse_lines([
            ("message_start", {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "model": "claude-opus-4-6",
                    "role": "assistant",
                    "content": [],
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                },
            }),
            ("content_block_start", {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }),
            ("content_block_delta", {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hello"},
            }),
            ("content_block_delta", {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": " world"},
            }),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 5},
            }),
            ("message_stop", {"type": "message_stop"}),
        ])

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=sse_text.encode(),
                headers={"content-type": "text/event-stream"},
            )

        adapter = _make_adapter(handler)
        events = list(adapter.stream(
            Request(model="claude-opus-4-6", messages=(Message.user("Hi"),))
        ))

        event_types = [e.type for e in events]
        assert StreamEventType.STREAM_START in event_types
        assert StreamEventType.TEXT_START in event_types
        assert StreamEventType.TEXT_DELTA in event_types
        assert StreamEventType.TEXT_END in event_types
        assert StreamEventType.FINISH in event_types

        # Check text deltas
        text_deltas = [e.delta for e in events if e.type == StreamEventType.TEXT_DELTA]
        assert "".join(text_deltas) == "Hello world"

        # Check finish event
        finish = [e for e in events if e.type == StreamEventType.FINISH][0]
        assert finish.finish_reason.reason == FinishReason.STOP
        assert finish.usage.output_tokens == 5

    def test_tool_call_streaming(self):
        """Tool call streaming should produce TOOL_CALL_START, TOOL_CALL_DELTA, TOOL_CALL_END."""
        sse_text = _sse_lines([
            ("message_start", {
                "type": "message_start",
                "message": {
                    "id": "msg_2",
                    "model": "claude-opus-4-6",
                    "role": "assistant",
                    "content": [],
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                },
            }),
            ("content_block_start", {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {}},
            }),
            ("content_block_delta", {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"city":'},
            }),
            ("content_block_delta", {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '"NYC"}'},
            }),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 15},
            }),
            ("message_stop", {"type": "message_stop"}),
        ])

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=sse_text.encode(),
                headers={"content-type": "text/event-stream"},
            )

        adapter = _make_adapter(handler)
        events = list(adapter.stream(
            Request(model="claude-opus-4-6", messages=(Message.user("Weather"),))
        ))

        event_types = [e.type for e in events]
        assert StreamEventType.TOOL_CALL_START in event_types
        assert StreamEventType.TOOL_CALL_DELTA in event_types
        assert StreamEventType.TOOL_CALL_END in event_types

        # Check tool call end has assembled arguments
        tc_end = [e for e in events if e.type == StreamEventType.TOOL_CALL_END][0]
        assert tc_end.tool_call.name == "get_weather"
        assert tc_end.tool_call.arguments == {"city": "NYC"}

    def test_thinking_streaming(self):
        """Thinking streaming should produce REASONING_START, REASONING_DELTA, REASONING_END."""
        sse_text = _sse_lines([
            ("message_start", {
                "type": "message_start",
                "message": {
                    "id": "msg_3",
                    "model": "claude-opus-4-6",
                    "role": "assistant",
                    "content": [],
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                },
            }),
            ("content_block_start", {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": ""},
            }),
            ("content_block_delta", {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "Let me think"},
            }),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ("content_block_start", {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "text", "text": ""},
            }),
            ("content_block_delta", {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "text_delta", "text": "Answer"},
            }),
            ("content_block_stop", {"type": "content_block_stop", "index": 1}),
            ("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 20},
            }),
            ("message_stop", {"type": "message_stop"}),
        ])

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=sse_text.encode(),
                headers={"content-type": "text/event-stream"},
            )

        adapter = _make_adapter(handler)
        events = list(adapter.stream(
            Request(model="claude-opus-4-6", messages=(Message.user("Think"),))
        ))

        event_types = [e.type for e in events]
        assert StreamEventType.REASONING_START in event_types
        assert StreamEventType.REASONING_DELTA in event_types
        assert StreamEventType.REASONING_END in event_types

        # Check reasoning delta
        reasoning_deltas = [
            e.reasoning_delta for e in events
            if e.type == StreamEventType.REASONING_DELTA
        ]
        assert "".join(d for d in reasoning_deltas if d) == "Let me think"

    def test_finish_event_contains_response(self):
        """The FINISH event should contain a response with usage and model info."""
        sse_text = _sse_lines([
            ("message_start", {
                "type": "message_start",
                "message": {
                    "id": "msg_4",
                    "model": "claude-opus-4-6",
                    "role": "assistant",
                    "content": [],
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                },
            }),
            ("content_block_start", {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }),
            ("content_block_delta", {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "OK"},
            }),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 3},
            }),
            ("message_stop", {"type": "message_stop"}),
        ])

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=sse_text.encode(),
                headers={"content-type": "text/event-stream"},
            )

        adapter = _make_adapter(handler)
        events = list(adapter.stream(
            Request(model="claude-opus-4-6", messages=(Message.user("Hi"),))
        ))

        finish = [e for e in events if e.type == StreamEventType.FINISH][0]
        assert finish.response is not None
        assert finish.response.id == "msg_4"
        assert finish.response.model == "claude-opus-4-6"
        assert finish.response.provider == "anthropic"

    def test_stream_request_has_stream_true(self):
        """The stream request body should have stream=true."""
        captured: dict[str, Any] = {}

        sse_text = _sse_lines([
            ("message_start", {
                "type": "message_start",
                "message": {
                    "id": "msg_5",
                    "model": "claude-opus-4-6",
                    "role": "assistant",
                    "content": [],
                    "usage": {"input_tokens": 5, "output_tokens": 0},
                },
            }),
            ("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 0},
            }),
            ("message_stop", {"type": "message_stop"}),
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
            Request(model="claude-opus-4-6", messages=(Message.user("Hi"),))
        ))
        assert captured["body"]["stream"] is True

    def test_stream_error_event(self):
        """Error events in the stream should produce ERROR events."""
        sse_text = _sse_lines([
            ("message_start", {
                "type": "message_start",
                "message": {
                    "id": "msg_6",
                    "model": "claude-opus-4-6",
                    "role": "assistant",
                    "content": [],
                    "usage": {"input_tokens": 5, "output_tokens": 0},
                },
            }),
            ("error", {
                "type": "error",
                "error": {"type": "overloaded_error", "message": "Overloaded"},
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
            Request(model="claude-opus-4-6", messages=(Message.user("Hi"),))
        ))

        error_events = [e for e in events if e.type == StreamEventType.ERROR]
        assert len(error_events) == 1

    def test_stream_http_error(self):
        """HTTP errors during streaming should raise appropriate exceptions."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                429,
                json={"type": "error", "error": {"type": "rate_limit_error", "message": "Rate limited"}},
            )

        adapter = _make_adapter(handler)
        with pytest.raises(RateLimitError):
            list(adapter.stream(
                Request(model="m", messages=(Message.user("Hi"),))
            ))

    def test_multiple_text_blocks_streaming(self):
        """Multiple text blocks should each get TEXT_START/TEXT_END events."""
        sse_text = _sse_lines([
            ("message_start", {
                "type": "message_start",
                "message": {
                    "id": "msg_7",
                    "model": "claude-opus-4-6",
                    "role": "assistant",
                    "content": [],
                    "usage": {"input_tokens": 5, "output_tokens": 0},
                },
            }),
            ("content_block_start", {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }),
            ("content_block_delta", {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "First"},
            }),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ("content_block_start", {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "text", "text": ""},
            }),
            ("content_block_delta", {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "text_delta", "text": "Second"},
            }),
            ("content_block_stop", {"type": "content_block_stop", "index": 1}),
            ("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 10},
            }),
            ("message_stop", {"type": "message_stop"}),
        ])

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=sse_text.encode(),
                headers={"content-type": "text/event-stream"},
            )

        adapter = _make_adapter(handler)
        events = list(adapter.stream(
            Request(model="claude-opus-4-6", messages=(Message.user("Hi"),))
        ))

        text_starts = [e for e in events if e.type == StreamEventType.TEXT_START]
        text_ends = [e for e in events if e.type == StreamEventType.TEXT_END]
        assert len(text_starts) == 2
        assert len(text_ends) == 2

    def test_text_id_in_stream_events(self):
        """Text events should include text_id for multi-block tracking."""
        sse_text = _sse_lines([
            ("message_start", {
                "type": "message_start",
                "message": {
                    "id": "msg_8",
                    "model": "claude-opus-4-6",
                    "role": "assistant",
                    "content": [],
                    "usage": {"input_tokens": 5, "output_tokens": 0},
                },
            }),
            ("content_block_start", {
                "type": "content_block_start",
                "index": 2,
                "content_block": {"type": "text", "text": ""},
            }),
            ("content_block_delta", {
                "type": "content_block_delta",
                "index": 2,
                "delta": {"type": "text_delta", "text": "Hello"},
            }),
            ("content_block_stop", {"type": "content_block_stop", "index": 2}),
            ("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 1},
            }),
            ("message_stop", {"type": "message_stop"}),
        ])

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=sse_text.encode(),
                headers={"content-type": "text/event-stream"},
            )

        adapter = _make_adapter(handler)
        events = list(adapter.stream(
            Request(model="claude-opus-4-6", messages=(Message.user("Hi"),))
        ))

        text_delta = [e for e in events if e.type == StreamEventType.TEXT_DELTA][0]
        assert text_delta.text_id == "2"

    def test_ping_events_ignored(self):
        """Ping events should be silently ignored."""
        sse_text = (
            "event: ping\ndata: {}\n\n"
            + _sse_lines([
                ("message_start", {
                    "type": "message_start",
                    "message": {
                        "id": "msg_9",
                        "model": "claude-opus-4-6",
                        "role": "assistant",
                        "content": [],
                        "usage": {"input_tokens": 5, "output_tokens": 0},
                    },
                }),
                ("message_delta", {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 0},
                }),
                ("message_stop", {"type": "message_stop"}),
            ])
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=sse_text.encode(),
                headers={"content-type": "text/event-stream"},
            )

        adapter = _make_adapter(handler)
        events = list(adapter.stream(
            Request(model="claude-opus-4-6", messages=(Message.user("Hi"),))
        ))

        # Should not have any ping-related events
        event_types = [e.type for e in events]
        assert StreamEventType.STREAM_START in event_types


# ===========================================================================
# Adapter Properties
# ===========================================================================


class TestAnthropicAdapterProperties:

    def test_name(self):
        adapter = AnthropicAdapter(api_key="test")
        assert adapter.name == "anthropic"

    def test_close(self):
        adapter = AnthropicAdapter(api_key="test")
        adapter.close()  # Should not raise
