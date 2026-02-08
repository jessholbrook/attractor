"""Tests for LLM client protocol and stub implementation."""

from __future__ import annotations

from agent_loop.client import (
    Client,
    CompletionRequest,
    CompletionResponse,
    Message,
    StubClient,
)
from agent_loop.turns import Role, ToolCall


# ---------------------------------------------------------------------------
# Message factory methods
# ---------------------------------------------------------------------------


class TestMessageSystem:
    def test_creates_system_role_message(self):
        msg = Message.system("You are a coding assistant.")
        assert msg.role == Role.SYSTEM
        assert msg.content == "You are a coding assistant."

    def test_system_message_has_no_tool_fields(self):
        msg = Message.system("system prompt")
        assert msg.tool_calls is None
        assert msg.tool_call_id is None
        assert msg.name is None


class TestMessageUser:
    def test_creates_user_role_message(self):
        msg = Message.user("Hello, world!")
        assert msg.role == Role.USER
        assert msg.content == "Hello, world!"

    def test_user_message_has_no_tool_fields(self):
        msg = Message.user("question")
        assert msg.tool_calls is None
        assert msg.tool_call_id is None


class TestMessageAssistant:
    def test_creates_assistant_role_message(self):
        msg = Message.assistant("I can help with that.")
        assert msg.role == Role.ASSISTANT
        assert msg.content == "I can help with that."

    def test_assistant_message_with_no_tool_calls(self):
        msg = Message.assistant("plain text")
        assert msg.tool_calls is None

    def test_assistant_message_with_tool_calls(self):
        tc = ToolCall(id="call_1", name="bash", arguments={"cmd": "ls"})
        msg = Message.assistant("Running command", tool_calls=[tc])
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "bash"

    def test_assistant_message_defaults_to_empty_content(self):
        msg = Message.assistant()
        assert msg.content == ""


class TestMessageTool:
    def test_creates_tool_role_message(self):
        msg = Message.tool(tool_call_id="call_1", content="file contents", name="read_file")
        assert msg.role == Role.TOOL
        assert msg.content == "file contents"
        assert msg.tool_call_id == "call_1"
        assert msg.name == "read_file"

    def test_tool_message_name_defaults_to_empty(self):
        msg = Message.tool(tool_call_id="call_2", content="ok")
        assert msg.name == ""


# ---------------------------------------------------------------------------
# CompletionRequest
# ---------------------------------------------------------------------------


class TestCompletionRequest:
    def test_construction_with_all_fields(self):
        msgs = [Message.system("sys"), Message.user("hi")]
        tools = [{"type": "function", "function": {"name": "bash"}}]
        req = CompletionRequest(
            messages=msgs,
            model="claude-opus-4-20250514",
            tools=tools,
            system="override system",
            temperature=0.7,
            max_tokens=4096,
            reasoning_effort="high",
            provider_options={"stream": True},
        )
        assert req.messages == msgs
        assert req.model == "claude-opus-4-20250514"
        assert len(req.tools) == 1
        assert req.system == "override system"
        assert req.temperature == 0.7
        assert req.max_tokens == 4096
        assert req.reasoning_effort == "high"
        assert req.provider_options == {"stream": True}

    def test_construction_with_defaults(self):
        req = CompletionRequest(messages=[Message.user("hello")])
        assert req.model == ""
        assert req.tools == []
        assert req.system is None
        assert req.temperature == 0.0
        assert req.max_tokens is None
        assert req.reasoning_effort is None
        assert req.provider_options == {}


# ---------------------------------------------------------------------------
# CompletionResponse
# ---------------------------------------------------------------------------


class TestCompletionResponse:
    def test_text_returns_message_content(self):
        resp = CompletionResponse(message=Message.assistant("answer"))
        assert resp.text == "answer"

    def test_tool_calls_returns_empty_list_when_no_tool_calls(self):
        resp = CompletionResponse(message=Message.assistant("no tools"))
        assert resp.tool_calls == []

    def test_tool_calls_returns_tool_calls_from_message(self):
        tc1 = ToolCall(id="call_1", name="bash", arguments={"cmd": "ls"})
        tc2 = ToolCall(id="call_2", name="read_file", arguments={"path": "/tmp"})
        resp = CompletionResponse(message=Message.assistant("", tool_calls=[tc1, tc2]))
        assert len(resp.tool_calls) == 2
        assert resp.tool_calls[0].id == "call_1"
        assert resp.tool_calls[1].id == "call_2"

    def test_default_fields(self):
        resp = CompletionResponse(message=Message.assistant("hi"))
        assert resp.usage == {}
        assert resp.model == ""
        assert resp.stop_reason == "end_turn"

    def test_custom_stop_reason_and_usage(self):
        resp = CompletionResponse(
            message=Message.assistant("done"),
            usage={"input_tokens": 100, "output_tokens": 50},
            model="test-model",
            stop_reason="tool_use",
        )
        assert resp.stop_reason == "tool_use"
        assert resp.usage["input_tokens"] == 100
        assert resp.model == "test-model"


# ---------------------------------------------------------------------------
# StubClient
# ---------------------------------------------------------------------------


class TestStubClient:
    def test_returns_first_response_on_first_call(self):
        r1 = CompletionResponse(message=Message.assistant("first"))
        r2 = CompletionResponse(message=Message.assistant("second"))
        client = StubClient(responses=[r1, r2])
        req = CompletionRequest(messages=[Message.user("hi")])
        resp = client.complete(req)
        assert resp.text == "first"

    def test_returns_second_response_on_second_call(self):
        r1 = CompletionResponse(message=Message.assistant("first"))
        r2 = CompletionResponse(message=Message.assistant("second"))
        client = StubClient(responses=[r1, r2])
        req = CompletionRequest(messages=[Message.user("hi")])
        client.complete(req)
        resp = client.complete(req)
        assert resp.text == "second"

    def test_cycles_last_response_when_exhausted(self):
        r1 = CompletionResponse(message=Message.assistant("only"))
        client = StubClient(responses=[r1])
        req = CompletionRequest(messages=[Message.user("hi")])
        client.complete(req)
        resp2 = client.complete(req)
        resp3 = client.complete(req)
        assert resp2.text == "only"
        assert resp3.text == "only"

    def test_default_response_when_no_responses_provided(self):
        client = StubClient()
        req = CompletionRequest(messages=[Message.user("hi")])
        resp = client.complete(req)
        assert resp.text == "Hello! How can I help?"

    def test_tracks_call_count(self):
        client = StubClient()
        req = CompletionRequest(messages=[Message.user("hi")])
        assert client.call_count == 0
        client.complete(req)
        assert client.call_count == 1
        client.complete(req)
        assert client.call_count == 2

    def test_tracks_requests_list(self):
        client = StubClient()
        req1 = CompletionRequest(messages=[Message.user("first")])
        req2 = CompletionRequest(messages=[Message.user("second")])
        client.complete(req1)
        client.complete(req2)
        assert len(client.requests) == 2
        assert client.requests[0].messages[0].content == "first"
        assert client.requests[1].messages[0].content == "second"

    def test_requests_returns_copy(self):
        client = StubClient()
        req = CompletionRequest(messages=[Message.user("hi")])
        client.complete(req)
        reqs = client.requests
        reqs.clear()
        assert client.call_count == 1  # internal list unaffected

    def test_with_tool_call_response(self):
        tc = ToolCall(id="call_abc", name="bash", arguments={"cmd": "pwd"})
        resp = CompletionResponse(
            message=Message.assistant("", tool_calls=[tc]),
            stop_reason="tool_use",
        )
        client = StubClient(responses=[resp])
        req = CompletionRequest(messages=[Message.user("run pwd")])
        result = client.complete(req)
        assert result.stop_reason == "tool_use"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "bash"


# ---------------------------------------------------------------------------
# Client protocol
# ---------------------------------------------------------------------------


class TestClientProtocol:
    def test_stub_client_satisfies_protocol(self):
        """StubClient structurally matches the Client protocol."""
        client: Client = StubClient()
        req = CompletionRequest(messages=[Message.user("hi")])
        resp = client.complete(req)
        assert isinstance(resp, CompletionResponse)
