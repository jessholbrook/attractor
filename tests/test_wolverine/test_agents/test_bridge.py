"""Tests for the UnifiedLLMBridge."""
from __future__ import annotations

import pytest

from agent_loop.client import CompletionRequest, CompletionResponse, Message as AgentMessage
from agent_loop.turns import Role as AgentRole, ToolCall as AgentToolCall
from unified_llm import (
    Client as ULMClient,
    FinishReason,
    FinishReasonInfo,
    Message as ULMMessage,
    Response as ULMResponse,
    StubAdapter,
    ToolCallData,
    Usage,
)

from wolverine.agents.bridge import UnifiedLLMBridge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stub_client(text: str, **kwargs) -> ULMClient:
    """Create a unified_llm client backed by a StubAdapter with a single text response."""
    resp = ULMResponse(
        message=ULMMessage.assistant(text),
        finish_reason=FinishReasonInfo(reason=FinishReason.STOP),
        usage=Usage(input_tokens=10, output_tokens=20),
        **kwargs,
    )
    adapter = StubAdapter(responses=[resp])
    return ULMClient(providers={"stub": adapter}, default_provider="stub")


def _make_stub_client_with_tool_calls(tool_calls: list[ToolCallData]) -> ULMClient:
    """Create a stub client whose response includes tool calls."""
    resp = ULMResponse(
        message=ULMMessage.assistant("", tool_calls=tool_calls),
        finish_reason=FinishReasonInfo(reason=FinishReason.TOOL_CALLS),
        usage=Usage(input_tokens=5, output_tokens=15),
    )
    adapter = StubAdapter(responses=[resp])
    return ULMClient(providers={"stub": adapter}, default_provider="stub")


def _simple_request(*messages: AgentMessage) -> CompletionRequest:
    """Build a simple CompletionRequest from messages."""
    return CompletionRequest(messages=list(messages), model="stub-model")


# ---------------------------------------------------------------------------
# Message translation tests
# ---------------------------------------------------------------------------

class TestTranslateMessage:
    """Tests for _translate_message (exercised through complete())."""

    def test_translate_user_message(self):
        bridge = UnifiedLLMBridge(_make_stub_client("ok"))
        msg = AgentMessage.user("hello")
        ulm_msg = bridge._translate_message(msg)
        assert ulm_msg.text == "hello"
        assert ulm_msg.role.value == "user"

    def test_translate_assistant_message(self):
        bridge = UnifiedLLMBridge(_make_stub_client("ok"))
        msg = AgentMessage.assistant("world")
        ulm_msg = bridge._translate_message(msg)
        assert ulm_msg.text == "world"
        assert ulm_msg.role.value == "assistant"

    def test_translate_system_message(self):
        bridge = UnifiedLLMBridge(_make_stub_client("ok"))
        msg = AgentMessage.system("you are helpful")
        ulm_msg = bridge._translate_message(msg)
        assert ulm_msg.text == "you are helpful"
        assert ulm_msg.role.value == "system"

    def test_translate_tool_message(self):
        bridge = UnifiedLLMBridge(_make_stub_client("ok"))
        msg = AgentMessage.tool(tool_call_id="tc-1", content="result data")
        ulm_msg = bridge._translate_message(msg)
        assert ulm_msg.role.value == "tool"
        assert ulm_msg.tool_call_id == "tc-1"

    def test_translate_assistant_with_tool_calls(self):
        bridge = UnifiedLLMBridge(_make_stub_client("ok"))
        tc = AgentToolCall(id="tc-1", name="read_file", arguments={"file_path": "/foo"})
        msg = AgentMessage.assistant("reading", tool_calls=[tc])
        ulm_msg = bridge._translate_message(msg)
        assert ulm_msg.role.value == "assistant"
        # The message should include tool call content parts
        assert len(ulm_msg.content) == 2  # text + tool_call

    def test_translate_tool_message_with_empty_tool_call_id(self):
        bridge = UnifiedLLMBridge(_make_stub_client("ok"))
        msg = AgentMessage(role=AgentRole.TOOL, content="data", tool_call_id=None)
        ulm_msg = bridge._translate_message(msg)
        assert ulm_msg.tool_call_id == ""

    def test_translate_empty_content(self):
        bridge = UnifiedLLMBridge(_make_stub_client("ok"))
        msg = AgentMessage.user("")
        ulm_msg = bridge._translate_message(msg)
        assert ulm_msg.text == ""


# ---------------------------------------------------------------------------
# Complete + response translation tests
# ---------------------------------------------------------------------------

class TestComplete:
    """Tests for the complete() method end-to-end."""

    def test_complete_returns_completion_response(self):
        bridge = UnifiedLLMBridge(_make_stub_client("Hello back"))
        request = _simple_request(AgentMessage.user("Hi"))
        result = bridge.complete(request)
        assert isinstance(result, CompletionResponse)

    def test_complete_text_round_trip(self):
        bridge = UnifiedLLMBridge(_make_stub_client("42"))
        request = _simple_request(AgentMessage.user("What is 6*7?"))
        result = bridge.complete(request)
        assert result.text == "42"

    def test_complete_usage_is_populated(self):
        bridge = UnifiedLLMBridge(_make_stub_client("ok"))
        request = _simple_request(AgentMessage.user("test"))
        result = bridge.complete(request)
        assert result.usage["input_tokens"] == 10
        assert result.usage["output_tokens"] == 20

    def test_complete_model_passthrough(self):
        resp = ULMResponse(
            message=ULMMessage.assistant("ok"),
            model="gpt-4o",
            finish_reason=FinishReasonInfo(reason=FinishReason.STOP),
            usage=Usage(input_tokens=1, output_tokens=1),
        )
        adapter = StubAdapter(responses=[resp])
        client = ULMClient(providers={"stub": adapter}, default_provider="stub")
        bridge = UnifiedLLMBridge(client)
        result = bridge.complete(_simple_request(AgentMessage.user("x")))
        assert result.model == "gpt-4o"

    def test_complete_with_tool_calls(self):
        tc = ToolCallData(id="tc-1", name="read_file", arguments={"file_path": "/a.py"})
        bridge = UnifiedLLMBridge(_make_stub_client_with_tool_calls([tc]))
        request = _simple_request(AgentMessage.user("read the file"))
        result = bridge.complete(request)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "tc-1"
        assert result.tool_calls[0].name == "read_file"
        assert result.tool_calls[0].arguments == {"file_path": "/a.py"}

    def test_complete_stop_reason_end_turn(self):
        bridge = UnifiedLLMBridge(_make_stub_client("done"))
        result = bridge.complete(_simple_request(AgentMessage.user("x")))
        assert result.stop_reason == "end_turn"

    def test_complete_stop_reason_tool_use(self):
        tc = ToolCallData(id="tc-1", name="shell", arguments={"command": "ls"})
        bridge = UnifiedLLMBridge(_make_stub_client_with_tool_calls([tc]))
        result = bridge.complete(_simple_request(AgentMessage.user("list files")))
        assert result.stop_reason == "tool_use"

    def test_complete_stop_reason_max_tokens(self):
        resp = ULMResponse(
            message=ULMMessage.assistant("truncated"),
            finish_reason=FinishReasonInfo(reason=FinishReason.LENGTH),
            usage=Usage(input_tokens=1, output_tokens=1),
        )
        adapter = StubAdapter(responses=[resp])
        client = ULMClient(providers={"stub": adapter}, default_provider="stub")
        bridge = UnifiedLLMBridge(client)
        result = bridge.complete(_simple_request(AgentMessage.user("x")))
        assert result.stop_reason == "max_tokens"

    def test_complete_with_tools_in_request(self):
        """Verify tool definitions from CompletionRequest are forwarded."""
        adapter = StubAdapter(responses=[
            ULMResponse(
                message=ULMMessage.assistant("ok"),
                finish_reason=FinishReasonInfo(reason=FinishReason.STOP),
                usage=Usage(input_tokens=1, output_tokens=1),
            )
        ])
        client = ULMClient(providers={"stub": adapter}, default_provider="stub")
        bridge = UnifiedLLMBridge(client)

        request = CompletionRequest(
            messages=[AgentMessage.user("test")],
            model="stub-model",
            tools=[{
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {}},
                },
            }],
        )
        result = bridge.complete(request)
        assert isinstance(result, CompletionResponse)

        # Verify the adapter received a request with tools
        assert len(adapter.requests) == 1
        assert adapter.requests[0].tools is not None
        assert len(adapter.requests[0].tools) == 1
        assert adapter.requests[0].tools[0].name == "read_file"

    def test_complete_multiple_tool_calls(self):
        tcs = [
            ToolCallData(id="tc-1", name="read_file", arguments={"file_path": "/a.py"}),
            ToolCallData(id="tc-2", name="write_file", arguments={"file_path": "/b.py", "content": "x"}),
        ]
        bridge = UnifiedLLMBridge(_make_stub_client_with_tool_calls(tcs))
        result = bridge.complete(_simple_request(AgentMessage.user("do stuff")))
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].name == "read_file"
        assert result.tool_calls[1].name == "write_file"

    def test_complete_no_tool_calls_returns_empty_list(self):
        bridge = UnifiedLLMBridge(_make_stub_client("plain text"))
        result = bridge.complete(_simple_request(AgentMessage.user("hi")))
        assert result.tool_calls == []

    def test_complete_with_string_arguments_falls_back_to_empty_dict(self):
        """When ToolCallData.arguments is a string, the bridge should produce an empty dict."""
        tc = ToolCallData(id="tc-1", name="shell", arguments='{"command": "ls"}')
        bridge = UnifiedLLMBridge(_make_stub_client_with_tool_calls([tc]))
        result = bridge.complete(_simple_request(AgentMessage.user("run")))
        # String arguments can't be used as dict, so bridge falls back to {}
        assert result.tool_calls[0].arguments == {}
