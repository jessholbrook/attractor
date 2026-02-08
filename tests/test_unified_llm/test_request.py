"""Tests for unified_llm.types.request."""
from __future__ import annotations

from unified_llm.types.enums import ToolChoiceMode
from unified_llm.types.messages import Message
from unified_llm.types.tools import Tool, ToolChoice
from unified_llm.types.request import Request, ResponseFormat


class TestResponseFormat:
    """Test ResponseFormat dataclass."""

    def test_defaults(self) -> None:
        rf = ResponseFormat()
        assert rf.type == "text"
        assert rf.json_schema is None
        assert rf.strict is False

    def test_json_schema(self) -> None:
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        rf = ResponseFormat(type="json_schema", json_schema=schema, strict=True)
        assert rf.type == "json_schema"
        assert rf.json_schema == schema
        assert rf.strict is True

    def test_frozen(self) -> None:
        rf = ResponseFormat()
        try:
            rf.type = "json"  # type: ignore[misc]
            assert False, "Expected FrozenInstanceError"
        except AttributeError:
            pass


class TestRequest:
    """Test Request dataclass."""

    def test_minimal_request(self) -> None:
        req = Request(model="gpt-4")
        assert req.model == "gpt-4"
        assert req.messages == ()
        assert req.provider is None
        assert req.tools is None
        assert req.tool_choice is None
        assert req.response_format is None
        assert req.temperature is None
        assert req.top_p is None
        assert req.max_tokens is None
        assert req.stop_sequences is None
        assert req.reasoning_effort is None
        assert req.metadata is None
        assert req.provider_options is None

    def test_request_with_messages(self) -> None:
        msgs = (Message.system("Be helpful."), Message.user("Hi"))
        req = Request(model="gpt-4", messages=msgs)
        assert len(req.messages) == 2
        assert req.messages[0].role.value == "system"
        assert req.messages[1].role.value == "user"

    def test_request_with_tools(self) -> None:
        tool = Tool(name="search", description="Search the web")
        req = Request(
            model="gpt-4",
            tools=(tool,),
            tool_choice=ToolChoice(mode=ToolChoiceMode.AUTO),
        )
        assert req.tools is not None
        assert len(req.tools) == 1
        assert req.tool_choice is not None
        assert req.tool_choice.mode == ToolChoiceMode.AUTO

    def test_request_with_all_fields(self) -> None:
        tool = Tool(name="calc", description="Calculate")
        rf = ResponseFormat(type="json_schema", json_schema={"type": "object"}, strict=True)
        req = Request(
            model="claude-3-opus",
            messages=(Message.user("test"),),
            provider="anthropic",
            tools=(tool,),
            tool_choice=ToolChoice(mode=ToolChoiceMode.REQUIRED),
            response_format=rf,
            temperature=0.7,
            top_p=0.9,
            max_tokens=1000,
            stop_sequences=("END",),
            reasoning_effort="high",
            metadata={"user_id": "u1"},
            provider_options={"stream": True},
        )
        assert req.model == "claude-3-opus"
        assert req.provider == "anthropic"
        assert req.temperature == 0.7
        assert req.top_p == 0.9
        assert req.max_tokens == 1000
        assert req.stop_sequences == ("END",)
        assert req.reasoning_effort == "high"
        assert req.metadata == {"user_id": "u1"}
        assert req.provider_options == {"stream": True}
        assert req.response_format is not None
        assert req.response_format.strict is True

    def test_request_is_frozen(self) -> None:
        req = Request(model="gpt-4")
        try:
            req.model = "other"  # type: ignore[misc]
            assert False, "Expected FrozenInstanceError"
        except AttributeError:
            pass

    def test_request_provider_field(self) -> None:
        req = Request(model="gpt-4", provider="openai")
        assert req.provider == "openai"

    def test_request_stop_sequences(self) -> None:
        req = Request(model="m", stop_sequences=("STOP", "END"))
        assert req.stop_sequences == ("STOP", "END")
