"""Tests for unified_llm.types.messages."""
from __future__ import annotations

from unified_llm.types.enums import ContentKind, Role
from unified_llm.types.content import ContentPart, ToolCallData, ToolResultData
from unified_llm.types.messages import Message


class TestMessageBasics:
    """Test basic Message construction."""

    def test_message_is_frozen(self) -> None:
        msg = Message(role=Role.USER)
        try:
            msg.role = Role.SYSTEM  # type: ignore[misc]
            assert False, "Expected FrozenInstanceError"
        except AttributeError:
            pass

    def test_default_content_is_empty_tuple(self) -> None:
        msg = Message(role=Role.USER)
        assert msg.content == ()

    def test_default_name_is_none(self) -> None:
        msg = Message(role=Role.USER)
        assert msg.name is None

    def test_default_tool_call_id_is_none(self) -> None:
        msg = Message(role=Role.USER)
        assert msg.tool_call_id is None

    def test_explicit_content(self) -> None:
        part = ContentPart.of_text("hello")
        msg = Message(role=Role.USER, content=(part,))
        assert len(msg.content) == 1
        assert msg.content[0].text == "hello"


class TestSystemFactory:
    """Test Message.system() factory."""

    def test_system_role(self) -> None:
        msg = Message.system("You are helpful.")
        assert msg.role == Role.SYSTEM

    def test_system_single_text_part(self) -> None:
        msg = Message.system("Be concise.")
        assert len(msg.content) == 1
        assert msg.content[0].kind == ContentKind.TEXT
        assert msg.content[0].text == "Be concise."


class TestUserFactory:
    """Test Message.user() factory."""

    def test_user_role(self) -> None:
        msg = Message.user("Hello!")
        assert msg.role == Role.USER

    def test_user_text(self) -> None:
        msg = Message.user("What is 2+2?")
        assert msg.text == "What is 2+2?"


class TestAssistantFactory:
    """Test Message.assistant() factory."""

    def test_assistant_text_only(self) -> None:
        msg = Message.assistant("Sure, I can help.")
        assert msg.role == Role.ASSISTANT
        assert msg.text == "Sure, I can help."
        assert len(msg.content) == 1

    def test_assistant_empty_text(self) -> None:
        msg = Message.assistant()
        assert msg.role == Role.ASSISTANT
        assert msg.content == ()
        assert msg.text == ""

    def test_assistant_with_tool_calls(self) -> None:
        tc = ToolCallData(id="tc1", name="search", arguments={"q": "weather"})
        msg = Message.assistant(tool_calls=[tc])
        assert msg.role == Role.ASSISTANT
        assert len(msg.content) == 1
        assert msg.content[0].kind == ContentKind.TOOL_CALL

    def test_assistant_text_and_tool_calls(self) -> None:
        tc = ToolCallData(id="tc1", name="search", arguments={"q": "test"})
        msg = Message.assistant("Let me search.", tool_calls=[tc])
        assert len(msg.content) == 2
        assert msg.content[0].kind == ContentKind.TEXT
        assert msg.content[1].kind == ContentKind.TOOL_CALL

    def test_assistant_multiple_tool_calls(self) -> None:
        tc1 = ToolCallData(id="tc1", name="search", arguments={"q": "a"})
        tc2 = ToolCallData(id="tc2", name="calc", arguments={"expr": "1+1"})
        msg = Message.assistant(tool_calls=[tc1, tc2])
        assert len(msg.content) == 2
        assert msg.content[0].tool_call is not None
        assert msg.content[0].tool_call.name == "search"
        assert msg.content[1].tool_call is not None
        assert msg.content[1].tool_call.name == "calc"


class TestDeveloperFactory:
    """Test Message.developer() factory."""

    def test_developer_role(self) -> None:
        msg = Message.developer("Internal instruction.")
        assert msg.role == Role.DEVELOPER

    def test_developer_text(self) -> None:
        msg = Message.developer("Do X.")
        assert msg.text == "Do X."


class TestToolResultFactory:
    """Test Message.tool_result() factory."""

    def test_tool_result_role(self) -> None:
        msg = Message.tool_result("tc1", "result data")
        assert msg.role == Role.TOOL

    def test_tool_result_tool_call_id(self) -> None:
        msg = Message.tool_result("tc1", "result data")
        assert msg.tool_call_id == "tc1"

    def test_tool_result_content(self) -> None:
        msg = Message.tool_result("tc1", "42")
        assert len(msg.content) == 1
        assert msg.content[0].kind == ContentKind.TOOL_RESULT
        assert msg.content[0].tool_result is not None
        assert msg.content[0].tool_result.content == "42"

    def test_tool_result_is_error(self) -> None:
        msg = Message.tool_result("tc1", "failed", is_error=True)
        assert msg.content[0].tool_result is not None
        assert msg.content[0].tool_result.is_error is True


class TestTextProperty:
    """Test Message.text property."""

    def test_text_from_single_part(self) -> None:
        msg = Message.user("hello")
        assert msg.text == "hello"

    def test_text_from_multiple_text_parts(self) -> None:
        parts = (
            ContentPart.of_text("Hello "),
            ContentPart.of_text("world!"),
        )
        msg = Message(role=Role.USER, content=parts)
        assert msg.text == "Hello world!"

    def test_text_ignores_non_text_parts(self) -> None:
        parts = (
            ContentPart.of_text("hi"),
            ContentPart.of_tool_call(id="tc1", name="f", arguments={}),
        )
        msg = Message(role=Role.ASSISTANT, content=parts)
        assert msg.text == "hi"

    def test_text_empty_when_no_text_parts(self) -> None:
        parts = (
            ContentPart.of_tool_call(id="tc1", name="f", arguments={}),
        )
        msg = Message(role=Role.ASSISTANT, content=parts)
        assert msg.text == ""

    def test_text_empty_for_default_message(self) -> None:
        msg = Message(role=Role.USER)
        assert msg.text == ""
