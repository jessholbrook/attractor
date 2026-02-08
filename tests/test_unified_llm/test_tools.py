"""Tests for unified_llm.types.tools."""
from __future__ import annotations

from unified_llm.types.enums import ToolChoiceMode
from unified_llm.types.tools import Tool, ToolCall, ToolChoice, ToolResult


class TestTool:
    """Test Tool dataclass."""

    def test_tool_basic(self) -> None:
        tool = Tool(name="search", description="Search the web")
        assert tool.name == "search"
        assert tool.description == "Search the web"
        assert tool.parameters == {}
        assert tool.execute is None

    def test_tool_with_parameters(self) -> None:
        params = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
        }
        tool = Tool(name="search", description="Search", parameters=params)
        assert tool.parameters["type"] == "object"

    def test_tool_with_execute(self) -> None:
        def my_fn(x: int) -> int:
            return x * 2

        tool = Tool(name="double", description="Double a number", execute=my_fn)
        assert tool.execute is not None
        assert tool.execute(5) == 10

    def test_tool_is_frozen(self) -> None:
        tool = Tool(name="t", description="d")
        try:
            tool.name = "other"  # type: ignore[misc]
            assert False, "Expected FrozenInstanceError"
        except AttributeError:
            pass

    def test_tool_equality_ignores_execute(self) -> None:
        """Two tools with same name/desc/params but different execute should be equal."""
        fn_a = lambda: "a"
        fn_b = lambda: "b"
        t1 = Tool(name="t", description="d", execute=fn_a)
        t2 = Tool(name="t", description="d", execute=fn_b)
        assert t1 == t2

    def test_tool_hash_ignores_execute(self) -> None:
        fn = lambda: None
        t1 = Tool(name="t", description="d")
        t2 = Tool(name="t", description="d", execute=fn)
        assert hash(t1) == hash(t2)


class TestToolChoice:
    """Test ToolChoice dataclass."""

    def test_default_mode_is_auto(self) -> None:
        tc = ToolChoice()
        assert tc.mode == ToolChoiceMode.AUTO
        assert tc.tool_name is None

    def test_none_mode(self) -> None:
        tc = ToolChoice(mode=ToolChoiceMode.NONE)
        assert tc.mode == ToolChoiceMode.NONE

    def test_named_mode_with_tool_name(self) -> None:
        tc = ToolChoice(mode=ToolChoiceMode.NAMED, tool_name="search")
        assert tc.mode == ToolChoiceMode.NAMED
        assert tc.tool_name == "search"

    def test_required_mode(self) -> None:
        tc = ToolChoice(mode=ToolChoiceMode.REQUIRED)
        assert tc.mode == ToolChoiceMode.REQUIRED


class TestToolCall:
    """Test ToolCall dataclass."""

    def test_tool_call_basic(self) -> None:
        tc = ToolCall(id="tc1", name="search")
        assert tc.id == "tc1"
        assert tc.name == "search"
        assert tc.arguments == {}
        assert tc.raw_arguments is None

    def test_tool_call_with_arguments(self) -> None:
        tc = ToolCall(id="tc1", name="search", arguments={"q": "test"})
        assert tc.arguments == {"q": "test"}

    def test_tool_call_with_raw_arguments(self) -> None:
        tc = ToolCall(
            id="tc1", name="search",
            arguments={"q": "test"},
            raw_arguments='{"q": "test"}',
        )
        assert tc.raw_arguments == '{"q": "test"}'


class TestToolResult:
    """Test ToolResult dataclass."""

    def test_tool_result_defaults(self) -> None:
        tr = ToolResult(tool_call_id="tc1")
        assert tr.tool_call_id == "tc1"
        assert tr.content == ""
        assert tr.is_error is False

    def test_tool_result_with_string_content(self) -> None:
        tr = ToolResult(tool_call_id="tc1", content="42")
        assert tr.content == "42"

    def test_tool_result_with_dict_content(self) -> None:
        tr = ToolResult(tool_call_id="tc1", content={"answer": 42})
        assert tr.content == {"answer": 42}

    def test_tool_result_with_list_content(self) -> None:
        tr = ToolResult(tool_call_id="tc1", content=[1, 2, 3])
        assert tr.content == [1, 2, 3]

    def test_tool_result_is_error(self) -> None:
        tr = ToolResult(tool_call_id="tc1", content="boom", is_error=True)
        assert tr.is_error is True
