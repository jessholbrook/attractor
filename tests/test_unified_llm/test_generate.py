"""Tests for unified_llm.generate — high-level blocking generation."""
from __future__ import annotations

import json
from typing import Any

import pytest

from unified_llm.adapter import StubAdapter
from unified_llm.client import Client
from unified_llm.errors import AbortError, NoObjectGeneratedError
from unified_llm.generate import generate, generate_object, _execute_tools
from unified_llm.types.config import AbortController
from unified_llm.types.content import ContentPart, ToolCallData
from unified_llm.types.enums import ContentKind, FinishReason, Role
from unified_llm.types.messages import Message
from unified_llm.types.request import ResponseFormat
from unified_llm.types.response import FinishReasonInfo, Response, Usage
from unified_llm.types.results import GenerateResult, StepResult
from unified_llm.types.tools import Tool, ToolCall, ToolChoice, ToolResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_text_response(text: str, model: str = "test-model") -> Response:
    """Create a simple text response."""
    return Response(
        id="resp_1",
        model=model,
        provider="stub",
        message=Message.assistant(text),
        finish_reason=FinishReasonInfo(reason=FinishReason.STOP),
        usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
    )


def _make_tool_response(
    tool_name: str,
    tool_id: str = "call_1",
    args: dict[str, Any] | None = None,
    model: str = "test-model",
) -> Response:
    """Create a response containing a single tool call."""
    tc = ToolCallData(id=tool_id, name=tool_name, arguments=args or {})
    msg = Message(
        role=Role.ASSISTANT,
        content=(ContentPart.of_tool_call(tool_id, tool_name, args or {}),),
    )
    return Response(
        id="resp_1",
        model=model,
        provider="stub",
        message=msg,
        finish_reason=FinishReasonInfo(reason=FinishReason.TOOL_CALLS),
        usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
    )


def _make_multi_tool_response(
    calls: list[tuple[str, str, dict[str, Any]]],
) -> Response:
    """Create a response with multiple tool calls.

    *calls* is a list of (tool_name, tool_id, arguments) tuples.
    """
    parts = tuple(
        ContentPart.of_tool_call(tid, name, args)
        for name, tid, args in calls
    )
    msg = Message(role=Role.ASSISTANT, content=parts)
    return Response(
        id="resp_1",
        model="test-model",
        provider="stub",
        message=msg,
        finish_reason=FinishReasonInfo(reason=FinishReason.TOOL_CALLS),
        usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
    )


def _stub_client(*responses: Response) -> Client:
    """Create a Client backed by a StubAdapter with the given responses."""
    adapter = StubAdapter(responses=list(responses))
    client = Client(providers={"stub": adapter}, default_provider="stub")
    return client


def _stub_client_and_adapter(*responses: Response) -> tuple[Client, StubAdapter]:
    """Return both client and adapter for request inspection."""
    adapter = StubAdapter(responses=list(responses))
    client = Client(providers={"stub": adapter}, default_provider="stub")
    return client, adapter


# ===================================================================
# TestGenerate
# ===================================================================


class TestGenerate:
    """Tests for the generate() function."""

    def test_simple_text_with_prompt(self) -> None:
        """Simple text generation using a prompt string."""
        resp = _make_text_response("Hello!")
        client = _stub_client(resp)
        result = generate("test-model", prompt="Hi", client=client)

        assert result.text == "Hello!"
        assert result.finish_reason.reason == FinishReason.STOP
        assert len(result.steps) == 1

    def test_generation_with_messages(self) -> None:
        """Generation using a list of Message objects."""
        resp = _make_text_response("Reply")
        client = _stub_client(resp)
        msgs = [Message.user("Hello")]
        result = generate("test-model", messages=msgs, client=client)

        assert result.text == "Reply"
        assert len(result.steps) == 1

    def test_system_message_prepended(self) -> None:
        """System message is prepended to the conversation."""
        resp = _make_text_response("OK")
        client, adapter = _stub_client_and_adapter(resp)
        generate("test-model", prompt="Hi", system="Be helpful", client=client)

        request = adapter.requests[0]
        assert request.messages[0].role == Role.SYSTEM
        assert request.messages[0].text == "Be helpful"
        assert request.messages[1].role == Role.USER
        assert request.messages[1].text == "Hi"

    def test_both_prompt_and_messages_raises(self) -> None:
        """Providing both prompt and messages raises ValueError."""
        with pytest.raises(ValueError, match="not both"):
            generate(
                "test-model",
                prompt="Hi",
                messages=[Message.user("Hello")],
                client=_stub_client(_make_text_response("x")),
            )

    def test_neither_prompt_nor_messages_raises(self) -> None:
        """Providing neither prompt nor messages raises ValueError."""
        with pytest.raises(ValueError, match="must be provided"):
            generate(
                "test-model",
                client=_stub_client(_make_text_response("x")),
            )

    def test_custom_client_passed_through(self) -> None:
        """A custom client is used instead of the default."""
        resp = _make_text_response("Custom")
        client, adapter = _stub_client_and_adapter(resp)
        result = generate("test-model", prompt="Hi", client=client)

        assert result.text == "Custom"
        assert len(adapter.requests) == 1

    def test_provider_passed_through(self) -> None:
        """Provider name is forwarded in the request."""
        resp = _make_text_response("OK")
        adapter = StubAdapter()
        # Register with the provider name "custom"
        client = Client(providers={"custom": adapter}, default_provider="custom")
        # Also register as default responses
        adapter._responses = [resp]
        generate("test-model", prompt="Hi", provider="custom", client=client)

        assert adapter.requests[0].provider == "custom"

    def test_temperature_and_max_tokens_passed_through(self) -> None:
        """Temperature and max_tokens are included in the request."""
        resp = _make_text_response("OK")
        client, adapter = _stub_client_and_adapter(resp)
        generate(
            "test-model",
            prompt="Hi",
            temperature=0.7,
            max_tokens=100,
            client=client,
        )

        req = adapter.requests[0]
        assert req.temperature == 0.7
        assert req.max_tokens == 100

    def test_result_has_correct_usage(self) -> None:
        """Result carries usage from the response."""
        resp = _make_text_response("OK")
        client = _stub_client(resp)
        result = generate("test-model", prompt="Hi", client=client)

        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5
        assert result.usage.total_tokens == 15

    def test_result_finish_reason(self) -> None:
        """Result carries the correct finish reason."""
        resp = _make_text_response("Done")
        client = _stub_client(resp)
        result = generate("test-model", prompt="Hi", client=client)

        assert result.finish_reason.reason == FinishReason.STOP

    def test_multiple_sequential_calls(self) -> None:
        """Multiple calls to generate() work independently."""
        resp1 = _make_text_response("First")
        resp2 = _make_text_response("Second")
        client = _stub_client(resp1, resp2)

        r1 = generate("test-model", prompt="A", client=client)
        r2 = generate("test-model", prompt="B", client=client)

        assert r1.text == "First"
        assert r2.text == "Second"

    def test_stop_sequences_forwarded(self) -> None:
        """Stop sequences are passed through to the request."""
        resp = _make_text_response("OK")
        client, adapter = _stub_client_and_adapter(resp)
        generate(
            "test-model",
            prompt="Hi",
            stop_sequences=["STOP", "END"],
            client=client,
        )

        assert adapter.requests[0].stop_sequences == ("STOP", "END")

    def test_reasoning_effort_forwarded(self) -> None:
        """Reasoning effort is passed through to the request."""
        resp = _make_text_response("OK")
        client, adapter = _stub_client_and_adapter(resp)
        generate(
            "test-model",
            prompt="Hi",
            reasoning_effort="high",
            client=client,
        )

        assert adapter.requests[0].reasoning_effort == "high"

    def test_top_p_forwarded(self) -> None:
        """top_p is passed through."""
        resp = _make_text_response("OK")
        client, adapter = _stub_client_and_adapter(resp)
        generate("test-model", prompt="Hi", top_p=0.9, client=client)

        assert adapter.requests[0].top_p == 0.9

    def test_abort_signal_raises(self) -> None:
        """AbortError raised when signal is already aborted."""
        ctrl = AbortController()
        ctrl.abort()
        with pytest.raises(AbortError):
            generate(
                "test-model",
                prompt="Hi",
                abort_signal=ctrl.signal,
                client=_stub_client(_make_text_response("x")),
            )


# ===================================================================
# TestGenerateToolLoop
# ===================================================================


class TestGenerateToolLoop:
    """Tests for the tool execution loop in generate()."""

    def _weather_tool(self) -> Tool:
        """Create a simple weather tool with an execute handler."""
        def get_weather(city: str = "NYC") -> str:
            return f"Sunny in {city}"

        return Tool(
            name="get_weather",
            description="Get weather for a city",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}},
            execute=get_weather,
        )

    def test_single_tool_round(self) -> None:
        """Tool call -> execute -> follow-up. Result has 2 steps."""
        tool = self._weather_tool()
        tool_resp = _make_tool_response("get_weather", args={"city": "NYC"})
        text_resp = _make_text_response("It's sunny in NYC!")

        client = _stub_client(tool_resp, text_resp)
        result = generate(
            "test-model",
            prompt="What's the weather?",
            tools=[tool],
            max_tool_rounds=1,
            client=client,
        )

        assert len(result.steps) == 2
        assert result.text == "It's sunny in NYC!"
        assert result.steps[0].tool_calls[0].name == "get_weather"
        assert result.steps[0].tool_results[0].content == "Sunny in NYC"

    def test_max_tool_rounds_zero_no_execution(self) -> None:
        """With max_tool_rounds=0, tool calls are returned without execution."""
        tool = self._weather_tool()
        tool_resp = _make_tool_response("get_weather", args={"city": "LA"})

        client = _stub_client(tool_resp)
        result = generate(
            "test-model",
            prompt="Weather?",
            tools=[tool],
            max_tool_rounds=0,
            client=client,
        )

        assert len(result.steps) == 1
        assert len(result.tool_calls) > 0
        assert len(result.tool_results) == 0

    def test_max_tool_rounds_exceeded(self) -> None:
        """Stops after max_tool_rounds and returns last step."""
        tool = self._weather_tool()
        # Model keeps calling tools
        tool_resp1 = _make_tool_response("get_weather", "c1", {"city": "A"})
        tool_resp2 = _make_tool_response("get_weather", "c2", {"city": "B"})
        # After rounds exhausted, still returns tool call
        tool_resp3 = _make_tool_response("get_weather", "c3", {"city": "C"})

        client = _stub_client(tool_resp1, tool_resp2, tool_resp3)
        result = generate(
            "test-model",
            prompt="Weather?",
            tools=[tool],
            max_tool_rounds=2,
            client=client,
        )

        # 2 tool rounds executed + 1 final call = 3 steps
        assert len(result.steps) == 3

    def test_tool_execution_error(self) -> None:
        """Tool that raises an exception produces is_error=True result."""
        def failing_tool(**kwargs: Any) -> str:
            raise RuntimeError("Tool broke")

        tool = Tool(name="broken", description="breaks", execute=failing_tool)
        tool_resp = _make_tool_response("broken")
        text_resp = _make_text_response("Sorry")

        client = _stub_client(tool_resp, text_resp)
        result = generate(
            "test-model",
            prompt="Run it",
            tools=[tool],
            client=client,
        )

        assert len(result.steps) == 2
        assert result.steps[0].tool_results[0].is_error is True
        assert "Tool broke" in result.steps[0].tool_results[0].content

    def test_unknown_tool_returns_error(self) -> None:
        """A tool call for an unregistered tool returns an error result."""
        tool = self._weather_tool()
        # Model calls a tool that doesn't exist
        tool_resp = _make_tool_response("nonexistent_tool")
        text_resp = _make_text_response("Sorry")

        client = _stub_client(tool_resp, text_resp)
        result = generate(
            "test-model",
            prompt="Run it",
            tools=[tool],
            client=client,
        )

        assert result.steps[0].tool_results[0].is_error is True
        assert "Unknown tool" in result.steps[0].tool_results[0].content

    def test_parallel_tool_execution(self) -> None:
        """Two tool calls in one response are both executed."""
        results_collected: list[str] = []

        def tool_a(x: str = "a") -> str:
            results_collected.append(f"a:{x}")
            return f"result_a_{x}"

        def tool_b(x: str = "b") -> str:
            results_collected.append(f"b:{x}")
            return f"result_b_{x}"

        tools = [
            Tool(name="tool_a", description="A", execute=tool_a),
            Tool(name="tool_b", description="B", execute=tool_b),
        ]

        multi_resp = _make_multi_tool_response([
            ("tool_a", "c1", {"x": "1"}),
            ("tool_b", "c2", {"x": "2"}),
        ])
        text_resp = _make_text_response("Done")

        client = _stub_client(multi_resp, text_resp)
        result = generate(
            "test-model",
            prompt="Do both",
            tools=tools,
            client=client,
        )

        assert len(result.steps) == 2
        step0 = result.steps[0]
        assert len(step0.tool_calls) == 2
        assert len(step0.tool_results) == 2
        # Results should be in the same order as calls
        assert step0.tool_results[0].content == "result_a_1"
        assert step0.tool_results[1].content == "result_b_2"

    def test_tool_results_appended_to_messages(self) -> None:
        """After executing tools, results are sent back in messages."""
        tool = self._weather_tool()
        tool_resp = _make_tool_response("get_weather", args={"city": "NYC"})
        text_resp = _make_text_response("Sunny!")

        client, adapter = _stub_client_and_adapter(tool_resp, text_resp)
        generate(
            "test-model",
            prompt="Weather?",
            tools=[tool],
            client=client,
        )

        # Second request should include tool result messages
        second_req = adapter.requests[1]
        msgs = second_req.messages
        # Should have: user msg, assistant msg with tool call, tool result
        tool_result_msgs = [m for m in msgs if m.role == Role.TOOL]
        assert len(tool_result_msgs) == 1
        assert tool_result_msgs[0].tool_call_id == "call_1"

    def test_active_tool_with_execute_handler(self) -> None:
        """Active tool (with execute handler) is automatically called."""
        call_log: list[str] = []

        def my_fn(q: str = "") -> str:
            call_log.append(q)
            return f"answer: {q}"

        tool = Tool(name="search", description="Search", execute=my_fn)
        tool_resp = _make_tool_response("search", args={"q": "test"})
        text_resp = _make_text_response("Found it")

        client = _stub_client(tool_resp, text_resp)
        result = generate(
            "test-model",
            prompt="Search",
            tools=[tool],
            client=client,
        )

        assert call_log == ["test"]
        assert result.steps[0].tool_results[0].content == "answer: test"

    def test_passive_tool_no_execute(self) -> None:
        """Passive tool (no execute handler) returns tool calls in result without execution."""
        tool = Tool(name="calculator", description="Calculate")
        tool_resp = _make_tool_response("calculator", args={"expr": "1+1"})

        client = _stub_client(tool_resp)
        result = generate(
            "test-model",
            prompt="Calculate",
            tools=[tool],
            max_tool_rounds=1,
            client=client,
        )

        # Should NOT execute (no handler) -- breaks out immediately
        assert len(result.steps) == 1
        assert len(result.tool_calls) > 0
        assert len(result.tool_results) == 0

    def test_total_usage_aggregates(self) -> None:
        """total_usage sums usage across all steps."""
        tool = self._weather_tool()
        tool_resp = _make_tool_response("get_weather", args={"city": "NYC"})
        text_resp = _make_text_response("Sunny!")

        client = _stub_client(tool_resp, text_resp)
        result = generate(
            "test-model",
            prompt="Weather?",
            tools=[tool],
            client=client,
        )

        assert len(result.steps) == 2
        # Each response has 10+5=15 tokens
        assert result.total_usage.input_tokens == 20
        assert result.total_usage.output_tokens == 10
        assert result.total_usage.total_tokens == 30

    def test_tool_returns_dict(self) -> None:
        """Tool returning a dict is preserved as-is in the result."""
        def dict_tool() -> dict[str, Any]:
            return {"status": "ok", "value": 42}

        tool = Tool(name="info", description="Info", execute=dict_tool)
        tool_resp = _make_tool_response("info")
        text_resp = _make_text_response("Done")

        client = _stub_client(tool_resp, text_resp)
        result = generate(
            "test-model",
            prompt="Info",
            tools=[tool],
            client=client,
        )

        assert result.steps[0].tool_results[0].content == {"status": "ok", "value": 42}

    def test_tool_returns_non_string_non_dict(self) -> None:
        """Tool returning an int is converted to a string."""
        def num_tool() -> int:
            return 42

        tool = Tool(name="num", description="Number", execute=num_tool)
        tool_resp = _make_tool_response("num")
        text_resp = _make_text_response("42")

        client = _stub_client(tool_resp, text_resp)
        result = generate(
            "test-model",
            prompt="Number",
            tools=[tool],
            client=client,
        )

        assert result.steps[0].tool_results[0].content == "42"


# ===================================================================
# TestExecuteTools
# ===================================================================


class TestExecuteTools:
    """Direct tests for the _execute_tools helper."""

    def test_single_tool_call(self) -> None:
        """Single tool call goes through execute path."""
        def add(a: int = 0, b: int = 0) -> str:
            return str(a + b)

        tool = Tool(name="add", description="Add", execute=add)
        tc = ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2})
        results = _execute_tools([tool], [tc])

        assert len(results) == 1
        assert results[0].content == "3"
        assert results[0].tool_call_id == "c1"
        assert results[0].is_error is False

    def test_missing_tool_returns_error(self) -> None:
        """Unknown tool name results in error ToolResult."""
        tool = Tool(name="known", description="Known", execute=lambda: "ok")
        tc = ToolCall(id="c1", name="unknown", arguments={})
        results = _execute_tools([tool], [tc])

        assert results[0].is_error is True
        assert "Unknown tool" in results[0].content


# ===================================================================
# TestGenerateObject
# ===================================================================


class TestGenerateObject:
    """Tests for generate_object()."""

    def test_valid_json_parsed(self) -> None:
        """Valid JSON in response text is parsed into output."""
        json_text = json.dumps({"name": "Alice", "age": 30})
        resp = _make_text_response(json_text)
        client = _stub_client(resp)

        result = generate_object(
            "test-model",
            prompt="Generate person",
            schema={"type": "object", "properties": {"name": {"type": "string"}}},
            client=client,
        )

        assert result.output == {"name": "Alice", "age": 30}
        assert result.text == json_text

    def test_invalid_json_raises(self) -> None:
        """Non-JSON response raises NoObjectGeneratedError."""
        resp = _make_text_response("This is not JSON")
        client = _stub_client(resp)

        with pytest.raises(NoObjectGeneratedError, match="Failed to parse"):
            generate_object(
                "test-model",
                prompt="Generate",
                schema={"type": "object"},
                client=client,
            )

    def test_json_with_whitespace(self) -> None:
        """JSON surrounded by whitespace is parsed correctly."""
        json_text = '  {"key": "value"}  '
        # json.loads handles leading/trailing whitespace
        resp = _make_text_response(json_text)
        client = _stub_client(resp)

        result = generate_object(
            "test-model",
            prompt="Generate",
            schema={"type": "object"},
            client=client,
        )

        assert result.output == {"key": "value"}

    def test_response_format_set_to_json_schema(self) -> None:
        """The request uses json_schema response format."""
        resp = _make_text_response('{"x": 1}')
        client, adapter = _stub_client_and_adapter(resp)

        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        generate_object(
            "test-model",
            prompt="Generate",
            schema=schema,
            client=client,
        )

        req = adapter.requests[0]
        assert req.response_format is not None
        assert req.response_format.type == "json_schema"
        assert req.response_format.json_schema == schema
        assert req.response_format.strict is True

    def test_schema_passed_through(self) -> None:
        """Complex schema is correctly forwarded."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "scores": {"type": "array", "items": {"type": "number"}},
            },
            "required": ["name", "scores"],
        }
        resp = _make_text_response('{"name": "Bob", "scores": [1, 2, 3]}')
        client, adapter = _stub_client_and_adapter(resp)

        result = generate_object(
            "test-model",
            prompt="Generate",
            schema=schema,
            client=client,
        )

        assert result.output["name"] == "Bob"
        assert result.output["scores"] == [1, 2, 3]
        assert adapter.requests[0].response_format.json_schema == schema

    def test_result_preserves_steps(self) -> None:
        """GenerateResult from generate_object has correct steps."""
        resp = _make_text_response('{"ok": true}')
        client = _stub_client(resp)

        result = generate_object(
            "test-model",
            prompt="Generate",
            schema={"type": "object"},
            client=client,
        )

        assert len(result.steps) == 1
        assert result.output == {"ok": True}

    def test_json_array_parsed(self) -> None:
        """JSON arrays are also valid structured output."""
        resp = _make_text_response('[1, 2, 3]')
        client = _stub_client(resp)

        result = generate_object(
            "test-model",
            prompt="Generate",
            schema={"type": "array"},
            client=client,
        )

        assert result.output == [1, 2, 3]

    def test_empty_object_parsed(self) -> None:
        """Empty JSON object is valid."""
        resp = _make_text_response('{}')
        client = _stub_client(resp)

        result = generate_object(
            "test-model",
            prompt="Generate",
            schema={"type": "object"},
            client=client,
        )

        assert result.output == {}
