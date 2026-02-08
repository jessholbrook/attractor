"""Tests for unified_llm.types.results."""
from __future__ import annotations

import pytest

from unified_llm.types.enums import FinishReason
from unified_llm.types.tools import ToolCall, ToolResult
from unified_llm.types.response import FinishReasonInfo, Response, Usage, Warning
from unified_llm.types.results import GenerateResult, StepResult


# ---------------------------------------------------------------------------
# StepResult
# ---------------------------------------------------------------------------


class TestStepResult:
    def test_defaults(self) -> None:
        sr = StepResult()
        assert sr.text == ""
        assert sr.reasoning is None
        assert sr.tool_calls == ()
        assert sr.tool_results == ()
        assert sr.finish_reason == FinishReasonInfo()
        assert sr.usage == Usage()
        assert sr.response == Response()
        assert sr.warnings == ()

    def test_custom_values(self) -> None:
        tc = ToolCall(id="c1", name="fn", arguments="{}")
        tr = ToolResult(tool_call_id="c1", content="ok")
        w = Warning(message="old model", code="deprecation")
        sr = StepResult(
            text="hello",
            reasoning="thought",
            tool_calls=(tc,),
            tool_results=(tr,),
            warnings=(w,),
        )
        assert sr.text == "hello"
        assert sr.reasoning == "thought"
        assert sr.tool_calls == (tc,)
        assert sr.tool_results == (tr,)
        assert sr.warnings == (w,)

    def test_frozen(self) -> None:
        sr = StepResult()
        with pytest.raises(AttributeError):
            sr.text = "nope"  # type: ignore[misc]

    def test_equality(self) -> None:
        assert StepResult() == StepResult()
        assert StepResult(text="a") != StepResult(text="b")


# ---------------------------------------------------------------------------
# GenerateResult
# ---------------------------------------------------------------------------


class TestGenerateResult:
    def test_defaults(self) -> None:
        gr = GenerateResult()
        assert gr.text == ""
        assert gr.reasoning is None
        assert gr.tool_calls == ()
        assert gr.tool_results == ()
        assert gr.finish_reason == FinishReasonInfo()
        assert gr.usage == Usage()
        assert gr.total_usage == Usage()
        assert gr.steps == ()
        assert gr.response == Response()
        assert gr.output is None

    def test_with_steps(self) -> None:
        step = StepResult(text="step1")
        gr = GenerateResult(steps=(step,), text="final")
        assert len(gr.steps) == 1
        assert gr.steps[0].text == "step1"
        assert gr.text == "final"

    def test_frozen(self) -> None:
        gr = GenerateResult()
        with pytest.raises(AttributeError):
            gr.text = "nope"  # type: ignore[misc]

    def test_output_arbitrary(self) -> None:
        gr = GenerateResult(output={"key": "value"})
        assert gr.output == {"key": "value"}

    def test_total_usage_separate(self) -> None:
        gr = GenerateResult(
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            total_usage=Usage(input_tokens=20, output_tokens=10, total_tokens=30),
        )
        assert gr.usage.total_tokens == 15
        assert gr.total_usage.total_tokens == 30

    def test_equality(self) -> None:
        assert GenerateResult() == GenerateResult()
        assert GenerateResult(text="x") != GenerateResult(text="y")
