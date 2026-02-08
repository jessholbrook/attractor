from __future__ import annotations

import json
from pathlib import Path

from attractor.model.context import Context
from attractor.model.outcome import Status
from unified_llm import (
    Client,
    FinishReason,
    FinishReasonInfo,
    Message,
    Response,
    StubAdapter,
    Usage,
)

from wolverine.pipeline.graph import build_wolverine_graph
from wolverine.pipeline.handlers import LLMClassifyHandler, LLMDiagnoseHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_CLASSIFICATION = {
    "severity": "high",
    "category": "bug",
    "title": "Login button unresponsive",
    "description": "The login button does not respond to clicks on mobile devices.",
    "tags": ["mobile", "auth", "ui"],
    "is_duplicate": False,
}


def make_stub_client(text: str) -> Client:
    """Create a Client with a StubAdapter that returns the given text."""
    response = Response(
        message=Message.assistant(text),
        finish_reason=FinishReasonInfo(reason=FinishReason.STOP),
        usage=Usage(input_tokens=10, output_tokens=10),
    )
    adapter = StubAdapter(responses=[response])
    return Client(providers={"stub": adapter}, default_provider="stub")


def _make_context(**kwargs: str) -> Context:
    """Create a Context pre-populated with the given key-value pairs."""
    ctx = Context()
    for k, v in kwargs.items():
        ctx.set(k, v)
    return ctx


def _graph():
    return build_wolverine_graph()


# ---------------------------------------------------------------------------
# LLMClassifyHandler
# ---------------------------------------------------------------------------


class TestLLMClassifyHandler:
    def test_returns_success(self, tmp_path: Path) -> None:
        client = make_stub_client(json.dumps(SAMPLE_CLASSIFICATION))
        handler = LLMClassifyHandler(client)
        graph = _graph()
        context = _make_context(signal_title="Bug", signal_body="It crashes")
        outcome = handler.execute(graph.nodes["classify"], context, graph, tmp_path)
        assert outcome.status == Status.SUCCESS

    def test_sets_severity(self, tmp_path: Path) -> None:
        client = make_stub_client(json.dumps(SAMPLE_CLASSIFICATION))
        handler = LLMClassifyHandler(client)
        graph = _graph()
        context = _make_context(signal_title="Bug", signal_body="It crashes")
        outcome = handler.execute(graph.nodes["classify"], context, graph, tmp_path)
        assert outcome.context_updates["severity"] == "high"

    def test_sets_category(self, tmp_path: Path) -> None:
        client = make_stub_client(json.dumps(SAMPLE_CLASSIFICATION))
        handler = LLMClassifyHandler(client)
        graph = _graph()
        context = _make_context(signal_title="Bug", signal_body="It crashes")
        outcome = handler.execute(graph.nodes["classify"], context, graph, tmp_path)
        assert outcome.context_updates["category"] == "bug"

    def test_sets_issue_title(self, tmp_path: Path) -> None:
        client = make_stub_client(json.dumps(SAMPLE_CLASSIFICATION))
        handler = LLMClassifyHandler(client)
        graph = _graph()
        context = _make_context(signal_title="Bug", signal_body="It crashes")
        outcome = handler.execute(graph.nodes["classify"], context, graph, tmp_path)
        assert outcome.context_updates["issue_title"] == "Login button unresponsive"

    def test_sets_issue_description(self, tmp_path: Path) -> None:
        client = make_stub_client(json.dumps(SAMPLE_CLASSIFICATION))
        handler = LLMClassifyHandler(client)
        graph = _graph()
        context = _make_context(signal_title="Bug", signal_body="It crashes")
        outcome = handler.execute(graph.nodes["classify"], context, graph, tmp_path)
        assert "login button" in outcome.context_updates["issue_description"].lower()

    def test_sets_tags_as_json(self, tmp_path: Path) -> None:
        client = make_stub_client(json.dumps(SAMPLE_CLASSIFICATION))
        handler = LLMClassifyHandler(client)
        graph = _graph()
        context = _make_context(signal_title="Bug", signal_body="It crashes")
        outcome = handler.execute(graph.nodes["classify"], context, graph, tmp_path)
        tags = json.loads(outcome.context_updates["tags"])
        assert tags == ["mobile", "auth", "ui"]

    def test_sets_is_duplicate_as_string(self, tmp_path: Path) -> None:
        client = make_stub_client(json.dumps(SAMPLE_CLASSIFICATION))
        handler = LLMClassifyHandler(client)
        graph = _graph()
        context = _make_context(signal_title="Bug", signal_body="It crashes")
        outcome = handler.execute(graph.nodes["classify"], context, graph, tmp_path)
        assert outcome.context_updates["is_duplicate"] == "false"

    def test_is_duplicate_true(self, tmp_path: Path) -> None:
        dup_classification = {**SAMPLE_CLASSIFICATION, "is_duplicate": True}
        client = make_stub_client(json.dumps(dup_classification))
        handler = LLMClassifyHandler(client)
        graph = _graph()
        context = _make_context(signal_title="Bug", signal_body="crashes")
        outcome = handler.execute(graph.nodes["classify"], context, graph, tmp_path)
        assert outcome.context_updates["is_duplicate"] == "true"

    def test_handles_missing_signal_title(self, tmp_path: Path) -> None:
        client = make_stub_client(json.dumps(SAMPLE_CLASSIFICATION))
        handler = LLMClassifyHandler(client)
        graph = _graph()
        context = _make_context(signal_body="Some body only")
        outcome = handler.execute(graph.nodes["classify"], context, graph, tmp_path)
        assert outcome.status == Status.SUCCESS

    def test_handles_missing_signal_body(self, tmp_path: Path) -> None:
        client = make_stub_client(json.dumps(SAMPLE_CLASSIFICATION))
        handler = LLMClassifyHandler(client)
        graph = _graph()
        context = _make_context(signal_title="Title only")
        outcome = handler.execute(graph.nodes["classify"], context, graph, tmp_path)
        assert outcome.status == Status.SUCCESS

    def test_handles_empty_context(self, tmp_path: Path) -> None:
        client = make_stub_client(json.dumps(SAMPLE_CLASSIFICATION))
        handler = LLMClassifyHandler(client)
        graph = _graph()
        context = Context()
        outcome = handler.execute(graph.nodes["classify"], context, graph, tmp_path)
        assert outcome.status == Status.SUCCESS

    def test_uses_custom_model(self, tmp_path: Path) -> None:
        client = make_stub_client(json.dumps(SAMPLE_CLASSIFICATION))
        handler = LLMClassifyHandler(client, model="gpt-4o")
        graph = _graph()
        context = _make_context(signal_title="Bug", signal_body="crash")
        handler.execute(graph.nodes["classify"], context, graph, tmp_path)
        adapter = client.providers["stub"]
        assert adapter.requests[0].model == "gpt-4o"

    def test_sends_system_prompt(self, tmp_path: Path) -> None:
        client = make_stub_client(json.dumps(SAMPLE_CLASSIFICATION))
        handler = LLMClassifyHandler(client)
        graph = _graph()
        context = _make_context(signal_title="Bug", signal_body="crash")
        handler.execute(graph.nodes["classify"], context, graph, tmp_path)
        adapter = client.providers["stub"]
        request = adapter.requests[0]
        # System message should be first
        system_msgs = [m for m in request.messages if m.role.value == "system"]
        assert len(system_msgs) == 1
        assert "classifier" in system_msgs[0].text.lower()

    def test_prompt_contains_signal_title(self, tmp_path: Path) -> None:
        client = make_stub_client(json.dumps(SAMPLE_CLASSIFICATION))
        handler = LLMClassifyHandler(client)
        graph = _graph()
        context = _make_context(signal_title="My Bug Title", signal_body="body")
        handler.execute(graph.nodes["classify"], context, graph, tmp_path)
        adapter = client.providers["stub"]
        request = adapter.requests[0]
        user_msgs = [m for m in request.messages if m.role.value == "user"]
        assert any("My Bug Title" in m.text for m in user_msgs)

    def test_classification_with_different_severity(self, tmp_path: Path) -> None:
        low_classification = {**SAMPLE_CLASSIFICATION, "severity": "low"}
        client = make_stub_client(json.dumps(low_classification))
        handler = LLMClassifyHandler(client)
        graph = _graph()
        context = _make_context(signal_title="Minor", signal_body="typo")
        outcome = handler.execute(graph.nodes["classify"], context, graph, tmp_path)
        assert outcome.context_updates["severity"] == "low"

    def test_classification_with_empty_tags(self, tmp_path: Path) -> None:
        no_tags = {**SAMPLE_CLASSIFICATION, "tags": []}
        client = make_stub_client(json.dumps(no_tags))
        handler = LLMClassifyHandler(client)
        graph = _graph()
        context = _make_context(signal_title="Bug", signal_body="crash")
        outcome = handler.execute(graph.nodes["classify"], context, graph, tmp_path)
        assert json.loads(outcome.context_updates["tags"]) == []

    def test_returns_fail_on_invalid_json(self, tmp_path: Path) -> None:
        client = make_stub_client("not valid json{{{")
        handler = LLMClassifyHandler(client)
        graph = _graph()
        context = _make_context(signal_title="Bug", signal_body="crash")
        outcome = handler.execute(graph.nodes["classify"], context, graph, tmp_path)
        assert outcome.status == Status.FAIL
        assert "Classification error" in outcome.failure_reason


# ---------------------------------------------------------------------------
# LLMDiagnoseHandler
# ---------------------------------------------------------------------------


class TestLLMDiagnoseHandler:
    def test_returns_success(self, tmp_path: Path) -> None:
        client = make_stub_client("Root cause: null pointer dereference in auth module")
        handler = LLMDiagnoseHandler(client)
        graph = _graph()
        context = _make_context(issue_title="Auth crash", issue_description="Login fails")
        outcome = handler.execute(graph.nodes["diagnose"], context, graph, tmp_path)
        assert outcome.status == Status.SUCCESS

    def test_sets_root_cause(self, tmp_path: Path) -> None:
        diagnosis_text = "The auth module has a null pointer dereference on line 42"
        client = make_stub_client(diagnosis_text)
        handler = LLMDiagnoseHandler(client)
        graph = _graph()
        context = _make_context(issue_title="Auth crash", issue_description="Login fails")
        outcome = handler.execute(graph.nodes["diagnose"], context, graph, tmp_path)
        assert outcome.context_updates["root_cause"] == diagnosis_text

    def test_sets_affected_files(self, tmp_path: Path) -> None:
        client = make_stub_client("diagnosis text")
        handler = LLMDiagnoseHandler(client)
        graph = _graph()
        context = _make_context(issue_title="Bug", issue_description="crashes")
        outcome = handler.execute(graph.nodes["diagnose"], context, graph, tmp_path)
        assert outcome.context_updates["affected_files"] == "[]"

    def test_handles_missing_title(self, tmp_path: Path) -> None:
        client = make_stub_client("diagnosis")
        handler = LLMDiagnoseHandler(client)
        graph = _graph()
        context = _make_context(issue_description="Some desc")
        outcome = handler.execute(graph.nodes["diagnose"], context, graph, tmp_path)
        assert outcome.status == Status.SUCCESS

    def test_handles_missing_description(self, tmp_path: Path) -> None:
        client = make_stub_client("diagnosis")
        handler = LLMDiagnoseHandler(client)
        graph = _graph()
        context = _make_context(issue_title="Title only")
        outcome = handler.execute(graph.nodes["diagnose"], context, graph, tmp_path)
        assert outcome.status == Status.SUCCESS

    def test_handles_empty_context(self, tmp_path: Path) -> None:
        client = make_stub_client("diagnosis")
        handler = LLMDiagnoseHandler(client)
        graph = _graph()
        context = Context()
        outcome = handler.execute(graph.nodes["diagnose"], context, graph, tmp_path)
        assert outcome.status == Status.SUCCESS

    def test_uses_custom_model(self, tmp_path: Path) -> None:
        client = make_stub_client("diagnosis")
        handler = LLMDiagnoseHandler(client, model="gpt-4o")
        graph = _graph()
        context = _make_context(issue_title="Bug", issue_description="crash")
        handler.execute(graph.nodes["diagnose"], context, graph, tmp_path)
        adapter = client.providers["stub"]
        assert adapter.requests[0].model == "gpt-4o"

    def test_sends_system_prompt(self, tmp_path: Path) -> None:
        client = make_stub_client("diagnosis")
        handler = LLMDiagnoseHandler(client)
        graph = _graph()
        context = _make_context(issue_title="Bug", issue_description="crash")
        handler.execute(graph.nodes["diagnose"], context, graph, tmp_path)
        adapter = client.providers["stub"]
        request = adapter.requests[0]
        system_msgs = [m for m in request.messages if m.role.value == "system"]
        assert len(system_msgs) == 1
        assert "diagnostician" in system_msgs[0].text.lower()

    def test_prompt_contains_title_and_description(self, tmp_path: Path) -> None:
        client = make_stub_client("diagnosis")
        handler = LLMDiagnoseHandler(client)
        graph = _graph()
        context = _make_context(issue_title="Login Bug", issue_description="Auth crash on mobile")
        handler.execute(graph.nodes["diagnose"], context, graph, tmp_path)
        adapter = client.providers["stub"]
        request = adapter.requests[0]
        user_msgs = [m for m in request.messages if m.role.value == "user"]
        assert any("Login Bug" in m.text for m in user_msgs)
        assert any("Auth crash on mobile" in m.text for m in user_msgs)

    def test_root_cause_preserves_full_response_text(self, tmp_path: Path) -> None:
        long_diagnosis = "Line 1\nLine 2\nLine 3\n" * 10
        client = make_stub_client(long_diagnosis)
        handler = LLMDiagnoseHandler(client)
        graph = _graph()
        context = _make_context(issue_title="Bug", issue_description="crash")
        outcome = handler.execute(graph.nodes["diagnose"], context, graph, tmp_path)
        assert outcome.context_updates["root_cause"] == long_diagnosis
