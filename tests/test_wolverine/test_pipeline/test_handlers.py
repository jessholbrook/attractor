from __future__ import annotations

from pathlib import Path

from attractor.model.context import Context
from attractor.model.outcome import Status

from wolverine.pipeline.graph import build_wolverine_graph
from wolverine.pipeline.handlers import (
    ApplyToolHandler,
    ClassifyHandler,
    DeduplicateToolHandler,
    DiagnoseHandler,
    HealHandler,
    IngestHandler,
    IngestToolHandler,
    ReviseHandler,
    ValidateHandler,
)


def _make_args(tmp_path: Path):
    graph = build_wolverine_graph()
    context = Context()
    node = graph.nodes["Start"]  # placeholder, overridden per test
    return graph, context, tmp_path


class TestIngestHandler:
    def test_returns_success(self, tmp_path) -> None:
        graph, context, logs = _make_args(tmp_path)
        handler = IngestHandler()
        outcome = handler.execute(graph.nodes["ingest"], context, graph, logs)
        assert outcome.status == Status.SUCCESS

    def test_sets_ingested(self, tmp_path) -> None:
        graph, context, logs = _make_args(tmp_path)
        handler = IngestHandler()
        outcome = handler.execute(graph.nodes["ingest"], context, graph, logs)
        assert outcome.context_updates["ingested"] is True


class TestClassifyHandler:
    def test_returns_success(self, tmp_path) -> None:
        graph, context, logs = _make_args(tmp_path)
        handler = ClassifyHandler()
        outcome = handler.execute(graph.nodes["classify"], context, graph, logs)
        assert outcome.status == Status.SUCCESS

    def test_sets_severity(self, tmp_path) -> None:
        graph, context, logs = _make_args(tmp_path)
        handler = ClassifyHandler()
        outcome = handler.execute(graph.nodes["classify"], context, graph, logs)
        assert outcome.context_updates["severity"] == "medium"

    def test_sets_is_duplicate(self, tmp_path) -> None:
        graph, context, logs = _make_args(tmp_path)
        handler = ClassifyHandler()
        outcome = handler.execute(graph.nodes["classify"], context, graph, logs)
        assert outcome.context_updates["is_duplicate"] == "false"

    def test_sets_category(self, tmp_path) -> None:
        graph, context, logs = _make_args(tmp_path)
        handler = ClassifyHandler()
        outcome = handler.execute(graph.nodes["classify"], context, graph, logs)
        assert outcome.context_updates["category"] == "other"


class TestDiagnoseHandler:
    def test_returns_success(self, tmp_path) -> None:
        graph, context, logs = _make_args(tmp_path)
        handler = DiagnoseHandler()
        outcome = handler.execute(graph.nodes["diagnose"], context, graph, logs)
        assert outcome.status == Status.SUCCESS

    def test_sets_root_cause(self, tmp_path) -> None:
        graph, context, logs = _make_args(tmp_path)
        handler = DiagnoseHandler()
        outcome = handler.execute(graph.nodes["diagnose"], context, graph, logs)
        assert outcome.context_updates["root_cause"] == "Stub root cause"


class TestHealHandler:
    def test_returns_success(self, tmp_path) -> None:
        graph, context, logs = _make_args(tmp_path)
        handler = HealHandler()
        outcome = handler.execute(graph.nodes["heal"], context, graph, logs)
        assert outcome.status == Status.SUCCESS

    def test_sets_solution(self, tmp_path) -> None:
        graph, context, logs = _make_args(tmp_path)
        handler = HealHandler()
        outcome = handler.execute(graph.nodes["heal"], context, graph, logs)
        assert outcome.context_updates["solution_summary"] == "Stub solution"


class TestValidateHandler:
    def test_returns_success(self, tmp_path) -> None:
        graph, context, logs = _make_args(tmp_path)
        handler = ValidateHandler()
        outcome = handler.execute(graph.nodes["validate"], context, graph, logs)
        assert outcome.status == Status.SUCCESS


class TestReviseHandler:
    def test_returns_success(self, tmp_path) -> None:
        graph, context, logs = _make_args(tmp_path)
        handler = ReviseHandler()
        outcome = handler.execute(graph.nodes["revise"], context, graph, logs)
        assert outcome.status == Status.SUCCESS

    def test_sets_revised_solution(self, tmp_path) -> None:
        graph, context, logs = _make_args(tmp_path)
        handler = ReviseHandler()
        outcome = handler.execute(graph.nodes["revise"], context, graph, logs)
        assert outcome.context_updates["solution_summary"] == "Revised stub solution"


class TestIngestToolHandler:
    def test_returns_success(self, tmp_path) -> None:
        graph, context, logs = _make_args(tmp_path)
        handler = IngestToolHandler()
        outcome = handler.execute(graph.nodes["ingest"], context, graph, logs)
        assert outcome.status == Status.SUCCESS
        assert outcome.context_updates["ingested"] is True


class TestDeduplicateToolHandler:
    def test_returns_success(self, tmp_path) -> None:
        graph, context, logs = _make_args(tmp_path)
        handler = DeduplicateToolHandler()
        outcome = handler.execute(graph.nodes["deduplicate"], context, graph, logs)
        assert outcome.status == Status.SUCCESS
        assert outcome.context_updates["deduplicated"] is True


class TestApplyToolHandler:
    def test_returns_success(self, tmp_path) -> None:
        graph, context, logs = _make_args(tmp_path)
        handler = ApplyToolHandler()
        outcome = handler.execute(graph.nodes["apply"], context, graph, logs)
        assert outcome.status == Status.SUCCESS
        assert outcome.context_updates["applied"] is True
