from __future__ import annotations

import json
from pathlib import Path

from attractor.model.context import Context
from attractor.model.graph import Graph, Node
from attractor.model.outcome import Outcome, Status
from unified_llm import Client as UnifiedClient
from unified_llm import generate, generate_object

from wolverine.pipeline.prompts import CLASSIFY_SCHEMA, CLASSIFY_SYSTEM, DIAGNOSE_SYSTEM


class IngestHandler:
    """Stub: stores signal in context."""

    def execute(self, node: Node, context: Context, graph: Graph, logs_root: Path) -> Outcome:
        return Outcome(status=Status.SUCCESS, context_updates={"ingested": True})


class ClassifyHandler:
    """Stub: classifies as medium severity, other category."""

    def execute(self, node: Node, context: Context, graph: Graph, logs_root: Path) -> Outcome:
        return Outcome(
            status=Status.SUCCESS,
            context_updates={
                "severity": "medium",
                "category": "other",
                "is_duplicate": "false",
            },
        )


class DiagnoseHandler:
    """Stub: provides fake diagnosis."""

    def execute(self, node: Node, context: Context, graph: Graph, logs_root: Path) -> Outcome:
        return Outcome(
            status=Status.SUCCESS,
            context_updates={
                "root_cause": "Stub root cause",
                "affected_files": "[]",
            },
        )


class HealHandler:
    """Stub: generates a fake solution."""

    def execute(self, node: Node, context: Context, graph: Graph, logs_root: Path) -> Outcome:
        return Outcome(
            status=Status.SUCCESS,
            context_updates={
                "solution_summary": "Stub solution",
                "solution_diffs": "[]",
            },
        )


class ValidateHandler:
    """Stub: always validates successfully."""

    def execute(self, node: Node, context: Context, graph: Graph, logs_root: Path) -> Outcome:
        return Outcome(status=Status.SUCCESS)


class ReviseHandler:
    """Stub: generates a revised solution."""

    def execute(self, node: Node, context: Context, graph: Graph, logs_root: Path) -> Outcome:
        return Outcome(
            status=Status.SUCCESS,
            context_updates={"solution_summary": "Revised stub solution"},
        )


class IngestToolHandler:
    """Tool handler for ingest node."""

    def execute(self, node: Node, context: Context, graph: Graph, logs_root: Path) -> Outcome:
        return Outcome(status=Status.SUCCESS, context_updates={"ingested": True})


class DeduplicateToolHandler:
    """Tool handler for deduplicate node."""

    def execute(self, node: Node, context: Context, graph: Graph, logs_root: Path) -> Outcome:
        return Outcome(status=Status.SUCCESS, context_updates={"deduplicated": True})


class ApplyToolHandler:
    """Tool handler for apply node."""

    def execute(self, node: Node, context: Context, graph: Graph, logs_root: Path) -> Outcome:
        return Outcome(status=Status.SUCCESS, context_updates={"applied": True})


class LLMClassifyHandler:
    """Classifies signals using unified_llm generate_object()."""

    def __init__(
        self,
        client: UnifiedClient,
        model: str = "claude-sonnet-4-20250514",
    ) -> None:
        self._client = client
        self._model = model

    def execute(self, node: Node, context: Context, graph: Graph, logs_root: Path) -> Outcome:
        signal_title = context.get("signal_title", "")
        signal_body = context.get("signal_body", "")

        prompt = f"Classify this signal:\n\nTitle: {signal_title}\nBody: {signal_body}"

        try:
            result = generate_object(
                self._model,
                prompt=prompt,
                system=CLASSIFY_SYSTEM,
                schema=CLASSIFY_SCHEMA,
                client=self._client,
            )
        except Exception as exc:
            return Outcome(
                status=Status.FAIL,
                failure_reason=f"Classification error: {exc}",
            )

        classification = result.output
        return Outcome(
            status=Status.SUCCESS,
            context_updates={
                "severity": classification["severity"],
                "category": classification["category"],
                "issue_title": classification["title"],
                "issue_description": classification["description"],
                "tags": json.dumps(classification.get("tags", [])),
                "is_duplicate": str(classification.get("is_duplicate", False)).lower(),
            },
        )


class LLMDiagnoseHandler:
    """Diagnoses root cause using unified_llm generate()."""

    def __init__(
        self,
        client: UnifiedClient,
        model: str = "claude-sonnet-4-20250514",
    ) -> None:
        self._client = client
        self._model = model

    def execute(self, node: Node, context: Context, graph: Graph, logs_root: Path) -> Outcome:
        title = context.get("issue_title", "")
        description = context.get("issue_description", "")

        prompt = f"Diagnose this issue:\n\nTitle: {title}\nDescription: {description}"

        try:
            result = generate(
                self._model,
                prompt=prompt,
                system=DIAGNOSE_SYSTEM,
                client=self._client,
            )
        except Exception as exc:
            return Outcome(
                status=Status.FAIL,
                failure_reason=f"Diagnosis error: {exc}",
            )

        return Outcome(
            status=Status.SUCCESS,
            context_updates={
                "root_cause": result.text,
                "affected_files": "[]",
            },
        )
