from __future__ import annotations

import pytest

from wolverine.model.run import HealingRun, RunStatus


class TestRunStatus:
    def test_all_values(self) -> None:
        expected = {
            "pending", "ingesting", "classifying", "diagnosing",
            "healing", "validating", "awaiting_review", "completed", "failed",
        }
        assert {v.value for v in RunStatus} == expected

    def test_status_count(self) -> None:
        assert len(RunStatus) == 9

    def test_is_str(self) -> None:
        assert isinstance(RunStatus.PENDING, str)
        assert RunStatus.PENDING == "pending"

    def test_lifecycle_order(self) -> None:
        statuses = list(RunStatus)
        assert statuses.index(RunStatus.PENDING) < statuses.index(RunStatus.INGESTING)
        assert statuses.index(RunStatus.INGESTING) < statuses.index(RunStatus.CLASSIFYING)
        assert statuses.index(RunStatus.HEALING) < statuses.index(RunStatus.VALIDATING)
        assert statuses.index(RunStatus.VALIDATING) < statuses.index(RunStatus.AWAITING_REVIEW)


class TestHealingRun:
    def test_construction_minimal(self) -> None:
        run = HealingRun(id="run-001", signal_id="sig-001")
        assert run.id == "run-001"
        assert run.signal_id == "sig-001"

    def test_default_status_is_pending(self) -> None:
        run = HealingRun(id="run-002", signal_id="sig-002")
        assert run.status == RunStatus.PENDING

    def test_default_strings_are_empty(self) -> None:
        run = HealingRun(id="run-003", signal_id="sig-003")
        assert run.issue_id == ""
        assert run.solution_id == ""
        assert run.review_id == ""
        assert run.pipeline_checkpoint == ""
        assert run.started_at == ""
        assert run.completed_at == ""
        assert run.error == ""

    def test_construction_all_fields(self) -> None:
        run = HealingRun(
            id="run-004",
            signal_id="sig-004",
            status=RunStatus.COMPLETED,
            issue_id="iss-004",
            solution_id="sol-004",
            review_id="rev-004",
            pipeline_checkpoint="healing",
            started_at="2025-01-15T10:00:00Z",
            completed_at="2025-01-15T10:05:00Z",
            error="",
        )
        assert run.status == RunStatus.COMPLETED
        assert run.issue_id == "iss-004"
        assert run.solution_id == "sol-004"
        assert run.review_id == "rev-004"

    def test_frozen_immutability(self) -> None:
        run = HealingRun(id="run-005", signal_id="sig-005")
        with pytest.raises(AttributeError):
            run.status = RunStatus.FAILED  # type: ignore[misc]
        with pytest.raises(AttributeError):
            run.error = "something broke"  # type: ignore[misc]

    def test_equality(self) -> None:
        kwargs = dict(id="run-eq", signal_id="sig-eq")
        assert HealingRun(**kwargs) == HealingRun(**kwargs)

    def test_hashable(self) -> None:
        run = HealingRun(id="run-hash", signal_id="sig-hash")
        assert isinstance(hash(run), int)
        s = {run}
        assert run in s
