from __future__ import annotations

import pytest

from wolverine.model.issue import Issue, IssueCategory, IssueSeverity, IssueStatus
from wolverine.model.review import Review, ReviewComment, ReviewDecision
from wolverine.model.run import HealingRun, RunStatus
from wolverine.model.signal import RawSignal, SignalKind, SignalSource
from wolverine.model.solution import FileDiff, Solution, SolutionStatus
from wolverine.store.db import Database
from wolverine.store.migrations import run_migrations
from wolverine.store.repositories import (
    IssueRepository,
    ReviewRepository,
    RunRepository,
    SignalRepository,
    SolutionRepository,
)


@pytest.fixture
def db() -> Database:
    """Create a fresh in-memory database with migrations for each test."""
    database = Database(":memory:")
    database.connect()
    run_migrations(database)
    yield database
    database.close()


# ---------------------------------------------------------------------------
# Shared factory helpers
# ---------------------------------------------------------------------------

def _make_signal(
    id: str = "sig-1",
    kind: SignalKind = SignalKind.ERROR_LOG,
    source: SignalSource = SignalSource.SENTRY,
    title: str = "NullPointerException in UserService",
    body: str = "Stack trace...",
    received_at: str = "2025-01-15T10:00:00Z",
    metadata: dict | None = None,
    raw_payload: str = '{"error": "NPE"}',
) -> RawSignal:
    return RawSignal(
        id=id,
        kind=kind,
        source=source,
        title=title,
        body=body,
        received_at=received_at,
        metadata=metadata or {"env": "prod"},
        raw_payload=raw_payload,
    )


def _make_issue(
    id: str = "iss-1",
    title: str = "Login fails for SSO users",
    description: str = "Users report 500 errors during SSO login flow.",
    severity: IssueSeverity = IssueSeverity.HIGH,
    status: IssueStatus = IssueStatus.NEW,
    category: IssueCategory = IssueCategory.BUG,
    signal_ids: tuple[str, ...] = (),
    root_cause: str = "",
    affected_files: tuple[str, ...] = (),
    tags: tuple[str, ...] = ("auth", "sso"),
    created_at: str = "2025-01-15T10:00:00Z",
    updated_at: str = "2025-01-15T10:00:00Z",
) -> Issue:
    return Issue(
        id=id,
        title=title,
        description=description,
        severity=severity,
        status=status,
        category=category,
        signal_ids=signal_ids,
        root_cause=root_cause,
        affected_files=affected_files,
        tags=tags,
        created_at=created_at,
        updated_at=updated_at,
    )


def _make_solution(
    id: str = "sol-1",
    issue_id: str = "iss-1",
    status: SolutionStatus = SolutionStatus.GENERATING,
    summary: str = "Fix null check in SSO handler",
    reasoning: str = "The SSO callback handler does not check...",
    diffs: tuple[FileDiff, ...] = (),
    test_results: str = "",
    created_at: str = "2025-01-15T11:00:00Z",
    attempt_number: int = 1,
    llm_model: str = "claude-opus-4-20250514",
    token_usage: dict | None = None,
) -> Solution:
    return Solution(
        id=id,
        issue_id=issue_id,
        status=status,
        summary=summary,
        reasoning=reasoning,
        diffs=diffs,
        test_results=test_results,
        created_at=created_at,
        attempt_number=attempt_number,
        llm_model=llm_model,
        token_usage=token_usage or {"input": 1000, "output": 500},
    )


def _make_review(
    id: str = "rev-1",
    solution_id: str = "sol-1",
    issue_id: str = "iss-1",
    reviewer: str = "human",
    decision: ReviewDecision = ReviewDecision.APPROVED,
    feedback: str = "Looks good.",
    comments: tuple[ReviewComment, ...] = (),
    created_at: str = "2025-01-15T12:00:00Z",
) -> Review:
    return Review(
        id=id,
        solution_id=solution_id,
        issue_id=issue_id,
        reviewer=reviewer,
        decision=decision,
        feedback=feedback,
        comments=comments,
        created_at=created_at,
    )


def _make_run(
    id: str = "run-1",
    signal_id: str = "sig-1",
    status: RunStatus = RunStatus.PENDING,
    issue_id: str = "",
    solution_id: str = "",
    review_id: str = "",
    started_at: str = "2025-01-15T10:00:00Z",
    completed_at: str = "",
    error: str = "",
) -> HealingRun:
    return HealingRun(
        id=id,
        signal_id=signal_id,
        status=status,
        issue_id=issue_id,
        solution_id=solution_id,
        review_id=review_id,
        started_at=started_at,
        completed_at=completed_at,
        error=error,
    )


# ===========================================================================
# SignalRepository tests
# ===========================================================================


class TestSignalRepository:
    def test_create_and_get(self, db: Database) -> None:
        repo = SignalRepository(db)
        signal = _make_signal()
        repo.create(signal)
        result = repo.get("sig-1")
        assert result is not None
        assert result.id == "sig-1"
        assert result.kind == SignalKind.ERROR_LOG
        assert result.source == SignalSource.SENTRY
        assert result.title == "NullPointerException in UserService"
        assert result.body == "Stack trace..."
        assert result.received_at == "2025-01-15T10:00:00Z"
        assert result.metadata == {"env": "prod"}
        assert result.raw_payload == '{"error": "NPE"}'

    def test_get_nonexistent_returns_none(self, db: Database) -> None:
        repo = SignalRepository(db)
        assert repo.get("no-such-id") is None

    def test_list_all(self, db: Database) -> None:
        repo = SignalRepository(db)
        for i in range(5):
            repo.create(_make_signal(id=f"sig-{i}", received_at=f"2025-01-15T1{i}:00:00Z"))
        results = repo.list_all()
        assert len(results) == 5

    def test_list_all_with_pagination(self, db: Database) -> None:
        repo = SignalRepository(db)
        for i in range(10):
            repo.create(_make_signal(id=f"sig-{i}", received_at=f"2025-01-{15 + i}T10:00:00Z"))
        page1 = repo.list_all(limit=3, offset=0)
        page2 = repo.list_all(limit=3, offset=3)
        assert len(page1) == 3
        assert len(page2) == 3
        # No overlap
        ids1 = {s.id for s in page1}
        ids2 = {s.id for s in page2}
        assert ids1.isdisjoint(ids2)

    def test_count(self, db: Database) -> None:
        repo = SignalRepository(db)
        assert repo.count() == 0
        repo.create(_make_signal(id="sig-1"))
        repo.create(_make_signal(id="sig-2"))
        assert repo.count() == 2

    def test_metadata_preserved(self, db: Database) -> None:
        repo = SignalRepository(db)
        signal = _make_signal(metadata={"key1": "val1", "key2": "val2"})
        repo.create(signal)
        result = repo.get("sig-1")
        assert result is not None
        assert result.metadata == {"key1": "val1", "key2": "val2"}


# ===========================================================================
# IssueRepository tests
# ===========================================================================


class TestIssueRepository:
    def test_create_and_get(self, db: Database) -> None:
        repo = IssueRepository(db)
        issue = _make_issue()
        repo.create(issue)
        result = repo.get("iss-1")
        assert result is not None
        assert result.id == "iss-1"
        assert result.title == "Login fails for SSO users"
        assert result.severity == IssueSeverity.HIGH
        assert result.status == IssueStatus.NEW
        assert result.category == IssueCategory.BUG
        assert result.tags == ("auth", "sso")

    def test_get_nonexistent_returns_none(self, db: Database) -> None:
        repo = IssueRepository(db)
        assert repo.get("no-such-id") is None

    def test_create_with_signal_ids(self, db: Database) -> None:
        # Must create signals first (FK constraint)
        sig_repo = SignalRepository(db)
        sig_repo.create(_make_signal(id="sig-1"))
        sig_repo.create(_make_signal(id="sig-2"))

        repo = IssueRepository(db)
        issue = _make_issue(signal_ids=("sig-1", "sig-2"))
        repo.create(issue)

        result = repo.get("iss-1")
        assert result is not None
        assert set(result.signal_ids) == {"sig-1", "sig-2"}

    def test_update_status(self, db: Database) -> None:
        repo = IssueRepository(db)
        repo.create(_make_issue())
        repo.update_status("iss-1", IssueStatus.TRIAGED)
        result = repo.get("iss-1")
        assert result is not None
        assert result.status == IssueStatus.TRIAGED

    def test_update_root_cause(self, db: Database) -> None:
        repo = IssueRepository(db)
        repo.create(_make_issue())
        repo.update_root_cause(
            "iss-1",
            "Missing null check in SSO callback",
            ("src/auth/sso.py", "src/auth/middleware.py"),
        )
        result = repo.get("iss-1")
        assert result is not None
        assert result.root_cause == "Missing null check in SSO callback"
        assert result.affected_files == ("src/auth/sso.py", "src/auth/middleware.py")

    def test_list_by_status(self, db: Database) -> None:
        repo = IssueRepository(db)
        repo.create(_make_issue(id="iss-1", status=IssueStatus.NEW))
        repo.create(_make_issue(id="iss-2", status=IssueStatus.TRIAGED))
        repo.create(_make_issue(id="iss-3", status=IssueStatus.NEW))

        new_issues = repo.list_by_status(IssueStatus.NEW)
        assert len(new_issues) == 2
        triaged_issues = repo.list_by_status(IssueStatus.TRIAGED)
        assert len(triaged_issues) == 1

    def test_list_all(self, db: Database) -> None:
        repo = IssueRepository(db)
        for i in range(5):
            repo.create(_make_issue(id=f"iss-{i}"))
        results = repo.list_all()
        assert len(results) == 5

    def test_list_all_with_pagination(self, db: Database) -> None:
        repo = IssueRepository(db)
        for i in range(10):
            repo.create(_make_issue(id=f"iss-{i}"))
        page1 = repo.list_all(limit=4, offset=0)
        page2 = repo.list_all(limit=4, offset=4)
        assert len(page1) == 4
        assert len(page2) == 4

    def test_link_signal(self, db: Database) -> None:
        sig_repo = SignalRepository(db)
        sig_repo.create(_make_signal(id="sig-1"))

        repo = IssueRepository(db)
        repo.create(_make_issue(id="iss-1"))
        repo.link_signal("iss-1", "sig-1")

        result = repo.get("iss-1")
        assert result is not None
        assert "sig-1" in result.signal_ids

    def test_link_signal_idempotent(self, db: Database) -> None:
        sig_repo = SignalRepository(db)
        sig_repo.create(_make_signal(id="sig-1"))

        repo = IssueRepository(db)
        repo.create(_make_issue(id="iss-1"))
        repo.link_signal("iss-1", "sig-1")
        repo.link_signal("iss-1", "sig-1")  # Should not raise

        result = repo.get("iss-1")
        assert result is not None
        assert result.signal_ids == ("sig-1",)

    def test_find_by_title(self, db: Database) -> None:
        repo = IssueRepository(db)
        repo.create(_make_issue(id="iss-1", title="Login fails for SSO users"))
        repo.create(_make_issue(id="iss-2", title="Dashboard loading slow"))
        repo.create(_make_issue(id="iss-3", title="SSO token expired error"))

        results = repo.find_by_title("SSO")
        assert len(results) == 2
        titles = {r.title for r in results}
        assert "Login fails for SSO users" in titles
        assert "SSO token expired error" in titles

    def test_find_by_title_no_match(self, db: Database) -> None:
        repo = IssueRepository(db)
        repo.create(_make_issue(id="iss-1", title="Login fails"))
        results = repo.find_by_title("nonexistent")
        assert len(results) == 0

    def test_count_by_status(self, db: Database) -> None:
        repo = IssueRepository(db)
        repo.create(_make_issue(id="iss-1", status=IssueStatus.NEW))
        repo.create(_make_issue(id="iss-2", status=IssueStatus.NEW))
        repo.create(_make_issue(id="iss-3", status=IssueStatus.TRIAGED))

        counts = repo.count_by_status()
        assert counts["new"] == 2
        assert counts["triaged"] == 1

    def test_count_by_status_empty(self, db: Database) -> None:
        repo = IssueRepository(db)
        counts = repo.count_by_status()
        assert counts == {}


# ===========================================================================
# SolutionRepository tests
# ===========================================================================


class TestSolutionRepository:
    @pytest.fixture(autouse=True)
    def _setup_issue(self, db: Database) -> None:
        """Create prerequisite issue for FK constraints."""
        IssueRepository(db).create(_make_issue(id="iss-1"))

    def test_create_and_get(self, db: Database) -> None:
        repo = SolutionRepository(db)
        solution = _make_solution()
        repo.create(solution)
        result = repo.get("sol-1")
        assert result is not None
        assert result.id == "sol-1"
        assert result.issue_id == "iss-1"
        assert result.status == SolutionStatus.GENERATING
        assert result.summary == "Fix null check in SSO handler"
        assert result.llm_model == "claude-opus-4-20250514"
        assert result.token_usage == {"input": 1000, "output": 500}
        assert result.attempt_number == 1

    def test_get_nonexistent_returns_none(self, db: Database) -> None:
        repo = SolutionRepository(db)
        assert repo.get("no-such-id") is None

    def test_create_with_diffs(self, db: Database) -> None:
        diff = FileDiff(
            file_path="src/auth/sso.py",
            original_content="def handle(): pass",
            modified_content="def handle():\n    if not user: raise",
            diff_text="--- a/src/auth/sso.py\n+++ b/src/auth/sso.py",
        )
        solution = _make_solution(diffs=(diff,))
        repo = SolutionRepository(db)
        repo.create(solution)
        result = repo.get("sol-1")
        assert result is not None
        assert len(result.diffs) == 1
        assert result.diffs[0].file_path == "src/auth/sso.py"
        assert result.diffs[0].diff_text == "--- a/src/auth/sso.py\n+++ b/src/auth/sso.py"

    def test_update_status(self, db: Database) -> None:
        repo = SolutionRepository(db)
        repo.create(_make_solution())
        repo.update_status("sol-1", SolutionStatus.GENERATED)
        result = repo.get("sol-1")
        assert result is not None
        assert result.status == SolutionStatus.GENERATED

    def test_list_by_issue(self, db: Database) -> None:
        repo = SolutionRepository(db)
        repo.create(_make_solution(id="sol-1", issue_id="iss-1", attempt_number=1))
        repo.create(_make_solution(id="sol-2", issue_id="iss-1", attempt_number=2))
        results = repo.list_by_issue("iss-1")
        assert len(results) == 2
        assert results[0].attempt_number == 1
        assert results[1].attempt_number == 2

    def test_list_by_issue_empty(self, db: Database) -> None:
        repo = SolutionRepository(db)
        results = repo.list_by_issue("iss-1")
        assert results == ()

    def test_get_latest_for_issue(self, db: Database) -> None:
        repo = SolutionRepository(db)
        repo.create(_make_solution(id="sol-1", issue_id="iss-1", attempt_number=1))
        repo.create(_make_solution(id="sol-2", issue_id="iss-1", attempt_number=2))
        repo.create(_make_solution(id="sol-3", issue_id="iss-1", attempt_number=3))
        result = repo.get_latest_for_issue("iss-1")
        assert result is not None
        assert result.id == "sol-3"
        assert result.attempt_number == 3

    def test_get_latest_for_issue_none(self, db: Database) -> None:
        repo = SolutionRepository(db)
        assert repo.get_latest_for_issue("iss-1") is None


# ===========================================================================
# ReviewRepository tests
# ===========================================================================


class TestReviewRepository:
    @pytest.fixture(autouse=True)
    def _setup_deps(self, db: Database) -> None:
        """Create prerequisite issue and solution for FK constraints."""
        IssueRepository(db).create(_make_issue(id="iss-1"))
        SolutionRepository(db).create(_make_solution(id="sol-1", issue_id="iss-1"))

    def test_create_and_get(self, db: Database) -> None:
        repo = ReviewRepository(db)
        review = _make_review()
        repo.create(review)
        result = repo.get("rev-1")
        assert result is not None
        assert result.id == "rev-1"
        assert result.solution_id == "sol-1"
        assert result.reviewer == "human"
        assert result.decision == ReviewDecision.APPROVED
        assert result.feedback == "Looks good."

    def test_get_nonexistent_returns_none(self, db: Database) -> None:
        repo = ReviewRepository(db)
        assert repo.get("no-such-id") is None

    def test_create_with_comments(self, db: Database) -> None:
        comments = (
            ReviewComment(file_path="src/auth/sso.py", line_number=42, comment="Consider edge case"),
            ReviewComment(file_path="src/auth/middleware.py", line_number=10, comment="Add test"),
        )
        review = _make_review(comments=comments)
        repo = ReviewRepository(db)
        repo.create(review)
        result = repo.get("rev-1")
        assert result is not None
        assert len(result.comments) == 2
        assert result.comments[0].file_path == "src/auth/sso.py"
        assert result.comments[0].line_number == 42
        assert result.comments[1].comment == "Add test"

    def test_list_by_issue(self, db: Database) -> None:
        repo = ReviewRepository(db)
        repo.create(_make_review(id="rev-1", issue_id="iss-1"))
        repo.create(_make_review(id="rev-2", issue_id="iss-1", created_at="2025-01-15T13:00:00Z"))
        results = repo.list_by_issue("iss-1")
        assert len(results) == 2

    def test_list_by_issue_empty(self, db: Database) -> None:
        repo = ReviewRepository(db)
        results = repo.list_by_issue("iss-1")
        assert results == ()

    def test_list_by_solution(self, db: Database) -> None:
        repo = ReviewRepository(db)
        repo.create(_make_review(id="rev-1", solution_id="sol-1"))
        repo.create(_make_review(id="rev-2", solution_id="sol-1", created_at="2025-01-15T13:00:00Z"))
        results = repo.list_by_solution("sol-1")
        assert len(results) == 2

    def test_list_by_solution_empty(self, db: Database) -> None:
        repo = ReviewRepository(db)
        results = repo.list_by_solution("sol-1")
        assert results == ()


# ===========================================================================
# RunRepository tests
# ===========================================================================


class TestRunRepository:
    @pytest.fixture(autouse=True)
    def _setup_signal(self, db: Database) -> None:
        """Create prerequisite signal for FK constraint."""
        SignalRepository(db).create(_make_signal(id="sig-1"))

    def test_create_and_get(self, db: Database) -> None:
        repo = RunRepository(db)
        run = _make_run()
        repo.create(run)
        result = repo.get("run-1")
        assert result is not None
        assert result.id == "run-1"
        assert result.signal_id == "sig-1"
        assert result.status == RunStatus.PENDING

    def test_get_nonexistent_returns_none(self, db: Database) -> None:
        repo = RunRepository(db)
        assert repo.get("no-such-id") is None

    def test_update_status(self, db: Database) -> None:
        repo = RunRepository(db)
        repo.create(_make_run())
        repo.update_status("run-1", RunStatus.INGESTING)
        result = repo.get("run-1")
        assert result is not None
        assert result.status == RunStatus.INGESTING

    def test_update_field_issue_id(self, db: Database) -> None:
        repo = RunRepository(db)
        repo.create(_make_run())
        repo.update_field("run-1", "issue_id", "iss-123")
        result = repo.get("run-1")
        assert result is not None
        assert result.issue_id == "iss-123"

    def test_update_field_solution_id(self, db: Database) -> None:
        repo = RunRepository(db)
        repo.create(_make_run())
        repo.update_field("run-1", "solution_id", "sol-456")
        result = repo.get("run-1")
        assert result is not None
        assert result.solution_id == "sol-456"

    def test_update_field_disallowed_raises(self, db: Database) -> None:
        repo = RunRepository(db)
        repo.create(_make_run())
        with pytest.raises(ValueError, match="not in allowed fields"):
            repo.update_field("run-1", "status", "completed")

    def test_list_by_status(self, db: Database) -> None:
        repo = RunRepository(db)
        SignalRepository(db).create(_make_signal(id="sig-2"))
        repo.create(_make_run(id="run-1", signal_id="sig-1", status=RunStatus.PENDING))
        repo.create(_make_run(id="run-2", signal_id="sig-2", status=RunStatus.INGESTING))

        pending = repo.list_by_status(RunStatus.PENDING)
        assert len(pending) == 1
        assert pending[0].id == "run-1"

    def test_list_by_status_empty(self, db: Database) -> None:
        repo = RunRepository(db)
        results = repo.list_by_status(RunStatus.COMPLETED)
        assert results == ()

    def test_list_recent(self, db: Database) -> None:
        repo = RunRepository(db)
        for i in range(5):
            sig_id = f"sig-{i + 10}"
            SignalRepository(db).create(_make_signal(id=sig_id))
            repo.create(
                _make_run(
                    id=f"run-{i}",
                    signal_id=sig_id,
                    started_at=f"2025-01-{15 + i}T10:00:00Z",
                )
            )
        results = repo.list_recent(limit=3)
        assert len(results) == 3

    def test_list_recent_ordering(self, db: Database) -> None:
        repo = RunRepository(db)
        SignalRepository(db).create(_make_signal(id="sig-2"))
        repo.create(_make_run(id="run-old", signal_id="sig-1", started_at="2025-01-10T10:00:00Z"))
        repo.create(_make_run(id="run-new", signal_id="sig-2", started_at="2025-01-20T10:00:00Z"))
        results = repo.list_recent(limit=10)
        assert len(results) == 2
        assert results[0].id == "run-new"
        assert results[1].id == "run-old"

    def test_count_by_status(self, db: Database) -> None:
        repo = RunRepository(db)
        SignalRepository(db).create(_make_signal(id="sig-2"))
        SignalRepository(db).create(_make_signal(id="sig-3"))
        repo.create(_make_run(id="run-1", signal_id="sig-1", status=RunStatus.PENDING))
        repo.create(_make_run(id="run-2", signal_id="sig-2", status=RunStatus.PENDING))
        repo.create(_make_run(id="run-3", signal_id="sig-3", status=RunStatus.COMPLETED))

        counts = repo.count_by_status()
        assert counts["pending"] == 2
        assert counts["completed"] == 1

    def test_count_by_status_empty(self, db: Database) -> None:
        repo = RunRepository(db)
        counts = repo.count_by_status()
        assert counts == {}
