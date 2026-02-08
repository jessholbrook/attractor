from __future__ import annotations

import pytest

from wolverine.model.issue import Issue, IssueCategory, IssueSeverity, IssueStatus
from wolverine.model.run import HealingRun, RunStatus
from wolverine.model.signal import RawSignal, SignalKind, SignalSource
from wolverine.model.review import Review, ReviewDecision
from wolverine.model.solution import FileDiff, Solution, SolutionStatus
from wolverine.store.db import Database
from wolverine.store.migrations import run_migrations
from wolverine.store.repositories import (
    IssueRepository,
    RunRepository,
    SignalRepository,
    SolutionRepository,
)
from wolverine.web.app import create_app


@pytest.fixture
def db():
    """Create a fresh in-memory database with migrations for each test."""
    database = Database(":memory:")
    database.connect()
    run_migrations(database)
    yield database
    database.close()


@pytest.fixture
def app(db):
    """Create a Flask app for testing."""
    application = create_app(db=db)
    application.config["TESTING"] = True
    return application


@pytest.fixture
def client(app):
    """Create a Flask test client."""
    return app.test_client()


# ---------------------------------------------------------------------------
# Shared factory helpers
# ---------------------------------------------------------------------------


def make_signal(
    id: str = "sig-1",
    kind: SignalKind = SignalKind.ERROR_LOG,
    source: SignalSource = SignalSource.SENTRY,
    title: str = "NullPointerException in UserService",
    body: str = "Stack trace...",
    received_at: str = "2025-01-15T10:00:00Z",
) -> RawSignal:
    return RawSignal(
        id=id,
        kind=kind,
        source=source,
        title=title,
        body=body,
        received_at=received_at,
    )


def make_issue(
    id: str = "iss-1",
    title: str = "Login fails for SSO users",
    description: str = "Users report 500 errors during SSO login flow.",
    severity: IssueSeverity = IssueSeverity.HIGH,
    status: IssueStatus = IssueStatus.NEW,
    category: IssueCategory = IssueCategory.BUG,
    signal_ids: tuple[str, ...] = (),
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
        created_at=created_at,
        updated_at=updated_at,
    )


def make_solution(
    id: str = "sol-1",
    issue_id: str = "iss-1",
    status: SolutionStatus = SolutionStatus.GENERATED,
    summary: str = "Fix null check in SSO handler",
    reasoning: str = "",
    diffs: tuple[FileDiff, ...] = (),
    test_results: str = "",
    created_at: str = "2025-01-15T11:00:00Z",
    attempt_number: int = 1,
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
    )


def make_review(
    id: str = "rev-1",
    solution_id: str = "sol-1",
    issue_id: str = "iss-1",
    reviewer: str = "tester",
    decision: ReviewDecision = ReviewDecision.APPROVED,
    feedback: str = "",
    created_at: str = "2025-01-15T12:00:00Z",
) -> Review:
    return Review(
        id=id,
        solution_id=solution_id,
        issue_id=issue_id,
        reviewer=reviewer,
        decision=decision,
        feedback=feedback,
        created_at=created_at,
    )


def make_run(
    id: str = "run-1",
    signal_id: str = "sig-1",
    status: RunStatus = RunStatus.PENDING,
    started_at: str = "2025-01-15T10:00:00Z",
) -> HealingRun:
    return HealingRun(
        id=id,
        signal_id=signal_id,
        status=status,
        started_at=started_at,
    )


def seed_signal(app, **kwargs) -> RawSignal:
    """Create a signal in the database and return it."""
    signal = make_signal(**kwargs)
    app.extensions["signal_repo"].create(signal)
    return signal


def seed_issue(app, **kwargs) -> Issue:
    """Create an issue in the database and return it."""
    issue = make_issue(**kwargs)
    app.extensions["issue_repo"].create(issue)
    return issue


def seed_solution(app, **kwargs) -> Solution:
    """Create a solution in the database and return it."""
    solution = make_solution(**kwargs)
    app.extensions["solution_repo"].create(solution)
    return solution


def seed_review(app, **kwargs) -> Review:
    """Create a review in the database and return it."""
    review = make_review(**kwargs)
    app.extensions["review_repo"].create(review)
    return review


def seed_run(app, **kwargs) -> HealingRun:
    """Create a healing run in the database and return it."""
    run = make_run(**kwargs)
    app.extensions["run_repo"].create(run)
    return run
