from __future__ import annotations

import json
from dataclasses import asdict

from wolverine.model.issue import Issue, IssueCategory, IssueSeverity, IssueStatus
from wolverine.model.review import Review, ReviewComment, ReviewDecision
from wolverine.model.run import HealingRun, RunStatus
from wolverine.model.signal import RawSignal, SignalKind, SignalSource
from wolverine.model.solution import FileDiff, Solution, SolutionStatus
from wolverine.store.db import Database


class SignalRepository:
    """Repository for RawSignal persistence."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def create(self, signal: RawSignal) -> None:
        """Insert a new signal."""
        self._db.execute(
            """INSERT INTO signals (id, kind, source, title, body, received_at, metadata, raw_payload)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                signal.id,
                signal.kind.value,
                signal.source.value,
                signal.title,
                signal.body,
                signal.received_at,
                json.dumps(signal.metadata),
                signal.raw_payload,
            ),
        )
        self._db.commit()

    def get(self, signal_id: str) -> RawSignal | None:
        """Retrieve a signal by ID, or None if not found."""
        row = self._db.fetch_one("SELECT * FROM signals WHERE id = ?", (signal_id,))
        if row is None:
            return None
        return _row_to_signal(row)

    def list_all(self, limit: int = 100, offset: int = 0) -> tuple[RawSignal, ...]:
        """List signals with pagination."""
        rows = self._db.fetch_all(
            "SELECT * FROM signals ORDER BY received_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        return tuple(_row_to_signal(r) for r in rows)

    def count(self) -> int:
        """Return the total number of signals."""
        row = self._db.fetch_one("SELECT COUNT(*) as cnt FROM signals")
        assert row is not None
        return row["cnt"]


class IssueRepository:
    """Repository for Issue persistence."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def create(self, issue: Issue) -> None:
        """Insert a new issue and link its signals."""
        self._db.execute(
            """INSERT INTO issues
               (id, title, description, severity, status, category, root_cause,
                affected_files, tags, created_at, updated_at, duplicate_of)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                issue.id,
                issue.title,
                issue.description,
                issue.severity.value,
                issue.status.value,
                issue.category.value,
                issue.root_cause,
                json.dumps(list(issue.affected_files)),
                json.dumps(list(issue.tags)),
                issue.created_at,
                issue.updated_at,
                issue.duplicate_of,
            ),
        )
        for signal_id in issue.signal_ids:
            self._db.execute(
                "INSERT OR IGNORE INTO issue_signals (issue_id, signal_id) VALUES (?, ?)",
                (issue.id, signal_id),
            )
        self._db.commit()

    def get(self, issue_id: str) -> Issue | None:
        """Retrieve an issue by ID, or None if not found."""
        row = self._db.fetch_one("SELECT * FROM issues WHERE id = ?", (issue_id,))
        if row is None:
            return None
        signal_ids = self._get_signal_ids(issue_id)
        return _row_to_issue(row, signal_ids)

    def update_status(self, issue_id: str, status: IssueStatus) -> None:
        """Update an issue's status."""
        self._db.execute(
            "UPDATE issues SET status = ? WHERE id = ?",
            (status.value, issue_id),
        )
        self._db.commit()

    def update_root_cause(
        self, issue_id: str, root_cause: str, affected_files: tuple[str, ...]
    ) -> None:
        """Update the root cause and affected files for an issue."""
        self._db.execute(
            "UPDATE issues SET root_cause = ?, affected_files = ? WHERE id = ?",
            (root_cause, json.dumps(list(affected_files)), issue_id),
        )
        self._db.commit()

    def list_by_status(
        self, status: IssueStatus, limit: int = 50
    ) -> tuple[Issue, ...]:
        """List issues filtered by status."""
        rows = self._db.fetch_all(
            "SELECT * FROM issues WHERE status = ? ORDER BY created_at DESC LIMIT ?",
            (status.value, limit),
        )
        return tuple(
            _row_to_issue(r, self._get_signal_ids(r["id"])) for r in rows
        )

    def list_all(self, limit: int = 100, offset: int = 0) -> tuple[Issue, ...]:
        """List issues with pagination."""
        rows = self._db.fetch_all(
            "SELECT * FROM issues ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        return tuple(
            _row_to_issue(r, self._get_signal_ids(r["id"])) for r in rows
        )

    def link_signal(self, issue_id: str, signal_id: str) -> None:
        """Link a signal to an issue."""
        self._db.execute(
            "INSERT OR IGNORE INTO issue_signals (issue_id, signal_id) VALUES (?, ?)",
            (issue_id, signal_id),
        )
        self._db.commit()

    def find_by_title(self, title: str) -> tuple[Issue, ...]:
        """Find issues whose title contains the given string (for dedup)."""
        rows = self._db.fetch_all(
            "SELECT * FROM issues WHERE title LIKE ?",
            (f"%{title}%",),
        )
        return tuple(
            _row_to_issue(r, self._get_signal_ids(r["id"])) for r in rows
        )

    def count_by_status(self) -> dict[str, int]:
        """Return counts of issues grouped by status."""
        rows = self._db.fetch_all(
            "SELECT status, COUNT(*) as cnt FROM issues GROUP BY status"
        )
        return {row["status"]: row["cnt"] for row in rows}

    def _get_signal_ids(self, issue_id: str) -> tuple[str, ...]:
        """Get all signal IDs linked to an issue."""
        rows = self._db.fetch_all(
            "SELECT signal_id FROM issue_signals WHERE issue_id = ?",
            (issue_id,),
        )
        return tuple(r["signal_id"] for r in rows)


class SolutionRepository:
    """Repository for Solution persistence."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def create(self, solution: Solution) -> None:
        """Insert a new solution."""
        diffs_json = json.dumps([asdict(d) for d in solution.diffs])
        self._db.execute(
            """INSERT INTO solutions
               (id, issue_id, status, summary, reasoning, diffs, test_results,
                agent_session_id, created_at, attempt_number, llm_model, token_usage)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                solution.id,
                solution.issue_id,
                solution.status.value,
                solution.summary,
                solution.reasoning,
                diffs_json,
                solution.test_results,
                solution.agent_session_id,
                solution.created_at,
                solution.attempt_number,
                solution.llm_model,
                json.dumps(solution.token_usage),
            ),
        )
        self._db.commit()

    def get(self, solution_id: str) -> Solution | None:
        """Retrieve a solution by ID, or None if not found."""
        row = self._db.fetch_one(
            "SELECT * FROM solutions WHERE id = ?", (solution_id,)
        )
        if row is None:
            return None
        return _row_to_solution(row)

    def update_status(self, solution_id: str, status: SolutionStatus) -> None:
        """Update a solution's status."""
        self._db.execute(
            "UPDATE solutions SET status = ? WHERE id = ?",
            (status.value, solution_id),
        )
        self._db.commit()

    def list_by_issue(self, issue_id: str) -> tuple[Solution, ...]:
        """List all solutions for an issue."""
        rows = self._db.fetch_all(
            "SELECT * FROM solutions WHERE issue_id = ? ORDER BY attempt_number ASC",
            (issue_id,),
        )
        return tuple(_row_to_solution(r) for r in rows)

    def get_latest_for_issue(self, issue_id: str) -> Solution | None:
        """Get the most recent solution for an issue."""
        row = self._db.fetch_one(
            "SELECT * FROM solutions WHERE issue_id = ? ORDER BY attempt_number DESC LIMIT 1",
            (issue_id,),
        )
        if row is None:
            return None
        return _row_to_solution(row)


class ReviewRepository:
    """Repository for Review persistence."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def create(self, review: Review) -> None:
        """Insert a new review."""
        comments_json = json.dumps([asdict(c) for c in review.comments])
        self._db.execute(
            """INSERT INTO reviews
               (id, solution_id, issue_id, reviewer, decision, feedback, comments, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                review.id,
                review.solution_id,
                review.issue_id,
                review.reviewer,
                review.decision.value,
                review.feedback,
                comments_json,
                review.created_at,
            ),
        )
        self._db.commit()

    def get(self, review_id: str) -> Review | None:
        """Retrieve a review by ID, or None if not found."""
        row = self._db.fetch_one("SELECT * FROM reviews WHERE id = ?", (review_id,))
        if row is None:
            return None
        return _row_to_review(row)

    def list_by_issue(self, issue_id: str) -> tuple[Review, ...]:
        """List all reviews for an issue."""
        rows = self._db.fetch_all(
            "SELECT * FROM reviews WHERE issue_id = ? ORDER BY created_at DESC",
            (issue_id,),
        )
        return tuple(_row_to_review(r) for r in rows)

    def list_by_solution(self, solution_id: str) -> tuple[Review, ...]:
        """List all reviews for a solution."""
        rows = self._db.fetch_all(
            "SELECT * FROM reviews WHERE solution_id = ? ORDER BY created_at DESC",
            (solution_id,),
        )
        return tuple(_row_to_review(r) for r in rows)


class RunRepository:
    """Repository for HealingRun persistence."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def create(self, run: HealingRun) -> None:
        """Insert a new healing run."""
        self._db.execute(
            """INSERT INTO healing_runs
               (id, signal_id, status, issue_id, solution_id, review_id,
                pipeline_checkpoint, started_at, completed_at, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run.id,
                run.signal_id,
                run.status.value,
                run.issue_id,
                run.solution_id,
                run.review_id,
                run.pipeline_checkpoint,
                run.started_at,
                run.completed_at,
                run.error,
            ),
        )
        self._db.commit()

    def get(self, run_id: str) -> HealingRun | None:
        """Retrieve a healing run by ID, or None if not found."""
        row = self._db.fetch_one(
            "SELECT * FROM healing_runs WHERE id = ?", (run_id,)
        )
        if row is None:
            return None
        return _row_to_run(row)

    def update_status(self, run_id: str, status: RunStatus) -> None:
        """Update a run's status."""
        self._db.execute(
            "UPDATE healing_runs SET status = ? WHERE id = ?",
            (status.value, run_id),
        )
        self._db.commit()

    def update_field(self, run_id: str, field: str, value: str) -> None:
        """Update a single field on a healing run (e.g., issue_id, solution_id)."""
        allowed_fields = {
            "issue_id",
            "solution_id",
            "review_id",
            "pipeline_checkpoint",
            "started_at",
            "completed_at",
            "error",
        }
        if field not in allowed_fields:
            raise ValueError(
                f"Field {field!r} not in allowed fields: {sorted(allowed_fields)}"
            )
        self._db.execute(
            f"UPDATE healing_runs SET {field} = ? WHERE id = ?",  # noqa: S608
            (value, run_id),
        )
        self._db.commit()

    def list_by_status(self, status: RunStatus) -> tuple[HealingRun, ...]:
        """List runs filtered by status."""
        rows = self._db.fetch_all(
            "SELECT * FROM healing_runs WHERE status = ? ORDER BY started_at DESC",
            (status.value,),
        )
        return tuple(_row_to_run(r) for r in rows)

    def list_recent(self, limit: int = 20) -> tuple[HealingRun, ...]:
        """List the most recent runs."""
        rows = self._db.fetch_all(
            "SELECT * FROM healing_runs ORDER BY started_at DESC LIMIT ?",
            (limit,),
        )
        return tuple(_row_to_run(r) for r in rows)

    def count_by_status(self) -> dict[str, int]:
        """Return counts of runs grouped by status."""
        rows = self._db.fetch_all(
            "SELECT status, COUNT(*) as cnt FROM healing_runs GROUP BY status"
        )
        return {row["status"]: row["cnt"] for row in rows}


# ---------------------------------------------------------------------------
# Row-to-dataclass conversion helpers
# ---------------------------------------------------------------------------


def _row_to_signal(row: dict) -> RawSignal:
    return RawSignal(
        id=row["id"],
        kind=SignalKind(row["kind"]),
        source=SignalSource(row["source"]),
        title=row["title"],
        body=row["body"],
        received_at=row["received_at"],
        metadata=json.loads(row["metadata"]),
        raw_payload=row["raw_payload"],
    )


def _row_to_issue(row: dict, signal_ids: tuple[str, ...]) -> Issue:
    return Issue(
        id=row["id"],
        title=row["title"],
        description=row["description"],
        severity=IssueSeverity(row["severity"]),
        status=IssueStatus(row["status"]),
        category=IssueCategory(row["category"]),
        signal_ids=signal_ids,
        root_cause=row["root_cause"],
        affected_files=tuple(json.loads(row["affected_files"])),
        tags=tuple(json.loads(row["tags"])),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        duplicate_of=row["duplicate_of"],
    )


def _row_to_solution(row: dict) -> Solution:
    raw_diffs = json.loads(row["diffs"])
    diffs = tuple(
        FileDiff(
            file_path=d["file_path"],
            original_content=d["original_content"],
            modified_content=d["modified_content"],
            diff_text=d["diff_text"],
        )
        for d in raw_diffs
    )
    return Solution(
        id=row["id"],
        issue_id=row["issue_id"],
        status=SolutionStatus(row["status"]),
        summary=row["summary"],
        reasoning=row["reasoning"],
        diffs=diffs,
        test_results=row["test_results"],
        agent_session_id=row["agent_session_id"],
        created_at=row["created_at"],
        attempt_number=row["attempt_number"],
        llm_model=row["llm_model"],
        token_usage=json.loads(row["token_usage"]),
    )


def _row_to_review(row: dict) -> Review:
    raw_comments = json.loads(row["comments"])
    comments = tuple(
        ReviewComment(
            file_path=c["file_path"],
            line_number=c["line_number"],
            comment=c["comment"],
        )
        for c in raw_comments
    )
    return Review(
        id=row["id"],
        solution_id=row["solution_id"],
        issue_id=row["issue_id"],
        reviewer=row["reviewer"],
        decision=ReviewDecision(row["decision"]),
        feedback=row["feedback"],
        comments=comments,
        created_at=row["created_at"],
    )


def _row_to_run(row: dict) -> HealingRun:
    return HealingRun(
        id=row["id"],
        signal_id=row["signal_id"],
        status=RunStatus(row["status"]),
        issue_id=row["issue_id"],
        solution_id=row["solution_id"],
        review_id=row["review_id"],
        pipeline_checkpoint=row["pipeline_checkpoint"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        error=row["error"],
    )
