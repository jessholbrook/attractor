from __future__ import annotations

import pytest

from wolverine.store.db import Database
from wolverine.store.migrations import run_migrations


@pytest.fixture
def db() -> Database:
    """Create a fresh in-memory database for each test."""
    database = Database(":memory:")
    database.connect()
    yield database
    database.close()


class TestRunMigrations:
    def test_creates_all_tables(self, db: Database) -> None:
        run_migrations(db)
        rows = db.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        table_names = {r["name"] for r in rows}
        expected = {
            "signals",
            "issues",
            "issue_signals",
            "solutions",
            "reviews",
            "healing_runs",
        }
        assert expected.issubset(table_names)

    def test_idempotent_migrations(self, db: Database) -> None:
        run_migrations(db)
        run_migrations(db)  # Should not raise
        rows = db.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        table_names = {r["name"] for r in rows}
        assert "signals" in table_names

    def test_signals_table_columns(self, db: Database) -> None:
        run_migrations(db)
        rows = db.fetch_all("PRAGMA table_info(signals)")
        col_names = {r["name"] for r in rows}
        assert col_names == {
            "id", "kind", "source", "title", "body",
            "received_at", "metadata", "raw_payload",
        }

    def test_issues_table_columns(self, db: Database) -> None:
        run_migrations(db)
        rows = db.fetch_all("PRAGMA table_info(issues)")
        col_names = {r["name"] for r in rows}
        assert col_names == {
            "id", "title", "description", "severity", "status", "category",
            "root_cause", "affected_files", "tags", "created_at", "updated_at",
            "duplicate_of",
        }

    def test_healing_runs_table_columns(self, db: Database) -> None:
        run_migrations(db)
        rows = db.fetch_all("PRAGMA table_info(healing_runs)")
        col_names = {r["name"] for r in rows}
        assert col_names == {
            "id", "signal_id", "status", "issue_id", "solution_id",
            "review_id", "pipeline_checkpoint", "started_at",
            "completed_at", "error",
        }

    def test_solutions_table_columns(self, db: Database) -> None:
        run_migrations(db)
        rows = db.fetch_all("PRAGMA table_info(solutions)")
        col_names = {r["name"] for r in rows}
        assert col_names == {
            "id", "issue_id", "status", "summary", "reasoning", "diffs",
            "test_results", "agent_session_id", "created_at",
            "attempt_number", "llm_model", "token_usage",
        }

    def test_reviews_table_columns(self, db: Database) -> None:
        run_migrations(db)
        rows = db.fetch_all("PRAGMA table_info(reviews)")
        col_names = {r["name"] for r in rows}
        assert col_names == {
            "id", "solution_id", "issue_id", "reviewer", "decision",
            "feedback", "comments", "created_at",
        }
