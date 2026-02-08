from __future__ import annotations

from wolverine.store.db import Database

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    received_at TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    raw_payload TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS issues (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    severity TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'new',
    category TEXT NOT NULL DEFAULT 'other',
    root_cause TEXT NOT NULL DEFAULT '',
    affected_files TEXT NOT NULL DEFAULT '[]',
    tags TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT '',
    duplicate_of TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS issue_signals (
    issue_id TEXT NOT NULL,
    signal_id TEXT NOT NULL,
    PRIMARY KEY (issue_id, signal_id),
    FOREIGN KEY (issue_id) REFERENCES issues(id),
    FOREIGN KEY (signal_id) REFERENCES signals(id)
);

CREATE TABLE IF NOT EXISTS solutions (
    id TEXT PRIMARY KEY,
    issue_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'generating',
    summary TEXT NOT NULL DEFAULT '',
    reasoning TEXT NOT NULL DEFAULT '',
    diffs TEXT NOT NULL DEFAULT '[]',
    test_results TEXT NOT NULL DEFAULT '',
    agent_session_id TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT '',
    attempt_number INTEGER NOT NULL DEFAULT 1,
    llm_model TEXT NOT NULL DEFAULT '',
    token_usage TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (issue_id) REFERENCES issues(id)
);

CREATE TABLE IF NOT EXISTS reviews (
    id TEXT PRIMARY KEY,
    solution_id TEXT NOT NULL,
    issue_id TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    decision TEXT NOT NULL,
    feedback TEXT NOT NULL DEFAULT '',
    comments TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (solution_id) REFERENCES solutions(id),
    FOREIGN KEY (issue_id) REFERENCES issues(id)
);

CREATE TABLE IF NOT EXISTS healing_runs (
    id TEXT PRIMARY KEY,
    signal_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    issue_id TEXT NOT NULL DEFAULT '',
    solution_id TEXT NOT NULL DEFAULT '',
    review_id TEXT NOT NULL DEFAULT '',
    pipeline_checkpoint TEXT NOT NULL DEFAULT '',
    started_at TEXT NOT NULL DEFAULT '',
    completed_at TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (signal_id) REFERENCES signals(id)
);
"""


def run_migrations(db: Database) -> None:
    """Create all tables."""
    db.connection.executescript(SCHEMA)
    db.commit()
