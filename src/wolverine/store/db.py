from __future__ import annotations

import sqlite3


class Database:
    """SQLite database wrapper with WAL mode for concurrent access."""

    def __init__(self, path: str = ":memory:") -> None:
        self._path = path
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        """Open connection and enable WAL mode."""
        self._conn = sqlite3.connect(self._path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a single SQL statement."""
        assert self._conn is not None, "Database not connected"
        return self._conn.execute(sql, params)

    def executemany(self, sql: str, params_list: list[tuple]) -> sqlite3.Cursor:
        """Execute a SQL statement against multiple parameter sets."""
        assert self._conn is not None, "Database not connected"
        return self._conn.executemany(sql, params_list)

    def fetch_one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        """Execute a query and return the first row, or None."""
        cursor = self.execute(sql, params)
        return cursor.fetchone()

    def fetch_all(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        """Execute a query and return all rows."""
        cursor = self.execute(sql, params)
        return cursor.fetchall()

    def commit(self) -> None:
        """Commit the current transaction."""
        assert self._conn is not None, "Database not connected"
        self._conn.commit()

    @property
    def connection(self) -> sqlite3.Connection:
        """Return the underlying connection."""
        assert self._conn is not None, "Database not connected"
        return self._conn
