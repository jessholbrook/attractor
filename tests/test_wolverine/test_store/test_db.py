from __future__ import annotations

import sqlite3
import tempfile

import pytest

from wolverine.store.db import Database


@pytest.fixture
def db() -> Database:
    """Create a fresh in-memory database for each test."""
    database = Database(":memory:")
    database.connect()
    yield database
    database.close()


class TestDatabaseConnect:
    def test_connect_creates_connection(self) -> None:
        db = Database(":memory:")
        assert db._conn is None
        db.connect()
        assert db._conn is not None
        db.close()

    def test_wal_mode_enabled_on_disk(self) -> None:
        """WAL mode only applies to file-based databases (not :memory:)."""
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            db = Database(f.name)
            db.connect()
            row = db.fetch_one("PRAGMA journal_mode")
            assert row is not None
            assert row[0] == "wal"
            db.close()

    def test_foreign_keys_enabled(self, db: Database) -> None:
        row = db.fetch_one("PRAGMA foreign_keys")
        assert row is not None
        assert row[0] == 1

    def test_row_factory_is_sqlite_row(self, db: Database) -> None:
        db.execute("CREATE TABLE t (id INTEGER, name TEXT)")
        db.execute("INSERT INTO t VALUES (1, 'alice')")
        db.commit()
        row = db.fetch_one("SELECT * FROM t")
        assert row is not None
        assert isinstance(row, sqlite3.Row)
        assert row["id"] == 1
        assert row["name"] == "alice"


class TestDatabaseClose:
    def test_close_sets_conn_to_none(self) -> None:
        db = Database(":memory:")
        db.connect()
        db.close()
        assert db._conn is None

    def test_close_on_unconnected_is_noop(self) -> None:
        db = Database(":memory:")
        db.close()  # Should not raise


class TestDatabaseExecute:
    def test_execute_and_fetch_one(self, db: Database) -> None:
        db.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, val TEXT)")
        db.execute("INSERT INTO items VALUES (1, 'hello')")
        db.commit()
        row = db.fetch_one("SELECT * FROM items WHERE id = ?", (1,))
        assert row is not None
        assert row["val"] == "hello"

    def test_fetch_one_returns_none_for_no_match(self, db: Database) -> None:
        db.execute("CREATE TABLE items (id INTEGER PRIMARY KEY)")
        db.commit()
        row = db.fetch_one("SELECT * FROM items WHERE id = ?", (999,))
        assert row is None

    def test_fetch_all(self, db: Database) -> None:
        db.execute("CREATE TABLE nums (n INTEGER)")
        for i in range(5):
            db.execute("INSERT INTO nums VALUES (?)", (i,))
        db.commit()
        rows = db.fetch_all("SELECT * FROM nums ORDER BY n")
        assert len(rows) == 5
        assert [r["n"] for r in rows] == [0, 1, 2, 3, 4]

    def test_executemany(self, db: Database) -> None:
        db.execute("CREATE TABLE nums (n INTEGER)")
        db.executemany("INSERT INTO nums VALUES (?)", [(i,) for i in range(3)])
        db.commit()
        rows = db.fetch_all("SELECT * FROM nums ORDER BY n")
        assert len(rows) == 3

    def test_connection_property(self, db: Database) -> None:
        conn = db.connection
        assert isinstance(conn, sqlite3.Connection)

    def test_execute_without_connect_raises(self) -> None:
        db = Database(":memory:")
        with pytest.raises(AssertionError):
            db.execute("SELECT 1")

    def test_commit_without_connect_raises(self) -> None:
        db = Database(":memory:")
        with pytest.raises(AssertionError):
            db.commit()

    def test_connection_property_without_connect_raises(self) -> None:
        db = Database(":memory:")
        with pytest.raises(AssertionError):
            _ = db.connection
