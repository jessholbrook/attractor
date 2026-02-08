"""Tests for the Wolverine CLI commands."""
from __future__ import annotations

import pytest
from click.testing import CliRunner

from wolverine.cli.main import cli
from wolverine.store.db import Database
from wolverine.store.migrations import run_migrations
from wolverine.store.repositories import SignalRepository


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


class TestCLIGroup:
    def test_cli_group_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Self-healing software system" in result.output

    def test_cli_lists_commands(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "serve" in result.output
        assert "ingest" in result.output
        assert "import-csv" in result.output


# ---------------------------------------------------------------------------
# serve command
# ---------------------------------------------------------------------------


class TestServeCommand:
    def test_serve_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--help"])
        assert result.exit_code == 0
        assert "Start the Wolverine web server" in result.output

    def test_serve_help_shows_options(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--help"])
        assert "--host" in result.output
        assert "--port" in result.output
        assert "--db" in result.output
        assert "--debug" in result.output


# ---------------------------------------------------------------------------
# ingest command
# ---------------------------------------------------------------------------


class TestIngestCommand:
    def test_ingest_creates_signal(self, tmp_path) -> None:
        db_path = str(tmp_path / "test.db")
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "ingest",
                "--title", "Test signal",
                "--body", "Test body",
                "--db", db_path,
            ],
        )
        assert result.exit_code == 0
        assert "Signal ingested" in result.output

    def test_ingest_stores_in_database(self, tmp_path) -> None:
        db_path = str(tmp_path / "test.db")
        runner = CliRunner()
        runner.invoke(
            cli,
            [
                "ingest",
                "--title", "DB check signal",
                "--body", "Checking storage",
                "--db", db_path,
            ],
        )

        # Verify signal is in the database
        database = Database(db_path)
        database.connect()
        repo = SignalRepository(database)
        signals = repo.list_all()
        database.close()

        assert len(signals) == 1
        assert signals[0].title == "DB check signal"

    def test_ingest_missing_title(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["ingest", "--body", "test"])
        assert result.exit_code != 0

    def test_ingest_missing_body(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["ingest", "--title", "test"])
        assert result.exit_code != 0

    def test_ingest_with_kind(self, tmp_path) -> None:
        db_path = str(tmp_path / "test.db")
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "ingest",
                "--title", "Error signal",
                "--body", "Stack trace",
                "--kind", "error_log",
                "--db", db_path,
            ],
        )
        assert result.exit_code == 0
        assert "Signal ingested" in result.output

        # Verify kind is correct
        database = Database(db_path)
        database.connect()
        repo = SignalRepository(database)
        signals = repo.list_all()
        database.close()

        assert signals[0].kind.value == "error_log"


# ---------------------------------------------------------------------------
# import-csv command
# ---------------------------------------------------------------------------


class TestImportCSVCommand:
    def test_import_csv_single_row(self, tmp_path) -> None:
        csv_path = tmp_path / "signals.csv"
        csv_path.write_text("title,body,kind\nBug report,Something broken,manual\n")

        db_path = str(tmp_path / "test.db")
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "import-csv",
                "--file", str(csv_path),
                "--db", db_path,
            ],
        )
        assert result.exit_code == 0
        assert "Imported 1 signals" in result.output

    def test_import_csv_multiple_rows(self, tmp_path) -> None:
        csv_path = tmp_path / "signals.csv"
        csv_path.write_text(
            "title,body,kind\n"
            "Bug 1,First bug,manual\n"
            "Bug 2,Second bug,error_log\n"
            "Bug 3,Third bug,user_feedback\n"
        )

        db_path = str(tmp_path / "test.db")
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "import-csv",
                "--file", str(csv_path),
                "--db", db_path,
            ],
        )
        assert result.exit_code == 0
        assert "Imported 3 signals" in result.output

    def test_import_csv_stores_in_database(self, tmp_path) -> None:
        csv_path = tmp_path / "signals.csv"
        csv_path.write_text("title,body,kind\nCSV bug,CSV body,manual\n")

        db_path = str(tmp_path / "test.db")
        runner = CliRunner()
        runner.invoke(
            cli,
            [
                "import-csv",
                "--file", str(csv_path),
                "--db", db_path,
            ],
        )

        database = Database(db_path)
        database.connect()
        repo = SignalRepository(database)
        signals = repo.list_all()
        database.close()

        assert len(signals) == 1
        assert signals[0].title == "CSV bug"

    def test_import_csv_missing_file(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "import-csv",
                "--file", "/nonexistent/file.csv",
                "--db", "test.db",
            ],
        )
        assert result.exit_code != 0
