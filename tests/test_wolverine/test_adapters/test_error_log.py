from __future__ import annotations

from wolverine.adapters.error_log import ErrorLogAdapter
from wolverine.model.signal import SignalKind, SignalSource


class TestErrorLogAdapter:
    def test_reads_log_files(self, tmp_path) -> None:
        (tmp_path / "app.log").write_text("Error: something failed")
        adapter = ErrorLogAdapter(tmp_path)
        signals = adapter.fetch()
        assert len(signals) == 1
        assert signals[0].title == "app.log"
        assert signals[0].body == "Error: something failed"
        assert signals[0].kind == SignalKind.ERROR_LOG

    def test_reads_json_files(self, tmp_path) -> None:
        (tmp_path / "errors.json").write_text('{"error": "oops"}')
        adapter = ErrorLogAdapter(tmp_path)
        signals = adapter.fetch()
        assert len(signals) == 1
        assert signals[0].title == "errors.json"

    def test_tracks_processed_files(self, tmp_path) -> None:
        (tmp_path / "app.log").write_text("Error 1")
        adapter = ErrorLogAdapter(tmp_path)
        first = adapter.fetch()
        assert len(first) == 1

        # Add another file
        (tmp_path / "app2.log").write_text("Error 2")
        second = adapter.fetch()
        assert len(second) == 1
        assert second[0].title == "app2.log"

    def test_empty_directory_returns_empty(self, tmp_path) -> None:
        adapter = ErrorLogAdapter(tmp_path)
        signals = adapter.fetch()
        assert signals == ()

    def test_nonexistent_directory_returns_empty(self, tmp_path) -> None:
        adapter = ErrorLogAdapter(tmp_path / "nonexistent")
        signals = adapter.fetch()
        assert signals == ()

    def test_ignores_non_log_files(self, tmp_path) -> None:
        (tmp_path / "readme.txt").write_text("not a log")
        (tmp_path / "data.csv").write_text("col1,col2")
        adapter = ErrorLogAdapter(tmp_path)
        signals = adapter.fetch()
        assert signals == ()

    def test_source_is_cli(self) -> None:
        adapter = ErrorLogAdapter("/fake")
        assert adapter.source == SignalSource.CLI
