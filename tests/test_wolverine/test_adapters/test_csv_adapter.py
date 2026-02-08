from __future__ import annotations

import pytest

from wolverine.adapters.csv_adapter import CSVAdapter
from wolverine.model.signal import SignalKind, SignalSource


class TestCSVAdapter:
    def test_read_csv_with_all_columns(self, tmp_path) -> None:
        csv_file = tmp_path / "signals.csv"
        csv_file.write_text(
            'title,body,kind,metadata\n'
            'Bug 1,Body 1,error_log,"{""env"": ""prod""}"\n'
            'Bug 2,Body 2,user_feedback,\n'
        )
        adapter = CSVAdapter(csv_file)
        signals = adapter.fetch()
        assert len(signals) == 2
        assert signals[0].title == "Bug 1"
        assert signals[0].kind == SignalKind.ERROR_LOG
        assert signals[0].metadata == {"env": "prod"}
        assert signals[1].title == "Bug 2"
        assert signals[1].kind == SignalKind.USER_FEEDBACK

    def test_read_csv_with_minimal_columns(self, tmp_path) -> None:
        csv_file = tmp_path / "signals.csv"
        csv_file.write_text("title,body\nBug,Broken\n")
        adapter = CSVAdapter(csv_file)
        signals = adapter.fetch()
        assert len(signals) == 1
        assert signals[0].title == "Bug"
        assert signals[0].kind == SignalKind.MANUAL  # default
        assert signals[0].source == SignalSource.CSV

    def test_empty_csv_returns_empty(self, tmp_path) -> None:
        csv_file = tmp_path / "signals.csv"
        csv_file.write_text("title,body\n")
        adapter = CSVAdapter(csv_file)
        signals = adapter.fetch()
        assert signals == ()

    def test_file_not_found_raises(self, tmp_path) -> None:
        adapter = CSVAdapter(tmp_path / "nonexistent.csv")
        with pytest.raises(FileNotFoundError):
            adapter.fetch()

    def test_fetch_consumes_file(self, tmp_path) -> None:
        csv_file = tmp_path / "signals.csv"
        csv_file.write_text("title,body\nBug,Broken\n")
        adapter = CSVAdapter(csv_file)
        first = adapter.fetch()
        assert len(first) == 1
        second = adapter.fetch()
        assert second == ()

    def test_source_is_csv(self) -> None:
        adapter = CSVAdapter("/fake/path.csv")
        assert adapter.source == SignalSource.CSV

    def test_invalid_kind_defaults_to_manual(self, tmp_path) -> None:
        csv_file = tmp_path / "signals.csv"
        csv_file.write_text("title,body,kind\nBug,Body,invalid_kind\n")
        adapter = CSVAdapter(csv_file)
        signals = adapter.fetch()
        assert len(signals) == 1
        assert signals[0].kind == SignalKind.MANUAL
