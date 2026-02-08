from __future__ import annotations

from wolverine.adapters.manual import ManualAdapter
from wolverine.model.signal import SignalKind, SignalSource


class TestManualAdapter:
    def test_submit_and_fetch_single(self) -> None:
        adapter = ManualAdapter()
        adapter.submit("Bug", "Something broke")
        signals = adapter.fetch()
        assert len(signals) == 1
        assert signals[0].title == "Bug"
        assert signals[0].body == "Something broke"

    def test_submit_and_fetch_multiple(self) -> None:
        adapter = ManualAdapter()
        adapter.submit("Bug 1", "Body 1")
        adapter.submit("Bug 2", "Body 2")
        adapter.submit("Bug 3", "Body 3")
        signals = adapter.fetch()
        assert len(signals) == 3
        assert signals[0].title == "Bug 1"
        assert signals[2].title == "Bug 3"

    def test_fetch_drains_queue(self) -> None:
        adapter = ManualAdapter()
        adapter.submit("Bug", "Body")
        first = adapter.fetch()
        assert len(first) == 1
        second = adapter.fetch()
        assert len(second) == 0

    def test_empty_fetch_returns_empty_tuple(self) -> None:
        adapter = ManualAdapter()
        signals = adapter.fetch()
        assert signals == ()

    def test_source_is_cli(self) -> None:
        adapter = ManualAdapter()
        assert adapter.source == SignalSource.CLI

    def test_signal_fields_correct(self) -> None:
        adapter = ManualAdapter()
        signal = adapter.submit("Title", "Body", kind=SignalKind.USER_FEEDBACK)
        assert signal.kind == SignalKind.USER_FEEDBACK
        assert signal.source == SignalSource.CLI
        assert signal.title == "Title"
        assert signal.body == "Body"
        assert signal.id  # non-empty
        assert signal.received_at  # non-empty

    def test_submit_returns_signal(self) -> None:
        adapter = ManualAdapter()
        signal = adapter.submit("T", "B")
        fetched = adapter.fetch()
        assert fetched[0].id == signal.id
