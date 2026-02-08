from __future__ import annotations

import pytest

from wolverine.model.signal import RawSignal, SignalKind, SignalSource


class TestSignalKind:
    def test_all_values(self) -> None:
        expected = {"error_log", "user_feedback", "support_ticket", "ux_research", "social_media", "manual"}
        assert {v.value for v in SignalKind} == expected

    def test_is_str(self) -> None:
        assert isinstance(SignalKind.ERROR_LOG, str)
        assert SignalKind.ERROR_LOG == "error_log"

    def test_string_comparison(self) -> None:
        assert SignalKind.MANUAL == "manual"
        assert "manual" == SignalKind.MANUAL
        assert SignalKind.ERROR_LOG < SignalKind.USER_FEEDBACK  # str ordering


class TestSignalSource:
    def test_all_values(self) -> None:
        expected = {"sentry", "datadog", "form", "csv", "api", "cli"}
        assert {v.value for v in SignalSource} == expected

    def test_is_str(self) -> None:
        assert isinstance(SignalSource.SENTRY, str)
        assert SignalSource.SENTRY == "sentry"


class TestRawSignal:
    def test_construction_all_fields(self) -> None:
        sig = RawSignal(
            id="sig-001",
            kind=SignalKind.ERROR_LOG,
            source=SignalSource.SENTRY,
            title="NullPointerException in checkout",
            body="Stack trace...",
            received_at="2025-01-15T10:30:00Z",
            metadata={"project": "web-app"},
            raw_payload='{"error": "NPE"}',
        )
        assert sig.id == "sig-001"
        assert sig.kind == SignalKind.ERROR_LOG
        assert sig.source == SignalSource.SENTRY
        assert sig.title == "NullPointerException in checkout"
        assert sig.body == "Stack trace..."
        assert sig.received_at == "2025-01-15T10:30:00Z"
        assert sig.metadata == {"project": "web-app"}
        assert sig.raw_payload == '{"error": "NPE"}'

    def test_default_metadata_is_empty_dict(self) -> None:
        sig = RawSignal(
            id="sig-002",
            kind=SignalKind.USER_FEEDBACK,
            source=SignalSource.FORM,
            title="Button broken",
            body="Can't click submit",
            received_at="2025-01-15T11:00:00Z",
        )
        assert sig.metadata == {}
        assert sig.raw_payload == ""

    def test_default_metadata_is_distinct_per_instance(self) -> None:
        a = RawSignal(id="a", kind=SignalKind.MANUAL, source=SignalSource.CLI, title="", body="", received_at="")
        b = RawSignal(id="b", kind=SignalKind.MANUAL, source=SignalSource.CLI, title="", body="", received_at="")
        assert a.metadata is not b.metadata

    def test_frozen_immutability(self) -> None:
        sig = RawSignal(
            id="sig-003",
            kind=SignalKind.SUPPORT_TICKET,
            source=SignalSource.API,
            title="Help",
            body="Need help",
            received_at="2025-01-15T12:00:00Z",
        )
        with pytest.raises(AttributeError):
            sig.id = "changed"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            sig.title = "changed"  # type: ignore[misc]

    def test_equality(self) -> None:
        kwargs = dict(
            id="sig-eq",
            kind=SignalKind.ERROR_LOG,
            source=SignalSource.DATADOG,
            title="Err",
            body="body",
            received_at="2025-01-15T00:00:00Z",
        )
        # metadata is excluded from compare
        a = RawSignal(**kwargs, metadata={"a": "1"})
        b = RawSignal(**kwargs, metadata={"b": "2"})
        assert a == b

    def test_hash_excludes_metadata(self) -> None:
        kwargs = dict(
            id="sig-hash",
            kind=SignalKind.ERROR_LOG,
            source=SignalSource.DATADOG,
            title="Err",
            body="body",
            received_at="2025-01-15T00:00:00Z",
        )
        a = RawSignal(**kwargs, metadata={"a": "1"})
        b = RawSignal(**kwargs, metadata={"b": "2"})
        assert hash(a) == hash(b)

    def test_kind_values_accessible_by_name(self) -> None:
        assert SignalKind["ERROR_LOG"] == SignalKind.ERROR_LOG
        assert SignalKind["UX_RESEARCH"] == SignalKind.UX_RESEARCH

    def test_source_values_accessible_by_name(self) -> None:
        assert SignalSource["SENTRY"] == SignalSource.SENTRY
        assert SignalSource["CSV"] == SignalSource.CSV
