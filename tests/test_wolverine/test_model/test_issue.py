from __future__ import annotations

import pytest

from wolverine.model.issue import Issue, IssueCategory, IssueSeverity, IssueStatus


class TestIssueSeverity:
    def test_all_values(self) -> None:
        expected = {"critical", "high", "medium", "low"}
        assert {v.value for v in IssueSeverity} == expected

    def test_is_str(self) -> None:
        assert isinstance(IssueSeverity.CRITICAL, str)
        assert IssueSeverity.CRITICAL == "critical"


class TestIssueCategory:
    def test_all_values(self) -> None:
        expected = {"bug", "missing_content", "ux_issue", "performance", "accessibility", "other"}
        assert {v.value for v in IssueCategory} == expected

    def test_string_comparison(self) -> None:
        assert IssueCategory.BUG == "bug"
        assert "other" == IssueCategory.OTHER


class TestIssueStatus:
    def test_all_values(self) -> None:
        expected = {
            "new", "triaged", "diagnosing", "diagnosed", "healing",
            "awaiting_review", "approved", "rejected", "deployed", "closed",
        }
        assert {v.value for v in IssueStatus} == expected

    def test_status_count(self) -> None:
        assert len(IssueStatus) == 10

    def test_status_ordering_makes_sense(self) -> None:
        """Verify the lifecycle order is reflected in member definition order."""
        statuses = list(IssueStatus)
        assert statuses.index(IssueStatus.NEW) < statuses.index(IssueStatus.TRIAGED)
        assert statuses.index(IssueStatus.TRIAGED) < statuses.index(IssueStatus.DIAGNOSING)
        assert statuses.index(IssueStatus.HEALING) < statuses.index(IssueStatus.AWAITING_REVIEW)
        assert statuses.index(IssueStatus.APPROVED) < statuses.index(IssueStatus.DEPLOYED)


class TestIssue:
    def test_construction_required_fields(self) -> None:
        issue = Issue(
            id="iss-001",
            title="Login fails on Safari",
            description="Users on Safari cannot log in",
            severity=IssueSeverity.HIGH,
            status=IssueStatus.NEW,
        )
        assert issue.id == "iss-001"
        assert issue.title == "Login fails on Safari"
        assert issue.severity == IssueSeverity.HIGH
        assert issue.status == IssueStatus.NEW

    def test_default_category_is_other(self) -> None:
        issue = Issue(
            id="iss-002",
            title="t",
            description="d",
            severity=IssueSeverity.LOW,
            status=IssueStatus.NEW,
        )
        assert issue.category == IssueCategory.OTHER

    def test_default_tuples_are_empty(self) -> None:
        issue = Issue(
            id="iss-003",
            title="t",
            description="d",
            severity=IssueSeverity.MEDIUM,
            status=IssueStatus.NEW,
        )
        assert issue.signal_ids == ()
        assert issue.affected_files == ()
        assert issue.tags == ()

    def test_default_strings_are_empty(self) -> None:
        issue = Issue(
            id="iss-004",
            title="t",
            description="d",
            severity=IssueSeverity.LOW,
            status=IssueStatus.NEW,
        )
        assert issue.root_cause == ""
        assert issue.created_at == ""
        assert issue.updated_at == ""
        assert issue.duplicate_of == ""

    def test_construction_all_fields(self) -> None:
        issue = Issue(
            id="iss-005",
            title="Crash on checkout",
            description="App crashes when user clicks checkout",
            severity=IssueSeverity.CRITICAL,
            status=IssueStatus.DIAGNOSED,
            category=IssueCategory.BUG,
            signal_ids=("sig-001", "sig-002"),
            root_cause="Null reference in payment module",
            affected_files=("src/payment.py", "src/checkout.py"),
            tags=("payment", "critical"),
            created_at="2025-01-15T10:00:00Z",
            updated_at="2025-01-15T12:00:00Z",
            duplicate_of="",
        )
        assert issue.category == IssueCategory.BUG
        assert issue.signal_ids == ("sig-001", "sig-002")
        assert issue.affected_files == ("src/payment.py", "src/checkout.py")
        assert issue.tags == ("payment", "critical")

    def test_frozen_immutability(self) -> None:
        issue = Issue(
            id="iss-006",
            title="t",
            description="d",
            severity=IssueSeverity.LOW,
            status=IssueStatus.NEW,
        )
        with pytest.raises(AttributeError):
            issue.status = IssueStatus.TRIAGED  # type: ignore[misc]
        with pytest.raises(AttributeError):
            issue.title = "changed"  # type: ignore[misc]

    def test_equality(self) -> None:
        kwargs = dict(
            id="iss-eq",
            title="t",
            description="d",
            severity=IssueSeverity.HIGH,
            status=IssueStatus.NEW,
        )
        assert Issue(**kwargs) == Issue(**kwargs)

    def test_hashable(self) -> None:
        issue = Issue(
            id="iss-hash",
            title="t",
            description="d",
            severity=IssueSeverity.LOW,
            status=IssueStatus.NEW,
        )
        assert isinstance(hash(issue), int)
        # Can be used in sets
        s = {issue}
        assert issue in s
