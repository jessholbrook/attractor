from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class IssueSeverity(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class IssueCategory(StrEnum):
    BUG = "bug"
    MISSING_CONTENT = "missing_content"
    UX_ISSUE = "ux_issue"
    PERFORMANCE = "performance"
    ACCESSIBILITY = "accessibility"
    OTHER = "other"


class IssueStatus(StrEnum):
    NEW = "new"
    TRIAGED = "triaged"
    DIAGNOSING = "diagnosing"
    DIAGNOSED = "diagnosed"
    HEALING = "healing"
    AWAITING_REVIEW = "awaiting_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEPLOYED = "deployed"
    CLOSED = "closed"


@dataclass(frozen=True)
class Issue:
    id: str
    title: str
    description: str
    severity: IssueSeverity
    status: IssueStatus
    category: IssueCategory = IssueCategory.OTHER
    signal_ids: tuple[str, ...] = ()
    root_cause: str = ""
    affected_files: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    created_at: str = ""
    updated_at: str = ""
    duplicate_of: str = ""
