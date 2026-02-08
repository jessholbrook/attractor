from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ReviewDecision(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"
    REQUEST_CHANGES = "request_changes"


@dataclass(frozen=True)
class ReviewComment:
    file_path: str
    line_number: int
    comment: str


@dataclass(frozen=True)
class Review:
    id: str
    solution_id: str
    issue_id: str
    reviewer: str
    decision: ReviewDecision
    feedback: str = ""
    comments: tuple[ReviewComment, ...] = ()
    created_at: str = ""
