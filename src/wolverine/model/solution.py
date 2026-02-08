from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class SolutionStatus(StrEnum):
    GENERATING = "generating"
    GENERATED = "generated"
    VALIDATED = "validated"
    FAILED = "failed"
    APPROVED = "approved"
    REJECTED = "rejected"
    EDITED = "edited"


@dataclass(frozen=True)
class FileDiff:
    file_path: str
    original_content: str
    modified_content: str
    diff_text: str  # unified diff format


@dataclass(frozen=True)
class Solution:
    id: str
    issue_id: str
    status: SolutionStatus
    summary: str = ""
    reasoning: str = ""
    diffs: tuple[FileDiff, ...] = ()
    test_results: str = ""
    agent_session_id: str = ""
    created_at: str = ""
    attempt_number: int = 1
    llm_model: str = ""
    token_usage: dict[str, int] = field(default_factory=dict, hash=False, compare=False)
