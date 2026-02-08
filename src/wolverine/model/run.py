from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RunStatus(StrEnum):
    PENDING = "pending"
    INGESTING = "ingesting"
    CLASSIFYING = "classifying"
    DIAGNOSING = "diagnosing"
    HEALING = "healing"
    VALIDATING = "validating"
    AWAITING_REVIEW = "awaiting_review"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class HealingRun:
    id: str
    signal_id: str
    status: RunStatus = RunStatus.PENDING
    issue_id: str = ""
    solution_id: str = ""
    review_id: str = ""
    pipeline_checkpoint: str = ""
    started_at: str = ""
    completed_at: str = ""
    error: str = ""
