"""Outcome model: status and result data from node execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Status(Enum):
    """Possible outcomes of a node execution."""

    SUCCESS = "success"
    FAIL = "fail"
    PARTIAL_SUCCESS = "partial_success"
    RETRY = "retry"
    SKIPPED = "skipped"


@dataclass
class Outcome:
    """Result produced by a node handler after execution."""

    status: Status
    preferred_label: str = ""
    suggested_next_ids: list[str] = field(default_factory=list)
    context_updates: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    failure_reason: str = ""

    @property
    def succeeded(self) -> bool:
        """True if status is SUCCESS or PARTIAL_SUCCESS."""
        return self.status in (Status.SUCCESS, Status.PARTIAL_SUCCESS)

    @property
    def failed(self) -> bool:
        """True if status is FAIL."""
        return self.status is Status.FAIL
