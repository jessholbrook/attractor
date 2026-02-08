"""Question model: types for interactive interviewer prompts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QuestionType(Enum):
    """Kind of question presented to the user."""

    YES_NO = "YES_NO"
    MULTIPLE_CHOICE = "MULTIPLE_CHOICE"
    FREEFORM = "FREEFORM"
    CONFIRMATION = "CONFIRMATION"


class AnswerValue(Enum):
    """Canonical answer values for structured question types."""

    YES = "YES"
    NO = "NO"
    SKIPPED = "SKIPPED"
    TIMEOUT = "TIMEOUT"


@dataclass(frozen=True)
class Option:
    """A single selectable option for multiple-choice questions."""

    key: str
    label: str


@dataclass(frozen=True)
class Question:
    """A question to be posed to the user during pipeline execution."""

    text: str
    type: QuestionType
    options: list[Option] = field(default_factory=list)
    default: str | None = None
    timeout_seconds: float | None = None
    stage: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Answer:
    """The user's response to a Question."""

    value: AnswerValue | str | None = None
    selected_option: Option | None = None
    text: str = ""

    @property
    def is_yes(self) -> bool:
        return self.value is AnswerValue.YES

    @property
    def is_no(self) -> bool:
        return self.value is AnswerValue.NO

    @property
    def was_skipped(self) -> bool:
        return self.value is AnswerValue.SKIPPED

    @property
    def timed_out(self) -> bool:
        return self.value is AnswerValue.TIMEOUT
