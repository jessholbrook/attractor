"""RecordingInterviewer: wraps another interviewer and records all Q&A pairs."""

from __future__ import annotations

from dataclasses import dataclass, field

from attractor.model.question import Answer, Question


@dataclass(frozen=True)
class QAPair:
    """A recorded question-answer exchange."""

    question: Question
    answer: Answer


class RecordingInterviewer:
    """Interviewer decorator that records all Q&A exchanges.

    Wraps an inner interviewer. Every call to ask() is forwarded to the
    inner interviewer, and both the question and answer are recorded.
    """

    def __init__(self, inner: object) -> None:
        self._inner = inner
        self._records: list[QAPair] = []

    def ask(self, question: Question) -> Answer:
        answer = self._inner.ask(question)  # type: ignore[union-attr]
        self._records.append(QAPair(question=question, answer=answer))
        return answer

    def transcript(self) -> list[QAPair]:
        """Return the list of all recorded Q&A pairs."""
        return list(self._records)

    def clear(self) -> None:
        """Clear the recording history."""
        self._records.clear()
