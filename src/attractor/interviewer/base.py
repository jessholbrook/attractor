"""Interviewer protocol definition."""

from __future__ import annotations

from typing import Protocol

from attractor.model.question import Answer, Question


class Interviewer(Protocol):
    """Protocol for objects that can ask a question and return an answer."""

    def ask(self, question: Question) -> Answer: ...
