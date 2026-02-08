"""CallbackInterviewer: delegates to a user-supplied callback function."""

from __future__ import annotations

from typing import Callable

from attractor.model.question import Answer, Question


class CallbackInterviewer:
    """Interviewer that delegates question answering to a callback.

    The callback receives a Question and must return an Answer.
    """

    def __init__(self, callback: Callable[[Question], Answer]) -> None:
        self._callback = callback

    def ask(self, question: Question) -> Answer:
        return self._callback(question)
