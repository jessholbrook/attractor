"""QueueInterviewer: uses a queue for async question/answer exchange."""

from __future__ import annotations

import queue

from attractor.model.question import Answer, Question


class QueueInterviewer:
    """Interviewer that uses a thread-safe queue pair for Q&A exchange.

    Questions are put onto the question_queue; answers are read from the
    answer_queue. This enables async communication between the engine
    thread and a separate UI thread.
    """

    def __init__(
        self,
        question_queue: queue.Queue[Question] | None = None,
        answer_queue: queue.Queue[Answer] | None = None,
        timeout: float | None = None,
    ) -> None:
        self.question_queue: queue.Queue[Question] = question_queue or queue.Queue()
        self.answer_queue: queue.Queue[Answer] = answer_queue or queue.Queue()
        self._timeout = timeout

    def ask(self, question: Question) -> Answer:
        self.question_queue.put(question)
        try:
            return self.answer_queue.get(timeout=self._timeout)
        except queue.Empty:
            from attractor.model.question import AnswerValue

            return Answer(value=AnswerValue.TIMEOUT, text="")

    def respond(self, answer: Answer) -> None:
        """Convenience method for the answering side to submit an answer."""
        self.answer_queue.put(answer)

    def pending_question(self, timeout: float | None = None) -> Question | None:
        """Convenience method to retrieve a pending question, if any."""
        try:
            return self.question_queue.get(timeout=timeout)
        except queue.Empty:
            return None
