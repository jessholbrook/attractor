"""WaitHumanHandler: pauses execution to ask the user a question."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from attractor.model.context import Context
from attractor.model.graph import Graph, Node
from attractor.model.outcome import Outcome, Status
from attractor.model.question import Answer, Option, Question, QuestionType


class Interviewer(Protocol):
    """Protocol for objects that can ask a question and return an answer."""

    def ask(self, question: Question) -> Answer: ...


class WaitHumanHandler:
    """Handler for wait.human (hexagon) nodes.

    Creates a Question from the node's prompt/label and options from
    outgoing edge labels, then delegates to an Interviewer to get an Answer.
    Sets preferred_label to the answer text or selected option label.
    """

    def __init__(self, interviewer: Interviewer) -> None:
        self._interviewer = interviewer

    def execute(self, node: Node, context: Context, graph: Graph, logs_root: Path) -> Outcome:
        text = node.prompt or node.label
        outgoing = graph.outgoing_edges(node.id)

        # Build options from outgoing edge labels
        options = [
            Option(key=str(i), label=edge.label)
            for i, edge in enumerate(outgoing)
            if edge.label
        ]

        # Determine question type
        if len(options) == 2 and _is_yes_no_pair(options):
            q_type = QuestionType.YES_NO
        elif options:
            q_type = QuestionType.MULTIPLE_CHOICE
        else:
            q_type = QuestionType.FREEFORM

        question = Question(
            text=text,
            type=q_type,
            options=options,
            stage=node.id,
        )

        answer = self._interviewer.ask(question)

        # Determine the preferred label from the answer
        preferred = ""
        if answer.selected_option:
            preferred = answer.selected_option.label
        elif answer.text:
            preferred = answer.text
        elif answer.value is not None:
            preferred = str(answer.value.value) if hasattr(answer.value, "value") else str(answer.value)

        return Outcome(
            status=Status.SUCCESS,
            preferred_label=preferred,
            context_updates={f"{node.id}.answer": preferred},
        )


def _is_yes_no_pair(options: list[Option]) -> bool:
    """Check if two options form a yes/no pair."""
    labels = {opt.label.strip().lower() for opt in options}
    return labels <= {"yes", "no", "y", "n", "true", "false"}
