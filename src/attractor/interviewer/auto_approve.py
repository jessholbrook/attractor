"""AutoApproveInterviewer: automatically approves all questions."""

from __future__ import annotations

from attractor.model.question import Answer, AnswerValue, Question, QuestionType


class AutoApproveInterviewer:
    """Interviewer that auto-approves without user interaction.

    - YES_NO / CONFIRMATION: returns YES
    - MULTIPLE_CHOICE: returns the first option
    - FREEFORM: returns "approved"
    """

    def ask(self, question: Question) -> Answer:
        if question.type in (QuestionType.YES_NO, QuestionType.CONFIRMATION):
            return Answer(value=AnswerValue.YES, text="YES")

        if question.type == QuestionType.MULTIPLE_CHOICE and question.options:
            first = question.options[0]
            return Answer(
                value=first.key,
                selected_option=first,
                text=first.label,
            )

        # FREEFORM or MULTIPLE_CHOICE with no options
        return Answer(value="approved", text="approved")
