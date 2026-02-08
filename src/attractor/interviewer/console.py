"""ConsoleInterviewer: prompts the user at the terminal via input()."""

from __future__ import annotations

import signal
from typing import Any

from attractor.interviewer.accelerators import parse_accelerator
from attractor.model.question import Answer, AnswerValue, Option, Question, QuestionType


class ConsoleInterviewer:
    """Interviewer that uses stdin/stdout for interactive prompts.

    Displays the question text and options with accelerator keys,
    reads user input, and parses it into an Answer.
    Supports an optional timeout (POSIX only, via SIGALRM).
    """

    def ask(self, question: Question) -> Answer:
        print(f"\n{'=' * 60}")
        print(f"  {question.text}")
        print(f"{'=' * 60}")

        if question.type in (QuestionType.YES_NO, QuestionType.CONFIRMATION):
            return self._ask_yes_no(question)
        elif question.type == QuestionType.MULTIPLE_CHOICE:
            return self._ask_multiple_choice(question)
        else:
            return self._ask_freeform(question)

    def _ask_yes_no(self, question: Question) -> Answer:
        prompt = "  [Y]es / [N]o: "
        raw = self._get_input(prompt, question.timeout_seconds)
        if raw is None:
            return Answer(value=AnswerValue.TIMEOUT, text="")
        raw = raw.strip().lower()
        if raw in ("y", "yes"):
            return Answer(value=AnswerValue.YES, text="YES")
        elif raw in ("n", "no"):
            return Answer(value=AnswerValue.NO, text="NO")
        elif raw == "":
            if question.default:
                if question.default.lower() in ("y", "yes"):
                    return Answer(value=AnswerValue.YES, text="YES")
                return Answer(value=AnswerValue.NO, text="NO")
            return Answer(value=AnswerValue.SKIPPED, text="")
        return Answer(value=AnswerValue.NO, text=raw)

    def _ask_multiple_choice(self, question: Question) -> Answer:
        accel_map: dict[str, Option] = {}
        for opt in question.options:
            key, clean = parse_accelerator(opt.label)
            if key:
                accel_map[key.lower()] = opt
            print(f"  [{opt.key}] {opt.label}")

        prompt = "  Choice: "
        raw = self._get_input(prompt, question.timeout_seconds)
        if raw is None:
            return Answer(value=AnswerValue.TIMEOUT, text="")

        raw = raw.strip()

        # Match by option key
        for opt in question.options:
            if raw == opt.key:
                return Answer(value=opt.key, selected_option=opt, text=opt.label)

        # Match by accelerator
        if raw.lower() in accel_map:
            opt = accel_map[raw.lower()]
            return Answer(value=opt.key, selected_option=opt, text=opt.label)

        # Match by label (case-insensitive)
        for opt in question.options:
            if opt.label.strip().lower() == raw.lower():
                return Answer(value=opt.key, selected_option=opt, text=opt.label)

        # Default if nothing matched
        if question.default:
            for opt in question.options:
                if opt.key == question.default or opt.label.strip().lower() == question.default.lower():
                    return Answer(value=opt.key, selected_option=opt, text=opt.label)

        return Answer(value=raw, text=raw)

    def _ask_freeform(self, question: Question) -> Answer:
        prompt = "  > "
        raw = self._get_input(prompt, question.timeout_seconds)
        if raw is None:
            return Answer(value=AnswerValue.TIMEOUT, text="")
        raw = raw.strip()
        if not raw and question.default:
            raw = question.default
        return Answer(value=raw, text=raw)

    @staticmethod
    def _get_input(prompt: str, timeout: float | None) -> str | None:
        """Read input with optional timeout. Returns None on timeout."""
        if timeout is None or timeout <= 0:
            return input(prompt)

        # POSIX timeout via SIGALRM
        def _handler(signum: Any, frame: Any) -> None:
            raise TimeoutError

        old_handler = signal.signal(signal.SIGALRM, _handler)
        try:
            signal.alarm(int(timeout))
            result = input(prompt)
            signal.alarm(0)
            return result
        except (TimeoutError, EOFError):
            return None
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
