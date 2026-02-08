"""Tests for all interviewer implementations and accelerator parsing."""

from __future__ import annotations

import queue
import threading

import pytest

from attractor.model.question import Answer, AnswerValue, Option, Question, QuestionType


# ===========================================================================
# AutoApproveInterviewer
# ===========================================================================


class TestAutoApproveInterviewer:
    def test_yes_no_returns_yes(self) -> None:
        from attractor.interviewer.auto_approve import AutoApproveInterviewer

        interviewer = AutoApproveInterviewer()
        q = Question(text="Continue?", type=QuestionType.YES_NO)

        answer = interviewer.ask(q)

        assert answer.value is AnswerValue.YES
        assert answer.text == "YES"

    def test_confirmation_returns_yes(self) -> None:
        from attractor.interviewer.auto_approve import AutoApproveInterviewer

        interviewer = AutoApproveInterviewer()
        q = Question(text="Are you sure?", type=QuestionType.CONFIRMATION)

        answer = interviewer.ask(q)

        assert answer.value is AnswerValue.YES

    def test_multiple_choice_returns_first_option(self) -> None:
        from attractor.interviewer.auto_approve import AutoApproveInterviewer

        interviewer = AutoApproveInterviewer()
        options = [
            Option(key="1", label="Python"),
            Option(key="2", label="Rust"),
            Option(key="3", label="TypeScript"),
        ]
        q = Question(text="Choose language", type=QuestionType.MULTIPLE_CHOICE, options=options)

        answer = interviewer.ask(q)

        assert answer.selected_option is not None
        assert answer.selected_option.label == "Python"
        assert answer.text == "Python"

    def test_freeform_returns_approved(self) -> None:
        from attractor.interviewer.auto_approve import AutoApproveInterviewer

        interviewer = AutoApproveInterviewer()
        q = Question(text="Enter your name", type=QuestionType.FREEFORM)

        answer = interviewer.ask(q)

        assert answer.text == "approved"

    def test_multiple_choice_no_options(self) -> None:
        from attractor.interviewer.auto_approve import AutoApproveInterviewer

        interviewer = AutoApproveInterviewer()
        q = Question(text="Choose", type=QuestionType.MULTIPLE_CHOICE)

        answer = interviewer.ask(q)

        assert answer.text == "approved"


# ===========================================================================
# CallbackInterviewer
# ===========================================================================


class TestCallbackInterviewer:
    def test_delegates_to_callback(self) -> None:
        from attractor.interviewer.callback import CallbackInterviewer

        def my_callback(question: Question) -> Answer:
            return Answer(value="custom", text=f"answer to: {question.text}")

        interviewer = CallbackInterviewer(my_callback)
        q = Question(text="What color?", type=QuestionType.FREEFORM)

        answer = interviewer.ask(q)

        assert answer.text == "answer to: What color?"

    def test_callback_receives_full_question(self) -> None:
        from attractor.interviewer.callback import CallbackInterviewer

        received: list[Question] = []

        def capture(question: Question) -> Answer:
            received.append(question)
            return Answer(text="ok")

        interviewer = CallbackInterviewer(capture)
        options = [Option(key="a", label="Alpha")]
        q = Question(
            text="Pick one",
            type=QuestionType.MULTIPLE_CHOICE,
            options=options,
            stage="stage1",
        )

        interviewer.ask(q)

        assert len(received) == 1
        assert received[0].text == "Pick one"
        assert received[0].stage == "stage1"
        assert len(received[0].options) == 1


# ===========================================================================
# QueueInterviewer
# ===========================================================================


class TestQueueInterviewer:
    def test_exchanges_via_queues(self) -> None:
        from attractor.interviewer.queue_interviewer import QueueInterviewer

        interviewer = QueueInterviewer()
        q = Question(text="Ready?", type=QuestionType.YES_NO)

        # Simulate the answering side in a thread
        def answerer():
            question = interviewer.pending_question(timeout=2.0)
            assert question is not None
            assert question.text == "Ready?"
            interviewer.respond(Answer(value=AnswerValue.YES, text="YES"))

        thread = threading.Thread(target=answerer)
        thread.start()

        answer = interviewer.ask(q)
        thread.join(timeout=3.0)

        assert answer.value is AnswerValue.YES
        assert answer.text == "YES"

    def test_timeout_returns_timeout_answer(self) -> None:
        from attractor.interviewer.queue_interviewer import QueueInterviewer

        interviewer = QueueInterviewer(timeout=0.1)
        q = Question(text="Will timeout", type=QuestionType.FREEFORM)

        answer = interviewer.ask(q)

        assert answer.value is AnswerValue.TIMEOUT

    def test_respond_convenience(self) -> None:
        from attractor.interviewer.queue_interviewer import QueueInterviewer

        interviewer = QueueInterviewer()

        # Pre-load an answer
        interviewer.respond(Answer(text="preloaded"))

        q = Question(text="Test", type=QuestionType.FREEFORM)
        answer = interviewer.ask(q)

        assert answer.text == "preloaded"

    def test_pending_question_returns_none_on_empty(self) -> None:
        from attractor.interviewer.queue_interviewer import QueueInterviewer

        interviewer = QueueInterviewer()
        result = interviewer.pending_question(timeout=0.05)
        assert result is None


# ===========================================================================
# RecordingInterviewer
# ===========================================================================


class TestRecordingInterviewer:
    def test_records_transcript(self) -> None:
        from attractor.interviewer.auto_approve import AutoApproveInterviewer
        from attractor.interviewer.recording import RecordingInterviewer

        inner = AutoApproveInterviewer()
        recorder = RecordingInterviewer(inner)

        q1 = Question(text="Q1", type=QuestionType.YES_NO)
        q2 = Question(text="Q2", type=QuestionType.CONFIRMATION)

        recorder.ask(q1)
        recorder.ask(q2)

        transcript = recorder.transcript()
        assert len(transcript) == 2
        assert transcript[0].question.text == "Q1"
        assert transcript[0].answer.value is AnswerValue.YES
        assert transcript[1].question.text == "Q2"

    def test_delegates_to_inner(self) -> None:
        from attractor.interviewer.callback import CallbackInterviewer
        from attractor.interviewer.recording import RecordingInterviewer

        def echo(q: Question) -> Answer:
            return Answer(text=q.text)

        recorder = RecordingInterviewer(CallbackInterviewer(echo))
        q = Question(text="Hello", type=QuestionType.FREEFORM)

        answer = recorder.ask(q)

        assert answer.text == "Hello"

    def test_clear_empties_transcript(self) -> None:
        from attractor.interviewer.auto_approve import AutoApproveInterviewer
        from attractor.interviewer.recording import RecordingInterviewer

        recorder = RecordingInterviewer(AutoApproveInterviewer())
        recorder.ask(Question(text="Q1", type=QuestionType.YES_NO))
        assert len(recorder.transcript()) == 1

        recorder.clear()
        assert len(recorder.transcript()) == 0

    def test_transcript_returns_copy(self) -> None:
        from attractor.interviewer.auto_approve import AutoApproveInterviewer
        from attractor.interviewer.recording import RecordingInterviewer

        recorder = RecordingInterviewer(AutoApproveInterviewer())
        recorder.ask(Question(text="Q1", type=QuestionType.YES_NO))

        t1 = recorder.transcript()
        t2 = recorder.transcript()
        assert t1 == t2
        assert t1 is not t2  # different list objects


# ===========================================================================
# parse_accelerator
# ===========================================================================


class TestParseAccelerator:
    def test_bracket_pattern(self) -> None:
        from attractor.interviewer.accelerators import parse_accelerator

        key, label = parse_accelerator("[Y] Yes please")
        assert key == "Y"
        assert label == "Yes please"

    def test_paren_pattern(self) -> None:
        from attractor.interviewer.accelerators import parse_accelerator

        key, label = parse_accelerator("N) No thanks")
        assert key == "N"
        assert label == "No thanks"

    def test_dash_pattern(self) -> None:
        from attractor.interviewer.accelerators import parse_accelerator

        key, label = parse_accelerator("A - Accept")
        assert key == "A"
        assert label == "Accept"

    def test_no_accelerator(self) -> None:
        from attractor.interviewer.accelerators import parse_accelerator

        key, label = parse_accelerator("Plain label")
        assert key == ""
        assert label == "Plain label"

    def test_numeric_key(self) -> None:
        from attractor.interviewer.accelerators import parse_accelerator

        key, label = parse_accelerator("[1] First option")
        assert key == "1"
        assert label == "First option"

    def test_lowercase_bracket(self) -> None:
        from attractor.interviewer.accelerators import parse_accelerator

        key, label = parse_accelerator("[y] yes")
        assert key == "y"
        assert label == "yes"

    def test_strips_whitespace(self) -> None:
        from attractor.interviewer.accelerators import parse_accelerator

        key, label = parse_accelerator("  [X] Extra spaces  ")
        assert key == "X"
        assert label == "Extra spaces"

    def test_empty_string(self) -> None:
        from attractor.interviewer.accelerators import parse_accelerator

        key, label = parse_accelerator("")
        assert key == ""
        assert label == ""


# ===========================================================================
# Imports from __init__.py
# ===========================================================================


class TestInterviewerExports:
    def test_all_exports_importable(self) -> None:
        from attractor.interviewer import (
            AutoApproveInterviewer,
            CallbackInterviewer,
            ConsoleInterviewer,
            Interviewer,
            QAPair,
            QueueInterviewer,
            RecordingInterviewer,
            parse_accelerator,
        )

        assert AutoApproveInterviewer is not None
        assert CallbackInterviewer is not None
        assert ConsoleInterviewer is not None
        assert Interviewer is not None
        assert QueueInterviewer is not None
        assert RecordingInterviewer is not None
        assert QAPair is not None
        assert parse_accelerator is not None


class TestHandlerExports:
    def test_all_exports_importable(self) -> None:
        from attractor.handlers import (
            CodergenBackend,
            CodergenHandler,
            ConditionalHandler,
            ExitHandler,
            FanInHandler,
            Handler,
            Interviewer,
            ParallelHandler,
            StackManagerHandler,
            StartHandler,
            StubBackend,
            ToolHandler,
            WaitHumanHandler,
            create_default_registry,
        )

        assert StartHandler is not None
        assert ExitHandler is not None
        assert ConditionalHandler is not None
        assert CodergenHandler is not None
        assert CodergenBackend is not None
        assert StubBackend is not None
        assert WaitHumanHandler is not None
        assert ParallelHandler is not None
        assert FanInHandler is not None
        assert ToolHandler is not None
        assert StackManagerHandler is not None
        assert Handler is not None
        assert Interviewer is not None
        assert create_default_registry is not None
