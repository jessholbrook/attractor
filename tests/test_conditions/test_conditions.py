"""Tests for the condition expression evaluator."""

import pytest

from attractor.conditions import evaluate_condition, resolve_key
from attractor.model.context import Context
from attractor.model.outcome import Outcome, Status


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _outcome(status: Status, preferred_label: str = "") -> Outcome:
    return Outcome(status=status, preferred_label=preferred_label)


def _context(data: dict | None = None) -> Context:
    return Context(initial=data)


# ---------------------------------------------------------------------------
# evaluate_condition
# ---------------------------------------------------------------------------


class TestEmptyCondition:
    def test_empty_string_returns_true(self):
        assert evaluate_condition("", _outcome(Status.SUCCESS), _context()) is True

    def test_whitespace_only_returns_true(self):
        assert evaluate_condition("   ", _outcome(Status.FAIL), _context()) is True

    def test_none_like_empty(self):
        # Edge case: None-ish -- the function signature says str, but let's
        # confirm empty string behaviour.
        assert evaluate_condition("", _outcome(Status.RETRY), _context()) is True


class TestOutcomeMatching:
    def test_outcome_equals_success(self):
        assert evaluate_condition("outcome=success", _outcome(Status.SUCCESS), _context()) is True

    def test_outcome_equals_fail(self):
        assert evaluate_condition("outcome=fail", _outcome(Status.FAIL), _context()) is True

    def test_outcome_equals_partial_success(self):
        assert evaluate_condition(
            "outcome=partial_success", _outcome(Status.PARTIAL_SUCCESS), _context()
        ) is True

    def test_outcome_equals_retry(self):
        assert evaluate_condition("outcome=retry", _outcome(Status.RETRY), _context()) is True

    def test_outcome_equals_skipped(self):
        assert evaluate_condition("outcome=skipped", _outcome(Status.SKIPPED), _context()) is True

    def test_outcome_mismatch(self):
        assert evaluate_condition("outcome=success", _outcome(Status.FAIL), _context()) is False

    def test_outcome_not_equals_success_when_fail(self):
        assert evaluate_condition("outcome!=success", _outcome(Status.FAIL), _context()) is True

    def test_outcome_not_equals_success_when_success(self):
        assert evaluate_condition("outcome!=success", _outcome(Status.SUCCESS), _context()) is False


class TestPreferredLabel:
    def test_preferred_label_match(self):
        assert evaluate_condition(
            "preferred_label=Fix", _outcome(Status.SUCCESS, preferred_label="Fix"), _context()
        ) is True

    def test_preferred_label_mismatch(self):
        assert evaluate_condition(
            "preferred_label=Fix", _outcome(Status.SUCCESS, preferred_label="Retry"), _context()
        ) is False

    def test_preferred_label_case_sensitive(self):
        assert evaluate_condition(
            "preferred_label=fix", _outcome(Status.SUCCESS, preferred_label="Fix"), _context()
        ) is False


class TestContextKeys:
    def test_context_dot_key_resolves(self):
        ctx = _context({"tests_passed": "true"})
        assert evaluate_condition(
            "context.tests_passed=true", _outcome(Status.SUCCESS), ctx
        ) is True

    def test_context_dot_key_full_key(self):
        """context.get('context.tests_passed') should also work."""
        ctx = _context({"context.tests_passed": "true"})
        assert evaluate_condition(
            "context.tests_passed=true", _outcome(Status.SUCCESS), ctx
        ) is True

    def test_bare_key_resolves_from_context(self):
        ctx = _context({"role": "admin"})
        assert evaluate_condition("role=admin", _outcome(Status.SUCCESS), ctx) is True

    def test_missing_context_key_is_empty_string(self):
        ctx = _context()
        assert evaluate_condition("missing_key=", _outcome(Status.SUCCESS), ctx) is True

    def test_missing_context_key_never_equals_nonempty(self):
        ctx = _context()
        assert evaluate_condition(
            "missing_key=something", _outcome(Status.SUCCESS), ctx
        ) is False


class TestMultipleClauses:
    def test_two_clauses_both_true(self):
        ctx = _context({"tests_passed": "true"})
        assert evaluate_condition(
            "outcome=success && context.tests_passed=true",
            _outcome(Status.SUCCESS),
            ctx,
        ) is True

    def test_two_clauses_first_false(self):
        ctx = _context({"tests_passed": "true"})
        assert evaluate_condition(
            "outcome=fail && context.tests_passed=true",
            _outcome(Status.SUCCESS),
            ctx,
        ) is False

    def test_two_clauses_second_false(self):
        ctx = _context({"tests_passed": "false"})
        assert evaluate_condition(
            "outcome=success && context.tests_passed=true",
            _outcome(Status.SUCCESS),
            ctx,
        ) is False

    def test_three_clauses_all_true(self):
        ctx = _context({"tests_passed": "true", "env": "prod"})
        assert evaluate_condition(
            "outcome=success && context.tests_passed=true && env=prod",
            _outcome(Status.SUCCESS),
            ctx,
        ) is True


class TestWhitespace:
    def test_spaces_around_operator(self):
        assert evaluate_condition(
            "outcome = success", _outcome(Status.SUCCESS), _context()
        ) is True

    def test_spaces_around_clauses(self):
        ctx = _context({"x": "1"})
        assert evaluate_condition(
            "  outcome=success  &&  x=1  ", _outcome(Status.SUCCESS), ctx
        ) is True

    def test_no_spaces(self):
        assert evaluate_condition(
            "outcome=fail", _outcome(Status.FAIL), _context()
        ) is True


# ---------------------------------------------------------------------------
# resolve_key
# ---------------------------------------------------------------------------


class TestResolveKey:
    def test_outcome_key(self):
        assert resolve_key("outcome", _outcome(Status.SUCCESS), _context()) == "success"

    def test_preferred_label_key(self):
        o = _outcome(Status.SUCCESS, preferred_label="Go")
        assert resolve_key("preferred_label", o, _context()) == "Go"

    def test_context_dot_key(self):
        ctx = _context({"lang": "en"})
        assert resolve_key("context.lang", _outcome(Status.SUCCESS), ctx) == "en"

    def test_bare_key(self):
        ctx = _context({"lang": "en"})
        assert resolve_key("lang", _outcome(Status.SUCCESS), ctx) == "en"

    def test_missing_key_returns_empty(self):
        assert resolve_key("nope", _outcome(Status.SUCCESS), _context()) == ""

    def test_missing_context_dot_key_returns_empty(self):
        assert resolve_key("context.nope", _outcome(Status.SUCCESS), _context()) == ""
