"""Tests for the 5-step edge selection algorithm."""

import pytest

from attractor.engine.edge_selector import select_edge, _normalize_label
from attractor.model.context import Context
from attractor.model.graph import Edge
from attractor.model.outcome import Outcome, Status


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _outcome(
    status: Status = Status.SUCCESS,
    preferred_label: str = "",
    suggested_next_ids: list[str] | None = None,
) -> Outcome:
    return Outcome(
        status=status,
        preferred_label=preferred_label,
        suggested_next_ids=suggested_next_ids or [],
    )


def _edge(
    from_node: str = "A",
    to_node: str = "B",
    label: str = "",
    condition: str = "",
    weight: int = 0,
) -> Edge:
    return Edge(
        from_node=from_node,
        to_node=to_node,
        label=label,
        condition=condition,
        weight=weight,
    )


def _context(data: dict | None = None) -> Context:
    return Context(initial=data)


# ---------------------------------------------------------------------------
# No edges
# ---------------------------------------------------------------------------

class TestNoEdges:
    def test_empty_list_returns_none(self):
        assert select_edge([], _outcome(), _context()) is None


# ---------------------------------------------------------------------------
# Single unconditional edge
# ---------------------------------------------------------------------------

class TestSingleEdge:
    def test_single_unconditional_edge_selected(self):
        e = _edge(to_node="B")
        result = select_edge([e], _outcome(), _context())
        assert result is e


# ---------------------------------------------------------------------------
# Step 1: Condition matching
# ---------------------------------------------------------------------------

class TestConditionMatch:
    def test_condition_match_wins_over_weight(self):
        high_weight = _edge(to_node="high", weight=100)
        cond_edge = _edge(to_node="cond", condition="outcome=success", weight=1)
        result = select_edge(
            [high_weight, cond_edge],
            _outcome(Status.SUCCESS),
            _context(),
        )
        assert result is cond_edge

    def test_condition_no_match_falls_through(self):
        cond_edge = _edge(to_node="cond", condition="outcome=fail")
        fallback = _edge(to_node="fallback", weight=1)
        result = select_edge(
            [cond_edge, fallback],
            _outcome(Status.SUCCESS),
            _context(),
        )
        assert result is fallback

    def test_multiple_condition_edges_best_by_weight(self):
        e1 = _edge(to_node="low", condition="outcome=success", weight=1)
        e2 = _edge(to_node="high", condition="outcome=success", weight=10)
        result = select_edge([e1, e2], _outcome(Status.SUCCESS), _context())
        assert result is e2

    def test_multiple_condition_edges_lexical_tiebreak(self):
        e1 = _edge(to_node="B", condition="outcome=success", weight=5)
        e2 = _edge(to_node="A", condition="outcome=success", weight=5)
        result = select_edge([e1, e2], _outcome(Status.SUCCESS), _context())
        assert result is e2  # A < B lexically


# ---------------------------------------------------------------------------
# Step 2: Preferred label matching
# ---------------------------------------------------------------------------

class TestPreferredLabel:
    def test_preferred_label_exact_match(self):
        e1 = _edge(to_node="yes_node", label="Yes")
        e2 = _edge(to_node="no_node", label="No")
        result = select_edge(
            [e1, e2],
            _outcome(preferred_label="Yes"),
            _context(),
        )
        assert result is e1

    def test_preferred_label_normalized_with_bracket_prefix(self):
        """[Y] Yes should match preferred_label='yes' after normalization."""
        e1 = _edge(to_node="yes_node", label="[Y] Yes")
        e2 = _edge(to_node="no_node", label="[N] No")
        result = select_edge(
            [e1, e2],
            _outcome(preferred_label="yes"),
            _context(),
        )
        assert result is e1

    def test_preferred_label_case_insensitive(self):
        e = _edge(to_node="target", label="APPROVE")
        result = select_edge(
            [e],
            _outcome(preferred_label="approve"),
            _context(),
        )
        assert result is e

    def test_preferred_label_no_match_falls_through(self):
        e = _edge(to_node="target", label="Yes")
        result = select_edge(
            [e],
            _outcome(preferred_label="No"),
            _context(),
        )
        # Falls through to weight/lexical
        assert result is e  # only edge, so selected by fallback


# ---------------------------------------------------------------------------
# Step 3: Suggested next IDs
# ---------------------------------------------------------------------------

class TestSuggestedNextIds:
    def test_suggested_id_matches_edge_target(self):
        e1 = _edge(to_node="alpha")
        e2 = _edge(to_node="beta")
        result = select_edge(
            [e1, e2],
            _outcome(suggested_next_ids=["beta"]),
            _context(),
        )
        assert result is e2

    def test_suggested_id_first_match_wins(self):
        e1 = _edge(to_node="alpha")
        e2 = _edge(to_node="beta")
        result = select_edge(
            [e1, e2],
            _outcome(suggested_next_ids=["alpha", "beta"]),
            _context(),
        )
        assert result is e1

    def test_suggested_id_no_match_falls_through(self):
        e = _edge(to_node="gamma", weight=5)
        result = select_edge(
            [e],
            _outcome(suggested_next_ids=["delta"]),
            _context(),
        )
        assert result is e  # falls through to weight


# ---------------------------------------------------------------------------
# Steps 4 & 5: Weight and lexical tiebreak
# ---------------------------------------------------------------------------

class TestWeightAndLexical:
    def test_highest_weight_wins(self):
        e1 = _edge(to_node="low", weight=1)
        e2 = _edge(to_node="high", weight=10)
        result = select_edge([e1, e2], _outcome(), _context())
        assert result is e2

    def test_lexical_tiebreak_alphabetical(self):
        e1 = _edge(to_node="C", weight=5)
        e2 = _edge(to_node="A", weight=5)
        e3 = _edge(to_node="B", weight=5)
        result = select_edge([e1, e2, e3], _outcome(), _context())
        assert result is e2  # A < B < C

    def test_unconditional_edges_only_for_weight(self):
        """Unconditional edges are preferred for weight selection, not conditional ones."""
        cond = _edge(to_node="cond", condition="outcome=fail", weight=100)
        uncond = _edge(to_node="uncond", weight=1)
        result = select_edge(
            [cond, uncond],
            _outcome(Status.SUCCESS),
            _context(),
        )
        assert result is uncond


# ---------------------------------------------------------------------------
# Mixed scenarios
# ---------------------------------------------------------------------------

class TestMixed:
    def test_conditional_wins_over_everything(self):
        """Condition match has highest priority over label, id, and weight."""
        cond = _edge(to_node="cond", condition="outcome=success", weight=1)
        labeled = _edge(to_node="labeled", label="Yes", weight=50)
        heavy = _edge(to_node="heavy", weight=100)
        result = select_edge(
            [cond, labeled, heavy],
            _outcome(Status.SUCCESS, preferred_label="Yes"),
            _context(),
        )
        assert result is cond

    def test_fallback_to_all_edges_when_no_unconditional(self):
        """If all edges have conditions and none match, fall back to best of all."""
        e1 = _edge(to_node="A", condition="outcome=fail", weight=5)
        e2 = _edge(to_node="B", condition="outcome=fail", weight=10)
        result = select_edge([e1, e2], _outcome(Status.SUCCESS), _context())
        assert result is e2  # fallback: best by weight


# ---------------------------------------------------------------------------
# Label normalization
# ---------------------------------------------------------------------------

class TestNormalizeLabel:
    def test_lowercase(self):
        assert _normalize_label("YES") == "yes"

    def test_strip_bracket_prefix(self):
        assert _normalize_label("[Y] Yes") == "yes"

    def test_strip_paren_prefix(self):
        assert _normalize_label("y) Yes") == "yes"

    def test_strip_dash_prefix(self):
        assert _normalize_label("y - Yes") == "yes"

    def test_strip_whitespace(self):
        assert _normalize_label("  Yes  ") == "yes"

    def test_no_prefix(self):
        assert _normalize_label("approve") == "approve"
