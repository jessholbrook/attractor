"""Tests for graph validation rules and the validator."""

import pytest

from attractor.model.diagnostic import Diagnostic, Severity
from attractor.model.graph import Edge, Graph, Node
from attractor.validation import validate, validate_or_raise, ValidationError
from attractor.validation.rules import (
    check_condition_syntax,
    check_edge_target_exists,
    check_exit_no_outgoing,
    check_fidelity_valid,
    check_goal_gate_has_retry,
    check_prompt_on_llm_nodes,
    check_reachability,
    check_retry_target_exists,
    check_start_no_incoming,
    check_start_node,
    check_terminal_node,
    check_type_known,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_graph(**overrides) -> Graph:
    """Build a minimal valid graph (start -> exit)."""
    defaults = dict(
        name="Test",
        nodes={
            "start": Node(id="start", shape="Mdiamond", label="Start"),
            "exit": Node(id="exit", shape="Msquare", label="Exit"),
        },
        edges=[Edge(from_node="start", to_node="exit")],
        attributes={},
    )
    defaults.update(overrides)
    return Graph(**defaults)


# ---------------------------------------------------------------------------
# check_start_node
# ---------------------------------------------------------------------------


class TestCheckStartNode:
    def test_no_start_node(self):
        g = Graph(
            name="T",
            nodes={"a": Node(id="a"), "b": Node(id="b", shape="Msquare")},
            edges=[Edge(from_node="a", to_node="b")],
        )
        diags = check_start_node(g)
        assert len(diags) == 1
        assert diags[0].severity is Severity.ERROR
        assert "No start node" in diags[0].message

    def test_multiple_start_nodes(self):
        g = Graph(
            name="T",
            nodes={
                "s1": Node(id="s1", shape="Mdiamond"),
                "s2": Node(id="s2", shape="Mdiamond"),
                "exit": Node(id="exit", shape="Msquare"),
            },
            edges=[
                Edge(from_node="s1", to_node="exit"),
                Edge(from_node="s2", to_node="exit"),
            ],
        )
        diags = check_start_node(g)
        assert len(diags) == 2
        assert all(d.severity is Severity.ERROR for d in diags)

    def test_exactly_one_start_node(self):
        g = _minimal_graph()
        diags = check_start_node(g)
        assert diags == []


# ---------------------------------------------------------------------------
# check_terminal_node
# ---------------------------------------------------------------------------


class TestCheckTerminalNode:
    def test_no_exit_node(self):
        g = Graph(
            name="T",
            nodes={
                "start": Node(id="start", shape="Mdiamond"),
                "work": Node(id="work"),
            },
            edges=[Edge(from_node="start", to_node="work")],
        )
        diags = check_terminal_node(g)
        assert len(diags) == 1
        assert diags[0].severity is Severity.ERROR

    def test_has_exit_node(self):
        g = _minimal_graph()
        assert check_terminal_node(g) == []


# ---------------------------------------------------------------------------
# check_reachability
# ---------------------------------------------------------------------------


class TestCheckReachability:
    def test_unreachable_node(self):
        g = Graph(
            name="T",
            nodes={
                "start": Node(id="start", shape="Mdiamond"),
                "exit": Node(id="exit", shape="Msquare"),
                "orphan": Node(id="orphan"),
            },
            edges=[Edge(from_node="start", to_node="exit")],
        )
        diags = check_reachability(g)
        assert len(diags) == 1
        assert diags[0].node_id == "orphan"
        assert diags[0].severity is Severity.ERROR

    def test_all_reachable(self):
        g = _minimal_graph()
        assert check_reachability(g) == []


# ---------------------------------------------------------------------------
# check_edge_target_exists
# ---------------------------------------------------------------------------


class TestCheckEdgeTargetExists:
    def test_edge_to_nonexistent_node(self):
        g = Graph(
            name="T",
            nodes={"start": Node(id="start", shape="Mdiamond")},
            edges=[Edge(from_node="start", to_node="ghost")],
        )
        diags = check_edge_target_exists(g)
        assert len(diags) == 1
        assert diags[0].severity is Severity.ERROR
        assert "ghost" in diags[0].message

    def test_all_edges_valid(self):
        g = _minimal_graph()
        assert check_edge_target_exists(g) == []


# ---------------------------------------------------------------------------
# check_start_no_incoming
# ---------------------------------------------------------------------------


class TestCheckStartNoIncoming:
    def test_start_with_incoming(self):
        g = Graph(
            name="T",
            nodes={
                "start": Node(id="start", shape="Mdiamond"),
                "exit": Node(id="exit", shape="Msquare"),
            },
            edges=[
                Edge(from_node="start", to_node="exit"),
                Edge(from_node="exit", to_node="start"),
            ],
        )
        diags = check_start_no_incoming(g)
        assert len(diags) == 1
        assert diags[0].severity is Severity.ERROR

    def test_start_no_incoming(self):
        g = _minimal_graph()
        assert check_start_no_incoming(g) == []


# ---------------------------------------------------------------------------
# check_exit_no_outgoing
# ---------------------------------------------------------------------------


class TestCheckExitNoOutgoing:
    def test_exit_with_outgoing(self):
        g = Graph(
            name="T",
            nodes={
                "start": Node(id="start", shape="Mdiamond"),
                "exit": Node(id="exit", shape="Msquare"),
                "extra": Node(id="extra"),
            },
            edges=[
                Edge(from_node="start", to_node="exit"),
                Edge(from_node="exit", to_node="extra"),
            ],
        )
        diags = check_exit_no_outgoing(g)
        assert len(diags) == 1
        assert diags[0].severity is Severity.ERROR

    def test_exit_no_outgoing(self):
        g = _minimal_graph()
        assert check_exit_no_outgoing(g) == []


# ---------------------------------------------------------------------------
# check_condition_syntax
# ---------------------------------------------------------------------------


class TestCheckConditionSyntax:
    def test_valid_condition(self):
        g = Graph(
            name="T",
            nodes={
                "a": Node(id="a"),
                "b": Node(id="b"),
            },
            edges=[Edge(from_node="a", to_node="b", condition="outcome=success")],
        )
        assert check_condition_syntax(g) == []

    def test_valid_not_equals(self):
        g = Graph(
            name="T",
            nodes={"a": Node(id="a"), "b": Node(id="b")},
            edges=[Edge(from_node="a", to_node="b", condition="outcome!=success")],
        )
        assert check_condition_syntax(g) == []

    def test_valid_compound(self):
        g = Graph(
            name="T",
            nodes={"a": Node(id="a"), "b": Node(id="b")},
            edges=[
                Edge(
                    from_node="a",
                    to_node="b",
                    condition="outcome=success && context.tests=true",
                )
            ],
        )
        assert check_condition_syntax(g) == []

    def test_invalid_condition(self):
        g = Graph(
            name="T",
            nodes={"a": Node(id="a"), "b": Node(id="b")},
            edges=[Edge(from_node="a", to_node="b", condition="this is garbage")],
        )
        diags = check_condition_syntax(g)
        assert len(diags) == 1
        assert diags[0].severity is Severity.ERROR

    def test_empty_condition_ok(self):
        g = Graph(
            name="T",
            nodes={"a": Node(id="a"), "b": Node(id="b")},
            edges=[Edge(from_node="a", to_node="b", condition="")],
        )
        assert check_condition_syntax(g) == []


# ---------------------------------------------------------------------------
# check_type_known
# ---------------------------------------------------------------------------


class TestCheckTypeKnown:
    def test_unknown_type(self):
        g = Graph(
            name="T",
            nodes={"a": Node(id="a", type="totally_made_up")},
        )
        diags = check_type_known(g)
        assert len(diags) == 1
        assert diags[0].severity is Severity.WARNING

    def test_known_type(self):
        g = Graph(
            name="T",
            nodes={"a": Node(id="a", type="codergen")},
        )
        assert check_type_known(g) == []

    def test_empty_type_ok(self):
        g = Graph(name="T", nodes={"a": Node(id="a")})
        assert check_type_known(g) == []


# ---------------------------------------------------------------------------
# check_fidelity_valid
# ---------------------------------------------------------------------------


class TestCheckFidelityValid:
    def test_invalid_fidelity(self):
        g = Graph(
            name="T",
            nodes={"a": Node(id="a", fidelity="garbage")},
        )
        diags = check_fidelity_valid(g)
        assert len(diags) == 1
        assert diags[0].severity is Severity.WARNING

    def test_valid_fidelity(self):
        g = Graph(
            name="T",
            nodes={"a": Node(id="a", fidelity="summary:high")},
        )
        assert check_fidelity_valid(g) == []

    def test_empty_fidelity_ok(self):
        g = Graph(name="T", nodes={"a": Node(id="a")})
        assert check_fidelity_valid(g) == []


# ---------------------------------------------------------------------------
# check_retry_target_exists
# ---------------------------------------------------------------------------


class TestCheckRetryTargetExists:
    def test_retry_target_missing(self):
        g = Graph(
            name="T",
            nodes={"a": Node(id="a", retry_target="nonexistent")},
        )
        diags = check_retry_target_exists(g)
        assert len(diags) == 1
        assert diags[0].severity is Severity.WARNING

    def test_fallback_retry_target_missing(self):
        g = Graph(
            name="T",
            nodes={"a": Node(id="a", fallback_retry_target="gone")},
        )
        diags = check_retry_target_exists(g)
        assert len(diags) == 1
        assert diags[0].severity is Severity.WARNING

    def test_retry_target_exists(self):
        g = Graph(
            name="T",
            nodes={
                "a": Node(id="a", retry_target="b"),
                "b": Node(id="b"),
            },
        )
        assert check_retry_target_exists(g) == []


# ---------------------------------------------------------------------------
# check_goal_gate_has_retry
# ---------------------------------------------------------------------------


class TestCheckGoalGateHasRetry:
    def test_goal_gate_without_retry(self):
        g = Graph(
            name="T",
            nodes={"a": Node(id="a", goal_gate=True)},
        )
        diags = check_goal_gate_has_retry(g)
        assert len(diags) == 1
        assert diags[0].severity is Severity.WARNING

    def test_goal_gate_with_node_retry(self):
        g = Graph(
            name="T",
            nodes={
                "a": Node(id="a", goal_gate=True, retry_target="b"),
                "b": Node(id="b"),
            },
        )
        assert check_goal_gate_has_retry(g) == []

    def test_goal_gate_with_graph_retry(self):
        g = Graph(
            name="T",
            nodes={"a": Node(id="a", goal_gate=True)},
            attributes={"retry_target": "somewhere"},
        )
        assert check_goal_gate_has_retry(g) == []


# ---------------------------------------------------------------------------
# check_prompt_on_llm_nodes
# ---------------------------------------------------------------------------


class TestCheckPromptOnLLMNodes:
    def test_llm_node_without_prompt(self):
        g = Graph(
            name="T",
            nodes={"a": Node(id="a", shape="box", type="codergen")},
        )
        diags = check_prompt_on_llm_nodes(g)
        assert len(diags) == 1
        assert diags[0].severity is Severity.WARNING

    def test_llm_node_with_prompt(self):
        g = Graph(
            name="T",
            nodes={"a": Node(id="a", shape="box", type="codergen", prompt="Do work")},
        )
        assert check_prompt_on_llm_nodes(g) == []

    def test_llm_node_with_label(self):
        g = Graph(
            name="T",
            nodes={"a": Node(id="a", shape="box", type="codergen", label="Write code")},
        )
        assert check_prompt_on_llm_nodes(g) == []

    def test_non_llm_node_ok(self):
        g = Graph(
            name="T",
            nodes={"a": Node(id="a", shape="diamond", type="conditional")},
        )
        assert check_prompt_on_llm_nodes(g) == []

    def test_default_shape_box_no_type_is_llm(self):
        """A node with shape=box and no explicit type defaults to LLM."""
        g = Graph(
            name="T",
            nodes={"a": Node(id="a", shape="box")},
        )
        diags = check_prompt_on_llm_nodes(g)
        assert len(diags) == 1
        assert diags[0].severity is Severity.WARNING


# ---------------------------------------------------------------------------
# Validator integration
# ---------------------------------------------------------------------------


class TestValidate:
    def test_valid_graph_no_diagnostics(self):
        g = _minimal_graph()
        diags = validate(g)
        assert all(not d.is_error for d in diags)

    def test_extra_rules(self):
        def custom_rule(graph: Graph) -> list[Diagnostic]:
            return [
                Diagnostic(
                    rule="custom",
                    severity=Severity.INFO,
                    message="custom info",
                )
            ]

        g = _minimal_graph()
        diags = validate(g, extra_rules=[custom_rule])
        assert any(d.rule == "custom" for d in diags)


class TestValidateOrRaise:
    def test_raises_on_errors(self):
        g = Graph(
            name="T",
            nodes={"a": Node(id="a")},
            edges=[],
        )
        with pytest.raises(ValidationError) as exc_info:
            validate_or_raise(g)
        assert len(exc_info.value.diagnostics) > 0

    def test_returns_warnings_without_raising(self):
        g = Graph(
            name="T",
            nodes={
                "start": Node(id="start", shape="Mdiamond"),
                "work": Node(id="work", shape="box", type="codergen"),
                "exit": Node(id="exit", shape="Msquare"),
            },
            edges=[
                Edge(from_node="start", to_node="work"),
                Edge(from_node="work", to_node="exit"),
            ],
        )
        # work is an LLM node with no prompt/label -> WARNING (not error)
        warnings = validate_or_raise(g)
        assert any(d.is_warning for d in warnings)
        # Should NOT raise

    def test_valid_graph_returns_empty(self):
        g = _minimal_graph()
        result = validate_or_raise(g)
        # Minimal graph has no warnings either
        assert all(not d.is_error for d in result)
