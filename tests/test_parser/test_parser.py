"""Comprehensive tests for the DOT parser."""

from pathlib import Path

import pytest

from attractor.parser import ParseError, parse_dot
from attractor.model.graph import Graph, Node, Edge

FIXTURES = Path(__file__).parent.parent / "fixtures"


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _load(name: str) -> Graph:
    return parse_dot((FIXTURES / name).read_text())


# ---------------------------------------------------------------------------
# Fixture file tests
# ---------------------------------------------------------------------------


class TestSimplePipeline:
    @pytest.fixture()
    def graph(self) -> Graph:
        return _load("simple_pipeline.dot")

    def test_graph_name(self, graph: Graph) -> None:
        assert graph.name == "Simple"

    def test_node_count(self, graph: Graph) -> None:
        assert len(graph.nodes) == 4

    def test_edge_count(self, graph: Graph) -> None:
        assert len(graph.edges) == 3

    def test_goal_attribute(self, graph: Graph) -> None:
        assert graph.goal == "Run tests and report"

    def test_graph_rankdir(self, graph: Graph) -> None:
        assert graph.attributes.get("rankdir") == "LR"

    def test_start_node_shape(self, graph: Graph) -> None:
        assert graph.nodes["start"].shape == "Mdiamond"
        assert graph.nodes["start"].label == "Start"

    def test_exit_node_shape(self, graph: Graph) -> None:
        assert graph.nodes["exit"].shape == "Msquare"
        assert graph.nodes["exit"].label == "Exit"

    def test_chained_edges_expand(self, graph: Graph) -> None:
        edges = graph.edges
        assert edges[0].from_node == "start"
        assert edges[0].to_node == "run_tests"
        assert edges[1].from_node == "run_tests"
        assert edges[1].to_node == "report"
        assert edges[2].from_node == "report"
        assert edges[2].to_node == "exit"

    def test_node_prompt(self, graph: Graph) -> None:
        assert graph.nodes["run_tests"].prompt == "Run the test suite"
        assert graph.nodes["report"].prompt == "Summarize results"

    def test_start_exit_helpers(self, graph: Graph) -> None:
        assert graph.start_node() is not None
        assert graph.start_node().id == "start"  # type: ignore[union-attr]
        assert graph.exit_node() is not None
        assert graph.exit_node().id == "exit"  # type: ignore[union-attr]


class TestBranching:
    @pytest.fixture()
    def graph(self) -> Graph:
        return _load("branching.dot")

    def test_goal(self, graph: Graph) -> None:
        assert graph.goal == "Implement and validate"

    def test_node_count(self, graph: Graph) -> None:
        assert len(graph.nodes) == 6

    def test_conditional_edges(self, graph: Graph) -> None:
        gate_edges = [e for e in graph.edges if e.from_node == "gate"]
        assert len(gate_edges) == 2
        labels = {e.label for e in gate_edges}
        assert "Yes" in labels
        assert "No" in labels

    def test_condition_strings(self, graph: Graph) -> None:
        gate_edges = [e for e in graph.edges if e.from_node == "gate"]
        conditions = {e.condition for e in gate_edges}
        assert "outcome=success" in conditions
        assert "outcome!=success" in conditions

    def test_chained_edges(self, graph: Graph) -> None:
        # start -> plan -> implement -> validate -> gate = 4 chain edges
        chain = [e for e in graph.edges if e.from_node != "gate"]
        assert len(chain) == 4

    def test_node_defaults_timeout(self, graph: Graph) -> None:
        # node [timeout=900] should apply to nodes declared after
        assert graph.nodes["plan"].timeout == 900.0
        assert graph.nodes["implement"].timeout == 900.0

    def test_node_defaults_shape(self, graph: Graph) -> None:
        # node [shape=box] default should apply, but explicit overrides
        assert graph.nodes["plan"].shape == "box"
        assert graph.nodes["gate"].shape == "diamond"  # overridden

    def test_start_exit_shapes(self, graph: Graph) -> None:
        assert graph.nodes["start"].shape == "Mdiamond"
        assert graph.nodes["exit"].shape == "Msquare"


class TestGoalGate:
    @pytest.fixture()
    def graph(self) -> Graph:
        return _load("goal_gate.dot")

    def test_goal(self, graph: Graph) -> None:
        assert graph.goal == "Build and verify"

    def test_retry_target_graph_attr(self, graph: Graph) -> None:
        assert graph.attributes.get("retry_target") == "plan"

    def test_goal_gate_flag(self, graph: Graph) -> None:
        assert graph.nodes["implement"].goal_gate is True

    def test_non_goal_gate_nodes(self, graph: Graph) -> None:
        assert graph.nodes["plan"].goal_gate is False
        assert graph.nodes["review"].goal_gate is False

    def test_edge_count(self, graph: Graph) -> None:
        assert len(graph.edges) == 4


class TestHumanGate:
    @pytest.fixture()
    def graph(self) -> Graph:
        return _load("human_gate.dot")

    def test_hexagon_shape(self, graph: Graph) -> None:
        assert graph.nodes["review_gate"].shape == "hexagon"

    def test_review_gate_label(self, graph: Graph) -> None:
        assert graph.nodes["review_gate"].label == "Review Changes"

    def test_edge_labels(self, graph: Graph) -> None:
        rg_edges = [e for e in graph.edges if e.from_node == "review_gate"]
        labels = {e.label for e in rg_edges}
        assert "[A] Approve" in labels
        assert "[F] Fix" in labels

    def test_edge_count(self, graph: Graph) -> None:
        assert len(graph.edges) == 5


# ---------------------------------------------------------------------------
# Comment stripping
# ---------------------------------------------------------------------------


class TestComments:
    def test_line_comments(self) -> None:
        src = """
        digraph C {
            // This is a comment
            start [shape=Mdiamond]  // inline comment
            exit [shape=Msquare]
            start -> exit
        }
        """
        g = parse_dot(src)
        assert len(g.nodes) == 2

    def test_block_comments(self) -> None:
        src = """
        digraph C {
            /* multi
               line
               comment */
            start [shape=Mdiamond]
            exit [shape=Msquare]
            start -> exit
        }
        """
        g = parse_dot(src)
        assert len(g.nodes) == 2

    def test_mixed_comments(self) -> None:
        src = """
        digraph C {
            // line comment
            start [shape=Mdiamond]
            /* block */
            exit [shape=Msquare]
            start -> exit  // trailing
        }
        """
        g = parse_dot(src)
        assert len(g.nodes) == 2
        assert len(g.edges) == 1


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


class TestValueTypes:
    def test_string_value(self) -> None:
        g = parse_dot('digraph V { a [label="Hello World"] }')
        assert g.nodes["a"].label == "Hello World"

    def test_integer_value(self) -> None:
        g = parse_dot("digraph V { a [max_retries=3] }")
        assert g.nodes["a"].max_retries == 3

    def test_float_value(self) -> None:
        g = parse_dot("digraph V { a [timeout=1.5] }")
        assert g.nodes["a"].timeout == 1.5

    def test_boolean_true(self) -> None:
        g = parse_dot("digraph V { a [goal_gate=true] }")
        assert g.nodes["a"].goal_gate is True

    def test_boolean_false(self) -> None:
        g = parse_dot("digraph V { a [goal_gate=false] }")
        assert g.nodes["a"].goal_gate is False

    def test_duration_seconds(self) -> None:
        g = parse_dot("digraph V { a [timeout=900s] }")
        assert g.nodes["a"].timeout == 900.0

    def test_duration_minutes(self) -> None:
        g = parse_dot("digraph V { a [timeout=15m] }")
        assert g.nodes["a"].timeout == 900.0

    def test_duration_hours(self) -> None:
        g = parse_dot("digraph V { a [timeout=2h] }")
        assert g.nodes["a"].timeout == 7200.0

    def test_duration_milliseconds(self) -> None:
        g = parse_dot("digraph V { a [timeout=250ms] }")
        assert g.nodes["a"].timeout == 0.25

    def test_duration_days(self) -> None:
        g = parse_dot("digraph V { a [timeout=1d] }")
        assert g.nodes["a"].timeout == 86400.0

    def test_bare_id_value(self) -> None:
        g = parse_dot("digraph V { a [shape=diamond] }")
        assert g.nodes["a"].shape == "diamond"


# ---------------------------------------------------------------------------
# Node defaults
# ---------------------------------------------------------------------------


class TestNodeDefaults:
    def test_defaults_applied(self) -> None:
        src = """
        digraph D {
            node [shape=box, prompt="default prompt"]
            a [label="A"]
            b [label="B"]
        }
        """
        g = parse_dot(src)
        assert g.nodes["a"].shape == "box"
        assert g.nodes["b"].shape == "box"
        assert g.nodes["a"].prompt == "default prompt"

    def test_explicit_overrides_default(self) -> None:
        src = """
        digraph D {
            node [shape=box]
            a [shape=diamond, label="A"]
        }
        """
        g = parse_dot(src)
        assert g.nodes["a"].shape == "diamond"

    def test_defaults_accumulate(self) -> None:
        src = """
        digraph D {
            node [shape=box]
            a [label="A"]
            node [prompt="later"]
            b [label="B"]
        }
        """
        g = parse_dot(src)
        # a gets shape=box but no prompt default
        assert g.nodes["a"].shape == "box"
        assert g.nodes["a"].prompt == ""
        # b gets shape=box and prompt="later"
        assert g.nodes["b"].shape == "box"
        assert g.nodes["b"].prompt == "later"


# ---------------------------------------------------------------------------
# Edge defaults
# ---------------------------------------------------------------------------


class TestEdgeDefaults:
    def test_edge_defaults_applied(self) -> None:
        src = """
        digraph D {
            edge [weight=5]
            a -> b
            a -> c
        }
        """
        g = parse_dot(src)
        assert all(e.weight == 5 for e in g.edges)

    def test_edge_explicit_overrides(self) -> None:
        src = """
        digraph D {
            edge [weight=5]
            a -> b [weight=10]
        }
        """
        g = parse_dot(src)
        assert g.edges[0].weight == 10


# ---------------------------------------------------------------------------
# Chained edge expansion
# ---------------------------------------------------------------------------


class TestChainedEdges:
    def test_three_node_chain(self) -> None:
        g = parse_dot("digraph C { a -> b -> c }")
        assert len(g.edges) == 2
        assert g.edges[0].from_node == "a"
        assert g.edges[0].to_node == "b"
        assert g.edges[1].from_node == "b"
        assert g.edges[1].to_node == "c"

    def test_four_node_chain(self) -> None:
        g = parse_dot("digraph C { a -> b -> c -> d }")
        assert len(g.edges) == 3

    def test_chain_with_attrs(self) -> None:
        g = parse_dot('digraph C { a -> b -> c [label="chain"] }')
        assert len(g.edges) == 2
        assert g.edges[0].label == "chain"
        assert g.edges[1].label == "chain"

    def test_chain_creates_implicit_nodes(self) -> None:
        g = parse_dot("digraph C { a -> b -> c }")
        assert "a" in g.nodes
        assert "b" in g.nodes
        assert "c" in g.nodes


# ---------------------------------------------------------------------------
# Subgraph scoping
# ---------------------------------------------------------------------------


class TestSubgraph:
    def test_subgraph_nodes_in_graph(self) -> None:
        src = """
        digraph S {
            subgraph cluster_0 {
                graph [label="Testing"]
                a [label="A"]
                b [label="B"]
            }
            c [label="C"]
        }
        """
        g = parse_dot(src)
        assert "a" in g.nodes
        assert "b" in g.nodes
        assert "c" in g.nodes

    def test_subgraph_class_derived(self) -> None:
        src = """
        digraph S {
            subgraph cluster_0 {
                graph [label="My Phase"]
                a [label="A"]
            }
        }
        """
        g = parse_dot(src)
        assert g.nodes["a"].node_class == "my_phase"

    def test_subgraph_node_defaults_scoped(self) -> None:
        src = """
        digraph S {
            node [shape=box]
            subgraph cluster_0 {
                node [shape=ellipse]
                a [label="A"]
            }
            b [label="B"]
        }
        """
        g = parse_dot(src)
        # Inside subgraph, shape=ellipse
        assert g.nodes["a"].shape == "ellipse"
        # Outside subgraph, shape=box still
        assert g.nodes["b"].shape == "box"


# ---------------------------------------------------------------------------
# Parse errors
# ---------------------------------------------------------------------------


class TestParseErrors:
    def test_invalid_syntax(self) -> None:
        with pytest.raises(ParseError):
            parse_dot("not valid dot")

    def test_missing_closing_brace(self) -> None:
        with pytest.raises(ParseError):
            parse_dot("digraph X { a -> b")

    def test_undirected_edge_fails(self) -> None:
        with pytest.raises(ParseError):
            parse_dot("digraph X { a -- b }")

    def test_error_has_message(self) -> None:
        with pytest.raises(ParseError) as exc_info:
            parse_dot("digraph { ??? }")
        assert str(exc_info.value) != ""


# ---------------------------------------------------------------------------
# Multi-line attribute blocks
# ---------------------------------------------------------------------------


class TestMultilineAttrs:
    def test_multiline_attr_list(self) -> None:
        src = """
        digraph M {
            a [
                label="Hello",
                shape=box,
                prompt="Do things"
            ]
        }
        """
        g = parse_dot(src)
        assert g.nodes["a"].label == "Hello"
        assert g.nodes["a"].shape == "box"
        assert g.nodes["a"].prompt == "Do things"


# ---------------------------------------------------------------------------
# Graph-level attributes
# ---------------------------------------------------------------------------


class TestGraphAttributes:
    def test_graph_block_attrs(self) -> None:
        src = 'digraph G { graph [goal="test", label="My Graph"] }'
        g = parse_dot(src)
        assert g.attributes["goal"] == "test"
        assert g.attributes["label"] == "My Graph"

    def test_bare_graph_attr(self) -> None:
        src = "digraph G { rankdir=LR }"
        g = parse_dot(src)
        assert g.attributes["rankdir"] == "LR"

    def test_multiple_graph_attrs(self) -> None:
        src = """
        digraph G {
            graph [goal="build"]
            rankdir=LR
            graph [label="Pipeline"]
        }
        """
        g = parse_dot(src)
        assert g.attributes["goal"] == "build"
        assert g.attributes["rankdir"] == "LR"
        assert g.attributes["label"] == "Pipeline"


# ---------------------------------------------------------------------------
# Dotted keys
# ---------------------------------------------------------------------------


class TestDottedKeys:
    def test_dotted_key_in_node_attrs(self) -> None:
        src = 'digraph D { a [tool_hooks.pre="lint"] }'
        g = parse_dot(src)
        # tool_hooks.pre is not a known Node field, goes to attrs
        assert g.nodes["a"].attrs["tool_hooks.pre"] == "lint"


# ---------------------------------------------------------------------------
# Semicolons
# ---------------------------------------------------------------------------


class TestSemicolons:
    def test_with_semicolons(self) -> None:
        src = """
        digraph S {
            a [label="A"];
            b [label="B"];
            a -> b;
        }
        """
        g = parse_dot(src)
        assert len(g.nodes) == 2
        assert len(g.edges) == 1

    def test_without_semicolons(self) -> None:
        src = """
        digraph S {
            a [label="A"]
            b [label="B"]
            a -> b
        }
        """
        g = parse_dot(src)
        assert len(g.nodes) == 2
        assert len(g.edges) == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_graph(self) -> None:
        g = parse_dot("digraph E { }")
        assert g.name == "E"
        assert len(g.nodes) == 0
        assert len(g.edges) == 0

    def test_empty_attr_list(self) -> None:
        g = parse_dot("digraph E { a [] }")
        assert "a" in g.nodes

    def test_trailing_comma_in_attrs(self) -> None:
        g = parse_dot('digraph E { a [label="x", shape=box,] }')
        assert g.nodes["a"].label == "x"
        assert g.nodes["a"].shape == "box"

    def test_node_referenced_only_in_edge(self) -> None:
        """Nodes referenced in edges but never declared should be created."""
        g = parse_dot("digraph E { a -> b }")
        assert "a" in g.nodes
        assert "b" in g.nodes
