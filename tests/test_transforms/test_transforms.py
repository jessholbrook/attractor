"""Tests for graph transforms."""

import pytest

from attractor.model.graph import Edge, Graph, Node
from attractor.transforms import apply_transforms
from attractor.transforms.variable_expansion import VariableExpansionTransform
from attractor.transforms.stylesheet import StylesheetApplicationTransform


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _graph_with_prompt(prompt: str, goal: str = "") -> Graph:
    attrs = {}
    if goal:
        attrs["goal"] = goal
    return Graph(
        name="T",
        nodes={
            "start": Node(id="start", shape="Mdiamond"),
            "work": Node(id="work", prompt=prompt),
            "exit": Node(id="exit", shape="Msquare"),
        },
        edges=[
            Edge(from_node="start", to_node="work"),
            Edge(from_node="work", to_node="exit"),
        ],
        attributes=attrs,
    )


def _graph_with_stylesheet(stylesheet: str, nodes: dict[str, Node] | None = None) -> Graph:
    if nodes is None:
        nodes = {
            "start": Node(id="start", shape="Mdiamond"),
            "work": Node(id="work"),
            "exit": Node(id="exit", shape="Msquare"),
        }
    return Graph(
        name="T",
        nodes=nodes,
        edges=[],
        attributes={"model_stylesheet": stylesheet},
    )


# ---------------------------------------------------------------------------
# VariableExpansionTransform
# ---------------------------------------------------------------------------


class TestVariableExpansion:
    def test_replaces_goal_in_prompt(self):
        g = _graph_with_prompt("Achieve $goal now", goal="world peace")
        result = VariableExpansionTransform().apply(g)
        assert result.nodes["work"].prompt == "Achieve world peace now"

    def test_noop_when_no_goal(self):
        g = _graph_with_prompt("Achieve $goal now", goal="")
        result = VariableExpansionTransform().apply(g)
        # No goal set, so $goal stays as-is.
        assert result.nodes["work"].prompt == "Achieve $goal now"

    def test_noop_when_no_dollar_goal(self):
        g = _graph_with_prompt("Just do the work", goal="something")
        result = VariableExpansionTransform().apply(g)
        assert result.nodes["work"].prompt == "Just do the work"

    def test_multiple_occurrences(self):
        g = _graph_with_prompt("$goal is $goal", goal="fun")
        result = VariableExpansionTransform().apply(g)
        assert result.nodes["work"].prompt == "fun is fun"


# ---------------------------------------------------------------------------
# StylesheetApplicationTransform
# ---------------------------------------------------------------------------


class TestStylesheetApplication:
    def test_universal_rule(self):
        css = "* { llm_model: claude-sonnet-4-5; }"
        g = _graph_with_stylesheet(css)
        result = StylesheetApplicationTransform().apply(g)
        assert result.nodes["work"].llm_model == "claude-sonnet-4-5"

    def test_class_rule(self):
        css = ".code { llm_model: claude-opus-4-6; }"
        nodes = {
            "a": Node(id="a", node_class="code"),
            "b": Node(id="b", node_class="review"),
        }
        g = _graph_with_stylesheet(css, nodes=nodes)
        result = StylesheetApplicationTransform().apply(g)
        assert result.nodes["a"].llm_model == "claude-opus-4-6"
        assert result.nodes["b"].llm_model == ""  # unchanged

    def test_id_rule(self):
        css = "#special { llm_provider: anthropic; }"
        nodes = {
            "special": Node(id="special"),
            "other": Node(id="other"),
        }
        g = _graph_with_stylesheet(css, nodes=nodes)
        result = StylesheetApplicationTransform().apply(g)
        assert result.nodes["special"].llm_provider == "anthropic"
        assert result.nodes["other"].llm_provider == ""

    def test_specificity_id_overrides_class(self):
        css = ".code { llm_model: sonnet; } #special { llm_model: opus; }"
        nodes = {
            "special": Node(id="special", node_class="code"),
        }
        g = _graph_with_stylesheet(css, nodes=nodes)
        result = StylesheetApplicationTransform().apply(g)
        assert result.nodes["special"].llm_model == "opus"

    def test_specificity_class_overrides_universal(self):
        css = "* { llm_model: haiku; } .code { llm_model: sonnet; }"
        nodes = {
            "a": Node(id="a", node_class="code"),
            "b": Node(id="b"),
        }
        g = _graph_with_stylesheet(css, nodes=nodes)
        result = StylesheetApplicationTransform().apply(g)
        assert result.nodes["a"].llm_model == "sonnet"
        assert result.nodes["b"].llm_model == "haiku"

    def test_specificity_id_overrides_universal(self):
        css = "* { llm_model: haiku; } #a { llm_model: opus; }"
        nodes = {"a": Node(id="a")}
        g = _graph_with_stylesheet(css, nodes=nodes)
        result = StylesheetApplicationTransform().apply(g)
        assert result.nodes["a"].llm_model == "opus"

    def test_explicit_attribute_overrides_stylesheet(self):
        css = "* { llm_model: haiku; }"
        nodes = {
            "a": Node(id="a", llm_model="explicit-model"),
        }
        g = _graph_with_stylesheet(css, nodes=nodes)
        result = StylesheetApplicationTransform().apply(g)
        assert result.nodes["a"].llm_model == "explicit-model"

    def test_no_stylesheet_noop(self):
        g = Graph(
            name="T",
            nodes={"a": Node(id="a")},
            edges=[],
            attributes={},
        )
        result = StylesheetApplicationTransform().apply(g)
        assert result.nodes["a"].llm_model == ""

    def test_empty_stylesheet_noop(self):
        g = _graph_with_stylesheet("")
        result = StylesheetApplicationTransform().apply(g)
        assert result is g  # same object returned


# ---------------------------------------------------------------------------
# apply_transforms pipeline
# ---------------------------------------------------------------------------


class TestApplyTransforms:
    def test_runs_all_transforms_in_order(self):
        """Variable expansion should happen, then stylesheet application."""
        g = Graph(
            name="T",
            nodes={
                "start": Node(id="start", shape="Mdiamond"),
                "work": Node(id="work", prompt="$goal"),
                "exit": Node(id="exit", shape="Msquare"),
            },
            edges=[
                Edge(from_node="start", to_node="work"),
                Edge(from_node="work", to_node="exit"),
            ],
            attributes={
                "goal": "build stuff",
                "model_stylesheet": "* { llm_model: sonnet; }",
            },
        )
        result = apply_transforms(g)
        assert result.nodes["work"].prompt == "build stuff"
        assert result.nodes["work"].llm_model == "sonnet"

    def test_custom_transforms_appended(self):
        class AddSuffix:
            def apply(self, graph: Graph) -> Graph:
                from dataclasses import replace

                new_nodes = {}
                for nid, node in graph.nodes.items():
                    new_nodes[nid] = replace(node, label=node.label + "_custom")
                return Graph(
                    name=graph.name,
                    nodes=new_nodes,
                    edges=graph.edges,
                    attributes=graph.attributes,
                )

        g = Graph(
            name="T",
            nodes={"a": Node(id="a", label="hello")},
            edges=[],
            attributes={},
        )
        result = apply_transforms(g, custom_transforms=[AddSuffix()])
        assert result.nodes["a"].label == "hello_custom"
