"""Lark Transformer that converts a DOT parse tree into a Graph model."""

from __future__ import annotations

import copy
from pathlib import Path

from lark import Lark, Token, Transformer, Tree

from attractor.model.graph import Edge, Graph, Node
from attractor.parser.errors import ParseError

GRAMMAR_PATH = Path(__file__).parent / "grammar.lark"

# Fields that live directly on the Node dataclass (not in attrs).
_NODE_FIELDS: dict[str, type] = {
    "label": str,
    "shape": str,
    "type": str,
    "prompt": str,
    "max_retries": int,
    "goal_gate": bool,
    "retry_target": str,
    "fallback_retry_target": str,
    "fidelity": str,
    "thread_id": str,
    "node_class": str,
    "timeout": float,
    "llm_model": str,
    "llm_provider": str,
    "reasoning_effort": str,
    "auto_status": bool,
    "allow_partial": bool,
}

# Fields that live directly on the Edge dataclass.
_EDGE_FIELDS: dict[str, type] = {
    "label": str,
    "condition": str,
    "weight": int,
    "fidelity": str,
    "thread_id": str,
    "loop_restart": bool,
}


def _coerce(value: object, target: type) -> object:
    """Coerce a parsed value to the target type expected by the dataclass."""
    if target is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes")
        return bool(value)
    if target is int:
        return int(value)  # type: ignore[arg-type]
    if target is float:
        if isinstance(value, (int, float)):
            return float(value)
        return float(value)  # type: ignore[arg-type]
    return str(value)


def _build_node(node_id: str, attrs: dict[str, object]) -> Node:
    """Build a Node from an id and merged attribute dict."""
    kwargs: dict[str, object] = {"id": node_id}
    extra: dict[str, str] = {}
    for k, v in attrs.items():
        if k in _NODE_FIELDS:
            kwargs[k] = _coerce(v, _NODE_FIELDS[k])
        else:
            extra[k] = str(v)
    if extra:
        kwargs["attrs"] = extra
    return Node(**kwargs)  # type: ignore[arg-type]


def _build_edge(from_node: str, to_node: str, attrs: dict[str, object]) -> Edge:
    """Build an Edge from endpoints and merged attribute dict."""
    kwargs: dict[str, object] = {"from_node": from_node, "to_node": to_node}
    for k, v in attrs.items():
        if k in _EDGE_FIELDS:
            kwargs[k] = _coerce(v, _EDGE_FIELDS[k])
    return Edge(**kwargs)  # type: ignore[arg-type]


class _Sentinel:
    """Marker objects returned by transformer rules that represent side-effects."""


class _NodeDefault(_Sentinel):
    def __init__(self, attrs: dict[str, object]):
        self.attrs = attrs


class _EdgeDefault(_Sentinel):
    def __init__(self, attrs: dict[str, object]):
        self.attrs = attrs


class _GraphAttr(_Sentinel):
    def __init__(self, attrs: dict[str, object]):
        self.attrs = attrs


class _NodeDecl(_Sentinel):
    def __init__(self, node_id: str, attrs: dict[str, object]):
        self.node_id = node_id
        self.attrs = attrs


class _EdgeDecl(_Sentinel):
    def __init__(self, node_ids: list[str], attrs: dict[str, object]):
        self.node_ids = node_ids
        self.attrs = attrs


class _Subgraph(_Sentinel):
    def __init__(self, statements: list[_Sentinel]):
        self.statements = statements


class DotTransformer(Transformer):  # type: ignore[type-arg]
    """Transform a Lark parse tree into intermediate sentinel objects."""

    # ---- value coercion ----

    def string_value(self, items: list[Token]) -> str:
        raw = str(items[0])
        # Strip surrounding quotes and process escapes.
        return raw[1:-1].replace('\\"', '"').replace("\\n", "\n").replace("\\\\", "\\")

    def duration_value(self, items: list[Token]) -> float:
        raw = str(items[0])
        multipliers = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
        for suffix, mult in multipliers.items():
            if raw.endswith(suffix):
                return float(raw[: -len(suffix)]) * mult
        return float(raw)  # pragma: no cover

    def boolean_value(self, items: list[Token]) -> bool:
        return str(items[0]) == "true"

    def float_value(self, items: list[Token]) -> float:
        return float(items[0])

    def int_value(self, items: list[Token]) -> int:
        return int(items[0])

    def bare_id_value(self, items: list[Token]) -> str:
        return str(items[0])

    # ---- structural ----

    def key(self, items: list[Token]) -> str:
        return ".".join(str(t) for t in items)

    def attr(self, items: list[object]) -> tuple[str, object]:
        return (str(items[0]), items[1])

    def attr_list(self, items: list[tuple[str, object]]) -> dict[str, object]:
        return dict(items)

    def node_id(self, items: list[Token]) -> str:
        return str(items[0])

    def node_stmt(self, items: list[object]) -> _NodeDecl:
        node_id = str(items[0])
        attrs = items[1] if len(items) > 1 and isinstance(items[1], dict) else {}
        return _NodeDecl(node_id, attrs)

    def edge_stmt(self, items: list[object]) -> _EdgeDecl:
        # Items are: node_id, node_id, ..., optional attr_list
        node_ids: list[str] = []
        attrs: dict[str, object] = {}
        for item in items:
            if isinstance(item, dict):
                attrs = item
            else:
                node_ids.append(str(item))
        return _EdgeDecl(node_ids, attrs)

    def graph_attr_stmt(self, items: list[object]) -> _GraphAttr:
        attrs = items[0] if items and isinstance(items[0], dict) else {}
        return _GraphAttr(attrs)

    def graph_attr_assign(self, items: list[object]) -> _GraphAttr:
        key = str(items[0])
        value = items[1]
        return _GraphAttr({key: value})

    def node_defaults(self, items: list[object]) -> _NodeDefault:
        attrs = items[0] if items and isinstance(items[0], dict) else {}
        return _NodeDefault(attrs)

    def edge_defaults(self, items: list[object]) -> _EdgeDefault:
        attrs = items[0] if items and isinstance(items[0], dict) else {}
        return _EdgeDefault(attrs)

    def subgraph(self, items: list[object]) -> _Subgraph:
        statements: list[_Sentinel] = []
        for item in items:
            if isinstance(item, _Sentinel):
                statements.append(item)
        return _Subgraph(statements)

    def digraph(self, items: list[object]) -> Graph:
        # First item is the graph name (ID token).
        graph_name = str(items[0])

        # Collect all statements.
        statements: list[_Sentinel] = []
        for item in items[1:]:
            if isinstance(item, _Sentinel):
                statements.append(item)

        return _assemble_graph(graph_name, statements)

    def start(self, items: list[object]) -> Graph:
        return items[0]  # type: ignore[return-value]


def _assemble_graph(name: str, statements: list[_Sentinel]) -> Graph:
    """Walk the flat list of sentinel objects and build a Graph."""
    graph = Graph(name=name)
    _process_statements(statements, graph, {}, {}, node_class="")
    return graph


def _process_statements(
    statements: list[_Sentinel],
    graph: Graph,
    node_defaults: dict[str, object],
    edge_defaults: dict[str, object],
    node_class: str,
) -> None:
    """Process a list of statements, mutating graph in place.

    node_defaults and edge_defaults are copied so that subgraph scoping
    works correctly -- the caller's dicts are not modified.
    """
    current_node_defaults = dict(node_defaults)
    current_edge_defaults = dict(edge_defaults)

    for stmt in statements:
        if isinstance(stmt, _GraphAttr):
            graph.attributes.update({k: str(v) for k, v in stmt.attrs.items()})

        elif isinstance(stmt, _NodeDefault):
            current_node_defaults.update(stmt.attrs)

        elif isinstance(stmt, _EdgeDefault):
            current_edge_defaults.update(stmt.attrs)

        elif isinstance(stmt, _NodeDecl):
            merged = dict(current_node_defaults)
            merged.update(stmt.attrs)
            if node_class and "node_class" not in merged:
                merged["node_class"] = node_class
            node = _build_node(stmt.node_id, merged)
            graph.nodes[stmt.node_id] = node

        elif isinstance(stmt, _EdgeDecl):
            merged = dict(current_edge_defaults)
            merged.update(stmt.attrs)
            # Expand chained edges: A -> B -> C produces A->B and B->C.
            for i in range(len(stmt.node_ids) - 1):
                edge = _build_edge(stmt.node_ids[i], stmt.node_ids[i + 1], merged)
                graph.edges.append(edge)
            # Ensure implicitly-referenced nodes exist.
            for nid in stmt.node_ids:
                if nid not in graph.nodes:
                    node_merged = dict(current_node_defaults)
                    if node_class:
                        node_merged["node_class"] = node_class
                    graph.nodes[nid] = _build_node(nid, node_merged)

        elif isinstance(stmt, _Subgraph):
            # Derive a CSS-like class from subgraph label if present.
            sub_class = node_class
            for sub_stmt in stmt.statements:
                if isinstance(sub_stmt, _GraphAttr) and "label" in sub_stmt.attrs:
                    sub_class = str(sub_stmt.attrs["label"]).strip().lower().replace(" ", "_")
                    break
            _process_statements(
                stmt.statements,
                graph,
                current_node_defaults,
                current_edge_defaults,
                node_class=sub_class,
            )


def parse_dot(source: str) -> Graph:
    """Parse a DOT source string into a Graph model."""
    parser = Lark(
        GRAMMAR_PATH.read_text(),
        parser="lalr",
        start="start",
    )
    try:
        tree = parser.parse(source)
    except Exception as e:
        # Try to extract line/column from Lark exceptions.
        line = getattr(e, "line", None)
        column = getattr(e, "column", None)
        raise ParseError(str(e), line=line, column=column) from e
    transformer = DotTransformer()
    return transformer.transform(tree)
