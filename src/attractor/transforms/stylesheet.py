"""Stylesheet application transform: applies model_stylesheet rules to nodes."""

from __future__ import annotations

from dataclasses import fields, replace

from attractor.model.graph import Graph, Node


# Node field names that can be set via stylesheet properties.
_NODE_FIELDS = {f.name for f in fields(Node)} - {"id", "attrs"}

# Fields that have non-empty "set" defaults and therefore can't be reliably
# detected as "explicitly set" just by checking truthiness.  We use a sentinel
# approach instead: if the current value equals the dataclass default we treat
# it as "not explicitly set" and allow the stylesheet to override it.
_NODE_DEFAULTS: dict[str, object] = {}
for _f in fields(Node):
    if _f.default is not _f.default_factory:  # type: ignore[attr-defined]
        if _f.default is not _f.__class__:
            try:
                _NODE_DEFAULTS[_f.name] = _f.default
            except Exception:
                pass


def _is_explicitly_set(node: Node, field_name: str) -> bool:
    """Return True if the node's field looks like it was explicitly set."""
    value = getattr(node, field_name)
    default = _NODE_DEFAULTS.get(field_name)
    if default is not None:
        return value != default
    # For fields with default_factory (like attrs dict), treat non-empty as explicit.
    return bool(value)


def _coerce_value(field_name: str, raw: str) -> object:
    """Coerce a stylesheet string value to the appropriate Python type."""
    # Look up the type annotation from Node fields.
    for f in fields(Node):
        if f.name == field_name:
            annotation = f.type
            if annotation in ("int",):
                return int(raw)
            if annotation in ("float", "float | None"):
                return float(raw)
            if annotation in ("bool",):
                return raw.lower() in ("true", "1", "yes")
            return raw
    return raw


class StylesheetApplicationTransform:
    """Apply the graph's ``model_stylesheet`` attribute to nodes.

    Selector matching:
        - ``*`` (universal) matches all nodes.
        - ``.name`` (class) matches nodes whose ``node_class`` contains *name*.
        - ``#name`` (id) matches nodes whose ``id`` equals *name*.

    Rules are applied in specificity order (universal < class < id).  Properties
    that are already explicitly set on the node are never overridden.
    """

    def apply(self, graph: Graph) -> Graph:
        stylesheet_src = graph.attributes.get("model_stylesheet", "")
        if not stylesheet_src:
            return graph

        from attractor.stylesheet import parse_stylesheet

        stylesheet = parse_stylesheet(stylesheet_src)
        if not stylesheet.rules:
            return graph

        # Sort rules by specificity ascending so lower-specificity rules
        # are applied first and higher-specificity rules can override.
        sorted_rules = sorted(stylesheet.rules, key=lambda r: r.selector.specificity)

        new_nodes: dict[str, Node] = {}
        for nid, node in graph.nodes.items():
            updates: dict[str, object] = {}
            for rule in sorted_rules:
                if not self._matches(rule.selector, node):
                    continue
                for prop, value in rule.properties.items():
                    if prop not in _NODE_FIELDS:
                        continue
                    # Only apply if the node doesn't already have this set explicitly.
                    if _is_explicitly_set(node, prop):
                        continue
                    updates[prop] = _coerce_value(prop, value)

            if updates:
                new_nodes[nid] = replace(node, **updates)
            else:
                new_nodes[nid] = node

        return Graph(
            name=graph.name,
            nodes=new_nodes,
            edges=graph.edges,
            attributes=graph.attributes,
        )

    @staticmethod
    def _matches(selector: "object", node: Node) -> bool:
        """Check whether *selector* matches *node*."""
        kind = getattr(selector, "kind", "")
        value = getattr(selector, "value", "")
        if kind == "universal":
            return True
        if kind == "class":
            # node_class can be a space-separated list of class names.
            return value in node.node_class.split()
        if kind == "id":
            return node.id == value
        return False
