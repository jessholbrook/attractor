"""Variable expansion transform: replaces $goal placeholders in node prompts."""

from __future__ import annotations

from dataclasses import replace

from attractor.model.graph import Graph


class VariableExpansionTransform:
    """Replace ``$goal`` in every node's prompt with the graph-level goal string."""

    def apply(self, graph: Graph) -> Graph:
        if not graph.goal:
            return graph
        new_nodes = {}
        for nid, node in graph.nodes.items():
            if "$goal" in node.prompt:
                new_nodes[nid] = replace(node, prompt=node.prompt.replace("$goal", graph.goal))
            else:
                new_nodes[nid] = node
        return Graph(
            name=graph.name,
            nodes=new_nodes,
            edges=graph.edges,
            attributes=graph.attributes,
        )
