"""Graph validator: runs all validation rules and reports diagnostics."""

from __future__ import annotations

from typing import Callable

from attractor.model.diagnostic import Diagnostic, Severity
from attractor.model.graph import Graph
from attractor.validation.rules import ALL_RULES


class ValidationError(Exception):
    """Raised when validation produces ERROR-severity diagnostics."""

    def __init__(self, diagnostics: list[Diagnostic]) -> None:
        self.diagnostics = diagnostics
        messages = [str(d) for d in diagnostics if d.is_error]
        super().__init__(
            f"Validation failed with {len(messages)} error(s): " + "; ".join(messages)
        )


RuleFunc = Callable[[Graph], list[Diagnostic]]


def validate(
    graph: Graph, extra_rules: list[RuleFunc] | None = None
) -> list[Diagnostic]:
    """Run all validation rules against *graph*.

    Returns the full list of diagnostics (errors, warnings, info).
    """
    rules: list[RuleFunc] = list(ALL_RULES)
    if extra_rules:
        rules.extend(extra_rules)
    diagnostics: list[Diagnostic] = []
    for rule in rules:
        diagnostics.extend(rule(graph))
    return diagnostics


def validate_or_raise(
    graph: Graph, extra_rules: list[RuleFunc] | None = None
) -> list[Diagnostic]:
    """Run validation; raises :class:`ValidationError` if any ERROR diagnostics exist.

    Returns the non-error diagnostics (warnings/info) when no errors are found.
    """
    diagnostics = validate(graph, extra_rules=extra_rules)
    errors = [d for d in diagnostics if d.is_error]
    if errors:
        raise ValidationError(errors)
    return diagnostics
