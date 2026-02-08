"""Condition expression evaluator for edge traversal decisions.

Grammar:
    ConditionExpr = Clause ( '&&' Clause )*
    Clause        = Key Operator Literal
    Key           = 'outcome' | 'preferred_label' | 'context.' Path | bare_key
    Operator      = '=' | '!='
"""

from __future__ import annotations

from attractor.model.context import Context
from attractor.model.outcome import Outcome

__all__ = ["evaluate_condition", "resolve_key"]


def resolve_key(key: str, outcome: Outcome, context: Context) -> str:
    """Resolve a condition key to its string value.

    - 'outcome'          -> outcome.status.value (e.g., 'success', 'fail')
    - 'preferred_label'  -> outcome.preferred_label
    - 'context.X'        -> context.get('context.X') or context.get('X'), empty string if missing
    - bare key           -> context.get(key), empty string if missing
    """
    if key == "outcome":
        return outcome.status.value

    if key == "preferred_label":
        return outcome.preferred_label

    if key.startswith("context."):
        # Try the full key first, then the suffix after 'context.'
        value = context.get(key)
        if value is not None:
            return str(value)
        suffix = key[len("context."):]
        value = context.get(suffix)
        if value is not None:
            return str(value)
        return ""

    # Bare key -- resolve from context
    value = context.get(key)
    if value is not None:
        return str(value)
    return ""


def _parse_clause(clause: str) -> tuple[str, str, str]:
    """Parse a single clause like 'outcome=success' or 'outcome!=fail'.

    Returns (key, operator, literal).
    """
    clause = clause.strip()

    # Try '!=' first (longer operator) to avoid partial match on '='
    if "!=" in clause:
        idx = clause.index("!=")
        key = clause[:idx].strip()
        literal = clause[idx + 2:].strip()
        return key, "!=", literal

    if "=" in clause:
        idx = clause.index("=")
        key = clause[:idx].strip()
        literal = clause[idx + 1:].strip()
        return key, "=", literal

    raise ValueError(f"Invalid clause (no operator found): {clause!r}")


def evaluate_condition(expr: str, outcome: Outcome, context: Context) -> bool:
    """Evaluate a condition expression against the current outcome and context.

    Empty/whitespace-only expressions return True (unconditional).
    Clauses joined by '&&' are AND-combined.
    """
    if not expr or not expr.strip():
        return True

    clauses = expr.split("&&")

    for clause_str in clauses:
        key, operator, literal = _parse_clause(clause_str)
        resolved = resolve_key(key, outcome, context)

        if operator == "=":
            if resolved != literal:
                return False
        elif operator == "!=":
            if resolved == literal:
                return False
        else:
            raise ValueError(f"Unknown operator: {operator!r}")

    return True
