"""Hand-written parser for CSS-like model stylesheets.

Syntax example:
    * { llm_model: claude-sonnet-4-5; llm_provider: anthropic; }
    .code { llm_model: claude-opus-4-6; }
    #critical_review { reasoning_effort: high; }
"""

from __future__ import annotations

import re

from attractor.stylesheet.model import Selector, StyleRule, Stylesheet

__all__ = ["parse_stylesheet"]

# Matches a complete rule: selector { properties }
_RULE_RE = re.compile(
    r"""
    (?P<selector>[^{]+)     # everything before the opening brace
    \{                       # opening brace
    (?P<body>[^}]*)          # property declarations
    \}                       # closing brace
    """,
    re.VERBOSE,
)

# Matches a single property declaration: key: value;
_PROP_RE = re.compile(
    r"""
    (?P<key>[a-zA-Z_][a-zA-Z0-9_-]*)   # property name
    \s*:\s*                              # colon separator
    (?P<value>[^;]+?)                    # value (non-greedy up to semicolon)
    \s*;                                 # terminating semicolon
    """,
    re.VERBOSE,
)


def _parse_selector(raw: str) -> Selector:
    """Parse a raw selector string into a Selector object."""
    raw = raw.strip()
    if raw == "*":
        return Selector(kind="universal", value="*", specificity=0)
    if raw.startswith("."):
        class_name = raw[1:]
        return Selector(kind="class", value=class_name, specificity=1)
    if raw.startswith("#"):
        node_id = raw[1:]
        return Selector(kind="id", value=node_id, specificity=2)
    raise ValueError(f"Invalid selector: {raw!r}")


def _parse_properties(body: str) -> dict[str, str]:
    """Parse the body of a rule block into a property dictionary."""
    props: dict[str, str] = {}
    for match in _PROP_RE.finditer(body):
        key = match.group("key").strip()
        value = match.group("value").strip()
        props[key] = value
    return props


def parse_stylesheet(source: str) -> Stylesheet:
    """Parse a CSS-like model stylesheet string into a Stylesheet object.

    Returns a Stylesheet containing all parsed rules in source order.
    """
    rules: list[StyleRule] = []
    for match in _RULE_RE.finditer(source):
        selector = _parse_selector(match.group("selector"))
        properties = _parse_properties(match.group("body"))
        if properties:  # skip rules with no valid properties
            rules.append(StyleRule(selector=selector, properties=properties))
    return Stylesheet(rules=rules)
