"""Stylesheet model: Selector, StyleRule, and Stylesheet dataclasses."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Selector:
    """A CSS-like selector targeting nodes by universal, class, or id.

    Specificity values:
        0 = universal (*)
        1 = class (.classname)
        2 = id (#nodeid)
    """

    kind: str  # "universal", "class", "id"
    value: str  # "*", "classname", "nodeid"
    specificity: int  # 0, 1, 2


@dataclass(frozen=True)
class StyleRule:
    """A single rule pairing a selector with property declarations."""

    selector: Selector
    properties: dict[str, str]  # llm_model, llm_provider, reasoning_effort


@dataclass(frozen=True)
class Stylesheet:
    """A collection of style rules parsed from a model stylesheet."""

    rules: list[StyleRule]
