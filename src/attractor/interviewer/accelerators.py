"""Accelerator key parsing for menu/option labels."""

from __future__ import annotations

import re


def parse_accelerator(label: str) -> tuple[str, str]:
    """Extract an accelerator key from a label.

    Recognizes patterns:
    - "[K] label" -> ("K", "label")
    - "K) label"  -> ("K", "label")
    - "K - label" -> ("K", "label")

    Returns (key, clean_label). If no accelerator is found,
    returns ("", original_label).
    """
    label = label.strip()

    # Pattern: [K] label
    m = re.match(r"^\[([a-zA-Z0-9])\]\s*(.*)", label)
    if m:
        return m.group(1), m.group(2).strip()

    # Pattern: K) label
    m = re.match(r"^([a-zA-Z0-9])\)\s*(.*)", label)
    if m:
        return m.group(1), m.group(2).strip()

    # Pattern: K - label
    m = re.match(r"^([a-zA-Z0-9])\s*-\s+(.*)", label)
    if m:
        return m.group(1), m.group(2).strip()

    return "", label
