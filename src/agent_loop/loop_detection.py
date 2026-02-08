"""Loop detection: identifies repeating tool call patterns."""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolCallSignature:
    """Hashable signature of a tool call for pattern detection."""
    tool_name: str
    arguments_hash: str

    def __str__(self) -> str:
        return f"{self.tool_name}({self.arguments_hash[:8]})"


def make_signature(tool_name: str, arguments: dict) -> ToolCallSignature:
    """Create a signature from a tool call name and arguments."""
    # Deterministic hash: sort keys, stable JSON serialization
    args_json = json.dumps(arguments, sort_keys=True, default=str)
    args_hash = hashlib.sha256(args_json.encode()).hexdigest()
    return ToolCallSignature(tool_name=tool_name, arguments_hash=args_hash)


def detect_loop(
    signatures: list[ToolCallSignature],
    window: int = 10,
) -> str | None:
    """Check the last `window` signatures for repeating patterns.

    Looks for a pattern of length 1..window//2 that repeats across the window.
    Returns a description string if a loop is detected, None otherwise.
    """
    if len(signatures) < window:
        return None

    recent = signatures[-window:]

    # Check for repeating patterns of length 1 to window//2
    for pattern_len in range(1, window // 2 + 1):
        pattern = recent[:pattern_len]
        repeats = True
        for i in range(pattern_len, window):
            if recent[i] != pattern[i % pattern_len]:
                repeats = False
                break
        if repeats:
            pattern_str = ", ".join(str(s) for s in pattern)
            return (
                f"Loop detected: the last {window} tool calls follow a repeating pattern "
                f"[{pattern_str}]. Try a different approach."
            )

    return None
