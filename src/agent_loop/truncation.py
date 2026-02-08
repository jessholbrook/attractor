"""Tool output truncation with head/tail and tail-only modes."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class TruncationMode(Enum):
    HEAD_TAIL = "head_tail"
    TAIL = "tail"


@dataclass(frozen=True)
class TruncationConfig:
    """Configuration for truncating tool output."""
    max_chars: int = 30_000
    mode: TruncationMode = TruncationMode.HEAD_TAIL


# Default per-tool truncation limits (from the spec)
DEFAULT_TOOL_LIMITS: dict[str, TruncationConfig] = {
    "read_file": TruncationConfig(max_chars=50_000, mode=TruncationMode.HEAD_TAIL),
    "shell": TruncationConfig(max_chars=30_000, mode=TruncationMode.HEAD_TAIL),
    "grep": TruncationConfig(max_chars=20_000, mode=TruncationMode.TAIL),
    "glob": TruncationConfig(max_chars=20_000, mode=TruncationMode.TAIL),
    "edit_file": TruncationConfig(max_chars=10_000, mode=TruncationMode.TAIL),
    "apply_patch": TruncationConfig(max_chars=10_000, mode=TruncationMode.TAIL),
    "write_file": TruncationConfig(max_chars=1_000, mode=TruncationMode.TAIL),
    "spawn_agent": TruncationConfig(max_chars=20_000, mode=TruncationMode.HEAD_TAIL),
}


def truncate_output(output: str, config: TruncationConfig | None = None) -> str:
    """Truncate tool output according to the config.

    Character-based truncation runs first (handles all cases including single huge lines).

    For HEAD_TAIL mode: keeps first half and last half of chars, inserts marker.
    For TAIL mode: keeps only the last max_chars characters.

    Returns original if within limits.
    """
    if config is None:
        config = TruncationConfig()

    if len(output) <= config.max_chars:
        return output

    removed = len(output) - config.max_chars

    if config.mode == TruncationMode.HEAD_TAIL:
        half = config.max_chars // 2
        marker = (
            f"\n\n[WARNING: Output truncated. {removed} characters removed from the middle. "
            f"Full output available in TOOL_CALL_END event.]\n\n"
        )
        return output[:half] + marker + output[-half:]

    # TAIL mode
    marker = (
        f"[WARNING: Output truncated. First {removed} characters removed.]\n\n"
    )
    return marker + output[-config.max_chars:]


def get_tool_config(tool_name: str, overrides: dict[str, TruncationConfig] | None = None) -> TruncationConfig:
    """Get truncation config for a tool, with optional overrides."""
    if overrides and tool_name in overrides:
        return overrides[tool_name]
    return DEFAULT_TOOL_LIMITS.get(tool_name, TruncationConfig())


# --- Line-based truncation (secondary pass) ---

DEFAULT_LINE_LIMITS: dict[str, int | None] = {
    "shell": 256,
    "grep": 200,
    "glob": 500,
    "read_file": None,
    "edit_file": None,
    "apply_patch": None,
    "write_file": None,
    "spawn_agent": None,
}


def truncate_lines(output: str, max_lines: int) -> str:
    """Line-based truncation using head/tail strategy.

    Keeps first half and last half of lines with an omission marker.
    Returns original if within limits.
    """
    lines = output.split("\n")
    if len(lines) <= max_lines:
        return output

    head_count = max_lines // 2
    tail_count = max_lines - head_count
    omitted = len(lines) - head_count - tail_count

    head = "\n".join(lines[:head_count])
    tail = "\n".join(lines[-tail_count:])
    return head + f"\n[... {omitted} lines omitted ...]\n" + tail


def truncate_tool_output(
    output: str,
    tool_name: str,
    char_overrides: dict[str, TruncationConfig] | None = None,
    line_overrides: dict[str, int] | None = None,
) -> str:
    """Two-stage truncation pipeline: chars first, then lines.

    Step 1: Character truncation via truncate_output() — always runs first.
    Step 2: Line truncation via truncate_lines() — runs where configured.

    Character truncation MUST run first to handle pathological cases
    like single lines with millions of characters.
    """
    # Step 1: Character-based truncation
    char_config = get_tool_config(tool_name, overrides=char_overrides)
    result = truncate_output(output, char_config)

    # Step 2: Line-based truncation
    if line_overrides and tool_name in line_overrides:
        max_lines = line_overrides[tool_name]
    else:
        max_lines = DEFAULT_LINE_LIMITS.get(tool_name)

    if max_lines is not None:
        result = truncate_lines(result, max_lines)

    return result
