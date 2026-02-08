"""Server-Sent Events parser."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field


@dataclass
class SSEEvent:
    """Accumulated SSE event data."""

    event: str = "message"
    data: str = ""
    id: str = ""
    retry: int | None = None


def parse_sse_lines(lines: Iterator[str]) -> Iterator[SSEEvent]:
    """Parse raw SSE text lines into structured events.

    Follows the SSE specification:
    - Lines beginning with ``:`` are comments (ignored).
    - Blank lines dispatch the current event.
    - Field names: ``event``, ``data``, ``id``, ``retry``.
    - A leading space after the colon is stripped from the field value.
    """
    current = SSEEvent()
    data_parts: list[str] = []
    has_data = False

    for raw_line in lines:
        line = raw_line.rstrip("\n").rstrip("\r")

        # Comment
        if line.startswith(":"):
            continue

        # Blank line -> dispatch
        if line == "":
            if has_data:
                current.data = "\n".join(data_parts)
                yield current
            current = SSEEvent()
            data_parts = []
            has_data = False
            continue

        # Parse field
        if ":" in line:
            field_name, _, value = line.partition(":")
            # Strip at most one leading space per SSE spec
            if value.startswith(" "):
                value = value[1:]
        else:
            field_name = line
            value = ""

        if field_name == "event":
            current.event = value
        elif field_name == "data":
            data_parts.append(value)
            has_data = True
        elif field_name == "id":
            current.id = value
        elif field_name == "retry":
            try:
                current.retry = int(value)
            except ValueError:
                pass

    # Handle remaining data if the stream ends without a trailing blank line
    if has_data:
        current.data = "\n".join(data_parts)
        yield current
