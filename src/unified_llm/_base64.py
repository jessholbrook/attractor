"""Base64 encoding utilities for multimodal content."""
from __future__ import annotations

import base64
import mimetypes


def encode_to_base64(data: bytes) -> str:
    """Base64-encode raw bytes and return as a string."""
    return base64.b64encode(data).decode("ascii")


def make_data_uri(data: bytes, media_type: str) -> str:
    """Build a ``data:`` URI from raw bytes and a media type."""
    encoded = encode_to_base64(data)
    return f"data:{media_type};base64,{encoded}"


def decode_data_uri(uri: str) -> tuple[bytes, str]:
    """Parse a ``data:`` URI and return ``(bytes, media_type)``.

    Raises :class:`ValueError` if *uri* is not a valid data URI.
    """
    if not uri.startswith("data:"):
        raise ValueError(f"Not a valid data URI: {uri!r}")

    # data:<media_type>;base64,<data>
    rest = uri[len("data:"):]
    if ";base64," not in rest:
        raise ValueError(f"Not a valid data URI (missing ;base64,): {uri!r}")

    media_type, _, encoded = rest.partition(";base64,")
    return base64.b64decode(encoded), media_type


def infer_media_type(file_path: str) -> str:
    """Guess the MIME type for a file path, defaulting to ``application/octet-stream``."""
    mime, _ = mimetypes.guess_type(file_path)
    return mime or "application/octet-stream"


def is_file_path(value: str) -> bool:
    """Return ``True`` if *value* looks like a file-system path."""
    return value.startswith(("/", "./", "~/"))
