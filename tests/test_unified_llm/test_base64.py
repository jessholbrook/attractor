"""Tests for base64 encoding utilities."""
from __future__ import annotations

import pytest

from unified_llm._base64 import (
    decode_data_uri,
    encode_to_base64,
    infer_media_type,
    is_file_path,
    make_data_uri,
)


# ---------------------------------------------------------------------------
# encode / decode round-trip
# ---------------------------------------------------------------------------


def test_encode_to_base64() -> None:
    result = encode_to_base64(b"hello world")
    assert result == "aGVsbG8gd29ybGQ="


def test_encode_empty_bytes() -> None:
    result = encode_to_base64(b"")
    assert result == ""


def test_encode_decode_roundtrip() -> None:
    original = b"The quick brown fox jumps over the lazy dog"
    encoded = encode_to_base64(original)
    uri = make_data_uri(original, "text/plain")
    decoded_bytes, media_type = decode_data_uri(uri)
    assert decoded_bytes == original
    assert media_type == "text/plain"


# ---------------------------------------------------------------------------
# make_data_uri
# ---------------------------------------------------------------------------


def test_make_data_uri() -> None:
    data = b"PNG_DATA"
    uri = make_data_uri(data, "image/png")
    assert uri.startswith("data:image/png;base64,")
    assert "UE5HX0RBVEE=" in uri  # base64 of PNG_DATA


# ---------------------------------------------------------------------------
# decode_data_uri
# ---------------------------------------------------------------------------


def test_decode_data_uri_valid() -> None:
    original = b"test bytes"
    uri = make_data_uri(original, "application/octet-stream")
    decoded, mt = decode_data_uri(uri)
    assert decoded == original
    assert mt == "application/octet-stream"


def test_decode_data_uri_not_data_scheme() -> None:
    with pytest.raises(ValueError, match="Not a valid data URI"):
        decode_data_uri("https://example.com/image.png")


def test_decode_data_uri_missing_base64() -> None:
    with pytest.raises(ValueError, match="missing ;base64,"):
        decode_data_uri("data:text/plain,hello")


# ---------------------------------------------------------------------------
# infer_media_type
# ---------------------------------------------------------------------------


def test_infer_media_type_png() -> None:
    assert infer_media_type("photo.png") == "image/png"


def test_infer_media_type_jpeg() -> None:
    mt = infer_media_type("photo.jpg")
    assert mt in ("image/jpeg", "image/jpg")


def test_infer_media_type_unknown() -> None:
    assert infer_media_type("data.xyz123") == "application/octet-stream"


def test_infer_media_type_pdf() -> None:
    assert infer_media_type("doc.pdf") == "application/pdf"


# ---------------------------------------------------------------------------
# is_file_path
# ---------------------------------------------------------------------------


def test_is_file_path_absolute() -> None:
    assert is_file_path("/usr/local/file.txt") is True


def test_is_file_path_relative_dot() -> None:
    assert is_file_path("./image.png") is True


def test_is_file_path_home_tilde() -> None:
    assert is_file_path("~/docs/file.pdf") is True


def test_is_file_path_url() -> None:
    assert is_file_path("https://example.com/image.png") is False


def test_is_file_path_base64() -> None:
    assert is_file_path("data:image/png;base64,abc") is False


def test_is_file_path_plain_name() -> None:
    assert is_file_path("image.png") is False
