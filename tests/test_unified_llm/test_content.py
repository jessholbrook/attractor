"""Tests for unified LLM content data types."""
from __future__ import annotations

import dataclasses

import pytest

from unified_llm.types.enums import ContentKind
from unified_llm.types.content import (
    AudioData,
    CacheControl,
    ContentPart,
    DocumentData,
    ImageData,
    ThinkingData,
    ToolCallData,
    ToolResultData,
)


# --- ImageData ---


class TestImageData:
    def test_all_defaults_none(self) -> None:
        img = ImageData()
        assert img.url is None
        assert img.data is None
        assert img.media_type is None
        assert img.detail is None

    def test_with_url(self) -> None:
        img = ImageData(url="https://example.com/img.png", detail="high")
        assert img.url == "https://example.com/img.png"
        assert img.detail == "high"

    def test_frozen(self) -> None:
        img = ImageData(url="https://example.com/img.png")
        with pytest.raises(dataclasses.FrozenInstanceError):
            img.url = "changed"  # type: ignore[misc]


# --- AudioData ---


class TestAudioData:
    def test_all_defaults_none(self) -> None:
        audio = AudioData()
        assert audio.url is None
        assert audio.data is None
        assert audio.media_type is None

    def test_with_url_and_media_type(self) -> None:
        audio = AudioData(url="https://example.com/audio.wav", media_type="audio/wav")
        assert audio.url == "https://example.com/audio.wav"
        assert audio.media_type == "audio/wav"

    def test_frozen(self) -> None:
        audio = AudioData()
        with pytest.raises(dataclasses.FrozenInstanceError):
            audio.url = "changed"  # type: ignore[misc]


# --- DocumentData ---


class TestDocumentData:
    def test_all_defaults_none(self) -> None:
        doc = DocumentData()
        assert doc.url is None
        assert doc.data is None
        assert doc.media_type is None
        assert doc.file_name is None

    def test_with_all_fields(self) -> None:
        doc = DocumentData(
            url="https://example.com/doc.pdf",
            media_type="application/pdf",
            file_name="report.pdf",
        )
        assert doc.url == "https://example.com/doc.pdf"
        assert doc.media_type == "application/pdf"
        assert doc.file_name == "report.pdf"

    def test_frozen(self) -> None:
        doc = DocumentData()
        with pytest.raises(dataclasses.FrozenInstanceError):
            doc.file_name = "changed"  # type: ignore[misc]


# --- ToolCallData ---


class TestToolCallData:
    def test_required_fields(self) -> None:
        tc = ToolCallData(id="call-1", name="bash")
        assert tc.id == "call-1"
        assert tc.name == "bash"
        assert tc.arguments == {}
        assert tc.type == "function"

    def test_with_dict_arguments(self) -> None:
        args = {"command": "ls -la"}
        tc = ToolCallData(id="call-2", name="bash", arguments=args)
        assert tc.arguments == args

    def test_with_string_arguments(self) -> None:
        tc = ToolCallData(id="call-3", name="bash", arguments='{"command": "ls"}')
        assert tc.arguments == '{"command": "ls"}'

    def test_frozen(self) -> None:
        tc = ToolCallData(id="call-1", name="bash")
        with pytest.raises(dataclasses.FrozenInstanceError):
            tc.name = "changed"  # type: ignore[misc]


# --- ToolResultData ---


class TestToolResultData:
    def test_required_fields_and_defaults(self) -> None:
        tr = ToolResultData(tool_call_id="call-1")
        assert tr.tool_call_id == "call-1"
        assert tr.content == ""
        assert tr.is_error is False
        assert tr.image_data is None
        assert tr.image_media_type is None

    def test_with_string_content(self) -> None:
        tr = ToolResultData(tool_call_id="call-1", content="success output")
        assert tr.content == "success output"

    def test_with_dict_content(self) -> None:
        tr = ToolResultData(tool_call_id="call-1", content={"key": "value"})
        assert tr.content == {"key": "value"}

    def test_error_result(self) -> None:
        tr = ToolResultData(tool_call_id="call-1", content="error msg", is_error=True)
        assert tr.is_error is True

    def test_frozen(self) -> None:
        tr = ToolResultData(tool_call_id="call-1")
        with pytest.raises(dataclasses.FrozenInstanceError):
            tr.content = "changed"  # type: ignore[misc]


# --- ThinkingData ---


class TestThinkingData:
    def test_required_text_field(self) -> None:
        t = ThinkingData(text="reasoning here")
        assert t.text == "reasoning here"
        assert t.signature is None
        assert t.redacted is False

    def test_with_signature(self) -> None:
        t = ThinkingData(text="reasoning", signature="sig-abc")
        assert t.signature == "sig-abc"

    def test_redacted_flag(self) -> None:
        t = ThinkingData(text="", redacted=True)
        assert t.redacted is True

    def test_frozen(self) -> None:
        t = ThinkingData(text="reasoning")
        with pytest.raises(dataclasses.FrozenInstanceError):
            t.text = "changed"  # type: ignore[misc]


# --- CacheControl ---


class TestCacheControl:
    def test_default_type_is_ephemeral(self) -> None:
        cc = CacheControl()
        assert cc.type == "ephemeral"

    def test_custom_type(self) -> None:
        cc = CacheControl(type="persistent")
        assert cc.type == "persistent"

    def test_frozen(self) -> None:
        cc = CacheControl()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cc.type = "changed"  # type: ignore[misc]


# --- ContentPart ---


class TestContentPart:
    def test_minimal_construction(self) -> None:
        part = ContentPart(kind=ContentKind.TEXT, text="hello")
        assert part.kind == ContentKind.TEXT
        assert part.text == "hello"
        assert part.image is None
        assert part.audio is None
        assert part.document is None
        assert part.tool_call is None
        assert part.tool_result is None
        assert part.thinking is None
        assert part.cache_control is None

    def test_frozen(self) -> None:
        part = ContentPart(kind=ContentKind.TEXT, text="hello")
        with pytest.raises(dataclasses.FrozenInstanceError):
            part.text = "changed"  # type: ignore[misc]

    def test_kind_accepts_raw_string(self) -> None:
        part = ContentPart(kind="custom_type", text="data")
        assert part.kind == "custom_type"


class TestContentPartFactoryText:
    def test_of_text_basic(self) -> None:
        part = ContentPart.of_text("hello world")
        assert part.kind == ContentKind.TEXT
        assert part.text == "hello world"
        assert part.cache_control is None

    def test_of_text_with_cache_control(self) -> None:
        cc = CacheControl()
        part = ContentPart.of_text("hello", cache_control=cc)
        assert part.cache_control is cc
        assert part.cache_control.type == "ephemeral"


class TestContentPartFactoryImage:
    def test_image_url(self) -> None:
        part = ContentPart.image_url("https://example.com/img.png", detail="high")
        assert part.kind == ContentKind.IMAGE
        assert part.image is not None
        assert part.image.url == "https://example.com/img.png"
        assert part.image.detail == "high"
        assert part.image.data is None

    def test_image_base64(self) -> None:
        raw = b"\x89PNG\r\n"
        part = ContentPart.image_base64(raw, "image/png", detail="low")
        assert part.kind == ContentKind.IMAGE
        assert part.image is not None
        assert part.image.data == raw
        assert part.image.media_type == "image/png"
        assert part.image.detail == "low"
        assert part.image.url is None


class TestContentPartFactoryAudio:
    def test_audio_url(self) -> None:
        part = ContentPart.audio_url("https://example.com/clip.mp3", media_type="audio/mpeg")
        assert part.kind == ContentKind.AUDIO
        assert part.audio is not None
        assert part.audio.url == "https://example.com/clip.mp3"
        assert part.audio.media_type == "audio/mpeg"

    def test_audio_url_no_media_type(self) -> None:
        part = ContentPart.audio_url("https://example.com/clip.wav")
        assert part.audio is not None
        assert part.audio.media_type is None


class TestContentPartFactoryDocument:
    def test_document_url(self) -> None:
        part = ContentPart.document_url(
            "https://example.com/report.pdf",
            media_type="application/pdf",
            file_name="report.pdf",
        )
        assert part.kind == ContentKind.DOCUMENT
        assert part.document is not None
        assert part.document.url == "https://example.com/report.pdf"
        assert part.document.media_type == "application/pdf"
        assert part.document.file_name == "report.pdf"

    def test_document_url_minimal(self) -> None:
        part = ContentPart.document_url("https://example.com/file.txt")
        assert part.document is not None
        assert part.document.media_type is None
        assert part.document.file_name is None


class TestContentPartFactoryToolCall:
    def test_tool_call_with_dict_args(self) -> None:
        args = {"command": "ls"}
        part = ContentPart.of_tool_call("tc-1", "bash", arguments=args)
        assert part.kind == ContentKind.TOOL_CALL
        assert part.tool_call is not None
        assert part.tool_call.id == "tc-1"
        assert part.tool_call.name == "bash"
        assert part.tool_call.arguments == args

    def test_tool_call_default_arguments(self) -> None:
        part = ContentPart.of_tool_call("tc-2", "read_file")
        assert part.tool_call is not None
        assert part.tool_call.arguments == ""

    def test_tool_call_type_defaults_to_function(self) -> None:
        part = ContentPart.of_tool_call("tc-3", "bash")
        assert part.tool_call is not None
        assert part.tool_call.type == "function"


class TestContentPartFactoryToolResult:
    def test_tool_result_basic(self) -> None:
        part = ContentPart.of_tool_result("tc-1", content="output text")
        assert part.kind == ContentKind.TOOL_RESULT
        assert part.tool_result is not None
        assert part.tool_result.tool_call_id == "tc-1"
        assert part.tool_result.content == "output text"
        assert part.tool_result.is_error is False

    def test_tool_result_error(self) -> None:
        part = ContentPart.of_tool_result("tc-1", content="failed", is_error=True)
        assert part.tool_result is not None
        assert part.tool_result.is_error is True

    def test_tool_result_default_content(self) -> None:
        part = ContentPart.of_tool_result("tc-1")
        assert part.tool_result is not None
        assert part.tool_result.content == ""


class TestContentPartFactoryThinking:
    def test_thinking_basic(self) -> None:
        part = ContentPart.of_thinking("Let me reason about this")
        assert part.kind == ContentKind.THINKING
        assert part.thinking is not None
        assert part.thinking.text == "Let me reason about this"
        assert part.thinking.signature is None
        assert part.thinking.redacted is False

    def test_thinking_with_signature(self) -> None:
        part = ContentPart.of_thinking("reasoning", signature="sig-123")
        assert part.thinking is not None
        assert part.thinking.signature == "sig-123"


class TestContentPartFactoryRedactedThinking:
    def test_redacted_thinking(self) -> None:
        part = ContentPart.redacted_thinking("redacted data", signature="sig-456")
        assert part.kind == ContentKind.REDACTED_THINKING
        assert part.thinking is not None
        assert part.thinking.text == "redacted data"
        assert part.thinking.signature == "sig-456"
        assert part.thinking.redacted is True

    def test_redacted_thinking_no_signature(self) -> None:
        part = ContentPart.redacted_thinking("redacted data")
        assert part.thinking is not None
        assert part.thinking.signature is None
        assert part.thinking.redacted is True
