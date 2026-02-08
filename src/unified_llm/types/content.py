"""Content data types for multimodal messages."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from unified_llm.types.enums import ContentKind


@dataclass(frozen=True)
class ImageData:
    """Image content payload."""

    url: str | None = None
    data: bytes | None = None
    media_type: str | None = None
    detail: str | None = None


@dataclass(frozen=True)
class AudioData:
    """Audio content payload."""

    url: str | None = None
    data: bytes | None = None
    media_type: str | None = None


@dataclass(frozen=True)
class DocumentData:
    """Document content payload."""

    url: str | None = None
    data: bytes | None = None
    media_type: str | None = None
    file_name: str | None = None


@dataclass(frozen=True)
class ToolCallData:
    """Data for a tool/function call."""

    id: str
    name: str
    arguments: dict[str, Any] | str = field(default_factory=dict)
    type: str = "function"


@dataclass(frozen=True)
class ToolResultData:
    """Result returned from a tool execution."""

    tool_call_id: str
    content: str | dict[str, Any] = ""
    is_error: bool = False
    image_data: bytes | None = None
    image_media_type: str | None = None


@dataclass(frozen=True)
class ThinkingData:
    """Model reasoning/thinking content."""

    text: str
    signature: str | None = None
    redacted: bool = False


@dataclass(frozen=True)
class CacheControl:
    """Cache control directive for content parts."""

    type: str = "ephemeral"


@dataclass(frozen=True)
class ContentPart:
    """A single piece of content within a message (tagged union)."""

    kind: ContentKind | str
    text: str | None = None
    image: ImageData | None = None
    audio: AudioData | None = None
    document: DocumentData | None = None
    tool_call: ToolCallData | None = None
    tool_result: ToolResultData | None = None
    thinking: ThinkingData | None = None
    cache_control: CacheControl | None = None

    # --- Factory classmethods ---

    @classmethod
    def of_text(cls, text: str, cache_control: CacheControl | None = None) -> ContentPart:
        """Create a text content part."""
        return cls(kind=ContentKind.TEXT, text=text, cache_control=cache_control)

    @classmethod
    def image_url(
        cls, url: str, detail: str | None = None
    ) -> ContentPart:
        """Create an image content part from a URL."""
        return cls(
            kind=ContentKind.IMAGE,
            image=ImageData(url=url, detail=detail),
        )

    @classmethod
    def image_base64(
        cls, data: bytes, media_type: str, detail: str | None = None
    ) -> ContentPart:
        """Create an image content part from base64-encoded bytes."""
        return cls(
            kind=ContentKind.IMAGE,
            image=ImageData(data=data, media_type=media_type, detail=detail),
        )

    @classmethod
    def audio_url(cls, url: str, media_type: str | None = None) -> ContentPart:
        """Create an audio content part from a URL."""
        return cls(
            kind=ContentKind.AUDIO,
            audio=AudioData(url=url, media_type=media_type),
        )

    @classmethod
    def document_url(
        cls,
        url: str,
        media_type: str | None = None,
        file_name: str | None = None,
    ) -> ContentPart:
        """Create a document content part from a URL."""
        return cls(
            kind=ContentKind.DOCUMENT,
            document=DocumentData(url=url, media_type=media_type, file_name=file_name),
        )

    @classmethod
    def of_tool_call(
        cls, id: str, name: str, arguments: dict[str, Any] | str = ""
    ) -> ContentPart:
        """Create a tool call content part."""
        return cls(
            kind=ContentKind.TOOL_CALL,
            tool_call=ToolCallData(id=id, name=name, arguments=arguments),
        )

    @classmethod
    def of_tool_result(
        cls,
        tool_call_id: str,
        content: str | dict[str, Any] = "",
        is_error: bool = False,
    ) -> ContentPart:
        """Create a tool result content part."""
        return cls(
            kind=ContentKind.TOOL_RESULT,
            tool_result=ToolResultData(
                tool_call_id=tool_call_id, content=content, is_error=is_error
            ),
        )

    @classmethod
    def of_thinking(cls, text: str, signature: str | None = None) -> ContentPart:
        """Create a thinking content part."""
        return cls(
            kind=ContentKind.THINKING,
            thinking=ThinkingData(text=text, signature=signature),
        )

    @classmethod
    def redacted_thinking(cls, data: str, signature: str | None = None) -> ContentPart:
        """Create a redacted thinking content part."""
        return cls(
            kind=ContentKind.REDACTED_THINKING,
            thinking=ThinkingData(text=data, signature=signature, redacted=True),
        )
