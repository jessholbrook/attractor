"""Anthropic Messages API adapter."""
from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx

from unified_llm._base64 import encode_to_base64
from unified_llm._sse import SSEEvent, parse_sse_lines
from unified_llm.errors import (
    NetworkError,
    RequestTimeoutError,
    StreamError,
    error_from_status_code,
)
from unified_llm.types.content import (
    CacheControl,
    ContentPart,
    ThinkingData,
    ToolCallData,
    ToolResultData,
)
from unified_llm.types.enums import (
    ContentKind,
    FinishReason,
    Role,
    StreamEventType,
    ToolChoiceMode,
)
from unified_llm.types.messages import Message
from unified_llm.types.request import Request
from unified_llm.types.response import FinishReasonInfo, Response, Usage
from unified_llm.types.streaming import StreamEvent
from unified_llm.types.tools import ToolCall


class AnthropicAdapter:
    """Adapter for the Anthropic Messages API."""

    ANTHROPIC_VERSION = "2023-06-01"
    DEFAULT_MAX_TOKENS = 4096

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        timeout: float = 120.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = httpx.Client(
            base_url=self._base_url,
            headers={
                "x-api-key": api_key,
                "anthropic-version": self.ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            timeout=httpx.Timeout(timeout),
        )

    @property
    def name(self) -> str:
        return "anthropic"

    # ------------------------------------------------------------------
    # Request translation
    # ------------------------------------------------------------------

    def _build_request_body(self, request: Request) -> dict[str, Any]:
        """Translate a unified Request into an Anthropic Messages API body."""
        system_blocks: list[dict[str, Any]] = []
        api_messages: list[dict[str, Any]] = []

        for msg in request.messages:
            if msg.role in (Role.SYSTEM, Role.DEVELOPER):
                for part in msg.content:
                    if part.kind == ContentKind.TEXT and part.text:
                        block: dict[str, Any] = {"type": "text", "text": part.text}
                        system_blocks.append(block)
            elif msg.role == Role.TOOL:
                # Tool results become tool_result blocks inside a user message
                content_blocks = self._translate_tool_result_parts(msg)
                api_messages.append({"role": "user", "content": content_blocks})
            else:
                role = "user" if msg.role == Role.USER else "assistant"
                content_blocks = self._translate_content_parts(msg.content)
                api_messages.append({"role": role, "content": content_blocks})

        # Enforce strict alternation: merge consecutive same-role messages
        api_messages = self._enforce_alternation(api_messages)

        body: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens or self.DEFAULT_MAX_TOKENS,
            "messages": api_messages,
        }

        if system_blocks:
            # Inject cache_control on the last system block
            system_blocks[-1]["cache_control"] = {"type": "ephemeral"}
            body["system"] = system_blocks

        # Inject cache_control on the last content block of the last user message
        self._inject_user_cache_control(api_messages)

        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.top_p is not None:
            body["top_p"] = request.top_p
        if request.stop_sequences:
            body["stop_sequences"] = list(request.stop_sequences)

        # Tools
        if request.tools and (
            not request.tool_choice
            or request.tool_choice.mode != ToolChoiceMode.NONE
        ):
            body["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in request.tools
            ]

        # Tool choice
        if request.tool_choice:
            tc = request.tool_choice
            if tc.mode == ToolChoiceMode.AUTO:
                body["tool_choice"] = {"type": "auto"}
            elif tc.mode == ToolChoiceMode.REQUIRED:
                body["tool_choice"] = {"type": "any"}
            elif tc.mode == ToolChoiceMode.NAMED and tc.tool_name:
                body["tool_choice"] = {"type": "tool", "name": tc.tool_name}
            elif tc.mode == ToolChoiceMode.NONE:
                # Remove tools entirely when none is selected
                body.pop("tools", None)
                body.pop("tool_choice", None)

        # Provider options
        provider_opts = (request.provider_options or {}).get("anthropic", {})
        if provider_opts:
            body.update(provider_opts)

        return body

    def _translate_content_parts(
        self, parts: tuple[ContentPart, ...]
    ) -> list[dict[str, Any]]:
        """Convert unified content parts to Anthropic content blocks."""
        blocks: list[dict[str, Any]] = []
        for part in parts:
            block = self._translate_one_part(part)
            if block is not None:
                blocks.append(block)
        return blocks or [{"type": "text", "text": ""}]

    def _translate_one_part(self, part: ContentPart) -> dict[str, Any] | None:
        """Translate a single ContentPart to an Anthropic block."""
        if part.kind == ContentKind.TEXT:
            return {"type": "text", "text": part.text or ""}

        if part.kind == ContentKind.IMAGE and part.image:
            img = part.image
            if img.data is not None and img.media_type:
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img.media_type,
                        "data": (
                            encode_to_base64(img.data)
                            if isinstance(img.data, bytes)
                            else img.data
                        ),
                    },
                }
            if img.url:
                return {
                    "type": "image",
                    "source": {"type": "url", "url": img.url},
                }

        if part.kind == ContentKind.TOOL_CALL and part.tool_call:
            tc = part.tool_call
            arguments = tc.arguments
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except (json.JSONDecodeError, ValueError):
                    arguments = {}
            return {
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": arguments,
            }

        if part.kind == ContentKind.TOOL_RESULT and part.tool_result:
            tr = part.tool_result
            content = tr.content if isinstance(tr.content, str) else json.dumps(tr.content)
            return {
                "type": "tool_result",
                "tool_use_id": tr.tool_call_id,
                "content": content,
                "is_error": tr.is_error,
            }

        if part.kind == ContentKind.THINKING and part.thinking:
            if part.thinking.redacted:
                return {
                    "type": "redacted_thinking",
                    "data": part.thinking.text,
                }
            return {
                "type": "thinking",
                "thinking": part.thinking.text,
                "signature": part.thinking.signature or "",
            }

        if part.kind == ContentKind.REDACTED_THINKING and part.thinking:
            return {
                "type": "redacted_thinking",
                "data": part.thinking.text,
            }

        return None

    def _translate_tool_result_parts(
        self, msg: Message
    ) -> list[dict[str, Any]]:
        """Translate a TOOL message into Anthropic tool_result blocks."""
        blocks: list[dict[str, Any]] = []
        for part in msg.content:
            if part.kind == ContentKind.TOOL_RESULT and part.tool_result:
                tr = part.tool_result
                content = (
                    tr.content
                    if isinstance(tr.content, str)
                    else json.dumps(tr.content)
                )
                blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tr.tool_call_id,
                        "content": content,
                        "is_error": tr.is_error,
                    }
                )
        return blocks or [{"type": "text", "text": ""}]

    def _enforce_alternation(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Merge consecutive same-role messages to satisfy Anthropic alternation."""
        if not messages:
            return messages
        merged: list[dict[str, Any]] = [messages[0]]
        for msg in messages[1:]:
            if msg["role"] == merged[-1]["role"]:
                # Extend content of the previous message
                merged[-1]["content"].extend(msg["content"])
            else:
                merged.append(msg)
        return merged

    def _inject_user_cache_control(
        self, messages: list[dict[str, Any]]
    ) -> None:
        """Add cache_control to the last content block of the last user message."""
        for msg in reversed(messages):
            if msg["role"] == "user" and msg["content"]:
                msg["content"][-1]["cache_control"] = {"type": "ephemeral"}
                break

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, raw: dict[str, Any]) -> Response:
        """Parse an Anthropic API response into a unified Response."""
        content_blocks = raw.get("content", [])
        parts: list[ContentPart] = []

        for block in content_blocks:
            btype = block.get("type")
            if btype == "text":
                parts.append(ContentPart.of_text(block.get("text", "")))
            elif btype == "tool_use":
                parts.append(
                    ContentPart.of_tool_call(
                        id=block.get("id", ""),
                        name=block.get("name", ""),
                        arguments=block.get("input", {}),
                    )
                )
            elif btype == "thinking":
                parts.append(
                    ContentPart.of_thinking(
                        text=block.get("thinking", ""),
                        signature=block.get("signature"),
                    )
                )
            elif btype == "redacted_thinking":
                parts.append(
                    ContentPart.redacted_thinking(
                        data=block.get("data", ""),
                    )
                )

        message = Message(role=Role.ASSISTANT, content=tuple(parts))

        # Stop reason mapping
        stop_reason_raw = raw.get("stop_reason", "")
        finish_reason = self._map_finish_reason(stop_reason_raw)

        # Usage
        usage_raw = raw.get("usage", {})
        usage = Usage(
            input_tokens=usage_raw.get("input_tokens", 0),
            output_tokens=usage_raw.get("output_tokens", 0),
            total_tokens=(
                usage_raw.get("input_tokens", 0)
                + usage_raw.get("output_tokens", 0)
            ),
            cache_read_tokens=usage_raw.get("cache_read_input_tokens"),
            cache_write_tokens=usage_raw.get("cache_creation_input_tokens"),
            raw=usage_raw,
        )

        return Response(
            id=raw.get("id", ""),
            model=raw.get("model", ""),
            provider="anthropic",
            message=message,
            finish_reason=FinishReasonInfo(reason=finish_reason, raw=stop_reason_raw),
            usage=usage,
            raw=raw,
        )

    def _map_finish_reason(self, raw: str) -> FinishReason:
        """Map Anthropic stop_reason to FinishReason."""
        mapping = {
            "end_turn": FinishReason.STOP,
            "stop_sequence": FinishReason.STOP,
            "max_tokens": FinishReason.LENGTH,
            "tool_use": FinishReason.TOOL_CALLS,
        }
        return mapping.get(raw, FinishReason.OTHER)

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def _translate_error(
        self, response: httpx.Response
    ) -> Exception:
        """Translate an HTTP error response to a unified error."""
        try:
            body = response.json()
            error_info = body.get("error", {})
            message = error_info.get("message", response.text)
            error_code = error_info.get("type")
        except Exception:
            message = response.text
            error_code = None
            body = None

        retry_after = None
        if "retry-after" in response.headers:
            try:
                retry_after = float(response.headers["retry-after"])
            except (ValueError, TypeError):
                pass

        return error_from_status_code(
            status_code=response.status_code,
            message=message,
            provider="anthropic",
            error_code=error_code,
            raw=body,
            retry_after=retry_after,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def complete(self, request: Request) -> Response:
        """Send a request and return the full response."""
        body = self._build_request_body(request)
        headers = self._extra_headers(request)

        try:
            http_response = self._client.post(
                "/v1/messages",
                json=body,
                headers=headers,
            )
        except httpx.TimeoutException as exc:
            raise RequestTimeoutError(
                f"Request timed out: {exc}", cause=exc
            ) from exc
        except httpx.HTTPError as exc:
            raise NetworkError(
                f"Network error: {exc}", cause=exc
            ) from exc

        if http_response.status_code != 200:
            raise self._translate_error(http_response)

        return self._parse_response(http_response.json())

    def stream(self, request: Request) -> Iterator[StreamEvent]:
        """Send a request and yield streaming events."""
        body = self._build_request_body(request)
        body["stream"] = True
        headers = self._extra_headers(request)

        try:
            with self._client.stream(
                "POST",
                "/v1/messages",
                json=body,
                headers=headers,
            ) as http_response:
                if http_response.status_code != 200:
                    http_response.read()
                    raise self._translate_error(http_response)

                yield from self._translate_stream(http_response)
        except httpx.TimeoutException as exc:
            raise RequestTimeoutError(
                f"Stream timed out: {exc}", cause=exc
            ) from exc
        except httpx.HTTPError as exc:
            raise NetworkError(
                f"Network error during stream: {exc}", cause=exc
            ) from exc

    def _translate_stream(
        self, http_response: httpx.Response
    ) -> Iterator[StreamEvent]:
        """Parse SSE events from an Anthropic streaming response."""
        # Track state for building complete tool calls and response
        current_block_type: str | None = None
        current_block_index: int = 0
        tool_call_id: str = ""
        tool_call_name: str = ""
        tool_call_args_parts: list[str] = []
        accumulated_usage: dict[str, Any] = {}
        stop_reason: str = ""
        response_id: str = ""
        response_model: str = ""

        for sse_event in parse_sse_lines(http_response.iter_lines()):
            event_type = sse_event.event
            if not sse_event.data or event_type == "ping":
                continue

            try:
                data = json.loads(sse_event.data)
            except json.JSONDecodeError:
                continue

            if event_type == "message_start":
                msg = data.get("message", {})
                response_id = msg.get("id", "")
                response_model = msg.get("model", "")
                usage_raw = msg.get("usage", {})
                accumulated_usage.update(usage_raw)
                yield StreamEvent(type=StreamEventType.STREAM_START, raw=data)

            elif event_type == "content_block_start":
                current_block_index = data.get("index", 0)
                cb = data.get("content_block", {})
                cb_type = cb.get("type", "")
                current_block_type = cb_type

                if cb_type == "text":
                    yield StreamEvent(
                        type=StreamEventType.TEXT_START,
                        text_id=str(current_block_index),
                        raw=data,
                    )
                elif cb_type == "tool_use":
                    tool_call_id = cb.get("id", "")
                    tool_call_name = cb.get("name", "")
                    tool_call_args_parts = []
                    yield StreamEvent(
                        type=StreamEventType.TOOL_CALL_START,
                        tool_call=ToolCall(
                            id=tool_call_id,
                            name=tool_call_name,
                        ),
                        raw=data,
                    )
                elif cb_type == "thinking":
                    yield StreamEvent(
                        type=StreamEventType.REASONING_START,
                        raw=data,
                    )

            elif event_type == "content_block_delta":
                delta = data.get("delta", {})
                delta_type = delta.get("type", "")

                if delta_type == "text_delta":
                    yield StreamEvent(
                        type=StreamEventType.TEXT_DELTA,
                        delta=delta.get("text", ""),
                        text_id=str(current_block_index),
                        raw=data,
                    )
                elif delta_type == "input_json_delta":
                    partial = delta.get("partial_json", "")
                    tool_call_args_parts.append(partial)
                    yield StreamEvent(
                        type=StreamEventType.TOOL_CALL_DELTA,
                        delta=partial,
                        tool_call=ToolCall(
                            id=tool_call_id,
                            name=tool_call_name,
                            raw_arguments=partial,
                        ),
                        raw=data,
                    )
                elif delta_type == "thinking_delta":
                    yield StreamEvent(
                        type=StreamEventType.REASONING_DELTA,
                        reasoning_delta=delta.get("thinking", ""),
                        raw=data,
                    )

            elif event_type == "content_block_stop":
                if current_block_type == "text":
                    yield StreamEvent(
                        type=StreamEventType.TEXT_END,
                        text_id=str(current_block_index),
                        raw=data,
                    )
                elif current_block_type == "tool_use":
                    # Assemble the complete tool call
                    args_str = "".join(tool_call_args_parts)
                    try:
                        args = json.loads(args_str) if args_str else {}
                    except json.JSONDecodeError:
                        args = {}
                    yield StreamEvent(
                        type=StreamEventType.TOOL_CALL_END,
                        tool_call=ToolCall(
                            id=tool_call_id,
                            name=tool_call_name,
                            arguments=args,
                            raw_arguments=args_str,
                        ),
                        raw=data,
                    )
                    tool_call_args_parts = []
                elif current_block_type == "thinking":
                    yield StreamEvent(
                        type=StreamEventType.REASONING_END,
                        raw=data,
                    )
                current_block_type = None

            elif event_type == "message_delta":
                delta = data.get("delta", {})
                stop_reason = delta.get("stop_reason", stop_reason)
                usage_delta = data.get("usage", {})
                accumulated_usage.update(usage_delta)

            elif event_type == "message_stop":
                finish_reason = self._map_finish_reason(stop_reason)
                usage = Usage(
                    input_tokens=accumulated_usage.get("input_tokens", 0),
                    output_tokens=accumulated_usage.get("output_tokens", 0),
                    total_tokens=(
                        accumulated_usage.get("input_tokens", 0)
                        + accumulated_usage.get("output_tokens", 0)
                    ),
                    cache_read_tokens=accumulated_usage.get(
                        "cache_read_input_tokens"
                    ),
                    cache_write_tokens=accumulated_usage.get(
                        "cache_creation_input_tokens"
                    ),
                    raw=accumulated_usage,
                )
                yield StreamEvent(
                    type=StreamEventType.FINISH,
                    finish_reason=FinishReasonInfo(
                        reason=finish_reason, raw=stop_reason
                    ),
                    usage=usage,
                    response=Response(
                        id=response_id,
                        model=response_model,
                        provider="anthropic",
                        finish_reason=FinishReasonInfo(
                            reason=finish_reason, raw=stop_reason
                        ),
                        usage=usage,
                    ),
                    raw=data,
                )

            elif event_type == "error":
                error_data = data.get("error", data)
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    error=StreamError(
                        error_data.get("message", str(error_data))
                    ),
                    raw=data,
                )

    def _extra_headers(self, request: Request) -> dict[str, str]:
        """Build any extra headers needed for the request."""
        headers: dict[str, str] = {}
        provider_opts = (request.provider_options or {}).get("anthropic", {})
        beta_headers = provider_opts.pop("beta_headers", None)
        if beta_headers:
            if isinstance(beta_headers, list):
                headers["anthropic-beta"] = ",".join(beta_headers)
            else:
                headers["anthropic-beta"] = str(beta_headers)
        return headers

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()
