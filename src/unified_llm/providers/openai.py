"""OpenAI Responses API adapter."""
from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx

from unified_llm._base64 import encode_to_base64, make_data_uri
from unified_llm._sse import parse_sse_lines
from unified_llm.errors import (
    NetworkError,
    RequestTimeoutError,
    StreamError,
    error_from_status_code,
)
from unified_llm.types.content import ContentPart, ToolCallData
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


class OpenAIAdapter:
    """Adapter for the OpenAI Responses API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com",
        org_id: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._org_id = org_id
        self._timeout = timeout

        headers: dict[str, str] = {
            "authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        }
        if org_id:
            headers["openai-organization"] = org_id

        self._client = httpx.Client(
            base_url=self._base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout),
        )

    @property
    def name(self) -> str:
        return "openai"

    # ------------------------------------------------------------------
    # Request translation
    # ------------------------------------------------------------------

    def _build_request_body(self, request: Request) -> dict[str, Any]:
        """Translate a unified Request into an OpenAI Responses API body."""
        body: dict[str, Any] = {"model": request.model}

        # System / developer messages -> instructions
        instructions_parts: list[str] = []
        input_items: list[dict[str, Any]] = []

        for msg in request.messages:
            if msg.role in (Role.SYSTEM, Role.DEVELOPER):
                instructions_parts.append(msg.text)
            elif msg.role == Role.USER:
                content_blocks = self._translate_user_content(msg.content)
                input_items.append(
                    {"type": "message", "role": "user", "content": content_blocks}
                )
            elif msg.role == Role.ASSISTANT:
                # Build assistant message and any tool_call items
                self._translate_assistant_message(msg, input_items)
            elif msg.role == Role.TOOL:
                self._translate_tool_result_message(msg, input_items)

        if instructions_parts:
            body["instructions"] = "\n\n".join(instructions_parts)

        body["input"] = input_items

        # Max tokens
        if request.max_tokens is not None:
            body["max_output_tokens"] = request.max_tokens

        # Temperature
        if request.temperature is not None:
            body["temperature"] = request.temperature

        # Top P
        if request.top_p is not None:
            body["top_p"] = request.top_p

        # Reasoning effort
        if request.reasoning_effort:
            body["reasoning"] = {"effort": request.reasoning_effort}

        # Tools
        if request.tools and (
            not request.tool_choice
            or request.tool_choice.mode != ToolChoiceMode.NONE
        ):
            body["tools"] = [
                {
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
                for t in request.tools
            ]

        # Tool choice
        if request.tool_choice:
            tc = request.tool_choice
            if tc.mode == ToolChoiceMode.AUTO:
                body["tool_choice"] = "auto"
            elif tc.mode == ToolChoiceMode.NONE:
                body["tool_choice"] = "none"
                body.pop("tools", None)
            elif tc.mode == ToolChoiceMode.REQUIRED:
                body["tool_choice"] = "required"
            elif tc.mode == ToolChoiceMode.NAMED and tc.tool_name:
                body["tool_choice"] = {
                    "type": "function",
                    "name": tc.tool_name,
                }

        # Provider options
        provider_opts = (request.provider_options or {}).get("openai", {})
        if provider_opts:
            body.update(provider_opts)

        return body

    def _translate_user_content(
        self, parts: tuple[ContentPart, ...]
    ) -> list[dict[str, Any]]:
        """Convert user content parts to OpenAI input content blocks."""
        blocks: list[dict[str, Any]] = []
        for part in parts:
            if part.kind == ContentKind.TEXT:
                blocks.append({"type": "input_text", "text": part.text or ""})
            elif part.kind == ContentKind.IMAGE and part.image:
                img = part.image
                if img.data is not None and img.media_type:
                    data_uri = make_data_uri(img.data, img.media_type) if isinstance(img.data, bytes) else f"data:{img.media_type};base64,{img.data}"
                    blocks.append({"type": "input_image", "image_url": data_uri})
                elif img.url:
                    blocks.append({"type": "input_image", "image_url": img.url})
        return blocks or [{"type": "input_text", "text": ""}]

    def _translate_assistant_message(
        self, msg: Message, input_items: list[dict[str, Any]]
    ) -> None:
        """Translate assistant message: text goes into a message item, tool calls become function_call items."""
        text_parts: list[dict[str, Any]] = []
        for part in msg.content:
            if part.kind == ContentKind.TEXT and part.text:
                text_parts.append({"type": "output_text", "text": part.text})
            elif part.kind == ContentKind.TOOL_CALL and part.tool_call:
                # First, emit any pending text as a message
                if text_parts:
                    input_items.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": text_parts,
                        }
                    )
                    text_parts = []
                tc = part.tool_call
                args = tc.arguments
                if isinstance(args, dict):
                    args = json.dumps(args)
                input_items.append(
                    {
                        "type": "function_call",
                        "id": tc.id,
                        "name": tc.name,
                        "arguments": args,
                    }
                )
        # Emit remaining text
        if text_parts:
            input_items.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": text_parts,
                }
            )

    def _translate_tool_result_message(
        self, msg: Message, input_items: list[dict[str, Any]]
    ) -> None:
        """Translate tool result message into function_call_output items."""
        for part in msg.content:
            if part.kind == ContentKind.TOOL_RESULT and part.tool_result:
                tr = part.tool_result
                output = (
                    tr.content
                    if isinstance(tr.content, str)
                    else json.dumps(tr.content)
                )
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": tr.tool_call_id,
                        "output": output,
                    }
                )

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, raw: dict[str, Any]) -> Response:
        """Parse an OpenAI Responses API response."""
        output_items = raw.get("output", [])
        parts: list[ContentPart] = []
        has_function_calls = False

        for item in output_items:
            item_type = item.get("type", "")
            if item_type == "message":
                for block in item.get("content", []):
                    block_type = block.get("type", "")
                    if block_type == "output_text":
                        parts.append(ContentPart.of_text(block.get("text", "")))
            elif item_type == "function_call":
                has_function_calls = True
                args_str = item.get("arguments", "")
                try:
                    args = json.loads(args_str) if args_str else {}
                except (json.JSONDecodeError, ValueError):
                    args = args_str
                parts.append(
                    ContentPart.of_tool_call(
                        id=item.get("id", ""),
                        name=item.get("name", ""),
                        arguments=args,
                    )
                )

        message = Message(role=Role.ASSISTANT, content=tuple(parts))

        # Status mapping
        status = raw.get("status", "")
        if has_function_calls:
            finish_reason = FinishReason.TOOL_CALLS
        elif status == "completed":
            finish_reason = FinishReason.STOP
        elif status == "incomplete":
            finish_reason = FinishReason.LENGTH
        else:
            finish_reason = FinishReason.OTHER

        # Usage
        usage_raw = raw.get("usage", {})
        reasoning_tokens = None
        cache_read_tokens = None
        output_details = usage_raw.get("output_tokens_details", {})
        if output_details:
            reasoning_tokens = output_details.get("reasoning_tokens")
        input_details = usage_raw.get("input_tokens_details", {})
        if input_details:
            cache_read_tokens = input_details.get("cached_tokens")

        input_tokens = usage_raw.get("input_tokens", 0)
        output_tokens = usage_raw.get("output_tokens", 0)

        usage = Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=usage_raw.get("total_tokens", input_tokens + output_tokens),
            reasoning_tokens=reasoning_tokens,
            cache_read_tokens=cache_read_tokens,
            raw=usage_raw,
        )

        return Response(
            id=raw.get("id", ""),
            model=raw.get("model", ""),
            provider="openai",
            message=message,
            finish_reason=FinishReasonInfo(reason=finish_reason, raw=status),
            usage=usage,
            raw=raw,
        )

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def _translate_error(self, response: httpx.Response) -> Exception:
        """Translate an HTTP error response to a unified error."""
        try:
            body = response.json()
            error_info = body.get("error", {})
            message = error_info.get("message", response.text)
            error_code = error_info.get("type") or error_info.get("code")
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
            provider="openai",
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

        try:
            http_response = self._client.post("/v1/responses", json=body)
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

        try:
            with self._client.stream(
                "POST", "/v1/responses", json=body
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
        """Parse SSE events from an OpenAI Responses streaming response."""
        # Track state for building tool calls
        current_tool_call_id: str = ""
        current_tool_call_name: str = ""
        tool_call_args_parts: list[str] = []

        for sse_event in parse_sse_lines(http_response.iter_lines()):
            event_type = sse_event.event
            if not sse_event.data:
                continue

            try:
                data = json.loads(sse_event.data)
            except json.JSONDecodeError:
                continue

            if event_type == "response.created":
                yield StreamEvent(
                    type=StreamEventType.STREAM_START,
                    raw=data,
                )

            elif event_type == "response.output_item.added":
                item = data.get("item", {})
                item_type = item.get("type", "")
                if item_type == "function_call":
                    current_tool_call_id = item.get("id", "")
                    current_tool_call_name = item.get("name", "")
                    tool_call_args_parts = []
                    yield StreamEvent(
                        type=StreamEventType.TOOL_CALL_START,
                        tool_call=ToolCall(
                            id=current_tool_call_id,
                            name=current_tool_call_name,
                        ),
                        raw=data,
                    )
                elif item_type == "message":
                    pass  # Text content will come via deltas

            elif event_type == "response.output_text.delta":
                yield StreamEvent(
                    type=StreamEventType.TEXT_DELTA,
                    delta=data.get("delta", ""),
                    raw=data,
                )

            elif event_type == "response.function_call_arguments.delta":
                partial = data.get("delta", "")
                tool_call_args_parts.append(partial)
                yield StreamEvent(
                    type=StreamEventType.TOOL_CALL_DELTA,
                    delta=partial,
                    tool_call=ToolCall(
                        id=current_tool_call_id,
                        name=current_tool_call_name,
                        raw_arguments=partial,
                    ),
                    raw=data,
                )

            elif event_type == "response.output_item.done":
                item = data.get("item", {})
                item_type = item.get("type", "")
                if item_type == "message":
                    yield StreamEvent(
                        type=StreamEventType.TEXT_END,
                        raw=data,
                    )
                elif item_type == "function_call":
                    args_str = "".join(tool_call_args_parts)
                    try:
                        args = json.loads(args_str) if args_str else {}
                    except json.JSONDecodeError:
                        args = {}
                    yield StreamEvent(
                        type=StreamEventType.TOOL_CALL_END,
                        tool_call=ToolCall(
                            id=item.get("id", current_tool_call_id),
                            name=item.get("name", current_tool_call_name),
                            arguments=args,
                            raw_arguments=args_str,
                        ),
                        raw=data,
                    )
                    tool_call_args_parts = []

            elif event_type == "response.completed":
                response_data = data.get("response", data)
                parsed = self._parse_response(response_data)
                yield StreamEvent(
                    type=StreamEventType.FINISH,
                    finish_reason=parsed.finish_reason,
                    usage=parsed.usage,
                    response=parsed,
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

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()
