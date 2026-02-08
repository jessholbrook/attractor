"""Gemini Native API adapter."""
from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any
from uuid import uuid4

import httpx

from unified_llm._base64 import encode_to_base64
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


class GeminiAdapter:
    """Adapter for the Google Gemini generativeLanguage API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://generativelanguage.googleapis.com",
        timeout: float = 120.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = httpx.Client(
            base_url=self._base_url,
            headers={"content-type": "application/json"},
            timeout=httpx.Timeout(timeout),
        )

    @property
    def name(self) -> str:
        return "gemini"

    # ------------------------------------------------------------------
    # Request translation
    # ------------------------------------------------------------------

    def _build_request_body(self, request: Request) -> dict[str, Any]:
        """Translate a unified Request into a Gemini generateContent body."""
        body: dict[str, Any] = {}

        # System instruction
        system_parts: list[dict[str, Any]] = []
        contents: list[dict[str, Any]] = []

        for msg in request.messages:
            if msg.role in (Role.SYSTEM, Role.DEVELOPER):
                for part in msg.content:
                    if part.kind == ContentKind.TEXT and part.text:
                        system_parts.append({"text": part.text})
            elif msg.role == Role.USER:
                parts = self._translate_content_parts(msg.content)
                contents.append({"role": "user", "parts": parts})
            elif msg.role == Role.ASSISTANT:
                parts = self._translate_content_parts(msg.content)
                contents.append({"role": "model", "parts": parts})
            elif msg.role == Role.TOOL:
                parts = self._translate_tool_result_parts(msg)
                contents.append({"role": "user", "parts": parts})

        # Merge consecutive same-role entries
        contents = self._merge_consecutive(contents)

        if system_parts:
            body["system_instruction"] = {"parts": system_parts}

        body["contents"] = contents

        # Generation config
        gen_config: dict[str, Any] = {}
        if request.temperature is not None:
            gen_config["temperature"] = request.temperature
        if request.top_p is not None:
            gen_config["topP"] = request.top_p
        if request.max_tokens is not None:
            gen_config["maxOutputTokens"] = request.max_tokens
        if request.stop_sequences:
            gen_config["stopSequences"] = list(request.stop_sequences)
        if gen_config:
            body["generationConfig"] = gen_config

        # Tools
        if request.tools and (
            not request.tool_choice
            or request.tool_choice.mode != ToolChoiceMode.NONE
        ):
            func_declarations = [
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
                for t in request.tools
            ]
            body["tools"] = [{"functionDeclarations": func_declarations}]

        # Tool config (tool choice)
        if request.tool_choice:
            tc = request.tool_choice
            if tc.mode == ToolChoiceMode.AUTO:
                body["tool_config"] = {
                    "function_calling_config": {"mode": "AUTO"}
                }
            elif tc.mode == ToolChoiceMode.NONE:
                body["tool_config"] = {
                    "function_calling_config": {"mode": "NONE"}
                }
                body.pop("tools", None)
            elif tc.mode == ToolChoiceMode.REQUIRED:
                body["tool_config"] = {
                    "function_calling_config": {"mode": "ANY"}
                }
            elif tc.mode == ToolChoiceMode.NAMED and tc.tool_name:
                body["tool_config"] = {
                    "function_calling_config": {
                        "mode": "ANY",
                        "allowed_function_names": [tc.tool_name],
                    }
                }

        # Provider options
        provider_opts = (request.provider_options or {}).get("gemini", {})
        if provider_opts:
            body.update(provider_opts)

        return body

    def _translate_content_parts(
        self, parts: tuple[ContentPart, ...]
    ) -> list[dict[str, Any]]:
        """Convert unified content parts to Gemini parts."""
        blocks: list[dict[str, Any]] = []
        for part in parts:
            block = self._translate_one_part(part)
            if block is not None:
                blocks.append(block)
        return blocks or [{"text": ""}]

    def _translate_one_part(self, part: ContentPart) -> dict[str, Any] | None:
        """Translate a single ContentPart to a Gemini part."""
        if part.kind == ContentKind.TEXT:
            return {"text": part.text or ""}

        if part.kind == ContentKind.IMAGE and part.image:
            img = part.image
            if img.data is not None and img.media_type:
                encoded = (
                    encode_to_base64(img.data)
                    if isinstance(img.data, bytes)
                    else img.data
                )
                return {
                    "inlineData": {
                        "mimeType": img.media_type,
                        "data": encoded,
                    }
                }
            if img.url:
                return {
                    "fileData": {
                        "mimeType": img.media_type or "image/jpeg",
                        "fileUri": img.url,
                    }
                }

        if part.kind == ContentKind.TOOL_CALL and part.tool_call:
            tc = part.tool_call
            args = tc.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    args = {}
            return {"functionCall": {"name": tc.name, "args": args}}

        return None

    def _translate_tool_result_parts(
        self, msg: Message
    ) -> list[dict[str, Any]]:
        """Translate a TOOL message into Gemini functionResponse parts."""
        parts: list[dict[str, Any]] = []
        for part in msg.content:
            if part.kind == ContentKind.TOOL_RESULT and part.tool_result:
                tr = part.tool_result
                result_content = tr.content
                if isinstance(result_content, str):
                    result_content = {"result": result_content}
                elif isinstance(result_content, dict):
                    result_content = result_content
                else:
                    result_content = {"result": str(result_content)}

                # Use the tool_call_id as the function name lookup key
                # Gemini expects the function name, but we only have tool_call_id
                # The name should come from somewhere; use tool_call_id as fallback
                parts.append(
                    {
                        "functionResponse": {
                            "name": tr.tool_call_id,
                            "response": result_content,
                        }
                    }
                )
        return parts or [{"text": ""}]

    def _merge_consecutive(
        self, contents: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Merge consecutive same-role entries."""
        if not contents:
            return contents
        merged: list[dict[str, Any]] = [contents[0]]
        for item in contents[1:]:
            if item["role"] == merged[-1]["role"]:
                merged[-1]["parts"].extend(item["parts"])
            else:
                merged.append(item)
        return merged

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, raw: dict[str, Any]) -> Response:
        """Parse a Gemini generateContent response."""
        candidates = raw.get("candidates", [])
        parts_list: list[ContentPart] = []
        has_function_calls = False
        finish_reason_raw = ""

        if candidates:
            candidate = candidates[0]
            content = candidate.get("content", {})
            raw_parts = content.get("parts", [])
            finish_reason_raw = candidate.get("finishReason", "")

            for rp in raw_parts:
                if "text" in rp:
                    parts_list.append(ContentPart.of_text(rp["text"]))
                elif "functionCall" in rp:
                    has_function_calls = True
                    fc = rp["functionCall"]
                    call_id = f"call_{uuid4().hex[:8]}"
                    parts_list.append(
                        ContentPart.of_tool_call(
                            id=call_id,
                            name=fc.get("name", ""),
                            arguments=fc.get("args", {}),
                        )
                    )

        message = Message(role=Role.ASSISTANT, content=tuple(parts_list))

        # Finish reason mapping
        finish_reason = self._map_finish_reason(finish_reason_raw)
        if has_function_calls and finish_reason not in (
            FinishReason.TOOL_CALLS,
        ):
            finish_reason = FinishReason.TOOL_CALLS

        # Usage
        usage_meta = raw.get("usageMetadata", {})
        input_tokens = usage_meta.get("promptTokenCount", 0)
        output_tokens = usage_meta.get("candidatesTokenCount", 0)
        total_tokens = usage_meta.get("totalTokenCount", input_tokens + output_tokens)
        thoughts_tokens = usage_meta.get("thoughtsTokenCount")
        cached_tokens = usage_meta.get("cachedContentTokenCount")

        usage = Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            reasoning_tokens=thoughts_tokens,
            cache_read_tokens=cached_tokens,
            raw=usage_meta if usage_meta else None,
        )

        return Response(
            id=raw.get("id", ""),
            model=raw.get("modelVersion", ""),
            provider="gemini",
            message=message,
            finish_reason=FinishReasonInfo(reason=finish_reason, raw=finish_reason_raw),
            usage=usage,
            raw=raw,
        )

    def _map_finish_reason(self, raw: str) -> FinishReason:
        """Map Gemini finishReason to FinishReason."""
        mapping = {
            "STOP": FinishReason.STOP,
            "MAX_TOKENS": FinishReason.LENGTH,
            "SAFETY": FinishReason.CONTENT_FILTER,
            "RECITATION": FinishReason.CONTENT_FILTER,
            "OTHER": FinishReason.OTHER,
        }
        return mapping.get(raw, FinishReason.OTHER)

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def _translate_error(self, response: httpx.Response) -> Exception:
        """Translate an HTTP error response to a unified error."""
        try:
            body = response.json()
            error_info = body.get("error", {})
            message = error_info.get("message", response.text)
            error_code = error_info.get("status") or error_info.get("code")
        except Exception:
            message = response.text
            error_code = None
            body = None

        return error_from_status_code(
            status_code=response.status_code,
            message=message,
            provider="gemini",
            error_code=str(error_code) if error_code else None,
            raw=body,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def _endpoint(self, model: str, method: str = "generateContent") -> str:
        """Build the API endpoint path for a given model and method."""
        return f"/v1beta/models/{model}:{method}"

    def complete(self, request: Request) -> Response:
        """Send a request and return the full response."""
        body = self._build_request_body(request)
        url = self._endpoint(request.model)

        try:
            http_response = self._client.post(
                url,
                json=body,
                params={"key": self._api_key},
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
        url = self._endpoint(request.model, method="streamGenerateContent")

        try:
            with self._client.stream(
                "POST",
                url,
                json=body,
                params={"key": self._api_key, "alt": "sse"},
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
        """Parse SSE events from a Gemini streaming response."""
        is_first = True
        accumulated_usage: dict[str, Any] = {}
        finish_reason_raw = ""

        for sse_event in parse_sse_lines(http_response.iter_lines()):
            if not sse_event.data:
                continue

            try:
                data = json.loads(sse_event.data)
            except json.JSONDecodeError:
                continue

            if is_first:
                yield StreamEvent(
                    type=StreamEventType.STREAM_START,
                    raw=data,
                )
                is_first = False

            candidates = data.get("candidates", [])
            is_last_chunk = False

            if candidates:
                candidate = candidates[0]
                content = candidate.get("content", {})
                raw_parts = content.get("parts", [])
                chunk_finish_reason = candidate.get("finishReason", "")
                if chunk_finish_reason:
                    finish_reason_raw = chunk_finish_reason
                    is_last_chunk = True

                for rp in raw_parts:
                    if "text" in rp:
                        yield StreamEvent(
                            type=StreamEventType.TEXT_DELTA,
                            delta=rp["text"],
                            raw=data,
                        )
                    elif "functionCall" in rp:
                        fc = rp["functionCall"]
                        call_id = f"call_{uuid4().hex[:8]}"
                        yield StreamEvent(
                            type=StreamEventType.TOOL_CALL_START,
                            tool_call=ToolCall(
                                id=call_id,
                                name=fc.get("name", ""),
                            ),
                            raw=data,
                        )
                        yield StreamEvent(
                            type=StreamEventType.TOOL_CALL_END,
                            tool_call=ToolCall(
                                id=call_id,
                                name=fc.get("name", ""),
                                arguments=fc.get("args", {}),
                            ),
                            raw=data,
                        )

            # Usage metadata
            usage_meta = data.get("usageMetadata", {})
            if usage_meta:
                accumulated_usage.update(usage_meta)

            if is_last_chunk:
                finish_reason = self._map_finish_reason(finish_reason_raw)
                # Check if any function calls were in this response
                input_tokens = accumulated_usage.get("promptTokenCount", 0)
                output_tokens = accumulated_usage.get("candidatesTokenCount", 0)
                total_tokens = accumulated_usage.get(
                    "totalTokenCount", input_tokens + output_tokens
                )

                usage = Usage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    reasoning_tokens=accumulated_usage.get("thoughtsTokenCount"),
                    cache_read_tokens=accumulated_usage.get(
                        "cachedContentTokenCount"
                    ),
                    raw=accumulated_usage if accumulated_usage else None,
                )

                yield StreamEvent(
                    type=StreamEventType.FINISH,
                    finish_reason=FinishReasonInfo(
                        reason=finish_reason, raw=finish_reason_raw
                    ),
                    usage=usage,
                    response=Response(
                        id="",
                        model="",
                        provider="gemini",
                        finish_reason=FinishReasonInfo(
                            reason=finish_reason, raw=finish_reason_raw
                        ),
                        usage=usage,
                    ),
                    raw=data,
                )

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()
