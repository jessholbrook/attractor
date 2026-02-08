"""High-level blocking generation functions."""
from __future__ import annotations

import json
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from unified_llm.client import Client, get_default_client
from unified_llm.errors import AbortError, NoObjectGeneratedError
from unified_llm.types.config import AbortSignal, RetryPolicy, TimeoutConfig
from unified_llm.types.enums import ContentKind, FinishReason, Role
from unified_llm.types.messages import Message
from unified_llm.types.request import Request, ResponseFormat
from unified_llm.types.response import FinishReasonInfo, Response, Usage
from unified_llm.types.results import GenerateResult, StepResult
from unified_llm.types.tools import Tool, ToolCall, ToolChoice, ToolResult
from unified_llm._retry import with_retry


def generate(
    model: str,
    *,
    prompt: str | None = None,
    messages: Sequence[Message] | None = None,
    system: str | None = None,
    tools: Sequence[Tool] | None = None,
    tool_choice: ToolChoice | None = None,
    max_tool_rounds: int = 1,
    response_format: ResponseFormat | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    stop_sequences: Sequence[str] | None = None,
    reasoning_effort: str | None = None,
    provider: str | None = None,
    provider_options: dict[str, Any] | None = None,
    max_retries: int = 2,
    timeout: float | TimeoutConfig | None = None,
    abort_signal: AbortSignal | None = None,
    client: Client | None = None,
) -> GenerateResult:
    """Generate a completion, with optional multi-round tool execution.

    Either *prompt* (a plain string) or *messages* (a sequence of Message
    objects) must be provided, but not both.  An optional *system* string
    is prepended as a system message.

    When *tools* are supplied and the model returns tool calls, the
    function automatically executes tools that have an ``execute`` handler
    and feeds the results back for up to *max_tool_rounds* iterations.
    """
    # --- Validate inputs ---
    if prompt is not None and messages is not None:
        raise ValueError("Provide either 'prompt' or 'messages', not both")
    if prompt is None and messages is None:
        raise ValueError("Either 'prompt' or 'messages' must be provided")

    # --- Build initial message list ---
    msg_list: list[Message] = []
    if system:
        msg_list.append(Message.system(system))
    if prompt is not None:
        msg_list.append(Message.user(prompt))
    else:
        assert messages is not None
        msg_list.extend(messages)

    # --- Resolve client ---
    c = client or get_default_client()

    # --- Prepare tools tuple ---
    tools_tuple = tuple(tools) if tools else None

    # --- Build retry policy ---
    retry_policy = RetryPolicy(max_retries=max_retries) if max_retries > 0 else None

    # --- Tool execution loop ---
    steps: list[StepResult] = []
    rounds_left = max_tool_rounds

    while True:
        # Check abort
        if abort_signal is not None and abort_signal.aborted:
            raise AbortError("Operation aborted")

        # Build request
        request = Request(
            model=model,
            messages=tuple(msg_list),
            provider=provider,
            tools=tools_tuple,
            tool_choice=tool_choice,
            response_format=response_format,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop_sequences=tuple(stop_sequences) if stop_sequences else None,
            reasoning_effort=reasoning_effort,
            provider_options=provider_options,
        )

        # Call the provider (with optional retry)
        if retry_policy is not None:
            response = with_retry(lambda req=request: c.complete(req), retry_policy)
        else:
            response = c.complete(request)

        # Extract tool call data from response
        response_tool_calls = response.tool_calls  # tuple[ToolCallData, ...]

        # Check if we have tool calls and should execute them
        has_tool_calls = len(response_tool_calls) > 0
        is_tool_finish = response.finish_reason.reason == FinishReason.TOOL_CALLS
        has_active_tools = (
            tools_tuple is not None
            and any(t.execute is not None for t in tools_tuple)
        )

        if has_tool_calls and is_tool_finish and rounds_left > 0 and has_active_tools:
            # Convert ToolCallData to ToolCall
            tool_calls = tuple(
                ToolCall(
                    id=tc.id,
                    name=tc.name,
                    arguments=tc.arguments if isinstance(tc.arguments, dict) else {},
                )
                for tc in response_tool_calls
            )

            # Execute tools
            tool_results = _execute_tools(tools_tuple, tool_calls)

            # Create step result
            step = _response_to_step(response, tool_results)
            steps.append(step)

            # Append assistant message with tool calls + tool result messages
            msg_list.append(response.message)
            for tr in tool_results:
                content = tr.content if isinstance(tr.content, str) else json.dumps(tr.content)
                msg_list.append(
                    Message.tool_result(
                        tool_call_id=tr.tool_call_id,
                        content=content,
                        is_error=tr.is_error,
                    )
                )

            rounds_left -= 1
            continue

        # Natural completion (or no more rounds)
        step = _response_to_step(response)
        steps.append(step)
        break

    # --- Build final GenerateResult ---
    final_step = steps[-1]
    total_usage = Usage()
    for s in steps:
        total_usage = total_usage + s.usage

    return GenerateResult(
        text=final_step.text,
        reasoning=final_step.reasoning,
        tool_calls=final_step.tool_calls,
        tool_results=final_step.tool_results,
        finish_reason=final_step.finish_reason,
        usage=final_step.usage,
        total_usage=total_usage,
        steps=tuple(steps),
        response=final_step.response,
    )


def _execute_tools(
    tools: Sequence[Tool],
    tool_calls: Sequence[ToolCall],
) -> tuple[ToolResult, ...]:
    """Execute tool calls, running active tools concurrently."""
    tool_map = {t.name: t for t in tools}

    def execute_one(tc: ToolCall) -> ToolResult:
        tool = tool_map.get(tc.name)
        if tool is None or tool.execute is None:
            return ToolResult(
                tool_call_id=tc.id,
                content=f"Unknown tool: {tc.name}",
                is_error=True,
            )
        try:
            result = tool.execute(**tc.arguments)
            content: str | dict[str, Any] | list[Any]
            if isinstance(result, (str, dict, list)):
                content = result
            else:
                content = str(result)
            return ToolResult(tool_call_id=tc.id, content=content)
        except Exception as e:
            return ToolResult(tool_call_id=tc.id, content=str(e), is_error=True)

    if len(tool_calls) == 1:
        return (execute_one(tool_calls[0]),)

    # Parallel execution for multiple tool calls
    results: dict[str, ToolResult] = {}
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(execute_one, tc): tc for tc in tool_calls}
        for future in as_completed(futures):
            tc = futures[future]
            results[tc.id] = future.result()

    # Preserve original order
    return tuple(results[tc.id] for tc in tool_calls)


def _response_to_step(
    response: Response, tool_results: tuple[ToolResult, ...] = ()
) -> StepResult:
    """Convert a Response + tool results into a StepResult."""
    response_tool_calls = response.tool_calls  # tuple[ToolCallData, ...]

    tool_calls = tuple(
        ToolCall(
            id=tc.id,
            name=tc.name,
            arguments=tc.arguments if isinstance(tc.arguments, dict) else {},
        )
        for tc in response_tool_calls
    )

    return StepResult(
        text=response.text,
        reasoning=response.reasoning,
        tool_calls=tool_calls,
        tool_results=tool_results,
        finish_reason=response.finish_reason,
        usage=response.usage,
        response=response,
    )


def generate_object(
    model: str,
    *,
    prompt: str | None = None,
    messages: Sequence[Message] | None = None,
    system: str | None = None,
    schema: dict[str, Any],
    temperature: float | None = None,
    max_tokens: int | None = None,
    provider: str | None = None,
    provider_options: dict[str, Any] | None = None,
    max_retries: int = 2,
    timeout: float | TimeoutConfig | None = None,
    client: Client | None = None,
) -> GenerateResult:
    """Generate structured output matching a JSON schema.

    Uses ``response_format`` with ``json_schema`` type to request
    structured output from the model.  The response text is parsed as
    JSON and placed in the ``output`` field of the returned
    :class:`GenerateResult`.

    Raises :class:`NoObjectGeneratedError` if the response cannot be
    parsed as valid JSON.
    """
    result = generate(
        model=model,
        prompt=prompt,
        messages=messages,
        system=system,
        response_format=ResponseFormat(
            type="json_schema", json_schema=schema, strict=True,
        ),
        temperature=temperature,
        max_tokens=max_tokens,
        provider=provider,
        provider_options=provider_options,
        max_retries=max_retries,
        timeout=timeout,
        client=client,
    )

    # Parse the text as JSON
    try:
        parsed = json.loads(result.text)
    except (json.JSONDecodeError, ValueError) as e:
        raise NoObjectGeneratedError(
            f"Failed to parse structured output as JSON: {e}",
            cause=e,
        )

    # Return a new GenerateResult with the parsed output
    return GenerateResult(
        text=result.text,
        reasoning=result.reasoning,
        tool_calls=result.tool_calls,
        tool_results=result.tool_results,
        finish_reason=result.finish_reason,
        usage=result.usage,
        total_usage=result.total_usage,
        steps=result.steps,
        response=result.response,
        output=parsed,
    )
