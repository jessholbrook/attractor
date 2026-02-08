"""Bridge between agent_loop.Client and unified_llm.Client."""
from __future__ import annotations

from agent_loop.client import CompletionRequest, CompletionResponse, Message as AgentMessage
from agent_loop.turns import Role as AgentRole, ToolCall as AgentToolCall
from unified_llm import (
    Client as ULMClient,
    FinishReason,
    Message as ULMMessage,
    Request as ULMRequest,
    Response as ULMResponse,
    Tool as ULMTool,
    ToolCallData,
)


class UnifiedLLMBridge:
    """Adapts a unified_llm.Client to the agent_loop.Client protocol.

    Translates between agent_loop's string-based message format and
    unified_llm's ContentPart-based multimodal format.
    """

    def __init__(self, client: ULMClient) -> None:
        self._client = client

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Translate an agent_loop request, call unified_llm, translate response back."""
        # 1. Translate messages
        messages = tuple(self._translate_message(m) for m in request.messages)

        # 2. Translate tools
        tools: tuple[ULMTool, ...] | None = None
        if request.tools:
            tools = tuple(
                ULMTool(
                    name=t.get("function", t).get("name", t.get("name", ""))
                    if isinstance(t.get("function", t), dict)
                    else t.get("name", ""),
                    description=t.get("function", t).get("description", "")
                    if isinstance(t.get("function", t), dict)
                    else t.get("description", ""),
                    parameters=t.get("function", t).get("parameters", {})
                    if isinstance(t.get("function", t), dict)
                    else t.get("parameters", {}),
                )
                for t in request.tools
            )

        # 3. Build unified_llm Request
        ulm_request = ULMRequest(
            model=request.model or "stub-model",
            messages=messages,
            tools=tools,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        # 4. Call unified_llm
        response = self._client.complete(ulm_request)

        # 5. Translate back
        return self._translate_response(response)

    def _translate_message(self, msg: AgentMessage) -> ULMMessage:
        """Convert an agent_loop Message to a unified_llm Message."""
        if msg.role == AgentRole.SYSTEM:
            return ULMMessage.system(msg.content)
        elif msg.role == AgentRole.USER:
            return ULMMessage.user(msg.content)
        elif msg.role == AgentRole.ASSISTANT:
            tool_calls: list[ToolCallData] | None = None
            if msg.tool_calls:
                tool_calls = [
                    ToolCallData(
                        id=tc.id,
                        name=tc.name,
                        arguments=tc.arguments,
                    )
                    for tc in msg.tool_calls
                ]
            return ULMMessage.assistant(msg.content, tool_calls=tool_calls)
        elif msg.role == AgentRole.TOOL:
            return ULMMessage.tool_result(
                tool_call_id=msg.tool_call_id or "",
                content=msg.content,
            )
        # Fallback
        return ULMMessage.user(msg.content)

    def _translate_response(self, response: ULMResponse) -> CompletionResponse:
        """Convert a unified_llm Response to an agent_loop CompletionResponse."""
        # Extract tool calls
        tool_calls: list[AgentToolCall] | None = None
        if response.tool_calls:
            tool_calls = [
                AgentToolCall(
                    id=tc.id,
                    name=tc.name,
                    arguments=tc.arguments if isinstance(tc.arguments, dict) else {},
                )
                for tc in response.tool_calls
            ]

        message = AgentMessage.assistant(
            content=response.text or "",
            tool_calls=tool_calls,
        )

        usage: dict[str, int] = {}
        if response.usage:
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }

        stop_reason = "end_turn"
        if response.finish_reason:
            reason = response.finish_reason.reason
            if reason == FinishReason.TOOL_CALLS:
                stop_reason = "tool_use"
            elif reason == FinishReason.LENGTH:
                stop_reason = "max_tokens"
            elif reason == FinishReason.STOP:
                stop_reason = "end_turn"
            else:
                stop_reason = reason.value

        return CompletionResponse(
            message=message,
            usage=usage,
            model=response.model or "",
            stop_reason=stop_reason,
        )
