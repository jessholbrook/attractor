"""Core agentic loop: the Session orchestrator."""

from __future__ import annotations

import uuid
from collections import deque
from typing import Any

from agent_loop.client import Client, CompletionRequest, CompletionResponse, Message
from agent_loop.environment.types import ExecutionEnvironment
from agent_loop.events import (
    AssistantTextEndEvent,
    ErrorEvent,
    EventEmitter,
    LoopDetectionEvent,
    SessionEndEvent,
    SessionStartEvent,
    SteeringInjectedEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    TurnLimitEvent,
    UserInputEvent,
)
from agent_loop.loop_detection import ToolCallSignature, detect_loop, make_signature
from agent_loop.providers.profile import ProviderProfile
from agent_loop.session_config import SessionConfig, SessionState
from agent_loop.truncation import truncate_tool_output
from agent_loop.turns import (
    AssistantTurn,
    SteeringTurn,
    ToolCall,
    ToolResult,
    ToolResultsTurn,
    Turn,
    UserTurn,
)


class Session:
    """Central orchestrator for the coding agent loop.

    Holds conversation state, dispatches tool calls, manages the event stream,
    and enforces limits. This is the primary interface for host applications.
    """

    def __init__(
        self,
        llm_client: Client,
        provider_profile: ProviderProfile,
        execution_env: ExecutionEnvironment,
        config: SessionConfig | None = None,
        event_emitter: EventEmitter | None = None,
        session_id: str | None = None,
        depth: int = 0,
    ) -> None:
        self.id = session_id or str(uuid.uuid4())
        self.llm_client = llm_client
        self.provider_profile = provider_profile
        self.execution_env = execution_env
        self.config = config or SessionConfig()
        self.event_emitter = event_emitter or EventEmitter()
        self.state = SessionState.IDLE
        self.history: list[Turn] = []
        self.depth = depth

        self._steering_queue: deque[str] = deque()
        self._followup_queue: deque[str] = deque()
        self._tool_signatures: list[ToolCallSignature] = []
        self._abort = False
        self._subagents: dict[str, Any] = {}

    # --- Public API ---

    def process_input(self, user_input: str) -> AssistantTurn:
        """Run the core agentic loop for a user input.

        Returns the final AssistantTurn (text-only response or state at limit).
        """
        if self.state == SessionState.CLOSED:
            raise RuntimeError("Session is closed")

        self.state = SessionState.PROCESSING
        self.history.append(UserTurn(content=user_input))
        self.event_emitter.emit(UserInputEvent(content=user_input))

        # Drain any pending steering before first LLM call
        self._drain_steering()

        round_count = 0
        last_assistant_turn = AssistantTurn(content="")

        while True:
            # 1. Check limits
            if round_count >= self.config.max_tool_rounds_per_input:
                self.event_emitter.emit(TurnLimitEvent(
                    turns_used=round_count, max_turns=self.config.max_tool_rounds_per_input,
                ))
                break

            if self.config.max_turns > 0 and self._count_turns() >= self.config.max_turns:
                self.event_emitter.emit(TurnLimitEvent(
                    turns_used=self._count_turns(), max_turns=self.config.max_turns,
                ))
                break

            if self._abort:
                break

            # 2. Build LLM request
            system_prompt = self.provider_profile.build_system_prompt(self.execution_env)
            messages = self._convert_history_to_messages()
            tool_defs = self.provider_profile.tools()

            request = CompletionRequest(
                messages=[Message.system(system_prompt)] + messages,
                model=self.provider_profile.model,
                tools=[{"type": "function", "function": {"name": td.name, "description": td.description, "parameters": td.parameters}} for td in tool_defs],
                reasoning_effort=self.config.reasoning_effort,
            )

            # 3. Call LLM
            try:
                response = self.llm_client.complete(request)
            except Exception as e:
                self.event_emitter.emit(ErrorEvent(error=str(e), recoverable=False))
                self.state = SessionState.CLOSED
                self.event_emitter.emit(SessionEndEvent(session_id=self.id, reason="error"))
                raise

            # 4. Record assistant turn
            assistant_turn = AssistantTurn(
                content=response.text,
                tool_calls=response.tool_calls,
                usage=response.usage,
            )
            self.history.append(assistant_turn)
            last_assistant_turn = assistant_turn

            self.event_emitter.emit(AssistantTextEndEvent(
                full_text=response.text,
            ))

            # 5. If no tool calls, natural completion
            if not response.tool_calls:
                break

            # 6. Execute tool calls
            round_count += 1
            results = self._execute_tool_calls(response.tool_calls)
            self.history.append(ToolResultsTurn(results=results))

            # 7. Drain steering injected during tool execution
            self._drain_steering()

            # 8. Loop detection
            if self.config.enable_loop_detection:
                loop_msg = detect_loop(
                    self._tool_signatures,
                    window=self.config.loop_detection_window,
                )
                if loop_msg:
                    self.history.append(SteeringTurn(content=loop_msg))
                    self.event_emitter.emit(LoopDetectionEvent(message=loop_msg))

        # Process follow-up messages
        if self._followup_queue:
            next_input = self._followup_queue.popleft()
            return self.process_input(next_input)

        self.state = SessionState.IDLE
        return last_assistant_turn

    def steer(self, message: str) -> None:
        """Queue a steering message for injection after the current tool round."""
        self._steering_queue.append(message)

    def follow_up(self, message: str) -> None:
        """Queue a message to process after the current input completes."""
        self._followup_queue.append(message)

    def abort(self) -> None:
        """Signal the processing loop to stop."""
        self._abort = True

    def close(self) -> None:
        """Close the session, preventing further input."""
        self.state = SessionState.CLOSED
        self.event_emitter.emit(SessionEndEvent(session_id=self.id, reason="closed"))

    # --- Private methods ---

    def _drain_steering(self) -> None:
        """Flush all pending steering messages into history."""
        while self._steering_queue:
            msg = self._steering_queue.popleft()
            self.history.append(SteeringTurn(content=msg))
            self.event_emitter.emit(SteeringInjectedEvent(content=msg))

    def _execute_tool_calls(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute tool calls sequentially through the registry."""
        results = []
        for tc in tool_calls:
            result = self._execute_single_tool(tc)
            results.append(result)
        return results

    def _execute_single_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call: lookup -> execute -> truncate -> emit."""
        self.event_emitter.emit(ToolCallStartEvent(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            arguments=tool_call.arguments,
        ))

        # Track signature for loop detection
        sig = make_signature(tool_call.name, tool_call.arguments)
        self._tool_signatures.append(sig)

        # Lookup tool
        registered = self.provider_profile.tool_registry.get(tool_call.name)
        if registered is None:
            error_msg = f"Unknown tool: {tool_call.name}"
            self.event_emitter.emit(ToolCallEndEvent(
                tool_call_id=tool_call.id, tool_name=tool_call.name,
                output=error_msg, is_error=True,
            ))
            return ToolResult(tool_call_id=tool_call.id, output=error_msg, is_error=True)

        # Execute
        try:
            raw_output = registered.executor(tool_call.arguments, self.execution_env)

            # Truncate for LLM
            truncated = truncate_tool_output(raw_output, tool_call.name)

            # Emit full output via event
            self.event_emitter.emit(ToolCallEndEvent(
                tool_call_id=tool_call.id, tool_name=tool_call.name,
                output=raw_output,
            ))

            return ToolResult(tool_call_id=tool_call.id, output=truncated)

        except Exception as e:
            error_msg = f"Tool error ({tool_call.name}): {e}"
            self.event_emitter.emit(ToolCallEndEvent(
                tool_call_id=tool_call.id, tool_name=tool_call.name,
                output=error_msg, is_error=True,
            ))
            return ToolResult(tool_call_id=tool_call.id, output=error_msg, is_error=True)

    def _convert_history_to_messages(self) -> list[Message]:
        """Convert Turn history to Message list for LLM request."""
        messages: list[Message] = []
        for turn in self.history:
            if isinstance(turn, UserTurn):
                messages.append(Message.user(turn.content))
            elif isinstance(turn, AssistantTurn):
                tc_list = turn.tool_calls if turn.tool_calls else None
                messages.append(Message.assistant(turn.content, tool_calls=tc_list))
            elif isinstance(turn, ToolResultsTurn):
                for result in turn.results:
                    messages.append(Message.tool(
                        tool_call_id=result.tool_call_id,
                        content=result.output,
                    ))
            elif isinstance(turn, SteeringTurn):
                messages.append(Message.user(turn.content))
        return messages

    def _count_turns(self) -> int:
        """Count total turns in history (user + assistant pairs)."""
        return sum(1 for t in self.history if isinstance(t, (UserTurn, AssistantTurn)))
