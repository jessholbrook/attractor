"""HealerAgent: wraps agent_loop.Session to generate code fixes."""
from __future__ import annotations

import uuid

from agent_loop.client import Client
from agent_loop.environment.types import ExecutionEnvironment
from agent_loop.events import EventEmitter
from agent_loop.providers.profile import ProviderProfile
from agent_loop.session import Session
from agent_loop.session_config import SessionConfig
from agent_loop.tools.registry import RegisteredTool
from agent_loop.turns import AssistantTurn

from wolverine.agents.tools import (
    QUERY_ISSUE,
    RUN_TESTS,
    make_query_issue_executor,
    make_run_tests_executor,
)
from wolverine.model.issue import Issue
from wolverine.model.solution import Solution, SolutionStatus


class HealerAgent:
    """Generates code fixes by wrapping an agent_loop.Session.

    Given an Issue, constructs a prompt, registers custom tools (query_issue,
    run_tests), then runs the agent loop to produce a Solution.
    """

    def __init__(
        self,
        llm_client: Client,
        execution_env: ExecutionEnvironment,
        provider_profile: ProviderProfile,
        issue: Issue,
        *,
        test_command: str = "",
        max_turns: int = 30,
        event_emitter: EventEmitter | None = None,
    ) -> None:
        self._issue = issue
        self._test_command = test_command
        self._event_emitter = event_emitter or EventEmitter()

        # Register custom tools on the provider profile's registry
        issue_data = {
            "id": issue.id,
            "title": issue.title,
            "description": issue.description,
            "severity": issue.severity,
            "status": issue.status,
            "root_cause": issue.root_cause,
            "affected_files": list(issue.affected_files),
        }
        provider_profile.tool_registry.register(
            RegisteredTool(
                definition=QUERY_ISSUE,
                executor=make_query_issue_executor(issue_data),
            )
        )
        provider_profile.tool_registry.register(
            RegisteredTool(
                definition=RUN_TESTS,
                executor=make_run_tests_executor(test_command or "echo 'no tests configured'"),
            )
        )

        self._session = Session(
            llm_client=llm_client,
            provider_profile=provider_profile,
            execution_env=execution_env,
            config=SessionConfig(
                max_turns=max_turns,
                max_tool_rounds_per_input=100,
            ),
            event_emitter=self._event_emitter,
        )

    @property
    def session(self) -> Session:
        """Access the underlying agent_loop Session."""
        return self._session

    def generate_solution(self) -> Solution:
        """Run the agent loop and return a Solution."""
        prompt = self._build_prompt()
        assistant_turn = self._session.process_input(prompt)

        summary = self._extract_summary(assistant_turn)

        return Solution(
            id=uuid.uuid4().hex[:12],
            issue_id=self._issue.id,
            status=SolutionStatus.GENERATED,
            summary=summary,
            reasoning=f"Automated fix for: {self._issue.title}",
            agent_session_id=self._session.id,
        )

    def _build_prompt(self) -> str:
        """Build the initial user prompt from the issue details."""
        affected = ", ".join(self._issue.affected_files) if self._issue.affected_files else "unknown"
        return (
            f"Fix this issue:\n\n"
            f"Title: {self._issue.title}\n"
            f"Description: {self._issue.description}\n"
            f"Root Cause: {self._issue.root_cause}\n"
            f"Affected Files: {affected}\n\n"
            f"Generate a concrete code fix. Read the affected files, understand the issue, "
            f"write the fix, and run tests to verify."
        )

    def _extract_summary(self, last_turn: AssistantTurn) -> str:
        """Extract a summary from the agent's final turn or history."""
        if last_turn.content:
            return last_turn.content

        # Walk history backwards for the last assistant text
        for turn in reversed(self._session.history):
            if isinstance(turn, AssistantTurn) and turn.content:
                return turn.content

        return "Solution generated"
