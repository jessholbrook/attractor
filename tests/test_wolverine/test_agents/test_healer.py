"""Tests for the HealerAgent."""
from __future__ import annotations

import pytest

from agent_loop.client import CompletionResponse, Message as AgentMessage, StubClient
from agent_loop.environment.stub import StubExecutionEnvironment
from agent_loop.providers.profile import StubProfile
from agent_loop.turns import AssistantTurn

from wolverine.agents.healer import HealerAgent
from wolverine.model.issue import Issue, IssueCategory, IssueSeverity, IssueStatus
from wolverine.model.solution import Solution, SolutionStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_issue(**overrides) -> Issue:
    """Create a test Issue with sensible defaults."""
    defaults = dict(
        id="ISS-42",
        title="NullPointerException in UserService",
        description="The getUserById method throws NPE when user does not exist.",
        severity=IssueSeverity.HIGH,
        status=IssueStatus.DIAGNOSED,
        category=IssueCategory.BUG,
        root_cause="Missing null check in getUserById before accessing user.name",
        affected_files=("/src/services/user_service.py", "/src/models/user.py"),
    )
    defaults.update(overrides)
    return Issue(**defaults)


def _make_healer(
    responses: list[CompletionResponse] | None = None,
    issue: Issue | None = None,
    test_command: str = "",
    max_turns: int = 30,
) -> HealerAgent:
    """Create a HealerAgent with stub dependencies."""
    if responses is None:
        responses = [
            CompletionResponse(message=AgentMessage.assistant("Fixed the null check bug.")),
        ]
    client = StubClient(responses=responses)
    env = StubExecutionEnvironment()
    profile = StubProfile()
    return HealerAgent(
        llm_client=client,
        execution_env=env,
        provider_profile=profile,
        issue=issue or _make_issue(),
        test_command=test_command,
        max_turns=max_turns,
    )


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestHealerConstruction:

    def test_constructs_with_stubs(self):
        healer = _make_healer()
        assert healer is not None

    def test_session_is_accessible(self):
        healer = _make_healer()
        assert healer.session is not None

    def test_registers_query_issue_tool(self):
        profile = StubProfile()
        client = StubClient()
        env = StubExecutionEnvironment()
        HealerAgent(
            llm_client=client,
            execution_env=env,
            provider_profile=profile,
            issue=_make_issue(),
        )
        names = profile.tool_registry.names()
        assert "query_issue" in names

    def test_registers_run_tests_tool(self):
        profile = StubProfile()
        client = StubClient()
        env = StubExecutionEnvironment()
        HealerAgent(
            llm_client=client,
            execution_env=env,
            provider_profile=profile,
            issue=_make_issue(),
        )
        names = profile.tool_registry.names()
        assert "run_tests" in names

    def test_custom_max_turns(self):
        healer = _make_healer(max_turns=5)
        assert healer.session.config.max_turns == 5


# ---------------------------------------------------------------------------
# generate_solution tests
# ---------------------------------------------------------------------------

class TestGenerateSolution:

    def test_returns_solution(self):
        healer = _make_healer()
        solution = healer.generate_solution()
        assert isinstance(solution, Solution)

    def test_solution_has_correct_issue_id(self):
        issue = _make_issue(id="ISS-99")
        healer = _make_healer(issue=issue)
        solution = healer.generate_solution()
        assert solution.issue_id == "ISS-99"

    def test_solution_status_is_generated(self):
        healer = _make_healer()
        solution = healer.generate_solution()
        assert solution.status == SolutionStatus.GENERATED

    def test_solution_has_summary(self):
        responses = [
            CompletionResponse(message=AgentMessage.assistant("Added null check to getUserById.")),
        ]
        healer = _make_healer(responses=responses)
        solution = healer.generate_solution()
        assert "null check" in solution.summary

    def test_solution_has_reasoning(self):
        healer = _make_healer()
        solution = healer.generate_solution()
        assert "NullPointerException" in solution.reasoning

    def test_solution_has_agent_session_id(self):
        healer = _make_healer()
        solution = healer.generate_solution()
        assert solution.agent_session_id == healer.session.id

    def test_solution_has_unique_id(self):
        healer = _make_healer()
        s1 = healer.generate_solution()
        # Need a fresh healer for a second solution (session state matters)
        healer2 = _make_healer()
        s2 = healer2.generate_solution()
        assert s1.id != s2.id


# ---------------------------------------------------------------------------
# Prompt building tests
# ---------------------------------------------------------------------------

class TestBuildPrompt:

    def test_prompt_includes_title(self):
        healer = _make_healer()
        prompt = healer._build_prompt()
        assert "NullPointerException in UserService" in prompt

    def test_prompt_includes_description(self):
        healer = _make_healer()
        prompt = healer._build_prompt()
        assert "getUserById" in prompt

    def test_prompt_includes_root_cause(self):
        healer = _make_healer()
        prompt = healer._build_prompt()
        assert "Missing null check" in prompt

    def test_prompt_includes_affected_files(self):
        healer = _make_healer()
        prompt = healer._build_prompt()
        assert "user_service.py" in prompt
        assert "user.py" in prompt

    def test_prompt_with_no_affected_files(self):
        issue = _make_issue(affected_files=())
        healer = _make_healer(issue=issue)
        prompt = healer._build_prompt()
        assert "unknown" in prompt

    def test_prompt_asks_for_concrete_fix(self):
        healer = _make_healer()
        prompt = healer._build_prompt()
        assert "concrete code fix" in prompt.lower() or "fix" in prompt.lower()


# ---------------------------------------------------------------------------
# Summary extraction tests
# ---------------------------------------------------------------------------

class TestExtractSummary:

    def test_extracts_from_last_turn(self):
        healer = _make_healer()
        turn = AssistantTurn(content="I fixed the bug by adding a null check.")
        summary = healer._extract_summary(turn)
        assert "null check" in summary

    def test_fallback_when_last_turn_empty(self):
        """When the last turn has no content, should walk history or use default."""
        healer = _make_healer()
        turn = AssistantTurn(content="")
        summary = healer._extract_summary(turn)
        assert summary == "Solution generated"
