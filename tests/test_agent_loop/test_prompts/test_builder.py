"""Tests for system prompt construction."""

import datetime

import pytest

from agent_loop.environment.stub import StubExecutionEnvironment
from agent_loop.prompts.builder import (
    PROMPT_BUDGET_BYTES,
    build_environment_context,
    build_system_prompt,
    discover_project_docs,
    format_tool_descriptions,
    get_git_context,
)
from agent_loop.tools.registry import ToolDefinition


# --- Environment context ---


class TestBuildEnvironmentContext:
    def test_includes_working_directory(self):
        env = StubExecutionEnvironment(working_dir="/app")
        result = build_environment_context(env, git_context={"is_repo": False})
        assert "/app" in result

    def test_includes_platform(self):
        env = StubExecutionEnvironment(plat="linux")
        result = build_environment_context(env, git_context={"is_repo": False})
        assert "linux" in result

    def test_includes_os_version(self):
        env = StubExecutionEnvironment(os_ver="Darwin 24.0.0")
        result = build_environment_context(env, git_context={"is_repo": False})
        assert "Darwin 24.0.0" in result

    def test_includes_today_date(self):
        env = StubExecutionEnvironment()
        result = build_environment_context(env, git_context={"is_repo": False})
        assert datetime.date.today().isoformat() in result

    def test_includes_model_when_provided(self):
        env = StubExecutionEnvironment()
        result = build_environment_context(env, model="claude-opus-4-6", git_context={"is_repo": False})
        assert "claude-opus-4-6" in result

    def test_omits_model_when_empty(self):
        env = StubExecutionEnvironment()
        result = build_environment_context(env, git_context={"is_repo": False})
        assert "Model:" not in result

    def test_format_is_environment_block(self):
        env = StubExecutionEnvironment()
        result = build_environment_context(env, git_context={"is_repo": False})
        assert result.startswith("<environment>")
        assert result.endswith("</environment>")

    def test_includes_git_branch_when_repo(self):
        env = StubExecutionEnvironment()
        git = {"is_repo": True, "branch": "main"}
        result = build_environment_context(env, git_context=git)
        assert "Is git repository: True" in result
        assert "Git branch: main" in result

    def test_omits_git_branch_when_not_repo(self):
        env = StubExecutionEnvironment()
        result = build_environment_context(env, git_context={"is_repo": False})
        assert "Git branch:" not in result


# --- Git context ---


class TestGetGitContext:
    def test_returns_is_repo_true_in_git_dir(self, tmp_path):
        """A git-initialized directory returns is_repo=True."""
        import subprocess
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        subprocess.run(["git", "commit", "--allow-empty", "-m", "init"],
                       cwd=str(tmp_path), capture_output=True,
                       env={"GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "t@t",
                            "GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "t@t",
                            "HOME": str(tmp_path), "PATH": "/usr/bin:/bin:/usr/local/bin"})
        context = get_git_context(str(tmp_path))
        assert context["is_repo"] is True
        assert "branch" in context

    def test_returns_is_repo_false_when_not_git(self, tmp_path):
        context = get_git_context(str(tmp_path))
        assert context["is_repo"] is False


# --- Project document discovery ---


class TestDiscoverProjectDocs:
    def test_discovers_agents_md(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text("# Agents")
        docs = discover_project_docs(str(tmp_path), provider_id="anthropic")
        assert len(docs) == 1
        assert "# Agents" in docs[0]

    def test_discovers_claude_md_for_anthropic(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("# Claude")
        docs = discover_project_docs(str(tmp_path), provider_id="anthropic")
        assert any("# Claude" in d for d in docs)

    def test_discovers_gemini_md_for_gemini(self, tmp_path):
        (tmp_path / "GEMINI.md").write_text("# Gemini")
        docs = discover_project_docs(str(tmp_path), provider_id="gemini")
        assert any("# Gemini" in d for d in docs)

    def test_ignores_claude_md_for_openai(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("# Claude")
        docs = discover_project_docs(str(tmp_path), provider_id="openai")
        assert not any("# Claude" in d for d in docs)

    def test_returns_empty_when_no_files(self, tmp_path):
        docs = discover_project_docs(str(tmp_path), provider_id="anthropic")
        assert docs == []

    def test_truncates_at_32kb_budget(self, tmp_path):
        huge_content = "x" * (PROMPT_BUDGET_BYTES + 1000)
        (tmp_path / "AGENTS.md").write_text(huge_content)
        docs = discover_project_docs(str(tmp_path), provider_id="anthropic")
        assert len(docs) == 1
        assert "truncated at 32KB" in docs[0]

    def test_discovers_codex_instructions_for_openai(self, tmp_path):
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        (codex_dir / "instructions.md").write_text("# Codex")
        docs = discover_project_docs(str(tmp_path), provider_id="openai")
        assert any("# Codex" in d for d in docs)


# --- Tool descriptions ---


class TestFormatToolDescriptions:
    def test_formats_single_tool(self):
        defs = [ToolDefinition(name="read_file", description="Read a file")]
        result = format_tool_descriptions(defs)
        assert "read_file" in result
        assert "Read a file" in result

    def test_formats_multiple_tools(self):
        defs = [
            ToolDefinition(name="read_file", description="Read a file"),
            ToolDefinition(name="shell", description="Run a command"),
        ]
        result = format_tool_descriptions(defs)
        assert "read_file" in result
        assert "shell" in result

    def test_empty_list_returns_empty(self):
        assert format_tool_descriptions([]) == ""


# --- System prompt assembly ---


class TestBuildSystemPrompt:
    def _make_env(self):
        return StubExecutionEnvironment(working_dir="/tmp/test")

    def test_includes_base_instructions(self):
        result = build_system_prompt(
            "You are an assistant.", self._make_env(), git_context={"is_repo": False},
        )
        assert "You are an assistant." in result

    def test_includes_environment_context(self):
        result = build_system_prompt("", self._make_env(), git_context={"is_repo": False})
        assert "<environment>" in result
        assert "/tmp/test" in result

    def test_includes_tool_descriptions(self):
        defs = [ToolDefinition(name="shell", description="Run a command")]
        result = build_system_prompt("", self._make_env(), tool_definitions=defs, git_context={"is_repo": False})
        assert "shell" in result
        assert "Run a command" in result

    def test_user_instructions_appended_last(self):
        result = build_system_prompt(
            "base", self._make_env(),
            user_instructions="OVERRIDE ME",
            git_context={"is_repo": False},
        )
        # User instructions should be the last layer
        assert result.endswith("OVERRIDE ME")

    def test_works_with_no_user_instructions(self):
        result = build_system_prompt("base", self._make_env(), git_context={"is_repo": False})
        assert "base" in result

    def test_works_with_empty_base(self):
        result = build_system_prompt("", self._make_env(), git_context={"is_repo": False})
        assert "<environment>" in result

    def test_layers_in_correct_order(self):
        defs = [ToolDefinition(name="test_tool", description="A test")]
        result = build_system_prompt(
            "BASE_INSTRUCTIONS",
            self._make_env(),
            tool_definitions=defs,
            user_instructions="USER_OVERRIDE",
            git_context={"is_repo": False},
        )
        base_pos = result.index("BASE_INSTRUCTIONS")
        env_pos = result.index("<environment>")
        tool_pos = result.index("test_tool")
        user_pos = result.index("USER_OVERRIDE")
        assert base_pos < env_pos < tool_pos < user_pos
