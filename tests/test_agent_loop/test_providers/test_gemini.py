"""Tests for the Gemini provider profile."""

from agent_loop.environment.stub import StubExecutionEnvironment
from agent_loop.environment.types import DirEntry
from agent_loop.providers.gemini import GeminiProfile, list_dir_executor
from agent_loop.providers.profile import ProviderProfile
from agent_loop.tools.registry import ToolDefinition


class TestGeminiProfileProtocol:
    def test_satisfies_provider_profile(self):
        profile: ProviderProfile = GeminiProfile()
        assert profile.id == "gemini"


class TestGeminiProfileProperties:
    def test_id(self):
        assert GeminiProfile().id == "gemini"

    def test_default_model(self):
        assert GeminiProfile().model == "gemini-2.5-pro"

    def test_custom_model(self):
        assert GeminiProfile(model="gemini-3-flash").model == "gemini-3-flash"

    def test_context_window(self):
        assert GeminiProfile().context_window_size == 1_000_000


class TestGeminiProfileTools:
    def test_has_core_tools_plus_list_dir(self):
        names = GeminiProfile().tool_registry.names()
        assert "read_file" in names
        assert "write_file" in names
        assert "edit_file" in names
        assert "shell" in names
        assert "grep" in names
        assert "glob" in names
        assert "list_dir" in names

    def test_has_seven_tools(self):
        assert len(GeminiProfile().tools()) == 7

    def test_tools_returns_definitions(self):
        defs = GeminiProfile().tools()
        assert all(isinstance(d, ToolDefinition) for d in defs)


class TestListDirExecutor:
    def test_formats_entries(self):
        env = StubExecutionEnvironment()
        # list_directory returns empty by default in stub
        result = list_dir_executor({"path": "/app"}, env)
        assert result == "Empty directory."


class TestGeminiProfileSystemPrompt:
    def test_returns_string(self):
        env = StubExecutionEnvironment()
        prompt = GeminiProfile().build_system_prompt(env)
        assert isinstance(prompt, str)
        assert len(prompt) > 0
