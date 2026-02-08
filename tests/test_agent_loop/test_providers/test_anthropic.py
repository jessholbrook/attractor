"""Tests for the Anthropic provider profile."""

from agent_loop.environment.stub import StubExecutionEnvironment
from agent_loop.providers.anthropic import AnthropicProfile
from agent_loop.providers.profile import ProviderProfile
from agent_loop.tools.registry import RegisteredTool, ToolDefinition, ToolRegistry


def _stub_executor(args, env):
    return "ok"


class TestAnthropicProfileProtocol:
    def test_satisfies_provider_profile(self):
        profile: ProviderProfile = AnthropicProfile()
        assert profile.id == "anthropic"


class TestAnthropicProfileProperties:
    def test_id(self):
        assert AnthropicProfile().id == "anthropic"

    def test_default_model(self):
        assert "claude" in AnthropicProfile().model

    def test_custom_model(self):
        p = AnthropicProfile(model="claude-opus-4-6")
        assert p.model == "claude-opus-4-6"

    def test_context_window_size(self):
        assert AnthropicProfile().context_window_size == 200_000

    def test_supports_reasoning(self):
        assert AnthropicProfile().supports_reasoning is True

    def test_supports_streaming(self):
        assert AnthropicProfile().supports_streaming is True

    def test_supports_parallel_tool_calls(self):
        assert AnthropicProfile().supports_parallel_tool_calls is True

    def test_provider_options_none(self):
        assert AnthropicProfile().provider_options() is None


class TestAnthropicProfileTools:
    def test_has_core_tools(self):
        names = AnthropicProfile().tool_registry.names()
        assert "read_file" in names
        assert "write_file" in names
        assert "edit_file" in names
        assert "shell" in names
        assert "grep" in names
        assert "glob" in names

    def test_has_edit_file_not_apply_patch(self):
        names = AnthropicProfile().tool_registry.names()
        assert "edit_file" in names
        assert "apply_patch" not in names

    def test_tools_returns_definitions(self):
        defs = AnthropicProfile().tools()
        assert len(defs) == 6
        assert all(isinstance(d, ToolDefinition) for d in defs)


class TestAnthropicProfileCustomTools:
    def test_custom_tool_can_be_registered(self):
        p = AnthropicProfile()
        p.tool_registry.register(RegisteredTool(
            definition=ToolDefinition(name="custom", description="Custom tool"),
            executor=_stub_executor,
        ))
        assert "custom" in p.tool_registry.names()

    def test_custom_tool_overwrites_core(self):
        p = AnthropicProfile()
        p.tool_registry.register(RegisteredTool(
            definition=ToolDefinition(name="shell", description="CUSTOM shell"),
            executor=_stub_executor,
        ))
        assert p.tool_registry.get("shell").definition.description == "CUSTOM shell"


class TestAnthropicProfileSystemPrompt:
    def test_returns_string(self):
        env = StubExecutionEnvironment()
        prompt = AnthropicProfile().build_system_prompt(env)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_includes_tool_guidance(self):
        env = StubExecutionEnvironment()
        prompt = AnthropicProfile().build_system_prompt(env)
        assert "edit_file" in prompt or "old_string" in prompt
