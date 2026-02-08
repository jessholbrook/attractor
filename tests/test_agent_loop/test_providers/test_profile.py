"""Tests for provider profile protocol and stub."""

from agent_loop.providers.profile import ProviderProfile, StubProfile
from agent_loop.tools.registry import RegisteredTool, ToolDefinition, ToolRegistry


def _stub_executor(args: dict, env: object) -> str:
    return "ok"


class TestProviderProfileProtocol:
    def test_stub_profile_satisfies_protocol(self):
        profile: ProviderProfile = StubProfile()
        assert profile.id == "stub"


class TestStubProfileDefaults:
    def test_default_id(self):
        assert StubProfile().id == "stub"

    def test_default_model(self):
        assert StubProfile().model == "stub-model"

    def test_default_context_window(self):
        assert StubProfile().context_window_size == 128_000

    def test_default_supports_parallel(self):
        assert StubProfile().supports_parallel_tool_calls is True

    def test_default_supports_reasoning_false(self):
        assert StubProfile().supports_reasoning is False

    def test_default_supports_streaming_false(self):
        assert StubProfile().supports_streaming is False

    def test_default_provider_options_none(self):
        assert StubProfile().provider_options() is None

    def test_default_system_prompt(self):
        assert StubProfile().build_system_prompt(None) == "You are a helpful assistant."


class TestStubProfileCustom:
    def test_custom_id_and_model(self):
        p = StubProfile(profile_id="anthropic", model="claude-opus-4-6")
        assert p.id == "anthropic"
        assert p.model == "claude-opus-4-6"

    def test_custom_context_window(self):
        p = StubProfile(context_window=200_000)
        assert p.context_window_size == 200_000

    def test_custom_system_prompt(self):
        p = StubProfile(system_prompt="Custom prompt.")
        assert p.build_system_prompt(None) == "Custom prompt."


class TestStubProfileToolRegistry:
    def test_tools_delegates_to_registry(self):
        reg = ToolRegistry()
        reg.register(RegisteredTool(
            definition=ToolDefinition(name="read_file", description="Read"),
            executor=_stub_executor,
        ))
        p = StubProfile(registry=reg)
        defs = p.tools()
        assert len(defs) == 1
        assert defs[0].name == "read_file"

    def test_empty_registry_returns_empty_tools(self):
        p = StubProfile()
        assert p.tools() == []

    def test_tool_registry_property(self):
        reg = ToolRegistry()
        p = StubProfile(registry=reg)
        assert p.tool_registry is reg
