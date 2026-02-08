"""Tests for the OpenAI provider profile and apply_patch."""

import pytest

from agent_loop.environment.stub import StubExecutionEnvironment
from agent_loop.providers.openai import OpenAIProfile, apply_patch_executor, _apply_v4a_patch
from agent_loop.providers.profile import ProviderProfile
from agent_loop.tools.registry import ToolDefinition


class TestOpenAIProfileProtocol:
    def test_satisfies_provider_profile(self):
        profile: ProviderProfile = OpenAIProfile()
        assert profile.id == "openai"


class TestOpenAIProfileProperties:
    def test_id(self):
        assert OpenAIProfile().id == "openai"

    def test_default_model(self):
        assert OpenAIProfile().model == "o4-mini"

    def test_custom_model(self):
        assert OpenAIProfile(model="gpt-5.2-codex").model == "gpt-5.2-codex"

    def test_context_window(self):
        assert OpenAIProfile().context_window_size == 128_000


class TestOpenAIProfileTools:
    def test_has_apply_patch_not_edit_file(self):
        names = OpenAIProfile().tool_registry.names()
        assert "apply_patch" in names
        assert "edit_file" not in names

    def test_has_core_tools_minus_edit_file(self):
        names = OpenAIProfile().tool_registry.names()
        assert "read_file" in names
        assert "write_file" in names
        assert "shell" in names
        assert "grep" in names
        assert "glob" in names

    def test_tools_returns_definitions(self):
        defs = OpenAIProfile().tools()
        assert all(isinstance(d, ToolDefinition) for d in defs)


# --- apply_patch v4a format ---


class TestApplyPatchAddFile:
    def test_creates_new_file(self):
        env = StubExecutionEnvironment()
        patch = (
            "*** Begin Patch\n"
            "*** Add File: src/hello.py\n"
            "+def greet():\n"
            '+    return "Hello!"\n'
            "*** End Patch\n"
        )
        result = _apply_v4a_patch(patch, env)
        assert "Created src/hello.py" in result
        content = env.read_file("src/hello.py")
        assert "def greet():" in content


class TestApplyPatchDeleteFile:
    def test_deletes_file(self):
        env = StubExecutionEnvironment(files={"old.py": "content"})
        patch = (
            "*** Begin Patch\n"
            "*** Delete File: old.py\n"
            "*** End Patch\n"
        )
        result = _apply_v4a_patch(patch, env)
        assert "Deleted old.py" in result


class TestApplyPatchUpdateFile:
    def test_updates_existing_file(self):
        env = StubExecutionEnvironment(files={"main.py": "def main():\n    print('Hello')\n    return 0\n"})
        patch = (
            "*** Begin Patch\n"
            "*** Update File: main.py\n"
            "@@ def main():\n"
            "     print('Hello')\n"
            "-    return 0\n"
            "+    print('World')\n"
            "+    return 1\n"
            "*** End Patch\n"
        )
        result = _apply_v4a_patch(patch, env)
        assert "Updated main.py" in result
        content = env.read_file("main.py")
        assert "print('World')" in content
        assert "return 1" in content

    def test_update_with_move(self):
        env = StubExecutionEnvironment(files={"old.py": "import os\nimport sys\n"})
        patch = (
            "*** Begin Patch\n"
            "*** Update File: old.py\n"
            "*** Move to: new.py\n"
            "@@ import os\n"
            " import os\n"
            "-import sys\n"
            "+import pathlib\n"
            "*** End Patch\n"
        )
        result = _apply_v4a_patch(patch, env)
        assert "new.py" in result
        content = env.read_file("new.py")
        assert "import pathlib" in content


class TestApplyPatchErrors:
    def test_invalid_patch_start(self):
        env = StubExecutionEnvironment()
        with pytest.raises(ValueError, match="Begin Patch"):
            _apply_v4a_patch("not a patch", env)

    def test_hunk_not_found(self):
        env = StubExecutionEnvironment(files={"f.py": "unrelated content\n"})
        patch = (
            "*** Begin Patch\n"
            "*** Update File: f.py\n"
            "@@ def nonexistent():\n"
            " def nonexistent():\n"
            "-    old_line\n"
            "+    new_line\n"
            "*** End Patch\n"
        )
        with pytest.raises(ValueError, match="Could not find"):
            _apply_v4a_patch(patch, env)


class TestApplyPatchMultiHunk:
    def test_multiple_hunks_in_one_file(self):
        env = StubExecutionEnvironment(files={
            "config.py": "TIMEOUT = 30\nDEBUG = False\nVERSION = 1\n"
        })
        patch = (
            "*** Begin Patch\n"
            "*** Update File: config.py\n"
            "@@ TIMEOUT\n"
            "-TIMEOUT = 30\n"
            "+TIMEOUT = 60\n"
            "@@ DEBUG\n"
            "-DEBUG = False\n"
            "+DEBUG = True\n"
            "*** End Patch\n"
        )
        result = _apply_v4a_patch(patch, env)
        content = env.read_file("config.py")
        assert "TIMEOUT = 60" in content
        assert "DEBUG = True" in content
        assert "VERSION = 1" in content
