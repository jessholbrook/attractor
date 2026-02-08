from __future__ import annotations

import pytest

from wolverine.config import WolverineConfig


class TestWolverineConfig:
    def test_default_values(self) -> None:
        cfg = WolverineConfig()
        assert cfg.db_path == "wolverine.db"
        assert cfg.working_dir == "."
        assert cfg.llm_model == "claude-sonnet-4-20250514"
        assert cfg.llm_provider == "anthropic"
        assert cfg.max_heal_turns == 50
        assert cfg.test_command == ""
        assert cfg.log_dir == "wolverine-runs"
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 5000

    def test_custom_values(self) -> None:
        cfg = WolverineConfig(
            db_path="/tmp/test.db",
            working_dir="/projects/myapp",
            llm_model="gpt-4o",
            llm_provider="openai",
            max_heal_turns=10,
            test_command="uv run pytest",
            log_dir="/tmp/runs",
            host="0.0.0.0",
            port=8080,
        )
        assert cfg.db_path == "/tmp/test.db"
        assert cfg.working_dir == "/projects/myapp"
        assert cfg.llm_model == "gpt-4o"
        assert cfg.llm_provider == "openai"
        assert cfg.max_heal_turns == 10
        assert cfg.test_command == "uv run pytest"
        assert cfg.log_dir == "/tmp/runs"
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 8080

    def test_frozen_immutability(self) -> None:
        cfg = WolverineConfig()
        with pytest.raises(AttributeError):
            cfg.db_path = "other.db"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            cfg.port = 9999  # type: ignore[misc]

    def test_equality(self) -> None:
        assert WolverineConfig() == WolverineConfig()
        assert WolverineConfig(port=8080) != WolverineConfig(port=9090)

    def test_hashable(self) -> None:
        cfg = WolverineConfig()
        assert isinstance(hash(cfg), int)
        s = {cfg}
        assert cfg in s
