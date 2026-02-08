from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WolverineConfig:
    db_path: str = "wolverine.db"
    working_dir: str = "."
    llm_model: str = "claude-sonnet-4-20250514"
    llm_provider: str = "anthropic"
    max_heal_turns: int = 50
    test_command: str = ""  # e.g., "uv run pytest"
    log_dir: str = "wolverine-runs"
    host: str = "127.0.0.1"
    port: int = 5000
