"""System prompt construction with layered priority."""

from __future__ import annotations

import datetime
import subprocess
from pathlib import Path
from typing import Any

from agent_loop.tools.registry import ToolDefinition

PROMPT_BUDGET_BYTES = 32 * 1024  # 32KB


# --- Environment context ---


def build_environment_context(
    env: Any,
    model: str = "",
    git_context: dict[str, Any] | None = None,
) -> str:
    """Generate the <environment> block from execution environment metadata.

    Includes working directory, git info, platform, date, and model.
    """
    git = git_context or get_git_context(env.working_directory)

    lines = [
        "<environment>",
        f"Working directory: {env.working_directory}",
        f"Is git repository: {git.get('is_repo', False)}",
    ]
    if git.get("is_repo"):
        lines.append(f"Git branch: {git.get('branch', 'unknown')}")
    lines.extend([
        f"Platform: {env.platform}",
        f"OS version: {env.os_version}",
        f"Today's date: {datetime.date.today().isoformat()}",
    ])
    if model:
        lines.append(f"Model: {model}")
    lines.append("</environment>")
    return "\n".join(lines)


# --- Git context ---


def get_git_context(working_dir: str) -> dict[str, Any]:
    """Snapshot git state: branch, status summary, recent commits.

    Returns a dict with: is_repo, branch, modified_count, untracked_count, recent_commits.
    If not a git repo, returns {is_repo: False}.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, cwd=working_dir, timeout=5,
        )
        if result.returncode != 0:
            return {"is_repo": False}
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {"is_repo": False}

    context: dict[str, Any] = {"is_repo": True}

    # Branch
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, cwd=working_dir, timeout=5,
        )
        context["branch"] = result.stdout.strip() if result.returncode == 0 else "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        context["branch"] = "unknown"

    # Status summary
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, cwd=working_dir, timeout=5,
        )
        if result.returncode == 0:
            lines = [l for l in result.stdout.splitlines() if l.strip()]
            context["modified_count"] = sum(1 for l in lines if not l.startswith("??"))
            context["untracked_count"] = sum(1 for l in lines if l.startswith("??"))
        else:
            context["modified_count"] = 0
            context["untracked_count"] = 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        context["modified_count"] = 0
        context["untracked_count"] = 0

    # Recent commits
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "-10"],
            capture_output=True, text=True, cwd=working_dir, timeout=5,
        )
        context["recent_commits"] = (
            result.stdout.strip().splitlines() if result.returncode == 0 else []
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        context["recent_commits"] = []

    return context


# --- Project document discovery ---

# Provider-specific file mappings
PROVIDER_DOC_FILES: dict[str, list[str]] = {
    "anthropic": ["AGENTS.md", "CLAUDE.md"],
    "openai": ["AGENTS.md", ".codex/instructions.md"],
    "gemini": ["AGENTS.md", "GEMINI.md"],
}


def discover_project_docs(
    working_dir: str,
    provider_id: str = "anthropic",
) -> list[str]:
    """Find and load project instruction files.

    Walks from git root (or working dir) upward to discover recognized
    instruction files. Root-level files loaded first, subdirectory files
    appended (deeper = higher precedence). AGENTS.md always loaded.

    Total budget: 32KB. Truncates with marker if exceeded.
    """
    allowed_names = set(PROVIDER_DOC_FILES.get(provider_id, ["AGENTS.md"]))

    # Try to find git root
    root = _find_git_root(working_dir) or working_dir
    root_path = Path(root)
    work_path = Path(working_dir)

    # Walk from root to working dir
    docs: list[str] = []
    search_dirs = [root_path]
    if work_path != root_path:
        # Add intermediate directories
        try:
            relative = work_path.relative_to(root_path)
            current = root_path
            for part in relative.parts:
                current = current / part
                if current != root_path:
                    search_dirs.append(current)
        except ValueError:
            pass

    total_bytes = 0
    for directory in search_dirs:
        for name in sorted(allowed_names):
            # Handle nested paths like .codex/instructions.md
            candidate = directory / name
            if candidate.is_file():
                try:
                    content = candidate.read_text(encoding="utf-8", errors="replace")
                    if total_bytes + len(content.encode("utf-8")) > PROMPT_BUDGET_BYTES:
                        remaining = PROMPT_BUDGET_BYTES - total_bytes
                        if remaining > 0:
                            docs.append(content[:remaining] + "\n[Project instructions truncated at 32KB]")
                        return docs
                    docs.append(content)
                    total_bytes += len(content.encode("utf-8"))
                except OSError:
                    continue

    return docs


def _find_git_root(working_dir: str) -> str | None:
    """Find the git repository root, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, cwd=working_dir, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


# --- Tool descriptions ---


def format_tool_descriptions(tool_definitions: list[ToolDefinition]) -> str:
    """Format tool definitions as a text block for the system prompt."""
    if not tool_definitions:
        return ""
    parts = ["## Available Tools\n"]
    for td in tool_definitions:
        parts.append(f"### {td.name}\n{td.description}\n")
    return "\n".join(parts)


# --- System prompt assembly ---


def build_system_prompt(
    base_instructions: str,
    environment: Any,
    tool_definitions: list[ToolDefinition] | None = None,
    model: str = "",
    provider_id: str = "anthropic",
    user_instructions: str | None = None,
    git_context: dict[str, Any] | None = None,
) -> str:
    """Assemble the full system prompt with layered priority.

    Layer order (lowest to highest priority):
    1. base_instructions (from ProviderProfile)
    2. environment context block
    3. tool descriptions
    4. project docs (AGENTS.md, CLAUDE.md, etc.)
    5. user_instructions (highest priority, appended last)
    """
    layers: list[str] = []

    # Layer 1: Base instructions
    if base_instructions:
        layers.append(base_instructions)

    # Layer 2: Environment context
    env_context = build_environment_context(environment, model=model, git_context=git_context)
    layers.append(env_context)

    # Layer 3: Tool descriptions
    if tool_definitions:
        tool_text = format_tool_descriptions(tool_definitions)
        if tool_text:
            layers.append(tool_text)

    # Layer 4: Project docs
    project_docs = discover_project_docs(environment.working_directory, provider_id)
    for doc in project_docs:
        layers.append(doc)

    # Layer 5: User instructions (highest priority)
    if user_instructions:
        layers.append(user_instructions)

    return "\n\n".join(layers)
