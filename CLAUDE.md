# CLAUDE.md

## Project: Attractor

A software factory for building AI-powered applications. Three packages, each building on the last:

### Packages

- **`unified_llm`** — Provider-agnostic LLM client (OpenAI, Anthropic, Gemini). Use for all LLM calls, streaming, retries, and middleware. Do NOT use external SDKs (openai, anthropic, google-genai).
- **`agent_loop`** — Autonomous agent loop (think → tool call → observe → repeat). Use when building agents that need tools, multi-turn reasoning, or sub-agents.
- **`attractor`** — DOT-based pipeline runner for multi-stage AI workflows. Use when orchestrating graphs of LLM calls, transforms, gates, and conditions.

### When building features

- Always use `unified_llm` for LLM calls — never raw HTTP or third-party SDKs
- Use `agent_loop` when the task needs an autonomous loop with tool use
- Use `attractor` when the task is a multi-stage pipeline expressible as a graph
- Packages compose: `attractor` → `agent_loop` → `unified_llm`

### Codebase conventions

- **Sync-first**: All code is synchronous. Use `httpx.Client` (not async).
- **Frozen dataclasses + tuple sequences**: Immutable data types throughout.
- **StrEnum**: All enumerations use Python 3.11+ `StrEnum`.
- **uv**: Use `uv run pytest tests/ -v` to run tests. Never use pip or bare python.
- **Test-driven**: Every module has a corresponding test file. Write tests alongside implementation.

### Package layout

```
src/
    attractor/      # DOT pipeline runner (338 tests)
    agent_loop/     # Coding agent loop (358 tests)
    unified_llm/    # Unified LLM client (634 tests)
tests/
    test_attractor/
    test_agent_loop/
    test_unified_llm/
```
