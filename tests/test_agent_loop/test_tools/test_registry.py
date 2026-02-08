"""Tests for tool registry: definitions, registration, and lookup."""

import dataclasses

import pytest

from agent_loop.tools.registry import RegisteredTool, ToolDefinition, ToolRegistry


# --- ToolDefinition ---


class TestToolDefinition:
    def test_construction_and_field_access(self):
        td = ToolDefinition(
            name="read_file",
            description="Read a file",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        )
        assert td.name == "read_file"
        assert td.description == "Read a file"
        assert td.parameters["type"] == "object"

    def test_parameters_default_to_empty_dict(self):
        td = ToolDefinition(name="noop", description="Does nothing")
        assert td.parameters == {}

    def test_frozen_immutability(self):
        td = ToolDefinition(name="x", description="y")
        with pytest.raises(dataclasses.FrozenInstanceError):
            td.name = "z"  # type: ignore[misc]


# --- RegisteredTool ---


def _stub_executor(arguments: dict, env: object) -> str:
    return "ok"


class TestRegisteredTool:
    def test_construction_with_callable_executor(self):
        defn = ToolDefinition(name="test", description="test tool")
        rt = RegisteredTool(definition=defn, executor=_stub_executor)
        assert rt.definition is defn
        assert rt.executor is _stub_executor

    def test_frozen_immutability(self):
        defn = ToolDefinition(name="test", description="test tool")
        rt = RegisteredTool(definition=defn, executor=_stub_executor)
        with pytest.raises(dataclasses.FrozenInstanceError):
            rt.definition = defn  # type: ignore[misc]

    def test_executor_is_callable(self):
        defn = ToolDefinition(name="test", description="test tool")
        rt = RegisteredTool(definition=defn, executor=_stub_executor)
        assert rt.executor({}, None) == "ok"


# --- ToolRegistry: register ---


def _make_tool(name: str) -> RegisteredTool:
    return RegisteredTool(
        definition=ToolDefinition(name=name, description=f"Tool {name}"),
        executor=_stub_executor,
    )


class TestToolRegistryRegister:
    def test_register_single_tool(self):
        reg = ToolRegistry()
        tool = _make_tool("read_file")
        reg.register(tool)
        assert reg.get("read_file") is tool

    def test_register_makes_tool_retrievable_by_name(self):
        reg = ToolRegistry()
        reg.register(_make_tool("shell"))
        assert reg.get("shell") is not None
        assert reg.get("shell").definition.name == "shell"

    def test_register_overwrites_existing_tool_latest_wins(self):
        reg = ToolRegistry()
        old = _make_tool("edit")
        new = RegisteredTool(
            definition=ToolDefinition(name="edit", description="NEW"),
            executor=_stub_executor,
        )
        reg.register(old)
        reg.register(new)
        assert reg.get("edit").definition.description == "NEW"


# --- ToolRegistry: unregister ---


class TestToolRegistryUnregister:
    def test_unregister_removes_tool(self):
        reg = ToolRegistry()
        reg.register(_make_tool("doomed"))
        reg.unregister("doomed")
        assert reg.get("doomed") is None

    def test_unregister_nonexistent_is_noop(self):
        reg = ToolRegistry()
        reg.unregister("ghost")  # should not raise


# --- ToolRegistry: get ---


class TestToolRegistryGet:
    def test_get_returns_registered_tool(self):
        reg = ToolRegistry()
        tool = _make_tool("grep")
        reg.register(tool)
        assert reg.get("grep") is tool

    def test_get_returns_none_for_unknown(self):
        reg = ToolRegistry()
        assert reg.get("nonexistent") is None


# --- ToolRegistry: definitions ---


class TestToolRegistryDefinitions:
    def test_returns_all_tool_definitions(self):
        reg = ToolRegistry()
        reg.register(_make_tool("a"))
        reg.register(_make_tool("b"))
        defs = reg.definitions()
        assert len(defs) == 2
        assert all(isinstance(d, ToolDefinition) for d in defs)

    def test_returns_empty_list_when_no_tools(self):
        reg = ToolRegistry()
        assert reg.definitions() == []

    def test_order_matches_insertion_order(self):
        reg = ToolRegistry()
        reg.register(_make_tool("second"))
        reg.register(_make_tool("first"))
        reg.register(_make_tool("third"))
        names = [d.name for d in reg.definitions()]
        assert names == ["second", "first", "third"]


# --- ToolRegistry: names ---


class TestToolRegistryNames:
    def test_returns_all_registered_names(self):
        reg = ToolRegistry()
        reg.register(_make_tool("read_file"))
        reg.register(_make_tool("shell"))
        assert reg.names() == ["read_file", "shell"]

    def test_returns_empty_list_when_no_tools(self):
        reg = ToolRegistry()
        assert reg.names() == []


# --- ToolRegistry: integration ---


class TestToolRegistryIntegration:
    def test_register_unregister_cycle(self):
        reg = ToolRegistry()
        reg.register(_make_tool("temp"))
        assert reg.get("temp") is not None
        reg.unregister("temp")
        assert reg.get("temp") is None
        assert reg.names() == []

    def test_multiple_tools_registered(self):
        reg = ToolRegistry()
        for name in ["read_file", "write_file", "edit_file", "shell", "grep", "glob"]:
            reg.register(_make_tool(name))
        assert len(reg.names()) == 6
        assert len(reg.definitions()) == 6
        assert reg.get("edit_file") is not None

    def test_overwrite_preserves_insertion_position(self):
        """When overwriting a tool, Python dict preserves original insertion order."""
        reg = ToolRegistry()
        reg.register(_make_tool("a"))
        reg.register(_make_tool("b"))
        reg.register(_make_tool("c"))
        # Overwrite "b"
        reg.register(RegisteredTool(
            definition=ToolDefinition(name="b", description="updated"),
            executor=_stub_executor,
        ))
        assert reg.names() == ["a", "b", "c"]
        assert reg.get("b").definition.description == "updated"
