"""Tests for execution environment types."""

import dataclasses

import pytest

from agent_loop.environment.types import DirEntry, ExecResult, GrepOptions


class TestExecResult:
    def test_defaults(self):
        r = ExecResult()
        assert r.stdout == ""
        assert r.stderr == ""
        assert r.exit_code == 0
        assert r.timed_out is False
        assert r.duration_ms == 0

    def test_construction_with_values(self):
        r = ExecResult(stdout="hello", stderr="warn", exit_code=1, timed_out=True, duration_ms=500)
        assert r.stdout == "hello"
        assert r.exit_code == 1
        assert r.timed_out is True
        assert r.duration_ms == 500

    def test_frozen_immutability(self):
        r = ExecResult()
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.exit_code = 1  # type: ignore[misc]


class TestDirEntry:
    def test_file_entry(self):
        e = DirEntry(name="main.py", is_dir=False, size=1024)
        assert e.name == "main.py"
        assert e.is_dir is False
        assert e.size == 1024

    def test_directory_entry(self):
        e = DirEntry(name="src", is_dir=True)
        assert e.is_dir is True
        assert e.size is None

    def test_size_defaults_to_none(self):
        e = DirEntry(name="x", is_dir=False)
        assert e.size is None

    def test_frozen_immutability(self):
        e = DirEntry(name="x", is_dir=False)
        with pytest.raises(dataclasses.FrozenInstanceError):
            e.name = "y"  # type: ignore[misc]


class TestGrepOptions:
    def test_defaults(self):
        o = GrepOptions()
        assert o.case_insensitive is False
        assert o.glob_filter is None
        assert o.max_results == 100

    def test_custom_options(self):
        o = GrepOptions(case_insensitive=True, glob_filter="*.py", max_results=50)
        assert o.case_insensitive is True
        assert o.glob_filter == "*.py"
        assert o.max_results == 50

    def test_frozen_immutability(self):
        o = GrepOptions()
        with pytest.raises(dataclasses.FrozenInstanceError):
            o.max_results = 200  # type: ignore[misc]
