"""Tests for tool output truncation."""
import pytest

from agent_loop.truncation import (
    DEFAULT_LINE_LIMITS,
    DEFAULT_TOOL_LIMITS,
    TruncationConfig,
    TruncationMode,
    get_tool_config,
    truncate_lines,
    truncate_output,
    truncate_tool_output,
)


class TestTruncateOutput:
    """Tests for the truncate_output function."""

    def test_short_output_not_truncated(self):
        """Output under the limit is returned unchanged."""
        config = TruncationConfig(max_chars=100)
        output = "hello world"
        assert truncate_output(output, config) == output

    def test_output_exactly_at_limit_not_truncated(self):
        """Output exactly at the limit is returned unchanged."""
        config = TruncationConfig(max_chars=10)
        output = "a" * 10
        assert truncate_output(output, config) == output

    def test_output_over_limit_head_tail_truncated(self):
        """Output over max_chars is truncated in HEAD_TAIL mode."""
        config = TruncationConfig(max_chars=20, mode=TruncationMode.HEAD_TAIL)
        output = "a" * 100
        result = truncate_output(output, config)
        # The content portions should total max_chars (20), even though
        # the marker text makes the overall string longer.
        assert result != output
        assert "[WARNING: Output truncated." in result

    def test_head_tail_keeps_first_and_last_half(self):
        """HEAD_TAIL mode keeps first half and last half of chars."""
        config = TruncationConfig(max_chars=20, mode=TruncationMode.HEAD_TAIL)
        output = "A" * 10 + "B" * 10 + "C" * 10  # 30 chars total
        result = truncate_output(output, config)
        # First 10 chars (half of 20) should be 'A' * 10
        assert result.startswith("A" * 10)
        # Last 10 chars (half of 20) should be 'C' * 10
        assert result.endswith("C" * 10)

    def test_head_tail_marker_contains_char_count(self):
        """HEAD_TAIL marker includes the number of removed characters."""
        config = TruncationConfig(max_chars=20, mode=TruncationMode.HEAD_TAIL)
        output = "x" * 50
        result = truncate_output(output, config)
        # 50 - 20 = 30 characters removed
        assert "30 characters removed from the middle" in result

    def test_tail_mode_keeps_last_chars(self):
        """TAIL mode keeps only the last max_chars characters."""
        config = TruncationConfig(max_chars=10, mode=TruncationMode.TAIL)
        output = "A" * 20 + "B" * 10  # 30 chars total
        result = truncate_output(output, config)
        # The tail portion should be the last 10 chars = "B" * 10
        assert result.endswith("B" * 10)

    def test_tail_mode_marker_contains_removed_count(self):
        """TAIL marker includes the number of removed characters."""
        config = TruncationConfig(max_chars=10, mode=TruncationMode.TAIL)
        output = "x" * 50
        result = truncate_output(output, config)
        # 50 - 10 = 40 characters removed
        assert "First 40 characters removed" in result

    def test_empty_string_not_truncated(self):
        """Empty string is returned unchanged."""
        config = TruncationConfig(max_chars=100)
        assert truncate_output("", config) == ""

    def test_custom_config_respected(self):
        """Custom TruncationConfig values are used."""
        config = TruncationConfig(max_chars=5, mode=TruncationMode.TAIL)
        output = "abcdefghij"  # 10 chars
        result = truncate_output(output, config)
        assert result.endswith("fghij")
        assert "[WARNING:" in result

    def test_default_config_when_none(self):
        """When config is None, default TruncationConfig (30k chars) is used."""
        short_output = "hello"
        assert truncate_output(short_output) == short_output


class TestGetToolConfig:
    """Tests for the get_tool_config function."""

    def test_returns_default_for_known_tool(self):
        """Known tools return their configured defaults."""
        config = get_tool_config("read_file")
        assert config.max_chars == 50_000
        assert config.mode == TruncationMode.HEAD_TAIL

        config = get_tool_config("grep")
        assert config.max_chars == 20_000
        assert config.mode == TruncationMode.TAIL

    def test_returns_generic_default_for_unknown_tool(self):
        """Unknown tools get the generic TruncationConfig default."""
        config = get_tool_config("unknown_tool")
        assert config.max_chars == 30_000
        assert config.mode == TruncationMode.HEAD_TAIL

    def test_respects_overrides(self):
        """Overrides take precedence over defaults."""
        custom = TruncationConfig(max_chars=999, mode=TruncationMode.TAIL)
        overrides = {"shell": custom}
        config = get_tool_config("shell", overrides=overrides)
        assert config.max_chars == 999
        assert config.mode == TruncationMode.TAIL


# --- Line-based truncation ---


class TestTruncateLines:
    def test_short_output_not_truncated(self):
        output = "line1\nline2\nline3"
        assert truncate_lines(output, max_lines=10) == output

    def test_output_at_limit_not_truncated(self):
        output = "\n".join(f"line{i}" for i in range(10))
        assert truncate_lines(output, max_lines=10) == output

    def test_output_over_limit_has_omission_marker(self):
        output = "\n".join(f"line{i}" for i in range(20))
        result = truncate_lines(output, max_lines=10)
        assert "[..." in result
        assert "lines omitted" in result

    def test_keeps_first_and_last_half(self):
        lines = [f"line{i}" for i in range(20)]
        output = "\n".join(lines)
        result = truncate_lines(output, max_lines=10)
        # First 5 lines preserved
        assert "line0" in result
        assert "line4" in result
        # Last 5 lines preserved
        assert "line15" in result
        assert "line19" in result

    def test_omission_marker_shows_correct_count(self):
        output = "\n".join(f"line{i}" for i in range(30))
        result = truncate_lines(output, max_lines=10)
        # 30 lines - 5 head - 5 tail = 20 omitted
        assert "20 lines omitted" in result

    def test_single_line_not_truncated(self):
        assert truncate_lines("single", max_lines=5) == "single"

    def test_empty_string_not_truncated(self):
        assert truncate_lines("", max_lines=5) == ""


class TestDefaultLineLimits:
    def test_shell_is_256(self):
        assert DEFAULT_LINE_LIMITS["shell"] == 256

    def test_grep_is_200(self):
        assert DEFAULT_LINE_LIMITS["grep"] == 200

    def test_glob_is_500(self):
        assert DEFAULT_LINE_LIMITS["glob"] == 500

    def test_read_file_is_none(self):
        assert DEFAULT_LINE_LIMITS["read_file"] is None


# --- Two-stage pipeline ---


class TestTruncateToolOutput:
    def test_char_truncation_runs_first(self):
        """Character truncation handles huge single-line output."""
        output = "x" * 100_000  # 100k chars, single line
        result = truncate_tool_output(output, "shell")
        # shell default is 30k chars
        assert len(result) < 100_000
        assert "[WARNING:" in result

    def test_line_truncation_runs_second(self):
        """Line truncation applied after char truncation."""
        # 500 short lines, well under char limit
        output = "\n".join(f"line{i}" for i in range(500))
        result = truncate_tool_output(output, "shell")
        # shell has 256 line limit
        assert "lines omitted" in result

    def test_both_stages_applied_to_large_output(self):
        """Both char and line truncation apply when needed."""
        # 1000 lines of moderate length
        output = "\n".join("a" * 100 for _ in range(1000))
        result = truncate_tool_output(output, "shell")
        # Should have char truncation marker and line truncation marker
        assert "[WARNING:" in result or "lines omitted" in result

    def test_no_line_limit_skips_line_truncation(self):
        """Tools with None line limit only get char truncation."""
        output = "\n".join(f"line{i}" for i in range(500))
        result = truncate_tool_output(output, "read_file")
        # read_file has no line limit, and 500 short lines under 50k char limit
        assert "lines omitted" not in result

    def test_char_overrides_respected(self):
        custom = {"shell": TruncationConfig(max_chars=50, mode=TruncationMode.HEAD_TAIL)}
        output = "x" * 200
        result = truncate_tool_output(output, "shell", char_overrides=custom)
        assert "[WARNING:" in result

    def test_line_overrides_respected(self):
        output = "\n".join(f"line{i}" for i in range(100))
        result = truncate_tool_output(output, "shell", line_overrides={"shell": 10})
        assert "lines omitted" in result

    def test_unknown_tool_uses_generic_defaults(self):
        output = "short"
        result = truncate_tool_output(output, "custom_tool")
        assert result == "short"
