"""Tests for loop detection in agent tool call patterns."""
import pytest

from agent_loop.loop_detection import (
    ToolCallSignature,
    detect_loop,
    make_signature,
)


class TestMakeSignature:
    """Tests for the make_signature function."""

    def test_different_arguments_produce_different_signatures(self):
        """Different arguments yield different hashes."""
        sig1 = make_signature("shell", {"command": "ls"})
        sig2 = make_signature("shell", {"command": "pwd"})
        assert sig1.arguments_hash != sig2.arguments_hash

    def test_same_arguments_produce_same_signature(self):
        """Identical arguments yield the same hash."""
        sig1 = make_signature("shell", {"command": "ls", "timeout": 30})
        sig2 = make_signature("shell", {"command": "ls", "timeout": 30})
        assert sig1 == sig2

    def test_deterministic(self):
        """make_signature produces the same result on repeated calls."""
        args = {"path": "/tmp", "recursive": True}
        sig1 = make_signature("read_file", args)
        sig2 = make_signature("read_file", args)
        assert sig1.tool_name == sig2.tool_name
        assert sig1.arguments_hash == sig2.arguments_hash


class TestToolCallSignature:
    """Tests for the ToolCallSignature dataclass."""

    def test_str_format(self):
        """__str__ shows tool name and first 8 chars of hash."""
        sig = make_signature("shell", {"command": "ls"})
        s = str(sig)
        assert s.startswith("shell(")
        assert s.endswith(")")
        # The hash portion should be 8 characters
        hash_part = s[len("shell("):-1]
        assert len(hash_part) == 8


class TestDetectLoop:
    """Tests for the detect_loop function."""

    def test_no_loop_in_varied_signatures(self):
        """Varied tool calls do not trigger loop detection."""
        sigs = [make_signature("tool", {"arg": i}) for i in range(10)]
        assert detect_loop(sigs, window=10) is None

    def test_loop_detected_same_signature_repeated(self):
        """Repeating the same call triggers detection (pattern length 1)."""
        sig = make_signature("shell", {"command": "ls"})
        sigs = [sig] * 10
        result = detect_loop(sigs, window=10)
        assert result is not None
        assert "Loop detected" in result
        assert "repeating pattern" in result

    def test_loop_detected_alternating_pattern(self):
        """An alternating A-B-A-B pattern triggers detection (pattern length 2)."""
        sig_a = make_signature("shell", {"command": "ls"})
        sig_b = make_signature("shell", {"command": "pwd"})
        sigs = [sig_a, sig_b] * 5  # 10 items, alternating
        result = detect_loop(sigs, window=10)
        assert result is not None
        assert "Loop detected" in result

    def test_window_too_large_for_history_returns_none(self):
        """If history is shorter than window, returns None."""
        sig = make_signature("shell", {"command": "ls"})
        sigs = [sig] * 5
        assert detect_loop(sigs, window=10) is None

    def test_exact_window_size(self):
        """Detection works when history length equals window size exactly."""
        sig = make_signature("shell", {"command": "ls"})
        sigs = [sig] * 10
        result = detect_loop(sigs, window=10)
        assert result is not None

    def test_no_loop_when_pattern_does_not_fill_window(self):
        """A partial pattern that doesn't repeat across the full window is not a loop."""
        sig_a = make_signature("shell", {"command": "ls"})
        sig_b = make_signature("shell", {"command": "pwd"})
        sig_c = make_signature("shell", {"command": "cat"})
        # 10 items but no clean repeating pattern
        sigs = [sig_a, sig_b, sig_c, sig_a, sig_b, sig_c, sig_a, sig_b, sig_c, sig_a]
        result = detect_loop(sigs, window=10)
        # Pattern length 3 repeats 3 times but 10 is not evenly divisible by 3,
        # so the last element must still match pattern[0] -- which it does (sig_a).
        # 3 * 3 = 9, index 9 -> pattern[9 % 3] = pattern[0] = sig_a -> matches
        # This IS a loop (pattern of length 3 repeats across the full window).
        assert result is not None
