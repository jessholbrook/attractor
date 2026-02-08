"""Tests for the model stylesheet parser."""

import pytest

from attractor.stylesheet import parse_stylesheet, Stylesheet, StyleRule, Selector


# ---------------------------------------------------------------------------
# Selector parsing
# ---------------------------------------------------------------------------


class TestUniversalSelector:
    def test_parse_universal(self):
        ss = parse_stylesheet("* { llm_model: gpt-4; }")
        assert len(ss.rules) == 1
        rule = ss.rules[0]
        assert rule.selector == Selector(kind="universal", value="*", specificity=0)
        assert rule.properties == {"llm_model": "gpt-4"}


class TestClassSelector:
    def test_parse_class(self):
        ss = parse_stylesheet(".fast { llm_model: gpt-4o-mini; }")
        assert len(ss.rules) == 1
        rule = ss.rules[0]
        assert rule.selector == Selector(kind="class", value="fast", specificity=1)
        assert rule.properties == {"llm_model": "gpt-4o-mini"}


class TestIdSelector:
    def test_parse_id(self):
        ss = parse_stylesheet("#review { reasoning_effort: high; }")
        assert len(ss.rules) == 1
        rule = ss.rules[0]
        assert rule.selector == Selector(kind="id", value="review", specificity=2)
        assert rule.properties == {"reasoning_effort": "high"}


# ---------------------------------------------------------------------------
# Multiple rules / properties
# ---------------------------------------------------------------------------


class TestMultipleRules:
    def test_two_rules(self):
        source = """
        * { llm_model: gpt-4; }
        .code { llm_model: claude-opus-4-6; }
        """
        ss = parse_stylesheet(source)
        assert len(ss.rules) == 2
        assert ss.rules[0].selector.kind == "universal"
        assert ss.rules[1].selector.kind == "class"

    def test_three_rules(self):
        source = """
        * { llm_model: claude-sonnet-4-5; llm_provider: anthropic; }
        .code { llm_model: claude-opus-4-6; }
        #critical_review { reasoning_effort: high; }
        """
        ss = parse_stylesheet(source)
        assert len(ss.rules) == 3


class TestMultipleProperties:
    def test_multiple_properties_in_one_rule(self):
        source = "* { llm_model: claude-sonnet-4-5; llm_provider: anthropic; reasoning_effort: medium; }"
        ss = parse_stylesheet(source)
        assert len(ss.rules) == 1
        props = ss.rules[0].properties
        assert props["llm_model"] == "claude-sonnet-4-5"
        assert props["llm_provider"] == "anthropic"
        assert props["reasoning_effort"] == "medium"


# ---------------------------------------------------------------------------
# Specificity ordering
# ---------------------------------------------------------------------------


class TestSpecificity:
    def test_universal_lowest(self):
        sel = Selector(kind="universal", value="*", specificity=0)
        assert sel.specificity == 0

    def test_class_middle(self):
        sel = Selector(kind="class", value="fast", specificity=1)
        assert sel.specificity == 1

    def test_id_highest(self):
        sel = Selector(kind="id", value="review", specificity=2)
        assert sel.specificity == 2

    def test_ordering(self):
        source = """
        #review { reasoning_effort: high; }
        * { llm_model: gpt-4; }
        .fast { llm_model: gpt-4o-mini; }
        """
        ss = parse_stylesheet(source)
        sorted_rules = sorted(ss.rules, key=lambda r: r.selector.specificity)
        assert sorted_rules[0].selector.kind == "universal"
        assert sorted_rules[1].selector.kind == "class"
        assert sorted_rules[2].selector.kind == "id"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEmptyStylesheet:
    def test_empty_string(self):
        ss = parse_stylesheet("")
        assert ss.rules == []

    def test_whitespace_only(self):
        ss = parse_stylesheet("   \n\t  ")
        assert ss.rules == []


class TestStylesheetDataclass:
    def test_stylesheet_is_frozen(self):
        ss = parse_stylesheet("* { llm_model: gpt-4; }")
        with pytest.raises(AttributeError):
            ss.rules = []  # type: ignore[misc]

    def test_selector_is_frozen(self):
        sel = Selector(kind="universal", value="*", specificity=0)
        with pytest.raises(AttributeError):
            sel.kind = "class"  # type: ignore[misc]
