"""Tests for AnnotatingFormatter and classify_variable."""

import pytest
from project_scripts.string_formatter import (
    AnnotatedSegment,
    AnnotatingFormatter,
    classify_variable,
)


class TestClassifyVariable:
    """Test classify_variable maps prefixes to correct categories."""

    def test_pov_variable(self):
        assert classify_variable("pov_1") == ("pov", "pov_1")

    def test_pov_high_number(self):
        assert classify_variable("pov_10") == ("pov", "pov_10")

    def test_switch_variable(self):
        assert classify_variable("switch_1") == ("switch", "switch_1")

    def test_filler_variable(self):
        assert classify_variable("filler_1") == ("filler", "filler_1")

    def test_coinflip_variable(self):
        assert classify_variable("coinflip_1") == ("coinflip", "coinflip_1")

    def test_otherside_variable(self):
        assert classify_variable("otherside_1") == ("coinflip", "otherside_1")

    def test_item_variable(self):
        assert classify_variable("item_1") == ("item", "item_1")

    def test_activity_variable(self):
        assert classify_variable("activity_1") == ("activity", "activity_1")

    def test_smallnumber_variable(self):
        assert classify_variable("smallnumber_1") == ("number", "smallnumber_1")

    def test_bignumber_variable(self):
        assert classify_variable("bignumber_1") == ("number", "bignumber_1")

    def test_regular_variable(self):
        assert classify_variable("name_1") == ("variable", "name_1")

    def test_unknown_prefix(self):
        assert classify_variable("color_2") == ("variable", "color_2")

    def test_no_underscore(self):
        assert classify_variable("standalone") == ("variable", "standalone")

    def test_pseudo_variable(self):
        assert classify_variable("pseudo_1") == ("variable", "pseudo_1")


class TestAnnotatingFormatter:
    """Test AnnotatingFormatter.annotated_format produces correct segments."""

    def test_literal_only(self):
        fmt = AnnotatingFormatter()
        segments = fmt.annotated_format("Hello world", {})
        assert len(segments) == 1
        assert segments[0].text == "Hello world"
        assert segments[0].source_type == "template"
        assert segments[0].source_key is None

    def test_single_variable(self):
        fmt = AnnotatingFormatter()
        segments = fmt.annotated_format("{vs[name_1]}", {"name_1": "Alice"})
        assert len(segments) == 1
        assert segments[0].text == "Alice"
        assert segments[0].source_type == "variable"
        assert segments[0].source_key == "name_1"

    def test_mixed_template_and_variable(self):
        fmt = AnnotatingFormatter()
        segments = fmt.annotated_format(
            "Hello {vs[name_1]}, welcome!",
            {"name_1": "Bob"},
        )
        assert len(segments) == 3
        assert segments[0].text == "Hello "
        assert segments[0].source_type == "template"
        assert segments[1].text == "Bob"
        assert segments[1].source_type == "variable"
        assert segments[1].source_key == "name_1"
        assert segments[2].text == ", welcome!"
        assert segments[2].source_type == "template"

    def test_pronoun_format_spec(self):
        fmt = AnnotatingFormatter()
        segments = fmt.annotated_format(
            "{vs[pov_1]:p1} went home",
            {"pov_1": "cat"},
        )
        assert segments[0].text == "it"
        assert segments[0].source_type == "pov"
        assert segments[0].source_key == "pov_1"
        assert segments[1].text == " went home"

    def test_pronoun_plural(self):
        fmt = AnnotatingFormatter()
        segments = fmt.annotated_format(
            "{vs[item_1]:p1}",
            {"item_1": "a pair of shoes"},
        )
        assert segments[0].text == "they"
        assert segments[0].source_type == "item"

    def test_case_conversion_upper(self):
        fmt = AnnotatingFormatter()
        segments = fmt.annotated_format(
            "{vs[name_1]!u}",
            {"name_1": "alice"},
        )
        assert segments[0].text == "ALICE"
        assert segments[0].source_type == "variable"

    def test_case_conversion_capitalize(self):
        fmt = AnnotatingFormatter()
        segments = fmt.annotated_format(
            "{vs[name_1]!c}",
            {"name_1": "alice"},
        )
        assert segments[0].text == "Alice"

    def test_case_conversion_lower(self):
        fmt = AnnotatingFormatter()
        segments = fmt.annotated_format(
            "{vs[name_1]!l}",
            {"name_1": "ALICE"},
        )
        assert segments[0].text == "alice"

    def test_switch_variable(self):
        fmt = AnnotatingFormatter()
        segments = fmt.annotated_format(
            "The {vs[switch_1]} option",
            {"switch_1": "left"},
        )
        assert segments[1].text == "left"
        assert segments[1].source_type == "switch"
        assert segments[1].source_key == "switch_1"

    def test_filler_variable(self):
        fmt = AnnotatingFormatter()
        segments = fmt.annotated_format(
            "Some {vs[filler_1]} text",
            {"filler_1": "extra context"},
        )
        assert segments[1].text == "extra context"
        assert segments[1].source_type == "filler"

    def test_coinflip_variable(self):
        fmt = AnnotatingFormatter()
        segments = fmt.annotated_format(
            "{vs[coinflip_1]}",
            {"coinflip_1": "heads"},
        )
        assert segments[0].text == "heads"
        assert segments[0].source_type == "coinflip"

    def test_multiple_variables(self):
        fmt = AnnotatingFormatter()
        segments = fmt.annotated_format(
            "{vs[name_1]} met {vs[name_2]}",
            {"name_1": "Alice", "name_2": "Bob"},
        )
        assert len(segments) == 3
        assert segments[0].text == "Alice"
        assert segments[0].source_key == "name_1"
        assert segments[1].text == " met "
        assert segments[1].source_type == "template"
        assert segments[2].text == "Bob"
        assert segments[2].source_key == "name_2"

    def test_empty_literal_between_vars_omitted(self):
        fmt = AnnotatingFormatter()
        segments = fmt.annotated_format(
            "{vs[name_1]}{vs[name_2]}",
            {"name_1": "A", "name_2": "B"},
        )
        assert len(segments) == 2
        assert segments[0].text == "A"
        assert segments[1].text == "B"

    def test_empty_string(self):
        fmt = AnnotatingFormatter()
        segments = fmt.annotated_format("", {})
        assert segments == []

    def test_number_variable(self):
        fmt = AnnotatingFormatter()
        segments = fmt.annotated_format(
            "{vs[smallnumber_1]} items",
            {"smallnumber_1": "three"},
        )
        assert segments[0].text == "three"
        assert segments[0].source_type == "number"
