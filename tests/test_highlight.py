"""Tests for web_editor.highlight HTML rendering."""

import pytest
from project_scripts.string_formatter import AnnotatedSegment
from web_editor.highlight import (
    SOURCE_COLORS,
    build_legend_html,
    segments_to_html,
)


class TestSegmentsToHtml:
    """Test segments_to_html converts segments to colored HTML."""

    def test_template_only_no_span(self):
        segments = [AnnotatedSegment("Hello world", "template", None)]
        html = segments_to_html(segments)
        assert "<span" not in html
        assert "Hello world" in html

    def test_variable_gets_colored_span(self):
        segments = [AnnotatedSegment("Alice", "variable", "name_1")]
        html = segments_to_html(segments)
        assert "<span" in html
        assert "Alice" in html
        assert SOURCE_COLORS["variable"] in html

    def test_html_escaping(self):
        segments = [AnnotatedSegment("<script>alert('xss')</script>", "template", None)]
        html = segments_to_html(segments)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_newline_to_br(self):
        segments = [AnnotatedSegment("line1\nline2", "template", None)]
        html = segments_to_html(segments)
        assert "<br>" in html

    def test_tooltip_shows_source_key(self):
        segments = [AnnotatedSegment("Alice", "variable", "name_1")]
        html = segments_to_html(segments)
        assert "name_1" in html
        assert "title=" in html

    def test_mixed_segments(self):
        segments = [
            AnnotatedSegment("Hello ", "template", None),
            AnnotatedSegment("Alice", "variable", "name_1"),
            AnnotatedSegment(" went to the ", "template", None),
            AnnotatedSegment("park", "item", "item_1"),
        ]
        html = segments_to_html(segments)
        assert "Hello " in html
        assert "Alice" in html
        assert SOURCE_COLORS["variable"] in html
        assert SOURCE_COLORS["item"] in html

    def test_empty_segments(self):
        html = segments_to_html([])
        assert html == ""

    def test_all_source_types_have_colors(self):
        source_types = [
            "variable", "switch", "filler", "pov",
            "coinflip", "item", "activity", "number",
        ]
        for st in source_types:
            assert st in SOURCE_COLORS

    def test_switch_type_colored(self):
        segments = [AnnotatedSegment("left", "switch", "switch_1")]
        html = segments_to_html(segments)
        assert SOURCE_COLORS["switch"] in html

    def test_pov_type_colored(self):
        segments = [AnnotatedSegment("you", "pov", "pov_1")]
        html = segments_to_html(segments)
        assert SOURCE_COLORS["pov"] in html


class TestBuildLegendHtml:
    """Test build_legend_html includes all source types."""

    def test_legend_contains_all_types(self):
        html = build_legend_html()
        for source_type in SOURCE_COLORS:
            assert source_type in html

    def test_legend_contains_color_samples(self):
        html = build_legend_html()
        for color in SOURCE_COLORS.values():
            assert color in html

    def test_legend_is_valid_html(self):
        html = build_legend_html()
        assert "<div" in html
        assert "</div>" in html
