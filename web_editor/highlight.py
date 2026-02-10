"""Convert AnnotatedSegments to colored HTML for the sample display."""

from __future__ import annotations

import html as html_lib

from project_scripts.string_formatter import AnnotatedSegment

SOURCE_COLORS: dict[str, str] = {
    "variable": "#dbeafe",   # light blue
    "switch": "#fef9c3",     # light yellow
    "filler": "#e0e7ff",     # light indigo
    "pov": "#dcfce7",        # light green
    "coinflip": "#fce7f3",   # light pink
    "item": "#ffedd5",       # light orange
    "activity": "#f3e8ff",   # light purple
    "number": "#ccfbf1",     # light teal
}


def segments_to_html(segments: list[AnnotatedSegment]) -> str:
    """Render segments as HTML with colored spans and tooltips."""
    if not segments:
        return ""
    parts: list[str] = []
    for seg in segments:
        escaped = html_lib.escape(seg.text).replace("\n", "<br>")
        if seg.source_type == "template":
            parts.append(escaped)
        else:
            color = SOURCE_COLORS.get(seg.source_type, "#f3f4f6")
            tooltip = html_lib.escape(seg.source_key or seg.source_type)
            parts.append(
                f'<span style="background-color:{color};padding:1px 3px;'
                f'border-radius:3px;cursor:help" '
                f'title="{tooltip} ({seg.source_type})">{escaped}</span>'
            )
    return "".join(parts)


def build_legend_html() -> str:
    """Render a compact color legend bar."""
    items: list[str] = []
    for source_type, color in SOURCE_COLORS.items():
        items.append(
            f'<span style="background-color:{color};padding:2px 8px;'
            f'border-radius:3px;margin-right:6px;font-size:0.85em">'
            f'{html_lib.escape(source_type)}</span>'
        )
    return f'<div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px">{"".join(items)}</div>'
