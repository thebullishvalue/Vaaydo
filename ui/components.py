"""
Sanket — Reusable UI primitives.

The component vocabulary is Tattva's, adopted wholesale. These two are the same
product family and a reader moving between them should not have to relearn what
a panel, a chip or a number looks like.

What a page is made of
----------------------
- ``render_section_header`` — names a section and says how to read it.
- ``panel`` / ``render_chart_panel`` / ``render_table_panel`` — ONE anatomy for
  every framed thing: header (title / context · meta · chip) · body · footer.
  A screen mixing charts, tables and iframes reads as one grid rather than as
  several products sharing a page.
- ``render_metric_card`` / ``render_kpi_strip`` — the figure tier.
- ``render_chip`` — state in a word. One badge system, six tones.
- ``render_empty_state`` / ``render_notice_rail`` / ``render_warning_box`` —
  everything that used to be a bare ``st.info``/``st.warning``/``st.caption``.
- ``render_data_table`` — the only table primitive. There is no bare
  ``st.dataframe`` anywhere, because Streamlit's grid brings its own typeface,
  row height, header treatment and hover, none of which the stylesheet reaches.
- ``table_shell_css`` — the same tokens, exposed for Sanket's own richly
  formatted screener tables, which carry per-cell glyphs and colouring that a
  generic DataFrame renderer cannot express. They stay bespoke; they do not get
  to look bespoke.

Amber is caution, never brand. Interaction is cobalt. See ui/theme.py.
"""

from __future__ import annotations

import datetime as _dt
import html as html_mod
import re as _re
from contextlib import contextmanager as _contextmanager

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as _components_html


# ── SVG Icons (inline, no external deps) — with ARIA labels for accessibility
ICONS = {
    "chart":      '<svg aria-label="Chart icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
    "cube":       '<svg aria-label="Cube icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>',
    "target":     '<svg aria-label="Target icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
    "layers":     '<svg aria-label="Layers icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>',
    "bar-chart":  '<svg aria-label="Bar chart icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>',
    "activity":   '<svg aria-label="Activity icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
    "crosshair":  '<svg aria-label="Crosshair icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><line x1="22" y1="12" x2="18" y2="12"/><line x1="6" y1="12" x2="2" y2="12"/><line x1="12" y1="6" x2="12" y2="2"/><line x1="12" y1="22" x2="12" y2="18"/></svg>',
    "cpu":        '<svg aria-label="CPU icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>',
    "zap":        '<svg aria-label="Zap icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>',
    "shield":     '<svg aria-label="Shield icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>',
    "grid":       '<svg aria-label="Grid icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>',
    "database":   '<svg aria-label="Database icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>',
    "trending":   '<svg aria-label="Trending icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>',
    "eye":        '<svg aria-label="Eye icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>',
    "play":       '<svg aria-label="Play icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/></svg>',
    "chevron-right": '<svg aria-label="Expand icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"/></svg>',
    "sun":        '<svg aria-label="Light mode icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>',
    "moon":       '<svg aria-label="Dark mode icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>',
    "download":   '<svg aria-label="Download icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>',
    "briefcase":  '<svg aria-label="Portfolio icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="2" y="7" width="20" height="14" rx="2" ry="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/></svg>',
    "compass":    '<svg aria-label="Regime icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/></svg>',
    "trending-up": '<svg aria-label="Bull icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg>',
    "trending-down": '<svg aria-label="Bear icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 17 13.5 8.5 8.5 13.5 2 7"/><polyline points="16 17 22 17 22 11"/></svg>',
    "arrow-up":   '<svg aria-label="Up" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/></svg>',
    "arrow-down": '<svg aria-label="Down" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"/><polyline points="19 12 12 19 5 12"/></svg>',
    "move-horizontal": '<svg aria-label="Chop icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="18 8 22 12 18 16"/><polyline points="6 8 2 12 6 16"/><line x1="2" y1="12" x2="22" y2="12"/></svg>',
    "alert-triangle": '<svg aria-label="Caution icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    "help-circle": '<svg aria-label="Unknown icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    "circle":     '<svg aria-label="Circle" role="img" viewBox="0 0 24 24" fill="currentColor" stroke="none"><circle cx="12" cy="12" r="10"/></svg>',
    "check-circle": '<svg aria-label="Check" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
    "scale":      '<svg aria-label="Weighting icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h18"/></svg>',
    "history":    '<svg aria-label="History icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v5h5"/><path d="M3.05 13A9 9 0 1 0 6 5.3L3 8"/><path d="M12 7v5l4 2"/></svg>',
    "list":       '<svg aria-label="List icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg>',
    "info":       '<svg aria-label="Info icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>',
}


#: The app's single icon drawing style. Every icon is normalised to these by
#: ``get_icon`` regardless of what its own SVG literal declares — the set was
#: assembled over time and carries three different stroke weights and a mix of
#: butt/round terminals, which is exactly how an icon set stops reading as a
#: set. One weight, round terminals, no fills.
ICON_STROKE = 1.6
_ICON_LINECAP = "round"


def get_icon(name: str, size: int = 18, stroke_width: float | None = None) -> str:
    """Return an SVG icon normalised to the app's one icon style.

    ``stroke_width`` is accepted for call sites that predate ``ICON_STROKE``
    but is deliberately clamped: a caller passing 2 is drawing the same icons
    as everything else, two notches heavier, inside components that sit side
    by side.
    """
    base_svg = ICONS.get(name, ICONS["chart"])
    # Only on the outer <svg> tag: <rect> carries its own width/height, and
    # stripping those drew the cpu / grid / briefcase icons blank or dotted.
    head, sep, body = base_svg.partition(">")
    for attr in ("width", "height", "stroke-width", "stroke-linecap", "stroke-linejoin"):
        head = _re.sub(rf'\s+{attr}="[^"]*"', "", head)
    for attr in ("stroke-width", "stroke-linecap", "stroke-linejoin"):
        body = _re.sub(rf'\s+{attr}="[^"]*"', "", body)
    base_svg = head + sep + body
    sw = ICON_STROKE if stroke_width is None else min(float(stroke_width), 1.75)
    return base_svg.replace(
        "<svg",
        f'<svg width="{size}" height="{size}" stroke-width="{sw}" '
        f'stroke-linecap="{_ICON_LINECAP}" stroke-linejoin="{_ICON_LINECAP}"',
    )


# ═══════════════════════════════════════════════════════════════════════
#  HEADINGS AND CAPTIONS
# ═══════════════════════════════════════════════════════════════════════

def render_section_header(
    title: str,
    description: str = "",
    icon: str = "chart",
    accent: str = "",
) -> None:
    """Render a section header with icon, title and optional description.

    ``accent`` is a CSS colour class — "", "cyan", "emerald", "violet", "rose",
    "amber".
    """
    svg = get_icon(icon, size=16)
    icon_class = f"icon {accent}" if accent else "icon"
    hdr_class = f"section-hdr {accent}" if accent else "section-hdr"
    # `.desc` is a DIRECT child of the header, not nested inside `.text`. The
    # header is a two-row grid — icon and title on row 1, description on row 2
    # under the title — and a nested description is not a grid item, so it
    # could not be placed and fell back to flowing under the title with its own
    # margin. That is the gap that made the subtitle read as a detached
    # paragraph ~15px below its own title.
    desc_html = f'<div class="desc">{html_mod.escape(description)}</div>' if description else ""
    st.markdown(
        f'<div class="{hdr_class}">'
        f'<div class="{icon_class}">{svg}</div>'
        f'<div class="text"><h3>{html_mod.escape(title)}</h3></div>'
        f'{desc_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_sub_header(title: str) -> None:
    """A labelled division INSIDE a section — one tier below a section header.

    Mono micro-label, no icon, no rule. Used where a section has two named
    parts, which call sites otherwise hand-roll as inline-styled divs at three
    different sizes.
    """
    st.markdown(f'<div class="sub-head">{html_mod.escape(title)}</div>',
                unsafe_allow_html=True)


def render_control_hint(text: str) -> None:
    """The canonical helper caption beneath a control.

    Single source of truth for the "sub-control hint" tier — 10px mono,
    tertiary ink, sentence case. Use it instead of ``st.caption`` so the fine
    print stays one coherent hierarchy.

    NOT uppercase, deliberately, even though this is a micro tier. Every other
    10px-and-below tier in the system (chip, card label, panel context) is a
    LABEL of one or two words, where uppercase plus wide tracking aids
    scanning. This tier carries sentences — it is shared with ``render_note``
    for chart and table footnotes — and uppercase sentences are slower to read,
    not faster. The rule is "uppercase terse labels", not "uppercase small
    text".
    """
    st.markdown(f'<div class="control-hint">{html_mod.escape(text)}</div>',
                unsafe_allow_html=True)


#: `**bold**` and `*italic*`, in that order — the two-star form must match
#: first or the single-star pattern eats one of its stars.
_EMPHASIS = ((_re.compile(r"\*\*(.+?)\*\*", _re.S), r"<strong>\1</strong>"),
             (_re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", _re.S), r"<em>\1</em>"))


def render_note(text: str) -> None:
    """The note under a chart or table — commentary, in the prose face (from Pragyam).

    It used to reuse the control-hint tier: 10px mono, which is a DATUM face, so
    a paragraph on how to read a chart looked like a number that had lost its
    column. A note is commentary and takes the prose treatment, the same as every
    panel body — data and commentary stay distinguishable with the page out of
    focus. `render_control_hint` keeps the mono tier for terse data under a control.

    EMPHASIS IS TRANSLATED. Wrapped in a ``<div>``, markdown stops being parsed,
    so ``**term**`` would reach the screen literally; it is converted here. HTML
    is also allowed.
    """
    for pattern, repl in _EMPHASIS:
        text = pattern.sub(repl, text)
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
#  PANEL SYSTEM — one anatomy for every framed thing in the app
# ═══════════════════════════════════════════════════════════════════════
#
# A panel is: header (title / context, meta and chip right) · body · footer.
# Charts, tables and embedded iframes all use it, so a screen mixing them
# reads as one grid instead of as several products sharing a page.
#
# It is a real ``st.container`` rather than an HTML string because the body
# holds WIDGETS — a Plotly figure, a components.v1 iframe — which no amount
# of markdown can wrap. The container carries `key="panel-<id>"`, and
# theme.css styles `[class*="st-key-panel-"]`.

def render_panel_header(
    title: str = "",
    *,
    context: str = "",
    meta: str = "",
    chip: "tuple[str, str] | None" = None,
) -> None:
    """Render a panel header.

    ``title`` — what the panel shows. Omit it when the section header directly
    above already names the panel; a panel header that restates the section
    header is a second title, not a header.
    ``context`` — the panel's own metadata (universe, timeframe, units).
    ``meta``/``chip`` — right-aligned status: as-of, source, freshness.
    """
    if not (title or context or meta or chip):
        return
    left = ""
    if title:
        left += f'<span class="ph-title">{html_mod.escape(title)}</span>'
    if context:
        left += f'<span class="ph-context">{html_mod.escape(context)}</span>'
    right = ""
    if meta:
        right += f'<span>{html_mod.escape(meta)}</span>'
    if chip:
        right += render_chip(chip[0], chip[1], as_html=True) or ""
    st.markdown(
        f'<div class="panel-hdr"><div class="ph-left">{left}</div>'
        f'<div class="ph-right">{right}</div></div>',
        unsafe_allow_html=True,
    )


@_contextmanager
def panel(
    key: str,
    title: str = "",
    *,
    context: str = "",
    meta: str = "",
    chip: "tuple[str, str] | None" = None,
    footer: str = "",
):
    """Context manager wrapping any content in the shared panel chrome.

    ``with panel("edge-study", context="NIFTY 50 · Daily"): st.plotly_chart(...)``

    Use it directly for anything that is neither a chart nor a table (an
    embedded widget, a bespoke layout) so that thing still belongs to the
    system rather than sitting on the page unframed.
    """
    with st.container(key=f"panel-{key}"):
        render_panel_header(title, context=context, meta=meta, chip=chip)
        yield
        if footer:
            st.markdown(f'<div class="panel-foot">{footer}</div>', unsafe_allow_html=True)


def default_chart_context(units: str = "") -> str:
    """The context line every chart panel gets for free: universe · timeframe.

    Read from session state rather than threaded through every call site, which
    is both less plumbing and strictly more correct — a context built from the
    same keys the command bar reads cannot disagree with it.
    """
    parts = [
        str(st.session_state.get("active_universe", "") or "").upper(),
        str(st.session_state.get("active_timeframe", "") or ""),
    ]
    if units:
        parts.append(units)
    return " · ".join(p for p in parts if p)


def render_chart_panel(
    fig,
    key: str,
    title: str = "",
    *,
    units: str = "",
    context: str | None = None,
    meta: str = "",
    chip: "tuple[str, str] | None" = None,
    footer: str = "",
) -> None:
    """Render a Plotly figure inside the shared panel chrome.

    The ONE way a chart reaches the screen in this app. Every call passes the
    same ``PLOTLY_CONFIG``, which is what removes the stock Plotly toolbar and
    its logo — without it every chart ships Plotly's own chrome, complete with
    a link out to plotly.com.

    ``title`` is normally EMPTY: every chart already sits under a
    ``render_section_header`` that names it and explains how to read it, so a
    panel title would be that title again, four pixels lower. What the section
    header cannot say is which universe and timeframe the plot is drawn on, so
    that is what the panel header carries.
    """
    from ui.theme import PLOTLY_CONFIG, fix_hover_formats   # local: avoids a circular import
    ctx = default_chart_context(units) if context is None else context
    with panel(key, title, context=ctx, meta=meta, chip=chip, footer=footer):
        fix_hover_formats(fig)
        st.plotly_chart(fig, width="stretch", key=f"chart-{key}", config=PLOTLY_CONFIG)


def render_table_panel(
    df,
    key: str,
    title: str = "",
    *,
    units: str = "",
    context: str | None = None,
    meta: str = "",
    chip: "tuple[str, str] | None" = None,
    footer: str = "",
    **table_kwargs,
) -> None:
    """Render a DataFrame inside the shared panel chrome.

    Same header anatomy — and the same ``units``/``context`` contract — as
    ``render_chart_panel``, deliberately, so a table and a chart sitting side
    by side are visibly the same kind of object.

    ``units`` MUST be declared here rather than left to ``**table_kwargs``:
    without it, every ``units=`` at a call site falls through to
    ``render_data_table``, which has no such parameter, and the page dies with
    ``render_data_table() got an unexpected keyword argument 'units'``.
    """
    ctx = default_chart_context(units) if context is None else context
    with panel(key, title, context=ctx, meta=meta, chip=chip, footer=footer):
        render_data_table(df, **table_kwargs)


@_contextmanager
def html_panel(
    key: str,
    title: str = "",
    *,
    context: str = "",
    meta: str = "",
    chip: "tuple[str, str] | None" = None,
    footer: str = "",
):
    """Panel chrome around a ``components.v1.html`` iframe.

    Sanket's screener tables carry per-cell glyphs, conviction bars and hold
    counters that ``render_data_table`` cannot express, so they stay bespoke.
    They do not get to look bespoke: this puts them in the same frame as every
    chart and every DataFrame, and ``table_shell_css`` gives them the same
    typeface, row height and header treatment.
    """
    with panel(key, title, context=context, meta=meta, chip=chip, footer=footer):
        yield


def render_chart_error(key: str, title: str, reason: str) -> None:
    """A failed chart keeps its panel and explains itself inside it.

    A chart that cannot draw must not vanish — a missing panel reads as "there
    was nothing to show", which is a different claim from "this could not be
    computed" and the wrong one to make silently.
    """
    with panel(key, title, chip=("UNAVAILABLE", "warning")):
        st.markdown(f'<div class="panel-state">{html_mod.escape(reason)}</div>',
                    unsafe_allow_html=True)


def render_loading_skeleton(key: str, *, rows: int = 1, height: int = 220,
                            title: str = "", context: str = "") -> None:
    """A panel-shaped placeholder at the size of the thing that is coming.

    Sized to the final content so the layout does not jump when the real panel
    replaces it. ``rows=1, height=N`` is the chart case; ``rows=N`` is the
    table case (a header rule plus N row bars).
    """
    with panel(key, title, context=context):
        if rows <= 1:
            body = f'<div class="skeleton" style="height:{height}px"></div>'
        else:
            body = ('<div class="skeleton sk-head"></div>'
                    + "".join('<div class="skeleton sk-row"></div>' for _ in range(rows)))
        st.markdown(f'<div class="skeleton-body">{body}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
#  FIGURES, CHIPS, MASTHEAD
# ═══════════════════════════════════════════════════════════════════════

def render_chip(label: str, tone: str = "neutral", *, as_html: bool = False) -> str | None:
    """Render (or return) a status chip — one badge system for the whole app.

    ``tone``: ``accent`` / ``success`` / ``danger`` / ``warning`` / ``info`` /
    ``neutral``. Pass ``as_html=True`` to get the raw ``<span>`` back for
    composition inside a larger ``st.markdown`` call.
    """
    html = f'<span class="chip chip-{html_mod.escape(tone)}">{html_mod.escape(label)}</span>'
    if as_html:
        return html
    st.markdown(html, unsafe_allow_html=True)
    return None


def render_metric_card(
    label: str,
    value: str,
    subtext: str = "",
    color_class: str = "neutral",
    tooltip: str = "",
    icon: str = "",
) -> None:
    """Render a metric card with optional tooltip.

    ``color_class`` — "neutral", "success", "danger", "warning", "info",
    "violet".
    """
    tooltip_html = ""
    if tooltip:
        tooltip_html = (
            f'<div class="metric-tooltip" data-tooltip="{html_mod.escape(tooltip)}">'
            f'<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">'
            f'<circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>'
            f'<line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
            f'<span class="metric-tooltip-text">{html_mod.escape(tooltip)}</span>'
            f'</div>'
        )
    sub_metric_html = f'<div class="sub-metric">{html_mod.escape(subtext)}</div>' if subtext else ""
    icon_html = f'<span class="card-icon">{get_icon(icon, size=12)}</span> ' if icon else ""
    st.markdown(
        f'<div class="metric-card {html_mod.escape(color_class)}">'
        f'<span class="label">{icon_html}{html_mod.escape(label)}</span>'
        f"<h2>{html_mod.escape(value)}</h2>"
        f"{sub_metric_html}"
        f"{tooltip_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_kpi_strip(items: "list[dict]", *, max_cols: int = 5, key: str = "kpi-strip") -> None:
    """Lay out ``render_metric_card`` items in rows of at most ``max_cols``.

    Caps row width so cards never squeeze past legibility — wraps to a second
    row instead of going tight on narrow viewports.
    """
    if not items:
        return
    with st.container(key=key):
        for i in range(0, len(items), max_cols):
            row = items[i:i + max_cols]
            cols = st.columns(len(row), gap="small")
            for c, item in zip(cols, row):
                with c:
                    render_metric_card(
                        label=item.get("label", ""),
                        value=item.get("value", ""),
                        subtext=item.get("subtext", ""),
                        color_class=item.get("color_class", "neutral"),
                        tooltip=item.get("tooltip", ""),
                        icon=item.get("icon", ""),
                    )


def render_header(title: str, tagline: str) -> None:
    """Render the cold-start masthead.

    Stacked, not inline: the mark reads at display size on its own line and the
    tagline sits under it as a rule-delimited subtitle. Inline, the two compete
    for the same optical line and the mark ends up the size of a section
    heading — which is what a masthead must not be, since it is the only thing
    on a cold-start screen that says what the application is.
    """
    head, tail = (title[:3], title[3:]) if len(title) > 3 else (title, "")
    st.markdown(
        f'<div class="premium-header">'
        f'<div class="title">{html_mod.escape(head)}'
        f'<span class="accent-ink">{html_mod.escape(tail)}</span></div>'
        f'<div class="tagline">{html_mod.escape(tagline)}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_nav_brand(title: str = "VAAYDO",
                     tagline: str = "वायदो · F&O Trade Screener") -> None:
    """Render the control rail's brand block.

    The mark is split so the second half carries the accent — a product mark
    that is one flat colour reads as a heading, not as a mark. Left-aligned
    (not centred) because everything below it in the rail is left-aligned, and
    a centred mark over a left-aligned column is the single most common tell of
    a template.
    """
    head, tail = (title[:3], title[3:]) if len(title) > 3 else (title, "")
    st.markdown(
        f'<div class="nav-brand">'
        f'<div class="mark">{html_mod.escape(head)}'
        f'<span class="accent-ink">{html_mod.escape(tail)}</span></div>'
        f'<div class="tagline">{html_mod.escape(tagline)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_top_bar(
    *,
    target: str = "",
    price: float | None = None,
    change_pct: float | None = None,
    status_label: str = "",
    status_tone: str = "neutral",
    meta: str = "",
    meta_items: "list[tuple[str, str]] | None" = None,
    open_strip: bool = False,
) -> None:
    """Render the command bar — the first element on every page.

    Reading order left to right is *identity → value → trust*: which universe,
    what it reads, and whether the data behind that is current. Nothing renders
    above this bar; data-quality notices hang BELOW it in the notice rail, so
    the thing being analysed is always the first thing on screen rather than an
    apology about it.

    ``change_pct`` is in PERCENT POINTS (-0.42 == -0.42%), the same unit it is
    printed in — passing a fraction makes every sub-1% session print "0.00%".
    """
    instrument_html = ""
    if target:
        instrument_html = (
            f'<div class="cb-instrument">'
            f'<span class="eyebrow">Universe</span>'
            f'<span class="sym">{html_mod.escape(target)}</span>'
            f'</div>'
        )
    quote_html = ""
    if price is not None:
        chg = change_pct if change_pct is not None else 0.0
        chg_cls = "up" if chg > 0.005 else "down" if chg < -0.005 else "flat"
        arrow = "▲" if chg_cls == "up" else "▼" if chg_cls == "down" else "▬"
        # Arrow carries the sign, so the number is unsigned — "▼ 0.42%", not
        # "▼ -0.42%". Direction is stated twice (glyph + colour) and never by
        # colour alone, for red/green deficiency.
        chg_html = (f'<span class="chg {chg_cls}">{arrow} {abs(chg):.2f}%</span>'
                    if change_pct is not None else "")
        quote_html = f'<div class="cb-quote"><span class="px">{price:,.2f}</span>{chg_html}</div>'

    items = list(meta_items or [])
    if meta:
        items.append(("As of", meta.replace("As of ", "")))
    meta_html = "".join(
        f'<div class="cb-meta"><span class="k">{html_mod.escape(str(k))}</span>'
        f'<span class="v">{html_mod.escape(str(v))}</span></div>'
        for k, v in items if v
    )
    chip_html = render_chip(status_label, status_tone, as_html=True) if status_label else ""
    open_cls = " open" if open_strip else ""
    st.markdown(
        f'<div class="command-bar{open_cls}">'
        f'<div class="cb-left">'
        f'<div class="cb-brand"><span class="mark">VAA<span class="accent-ink">YDO</span></span>'
        f'<span class="sub">वायदो</span></div>'
        f'{instrument_html}{quote_html}'
        f'</div>'
        f'<div class="cb-right">{meta_html}{chip_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_notice_rail(notices: "list[dict] | None") -> None:
    """Render queued data-quality notices as a compact rail under the chrome.

    Each notice is ``{"kind": "warning"|"info", "title": str, "body": str}``.
    These used to render as full-width boxes at the very top of the page, which
    pushed the actual reading below the fold — so on exactly the days the data
    most needed scrutiny the interface led with an apology instead of a result.
    One row each, severity on the left rule: same information, a third of the
    vertical cost, and BELOW the thing it qualifies rather than above it.
    """
    if not notices:
        return
    rows = "".join(
        f'<div class="notice {html_mod.escape(n.get("kind", "info"))}">'
        f'<div class="n-title">{html_mod.escape(n.get("title", ""))}</div>'
        f'<div class="n-body">{n.get("body", "")}</div>'
        f'</div>'
        for n in notices
    )
    st.markdown(f'<div class="notice-rail">{rows}</div>', unsafe_allow_html=True)


def render_rail_readout(rows: "list[tuple[str, str, str]]") -> None:
    """Render the rail's session readout — ``(label, value, tone)`` rows.

    ``tone`` is "" / "accent" / "long" / "short" / "caution". Values are
    right-aligned and tabular so a changed number is seen rather than read.
    """
    if not rows:
        return
    body = "".join(
        f'<div class="row"><span class="k">{html_mod.escape(str(k))}</span>'
        f'<span class="v {html_mod.escape(tone)}">{html_mod.escape(str(v))}</span></div>'
        for k, v, tone in rows
    )
    st.markdown(f'<div class="rail-readout">{body}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
#  STATES AND NOTICES
# ═══════════════════════════════════════════════════════════════════════

def render_empty_state(title: str, body: str, *, eyebrow: str = "",
                       action_label: str = "") -> None:
    """A professional, icon-free empty/degraded state.

    One system for every "nothing to show yet" moment — cold start, every
    symbol still warming up, no signals fired — rather than each call site
    hand-rolling its own notice. ``body`` may carry simple inline HTML.
    ``action_label`` is a short hint at what to do next, not a real button.
    """
    eyebrow_html = f'<div class="es-eyebrow">{html_mod.escape(eyebrow)}</div>' if eyebrow else ""
    action_html = f'<div class="es-action">{html_mod.escape(action_label)}</div>' if action_label else ""
    st.markdown(
        f'<div class="empty-state">{eyebrow_html}'
        f'<div class="es-title">{html_mod.escape(title)}</div>'
        f'<div class="es-body">{body}</div>{action_html}</div>',
        unsafe_allow_html=True,
    )


def render_info_box(title: str, content: str, color: str = "cyan") -> None:
    """An info box. ``color`` is a modifier class (cyan / amber / emerald /
    rose / violet)."""
    st.markdown(
        f'<div class="info-box {html_mod.escape(color)}">'
        f"<h4>{html_mod.escape(title)}</h4><p>{html_mod.escape(content)}</p></div>",
        unsafe_allow_html=True,
    )


def render_warning_box(title: str, content: str) -> None:
    """A themed alert box for something the reader must not miss."""
    st.markdown(
        f'<div class="warning-box"><div class="icon"></div><div>'
        f'<div class="title">{html_mod.escape(title)}</div>'
        f'<div class="content">{html_mod.escape(content)}</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def render_interpretation_card(title: str, body: str, color: str = "neutral") -> None:
    """A state-aware interpretation card — terminal readout style.

    ``title`` is a short state label; ``body`` is one paragraph (raw HTML
    allowed — the caller is trusted). ``color`` is "neutral", "success",
    "danger", "warning", "info".
    """
    st.markdown(
        f'<div class="interp-card {html_mod.escape(color)}">'
        f'<div class="interp-title">{html_mod.escape(title)}</div>'
        f'<div class="interp-body">{body}</div></div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════
#  DATA TABLES
# ═══════════════════════════════════════════════════════════════════════
# ``render_data_table`` renders into an isolated components.v1.html iframe,
# which does NOT inherit the app's CSS variables — so the theme values it needs
# are mirrored here as literals, in BOTH a dark and a light set. Any change to
# the corresponding --token in theme.css / ui.theme.LIGHT_TOKENS has to be made
# here too; there is no way around that while the table lives in an iframe, and
# a stale colour here is the visible symptom.
#
# Header rule and hover use the primary ACCENT — not amber. Amber is caution in
# this system and nothing else.
_TABLE_TOKENS_DARK = {
    "ink_primary":   "#E6EAF1",   # --ink
    "ink_secondary": "#AEB8C7",   # --ink-secondary
    "ink_tertiary":  "#8B95A6",   # --ink-tertiary
    "ink_quaternary": "#737D8E",  # --ink-quaternary
    "border":        "rgba(255, 255, 255, 0.07)",   # --line
    "border_subtle": "rgba(255, 255, 255, 0.035)",  # --line-faint
    "accent":        "#4C7DF0",   # --accent
    "emerald":       "#2CA36B",   # --long
    "rose":          "#DD5A5A",   # --short
    "amber":         "#D79A3C",   # --caution
    "cyan":          "#4E9FC4",   # --system
    "violet":        "#9B8FD4",   # --violet
    "slate":         "#7E8797",   # --neutral
    "long_fill":     "rgba(44, 163, 107, 0.10)",   # --long-fill
    "accent_border": "rgba(76, 125, 240, 0.34)",
    "accent_hover":  "rgba(76, 125, 240, 0.10)",
    "row_odd":       "rgba(255, 255, 255, 0.015)",
    "row_even":      "transparent",
    "surface_a":     "#0F1217",   # --surface-1
    "surface_b":     "#0F1217",
    "header_a":      "#151920",   # --surface-2
    "header_b":      "#151920",
}
_TABLE_TOKENS_LIGHT = {
    "ink_primary":   "#141920",
    "ink_secondary": "#3D4756",
    "ink_tertiary":  "#5E6979",
    "ink_quaternary": "#6B7482",
    "border":        "rgba(15, 23, 42, 0.10)",
    "border_subtle": "rgba(15, 23, 42, 0.05)",
    "accent":        "#2B5FD9",
    "emerald":       "#0F7A54",
    "rose":          "#C0392F",
    "amber":         "#96660F",
    "cyan":          "#15708C",
    "violet":        "#6A4BC0",
    "slate":         "#5A6472",
    "long_fill":     "rgba(15, 122, 84, 0.08)",
    "accent_border": "rgba(43, 95, 217, 0.32)",
    "accent_hover":  "rgba(43, 95, 217, 0.07)",
    "row_odd":       "rgba(15, 23, 42, 0.022)",
    "row_even":      "transparent",
    "surface_a":     "#FFFFFF",
    "surface_b":     "#FFFFFF",
    "header_a":      "#EEF1F5",
    "header_b":      "#EEF1F5",
}


def table_tokens() -> dict:
    """Active-theme token set for iframe-isolated tables.

    Public because Sanket's screener tables build their own markup and must
    draw from exactly this set — a bespoke table with its own palette is how
    one surface ends up looking like a different product.
    """
    return _TABLE_TOKENS_LIGHT if st.session_state.get("theme") == "light" else _TABLE_TOKENS_DARK


#: Back-compat/private alias used by ``render_data_table``.
_table_tokens = table_tokens

#: Webfont the iframe must import for itself, for the same isolation reason.
_TABLE_FONTS = ("https://fonts.googleapis.com/css2?"
                "family=JetBrains+Mono:wght@400;500;600;700&display=swap")

#: THE TYPE RAMP, mirrored for markup that cannot read a CSS variable.
#:
#: theme.css declares these as --fs-* on :root, and everything in the app DOM
#: must use the variable. An iframe cannot see it — ``render_data_table`` and
#: Sanket's screener tables render into an isolated document — so the same nine
#: values are restated here and read from Python instead. Two statements of one
#: ramp is the cost of the isolation; a THIRD, invented at a call site, is not.
#:
#: The tiers, and what each is for:
#:   3xs  9px  micro labels, eyebrows      — the floor. Nothing is smaller.
#:   2xs 10px  table headers, chips, cells one tier below body
#:   xs  11px  dense metadata, table body, prose in a panel
#:   sm  12px  secondary body, compact card values
#:   md  13px  BODY
#:   lg  15px  section titles, compact card headline
#:   xl  19px  card values
#:   2xl 24px  hero secondary
#:   3xl 32px  hero signal
FS = {
    "3xs": "0.5625rem", "2xs": "0.625rem", "xs": "0.6875rem", "sm": "0.75rem",
    "md": "0.8125rem", "lg": "0.9375rem", "xl": "1.1875rem",
    "2xl": "1.5rem", "3xl": "2rem",
}

#: The data face, for iframe markup. MUST match --mono in theme.css and the
#: family ``_TABLE_FONTS`` actually imports.
#:
#: Declaring 'IBM Plex Mono' here instead is a specific, documented bug: the
#: iframe imports JetBrains only, so IBM Plex never loads and the cell falls
#: through to the generic `monospace` keyword — the system default (Menlo,
#: Courier). That makes the tables the one surface in the app rendering in a
#: typeface the rest of the UI does not use, which is exactly how it reads.
MONO_STACK = "'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,monospace"

#: Row and header geometry, shared by ``render_data_table`` and the bespoke
#: tables so a Sanket screener table and a Tattva DataFrame are the same object.
#:
#: MEASURED, not estimated (Pragyam found this first): a body row is 0.6875rem at
#: One row is exactly TABLE_ROW_H: line-height 1.5 (16.5px) + a 1px rule + the cell
#: padding DERIVED from it (TABLE_PAD_V above and below). A fixed 0.4rem made
#: 30.3px rows against a 31px budget, and the slack piled up as a blank strip under
#: every long table (17px on the screener). 30px is the family's row (Pragyam's).
#: The header is ~29px plus the scroller's 2px.
TABLE_ROW_H = 30
TABLE_PAD_V = (TABLE_ROW_H - 17.5) / 2
TABLE_HEADER_H = 31
#: The empty-state row (`.empty`: 2rem padding above and below one line), measured 81px.
TABLE_EMPTY_H = 82


def table_shell_css(*, max_height: int = 520) -> str:
    """The stylesheet a bespoke iframe table needs to match the app's one table.

    Sanket's screener tables carry per-cell glyphs (▲ BUY / ◆ SELL), conviction
    readouts and hold counters that ``render_data_table`` cannot express, so
    they build their own rows. They must not build their own LOOK: this returns
    the same tokens, the same JetBrains Mono, the same 27px rows, the same
    hairline sticky header and the same accent hover, so the only difference
    between the two is the content of a cell.

    Emits the ``<style>`` body only — the caller wraps it.
    """
    t = table_tokens()
    return f"""
    @import url('{_TABLE_FONTS}');
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{ font-family:{MONO_STACK};
            background:transparent; color:{t['ink_primary']};
            font-variant-numeric:tabular-nums; font-feature-settings:"tnum" 1,"zero" 1; }}
    .tt-scroll {{ padding-top:2px; max-height:{max_height}px; overflow:auto;
                  scrollbar-width:thin; scrollbar-color:{t['ink_tertiary']} transparent; }}
    .tt-scroll::-webkit-scrollbar {{ width:9px; height:9px; }}
    .tt-scroll::-webkit-scrollbar-track {{ background:transparent; }}
    .tt-scroll::-webkit-scrollbar-thumb {{ background:{t['border']}; border-radius:100px; }}
    .tt-scroll::-webkit-scrollbar-thumb:hover {{ background:{t['ink_tertiary']}; }}
    .tt-scroll::-webkit-scrollbar-corner {{ background:transparent; }}
    table {{ width:100%; border-collapse:collapse; }}
    thead th {{ position:sticky; top:0; z-index:2; background:{t['header_a']};
        color:{t['ink_tertiary']}; font-size:{FS['2xs']}; font-weight:600;
        text-transform:uppercase; letter-spacing:0.12em; padding:0.5rem 0.75rem;
        border-bottom:1px solid {t['border']}; text-align:left; white-space:nowrap; }}
    thead th.numeric, thead th.num {{ text-align:right; }}
    tbody tr {{ border-bottom:1px solid {t['border_subtle']};
                transition:background 120ms cubic-bezier(0.2,0,0,1); }}
    tbody tr:last-child {{ border-bottom:none; }}
    tbody tr:hover {{ background:{t['accent_hover']}; }}
    tbody td {{ padding:{TABLE_PAD_V}px 0.75rem; color:{t['ink_primary']}; font-size:{FS['xs']};
                line-height:1.5; vertical-align:middle; white-space:nowrap; }}
    tbody td.numeric, tbody td.num {{ text-align:right; }}
    tbody td.symbol, tbody td.lbl {{ font-weight:600; color:{t['ink_primary']}; }}
    tbody td.txt {{ color:{t['ink_tertiary']}; }}
    tbody td.currency {{ text-align:right; color:{t['ink_secondary']}; }}
    tbody td.best {{ box-shadow:inset 2px 0 0 {t['emerald']}; background:{t['long_fill']}; }}
    .sect {{ background:{t['header_a']}; color:{t['ink_secondary']};
             font-size:{FS['2xs']}; font-weight:600; text-transform:uppercase;
             letter-spacing:0.12em; padding:0.5rem 0.75rem;
             border-top:1px solid {t['border']}; border-bottom:1px solid {t['border']}; }}
    .empty {{ text-align:center; color:{t['ink_quaternary']}; font-size:{FS['xs']};
              letter-spacing:0.06em; padding:2rem 1rem; }}
    """


def table_iframe_height(n_rows: int, *, extra_rows: int = 0, max_height: int = 520) -> int:
    """Pixel height for a bespoke table iframe — one geometry, one formula."""
    if n_rows <= 0 and not extra_rows:
        # An empty table shows its `.empty` row, 81px, not a 31px data row: sized
        # as one row it was 47px short and an empty table drew a scrollbar.
        return int(min(TABLE_HEADER_H + TABLE_EMPTY_H + 4, max_height))
    content = TABLE_HEADER_H + (n_rows + extra_rows) * TABLE_ROW_H + 4
    return int(min(content, max_height))


def _fmt_cell(value, precision: int) -> str:
    """Format one cell value for display (NaN → em dash; floats to `precision`).

    Dates render date-only: Sanket is a daily/weekly system, so a Timestamp's
    ``00:00:00`` time component is noise — never shown.
    """
    if value is None:
        return "—"
    if isinstance(value, (pd.Timestamp, _dt.date)):
        try:
            if pd.isna(value):
                return "—"
        except (TypeError, ValueError):
            pass
        return value.strftime("%Y-%m-%d")
    if isinstance(value, float):
        if value != value:            # NaN
            return "—"
        return f"{value:,.{precision}f}"
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}"
    try:
        if pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        pass
    return html_mod.escape(str(value))


# A number written as text: optional sign or ≈/±, optional currency, digits with
# separators and decimals, an optional unit (%, ×, σ, pts, pp, L, Cr …), optionally a
# range to a second such number, optionally a parenthetical ("1,625.48 (−14.5%)").
_NUM = r"[+\-−±≈~]?\s*[₹$]?\s*\d[\d,]*(?:\.\d+)?\s*(?:%|×|x|σ|pts|pp|bps|u|L|Cr|K|M|B)?"
_NUM_TEXT = _re.compile(rf"^{_NUM}(?:\s*–\s*{_NUM})?(?:\s*\([^)]*\))?$")
_EMPTY_TEXT = {"", "—", "–", "-"}

# A side, as a table writes it: the triangle, the position word, the order word.
_DIRECTION_TOKEN = _re.compile(r"▲|▼|\bLONG\b|\bSHORT\b|\bBUY\b|\bSELL\b")
_LONG_TOKENS = {"▲", "LONG", "BUY"}


def _is_numeric_text(s: "pd.Series") -> bool:
    """Is this object column a column of numbers written as text?

    Every non-empty value must read as a number (dashes for missing are allowed),
    and at least one must be present: a column of dashes is not evidence either way.
    """
    if s.dtype != object:
        return False
    seen = False
    for v in s:
        if v is None or (isinstance(v, float) and v != v):
            continue
        if not isinstance(v, str):
            return False
        t = v.strip()
        if t in _EMPTY_TEXT:
            continue
        if not _NUM_TEXT.match(t):
            return False
        seen = True
    return seen


# Column-name tokens that must stay UPPER-CASE when a raw column name is
# prettified into a professional header ("SID_Osc" → "SID Osc", not "Sid Osc").
_HEADER_ACRONYMS = {
    "SID", "CLR", "CVD", "RVOL", "POC", "VAH", "VAL", "VA", "ATR", "EMA", "SMA",
    "HMM", "GARCH", "CUSUM", "MA", "IC", "HR", "US", "FX", "ID", "N", "T", "Z",
    "R2", "OHLC", "OHLCV", "F1", "F2", "BUY", "SELL", "CI", "MDE",
}


def _prettify_header(name: str) -> str:
    """Turn a raw column/field name into a professional table header.

    ``SID_Hist_Z`` → ``SID Hist Z``; ``buy_cond`` → ``Buy Cond``. Already-clean
    headers pass through with only per-word acronym casing applied.
    """
    raw = str(name).replace("_", " ").strip()
    if not raw:
        return ""
    out = []
    for word in raw.split():
        up = word.upper()
        if up in _HEADER_ACRONYMS:
            out.append(up)
        elif word.isupper() and len(word) <= 4:   # keep short all-caps as-is
            out.append(word)
        else:
            out.append(word[:1].upper() + word[1:])
        if not word[:1].isalnum():                # Δ, %, … verbatim
            out[-1] = word
    return " ".join(out)


def render_data_table(
    df: "pd.DataFrame",
    *,
    index_label: str | None = None,
    show_index: bool | None = None,
    max_rows: int | None = None,
    precision: int = 2,
    col_precision: "dict[str, int] | None" = None,
    sign_color_cols: "set[str] | None" = None,
    lower_is_better_cols: "set[str] | None" = None,
    best_in_row: "list[bool] | None" = None,
    label_col: str | None = None,
    col_labels: "dict[str, str] | None" = None,
    max_height: int = 520,
    row_height: int = TABLE_ROW_H,
    direction_cols: "set[str] | None" = None,
) -> None:
    """Render a DataFrame as the app's one institutional table.

    There is no bare ``st.dataframe`` anywhere, because Streamlit's grid brings
    its own typeface, row height, header treatment and hover, none of which can
    be reached from the app's stylesheet. Sticky muted header, hairline row
    rules, right-aligned tabular numerics, a bolder "label" column, and scroll
    under a fixed ``max_height``.

    Rows are 27px: the usual 42 comes from 0.6rem padding at a 0.75rem font,
    which is a comfortable READING density, not a scanning one. A table a
    trader scans should fit twice as many rows in the same panel.

    Wrap it in ``render_table_panel`` rather than calling it directly, so the
    table gets the same header anatomy as every chart.

    Two readings carried from Pragyam:
    ``lower_is_better_cols`` — sign-coloured columns where NEGATIVE is the good
        outcome. Colour follows the outcome, never the sign of the arithmetic.
    ``best_in_row`` — marks the winning cell in each ROW of a comparison read
        horizontally, one bool per row (True where higher wins), because a table
        like that mixes quantities whose polarity differs by row.
    """
    if df is None or getattr(df, "empty", True):
        st.markdown('<div class="panel-state">No rows to display.</div>',
                    unsafe_allow_html=True)
        return

    view = df.tail(max_rows).copy() if max_rows else df.copy()
    if isinstance(view.columns, pd.MultiIndex):
        view.columns = [" · ".join(str(x) for x in c) for c in view.columns]

    if show_index is None:
        show_index = index_label is not None or not isinstance(view.index, pd.RangeIndex)
    idx_header = (index_label or _prettify_header(view.index.name or "")) if show_index else ""
    col_labels = col_labels or {}

    def _header(c: str) -> str:
        return col_labels.get(c) or _prettify_header(c)

    cols = list(view.columns)
    numeric_cols = {c for c in cols if pd.api.types.is_numeric_dtype(view[c])}
    # A column of numbers already formatted as text ("1,901.70", "+₹25,575", "+3.4%",
    # "1,625.48 (−14.5%)", "325 (1 lot)") is still a column of numbers: right-aligned in
    # the primary ink. Read by dtype alone it was text — left-aligned and tertiary, so a
    # table of prices and P&L read as prose. A pre-formatted table is how an app shows
    # rupees, lakh and percent exactly, so this is the common case, not the exception.
    # The label column is never a numeric column (resolved here as it is below).
    _lbl = label_col if label_col is not None else ("__index__" if show_index else (cols[0] if cols else None))
    text_numeric = {c for c in cols if c not in numeric_cols and c != _lbl and _is_numeric_text(view[c])}
    numeric_cols |= text_numeric
    lower_better = (lower_is_better_cols or set()) & numeric_cols
    sign_cols = ((sign_color_cols or set()) & numeric_cols) | lower_better
    col_precision = col_precision or {}
    if label_col is None:
        label_col = "__index__" if show_index else (cols[0] if cols else None)

    t = table_tokens()

    def _header_cells() -> str:
        cells = []
        if show_index:
            cells.append(f'<th class="lbl">{html_mod.escape(str(idx_header))}</th>')
        for c in cols:
            cls = "num" if c in numeric_cols and c != label_col else "lbl" if c == label_col else "txt"
            cells.append(f'<th class="{cls}">{html_mod.escape(_header(c))}</th>')
        return "".join(cells)

    direction_cols = direction_cols or set()

    def _value_html(c: str, val) -> str:
        p = col_precision.get(c, precision)
        text = _fmt_cell(val, p)
        if c in direction_cols and text != "—":
            # The direction, wherever it appears in the cell (a roll reads "SELL … · BUY …"),
            # in the side's ink: ▲ / LONG / BUY long, ▼ / SHORT / SELL short. Only the token is
            # inked — the glyph and the word stay, so a side is never told by colour alone.
            return _DIRECTION_TOKEN.sub(
                lambda m: f'<span style="color:{t["emerald"] if m.group(0) in _LONG_TOKENS else t["rose"]};'
                          f'font-weight:600;">{m.group(0)}</span>', text)
        if c in sign_cols and c in text_numeric and text != "—":
            # A formatted number carries its sign as its first character.
            s = str(val).strip()[:1]
            sgn = 1 if s == "+" else -1 if s in ("−", "-") else 0
            if c in lower_better:
                sgn = -sgn
            color = t["emerald"] if sgn > 0 else t["rose"] if sgn < 0 else t["ink_tertiary"]
            return f'<span style="color:{color};font-weight:600;">{text}</span>'
        if c in sign_cols and text != "—":
            try:
                fv = float(val)
                if c in lower_better:
                    fv = -fv
                color = (t["emerald"] if fv > 1e-12 else t["rose"] if fv < -1e-12
                         else t["ink_tertiary"])
                return f'<span style="color:{color};font-weight:600;">{text}</span>'
            except (TypeError, ValueError):
                pass
        return text

    def _winner(row, higher_wins: bool) -> "str | None":
        live = {}
        for c in cols:
            if c not in numeric_cols or c == label_col:
                continue
            try:
                v = float(row[c])
            except (TypeError, ValueError):
                continue
            if np.isfinite(v):
                live[c] = v
        if len(live) < 2:
            return None
        return max(live, key=live.get) if higher_wins else min(live, key=live.get)

    body_rows = []
    for r_i, (idx, row) in enumerate(view.iterrows()):
        win = (_winner(row, best_in_row[r_i])
               if best_in_row is not None and r_i < len(best_in_row) else None)
        tds = []
        if show_index:
            tds.append(f'<td class="lbl">{_fmt_cell(idx, precision)}</td>')
        for c in cols:
            cls = "num" if c in numeric_cols and c != label_col else "lbl" if c == label_col else "txt"
            if c == win:
                cls += " best"
            tds.append(f'<td class="{cls}">{_value_html(c, row[c])}</td>')
        body_rows.append(f"<tr>{''.join(tds)}</tr>")

    iframe_h = table_iframe_height(len(view), max_height=max_height)
    if row_height != TABLE_ROW_H:
        iframe_h = int(min(TABLE_HEADER_H + len(view) * row_height + 4, max_height))

    table_html = (
        '<!DOCTYPE html><html><head><meta charset="utf-8"><style>'
        + table_shell_css(max_height=max_height)
        + '</style></head><body><div class="tt-scroll"><table>'
        + f"<thead><tr>{_header_cells()}</tr></thead>"
        + f"<tbody>{''.join(body_rows)}</tbody>"
        + "</table></div></body></html>"
    )
    # +8: the panel pads the table's element container by --s2 (8px) under its
    # header (theme.css); the container's height is the one given here, so the
    # padding must be added to it or it comes out of the table.
    _components_html(table_html, height=iframe_h + 8, scrolling=False)


def render_kv_table(data: dict, header_left: str = "Setting",
                    header_right: str = "Value") -> None:
    """A run's settings as a two-column readout (from Pragyam).

    A grid rather than ``render_data_table`` because these rows are read DOWN the
    value column — "what was this screen run under?" — not scanned or sorted.
    Values are mono: several are numbers.
    """
    if not data:
        render_empty_state("No settings recorded", "Run an analysis to freeze a configuration.")
        return
    cells = (f'<div class="key hdr">{html_mod.escape(header_left)}</div>'
             f'<div class="value hdr">{html_mod.escape(header_right)}</div>')
    for k, v in data.items():
        cells += (f'<div class="key">{html_mod.escape(str(k))}</div>'
                  f'<div class="value">{html_mod.escape(str(v))}</div>')
    st.markdown(f'<div class="kv-table-container"><div class="kv-table">{cells}</div></div>',
                unsafe_allow_html=True)


def render_set_board(cards: "list[dict]") -> None:
    """The signal board: one card per signal set, side by side, read across in one pass.

    Each card is ``{key, title, line, tone, rate, sides: [(glyph, count, sub, tone)],
    watch: (label, value), edges: [(glyph, verdict, chip_tone, number, tip)]}``. The three
    sets are never a choice the reader makes; they are three columns of the same answer
    — what fired, what is about to, and whether this universe has ever paid for it.
    """
    if not cards:
        return
    out = []
    for c in cards:
        e = html_mod.escape
        sides = "".join(
            f'<div class="sc-side {e(tone)}"><span class="g">{e(g)}</span>'
            f'<span class="n">{e(str(n))}</span><span class="t">{e(sub)}</span></div>'
            for g, n, sub, tone in c.get("sides", []))
        wl, wv = c.get("watch", ("", ""))
        watch = (f'<div class="sc-watch"><span class="k">{e(wl)}</span>'
                 f'<span class="v">{e(wv)}</span></div>') if wl else ""
        edges = "".join(
            f'<div class="sc-edge-row" title="{e(tip)}"><span class="k">{e(g)} tested</span>'
            f'{render_chip(lbl, ctone, as_html=True)}<span class="e">{e(num)}</span></div>'
            for g, lbl, ctone, num, tip in c.get("edges", []))
        out.append(
            f'<div class="set-card {e(c.get("tone", "neutral"))}">'
            f'<div class="sc-head"><span class="sc-key">{e(c.get("key", ""))}</span>'
            f'<span class="sc-title">{e(c.get("title", ""))}</span>'
            f'<span class="sc-rate">{e(c.get("rate", ""))}</span></div>'
            f'<div class="sc-line">{e(c.get("line", ""))}</div>'
            f'<div class="sc-sides">{sides}</div>{watch}'
            f'<div class="sc-edge">{edges}</div></div>')
    st.markdown(f'<div class="set-board">{"".join(out)}</div>', unsafe_allow_html=True)


# ── The landing's segments ───────────────────────────────────────────────────────
# Every member's cold start reads as Arthagati's: Engines (three panels), Inputs & Scope
# (render_kpi_strip), Run Outputs (four cards). Panels and cards are static text, so each
# set is ONE grid of plain divs, styled in the fidelity block of ui/theme.css: equal
# heights and one baseline for the spec rows by construction.
def render_system_grid(panels) -> None:
    """Engines: one ``(name, kicker, body, ((label, value), ...))`` per panel, three of them."""
    esc = html_mod.escape
    st.markdown(
        '<div class="system-grid">' + "".join(
            '<div class="system-panel"><div class="panel-hdr"><div class="ph-left">'
            f'<span class="ph-title">{esc(name)}</span><span class="ph-context">{esc(kicker)}</span>'
            f'</div></div><div class="panel-copy">{esc(body)}</div><div class="panel-specs">'
            + "".join(f'<div class="lookback-row"><span class="lbl">{esc(k)}</span>'
                      f'<span class="val">{esc(v)}</span></div>' for k, v in specs)
            + "</div></div>" for name, kicker, body, specs in panels) + "</div>",
        unsafe_allow_html=True)


def render_outcome_grid(cards) -> None:
    """Run Outputs: one ``(title, description)`` per card; four, laid out 4-up, 2-up or 1-up."""
    esc = html_mod.escape
    st.markdown('<div class="outcome-grid">' + "".join(
        f'<div class="outcome"><div class="o-t">{esc(t)}</div><div class="o-d">{esc(d)}</div></div>'
        for t, d in cards) + "</div>", unsafe_allow_html=True)
