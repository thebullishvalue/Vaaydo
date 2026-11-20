"""
Sanket — Shared CSS, chart theming, and colour constants for the UI layer.
संकेत (Sanketa) — "Signal · Indicator · Forewarning"

UI — Institutional research terminal design language, adopted wholesale from
Tattva. These two are the same product family; a reader moving between them
should not have to relearn what a panel, a chip or a number looks like.

Aesthetic: "Graphite" — near-achromatic ground, semantic colour only
--------------------------------------------------------------------
- Display/UI:  Inter (prose, headings, labels)
- Body/Data:   JetBrains Mono (tabular numerals — every figure in the app)
- Ground:      Graphite (#0A0C10 -> #1C212A), deliberately neutral. The
               previous ramp was a saturated navy under an amber-gold accent,
               which tinted every panel and forced the semantic hues to shout
               over an already-coloured field.
- Semantic:    Cobalt #4C7DF0 (interactive), Green #2CA36B (long),
               Red #DD5A5A (short), Amber #D79A3C (caution ONLY),
               Steel #4E9FC4 (info). Muted, not the stock Tailwind-500 ramp;
               each clears WCAG AA on every surface it is used on.
- Surfaces:    Flat, told apart by a hairline border and one step of tone.
               No blur, no stacked shadows — a shadow is spent on overlays.
- Themes:      Slate (dark) and Paper (light, for reading and print). The
               light theme is a token swap, not a second stylesheet — see
               LIGHT_TOKENS below.

AMBER IS NO LONGER THE BRAND. Sanket's previous system made amber-gold the
product accent and also its caution colour, so "this is Sanket" and "be
careful" were the same signal. Amber now means caution and nothing else;
cobalt carries interaction.
"""

from __future__ import annotations

import re
import html
from pathlib import Path

import streamlit as st

# Path to external CSS file
CSS_PATH = Path(__file__).parent / "theme.css"


# ── The chart palette, dark ground ──────────────────────────────────────────
# Never read these by value at module scope in a tab/page file — see
# `chart_color`. They are the DARK ramp; the light one is `_PALETTE_LIGHT`.
_PALETTE_RGB: dict[str, tuple[int, int, int]] = {
    "emerald": (44, 163, 107),   # #2CA36B  long / bullish
    "rose":    (221, 90, 90),    # #DD5A5A  short / bearish
    "accent":  (76, 125, 240),   # #4C7DF0  primary / interactive
    "cyan":    (78, 159, 196),   # #4E9FC4  info (informational tone only)
    "amber":   (215, 154, 60),   # #D79A3C  caution ONLY, never brand
    "violet":  (155, 143, 212),  # #9B8FD4  secondary / attribution
    "slate":   (126, 135, 151),  # #7E8797  neutral / muted
}


def _palette_hex(name: str) -> str:
    r, g, b = _PALETTE_RGB[name]
    return f"#{r:02X}{g:02X}{b:02X}"


#: Dark-ground constants, for the handful of places that need a literal before
#: a session exists (page-config favicon, docstrings). Anything drawn INTO a
#: chart or a component must use `chart_color`/`chart_rgba` instead, which
#: resolve per render and therefore follow the appearance toggle.
COLOR_GREEN  = _palette_hex("emerald")
COLOR_RED    = _palette_hex("rose")
COLOR_ACCENT = _palette_hex("accent")
COLOR_CYAN   = _palette_hex("cyan")
COLOR_AMBER  = _palette_hex("amber")
COLOR_PURPLE = _palette_hex("violet")
COLOR_SLATE  = _palette_hex("slate")


# ── Light theme — token overrides only ──────────────────────────────────────
# theme.css defines the canonical dark :root token block; every component
# rule in it reads var(--token) with nothing hardcoded outside that block.
# Light mode is therefore just a second, smaller :root that redefines the
# same custom properties — injected AFTER the base stylesheet so it wins on
# source order, no runtime DOM attribute toggling required. Hues are
# deepened versions of the dark palette (not the same RGB) so text clears
# WCAG AA on a near-white surface.
LIGHT_TOKENS = """
:root {
    /* Counterpart to the dark block's declaration — see the note there. This
       is what keeps Paper light on a device whose system theme is dark; the
       two together mean the OS preference is never consulted in either
       direction. */
    color-scheme: light;

    /* Paper — the reporting/print theme. Not "dark inverted": a near-white
       ground reflects far more light than a graphite one, so the semantic
       hues are DEEPENED rather than reused (a #2CA36B that clears 5.9:1 on
       graphite manages 2.6:1 on white and would be illegible). Every value
       below clears WCAG AA on both --surface-1 and --surface-2. */
    --bg:            #F4F6F8;
    --surface-1:     #FFFFFF;
    --surface-2:     #EEF1F5;
    --surface-3:     #E2E7EE;

    --ink:           #141920;   /* 17.7:1 on white */
    --ink-secondary: #3D4756;   /*  9.4:1 */
    --ink-tertiary:  #5E6979;   /*  5.6:1 */
    --ink-quaternary:#6B7482;   /*  4.6:1 */
    --spike: rgba(90, 100, 114, 0.45);

    --accent:        #2B5FD9;   /* 5.6:1 */
    --long:          #0F7A54;   /* 5.3:1 */
    --short:         #C0392F;   /* 5.4:1 */
    --caution:       #96660F;   /* 5.0:1 */
    --system:        #15708C;   /* 5.6:1 */
    --neutral:       #5A6472;   /* 6.0:1 */

    --accent-fill:   rgba(43, 95, 217, 0.07);
    --long-fill:     rgba(15, 122, 84, 0.08);
    --short-fill:    rgba(192, 57, 47, 0.07);
    --caution-fill:  rgba(150, 102, 15, 0.08);
    --system-fill:   rgba(21, 112, 140, 0.07);
    --accent-edge:   rgba(43, 95, 217, 0.32);
    --long-edge:     rgba(15, 122, 84, 0.32);
    --short-edge:    rgba(192, 57, 47, 0.30);
    --caution-edge:  rgba(150, 102, 15, 0.30);
    --system-edge:   rgba(21, 112, 140, 0.28);

    --line:          rgba(15, 23, 42, 0.10);
    --line-strong:   rgba(15, 23, 42, 0.18);
    --line-faint:    rgba(15, 23, 42, 0.05);

    --violet:        #6A4BC0;   /* 6.2:1 */
    --violet-fill:   rgba(106, 75, 192, 0.07);
    --violet-edge:   rgba(106, 75, 192, 0.30);

    --shadow-sm:     0 1px 2px rgba(15, 23, 42, 0.06);
    --shadow-pop:    0 10px 24px rgba(15, 23, 42, 0.12);
}
/* Two rules cannot be expressed as a token swap. On paper the primary
   button's hover needs a DARKER accent (the dark theme's is lighter), and
   the sidebar rail reads better as the tinted surface with the content
   area white — the reverse of the dark theme's arrangement. */
[data-testid="stBaseButton-primary"]:hover { background: #244EB4 !important; border-color: #244EB4 !important; }
[data-testid="stSidebar"] { background: var(--surface-2); }

/* ── Reclaiming Streamlit's own natives ──────────────────────────────────
   THIS is why Paper mode looks half-broken without it. `.streamlit/config.toml`
   is a STATIC config — it cannot follow a runtime theme switch. So on Paper a
   token swap repaints every surface white while Streamlit keeps colouring its
   own internals for whichever base it resolved: navigation labels, button
   faces, input text and placeholders render near-white on near-white, while
   anything drawn through the app's own classes stays correct. That split is
   what makes the failure look arbitrary rather than total. */
[data-testid="stSidebarNav"] a span,
[data-testid="stSidebarNav"] a p,
[data-testid="stWidgetLabel"] label p,
[data-testid="stMarkdownContainer"] p:not(button p),
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div,
[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-secondaryFormSubmit"],
[data-testid="stDateInput"] input,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input { color: var(--ink) !important; }

[data-testid="stDateInput"] input::placeholder,
[data-testid="stTextInput"] input::placeholder { color: var(--ink-quaternary) !important; }

[data-baseweb="popover"] li,
[data-baseweb="popover"] div { color: var(--ink) !important; }
[data-baseweb="popover"] ul { background: var(--surface-1) !important; }

/* The collapse chevrons are drawn in app ink: at Streamlit's own near-white
   they were effectively invisible against Paper (now 4.9:1). */
[data-testid="stSidebarCollapseButton"] svg,
[data-testid="stExpandSidebarButton"] svg { color: var(--ink-secondary) !important; }
"""


# ── Theme resolution ────────────────────────────────────────────────────────
def _active_theme() -> str:
    """The active theme name — dark unless the appearance control has set light.

    Reads the DERIVED key, which `main()` writes once at the top of the run from
    the durable appearance choice. Everything in the script therefore agrees on
    one value for the whole render — the fix for a page that used to draw its
    chrome in one theme and its charts in the other.
    """
    return st.session_state.get("theme", "dark")


_CHART_THEME = {
    "dark": dict(
        font_color="#8B95A6",                 # --ink-tertiary
        hover_bg="rgba(21, 25, 32, 0.96)",    # --surface-2
        hover_border="rgba(255,255,255,0.13)",
        hover_text="#E6EAF1",
        grid="rgba(255,255,255,0.05)",
        grid_zero="rgba(255,255,255,0.11)",
        axis_line="rgba(255,255,255,0.09)",
        tick="#737D8E",
        spike="rgba(139,149,166,0.45)",
    ),
    "light": dict(
        font_color="#5E6979",
        hover_bg="rgba(255,255,255,0.97)",
        hover_border="rgba(15,23,42,0.18)",
        hover_text="#141920",
        grid="rgba(15,23,42,0.07)",
        grid_zero="rgba(15,23,42,0.16)",
        axis_line="rgba(15,23,42,0.12)",
        tick="#5E6979",
        spike="rgba(90,100,114,0.45)",
    ),
}


def _chart_theme() -> dict:
    return _CHART_THEME.get(_active_theme(), _CHART_THEME["dark"])


# ── Theme-aware CHART palette ───────────────────────────────────────────────
# The light values below are the SAME hexes LIGHT_TOKENS gives the chrome, so a
# green line equals the green value beside it in either theme, and each clears
# WCAG AA on its own ground.
_PALETTE_LIGHT: dict[str, tuple[int, int, int]] = {
    "emerald": (15, 122, 84),    # #0F7A54  5.3:1 on white
    "rose":    (192, 57, 47),    # #C0392F  5.4:1
    "accent":  (43, 95, 217),    # #2B5FD9  5.6:1
    "cyan":    (21, 112, 140),   # #15708C  5.6:1
    "amber":   (150, 102, 15),   # #96660F  5.0:1
    "violet":  (106, 75, 192),   # #6A4BC0  6.2:1
    "slate":   (90, 100, 114),   # #5A6472  6.0:1
}


def _palette() -> dict:
    return _PALETTE_LIGHT if _active_theme() == "light" else _PALETTE_RGB


def chart_color(name: str) -> str:
    """A semantic chart colour for the ACTIVE theme, as ``#RRGGBB``.

    The one way a call site names a colour. Use it in place of module-level
    constants, which bind at import time — when there is no session to read a
    theme from — and therefore cannot flip with the appearance toggle.
    """
    r, g, b = _palette()[name]
    return f"#{r:02X}{g:02X}{b:02X}"


def chart_rgba(name: str, alpha) -> str:
    """A semantic chart colour for the active theme, as ``rgba(...)``."""
    r, g, b = _palette()[name]
    return f"rgba({r},{g},{b},{alpha})"


def panel_bg() -> str:
    """The panel surface a chart is drawn on, as a solid hex.

    Also used for marker outlines, whose job is to separate overlapping points
    by painting a sliver of the BACKGROUND between them — a hardcoded dark
    value there draws a near-black halo around every marker on a white panel.
    """
    return "#FFFFFF" if _active_theme() == "light" else "#0F1217"


def rgba(name: str, alpha) -> str:
    """Semantic chart colour for the active theme -> ``rgba()`` string (Pragyam's name)."""
    return chart_rgba(name, alpha)


def diverging_scale(low: str = "rose", high: str = "emerald") -> list:
    """The app's one diverging colourscale, resolved for the ACTIVE theme (from Pragyam).

    THE MIDPOINT IS THE PANEL, not a colour. A cell at zero has nothing to say,
    and the honest way to draw nothing is to let the panel show through — so zero
    disappears and only the cells carrying a claim have ink. NO ALPHA IN THE RAMP:
    both stops are opaque semantic colours; the panel supplies the middle, so the
    scale does not change character with the surface behind it.
    """
    return [[0.0, chart_color(low)], [0.5, panel_bg()], [1.0, chart_color(high)]]


def grid_rgba(alpha: float = 1.0) -> str:
    """A hairline colour that works on BOTH grounds.

    In-plot rules drawn as literal ``rgba(255,255,255,…)`` are white on white
    in Paper mode, i.e. invisible. This returns white-alpha on the dark ground
    and slate-alpha on the light one.
    """
    if _active_theme() == "light":
        return f"rgba(15,23,42,{min(0.9, alpha * 1.6):.3f})"
    return f"rgba(255,255,255,{alpha:.3f})"


def signed_color(value: float, *, pos: str = "emerald", neg: str = "rose") -> str:
    """Theme-aware green/red for a signed number. Replaces hardcoded hex pairs."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return chart_color("slate")
    return chart_color(pos) if v >= 0 else chart_color(neg)


# ── Shared Plotly layout config ─────────────────────────────────────────────

#: Legend. Two things were wrong with the stock arrangement.
#: (1) Anchored top-right — exactly where Plotly puts the modebar, so the
#:     toolbar sat on top of the series names on every hover.
#: (2) Its font dict named a size and family but NO colour, which makes Plotly
#:     fall back to its own default ink rather than inheriting the layout font
#:     — invisible on Paper. The colour is supplied per theme in chart_layout().
#: It now sits BELOW the plot, right-aligned: clear of the toolbar, clear of
#: the y-axis, and reading as a caption to the chart rather than a header.
PLOTLY_LEGEND = dict(
    orientation="h",
    yanchor="top",
    y=-0.16,
    xanchor="right",
    x=1,
    font=dict(size=10, family="JetBrains Mono, monospace"),
    bgcolor="rgba(0,0,0,0)",
    itemsizing="constant",
)

#: Plot margins. `t`/`b` are set per-figure by ``chart_layout`` — a legend needs
#: room to sit in, and a single fixed margin either clips every legended chart
#: or wastes the same space on every chart without one.
PLOTLY_MARGIN = dict(t=28, l=60, r=24, b=38)

# ── The one Plotly config, passed to EVERY st.plotly_chart in the app ────────
# Without this every chart renders Plotly's stock toolbar — including the Plotly
# logo, a link out to plotly.com, and buttons for lasso/box-select that do
# nothing in a read-only research view. It is the one element in the app that
# visibly belongs to another product.
#
# What survives is what a research reader actually uses: zoom, pan, reset, and
# a PNG export named after the app. `displayModeBar="hover"` keeps the toolbar
# out of the composition until the pointer is inside the panel.
PLOTLY_CONFIG = dict(
    displaylogo=False,
    displayModeBar="hover",
    modeBarButtonsToRemove=[
        "lasso2d", "select2d", "autoScale2d", "toggleSpikelines",
        "hoverClosestCartesian", "hoverCompareCartesian", "zoom3d", "pan3d",
        "orbitRotation", "tableRotation", "resetCameraDefault3d",
        "resetCameraLastSave3d", "hoverClosest3d",
    ],
    toImageButtonOptions=dict(format="png", scale=2, filename="sanket-chart"),
    scrollZoom=False,
    doubleClick="reset",
    responsive=True,
)

#: Back-compat alias for anything still importing the old name.
PLOTLY_MODEBAR = PLOTLY_CONFIG


def chart_layout(
    height: int = 360,
    show_legend: bool = True,
    margin: dict | None = None,
    responsive: bool = False,
) -> dict:
    """Return a base Plotly layout dict for the active theme."""
    ct = _chart_theme()
    _margin = dict(PLOTLY_MARGIN)
    if show_legend:
        _margin["b"] = 58        # the legend sits under the x-axis
    else:
        _margin["t"] = 12
    base = dict(
        height=height,
        showlegend=show_legend,
        legend=({**PLOTLY_LEGEND,
                 "font": {**PLOTLY_LEGEND["font"], "color": ct["font_color"]}}
                if show_legend else None),
        # PAINT THE CANVAS, never leave it transparent.
        #
        # A transparent Plotly canvas renders nothing of its own and shows
        # whatever sits behind it, so the chart ground would never actually be
        # chosen by this app — it would be inherited. On a device whose SYSTEM
        # theme is light, any light bleed from the browser or from a Streamlit
        # surface that has not been overridden lands inside the plot area, and
        # Slate renders with pale patches behind dark-theme ink.
        #
        # `panel_bg()` makes the ground explicit AND keeps it appearance-aware,
        # which is the part a blanket "force dark" would get wrong: Paper stays
        # light on a dark-mode device by exactly the mechanism that keeps Slate
        # dark on a light-mode one.
        paper_bgcolor=panel_bg(),
        plot_bgcolor=panel_bg(),
        font=dict(family="JetBrains Mono, monospace", color=ct["font_color"], size=10),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=ct["hover_bg"],
            font=dict(family="JetBrains Mono, monospace", size=11, color=ct["hover_text"]),
            bordercolor=ct["hover_border"],
            align="left",
        ),
        margin=margin or _margin,
        spikedistance=-1,
        # Colourway: any trace that does not name a colour draws from the app's
        # own semantic ramp instead of Plotly's default D3 category-10 (the
        # orange/purple/brown sequence that reads as a different product).
        # Resolved per render, so an unnamed trace follows the theme too.
        colorway=[chart_color(n) for n in
                  ("accent", "cyan", "emerald", "amber", "rose", "violet")],
    )
    if responsive:
        base["autosize"] = True
    return base


#: Axis type. One family, one size, one colour across every plot — the same
#: mono the tables and cards use, at the app's 9px tick / 10px title tiers, so
#: a chart's axis labels are visibly the same kind of text as a table's column
#: headers rather than Plotly's default 12px sans.
_AXIS_TICK_FONT = dict(size=9, family="JetBrains Mono, monospace")
_AXIS_TITLE_FONT = dict(size=10, family="JetBrains Mono, monospace")


def style_axes(fig, y_title: str = "", x_title: str = "", y_range=None, row=None, col=None) -> None:
    """Apply the app's one axis grammar to a Plotly figure."""
    kw = {}
    if row is not None:
        kw["row"] = row
    if col is not None:
        kw["col"] = col

    ct = _chart_theme()
    fig.update_xaxes(
        showgrid=True,
        gridcolor=ct["grid"],
        gridwidth=0.5,
        zeroline=False,
        linecolor=ct["axis_line"],
        title_text=x_title,
        title_font=dict(**_AXIS_TITLE_FONT, color=ct["tick"]),
        tickfont=dict(**_AXIS_TICK_FONT, color=ct["tick"]),
        # Crosshair. Plotly's default renders as a hard white rule across the
        # plot, which is the loudest mark on the panel and belongs to no part of
        # the design system. A crosshair is a pointer, not a series: hairline,
        # dashed, at the theme's own low-alpha spike colour.
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
        spikedash="dash",
        spikecolor=ct["spike"],
        **kw,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=ct["grid"],
        gridwidth=0.5,
        zeroline=True,
        zerolinecolor=ct["grid_zero"],
        zerolinewidth=1,
        linecolor=ct["axis_line"],
        title_text=y_title,
        title_font=dict(**_AXIS_TITLE_FONT, color=ct["tick"]),
        range=y_range,
        tickfont=dict(**_AXIS_TICK_FONT, color=ct["tick"]),
        hoverformat=".2f",
        # NO horizontal spike. A second crosshair arm doubles the ink for a
        # reading the gridlines already give, and in `x unified` hover mode
        # Plotly draws it as a hard opaque rule regardless of the alpha asked
        # for. One dashed vertical crosshair is the whole crosshair.
        showspikes=False,
        # A FIXED standoff between the axis title and its tick labels. Plotly
        # otherwise sets it from each subplot's widest tick label, so a stacked
        # figure whose rows carry different magnitudes ("0.5" vs "-100") puts
        # each row's y-title at a different x.
        title_standoff=14,
        **kw,
    )
    # ── Crosshair, enforced on EVERY x-axis ──────────────────────────────
    # The spike settings above are applied with `row=`/`col=`, which addresses
    # one subplot's axis. On a stacked figure with `shared_xaxes=True` the
    # visible spike is drawn from a DIFFERENT axis object than the ones being
    # updated, so it keeps Plotly's default — an opaque white rule — no matter
    # what the per-row call said. A row-less update writes every x-axis.
    fig.update_xaxes(
        showspikes=True, spikemode="across", spikesnap="cursor",
        spikethickness=1, spikedash="dot", spikecolor=ct["spike"],
    )
    fig.update_yaxes(showspikes=False)

    apply_default_hover(fig)


def apply_default_hover(fig, precision: int = 2) -> None:
    """Give every visible trace a 2-decimal hover, robustly.

    We do NOT rely on a d3 number format inside the hovertemplate
    (``%{y:.2f}``): under ``hovermode="x unified"`` Plotly leaves that format
    UNAPPLIED and the hover leaks full float precision. Instead the values are
    pre-formatted to strings in Python and stashed in ``customdata``, then the
    template just inserts the finished string — no client-side number
    formatting involved, so it cannot be ignored.
    """
    for tr in fig.data:
        if getattr(tr, "hoverinfo", None) == "skip":
            continue
        # Preserve two kinds of intentional template: "%{x…}" (traces that show
        # the X value on hover, in closest mode where d3 formats fine) and
        # "%{customdata…}" (already pre-formatted — keeps this idempotent
        # across the multiple style_axes calls a subplot figure makes).
        _ht = getattr(tr, "hovertemplate", None)
        if _ht and ("%{x" in _ht or "%{customdata" in _ht):
            continue
        y = getattr(tr, "y", None)
        if y is None:
            continue
        cd = []
        for v in y:
            try:
                if v is None or (isinstance(v, float) and v != v):
                    cd.append("—")
                else:
                    cd.append(f"{float(v):.{precision}f}")
            except (TypeError, ValueError):
                cd.append("—")           # non-numeric (category/text) → dash
        try:
            tr.customdata = [[s] for s in cd]
        except (ValueError, TypeError):
            continue
        has_text = getattr(tr, "text", None) is not None
        name = getattr(tr, "name", None)
        if has_text:
            tr.hovertemplate = "%{customdata[0]} · %{text}<extra></extra>"
        elif name:
            tr.hovertemplate = "%{fullData.name}: %{customdata[0]}<extra></extra>"
        else:
            tr.hovertemplate = "%{customdata[0]}<extra></extra>"


# ── Sidebar pop-up guard ─────────────────────────────────────────────────────
# On a screen narrower than 768px Streamlit collapses the sidebar on any press
# outside it, so a tap dismisses the rail. A date picker's calendar is drawn in
# a pop-up layer outside the sidebar's DOM, so the first tap on a day or a month
# arrow counted as "outside" and the rail closed under the user's finger.
# Dropdown menus never had the problem: they stop their own presses before
# Streamlit's document-level listener sees them. This makes every pop-up do the
# same, and only below 768px, where that listener acts. The iframe only hands
# the listener to the page (so it outlives the iframe) and takes no space; the
# CSS rule for `sidebar-popover-guard` lifts it out of the layout.
_POPOVER_GUARD_JS = (  # no double quotes: it is embedded in one below
    "(function(){if(window.__sidebarPopoverGuard)return;window.__sidebarPopoverGuard=true;"
    "document.body.addEventListener('mousedown',function(e){var t=e.target;"
    "if(window.innerWidth<768&&t&&t.closest&&t.closest('[data-baseweb=popover]'))"
    "e.stopPropagation();});})();"
)
_POPOVER_GUARD_HTML = (
    "<script>/* sidebar-popover-guard */(function(){try{var d=window.parent.document;"
    "if(d.getElementById('sidebar-popover-guard'))return;"
    "var s=d.createElement('script');s.id='sidebar-popover-guard';"
    's.textContent="' + _POPOVER_GUARD_JS + '";'
    "d.head.appendChild(s);}catch(e){}})();</script>"
)


def _guard_sidebar_popovers() -> None:
    """Keep the sidebar open while a pop-up drawn from it (a calendar) is used."""
    from streamlit.components.v1 import html as _html

    _html(_POPOVER_GUARD_HTML, height=0)


def inject_css(theme: str = "dark") -> None:
    """Inject the design system into the Streamlit app.

    theme.css defines the canonical DARK token block; when ``theme == "light"``
    a second, small ``:root`` override (``LIGHT_TOKENS``) is appended after it —
    later source wins on identical specificity, so this repaints every component
    without touching a single component rule or the DOM. No runtime
    ``document.documentElement`` attribute toggling involved.

    Injects on every render — Streamlit deduplicates identical <style> blocks.
    """
    if CSS_PATH.exists():
        # Explicit UTF-8: theme.css embeds Devanagari in comments. read_text()
        # with no encoding= falls back to the OS locale encoding, which on many
        # Windows machines is cp1252 — that raises UnicodeDecodeError on the
        # non-ASCII bytes and crashes the app on startup before anything renders.
        css = CSS_PATH.read_text(encoding="utf-8")
    else:
        css = "/* theme.css not found */"

    if theme == "light":
        css += LIGHT_TOKENS

    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    _guard_sidebar_popovers()


# ── The run's phases, and which labels belong to each ────────────────────────
# One table, read by the progress bar so the card can say "Phase 2 of 4 ·
# Data Acquisition" instead of a bare percentage with a free-text label.
#
# Sanket derives the phase from the LABEL, not the percentage. Tattva keys its
# bands on percent because it has one fixed pipeline; Sanket's bar is SHARED —
# the edge study owns 0-35% when it measures and nothing at all when it serves
# a cached one, so the same percentage means different work on different runs.
# The label is the only thing that is accurate in both cases, and every call
# site already passes one.
RUN_PHASES = (
    # the screener
    (1, "Data",           ("loading the bundle", "fetching nse files")),
    (2, "Positioning",    ("scoring positioning",)),
    (3, "Book & Flow",    ("pragyam levels", "checking the f&o ban list", "replaying the book",
                           "reading option flow")),
    (4, "Final Assembly", ("assembling the screen", "analysis complete")),
    # the index desk: the same four phases, named for its work
    (1, "Data",           ("reading the index chains", "loading index history", "reading new sessions")),
    (2, "Readings",       ("participant positions", "india vix", "option chains", "market breadth")),
    (3, "Evidence",       ("verdicts",)),
    (4, "Final Assembly", ("desk ready",)),
)
_N_PHASES = max(n for n, _, _ in RUN_PHASES)


def _phase_of(label: str, pct: int) -> "tuple[int, int, str]":
    """Which phase a progress label belongs to, as ``(n, total, name)``.

    Falls back to the percentage only for a label the table does not know, so a
    new call site degrades to a plausible phase rather than an exception.
    """
    key = str(label).strip().lower()
    for n, name, labels in RUN_PHASES:
        if key in labels:
            return n, _N_PHASES, name
    if pct >= 98:
        return _N_PHASES, _N_PHASES, RUN_PHASES[_N_PHASES - 1][1]
    idx = min(int(pct) // (100 // _N_PHASES), _N_PHASES - 1)
    return RUN_PHASES[idx][0], _N_PHASES, RUN_PHASES[idx][1]


def progress_bar(slot, pct: int, label: str, sub: str = "") -> None:
    """Render the pipeline's progress card into an ``st.empty()`` slot.

    The fill is an ``<i>`` the stylesheet can actually reach, state is carried
    by a class rather than an inlined colour, and width is the only inline
    value — it is the datum. Notably absent: the ``box-shadow`` glow this
    element used to carry, on the one thing every user watches for a full
    minute, in a design system whose stated rule is that nothing glows.
    """
    pct = max(0, min(100, int(pct)))
    is_complete = pct >= 100
    state = " complete" if is_complete else ""
    n, total, phase = _phase_of(label, pct)
    slot.markdown(
        f'<div class="progress-card{state}">'
        f'<div class="progress-phase">Phase {n} of {total}'
        f'<span class="pp-name">{html.escape(phase)}</span></div>'
        f'<div class="progress-label">'
        f'<span class="pulse-dot"></span>{html.escape(str(label))}'
        f'<span class="progress-pct">{pct}%</span>'
        f'</div>'
        + (f'<div class="progress-sub">{html.escape(str(sub))}</div>' if sub else "")
        + f'<div class="progress-track"><i style="width:{pct}%"></i></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def apply_chart_theme(fig) -> None:
    """Apply the app's theme to a Plotly figure (mutates in place)."""
    fig.update_layout(**chart_layout())
    style_axes(fig)


# ── Hover numbers Plotly would print raw ─────────────────────────────────────
# Plotly.js 3.1 (bundled by Streamlit 1.51 and 1.52) rejects a d3 number format
# that pairs the sign flag with a fixed precision — `%{y:+.2f}`, `%{x:+,.2f}`,
# `%{customdata:+.0f}` — and prints the raw float in its place, in every hover
# mode: a tooltip read "Consensus : 0.2295574715461816 · NEUTRAL". Plain formats
# (`.2f`, `,.2f`, `.1%`) are applied. So every signed field is formatted HERE, in
# Python (which reads the same spec), and the template points at the finished
# string in customdata. A numeric x axis with no hover format gets one, so the
# header of a unified hover is "3.755", not "3.754925".
# Called by render_chart_panel on every figure, after all traces are added.
_TPL_FIELD = re.compile(r"%\{([A-Za-z_.]+(?:\[\d+\])?)(?::([^}]*))?\}")
_D3_SPEC = re.compile(r"(?:(.)?([<>=^]))?([+\- ])?([$#])?(0)?(\d+)?(,)?(?:\.(\d+))?(~)?([efgd%])?")


def _fmt_d3(v, spec: str):
    """`v` formatted as d3-format `spec` would, or None when the spec has no Python twin."""
    m = _D3_SPEC.fullmatch(spec)
    if m is None:
        return None
    fill, align, sign, sym, zero, width, comma, prec, trim, typ = m.groups()
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—" if v is None else str(v)
    if f != f:
        return "—"
    if typ is None:                       # d3's default type: shortest, trailing zeros trimmed
        typ, trim, prec = "g", "~", prec or "12"
    if typ == "d":
        typ, prec = "f", "0"
    py = (f"{fill or '' if align else ''}{align or ''}{sign or ''}{'#' if sym == '#' else ''}"
          f"{zero or ''}{width or ''}{comma or ''}{'.' + prec if prec else ''}{typ}")
    try:
        s = format(f, py)
    except ValueError:
        return None
    if trim:
        tail = ""
        if s.endswith("%"):
            s, tail = s[:-1], "%"
        elif "e" in s:
            s, e = s.split("e", 1)
            tail = "e" + e
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        s += tail
    if sym == "$":
        s = s[0] + "$" + s[1:] if s[:1] in "+- " else "$" + s
    return s


def _is_2d(vals) -> bool:
    try:
        first = vals[0]
    except (TypeError, IndexError, KeyError):
        return False
    return hasattr(first, "__len__") and not isinstance(first, str)


def _field_values(tr, name: str):
    """The per-point values a template field reads, or None."""
    if name.startswith("customdata"):
        cd = getattr(tr, "customdata", None)
        if cd is None:
            return None
        m = re.fullmatch(r"customdata\[(\d+)\]", name)
        if m:
            k = int(m.group(1))
            return [row[k] if _is_2d([row]) and len(row) > k else None for row in cd]
        return list(cd)
    obj = tr
    for part in name.split("."):
        obj = getattr(obj, part, None) if obj is not None else None
    if obj is None or isinstance(obj, (str, int, float)):
        return None
    return list(obj)


def _numeric_floats(vals) -> bool:
    """Do these axis values include a non-integer number (not a date, not a category)?"""
    import numbers
    seen = False
    for v in vals[:500]:
        if isinstance(v, bool) or not isinstance(v, numbers.Real):
            return False
        if float(v) != float(v):
            continue
        if float(v) != round(float(v)):
            seen = True
    return seen


def fix_hover_formats(fig) -> None:
    """Pre-format every signed number a hover or text template would print raw."""
    for tr in fig.data:
        tpls = {a: getattr(tr, a, None) for a in ("hovertemplate", "texttemplate")}
        tpls = {a: t for a, t in tpls.items() if isinstance(t, str)}
        fields = []
        for a, t in tpls.items():
            for m in _TPL_FIELD.finditer(t):
                if m.group(2) and "+" in m.group(2) and m.group(0) not in [f[0] for f in fields]:
                    fields.append((m.group(0), m.group(1), m.group(2)))
        if not fields:
            continue
        values = {ph: _field_values(tr, name) for ph, name, _ in fields}
        new = dict(tpls)
        twod = [f for f in fields if values[f[0]] is not None and _is_2d(values[f[0]])
                and not f[1].startswith("customdata")]
        if twod:                                         # heatmap-like: z / text are grids
            cd_free = getattr(tr, "customdata", None) is None
            for ph, name, spec in twod:
                grid = [[_fmt_d3(v, spec) for v in row] for row in values[ph]]
                if any(s is None for row in grid for s in row):
                    continue
                if name == "text":
                    tr.text = grid
                    target = "%{text}"
                elif cd_free:
                    tr.customdata = grid
                    cd_free = False
                    target = "%{customdata}"
                else:
                    continue
                new = {a: t.replace(ph, target) for a, t in new.items()}
            for a, t in new.items():
                setattr(tr, a, t)
            continue
        base = getattr(tr, "customdata", None)
        one_d = False
        if base is None:
            n = next((len(v) for v in values.values() if v is not None), 0)
            rows = [[] for _ in range(n)]
        else:
            lst = list(base)
            if lst and _is_2d(lst):
                rows = [list(r) for r in lst]
            else:
                rows, one_d = [[v] for v in lst], True
        if not rows or len({len(r) for r in rows}) != 1:
            continue
        for ph, name, spec in fields:
            vals = values[ph]
            if vals is None or len(vals) != len(rows):
                continue
            strs = [_fmt_d3(v, spec) for v in vals]
            if any(s is None for s in strs):
                continue
            k = len(rows[0])
            for r, s in zip(rows, strs):
                r.append(s)
            new = {a: t.replace(ph, f"%{{customdata[{k}]}}") for a, t in new.items()}
        if one_d:                                        # customdata became 2-D: column 0 is the old one
            new = {a: re.sub(r"%\{customdata(?=[:}])", "%{customdata[0]", t) for a, t in new.items()}
        tr.customdata = rows
        for a, t in new.items():
            setattr(tr, a, t)
    # A unified hover's header is the axis value, formatted by the axis.
    hm = getattr(fig.layout, "hovermode", None)
    letter = "x" if hm in ("x", "x unified") else "y" if hm in ("y", "y unified") else None
    if not letter:
        return
    refs = {getattr(tr, f"{letter}axis", None) or letter for tr in fig.data if hasattr(tr, f"{letter}axis")}
    for ref in sorted(refs):
        ax = fig.layout[f"{letter}axis{ref[1:]}"]
        if ax.hoverformat or ax.tickformat or ax.type not in (None, "-", "linear"):
            continue
        vals = []
        for tr in fig.data:
            if hasattr(tr, f"{letter}axis") and (getattr(tr, f"{letter}axis", None) or letter) == ref:
                v = getattr(tr, letter, None)
                if v is not None and not isinstance(v, str):
                    vals.extend(list(v)[:200])
        if vals and _numeric_floats(vals):
            ax.hoverformat = ",.3~f"
