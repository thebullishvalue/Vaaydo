# Changelog

All notable changes to **VAAYDO** (वायदो) are documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

VAAYDO shares its interface with the Pragyam family (Tattva, Nishkarsh, Nivesa, Pragyam,
Arthagati, Samhita, Swing, Sanket): one stylesheet, one set of components, one way a panel,
a table, a chart or a number looks. A change to that shared layer lands in every member.

---

## [7.3.0] — 2026-11-18 — *One family*

VAAYDO's design system already came from Sanket's, file for file; what had not come with it
was the way the family composes a page. Every view was measured against the family's
changelogs and rebuilt where it differed. Nothing the engines compute changed.

### Changed
- **Every table is a panel.** Twenty-seven tables sat bare under a section header, without the
  family's header band. Each now sits in a panel whose band carries its context (universe ·
  as-of date · units, or what the table is), as every chart already did.
- **Numbers read as numbers.** A table of prices, P&L and percentages arrives pre-formatted as
  text (₹ lakh, signs, units), and the table read text as prose: left-aligned in tertiary ink,
  so "1,901.70" and "+₹25,575" looked like labels. A column whose every value is a written
  number is now a numeric column, right-aligned in the primary ink, and a signed one (P&L,
  return, change, net) takes its sign's colour.
- **The tickets lead with what a trader acts on.** Positions read quantity, last, P&L, return,
  stop and risk first, then where the position came from; exits read P&L and return before the
  reason. Order cells carry side, shares and contract ("BUY 325 · FUT 29 Dec 26"); "market,
  next open" and "both at the close" are each table's context, not every row's.
- **Tabs carry their counts**, as Sanket's do: Signals · tonight's orders, Book · positions
  after the next open, Watchlist · names near an entry; Trades · n in a range.
- **Every tab opens with a section header and a one-line subtitle**, and every section title is
  in Title Case ("What It Takes, What to Expect", "By Industry", "Futures Build-Up, Five
  Sessions", "Rejected Myths"). Copy that floated between blocks became the family's note
  under the block it explains, or the subtitle of its section.
- **The appearance switch is the family's**: a Slate / Paper segmented control at the foot of
  the rail, held in a durable session key and read before anything is styled (it was a radio
  in Streamlit's own sans, under the run button, re-run from a widget key).
- **The index picker is a segmented control** (NIFTY · BANKNIFTY) on Gamma & OI, Volatility
  and Futures; it was a radio in Streamlit's own 16px sans.
- **The Long / Short / Exits cards** take the family's tones: the long card's rule and ▲ in
  the long ink, the short card's in the short ink.
- The run button reads RUN DESK for the index desk in either mode (a range read RUN HARVEST).
- The position picker on Book is a third of the width; the leg table sits under a one-line
  summary of the position.

### Fixed
- **Reference-line labels no longer collide**: each opens away from its nearest neighbour
  (the gamma chart printed "flipspot" where the flip sat 65 points from spot).
- The futures build-up chart reads top to bottom in the order its states are listed.
- A missing reading in the Screen's rank column is a dash, explained in a note, not a sentence
  in a numeric column.

### Shared with the family (the fidelity block in `ui/theme.css`)
- **The appearance switch reaches the rail's foot.** The rail's content column was only as
  tall as its content, and Streamlit 1.52 wraps a keyed container in a layout wrapper, so the
  switch stayed under the run button in every app with a short rail.
- A set card keyed by a side's glyph (`side`) inks the glyph in that side's colour.

---

## [7.2.0] — 2026-11-18 — *Audits*
- Coherence, system and UX audits: every leg checked against the exchange files, the simulator
  and the live ledger made identical, one lot per position (no capital sizing), orders-first
  Signals, a ranked watchlist, horizon-labelled flow, rupee range mode, desk wording.
- `panchang`: the input-update routine (holiday list, sector map, data freshness).

## [7.1.0] — 2026-11-18
- NIFTY breadth from NSE's own files; NIFTY evidence at Nishkarsh's standard of inference.

## [7.0.0] — 2026-11-17
- The index desk (NIFTY, BANKNIFTY): chain, gamma, volatility, futures, participants and the
  evidence; F&O ban-list realism.

## [6.2.0] — 2026-11-17
- A Sanket-style F&O trade screener on the family's Graphite design system: order blotter,
  real entries, trailing stops (stop look-ahead fixed).

## [6.0.0] — 2026-11-17
- The option-positioning futures curator.

## [5.0.0] — 2026-11-17
- An evidence-gated F&O desk on real NSE prices.
