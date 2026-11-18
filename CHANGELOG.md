# Changelog

All notable changes to **VAAYDO** (वायदो) are documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

VAAYDO shares its interface with the Pragyam family (Tattva, Nishkarsh, Nivesa, Pragyam,
Arthagati, Samhita, Swing, Sanket): one stylesheet, one set of components, one way a panel,
a table, a chart or a number looks. A change to that shared layer lands in every member.

---

## [7.3.5] — 2026-11-18 — *A progress bar that moves*

An audit of the run's progress card against every member's, step by step on real runs.

### Fixed
- **The bar moves on the first run of the hour.** A run of the latest session, and every index
  desk run, was cached by decorating the engine, and a decorated engine cannot take the progress
  callback: the card sat on "loading the bundle · 3%" (the desk on "reading the index chains ·
  5%") for the whole computation, then jumped to 100%. Finished runs are now kept by hand, for
  the hour as before, so the first run reports every step and a repeat is instant. A past date is
  kept the same way (it was recomputed on every click).
- **Every step has the family's detail line**, saying what the step is working on: "Fetching
  NSE Files · F&O bhavcopy · the sessions the bundle lacks", "Replaying the Book · fills, rolls,
  stops · one lot each", "Option Chains · OI by strike · dealer gamma · IV smile". The card was
  the family's without its third line (78px against every member's 97).
- **The desk's steps run forward.** "Verdicts · 90%" was reported before "Market Breadth ·
  85%"; it now follows the breadth read, where its work is.
- **The phases name the work and never step back.** The desk's steps were unknown to the phase
  table, so its card guessed a phase from the percentage; it now reads Data → Readings →
  Evidence → Final Assembly. The screener's ban-list check (55%) was filed under Positioning,
  after the book replay (45%) had already moved the card to phase 3; it is part of Book & Flow.
  A phase no step reported ("reading option chains") is gone.
- A table's label column is never re-read as a numeric column, whatever its values; the rule
  now goes to every family member.

---

## [7.3.4] — 2026-11-18 — *Paper first*

### Changed
- **Paper is the first-run appearance**, as in Tattva, Nishkarsh, Nivesa, Pragyam, Arthagati,
  Samhita and Swing; the switch reads PAPER · SLATE. VAAYDO had inherited Sanket's Slate-first
  order. A choice once made is kept for the session, as before. Streamlit's native layer stays
  on the dark base, as in every member: the stylesheet is built Slate-first and Paper's tokens
  reclaim the natives.

---

## [7.3.3] — 2026-11-18 — *The index picker without its label*

### Changed
- **The NIFTY · BANKNIFTY picker has no visible label**, as no segmented control in the family
  does (the chart-window strip, the appearance switch). The options name themselves, and every
  chart under the picker names its index. Shown, the "INDEX" label sat inside the control's
  bordered track, so the box carried an empty band above and to the right of the buttons; the
  track now wraps the two buttons alone. The label is kept for screen readers.

---

## [7.3.2] — 2026-11-18 — *Sides in their ink, positions by the day they opened*

An audit of every table for the order its rows come in and for how it says a side.

### Changed
- **Positions newest first, by the day each was opened.** The book came in the engine's order:
  entries, then every long A–Z, then every short A–Z, so a position opened yesterday sat
  wherever its letter put it (and the subtitle claimed "the positions held longest"). Every
  position-like table now reads newest first: tomorrow's entries, then each position by its
  entry date, the stronger signal first on a tie. The orders to enter, exit and roll, and the
  contract-legs picker, follow the same rule. Long and short are no longer grouped; the side is
  told by its ink. The engine's own order (and the CSV exports) are unchanged.
- **A side in its ink, written one way.** ▲ LONG in the long ink and ▼ SHORT in the short ink,
  BUY and SELL likewise in the order cells (both words in a roll), and the ▲ / ▼ of every state
  (▲ NEW, READY ▼) in Option flow and the Screen. Only the token is inked, and the glyph and the
  word stay, so a side is never told by colour alone; P&L keeps its own sign colour. The side
  was written four ways ("▲ LONG", "LONG", "long ▲", "long"); it is "▲ LONG" / "▼ SHORT" in
  every table, the index desk's paper side included.
- **Stopped out: most recent stop first** (it came in the screen's order). By Industry says
  its order (largest net first).

### Fixed
- **One minus sign.** Rupee amounts printed "−" (U+2212) beside percentages, prices and points
  printing "-", in the same rows; every number now uses one.
- **The evidence table in its registered order** (H1, H1b, H2 … H10, H11, H12): read as text it
  ran H1, H10, H11, H12, H1b, H2.
- **NIFTY before BANKNIFTY** in the paper-candidate and premium-selling tables, as everywhere
  else on the desk (alphabetical put BANKNIFTY first).

---

## [7.3.1] — 2026-11-18 — *The family's cold start*

7.3.0 brought the analysis pages into the family and left the landing page as it was; set
beside the others it was plainly a different product's front door. It is now built the way
Sanket's, Tattva's and Nishkarsh's are.

### Changed
- **The masthead.** The page opened straight on its claim; it now opens, as every member's
  does, on the wordmark (VAA·YDO in the accent) over its mono subtitle and a rule: "F&O Trade
  Screener · Option Positioning · Stock Futures".
- **One line on what to do**: "Pick a universe, a mode and a date in the rail, then Run." The
  two universes are now the first two Coverage cards.
- **Three sections, the family's three.** Coverage (F&O stocks, the index desk, history),
  System (signal, risk, book) and What a Run Returns (tonight's orders, the book, the market
  around it, the record). The five-card "What It Takes" strip that led the page is gone as a
  section: its numbers are the system panels' spec rows (the record and the worst drawdown
  computed from the track record, as before), so the three panels carry four rows each and
  stand at one height.
- **The family's footer**: "© 2026 VAAYDO · @thebullishvalue · v7.3.1 · the time drawn, IST"
  on every page. The line it replaced ("decision support, not investment advice") is a note
  under System Data's definitions.
- A failed run is reported under the masthead on the cold start and under the command bar on
  a loaded page, as in Sanket (it printed above everything).

---

## [7.3.0] — 2026-11-18 — *One family*

VAAYDO's design system already came from Sanket's, file for file; what had not come with it
was the way the family composes a page. Every view was measured against the family's
changelogs and rebuilt where it differed, then scanned as the family is: every page and tab
at 1440, 834 and 390px in Slate and Paper for clipped, overlapping or unreadable text, every
chart's tooltips for raw numbers, and every text element for its face, size and case. Nothing
the engines compute changed.

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
- **Charts on Paper have readable axes.** No chart was given the family's axis grammar
  (`style_axes`), so every axis kept Plotly's dark-theme defaults: on Paper the tick labels were
  near-white on white and the gridlines heavy black. All nineteen charts now take it: hairline
  grids, the type scale's ticks, the crosshair, in both appearances.
- **Reference-line labels no longer collide.** A label whose line sits within an eighth of the
  axis of the previous one drops a row, so max pain, the gamma flip and spot stack instead of
  printing over each other ("flipspot" at full width, "max pain" over "gamma flip" on a tablet).
- The track record's shaded validation period is labelled "validation period"; what it means
  moved to the section's subtitle (the long label ran 149px past the plot on a phone).
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
