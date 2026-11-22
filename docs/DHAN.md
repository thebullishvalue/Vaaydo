# VAAYDO × Dhan: what the data is, and how it will be used

Stable anchor before this work: branch `stable/v7.4.2` (main on 22 Nov 2026). To revert, point main at it.

## 1. Verified, not assumed (22 Nov 2026, against Dhan's API with the owner's account)

| Question | Answer |
|---|---|
| Accuracy | Dhan's NSE futures OHLC equals NSE's own bhavcopy to the paisa (RELIANCE, 55 sessions, open and close). |
| MCX futures history | Daily OHLC, volume and **open interest from Jan 2012**, for every MCX commodity (29 roots pulled: bullion, energy, base metals, agri). |
| Contract structure | `/charts/historical` with a contract's id returns a series **stitched by rank**: the 1st, 2nd and 3rd listed contracts, each back to 2012. `expiryCode` is ignored. The front series holds a contract to its expiry, then switches. |
| Term structure | Real prices of the 1st, 2nd and 3rd contracts on every day since 2012: honest rolls and carry (near-far spread) are computable. |
| Contract identity | Not given; recovered from the data (the new front's OI and price continue yesterday's 2nd contract). Recovered expiries follow MCX's calendar (gold mini ~5th, crude ~19th, silver mini and copper month-end, natural gas ~25th); a rules calendar anchors them. |
| MCX option chain (live) | Works: every strike with OI, volume, IV, greeks, bid/ask (crude: 194 strikes). |
| Expired options history | `/charts/rollingoption` rejects every request, including the documented example verbatim, for NSE and MCX. Not usable. NSE option history is already complete in VAAYDO from NSE's bhavcopy; MCX option history will be built by recording the live chain daily. |
| Lot size | The instrument master gives MCX lots as `1`; the quantity per lot (1 kg, 100 bbl, ...) comes from MCX's contract specifications (a hand-fed input, checked by `panchang`). |
| Tokens | 24-hour validity. `RenewToken` expires the current token and returns a new one, only while it is active. Order APIs need a registered static IP; data APIs do not. |
| Terms | Dhan's terms forbid public display, redistribution and sub-licensing of platform data. |

## 2. Architecture decisions

1. **NSE stays the source of record for F&O Stocks and Indices.** Public, audited, reproducible, and free to display.
   Dhan is the second source, for what NSE's files cannot give: MCX first; then new data points for the NSE
   universes (live IV and greeks, intraday open interest and prices) **only after a preregistered test shows they help**.
2. **Dhan data never enters this public repository.** It lives in a private store (a private GitHub repository
   the app reads with a read-only token from Streamlit secrets). Screens built from Dhan data sit behind a login;
   F&O Stocks and Indices stay public.
3. **No PIN or TOTP secret is stored anywhere.** A nightly job renews the token (`RenewToken`) and writes the new one
   back to its own secret (a fine-grained GitHub token scoped to the private repository's secrets). A missed night
   means one manual token from Dhan Web.
4. **MCX is not NSE**, and the book's rules follow MCX:
   * the session runs to ~23:30 IST: decisions on the settled close, the nightly job after MCX closes, fills at the
     09:00 open;
   * bullion and base metals are compulsorily deliverable: positions roll **before the tender period**, per
     commodity, not at NSE's T−3;
   * one lot of the **mini** contracts (GOLDM, SILVERM, CRUDEOILM, NATGASMINI) by default: a big gold lot is ~₹1 Cr+;
   * costs: CTT, MCX fees, GST, stamp duty;
   * MCX's own holiday calendar (evening sessions on some NSE holidays);
   * crude oil's negative settlement (April 2020) is a real event the engine must survive.
5. **Edge before screens.** VAAYDO trades only what passes a preregistered test (docs/PREREGISTRATION.md). Commodities
   cannot use the stock edge (a cross-section of ~220 names); their hypotheses are time-series and term-structure.

## 3. Build order (each step a gate)

| Step | What | Gate |
|---|---|---|
| 1 | Verify the data | **Passed** (section 1) |
| 2 | Contract identity, MCX calendar, contract specs, costs; research panel | Every recovered expiry within MCX's rule; every roll priced from a real contract |
| 3 | Preregister and test commodity hypotheses (trend, carry, OI build-up), 2012–2026 | Nishkarsh standard; out of sample; after costs |
| 4 | Private store, Dhan client, nightly job with token renewal; daily MCX option-chain record | Built (`vaaydo/dhan/`, vault workflow); live once on main with the Dhan secrets; gate: unattended for 5 sessions |
| 5 | Commodities universe in the app (login), book or paper per step 3 | Same audits as stocks: legs vs data, engines identical |
| 6 | Dhan data points for the NSE universes (intraday stops and fills, live IV), each tested first | Preregistered improvement or not shipped |

## 4. Needed from the owner (step 4)

* A **private** GitHub repository for Dhan data (e.g. `vaaydo-vault`), added to Claude's sessions.
* Streamlit Cloud secrets: a read-only token for that repository, and a password for the login.
* The Dhan token as a secret of that repository, plus a fine-grained GitHub token allowed to update its secrets.

## 5. How the nightly collection proves identity (step 4)

Dhan's daily history for a contract id is stitched by the contract's current rank, and its quote endpoint is
per real contract but its "close" is not the official close for every commodity (crude showed the previous
session's). So the collector takes official bars from history and keeps a row only if it is provably the
contract's own: the latest session when its open, high, low and volume equal the contract's quote; an earlier
session only when it falls after the commodity's latest expiry (known from the vault's own contract record).
Tested live (22 Nov 2026): 63 contracts verified across 24 commodities; crude's 19 Nov row, which history
files under December but belongs to the expired November contract, is rejected.
