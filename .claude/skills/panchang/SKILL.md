---
name: panchang
description: VAAYDO's input-update routine. Run when the user says "panchang" (or /panchang) - checks and updates the inputs VAAYDO cannot derive from the nightly NSE files (NSE holiday list, sector map) plus the freshness of every data file, then re-audits and commits to the dev branch.
---

# Panchang: update VAAYDO's hand-fed inputs

The nightly refresh appends NSE's files. A few inputs are not in those files and can go stale
silently. The NSE holiday list is the main one: it sets sessions-to-expiry, and so every roll
date shown on a ticket. This routine finds and fixes them.

## Steps

1. **Check.** From the repo root:
   `VAAYDO_DATA=/home/user/vaaydo-data python3 -W ignore scripts/check_inputs.py --fix`.
   `--fix` applies what needs no judgement:
   - Past weekdays with no NSE file go into `data/nse_holidays.csv` (the files are the authority).
   - The sector map is re-read from NSE's NIFTY 500 list.

   Everything marked ACTION needs you or the user.

2. **Holidays (ACTION lines).** You need NSE's own holiday circular for the year in question
   (equity / F&O segment).
   - First try fetching it with an honest user agent (`vaaydo/7 (end-of-day research)`).
   - Do not call NSE's live API with spoofed browser headers.
   - If you cannot get it, ask the user to paste the list or upload the circular, in one message.

   Then:
   - Replace the `PROVISIONAL` rows for that year with the circular's weekday dates. Use
     source `NSE circular <number or date>`.
   - Never mark a date confirmed without the source in hand. Never copy another project's
     code; dates are facts and are fine.
   - If listed expiries imply a holiday the list lacks, confirm it against the circular
     before adding it.

3. **Data behind.** If an ACTION line says a file is behind, run `python3 scripts/refresh_bundle.py`.

4. **Re-audit.** All of these must pass before committing:
   - `python3 -m pytest -q`
   - `PYTHONPATH=. python3 scripts/audit/invariants.py`
   - `PYTHONPATH=. python3 scripts/audit/legs.py`

   If holidays changed, roll dates can move: say which open positions' "roll on" dates changed.

5. **Re-run the check.** Run `scripts/check_inputs.py` again. Only INFO and OK lines should remain,
   unless the user still owes a source.

6. **Commit and push.** Commit to the session's development branch with the required trailer and
   push there. Push to main only when the user says so.

7. **Report.** In a few lines: what was stale, what changed, and what is still waiting on a source.
