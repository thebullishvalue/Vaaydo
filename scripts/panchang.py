"""
panchang without Claude: the same input-update routine as .claude/skills/panchang, as one script.

    python scripts/panchang.py                          check, fix what needs no judgement, refresh, re-audit
    python scripts/panchang.py --holidays nse2027.txt   also load NSE's holiday list for a year
    python scripts/panchang.py --check-only             report only, change nothing

--holidays takes any text file with the circular's dates in it (paste the table from NSE's circular
or its PDF text): 26-Jan-2027, 26 January 2027, January 26, 2027 and 2027-01-26 are all read.
Weekend dates are dropped (NSE lists them; they are not sessions anyway). The year's PROVISIONAL rows
are replaced; dates already listed from the files themselves are kept. Name the source with
--source "NSE circular NSE/CMTR/12345" (default: "NSE circular").

It does not commit or push: it prints the git commands to run after reviewing the diff.
Exit code 0 when everything passes and nothing needs a person, 1 otherwise.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HOL = ROOT / "data" / "nse_holidays.csv"
PY = sys.executable
ENV = {**os.environ, "PYTHONPATH": str(ROOT)}

MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
PATTERNS = [
    (r"\b(\d{4})-(\d{2})-(\d{2})\b", lambda g: (int(g[0]), int(g[1]), int(g[2]))),
    (r"\b(\d{1,2})[-\s/]([A-Za-z]{3,9})[-\s/,]+(\d{4})\b", lambda g: (int(g[2]), MONTHS.get(g[1][:3].lower()), int(g[0]))),
    (r"\b([A-Za-z]{3,9})\s+(\d{1,2}),?\s+(\d{4})\b", lambda g: (int(g[2]), MONTHS.get(g[0][:3].lower()), int(g[1]))),
    (r"\b(\d{1,2})[/.](\d{1,2})[/.](\d{4})\b", lambda g: (int(g[2]), int(g[1]), int(g[0]))),   # dd/mm/yyyy
]


def parse_dates(text: str) -> list[pd.Timestamp]:
    found = set()
    for pat, f in PATTERNS:
        for g in re.findall(pat, text):
            y, m, d = f(g)
            if not m:
                continue
            try:
                found.add(pd.Timestamp(year=y, month=m, day=d))
            except ValueError:
                pass
    return sorted(found)


def load_holidays(path: str, source: str) -> str:
    dates = parse_dates(Path(path).read_text(errors="ignore"))
    if not dates:
        return f"ACTION  holidays: no dates found in {path}"
    years = sorted({d.year for d in dates})
    if len(years) != 1:
        return f"ACTION  holidays: {path} spans {years}; give one year's circular at a time"
    yr = years[0]
    wk = [d for d in dates if d.weekday() < 5]
    hol = pd.read_csv(HOL)
    hol["date"] = pd.to_datetime(hol["date"])
    prov = hol["source"].astype(str).str.startswith("PROVISIONAL") & (hol["date"].dt.year == yr)
    keep = hol[~prov & ~((hol["date"].dt.year == yr) & hol["date"].isin(wk))]
    new = pd.DataFrame({"date": wk, "source": source})
    out = pd.concat([keep, new]).sort_values("date")
    out.assign(date=out["date"].dt.strftime("%Y-%m-%d")).to_csv(HOL, index=False)
    gone = sorted(set(hol.loc[prov, "date"]) - set(wk))
    return (f"FIX     holidays: {yr} loaded from {Path(path).name}: {len(wk)} weekday holiday(s), "
            f"{len(dates) - len(wk)} weekend date(s) dropped, {int(prov.sum())} provisional row(s) replaced"
            + (f"; provisional dates NSE did not list: {', '.join(f'{d:%d %b}' for d in gone)}" if gone else ""))


def run(title: str, cmd: list[str]) -> bool:
    print(f"\n== {title}")
    r = subprocess.run(cmd, cwd=ROOT, env=ENV, capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()[-6:]
    print("\n".join("   " + t for t in tail))
    return r.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holidays", help="text of NSE's holiday circular for one year")
    ap.add_argument("--source", default="NSE circular", help="source label written next to each date")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--no-refresh", action="store_true", help="skip fetching new NSE sessions")
    a = ap.parse_args()

    if a.check_only:
        return 0 if run("check", [PY, "-W", "ignore", "scripts/check_inputs.py"]) else 1

    before = HOL.read_text()
    if a.holidays:
        print("\n== holidays\n   " + load_holidays(a.holidays, a.source))

    ok_fix = run("check and fix", [PY, "-W", "ignore", "scripts/check_inputs.py", "--fix"])
    if not a.no_refresh:
        run("refresh NSE sessions", [PY, "-W", "ignore", "scripts/refresh_bundle.py"])

    passed = all([
        run("tests", [PY, "-m", "pytest", "-q"]),
        run("invariants", [PY, "-W", "ignore", "scripts/audit/invariants.py"]),
        run("legs", [PY, "-W", "ignore", "scripts/audit/legs.py"]),
    ])
    clean = run("check again", [PY, "-W", "ignore", "scripts/check_inputs.py"])

    if HOL.read_text() != before:
        print("\n   The holiday list changed: roll dates on open tickets may have moved. Compare the Book "
              "tab's 'roll on' column with yesterday's, or the diff of data/nse_holidays.csv.")

    print("\n== result")
    print("   all audits pass" if passed else "   AUDIT FAILURE: do not commit; see above")
    print("   inputs current" if clean else "   inputs still need a person (ACTION lines above)")
    print("\n   review, then:\n     git diff --stat\n     git add -A && git commit -m \"panchang: inputs updated\" && git push")
    return 0 if (passed and clean and ok_fix is not None) else 1


if __name__ == "__main__":
    sys.exit(main())
