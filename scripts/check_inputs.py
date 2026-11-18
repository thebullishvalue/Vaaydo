"""
The inputs VAAYDO cannot derive from the nightly NSE files, and whether each is current.

    python scripts/check_inputs.py          report
    python scripts/check_inputs.py --fix    also apply the fixes that need no judgement:
                                            past weekday holidays read off the files themselves,
                                            and the sector map re-read from NSE's NIFTY 500 list

Each line is OK, INFO, FIX (applied with --fix) or ACTION (needs a person or a source this
script does not read, e.g. NSE's holiday circular for next year). Exit code 1 if any ACTION.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"
HOL = DATA / "nse_holidays.csv"
FIX = "--fix" in sys.argv
out: list[tuple[str, str]] = []


def say(level: str, msg: str) -> None:
    out.append((level, msg))


today = pd.Timestamp.today().normalize()
F = pd.read_parquet(DATA / "futures.parquet", columns=["date", "symbol", "expiry", "open", "close", "lot"])
F["date"] = pd.to_datetime(F["date"])
F["expiry"] = pd.to_datetime(F["expiry"])
last = F["date"].max()
files = set(F["date"].unique())

# 1. holidays ------------------------------------------------------------------------------
hol = pd.read_csv(HOL)
hol["date"] = pd.to_datetime(hol["date"])
prov = hol[hol["source"].astype(str).str.startswith("PROVISIONAL")]
listed = set(hol["date"])

# past: weekdays with no NSE file are holidays; the files are the authority
wk = pd.bdate_range(F["date"].min(), last)
gaps = sorted(d for d in wk if d not in files and d not in listed)
if gaps:
    say("FIX", f"holidays: {len(gaps)} past weekday(s) with no NSE file not in the list: "
               + ", ".join(f"{d:%d %b %Y}" for d in gaps[:8]))
    if FIX:
        hol = pd.concat([hol, pd.DataFrame({"date": gaps, "source": "session gap in NSE files"})])
traded = sorted(d for d in listed if d in files)
if traded:
    say("ACTION", "holidays: listed as holidays but NSE published a session: "
                  + ", ".join(f"{d:%d %b %Y}" for d in traded) + " (remove them)")

# ahead: provisional dates, and whether next year exists at all
soon = prov[prov["date"] <= today + pd.Timedelta(days=120)]
if len(soon):
    say("ACTION", f"holidays: {len(soon)} PROVISIONAL date(s) within 120 days; replace with NSE's circular")
elif len(prov):
    say("INFO", f"holidays: {len(prov)} provisional date(s) for {sorted(prov['date'].dt.year.unique())}; "
                "replace when NSE publishes its circular (usually December)")
nxt = today.year + 1
if today.month >= 11 and not (hol["date"].dt.year == nxt).any():
    say("ACTION", f"holidays: nothing listed for {nxt}; NSE's circular is due")

from vaaydo import nse_calendar  # noqa: E402
imp = {pd.Timestamp(d) for d in nse_calendar.implied_by_expiries(F.loc[F["expiry"] > last, "expiry"].unique())}
miss = sorted(d for d in imp if d not in listed)
if miss:
    say("ACTION", "holidays: listed expiries imply holiday(s) the list lacks: "
                  + ", ".join(f"{d:%d %b %Y}" for d in miss) + " (check NSE's circular, then add)")
if not (gaps or traded or len(soon) or miss):
    say("OK", f"holidays: {len(hol)} dates, future ones to {hol['date'].max():%d %b %Y}")

if FIX and gaps:
    hol.sort_values("date").assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d")).to_csv(HOL, index=False)

# 2. sector map --------------------------------------------------------------------------
S = pd.read_parquet(DATA / "sectors.parquet")
live = set(F.loc[F["date"] == last, "symbol"])
nosec = sorted(live - set(S["symbol"]))
if nosec:
    say("FIX" if FIX else "INFO", f"sectors: {len(nosec)} F&O stock(s) with no industry: {', '.join(nosec[:12])}"
                                  + (" ..." if len(nosec) > 12 else ""))
    if FIX:
        try:
            from vaaydo import bundle
            new = bundle.sectors()
            keep = S[~S["symbol"].isin(new["symbol"])]          # names that left the NIFTY 500 keep their old industry
            pd.concat([new, keep], ignore_index=True).to_parquet(DATA / "sectors.parquet", index=False)
            left = sorted(live - set(new["symbol"]) - set(keep["symbol"]))
            if left:
                say("ACTION", f"sectors: still none after re-reading NIFTY 500 (outside it): {', '.join(left)}")
        except Exception as e:  # network
            say("ACTION", f"sectors: could not re-read the NIFTY 500 list ({e})")
else:
    say("OK", "sectors: every F&O stock has an industry")

# 3. renames (informational: detected automatically) ---------------------------------------
from vaaydo import renames  # noqa: E402
recent = F[F["date"] >= last - pd.Timedelta(days=60)]
rn = renames.detect(recent)
say("INFO" if rn else "OK", "renames: " + (", ".join(f"{a} -> {b}" for a, b in rn.items()) + " in the last 60 days "
                                           "(handled automatically)" if rn else "none in the last 60 days"))

# 4. the nightly files are in step ------------------------------------------------------------
for f in ["opi_inputs", "futures", "pragyam", "index_panel", "index_futures", "participant_oi", "banlist", "vix"]:
    p = DATA / f"{f}.parquet"
    if not p.exists():
        say("ACTION", f"data: {f}.parquet missing")
        continue
    d = pd.to_datetime(pd.read_parquet(p, columns=["date"])["date"]).max()
    if f == "banlist":
        ok = d >= last          # the ban list is read for the next session
    else:
        ok = d == last
    if not ok:
        say("ACTION", f"data: {f} ends {d:%d %b %Y}, futures end {last:%d %b %Y}; run scripts/refresh_bundle.py")
b = pd.read_parquet(DATA / "breadth_nse.parquet", columns=["DATE"])
bd = pd.to_datetime(b["DATE"]).max()
if bd < last:
    say("ACTION", f"data: breadth_nse ends {bd:%d %b %Y}; run scripts/refresh_bundle.py")
lag = len(pd.bdate_range(last + pd.Timedelta(days=1), today - pd.Timedelta(days=1)))
say("INFO" if lag <= 1 else "ACTION", f"data: last session {last:%d %b %Y} ({lag} weekday(s) behind yesterday)")

w = max(len(l) for l, _ in out)
for l, m in out:
    print(f"{l:<{w}}  {m}")
sys.exit(1 if any(l == "ACTION" for l, _ in out) else 0)
