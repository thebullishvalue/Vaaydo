"""
python -m vaaydo sync              download NSE bhavcopies up to today (publish ~18:00 IST)
python -m vaaydo research          extend the evidence with any newly completed cycles
python -m vaaydo playbook          every rule's record and status (live / watch / rejected)
python -m vaaydo scan [--date D]   today's plan from LIVE rules, plus the watchlist
python -m vaaydo mark              mark the journal's open positions, list actions
python -m vaaydo study             positioning (build-up) predictive study

Daily routine, after the close:  sync -> research -> scan -> mark
"""
from __future__ import annotations

import argparse
import warnings
from datetime import date, timedelta

import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 80)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vaaydo")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sync"); s.add_argument("--start", type=date.fromisoformat)
    sub.add_parser("research")
    sub.add_parser("playbook")
    s = sub.add_parser("scan"); s.add_argument("--date", type=date.fromisoformat); s.add_argument("--capital", type=float)
    sub.add_parser("mark")
    sub.add_parser("study")
    a = ap.parse_args(argv)

    from dataclasses import replace
    from .config import DEFAULT
    from .data import bhavcopy, store

    if a.cmd == "sync":
        have = bhavcopy.available_days()
        start = a.start or (have[-1] + timedelta(days=1) if have else bhavcopy.FIRST_UDIFF_DAY)
        bhavcopy.sync(start, date.today(), progress=lambda i, n, d: print(f"\r{i}/{n} {d}", end=""))
        print(f"\nlatest session: {bhavcopy.latest_day()}")
    elif a.cmd == "research":
        from . import research as R
        h = R.update_evidence(progress=lambda i, n, d: print(f"cycle {i + 1}/{n} {d.date()}"))
        print(h.groupby("universe")["entry"].agg(["nunique", "max"]))
    elif a.cmd == "playbook":
        from . import playbook as P, research as R
        t = P.table(R.load_evidence())
        print(t.drop(columns="_rule").round(3).to_string(index=False))
    elif a.cmd == "scan":
        from . import engine, playbook as P, portfolio, research as R
        d = a.date or bhavcopy.latest_day()
        cfg = DEFAULT if not a.capital else replace(DEFAULT, risk=replace(DEFAULT.risk, capital=a.capital))
        rules = P.table(R.load_evidence(), before=pd.Timestamp(d))
        res = engine.scan(d, cfg, rules=rules)
        live = rules[rules["status"] == "live"]
        print(f"VAAYDO · {d} · {len(res.market)} names · {len(live)} live rules · "
              f"{(rules['status'] == 'watch').sum()} on watch\n")
        if live.empty:
            print("No rule has earned live status on out-of-sample evidence. No trades today.\n")
        book = portfolio.construct(res.candidates, store.returns().loc[:pd.Timestamp(d)], cfg)
        if book:
            print(portfolio.plan_frame(book).round(3).to_string(index=False))
            print("\nStress (book P&L at expiry if NIFTY gaps):")
            print(portfolio.stress_table(book, cfg).round(0).to_string())
        if res.watchlist:
            print("Watchlist (NOT traded; rules still earning their record):")
            for c in res.watchlist[:15]:
                e = c.evaluation
                print(f"  {c.symbol:<12} {e.structure.kind:<15} {e.structure.label:<32} {c.reason}")
    elif a.cmd == "mark":
        from . import journal
        print(journal.mark().round(2).to_string(index=False))
    elif a.cmd == "study":
        from . import backtest as B
        print(B.flow_study().round(4).to_string(index=False))


if __name__ == "__main__":
    main()
