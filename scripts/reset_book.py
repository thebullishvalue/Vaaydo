"""
Set where the app's book starts by default, for every visitor and device (the in-app Reset book
saves to the page link only).

    python scripts/reset_book.py 2026-11-20    the book is decided from that session's close
    python scripts/reset_book.py --clear       full model history (the default)

Writes data/book.json; commit it for the deployed app to pick it up. A link with ?book=... still
overrides it in that browser.
"""
import json
import sys
from datetime import date
from pathlib import Path

P = Path(__file__).resolve().parent.parent / "data" / "book.json"

if len(sys.argv) != 2:
    sys.exit(__doc__)
if sys.argv[1] == "--clear":
    P.unlink(missing_ok=True)
    print("book: full model history")
else:
    d = date.fromisoformat(sys.argv[1])
    if d > date.today():
        sys.exit("that date is in the future")
    P.write_text(json.dumps({"start": d.isoformat()}) + "\n")
    print(f"book: from the {d:%d %b %Y} close (data/book.json); commit it to deploy")
