"""Append sessions NSE has published since the bundle ends to every file in data/
(option-positioning inputs, futures prices, Pragyam levels). Run after ~18:30 IST.

    python scripts/refresh_bundle.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vaaydo import bundle  # noqa: E402

bundle.refresh()
