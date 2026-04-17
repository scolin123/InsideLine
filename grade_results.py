#!/usr/bin/env python3
"""
grade_results.py — Standalone runner for the InsideLine bet result grader.

Loads credentials from .env, connects to the Google Sheet, and grades all
previous-day plays that have an empty result column (column M).

Usage
─────
  python grade_results.py

Environment variables (set in .env)
─────────────────────────────────────
  ODDS_API_KEY                    Primary Odds API key
  ODDS_API_KEY_2 … ODDS_API_KEY_9 Optional fallback keys
  GOOGLE_SHEETS_CREDS_PATH        Path to the service-account JSON file
  GOOGLE_SHEETS_SPREADSHEET_ID    Google Sheet ID
"""
import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s — %(message)s",
)

# Ensure the project root is importable when run from any working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env before any project imports so credentials are in os.environ
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass   # .env values already exported in the shell environment

from main   import _init_gsheets
from grader import grade_sheet


def main() -> None:
    print("InsideLine — Bet Result Grader")
    print("══════════════════════════════")

    sheet = _init_gsheets()
    if not sheet:
        print(
            "\nERROR: Could not connect to Google Sheet.\n"
            "Check that GOOGLE_SHEETS_CREDS_PATH and "
            "GOOGLE_SHEETS_SPREADSHEET_ID are set in .env."
        )
        sys.exit(1)

    # Build API key pool from environment (primary + up to 8 fallbacks)
    primary = os.environ.get("ODDS_API_KEY", "")
    extras  = [os.environ.get(f"ODDS_API_KEY_{i}", "") for i in range(2, 10)]
    pool    = [k for k in [primary] + extras if k and k != "YOUR_ODDS_API_KEY_HERE"]

    if not pool:
        print(
            "\nERROR: No valid ODDS_API_KEY found in .env.\n"
            "Add your key as ODDS_API_KEY=<your_key>."
        )
        sys.exit(1)

    print(f"API key pool : {len(pool)} key(s)")
    print()

    n = grade_sheet(sheet, pool)

    print(f"\nDone. {n} bet(s) graded.")


if __name__ == "__main__":
    main()
