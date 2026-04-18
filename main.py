"""
main.py — Unified Entry Point for the NBA + MLB Value Scanner.

Batch-processes every live game on today's slate for BOTH sports:
  Phase 1 — Supabase init (once, shared across both sports)
  Phase 2 — NBA pipeline  (fetch → train → predict → plays)
  Phase 3 — MLB pipeline  (fetch → train → predict → plays)
  Phase 4 — Unified +EV summary table across all sports

Season Toggles
──────────────
Set RUN_NBA / RUN_MLB to False here, or override via .env:
  RUN_NBA=false  RUN_MLB=true  python main.py
"""
import os
import logging
import datetime as dt
from datetime import datetime
from typing import Optional

from supabase import Client as SupabaseClient

try:
    import gspread
    from google.oauth2.service_account import Credentials as _GSCredentials
    _GSPREAD_AVAILABLE = True
except ImportError:
    _GSPREAD_AVAILABLE = False

# ── NBA package imports ───────────────────────────────────────────────────────
# NOTE: files live at NBA/config.py, NBA/insideLine.py, NBA/scanner.py
# Import each file as a module object so run_sport_pipeline can resolve
# symbols via getattr() — this is what eliminates the per-sport name aliases
# and keeps the helper signature to just (sport_name, line_module, config_module, …).
import NBA.config     as nba_config    # NBA/config.py
import NBA.insideLine as nba_line      # NBA/insideLine.py  ← was NBA.splitInsideLine (fixed)
import NBA.scanner    as nba_scanner   # NBA/scanner.py

# ── MLB package imports ───────────────────────────────────────────────────────
# NOTE: files live at MLB/MLB_config.py and MLB/MLB_insideLine.py
import MLB.MLB_config     as mlb_config   # MLB/MLB_config.py
import MLB.MLB_insideline as mlb_line     # MLB/MLB_insideLine.py

log = logging.getLogger(__name__)

# Colour helpers live in the NBA scanner; reused for the unified summary table
_C        = nba_scanner._C
_colorize = nba_scanner._colorize


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER — generic sport pipeline runner
# ══════════════════════════════════════════════════════════════════════════════
def run_sport_pipeline(
    sport_name:    str,
    line_module:   object,              # e.g. nba_line  / mlb_line
    config_module: object,              # e.g. nba_config / mlb_config
    bankroll:      float,
    sb_client:     Optional[SupabaseClient],
) -> list[dict]:
    """
    Generic pipeline runner shared by both the NBA and MLB flows.

    All callables are resolved from the supplied module objects via getattr(),
    so the signature stays at five arguments regardless of how many sports are
    added later.  To add a third sport, call this function with its two module
    references — no signature changes needed anywhere.

    Symbols consumed from ``line_module``
    ──────────────────────────────────────
      OddsClient               — class; instantiated with api_key
      build_pipeline_artifacts — callable() → PipelineArtifacts
      run_pipeline             — callable(home, away, spread, total, ml,
                                          bankroll, live_odds, artifacts)
      save_to_supabase         — callable(sb, matchup, result, game_odds)

    Symbols consumed from ``config_module``
    ────────────────────────────────────────
      ODDS_API_KEY    — str   (overridden by ODDS_API_KEY env var if set)
      EDGE_SPREAD_MIN — float
      EDGE_TOTAL_MIN  — float

    Returns
    ───────
    List of play dicts, each extended with:
      "sport"   : sport_name  (e.g. "NBA" / "MLB")
      "matchup" : "{AWAY} @ {HOME}"
    """
    sep = "═" * 64

    # ── Resolve callables and config from module references ───────────────────
    # Build a key pool: ODDS_API_KEY is primary, ODDS_API_KEY_2 / _3 / etc. are fallbacks.
    _primary_key = os.environ.get("ODDS_API_KEY", getattr(config_module, "ODDS_API_KEY", ""))
    _extra_keys  = [
        os.environ.get(f"ODDS_API_KEY_{i}", "")
        for i in range(2, 10)
    ]
    api_key_pool = [k for k in [_primary_key] + _extra_keys if k and k != "YOUR_ODDS_API_KEY_HERE"]

    edge_spread_min = getattr(config_module, "EDGE_SPREAD_MIN", 1.5)
    edge_total_min  = getattr(config_module, "EDGE_TOTAL_MIN",  0.5)

    OddsClientCls   = getattr(line_module, "OddsClient")
    build_artifacts = getattr(line_module, "build_pipeline_artifacts")
    run_pipeline_fn = getattr(line_module, "run_pipeline")
    save_fn         = getattr(line_module, "save_to_supabase")

    # ── Step 1: Fetch live odds slate (try each key in pool until one succeeds) ─
    print(f"\n{sep}")
    print(f"  {sport_name} VALUE SCANNER  ·  "
          f"{datetime.now().strftime('%A %B %d, %Y  %H:%M')}")
    print(sep)

    odds_client = None
    live_games  = None
    last_exc    = None

    for idx, key in enumerate(api_key_pool):
        try:
            client     = OddsClientCls(api_key=key)
            live_games = client.fetch()
            odds_client = client
            if idx > 0:
                log.info(f"{sport_name}: using fallback API key #{idx + 1} (primary exhausted).")
                print(f"  ⚠  Using fallback Odds API key #{idx + 1} (primary key exhausted).")
            break
        except RuntimeError as exc:
            last_exc = exc
            log.warning(f"{sport_name}: API key #{idx + 1} failed — {exc}")
            if idx + 1 < len(api_key_pool):
                log.info(f"{sport_name}: trying next key in pool …")

    if live_games is None:
        log.error(f"{sport_name} OddsClient: all {len(api_key_pool)} key(s) failed. Last error: {last_exc}")
        log.error(f"Skipping {sport_name} slate entirely.")
        return []

    if not live_games:
        print(f"\n  ✗  No {sport_name} games found in the odds feed right now.")
        print("     The slate may be empty, or the feed hasn't opened yet.\n")
        return []

    print(f"\n  {len(live_games)} game(s) on today's {sport_name} slate:\n")
    for g in live_games:
        total_str  = f"{g.consensus_total}"   if g.consensus_total else "n/a"
        spread_str = f"{g.home_spread:+.1f}"  if g.home_spread     else "n/a"
        home_ml_str = f"{g.home_ml:+d}" if g.home_ml is not None else "n/a"
        away_ml_str = f"{g.away_ml:+d}" if getattr(g, "away_ml", None) is not None else "n/a"
        print(
            f"    {g.away_abv:>3} @ {g.home_abv:<3}  "
            f"Spread {spread_str:<7}  "
            f"Total {total_str:<6}  "
            f"ML(home) {home_ml_str:<6}  "
            f"ML(away) {away_ml_str}"
        )
    print()

    # ── Step 2: Build shared pipeline artifacts ───────────────────────────────
    print(f"  Building {sport_name} pipeline artifacts "
          f"(data load + model training) …\n")
    try:
        artifacts = build_artifacts()
    except Exception as exc:
        log.error(f"Fatal error building {sport_name} pipeline artifacts: {exc}")
        return []

    # ── Step 3: Per-game inference loop ───────────────────────────────────────
    print(f"\n{sep}")
    print(f"  {sport_name} PER-GAME PROJECTIONS")
    print(sep)

    sport_plays: list[dict] = []
    skipped:     list[str]  = []

    for game_odds in live_games:
        home_abv = game_odds.home_abv
        away_abv = game_odds.away_abv
        matchup  = f"{away_abv} @ {home_abv}"

        try:
            # Forward per-side juice attrs that exist on this sport's GameOdds object.
            # NBA uses spread_juice; MLB uses run_line_juice. Both share over/under_juice.
            # Unknown attrs are skipped; run_pipeline defaults to -110 for anything missing.
            _juice = {
                k: v for k, v in {
                    "run_line_juice": getattr(game_odds, "run_line_juice", None),
                    "spread_juice":   getattr(game_odds, "spread_juice",   None),
                    "over_juice":     getattr(game_odds, "over_juice",     None),
                    "under_juice":    getattr(game_odds, "under_juice",    None),
                }.items() if v is not None
            }
            result = run_pipeline_fn(
                        home_team_abv  = home_abv,
                        away_team_abv  = away_abv,
                        market_spread  = game_odds.home_spread,
                        market_total   = game_odds.consensus_total,
                        market_ml_home = game_odds.home_ml,
                        market_ml_away = game_odds.away_ml,
                        bankroll       = bankroll,
                        live_odds      = False,
                        artifacts      = artifacts,
                        **_juice,
                    )

            for play in result.get("plays", []):
                sport_plays.append({
                    "sport":     sport_name,
                    "matchup":   matchup,
                    "game_time": getattr(game_odds, "commence_time", None) or "",
                    **play,
                })

            save_fn(sb_client, matchup, result, game_odds)

        except Exception as exc:
            log.warning(f"  ✗  [{sport_name}] {matchup} — skipped: {exc}")
            skipped.append(f"[{sport_name}] {matchup}")

    if skipped:
        print(f"\n  ⚠  {len(skipped)} {sport_name} game(s) skipped due to errors:")
        for s in skipped:
            print(f"       • {s}")

    # Surface quota info immediately after each sport's fetch
    if hasattr(odds_client, "quota_remaining") and odds_client.quota_remaining is not None:
        print(
            f"\n  {sport_name} Odds API quota: "
            f"{odds_client.quota_remaining} requests remaining "
            f"({odds_client.quota_used} used this billing cycle)."
        )

    return sport_plays


# ══════════════════════════════════════════════════════════════════════════════
#  UNIFIED SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
def _print_unified_summary(
    all_plays:      list[dict],
    bankroll:       float,
    nba_edge_sp:    float,
    nba_edge_tot:   float,
    mlb_edge_sp:    float,
    mlb_edge_tot:   float,
    active_sports:  list[str],          # e.g. ["NBA"], ["MLB"], ["NBA", "MLB"]
) -> None:
    """
    Render a single +EV summary table combining NBA and MLB plays.

    Plays are sorted first by grade (A → B → C), then by descending |edge|.
    The SPORT column differentiates NBA vs. MLB rows visually.

    For SPREAD / RUN-LINE plays the EDGE column shows 'pts' for NBA and
    'runs' for MLB so the unit is always unambiguous.

    ``active_sports`` drives the header label and the per-sport threshold
    footer — toggled-off sports are omitted entirely from both.
    """
    sep          = "═" * 72
    sports_label = " + ".join(active_sports) if active_sports else "—"

    print(f"\n{sep}")
    print(f"  +EV PLAY SUMMARY  ({sports_label})  —  "
          + datetime.now().strftime("%H:%M UTC"))
    print(sep)

    if not all_plays:
        print(f"\n  No +EV plays flagged on today's combined slate.")
        # Only print thresholds for sports that actually ran
        if "NBA" in active_sports:
            print(
                f"  (NBA thresholds  — Spread ≥ {nba_edge_sp} pts  |  "
                f"Total ≥ {nba_edge_tot} pts  |  ML ≥ 3.0%)"
            )
        if "MLB" in active_sports:
            print(
                f"  (MLB thresholds  — Run Line ≥ {mlb_edge_sp} runs |  "
                f"Total ≥ {mlb_edge_tot} runs |  ML ≥ 3.0%)"
            )
        print()
        return

    # Sort: grade A first, then B, then C; within grade by descending |edge|
    _GRADE_ORDER = {"A": 0, "B": 1, "C": 2}
    all_plays.sort(
        key=lambda p: (
            _GRADE_ORDER.get(p.get("grade", "C"), 2),
            -abs(p["edge"]),
        )
    )

    # ── Column widths ─────────────────────────────────────────────────────────
    COL_GRADE   =  7
    COL_SPORT   =  5
    COL_MATCHUP = 14
    COL_TIME    = 20
    COL_TYPE    = 10
    COL_SIDE    =  6
    COL_BOOK    =  8
    COL_PROJ    =  8
    COL_EDGE    = 10
    COL_EV      =  9
    COL_ODDS    =  7
    COL_KELLY   = 12

    _GRADE_COLOR = {"A": _C.GREEN, "B": _C.YELLOW, "C": _C.RED}

    header = (
        f"  {'GRADE':<{COL_GRADE}}"
        f"  {'SPORT':<{COL_SPORT}}"
        f"  {'MATCHUP':<{COL_MATCHUP}}"
        f"  {'TIME':<{COL_TIME}}"
        f"  {'TYPE':<{COL_TYPE}}"
        f"  {'SIDE':<{COL_SIDE}}"
        f"  {'BOOK':>{COL_BOOK}}"
        f"  {'PROJ':>{COL_PROJ}}"
        f"  {'EDGE':>{COL_EDGE}}"
        f"  {'EV($100)':>{COL_EV}}"
        f"  {'ODDS':>{COL_ODDS}}"
        f"  {'KELLY':>{COL_KELLY}}"
    )
    divider = "  " + "─" * (len(header) - 2)
    print(header)
    print(divider)

    for play in all_plays:
        sport       = play.get("sport", "?")
        grade       = play.get("grade", "?")
        grade_color = _GRADE_COLOR.get(grade, _C.RESET)
        grade_str   = _colorize(f"[{grade}]", grade_color)

        # Edge units depend on sport and play type
        if play["type"] == "MONEYLINE":
            edge_str = f"{play['edge'] * 100:+.2f}%"
        elif sport == "MLB":
            edge_str = f"{play['edge']:+.2f} runs"
        else:
            edge_str = f"{play['edge']:+.2f} pts"

        ev_str    = f"${play.get('ev', 0):+.2f}"
        kelly_str = (
            f"${play['kelly_$']:>8,.2f}"
            if grade in ("A", "B")
            else f"{'—':>9}  "
        )

        # BOOK / PROJ columns
        if play["type"] in ("SPREAD", "TOTAL"):
            book_val = play.get("market_line")
            proj_val = play.get("proj_line")
            if play["type"] == "SPREAD":
                book_str = f"{book_val:+.1f}" if book_val is not None else "—"
                proj_str = f"{proj_val:+.1f}" if proj_val is not None else "—"
            else:
                book_str = f"{book_val:.1f}" if book_val is not None else "—"
                proj_str = f"{proj_val:.1f}" if proj_val is not None else "—"
        else:
            book_str = "—"
            proj_str = "—"

        raw_grade_col    = f"[{grade}]"
        grade_col_padded = grade_str + " " * max(0, COL_GRADE - len(raw_grade_col))

        game_time = play.get("game_time", "") or ""

        odds_val = play.get("odds")
        odds_str = f"{odds_val:+d}" if odds_val is not None else "—"

        row = (
            f"  {grade_col_padded}"
            f"  {sport:<{COL_SPORT}}"
            f"  {play['matchup']:<{COL_MATCHUP}}"
            f"  {game_time:<{COL_TIME}}"
            f"  {play['type']:<{COL_TYPE}}"
            f"  {play['side']:<{COL_SIDE}}"
            f"  {book_str:>{COL_BOOK}}"
            f"  {proj_str:>{COL_PROJ}}"
            f"  {edge_str:>{COL_EDGE}}"
            f"  {ev_str:>{COL_EV}}"
            f"  {odds_str:>{COL_ODDS}}"
            f"  {kelly_str:>{COL_KELLY}}"
        )
        print(row)

    print(divider)
    print(
        f"  {_colorize('[A] High Value', _C.GREEN)}   "
        f"{_colorize('[B] Medium Value', _C.YELLOW)}   "
        f"{_colorize('[C] Low / -EV  ', _C.RED)}"
    )
    print()

    # ── Totals block ──────────────────────────────────────────────────────────
    graded_plays = [p for p in all_plays if p.get("grade") in ("A", "B")]
    total_kelly  = sum(p["kelly_$"] for p in graded_plays)
    label_width  = COL_GRADE + COL_SPORT + COL_MATCHUP + COL_TYPE + COL_SIDE + COL_EV + COL_EDGE + 14

    print(f"  {'TOTAL ALLOCATION  (Grade A+B only)':<{label_width}}  ${total_kelly:>8,.2f}")
    if bankroll > 0:
        pct_bankroll = total_kelly / bankroll * 100
        print(f"  {'(% of bankroll)':<{label_width}}  {pct_bankroll:>8.1f}%")
    print()

    # ── Per-sport / per-grade breakdown (only for active sports) ─────────────
    for sport_label in active_sports:
        sport_subset = [p for p in all_plays if p.get("sport") == sport_label]
        if not sport_subset:
            continue
        grade_counts = {
            g: sum(1 for p in sport_subset if p.get("grade") == g)
            for g in ("A", "B", "C")
        }
        game_count = len(set(p["matchup"] for p in sport_subset))
        print(
            f"  {sport_label}: {len(sport_subset)} play(s) across {game_count} game(s)  ·  "
            f"{_colorize(str(grade_counts['A']) + ' Grade A', _C.GREEN)}  "
            f"{_colorize(str(grade_counts['B']) + ' Grade B', _C.YELLOW)}  "
            f"{_colorize(str(grade_counts['C']) + ' Grade C', _C.RED)}"
        )

    print()


# ══════════════════════════════════════════════════════════════════════════════
#  GOOGLE SHEETS  (optional — gracefully skipped when credentials are absent)
# ══════════════════════════════════════════════════════════════════════════════
def _init_gsheets():
    """
    Initialise a gspread Worksheet from env credentials.
    Returns None (with a warning) when credentials are missing or gspread is
    not installed — the rest of the pipeline continues unaffected.
    """
    if not _GSPREAD_AVAILABLE:
        log.warning(
            "gspread / google-auth not installed — Google Sheets write skipped. "
            "Run: pip install gspread google-auth"
        )
        return None

    creds_path = os.environ.get("GOOGLE_SHEETS_CREDS_PATH", "")
    sheet_id   = os.environ.get("GOOGLE_SHEETS_SPREADSHEET_ID", "")

    if not creds_path or not sheet_id:
        log.warning(
            "Google Sheets credentials missing — "
            "GOOGLE_SHEETS_CREDS_PATH / GOOGLE_SHEETS_SPREADSHEET_ID "
            "not set in .env. Sheet write skipped."
        )
        return None

    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds  = _GSCredentials.from_service_account_file(creds_path, scopes=scopes)
        gc     = gspread.authorize(creds)
        sheet  = gc.open_by_key(sheet_id).sheet1
        log.info("Google Sheets client initialised successfully.")
        return sheet
    except Exception as exc:
        log.warning(f"Google Sheets init failed: {exc} — sheet write skipped.")
        return None


def _delete_todays_rows(sheet, run_date: dt.date) -> int:
    """
    Delete the divider row + all play rows written by the most recent run
    for run_date, so a re-run doesn't produce duplicate entries.
    Returns the number of rows deleted.
    """
    today_label = f"CURRENT - {run_date.strftime('%A, %B %d, %Y').upper()}"
    all_rows    = sheet.get_all_values()

    # Find the 1-based row number of today's divider
    divider_1 = None
    for i, row in enumerate(all_rows):
        if row and row[0].strip().upper() == today_label:
            divider_1 = i + 1   # gspread is 1-based
            break

    if divider_1 is None:
        return 0   # no previous run today — nothing to delete

    # Find where today's section ends (start of next top-level divider or end of sheet)
    # "UPCOMING - " sub-headers are part of this run's block, so they are NOT stop markers.
    _PREFIXES = ("CURRENT - ", "COMPLETED - ", "FUTURE - ")
    end_1 = len(all_rows) + 1   # default: one past last row
    for i in range(divider_1, len(all_rows)):   # 0-based index, rows after divider
        cell = (all_rows[i][0] or "").strip().upper()
        if any(cell.startswith(p) for p in _PREFIXES):
            end_1 = i + 1   # 1-based row number of next divider
            break

    n = end_1 - divider_1
    if n > 0:
        sheet.delete_rows(divider_1, divider_1 + n - 1)
    return n


def _game_date_iso(game_time: str, fallback: dt.date) -> str:
    """
    Extract an ISO date string from a formatted game_time like
    'April 17th @ 6:11 PM'.  Falls back to ``fallback`` if parsing fails.
    Handles year rollover (e.g. December games run in January).
    """
    import re as _re
    parts = (game_time or "").split(" @ ")
    if len(parts) < 2:
        return fallback.isoformat()
    date_part = parts[0].strip()                         # e.g. "April 17th"
    clean     = _re.sub(r"(\d+)(st|nd|rd|th)", r"\1", date_part)  # "April 17"
    for year in (fallback.year, fallback.year + 1):
        try:
            return dt.datetime.strptime(f"{clean} {year}", "%B %d %Y").date().isoformat()
        except ValueError:
            continue
    return fallback.isoformat()


def _write_plays_to_sheets(sheet, plays: list[dict], run_date: dt.date) -> None:
    """
    Append a date-divider row followed by one row per play to the Google Sheet.
    Columns (14 total):
      Date | Grade | League | Matchup | Date/Time | Market | Side |
      Line | Odds | Proj. Result | Margin | EV | Wager  (Bet Result written by grader)

    Column A uses the actual game date (parsed from game_time), NOT the run date,
    so future-dated plays land on the correct date in the sheet.

    NOTE: value_input_option="RAW" is intentional — prevents Google Sheets from
    interpreting margin strings like "+0.10 runs" as formulas (#ERROR!) or
    converting "+8.60%" to the decimal 0.086.
    """
    # Remove any rows from a previous run today before appending fresh ones
    n_deleted = _delete_todays_rows(sheet, run_date)
    if n_deleted:
        log.info(f"Google Sheets: cleared {n_deleted} stale row(s) from previous run.")

    import itertools as _itertools
    import re as _re

    # Grade rank for sorting within each date group (best → worst)
    _GRADE_RANK = {"A": 0, "A-dog": 1, "B+": 2, "B": 3, "C": 4}

    def _play_sort_key(p: dict) -> tuple:
        game_date = _game_date_iso(p.get("game_time", "") or "", run_date)
        grade_rank = _GRADE_RANK.get(p.get("grade", ""), 99)
        return (game_date, grade_rank)

    sorted_plays = sorted(plays, key=_play_sort_key)

    # ── Build the main "CURRENT" header (run date) ────────────────────────────
    run_date_iso = run_date.isoformat()
    run_label    = f"CURRENT - {run_date.strftime('%A, %B %d, %Y').upper()}"
    rows: list[list] = [[run_label] + [""] * 13]

    def _date_label(date_iso: str, prefix: str) -> str:
        """Format a date ISO string into a sheet header label."""
        try:
            d        = dt.date.fromisoformat(date_iso)
            raw      = d.strftime("%A, %B %d, %Y").upper()   # "SATURDAY, APRIL 18, 2026"
            day_str  = _re.sub(r"(\w+ )0(\d,)", r"\1\2", raw)  # strip leading zero
        except ValueError:
            day_str = date_iso.upper()
        return f"{prefix} - {day_str}"

    def _build_play_row(play: dict) -> list:
        sport  = play.get("sport", "")
        ptype  = play.get("type", "")
        edge   = play.get("edge", 0.0)
        kelly  = play.get("kelly_$", 0.0)
        grade  = play.get("grade", "")

        game_time_full = play.get("game_time", "") or ""
        date_iso       = _game_date_iso(game_time_full, fallback=run_date)
        time_str       = game_time_full.split(" @ ")[-1] if " @ " in game_time_full else game_time_full

        # Line / Proj. Result
        if ptype == "MONEYLINE":
            line_str = "-"
            proj_str = "-"
        elif ptype == "SPREAD":
            line_val = play.get("market_line")
            proj_val = play.get("proj_line")
            line_str = f"{line_val:+.1f}" if line_val is not None else "-"
            proj_str = f"{proj_val:+.1f}" if proj_val is not None else "-"
        else:  # TOTAL
            line_val = play.get("market_line")
            proj_val = play.get("proj_line")
            line_str = f"{line_val:.1f}" if line_val is not None else "-"
            proj_str = f"{proj_val:.1f}" if proj_val is not None else "-"

        # Margin
        if ptype == "MONEYLINE":
            margin_str = f"{edge * 100:.2f}%"
            if edge >= 0:
                margin_str = f"+{margin_str}"
        elif sport == "MLB":
            margin_str = f"{edge:.2f} runs"
            if edge >= 0:
                margin_str = f"+{margin_str}"
        else:
            margin_str = f"{edge:.2f} pts"
            if edge >= 0:
                margin_str = f"+{margin_str}"

        wager_str = f"${kelly:,.2f}" if kelly > 0 else "-"

        odds_val = play.get("odds")
        odds_str = f"{odds_val:+d}" if odds_val is not None else "-"

        return [
            date_iso,                       # A: game date
            f"[{grade}]",                   # B: grade
            sport,                          # C: league
            play.get("matchup", ""),        # D: matchup
            time_str,                       # E: tipoff time
            ptype,                          # F: market
            play.get("side", ""),           # G: side
            line_str,                       # H: book line
            odds_str,                       # I: american odds
            proj_str,                       # J: proj. result
            margin_str,                     # K: margin
            f"${play.get('ev', 0):+.2f}",  # L: EV
            wager_str,                      # M: wager
        ]

    # ── Group sorted plays by game date, inserting sub-headers ────────────────
    for date_iso, group in _itertools.groupby(
        sorted_plays,
        key=lambda p: _game_date_iso(p.get("game_time", "") or "", run_date),
    ):
        if date_iso > run_date_iso:
            # Future date — insert an UPCOMING sub-header
            rows.append([_date_label(date_iso, "UPCOMING")] + [""] * 13)

        for play in group:
            rows.append(_build_play_row(play))

    sheet.append_rows(rows, value_input_option="RAW")
    log.info("Google Sheets: %d play(s) written.", len(rows) - 1)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    # ── Configuration ─────────────────────────────────────────────────────────
    BANKROLL: float = float(os.environ.get("BANKROLL", 1_000))

    # ── Season Toggles ────────────────────────────────────────────────────────
    # Set to False here to permanently skip a sport, or override at runtime
    # via environment variables without touching this file:
    #
    #   RUN_NBA=false python main.py          ← skip NBA for today
    #   RUN_NBA=false RUN_MLB=false python main.py  ← dry run / off-season
    #
    RUN_NBA: bool = os.environ.get("RUN_NBA", "True").lower() == "true"
    RUN_MLB: bool = os.environ.get("RUN_MLB", "True").lower() == "true"


    # ── Phase 0: Grade previous bets ─────────────────────────────────────────
    # Reads all sheet rows with a blank result column (col M), fetches final
    # scores from The Odds API, and writes Hit / Miss before new bets are added.
    print("\n── Phase 0 · Grading previous bets ─────────────────────────────")
    _primary_key = os.environ.get("ODDS_API_KEY", getattr(nba_config, "ODDS_API_KEY", ""))
    _extra_keys  = [os.environ.get(f"ODDS_API_KEY_{i}", "") for i in range(2, 10)]
    api_key_pool = [k for k in [_primary_key] + _extra_keys if k and k != "YOUR_ODDS_API_KEY_HERE"]
    gs_sheet_grader = _init_gsheets()
    if gs_sheet_grader and api_key_pool:
        try:
            from grader import grade_sheet
            n_graded = grade_sheet(gs_sheet_grader, api_key_pool)
            print(f"  Graded {n_graded} previous bet(s).")
        except Exception as exc:
            log.warning(f"Result grading failed: {exc}")
            print(f"  Result grading skipped: {exc}")
    else:
        print("  Result grading skipped (no sheet credentials or API keys).")

    # ── Phase 1: Supabase — initialised once, shared by both pipelines ────────
    # init_supabase() reads SUPABASE_URL / SUPABASE_KEY from .env.
    # Both sports write to the same Supabase project but separate tables:
    #   NBA → `projections`       MLB → `mlb_projections`
    sb: Optional[SupabaseClient] = nba_line.init_supabase()

    all_plays:     list[dict] = []
    active_sports: list[str]  = []      # tracks which pipelines actually ran

    # ── Phase 2: NBA pipeline ─────────────────────────────────────────────────
    if RUN_NBA:
        nba_plays = run_sport_pipeline(
            sport_name    = "NBA",
            line_module   = nba_line,
            config_module = nba_config,
            bankroll      = BANKROLL,
            sb_client     = sb,
        )
        all_plays.extend(nba_plays)
        active_sports.append("NBA")
    else:
        print("[INFO] NBA Pipeline is toggled OFF. Skipping...")

    # ── Phase 3: MLB pipeline ─────────────────────────────────────────────────
    if RUN_MLB:
        mlb_plays = run_sport_pipeline(
            sport_name    = "MLB",
            line_module   = mlb_line,
            config_module = mlb_config,
            bankroll      = BANKROLL,
            sb_client     = sb,
        )
        all_plays.extend(mlb_plays)
        active_sports.append("MLB")
    else:
        print("[INFO] MLB Pipeline is toggled OFF. Skipping...")

    # ── Phase 4: Unified +EV summary table ───────────────────────────────────
    # Runs safely even when one or both lists are empty — all_plays may be []
    # and active_sports drives the header / threshold footer automatically.
    _print_unified_summary(
        all_plays     = all_plays,
        bankroll      = BANKROLL,
        nba_edge_sp   = getattr(nba_config, "EDGE_SPREAD_MIN", 1.5),
        nba_edge_tot  = getattr(nba_config, "EDGE_TOTAL_MIN",  2.0),
        mlb_edge_sp   = getattr(mlb_config, "EDGE_SPREAD_MIN", 0.4),
        mlb_edge_tot  = getattr(mlb_config, "EDGE_TOTAL_MIN",  0.5),
        active_sports = active_sports,
    )

    # ── Phase 5: Google Sheets ───────────────────────────────────────────────
    gs_sheet = _init_gsheets()
    if gs_sheet is not None and all_plays:
        try:
            _write_plays_to_sheets(gs_sheet, all_plays, dt.date.today())
            print(f"  Google Sheets: {len(all_plays)} play(s) appended.")
        except Exception as exc:
            log.warning(f"Google Sheets write failed: {exc}")
            print("  Google Sheets: write failed (see log).")
    elif not all_plays:
        print("  Google Sheets: no plays to write.")
    else:
        print("  Google Sheets: disabled (credentials not set in .env)")

    # ── Supabase status footer ────────────────────────────────────────────────
    # Show only the tables that were actually written to this run.
    supabase_url = getattr(nba_config, "SUPABASE_URL", "")
    if sb is not None:
        project_id = (
            supabase_url.split("//")[1].split(".")[0]
            if "//" in supabase_url else "project"
        )
        tables_written = []
        if RUN_NBA:
            tables_written.append("`projections` (NBA)")
        if RUN_MLB:
            tables_written.append("`mlb_projections` (MLB)")
        tables_str = " and ".join(tables_written) if tables_written else "none (all pipelines toggled off)"
        print(f"  Supabase: plays persisted to {tables_str}  [{project_id}]")
    else:
        print(
            "\n  Supabase: disabled "
            "(SUPABASE_URL / SUPABASE_KEY not set in .env)"
        )

    print("═" * 72 + "\n")


if __name__ == "__main__":
    main()