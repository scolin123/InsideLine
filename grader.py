"""
grader.py — Automatic bet result grader for InsideLine.

Reads all previous-day rows with an empty result column (col M) from the
Google Sheet, fetches completed game scores from The Odds API scores endpoint,
applies deterministic grading rules for each market type, and writes
Hit / Miss back to column M.

Public API
──────────
  fetch_scores(sport_key, api_key, days_from=3) → dict
  grade_play(market, side, line_str, home_score, away_score) → "Hit"|"Miss"|None
  grade_sheet(sheet, api_key_pool) → int
"""
from __future__ import annotations

import logging
import datetime as dt
from collections import defaultdict
from typing import Optional

import requests

log = logging.getLogger(__name__)

# ── Sport key constants ───────────────────────────────────────────────────────
SPORT_KEY: dict[str, str] = {
    "NBA": "basketball_nba",
    "MLB": "baseball_mlb",
}


# ── Team name → abbreviation maps ────────────────────────────────────────────
def _get_name_map(league: str) -> dict[str, str]:
    """
    Return the full-name → abbreviation map for the given league.
    Imported lazily to avoid circular imports at module load time.
    """
    if league == "NBA":
        from NBA.config import ODDS_TEAM_NAME_MAP
        return ODDS_TEAM_NAME_MAP
    if league == "MLB":
        from MLB.MLB_config import ODDS_TEAM_NAME_MAP
        return ODDS_TEAM_NAME_MAP
    return {}


# ══════════════════════════════════════════════════════════════════════════════
#  SCORES FETCH
# ══════════════════════════════════════════════════════════════════════════════

def fetch_scores(
    sport_key: str,
    api_key:   str,
    days_from: int = 3,
) -> dict[tuple[str, str], dict]:
    """
    Call The Odds API scores endpoint and return a keyed lookup.

    Parameters
    ----------
    sport_key : str
        e.g. "basketball_nba" or "baseball_mlb"
    api_key : str
        A valid Odds API key.
    days_from : int
        How many completed days back to include (API maximum is 3).

    Returns
    -------
    dict keyed by (away_abbr, home_abbr) →
        {
            "home":      int  | None,
            "away":      int  | None,
            "completed": bool,
        }

    Only games whose team names are in the known map are included.
    Incomplete games are stored with completed=False so callers can
    distinguish "game not finished" from "game not found".
    """
    # Derive league from sport key for team name resolution
    _sport_to_league = {
        "basketball_nba": "NBA",
        "baseball_mlb":   "MLB",
    }
    league   = _sport_to_league.get(sport_key)
    name_map = _get_name_map(league) if league else {}

    url    = f"https://api.the-odds-api.com/v4/sports/{sport_key}/scores/"
    params = {
        "apiKey":     api_key,
        "daysFrom":   days_from,
        "dateFormat": "iso",
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"fetch_scores({sport_key}): HTTP error — {exc}"
        ) from exc

    games  = resp.json()
    result: dict[tuple[str, str], dict] = {}

    for game in games:
        home_full  = game.get("home_team", "")
        away_full  = game.get("away_team", "")
        completed  = game.get("completed", False)
        scores_raw = game.get("scores") or []

        home_abbr = name_map.get(home_full, "")
        away_abbr = name_map.get(away_full, "")

        if not home_abbr or not away_abbr:
            log.debug(
                f"fetch_scores: unmapped team(s) '{home_full}' / '{away_full}' "
                f"— game skipped."
            )
            continue

        key = (away_abbr, home_abbr)

        if not completed or not scores_raw:
            result[key] = {"home": None, "away": None, "completed": False}
            continue

        # Build score lookup: team full name → final score (int)
        score_lookup: dict[str, int] = {}
        for entry in scores_raw:
            try:
                score_lookup[entry["name"]] = int(entry["score"])
            except (KeyError, ValueError, TypeError):
                pass

        home_score = score_lookup.get(home_full)
        away_score = score_lookup.get(away_full)

        if home_score is None or away_score is None:
            log.debug(
                f"fetch_scores: missing score for '{home_full}' or '{away_full}'"
            )
            continue

        result[key] = {
            "home":      home_score,
            "away":      away_score,
            "completed": True,
        }

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  GRADING LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def grade_play(
    market:     str,
    side:       str,
    line_str:   str,
    home_score: int,
    away_score: int,
) -> Optional[str]:
    """
    Apply deterministic grading rules for a single play.

    Parameters
    ----------
    market : str
        "MONEYLINE", "SPREAD", or "TOTAL"  (case-insensitive).
    side : str
        "HOME", "AWAY", "OVER", or "UNDER"  (case-insensitive).
    line_str : str
        The book line as stored in the sheet, e.g. "+1.5", "224.5", "-".
        Ignored for MONEYLINE plays.
    home_score : int
        Final home team score.
    away_score : int
        Final away team score.

    Returns
    -------
    "Hit" | "Miss" | None
        None is returned when the market / side combination is unrecognised
        or when the line cannot be parsed.

    Notes
    -----
    Push (exact tie after spread adjustment) is graded as "Miss" by
    convention — the conservative default when no push-refund rule is known.
    """
    market = market.upper().strip()
    side   = side.upper().strip()

    # ── MONEYLINE ──────────────────────────────────────────────────────────────
    if market == "MONEYLINE":
        if side == "HOME":
            return "Hit" if home_score > away_score else "Miss"
        if side == "AWAY":
            return "Hit" if away_score > home_score else "Miss"
        log.warning(f"grade_play: unrecognised MONEYLINE side '{side}'")
        return None

    # ── TOTAL / TOTALS ─────────────────────────────────────────────────────────
    if market in ("TOTAL", "TOTALS"):
        try:
            line_val = float(line_str.replace("+", "").strip())
        except (ValueError, AttributeError):
            log.warning(f"grade_play: could not parse TOTAL line '{line_str}'")
            return None
        total = home_score + away_score
        if side == "OVER":
            return "Hit" if total > line_val else "Miss"
        if side == "UNDER":
            return "Hit" if total < line_val else "Miss"
        log.warning(f"grade_play: unrecognised TOTAL side '{side}'")
        return None

    # ── SPREAD / RUN LINE ──────────────────────────────────────────────────────
    if market in ("SPREAD", "RUN LINE", "RUNLINE", "RUN-LINE"):
        try:
            line_val = float(line_str.replace("+", "").strip())
        except (ValueError, AttributeError):
            log.warning(f"grade_play: could not parse SPREAD line '{line_str}'")
            return None
        if side == "HOME":
            # Home team covers if: home_score - away_score + line > 0
            return "Hit" if (home_score - away_score + line_val) > 0 else "Miss"
        if side == "AWAY":
            # Away team covers if: away_score - home_score + line > 0
            return "Hit" if (away_score - home_score + line_val) > 0 else "Miss"
        log.warning(f"grade_play: unrecognised SPREAD side '{side}'")
        return None

    log.warning(f"grade_play: unrecognised market '{market}'")
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  SHEET GRADER
# ══════════════════════════════════════════════════════════════════════════════

def grade_sheet(sheet, api_key_pool: list[str]) -> int:
    """
    Read the Google Sheet, find all ungraded plays from previous days,
    fetch scores from The Odds API, grade each play, and batch-write
    Hit / Miss back to column M.

    Sheet column layout (0-indexed)
    ────────────────────────────────
      0  A  Date        (YYYY-MM-DD)
      1  B  Grade       ([A], [B+], etc.)
      2  C  League      (NBA / MLB)
      3  D  Matchup     (AWAY @ HOME)
      4  E  Time
      5  F  Market      (MONEYLINE / SPREAD / TOTAL)
      6  G  Side        (HOME / AWAY / OVER / UNDER)
      7  H  Line        (book line)
      8  I  Proj
      9  J  Margin
     10  K  EV
     11  L  Wager
     12  M  Bet Result  ← written here (Hit / Miss)

    A row is gradeable when:
      • Column A is a valid ISO date strictly before today
      • Column M is empty
      • Column C is NBA or MLB
      • Columns D, F, G are all non-empty

    Parameters
    ----------
    sheet : gspread.Worksheet
        An authenticated worksheet (sheet1 of the InsideLine spreadsheet).
    api_key_pool : list[str]
        One or more Odds API keys tried in order until one succeeds.

    Returns
    -------
    int
        Number of plays successfully graded and written to the sheet.
    """
    today    = dt.date.today()
    all_rows = sheet.get_all_values()   # list[list[str]], 0-indexed rows

    # ── Identify gradeable rows ───────────────────────────────────────────────
    gradeable: list[tuple[int, list[str]]] = []   # (1-based row #, row data)

    for row_idx, row in enumerate(all_rows):
        row_num = row_idx + 1   # gspread uses 1-based addressing

        # Must have at least columns A–H to be actionable
        if len(row) < 8:
            continue

        date_str   = (row[0]  if len(row) > 0  else "").strip()
        league     = (row[2]  if len(row) > 2  else "").strip().upper()
        matchup    = (row[3]  if len(row) > 3  else "").strip()
        market     = (row[5]  if len(row) > 5  else "").strip().upper()
        side       = (row[6]  if len(row) > 6  else "").strip().upper()
        bet_result = (row[12] if len(row) > 12 else "").strip()

        # Skip divider / header rows
        first = date_str.upper()
        if (
            first.startswith("CURRENT")
            or first.startswith("COMPLETED")
            or first.startswith("FUTURE")
            or first.startswith("DATE")
            or not first
        ):
            continue

        # Skip rows that are already graded
        if bet_result:
            continue

        # Skip rows with missing essential fields
        if not league or not matchup or not market or not side:
            continue

        # Skip unknown leagues
        if league not in ("NBA", "MLB"):
            log.debug(
                f"grade_sheet: row {row_num} — unknown league '{league}', skipping."
            )
            continue

        # Parse the date; skip if invalid or not in the past
        try:
            row_date = dt.date.fromisoformat(date_str)
        except ValueError:
            continue

        if row_date >= today:
            continue   # today's or future games aren't final yet

        gradeable.append((row_num, row))

    if not gradeable:
        log.info("grade_sheet: no ungraded rows found.")
        return 0

    print(f"  Found {len(gradeable)} ungraded play(s) to grade.")

    # ── Group by (league, date) to minimise API calls ─────────────────────────
    groups: dict[tuple[str, str], list[tuple[int, list[str]]]] = defaultdict(list)
    for row_num, row in gradeable:
        key = (row[2].strip().upper(), row[0].strip())
        groups[key].append((row_num, row))

    # ── Fetch scores and grade ────────────────────────────────────────────────
    scores_cache: dict[str, dict] = {}   # sport_key → scores dict (fetched once per sport)
    updates:      list[dict]      = []   # gspread batch_update payload

    for (league, date_str), rows_in_group in groups.items():
        sport_key = SPORT_KEY.get(league)
        if not sport_key:
            log.warning(
                f"grade_sheet: no sport key mapping for league '{league}' — skipping group."
            )
            continue

        # Fetch scores once per sport (cache avoids duplicate API calls)
        if sport_key not in scores_cache:
            fetched = False
            for api_key in api_key_pool:
                try:
                    scores_cache[sport_key] = fetch_scores(sport_key, api_key, days_from=3)
                    n_games = sum(
                        1 for v in scores_cache[sport_key].values() if v["completed"]
                    )
                    log.info(
                        f"grade_sheet: fetched {n_games} completed {league} game(s) "
                        f"from Odds API."
                    )
                    fetched = True
                    break
                except RuntimeError as exc:
                    log.warning(f"grade_sheet: API key failed for {sport_key}: {exc}")

            if not fetched:
                print(
                    f"  WARNING: Could not fetch {league} scores "
                    f"— all {len(api_key_pool)} API key(s) failed."
                )
                continue

        game_scores = scores_cache[sport_key]

        for row_num, row in rows_in_group:
            matchup  = (row[3] if len(row) > 3 else "").strip()
            market   = (row[5] if len(row) > 5 else "").strip().upper()
            side     = (row[6] if len(row) > 6 else "").strip().upper()
            line_str = (row[7] if len(row) > 7 else "").strip()

            # Parse "AWAY @ HOME" into abbreviation pair
            if " @ " not in matchup:
                log.warning(
                    f"grade_sheet: row {row_num} — can't parse matchup "
                    f"'{matchup}' (expected 'AWAY @ HOME' format)."
                )
                continue

            away_abbr, home_abbr = [p.strip() for p in matchup.split(" @ ", 1)]
            score_key = (away_abbr, home_abbr)

            if score_key not in game_scores:
                log.debug(
                    f"grade_sheet: row {row_num} — game '{matchup}' not found in "
                    f"{league} scores feed (may still be in progress)."
                )
                continue

            game_data = game_scores[score_key]
            if not game_data.get("completed"):
                log.debug(
                    f"grade_sheet: row {row_num} — game '{matchup}' not yet completed."
                )
                continue

            home_score = game_data["home"]
            away_score = game_data["away"]

            result = grade_play(market, side, line_str, home_score, away_score)

            if result is None:
                log.warning(
                    f"grade_sheet: row {row_num} — could not grade "
                    f"(market={market}, side={side}, line='{line_str}')."
                )
                continue

            print(
                f"    Row {row_num:>4} | {matchup:<14} | "
                f"{market:<10} {side:<5} | "
                f"{away_abbr} {away_score} – {home_abbr} {home_score} "
                f"→ {result}"
            )

            # Column M = column index 13 (1-based) in gspread A1 notation
            updates.append({
                "range":  f"M{row_num}",
                "values": [[result]],
            })

    # ── Batch-write all results in one API call ───────────────────────────────
    if updates:
        sheet.batch_update(updates, value_input_option="RAW")
        log.info(f"grade_sheet: wrote {len(updates)} result(s) to column M.")
    else:
        log.info(
            "grade_sheet: nothing written — all located games may still be in progress."
        )

    return len(updates)
