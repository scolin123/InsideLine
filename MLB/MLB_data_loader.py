"""
mlb_data_loader.py — MODULE 1: MLBDataLoader
Fetches raw game-log data from the MLB Stats API (statsapi) for the specified
seasons and returns a unified DataFrame with team + opponent box-score columns,
starting-pitcher metrics, bullpen fatigue, wOBA, ISO, and park-factor features.

Each row represents a single Team-Game and is self-contained: it carries both
the team's own statistics and its opponent's mirrored statistics (OPP_ prefix),
making it immediately usable for model training or prediction.

Dependencies
------------
    pip install mlb-statsapi pandas python-dotenv

Key design decisions
--------------------
  * Plate Appearances (PA) replace NBA Possessions as the primary rate
    denominator for offensive statistics.
  * Starting-pitcher identity, handedness, and recent-start ERA / WHIP / K9 /
    FIP windows replace individual-player load-bearing in the NBA version.
  * Bullpen fatigue (total relief pitches in the last 3 calendar days) replaces
    the travel-miles fatigue metric.
  * Park factors are joined from mlb_config.PARK_FACTORS so every run-scoring
    metric can be park-adjusted downstream.
"""

import math
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import statsapi

from .MLB_config import (
    SEASONS,
    API_SLEEP,
    BALLPARK_COORDS,
    PARK_FACTORS,
    WOBA_WEIGHTS,
    FIP_CONSTANT,
    SP_RECENT_STARTS_SHORT,
    SP_RECENT_STARTS_LONG,
    BULLPEN_FATIGUE_DAYS,
    STATSAPI_TEAM_ABV_MAP,
    STATSAPI_TEAM_ID_MAP,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Return numerator / denominator, or `default` when denominator is zero."""
    return numerator / denominator if denominator else default


def _haversine(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Great-circle distance in miles between two (lat, lon) coordinate pairs."""
    R = 3_958.8
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlat, dlon  = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class MLBDataLoader:
    """
    Fetches and enriches MLB game-log data from the MLB Stats API.

    Usage
    -----
        loader = MLBDataLoader()
        df     = loader.load()   # returns enriched team-game DataFrame

    The returned DataFrame contains one row per (team, game) pair and includes:
      - Raw box-score counts (R, H, HR, BB, SO, AB, PA, …)
      - Derived rate stats (wOBA, ISO, K%, BB%, OBP, SLG)
      - Starting-pitcher features (ERA_5, ERA_10, WHIP_5, WHIP_10,
                                   K9_5, K9_10, FIP_5, FIP_10, SP_HAND,
                                   SP_DAYS_REST)
      - Bullpen fatigue (BULLPEN_PITCHES_3D)
      - Opponent mirror columns (OPP_* prefix)
      - Pace / game-shape features (PACE_MATCHUP, PROJ_TOTAL_RAW)
      - Park factor (PARK_FACTOR) for the home team's stadium
    """

    def __init__(
        self,
        seasons:       list[str] = SEASONS,
        sleep:         float     = API_SLEEP,
        force_refresh: bool      = False,
        cache_dir:     str       = "data/mlb_cache",
        max_workers:   int       = 8,
    ) -> None:
        self.seasons       = seasons
        self.sleep         = sleep
        self.force_refresh = force_refresh
        self.cache_dir     = Path(cache_dir)
        self.max_workers   = max_workers
        self._raw: Optional[pd.DataFrame] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self) -> pd.DataFrame:
        """
        Orchestrate the full pipeline:
          1. Fetch raw game-level box scores for every season.
          2. Attach opponent-side mirror columns.
          3. Attach starting-pitcher metrics and handedness.
          4. Attach SP days-rest and bullpen fatigue.
          5. Attach park factors.

        Returns
        -------
        pd.DataFrame
            Enriched team-game log, one row per (team, game).
        """
        frames: list[pd.DataFrame] = []
        for season in self.seasons:
            log.info(f"Fetching season {season} …")
            df = self._fetch_season(season)
            if df is not None and not df.empty:
                frames.append(df)

        if not frames:
            raise RuntimeError("No data returned from MLB Stats API. Check connectivity.")

        combined = (
            pd.concat(frames, ignore_index=True)
              .drop_duplicates(subset=["GAME_ID", "TEAM_ID"])
              .sort_values(["GAME_DATE", "GAME_ID"])
              .reset_index(drop=True)
        )

        combined = self._attach_opponent_stats(combined)
        combined = self._attach_starting_pitcher_metrics(combined)
        combined = self._attach_pitcher_rest_and_bullpen_fatigue(combined)
        combined = self._attach_park_factors(combined)

        self._raw = combined
        log.info(f"MLBDataLoader complete — {len(combined):,} team-game rows loaded.")
        return combined

    @property
    def raw(self) -> pd.DataFrame:
        """Return the last DataFrame produced by `.load()`. Raises if not yet loaded."""
        if self._raw is None:
            raise RuntimeError("Call .load() first.")
        return self._raw

    # ── Private: caching helpers ──────────────────────────────────────────────

    def _cache_path(self, season: str) -> Path:
        return self.cache_dir / f"{season}.parquet"

    @staticmethod
    def _is_season_complete(season: str) -> bool:
        """True when the season year is strictly before the current calendar year."""
        return int(season) < datetime.utcnow().year

    def _load_from_cache(self, season: str) -> Optional[pd.DataFrame]:
        path = self._cache_path(season)
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
            # Backfill TEAM_ABV from TEAM_ID if the column is missing or all empty.
            if "TEAM_ABV" not in df.columns or df["TEAM_ABV"].eq("").all():
                df["TEAM_ABV"] = df["TEAM_ID"].map(STATSAPI_TEAM_ID_MAP).fillna("")
                log.info(f"  Season {season}: backfilled TEAM_ABV from TEAM_ID.")

            # Backfill derived columns that were missing from older caches.
            if "TEAM_ERA" in df.columns:
                df["TEAM_ERA"] = pd.to_numeric(df["TEAM_ERA"], errors="coerce").fillna(4.50)
            if "RA" not in df.columns and "TEAM_RUNS_ALLOWED" in df.columns:
                df["RA"] = df["TEAM_RUNS_ALLOWED"]
            if "BABIP" not in df.columns:
                babip_den = (df["AB"] - df["SO"] - df["HR"] + df["SF"]).clip(lower=1)
                df["BABIP"] = ((df["H"] - df["HR"]) / babip_den).round(3)
            ip_safe = df.get("TEAM_IP", pd.Series(9.0, index=df.index))
            ip_safe = pd.to_numeric(ip_safe, errors="coerce").fillna(9.0).clip(lower=0.1)
            if "P_ERA" not in df.columns and "TEAM_ERA" in df.columns:
                df["P_ERA"] = df["TEAM_ERA"]
            if "P_WHIP" not in df.columns and "TEAM_H_ALLOWED" in df.columns:
                df["P_WHIP"] = ((df["TEAM_H_ALLOWED"] + df["TEAM_BB_ALLOWED"]) / ip_safe).round(3)
            if "P_K9" not in df.columns and "TEAM_K" in df.columns:
                df["P_K9"]  = (df["TEAM_K"]          * 9 / ip_safe).round(2)
                df["P_BB9"] = (df["TEAM_BB_ALLOWED"] * 9 / ip_safe).round(2)
                df["P_HR9"] = (df["TEAM_HR_ALLOWED"] * 9 / ip_safe).round(2)
                df["P_FIP"] = (
                    (13 * df["TEAM_HR_ALLOWED"] + 3 * df["TEAM_BB_ALLOWED"] - 2 * df["TEAM_K"])
                    / ip_safe + FIP_CONSTANT
                ).round(2)
                log.info(f"  Season {season}: backfilled pitching rate columns (P_ERA/WHIP/K9/…).")
            return df
        except Exception as exc:
            log.warning(f"  Season {season}: cache read failed ({exc}) — falling back to API.")
            return None

    def _save_to_cache(self, season: str, df: pd.DataFrame) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(self._cache_path(season), index=False)
            log.info(f"  Season {season}: cached to {self._cache_path(season)}")
        except Exception as exc:
            log.warning(f"  Season {season}: cache write failed ({exc}) — continuing without cache.")

    @staticmethod
    def _month_ranges(season: str) -> list[tuple[str, str]]:
        """Return (start, end) date strings for March–October of `season`."""
        months = [
            ("03", "31"), ("04", "30"), ("05", "31"), ("06", "30"),
            ("07", "31"), ("08", "31"), ("09", "30"), ("10", "31"),
        ]
        return [(f"{season}-{m}-01", f"{season}-{m}-{d}") for m, d in months]

    # ── Private: data fetching ────────────────────────────────────────────────

    def _fetch_season(self, season: str) -> Optional[pd.DataFrame]:
        """
        Retrieve the full regular-season schedule for `season`, then pull a
        box score for every completed game.  Retries each failed game up to
        3 times with exponential back-off.

        For completed seasons (before the current year), results are cached
        locally as parquet and loaded from disk on subsequent runs.  The
        schedule fetch itself is broken into monthly chunks (March–October),
        each independently retried, to avoid 8-month single-request timeouts.

        Parameters
        ----------
        season : str
            Four-digit year string, e.g. ``"2024"``.

        Returns
        -------
        pd.DataFrame or None
            One row per team per game, or None if all attempts are exhausted.
        """
        # ── 1. Check cache for completed seasons ──────────────────────────
        if self._is_season_complete(season) and not self.force_refresh:
            cached = self._load_from_cache(season)
            if cached is not None:
                log.info(f"  Season {season}: loaded {len(cached):,} rows from cache.")
                return cached

        # ── 2. Fetch schedule in monthly chunks with retry ────────────────
        all_games: list[dict] = []
        for start, end in self._month_ranges(season):
            for attempt in range(3):
                try:
                    chunk = statsapi.schedule(
                        start_date=start,
                        end_date=end,
                        sportId=1,
                    )
                    all_games.extend(chunk)
                    break
                except Exception as exc:
                    wait = self.sleep * (2 ** attempt)
                    log.warning(
                        f"  Schedule chunk {start}/{end} attempt {attempt + 1} failed: "
                        f"{exc} — retry in {wait:.1f}s"
                    )
                    time.sleep(wait)
            else:
                log.error(f"  Schedule chunk {start}/{end}: all retries exhausted.")

        # Keep only final (completed) regular-season games
        completed = [g for g in all_games if g.get("status") == "Final"
                                          and g.get("game_type") == "R"]
        log.info(f"  Season {season}: {len(completed)} completed games found.")

        # ── 3. Fetch boxscores in parallel with per-worker retry ──────────
        rows: list[dict] = []
        rows_lock = threading.Lock()
        total = len(completed)
        done_count = [0]   # mutable counter shared across threads

        def fetch_one(game: dict) -> None:
            game_id = game["game_id"]
            for attempt in range(3):
                try:
                    box    = statsapi.boxscore_data(game_id)
                    parsed = self._parse_boxscore(box, game_id, season)
                    with rows_lock:
                        rows.extend(parsed)
                        done_count[0] += 1
                        if done_count[0] % 100 == 0:
                            log.info(f"  Season {season}: {done_count[0]}/{total} games fetched …")
                    return
                except Exception as exc:
                    wait = self.sleep * (2 ** attempt)
                    log.warning(
                        f"  Game {game_id} attempt {attempt + 1} failed: "
                        f"{exc} — retry in {wait:.1f}s"
                    )
                    time.sleep(wait)
            log.error(f"  Game {game_id}: all attempts exhausted.")

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [pool.submit(fetch_one, g) for g in completed]
            for f in as_completed(futures):
                f.result()   # re-raise any unexpected exception

        if not rows:
            return None

        df = pd.DataFrame(rows)
        df["SEASON"] = season
        df = self._standardize_columns(df)

        # ── 4. Cache result for completed seasons ─────────────────────────
        if self._is_season_complete(season):
            self._save_to_cache(season, df)

        return df

    def _parse_boxscore(
        self,
        box:     dict,
        game_id: int,
        season:  str,
    ) -> list[dict]:
        """
        Extract two team-game dicts (home + away) from a raw statsapi
        ``boxscore_data`` payload.

        Returns
        -------
        list[dict]
            Always a two-element list: [away_row, home_row].
        """
        rows = []
        game_date_str = box.get("gameId", "")[:10]   # "YYYY/MM/DD"
        game_date     = game_date_str.replace("/", "-")

        for side in ("away", "home"):
            team_data    = box.get(side, {})
            team_info    = team_data.get("team", {})
            team_name    = team_info.get("name", "")
            team_id      = team_info.get("id", -1)
            # Use numeric ID map first (most reliable); fall back to name lookup.
            team_abv     = STATSAPI_TEAM_ID_MAP.get(
                team_id,
                STATSAPI_TEAM_ABV_MAP.get(team_info.get("teamName", ""), ""),
            )
            batting      = team_data.get("teamStats", {}).get("batting",  {})
            pitching     = team_data.get("teamStats", {}).get("pitching", {})

            # ── Identify the starting pitcher ─────────────────────────────
            sp_id, sp_name, sp_hand = self._extract_starting_pitcher(team_data)

            # ── Relief corps: pitcher IDs and pitch counts ─────────────────
            relief_pitcher_ids, relief_pitches = self._extract_bullpen_usage(
                team_data, sp_id
            )

            # ── Raw counting stats ─────────────────────────────────────────
            ab  = batting.get("atBats",      0)
            bb  = batting.get("baseOnBalls", 0)
            hbp = batting.get("hitByPitch",  0)
            h   = batting.get("hits",        0)
            hr  = batting.get("homeRuns",    0)
            so  = batting.get("strikeOuts",  0)
            r   = batting.get("runs",        0)
            sf  = batting.get("sacFlies",    0)
            doubles = batting.get("doubles", 0)
            triples = batting.get("triples", 0)
            singles = h - doubles - triples - hr

            # Plate appearances: AB + BB + HBP + SF + SAC
            sac = batting.get("sacBunts", 0)
            pa  = ab + bb + hbp + sf + sac

            rows.append({
                "GAME_ID":              game_id,
                "GAME_DATE":            game_date,
                "SEASON":               season,
                "TEAM_ID":              team_id,
                "TEAM_ABV":             team_abv,
                "HOME":                 1 if side == "home" else 0,
                # ── Batting counts ────────────────────────────────────────
                "R":                    r,
                "H":                    h,
                "AB":                   ab,
                "PA":                   pa,
                "BB":                   bb,
                "HBP":                  hbp,
                "SO":                   so,
                "HR":                   hr,
                "1B":                   max(singles, 0),
                "2B":                   doubles,
                "3B":                   triples,
                "SF":                   sf,
                "SAC":                  sac,
                # ── Pitching counts ───────────────────────────────────────
                "TEAM_ERA":             pitching.get("era",            0.0),
                "TEAM_IP":              self._parse_ip(pitching.get("inningsPitched", "0.0")),
                "TEAM_K":               pitching.get("strikeOuts",     0),
                "TEAM_BB_ALLOWED":      pitching.get("baseOnBalls",    0),
                "TEAM_H_ALLOWED":       pitching.get("hits",           0),
                "TEAM_HR_ALLOWED":      pitching.get("homeRuns",       0),
                "TEAM_RUNS_ALLOWED":    pitching.get("runs",           0),
                # ── Starting pitcher identifiers ──────────────────────────
                "SP_ID":                sp_id,
                "SP_NAME":              sp_name,
                "SP_HAND":              sp_hand,         # "LHP" | "RHP" | "UNK"
                # ── Bullpen usage for fatigue calculation ─────────────────
                "RELIEF_PITCHER_IDS":   relief_pitcher_ids,   # list[int]
                "RELIEF_PITCHES":       relief_pitches,        # int
            })

        return rows

    # ── Private: column standardisation ──────────────────────────────────────

    @staticmethod
    def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Cast types and derive all first-order rate statistics.

        New columns added
        -----------------
        WOBA        : Weighted On-Base Average
        ISO         : Isolated Power  (SLG – AVG)
        OBP         : On-Base Percentage
        SLG         : Slugging Percentage
        K_PCT       : Strikeout rate  (SO / PA)
        BB_PCT      : Walk rate       (BB / PA)
        PA_CLIPPED  : PA clipped to ≥ 1 (safe denominator)
        WIN         : 1 if R > OPP_R (filled post opponent-merge; placeholder 0)
        """
        df = df.copy()
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

        # Safe denominators
        pa_safe = df["PA"].clip(lower=1)
        ab_safe = df["AB"].clip(lower=1)
        ip_safe = pd.to_numeric(df.get("TEAM_IP", 9.0), errors="coerce").fillna(9.0).clip(lower=0.1)
        df["PA_CLIPPED"] = pa_safe

        # ── TEAM_ERA: cast to float ───────────────────────────────────────
        df["TEAM_ERA"] = pd.to_numeric(df["TEAM_ERA"], errors="coerce").fillna(4.50)

        # ── wOBA ──────────────────────────────────────────────────────────
        w = WOBA_WEIGHTS
        woba_num = (
            w["BB"]  * df["BB"]  +
            w["HBP"] * df["HBP"] +
            w["1B"]  * df["1B"]  +
            w["2B"]  * df["2B"]  +
            w["3B"]  * df["3B"]  +
            w["HR"]  * df["HR"]
        )
        woba_den = df["AB"] + df["BB"] + df["HBP"] + df["SF"]
        df["WOBA"] = (woba_num / woba_den.clip(lower=1)).round(3)

        # ── ISO (Isolated Power) ──────────────────────────────────────────
        slg_num = df["1B"] + 2 * df["2B"] + 3 * df["3B"] + 4 * df["HR"]
        df["SLG"] = (slg_num / ab_safe).round(3)
        df["AVG"] = (df["H"]  / ab_safe).round(3)
        df["ISO"] = (df["SLG"] - df["AVG"]).round(3)

        # ── OBP ───────────────────────────────────────────────────────────
        obp_den = df["AB"] + df["BB"] + df["HBP"] + df["SF"]
        df["OBP"] = ((df["H"] + df["BB"] + df["HBP"]) / obp_den.clip(lower=1)).round(3)

        # ── Rate stats ────────────────────────────────────────────────────
        df["K_PCT"]  = (df["SO"] / pa_safe).round(4)
        df["BB_PCT"] = (df["BB"] / pa_safe).round(4)

        # ── BABIP ─────────────────────────────────────────────────────────
        babip_den = (df["AB"] - df["SO"] - df["HR"] + df["SF"]).clip(lower=1)
        df["BABIP"] = ((df["H"] - df["HR"]) / babip_den).round(3)

        # ── RA (Runs Allowed) ─────────────────────────────────────────────
        df["RA"] = df["TEAM_RUNS_ALLOWED"]

        # ── Team pitching rate stats ──────────────────────────────────────
        df["P_ERA"]  = df["TEAM_ERA"]   # already cast to float above
        df["P_WHIP"] = ((df["TEAM_H_ALLOWED"] + df["TEAM_BB_ALLOWED"]) / ip_safe).round(3)
        df["P_K9"]   = (df["TEAM_K"]          * 9 / ip_safe).round(2)
        df["P_BB9"]  = (df["TEAM_BB_ALLOWED"] * 9 / ip_safe).round(2)
        df["P_HR9"]  = (df["TEAM_HR_ALLOWED"] * 9 / ip_safe).round(2)
        df["P_FIP"]  = (
            (13 * df["TEAM_HR_ALLOWED"] + 3 * df["TEAM_BB_ALLOWED"] - 2 * df["TEAM_K"])
            / ip_safe + FIP_CONSTANT
        ).round(2)

        # Placeholder; filled correctly after _attach_opponent_stats
        df["WIN"] = 0

        return df

    # ── Private: opponent-side merge ─────────────────────────────────────────

    def _attach_opponent_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For every team-game row, locate the opponent row sharing the same
        GAME_ID and attach key opponent metrics with an ``OPP_`` prefix.

        Also derives matchup-level features:
          - PACE_MATCHUP   : mean PA of both teams (proxy for game pace)
          - PROJ_TOTAL_RAW : actual combined runs (R + OPP_R)
          - MARGIN         : R – OPP_R
          - WIN            : 1 if R > OPP_R
        """
        log.info("Attaching opponent-side statistics …")

        opp_cols = [
            "GAME_ID", "TEAM_ID",
            "R", "H", "AB", "PA", "BB", "SO", "HR", "WOBA", "ISO",
            "OBP", "SLG", "AVG", "K_PCT", "BB_PCT",
            "TEAM_ERA", "TEAM_IP", "TEAM_K", "TEAM_BB_ALLOWED",
            "TEAM_H_ALLOWED", "TEAM_HR_ALLOWED", "TEAM_RUNS_ALLOWED",
            "HOME",
        ]
        opp = df[opp_cols].copy()
        opp.columns = (
            ["GAME_ID", "OPP_TEAM_ID"]
            + [f"OPP_{c}" for c in opp_cols[2:]]
        )

        merged = df.merge(opp, on="GAME_ID", how="left")
        merged = merged[merged["TEAM_ID"] != merged["OPP_TEAM_ID"]].copy()

        # Matchup-level features
        merged["PACE_MATCHUP"]   = ((merged["PA"] + merged["OPP_PA"]) / 2).round(1)
        merged["PROJ_TOTAL_RAW"] = merged["R"] + merged["OPP_R"]
        merged["MARGIN"]         = merged["R"] - merged["OPP_R"]
        merged["WIN"]            = (merged["MARGIN"] > 0).astype(int)

        return merged.reset_index(drop=True)

    # ── Private: starting pitcher metrics ────────────────────────────────────

    def _attach_starting_pitcher_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For each game row, look back at the SP's prior appearances in the
        same DataFrame (within the same season) and compute rolling ERA,
        WHIP, K/9, and FIP over the last 5 and last 10 starts.

        New columns
        -----------
        SP_ERA_5,  SP_ERA_10  : rolling ERA  over last 5 / 10 starts
        SP_WHIP_5, SP_WHIP_10 : rolling WHIP over last 5 / 10 starts
        SP_K9_5,   SP_K9_10   : rolling K/9  over last 5 / 10 starts
        SP_FIP_5,  SP_FIP_10  : rolling FIP  over last 5 / 10 starts
        SP_GS_SEASON          : cumulative starts made so far this season
        """
        log.info("Computing rolling starting-pitcher metrics …")

        # We need to iterate in chronological order; SP_ID may appear on
        # multiple teams across seasons, so key = (SP_ID, SEASON).
        df = df.sort_values(["SEASON", "GAME_DATE", "GAME_ID"]).copy()

        # Initialise all new SP metric columns
        for window in (SP_RECENT_STARTS_SHORT, SP_RECENT_STARTS_LONG):
            sfx = str(window)
            df[f"SP_ERA_{sfx}"]  = float("nan")
            df[f"SP_WHIP_{sfx}"] = float("nan")
            df[f"SP_K9_{sfx}"]   = float("nan")
            df[f"SP_FIP_{sfx}"]  = float("nan")
        df["SP_GS_SEASON"] = 0

        # We need SP game-level stats to build rolling windows.
        # Pull them from the team pitching rows where SP_ID is present.
        sp_history: dict[tuple, list[dict]] = {}  # (sp_id, season) → list of start dicts

        for idx, row in df.iterrows():
            sp_id  = row["SP_ID"]
            season = row["SEASON"]
            key    = (sp_id, season)

            prior_starts = sp_history.get(key, [])

            df.at[idx, "SP_GS_SEASON"] = len(prior_starts)

            for window in (SP_RECENT_STARTS_SHORT, SP_RECENT_STARTS_LONG):
                sfx    = str(window)
                window_starts = prior_starts[-window:] if prior_starts else []

                if window_starts:
                    era  = self._compute_rolling_era(window_starts)
                    whip = self._compute_rolling_whip(window_starts)
                    k9   = self._compute_rolling_k9(window_starts)
                    fip  = self._compute_rolling_fip(window_starts)
                    df.at[idx, f"SP_ERA_{sfx}"]  = round(era,  2)
                    df.at[idx, f"SP_WHIP_{sfx}"] = round(whip, 3)
                    df.at[idx, f"SP_K9_{sfx}"]   = round(k9,   2)
                    df.at[idx, f"SP_FIP_{sfx}"]  = round(fip,  2)

            # Append this appearance to history AFTER computing metrics
            # (so we never include the current game in its own look-back).
            # We approximate per-start IP / K / BB / HR from team-level
            # totals scaled to the SP's innings share where individual
            # pitcher lines are unavailable in the free tier of the API.
            sp_history.setdefault(key, []).append({
                "ip":  row.get("TEAM_IP", 6.0),       # starter proxy
                "er":  row.get("TEAM_RUNS_ALLOWED", 3),
                "h":   row.get("TEAM_H_ALLOWED", 7),
                "bb":  row.get("TEAM_BB_ALLOWED", 2),
                "k":   row.get("TEAM_K", 6),
                "hr":  row.get("TEAM_HR_ALLOWED", 1),
            })

        return df

    # ── Private: pitcher rest + bullpen fatigue ───────────────────────────────

    def _attach_pitcher_rest_and_bullpen_fatigue(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add two fatigue features per team-game row:

        SP_DAYS_REST : calendar days since the starting pitcher's last
                       appearance. Defaults to 5 for season-opening starts.

        BULLPEN_PITCHES_3D : total pitches thrown by the team's relief corps
                             across the previous ``BULLPEN_FATIGUE_DAYS`` (3)
                             calendar days, indicating pen wear.
        """
        log.info("Computing SP rest days and bullpen fatigue …")

        df = df.sort_values(["TEAM_ID", "GAME_DATE"]).copy()
        df["DAYS_REST"]       = 5    # default: assume fresh arm
        df["BP_PITCHES_3D"]   = 0

        for team_id, grp in df.groupby("TEAM_ID"):
            grp    = grp.sort_values("GAME_DATE").reset_index()
            orig_ix = grp["index"].tolist()

            # ── SP rest ───────────────────────────────────────────────────
            sp_last_seen: dict[int, pd.Timestamp] = {}

            for i, row in grp.iterrows():
                sp_id     = row["SP_ID"]
                game_date = row["GAME_DATE"]

                if sp_id in sp_last_seen:
                    delta = (game_date - sp_last_seen[sp_id]).days
                    df.at[orig_ix[i], "DAYS_REST"] = min(delta, 30)
                # else: default 5 already set

                sp_last_seen[sp_id] = game_date

            # ── Bullpen fatigue ───────────────────────────────────────────
            for i, row in grp.iterrows():
                game_date    = row["GAME_DATE"]
                window_start = game_date - timedelta(days=BULLPEN_FATIGUE_DAYS)

                recent = grp[
                    (grp["GAME_DATE"] >= window_start) &
                    (grp["GAME_DATE"] <  game_date)
                ]
                total_relief_pitches = recent["RELIEF_PITCHES"].sum()
                df.at[orig_ix[i], "BP_PITCHES_3D"] = int(total_relief_pitches)

        return df

    # ── Private: park factors ─────────────────────────────────────────────────

    def _attach_park_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Join PARK_FACTOR (100 = neutral) to each row based on which team is
        at home (HOME == 1).  Away games inherit the home team's ballpark.

        Because each game produces two rows (home + away), both rows receive
        the same PARK_FACTOR value for the home stadium.
        """
        log.info("Attaching park factors …")

        # Build a game-id → park-factor lookup from the home-team rows
        home_pf = (
            df[df["HOME"] == 1][["GAME_ID", "TEAM_ABV"]]
            .copy()
            .rename(columns={"TEAM_ABV": "HOME_ABV"})
        )
        home_pf["PARK_FACTOR"] = home_pf["HOME_ABV"].map(PARK_FACTORS).fillna(100.0)

        df = df.merge(home_pf[["GAME_ID", "PARK_FACTOR"]], on="GAME_ID", how="left")
        df["PARK_FACTOR"] = df["PARK_FACTOR"].fillna(100.0)
        return df

    # ── Private: boxscore parsing helpers ────────────────────────────────────

    @staticmethod
    def _extract_starting_pitcher(
        team_data: dict,
    ) -> tuple[int, str, str]:
        """
        Identify the starting pitcher from a statsapi boxscore side-dict.

        Returns
        -------
        (sp_id, sp_name, sp_hand) : (int, str, str)
            sp_hand is one of "LHP", "RHP", or "UNK".
        """
        pitchers = team_data.get("pitchers", [])
        players  = team_data.get("players",  {})

        sp_id, sp_name, sp_hand = -1, "Unknown", "UNK"

        if pitchers:
            first_pitcher_id = pitchers[0]
            player_key       = f"ID{first_pitcher_id}"
            player_info      = players.get(player_key, {})

            sp_id   = first_pitcher_id
            sp_name = (
                player_info.get("person", {}).get("fullName", "Unknown")
            )
            # pitchHand may be directly on the player dict or nested under person
            pitch_hand_info = (
                player_info.get("pitchHand")
                or player_info.get("person", {}).get("pitchHand")
                or {}
            )
            pitch_hand_code = (
                pitch_hand_info.get("code", "")
                if isinstance(pitch_hand_info, dict)
                else str(pitch_hand_info)
            )
            if pitch_hand_code == "L":
                sp_hand = "LHP"
            elif pitch_hand_code == "R":
                sp_hand = "RHP"

        return sp_id, sp_name, sp_hand

    @staticmethod
    def _extract_bullpen_usage(
        team_data:  dict,
        sp_id:      int,
    ) -> tuple[list[int], int]:
        """
        Collect relief pitcher IDs and aggregate their pitch counts.

        Parameters
        ----------
        team_data : dict
            One side of a statsapi boxscore_data payload.
        sp_id : int
            The starting pitcher's player ID; excluded from relief totals.

        Returns
        -------
        (relief_ids, total_relief_pitches) : (list[int], int)
        """
        pitchers = team_data.get("pitchers",  [])
        players  = team_data.get("players",   {})

        relief_ids    = [pid for pid in pitchers if pid != sp_id]
        total_pitches = 0

        for pid in relief_ids:
            player_key  = f"ID{pid}"
            player_data = players.get(player_key, {})
            pitches     = (
                player_data
                .get("stats", {})
                .get("pitching", {})
                .get("numberOfPitches", 0)
            )
            total_pitches += pitches

        return relief_ids, total_pitches

    # ── Private: rolling metric calculators ──────────────────────────────────

    @staticmethod
    def _compute_rolling_era(starts: list[dict]) -> float:
        """
        ERA = (earned runs / total innings pitched) × 9.

        We use 'er' as a proxy for earned runs since the free-tier API
        does not always expose the ER / R split at the individual level.
        """
        total_er = sum(s.get("er", 0) for s in starts)
        total_ip = sum(s.get("ip", 0) for s in starts)
        return _safe_div(total_er * 9, total_ip, default=0.0)

    @staticmethod
    def _compute_rolling_whip(starts: list[dict]) -> float:
        """WHIP = (walks + hits allowed) / innings pitched."""
        total_bb = sum(s.get("bb", 0) for s in starts)
        total_h  = sum(s.get("h",  0) for s in starts)
        total_ip = sum(s.get("ip", 0) for s in starts)
        return _safe_div(total_bb + total_h, total_ip, default=0.0)

    @staticmethod
    def _compute_rolling_k9(starts: list[dict]) -> float:
        """K/9 = strikeouts / innings pitched × 9."""
        total_k  = sum(s.get("k",  0) for s in starts)
        total_ip = sum(s.get("ip", 0) for s in starts)
        return _safe_div(total_k * 9, total_ip, default=0.0)

    @staticmethod
    def _compute_rolling_fip(starts: list[dict]) -> float:
        """
        FIP = ((13 × HR + 3 × BB – 2 × K) / IP) + FIP_CONSTANT.

        A fielding-independent measure of pitcher effectiveness.
        """
        total_hr = sum(s.get("hr", 0) for s in starts)
        total_bb = sum(s.get("bb", 0) for s in starts)
        total_k  = sum(s.get("k",  0) for s in starts)
        total_ip = sum(s.get("ip", 0) for s in starts)
        if total_ip == 0:
            return 0.0
        return ((13 * total_hr + 3 * total_bb - 2 * total_k) / total_ip) + FIP_CONSTANT

    # ── Private: utility ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_ip(ip_str: str) -> float:
        """
        Convert the MLB Stats API "innings pitched" string to a decimal float.

        The API returns fractional innings as  ``"6.2"`` meaning 6 and 2/3
        innings (20 outs), not 6.2 decimal innings.  This method converts
        the ".1" and ".2" suffixes to their true fractions (0.333, 0.667).

        Examples
        --------
        >>> MLBDataLoader._parse_ip("6.2")
        6.667
        >>> MLBDataLoader._parse_ip("9.0")
        9.0
        """
        try:
            whole, frac = str(ip_str).split(".")
            return int(whole) + int(frac) / 3.0
        except (ValueError, AttributeError):
            return 0.0

# Backwards-compat alias so imports expecting `DataLoader` continue to work
DataLoader = MLBDataLoader