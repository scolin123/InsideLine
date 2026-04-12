"""
data_loader.py — MODULE 1: DataLoader
Fetches raw game-log data from nba_api for the specified seasons and returns
a unified DataFrame with team + opponent box-score columns, travel fatigue,
and basic derived rate statistics.
"""
import time
import math
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

from .config import SEASONS, API_SLEEP, ARENA_COORDS

log = logging.getLogger(__name__)


class DataLoader:
    """
    Fetches raw game-log data from nba_api for the specified seasons.
    Returns a unified DataFrame with both team and opponent box-score columns.
    """

    def __init__(
        self,
        seasons:       list[str] = SEASONS,
        sleep:         float     = API_SLEEP,
        force_refresh: bool      = False,
        cache_dir:     str       = "data/nba_cache",
    ):
        self.seasons       = seasons
        self.sleep         = sleep
        self.force_refresh = force_refresh
        self.cache_dir     = Path(cache_dir)
        self._raw: Optional[pd.DataFrame] = None

    # ── Public ────────────────────────────────────────────────────────────────
    def load(self) -> pd.DataFrame:
        """Pull & merge all seasons; return enriched game-log DataFrame."""
        frames = []
        for season in self.seasons:
            log.info(f"Fetching season {season} …")
            df = self._fetch_season(season)
            if df is not None and not df.empty:
                frames.append(df)

        if not frames:
            raise RuntimeError("No data returned from nba_api. Check connectivity.")

        combined = (
            pd.concat(frames, ignore_index=True)
              .drop_duplicates(subset=["GAME_ID", "TEAM_ID"])
              .sort_values(["GAME_DATE", "GAME_ID"])
              .reset_index(drop=True)
        )

        combined = self._attach_opponent_stats(combined)
        combined = self._attach_travel_fatigue(combined)
        self._raw = combined
        log.info(f"DataLoader complete — {len(combined):,} team-game rows loaded.")
        return combined

    @property
    def raw(self) -> pd.DataFrame:
        if self._raw is None:
            raise RuntimeError("Call .load() first.")
        return self._raw

    # ── Private: caching helpers ──────────────────────────────────────────────

    def _cache_path(self, season: str) -> Path:
        return self.cache_dir / f"{season}.parquet"

    @staticmethod
    def _is_season_complete(season: str) -> bool:
        """True when the season's ending year is before the current calendar year.

        NBA seasons use the format ``"2023-24"``, which ends in calendar year 2024.
        """
        end_year = int(season.split("-")[1]) + 2000
        return end_year < datetime.utcnow().year

    def _load_from_cache(self, season: str) -> Optional[pd.DataFrame]:
        path = self._cache_path(season)
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
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

    # ── Private ───────────────────────────────────────────────────────────────

    def _fetch_season(self, season: str) -> Optional[pd.DataFrame]:
        """Fetch LeagueGameLog for one season (retries on timeout).

        For completed seasons (ending before the current calendar year), results
        are cached locally as parquet and loaded from disk on subsequent runs.
        """
        # Check cache for completed seasons
        if self._is_season_complete(season) and not self.force_refresh:
            cached = self._load_from_cache(season)
            if cached is not None:
                log.info(f"  Season {season}: loaded {len(cached):,} rows from cache.")
                return cached

        for attempt in range(3):
            try:
                endpoint = leaguegamelog.LeagueGameLog(
                    season=season,
                    season_type_all_star="Regular Season",
                    direction="ASC",
                )
                time.sleep(self.sleep)
                df = endpoint.get_data_frames()[0]
                df["SEASON"] = season
                df = self._standardize_columns(df)

                # Cache result for completed seasons
                if self._is_season_complete(season):
                    self._save_to_cache(season, df)

                return df
            except Exception as exc:
                wait = self.sleep * (2 ** attempt)
                log.warning(f"Season {season} attempt {attempt+1} failed: {exc} — retry in {wait:.1f}s")
                time.sleep(wait)
        log.error(f"Season {season}: all attempts exhausted.")
        return None

    @staticmethod
    def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Rename, cast, and derive basic rate columns."""
        df = df.copy()
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

        # Possession estimate: FGA - OREB + TO + 0.44*FTA
        df["POSS"] = (
            df["FGA"] - df["OREB"] + df["TOV"] + 0.44 * df["FTA"]
        ).clip(lower=1)

        # Home / Away flag
        df["HOME"] = df["MATCHUP"].str.contains("vs\\.").astype(int)

        # Win flag
        df["WIN"] = (df["WL"] == "W").astype(int)

        # Points per possession
        df["PPP"] = df["PTS"] / df["POSS"]

        # Effective FG%  eFG = (FGM + 0.5*FG3M) / FGA
        df["EFG_PCT"] = (df["FGM"] + 0.5 * df["FG3M"]) / df["FGA"].clip(lower=1)

        # Turnover %  TOV% = TO / (FGA + 0.44*FTA + TO)
        df["TOV_PCT"] = df["TOV"] / (df["FGA"] + 0.44 * df["FTA"] + df["TOV"]).clip(lower=1)

        # 3-char team abbreviation from MATCHUP
        df["TEAM_ABV"] = df["MATCHUP"].str[:3]

        return df

    def _attach_opponent_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For every team-game row, locate the opponent row in the same GAME_ID
        and attach key opponent metrics with an 'OPP_' prefix.
        """
        log.info("Attaching opponent-side statistics …")
        opp_cols = [
            "GAME_ID", "TEAM_ID", "PTS", "POSS", "PPP",
            "EFG_PCT", "TOV_PCT", "FGA", "OREB", "DREB",
            "AST", "STL", "BLK", "FG_PCT", "FG3_PCT", "HOME",
        ]
        opp = df[opp_cols].copy()
        opp.columns = ["GAME_ID", "OPP_TEAM_ID"] + [f"OPP_{c}" for c in opp_cols[2:]]

        merged = df.merge(opp, on="GAME_ID", how="left")
        # Keep only rows where OPP is different from self
        merged = merged[merged["TEAM_ID"] != merged["OPP_TEAM_ID"]]

        # Derived matchup features
        merged["PACE_MATCHUP"]   = (merged["POSS"] + merged["OPP_POSS"]) / 2
        merged["PROJ_TOTAL_RAW"] = merged["PTS"] + merged["OPP_PTS"]
        merged["MARGIN"]         = merged["PTS"] - merged["OPP_PTS"]

        return merged.reset_index(drop=True)

    def _attach_travel_fatigue(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add:
          - DAYS_REST  : calendar days since the team's previous game
          - MILES_7D   : total miles traveled by the team in the last 7 days
        """
        log.info("Computing travel & fatigue features …")
        df = df.sort_values(["TEAM_ID", "GAME_DATE"]).copy()

        days_rest, miles_7d = [], []

        for team_id, grp in df.groupby("TEAM_ID"):
            grp = grp.sort_values("GAME_DATE").reset_index()
            abv_series = grp["TEAM_ABV"]

            for i, row in grp.iterrows():
                # ── Days rest ─────────────────────────────────────────────
                if i == 0:
                    days_rest.append(3)          # assume rested at season start
                else:
                    delta = (row["GAME_DATE"] - grp.loc[i-1, "GAME_DATE"]).days
                    days_rest.append(min(delta, 14))

                # ── Miles traveled in last 7 days ─────────────────────────
                window_start = row["GAME_DATE"] - timedelta(days=7)
                recent = grp[(grp["GAME_DATE"] >= window_start) &
                             (grp["GAME_DATE"] <  row["GAME_DATE"])]

                total_miles = 0.0
                prev_abv    = abv_series.iloc[max(0, i-1)]
                for _, r in recent.iterrows():
                    cur_abv = r["TEAM_ABV"]
                    total_miles += self._haversine(
                        ARENA_COORDS.get(prev_abv, (39.5, -98.35)),
                        ARENA_COORDS.get(cur_abv,  (39.5, -98.35)),
                    )
                    prev_abv = cur_abv
                miles_7d.append(total_miles)

        df["DAYS_REST"] = days_rest
        df["MILES_7D"]  = miles_7d
        return df

    @staticmethod
    def _haversine(p1: tuple[float, float], p2: tuple[float, float]) -> float:
        """Great-circle distance in miles between two (lat, lon) points."""
        R = 3958.8  # Earth radius in miles
        lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
        lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
        return R * 2 * math.asin(math.sqrt(a))
