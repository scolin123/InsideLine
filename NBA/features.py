"""
features.py — MODULE 2: FeatureEngine
Transforms the raw game-log DataFrame into rolling, opponent-adjusted features
ready for model training or inference.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


class FeatureEngine:
    """
    Transforms the raw game-log DataFrame into rolling, opponent-adjusted
    features ready for model training or inference.
    """

    ROLL_WINDOWS = [5, 10, 20]   # game windows for rolling averages
    EWM_SPANS    = [5, 10, 20]   # spans for exponential weighted averages

    # Core stats to roll
    ROLL_STATS = [
        "PTS", "PPP", "EFG_PCT", "TOV_PCT", "POSS",
        "OREB", "DREB", "FG_PCT", "FG3_PCT",
        "OPP_PTS", "OPP_PPP", "OPP_EFG_PCT", "OPP_TOV_PCT",
        "DAYS_REST", "MILES_7D", "HOME",
    ]

    # High-signal stats to also capture with exponential weighting
    EWM_STATS = [
        "PTS", "OPP_PTS", "PPP", "OPP_PPP",
        "EFG_PCT", "OPP_EFG_PCT", "POSS",
    ]

    def __init__(self):
        self.scaler      = StandardScaler()
        self.feature_cols: list[str] = []

    # ── Public ────────────────────────────────────────────────────────────────
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all engineered columns and return the enriched DataFrame."""
        df = df.copy()
        df = self._rolling_averages(df)
        df = self._ewm_averages(df)
        df = self._win_streak(df)
        df = self._season_win_pct(df)
        df = self._home_away_splits(df)
        df = self._rest_advantage(df)
        df = self._offensive_defensive_ratings(df)
        df = self._poisson_eppps(df)
        df = self._spread_and_totals(df)

        # Drop rows with NaN rolling stats (early season warm-up)
        min_window = min(self.ROLL_WINDOWS)
        df = df.dropna(subset=[c for c in df.columns if f"_R{min_window}" in c])

        self.feature_cols = self._detect_feature_cols(df)
        log.info(f"FeatureEngine built {len(self.feature_cols)} features on {len(df):,} rows.")
        return df.reset_index(drop=True)

    def get_feature_matrix(
        self, df: pd.DataFrame, scale: bool = False
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Return (X, y_pts, y_win) ready for training."""
        X     = df[self.feature_cols].copy()
        y_pts = df["PTS"].values.astype(float)
        y_win = df["WIN"].values.astype(int)
        if scale:
            X = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=self.feature_cols,
            )
        return X, y_pts, y_win

    # ── Private ───────────────────────────────────────────────────────────────
    def _rolling_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute per-team rolling means for all ROLL_STATS."""
        df = df.sort_values(["TEAM_ID", "GAME_DATE"])
        for col in self.ROLL_STATS:
            if col not in df.columns:
                continue
            for win in self.ROLL_WINDOWS:
                new_col = f"{col}_R{win}"
                df[new_col] = (
                    df.groupby("TEAM_ID")[col]
                      .transform(lambda s: s.shift(1).rolling(win, min_periods=max(2, win//2)).mean())
                )
        return df

    def _ewm_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Exponential weighted moving averages for high-signal stats.
        Unlike simple rolling means, EWM weights recent games exponentially
        higher — last night's game matters more than one from 3 weeks ago.
        Suffix: {col}_EWM{span}
        """
        df = df.sort_values(["TEAM_ID", "GAME_DATE"])
        for col in self.EWM_STATS:
            if col not in df.columns:
                continue
            for span in self.EWM_SPANS:
                df[f"{col}_EWM{span}"] = (
                    df.groupby("TEAM_ID")[col]
                      .transform(
                          lambda s, sp=span: s.shift(1)
                            .ewm(span=sp, min_periods=max(2, sp // 2))
                            .mean()
                      )
                )
        return df

    def _win_streak(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Signed consecutive win/loss streak entering the game.
        +4 = won last 4 games, -2 = lost last 2 games.
        Uses shift(1) so the current game's outcome is not included.
        """
        df = df.sort_values(["TEAM_ID", "GAME_DATE"])

        def _streak(series: pd.Series) -> pd.Series:
            shifted = series.shift(1)
            result, cur = [], 0
            for v in shifted:
                if pd.isna(v):
                    result.append(0)
                    continue
                if v == 1:
                    cur = cur + 1 if cur >= 0 else 1
                else:
                    cur = cur - 1 if cur <= 0 else -1
                result.append(cur)
            return pd.Series(result, index=series.index)

        df["WIN_STREAK"] = df.groupby("TEAM_ID")["WIN"].transform(_streak)
        return df

    def _season_win_pct(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cumulative season win percentage up to (not including) the current game.
        Captures overall team quality that short rolling windows miss —
        a 40-10 team and a 10-40 team can look identical in the last 5 games.
        """
        df = df.sort_values(["TEAM_ID", "SEASON", "GAME_DATE"])
        df["SEASON_WIN_PCT"] = (
            df.groupby(["TEAM_ID", "SEASON"])["WIN"]
              .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
        )
        return df

    def _home_away_splits(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Separate rolling scoring averages for home and away games.
        A team that's dominant at home but struggles on the road will have
        those very different profiles blended in a single rolling average.
        Computes PTS and OPP_PTS splits for windows [10, 20].
        Suffix: {col}_{HOME|AWAY}_R{win}
        """
        df = df.sort_values(["TEAM_ID", "GAME_DATE"])
        for col in ["PTS", "OPP_PTS"]:
            if col not in df.columns:
                continue
            for win in [10, 20]:
                for loc_name, loc_val in [("HOME", 1), ("AWAY", 0)]:
                    tmp = f"__split_{col}_{loc_name}"
                    df[tmp] = df[col].where(df["HOME"] == loc_val)
                    df[f"{col}_{loc_name}_R{win}"] = (
                        df.groupby("TEAM_ID")[tmp]
                          .transform(
                              lambda s, w=win: s.shift(1)
                                .rolling(w, min_periods=max(2, w // 3))
                                .mean()
                          )
                    )
                    df.drop(columns=[tmp], inplace=True)
        return df

    def _rest_advantage(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Difference in days rest between this team and the opponent.
        Positive = this team is more rested. Clipped to [-7, 7].
        Derived by self-joining on GAME_ID + OPP_TEAM_ID (both rows exist
        in the DataFrame since data_loader produces one row per team per game).
        """
        rest_lookup = (
            df[["GAME_ID", "TEAM_ID", "DAYS_REST"]]
            .rename(columns={"TEAM_ID": "OPP_TEAM_ID", "DAYS_REST": "OPP_DAYS_REST"})
        )
        df = df.merge(rest_lookup, on=["GAME_ID", "OPP_TEAM_ID"], how="left")
        df["REST_ADVANTAGE"] = (df["DAYS_REST"] - df["OPP_DAYS_REST"]).clip(-7, 7)
        return df

    def _offensive_defensive_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Off Rating  = (PTS / POSS) * 100
        Def Rating  = (OPP_PTS / POSS) * 100
        Net Rating  = Off - Def
        (All opponent-adjusted via rolling averages already computed)
        """
        for win in self.ROLL_WINDOWS:
            df[f"OFF_RTG_R{win}"] = df[f"PTS_R{win}"]     / df[f"POSS_R{win}"].clip(lower=1) * 100
            df[f"DEF_RTG_R{win}"] = df[f"OPP_PTS_R{win}"] / df[f"POSS_R{win}"].clip(lower=1) * 100
            df[f"NET_RTG_R{win}"] = df[f"OFF_RTG_R{win}"] - df[f"DEF_RTG_R{win}"]
        return df

    def _poisson_eppps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Expected Points per Possession (ePPP) via Poisson rate estimation.

        Poisson λ for a team = their rolling PPP × opponent's rolling defensive PPP
                               scaled by the projected possession count.

        ePPP_A = (OFF_PPP_A × DEF_PPP_B)^0.5  — geometric mean of attack vs defence
        """
        for win in self.ROLL_WINDOWS:
            off_col = f"PPP_R{win}"
            def_col = f"OPP_PPP_R{win}"
            df[f"ePPP_R{win}"] = np.sqrt(
                df[off_col].clip(lower=0.01) * df[def_col].clip(lower=0.01)
            )
        return df

    def _spread_and_totals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pre-compute raw projected spread and total (used as training targets)."""
        df["RAW_PROJ_SPREAD"] = df["MARGIN"]          # actual margin → target for regression
        df["RAW_PROJ_TOTAL"]  = df["PROJ_TOTAL_RAW"]  # actual total  → target for regression
        return df

    @staticmethod
    def _detect_feature_cols(df: pd.DataFrame) -> list[str]:
        """
        Return only safe pre-game feature columns — no current-game box score stats.

        Uses an allowlist instead of a denylist to prevent data leakage.
        Raw per-game stats (PPP, EFG_PCT, POSS, OPP_PPP, PACE_MATCHUP, etc.) are
        computed from the game being predicted and must never be used as features.
        Only rolling/lagged averages and known pre-game context columns are safe.
        """
        # Pre-game context columns that don't encode current game outcome
        pre_game_cols = {
            "HOME", "DAYS_REST", "MILES_7D",
            "WIN_STREAK", "SEASON_WIN_PCT", "REST_ADVANTAGE",
        }

        # Rolling average suffixes (all built with shift(1) — no look-ahead)
        roll_suffixes  = tuple(f"_R{w}"   for w  in FeatureEngine.ROLL_WINDOWS)
        ewm_suffixes   = tuple(f"_EWM{sp}" for sp in FeatureEngine.EWM_SPANS)
        split_suffixes = (
            tuple(f"_HOME_R{w}" for w in [10, 20]) +
            tuple(f"_AWAY_R{w}" for w in [10, 20])
        )
        all_suffixes = roll_suffixes + ewm_suffixes + split_suffixes

        allowed = []
        for c in df.columns:
            if df[c].dtype not in [np.float64, np.int64, float, int]:
                continue
            if df[c].isna().all():
                continue
            if c in pre_game_cols:
                allowed.append(c)
            elif any(c.endswith(s) for s in all_suffixes):
                allowed.append(c)

        return allowed
