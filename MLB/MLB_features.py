"""
MLB_features.py — MODULE 2: FeatureEngine (MLB)
Transforms the raw game-log DataFrame produced by MLB_data_loader into rolling,
opponent-adjusted, and park-factor-corrected features ready for model training
or inference.

Key MLB-specific design decisions vs. the NBA version
──────────────────────────────────────────────────────
• Possessions → Plate Appearances (PA) as the primary rate denominator.
• Starting Pitcher (SP) metrics (ERA, WHIP, K/9, FIP) are rolled over the
  pitcher's last 5 and 10 *starts* independently of the team's game log.
• Bullpen Fatigue (total pitches thrown by the bullpen in the previous 3 days)
  is attached as a game-level feature.
• Park Factors (run-scoring environment index centred on 1.0) are multiplied
  into projected totals — a park factor of 1.08 at Coors Field inflates both
  teams' expected run outputs by 8 %.
• Pitcher Handedness (SP_HAND: 0 = LHP, 1 = RHP) is encoded as a binary
  feature; XGBoost treats it as a numerical split, and LightGBM can optionally
  mark it categorical.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .MLB_config import PARK_FACTORS  # dict[team_abv → float], centred on 1.0

log = logging.getLogger(__name__)


class FeatureEngine:
    """
    Transforms raw MLB game-log data into a rich feature matrix.

    Rolling windows
    ───────────────
    TEAM_ROLL_WINDOWS  : [5, 10, 20] games — team-level offensive/defensive stats
    SP_ROLL_WINDOWS    : [5, 10]     starts — starting-pitcher-level stats

    Feature categories
    ──────────────────
    1. Team rolling averages  — OBP, SLG, wOBA, ISO, K%, BB%, BABIP, PA
    2. Pitching rolling avgs  — ERA, WHIP, K/9, BB/9, HR/9, FIP per team
    3. SP rolling metrics     — ERA, WHIP, K/9, FIP for the *named* starter
    4. Bullpen fatigue        — total BP pitches in last 3 days
    5. Offensive / defensive ratings — runs per PA, opponent runs per PA
    6. Park-factor adjustment — raw offensive ratings scaled by venue index
    7. Pitcher handedness     — SP_HAND_HOME, SP_HAND_AWAY  (0=LHP, 1=RHP)
    8. Travel / rest          — DAYS_REST, MILES_7D  (from data_loader)
    9. Run-expectation proxy  — ePPA (expected runs per PA, geometric-mean blend)
   10. Derived targets        — RAW_PROJ_SPREAD, RAW_PROJ_TOTAL
    """

    TEAM_ROLL_WINDOWS: list[int] = [5, 10, 20]
    SP_ROLL_WINDOWS:   list[int] = [5, 10]

    # Team-level box-score columns to roll
    TEAM_ROLL_STATS: list[str] = [
        "R",            # Runs scored
        "RA",           # Runs allowed
        "PA",           # Plate appearances
        "H", "HR", "BB", "SO",
        "OBP", "SLG", "WOBA",
        "ISO",          # Isolated power = SLG - BA
        "K_PCT",        # Strikeout %  = SO / PA
        "BB_PCT",       # Walk %       = BB / PA
        "BABIP",        # Batting average on balls in play
        # Team pitching aggregates (starters + bullpen combined)
        "P_ERA", "P_WHIP", "P_K9", "P_BB9", "P_HR9", "P_FIP",
        # Opponent mirror
        "OPP_R", "OPP_PA", "OPP_OBP", "OPP_SLG", "OPP_WOBA",
        "OPP_K_PCT", "OPP_BB_PCT",
        # Situational / contextual
        "HOME", "DAYS_REST", "MILES_7D",
        "BP_PITCHES_3D",    # bullpen fatigue (attached by data_loader)
    ]

    # SP-level columns to roll per *pitcher* (indexed by SP_ID)
    SP_ROLL_STATS: list[str] = [
        "SP_ERA", "SP_WHIP", "SP_K9", "SP_BB9", "SP_FIP",
        "SP_IP",  "SP_ER",
    ]

    def __init__(self) -> None:
        self.scaler:       StandardScaler = StandardScaler()
        self.feature_cols: list[str]      = []

    # ── Public ────────────────────────────────────────────────────────────────
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all engineered columns and return the enriched DataFrame.

        Processing order
        ────────────────
        1. Team rolling averages
        2. SP rolling averages (per pitcher)
        3. Offensive / defensive rate ratings
        4. Park-factor adjustments
        5. ePPA (Poisson-style run-expectation proxy)
        6. Spread and total target derivation
        7. NaN row pruning (early-season warm-up)
        """
        df = df.copy()
        df = self._team_rolling_averages(df)
        df = self._sp_rolling_averages(df)
        df = self._offensive_defensive_ratings(df)
        df = self._apply_park_factors(df)
        df = self._eppa(df)
        df = self._encode_pitcher_handedness(df)
        df = self._spread_and_totals(df)

        # Drop rows missing the smallest rolling window features
        min_win = min(self.TEAM_ROLL_WINDOWS)
        roll_cols = [c for c in df.columns if f"_R{min_win}" in c]
        df = df.dropna(subset=roll_cols)

        self.feature_cols = self._detect_feature_cols(df)
        log.info(
            f"FeatureEngine (MLB) built {len(self.feature_cols)} features "
            f"on {len(df):,} rows."
        )
        return df.reset_index(drop=True)

    def get_feature_matrix(
        self,
        df:    pd.DataFrame,
        scale: bool = False,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Return (X, y_runs, y_win) ready for model training."""
        X      = df[self.feature_cols].copy()
        y_runs = df["R"].values.astype(float)
        y_win  = df["WIN"].values.astype(int)
        if scale:
            X = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=self.feature_cols,
            )
        return X, y_runs, y_win

    # ── Private — Team Rolling ────────────────────────────────────────────────
    def _team_rolling_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute per-team rolling means for all TEAM_ROLL_STATS."""
        df = df.sort_values(["TEAM_ID", "GAME_DATE"])
        for col in self.TEAM_ROLL_STATS:
            if col not in df.columns:
                continue
            for win in self.TEAM_ROLL_WINDOWS:
                new_col = f"{col}_R{win}"
                df[new_col] = (
                    df.groupby("TEAM_ID")[col]
                      .transform(
                          lambda s, w=win: s.shift(1)
                            .rolling(w, min_periods=max(2, w // 2))
                            .mean()
                      )
                )
        return df

    # ── Private — SP Rolling ─────────────────────────────────────────────────
    def _sp_rolling_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Roll SP-level metrics over each pitcher's last N *starts*.

        The raw data_loader is expected to supply per-game SP stats in columns:
          SP_ID, SP_ERA, SP_WHIP, SP_K9, SP_BB9, SP_FIP, SP_IP, SP_ER,
          SP_HAND  (0=LHP, 1=RHP)

        For each rolling window W, this produces:
          SP_ERA_RS{W},  SP_WHIP_RS{W},  SP_K9_RS{W},  SP_FIP_RS{W}
        (suffix RS = "rolling starts" to distinguish from team-game rolls)
        """
        if "SP_ID" not in df.columns:
            log.warning(
                "SP_ID column absent — skipping SP rolling averages.  "
                "Ensure MLB_data_loader populates per-game starting pitcher rows."
            )
            return df

        df = df.sort_values(["SP_ID", "GAME_DATE"])
        for col in self.SP_ROLL_STATS:
            if col not in df.columns:
                continue
            for win in self.SP_ROLL_WINDOWS:
                new_col = f"{col}_RS{win}"
                df[new_col] = (
                    df.groupby("SP_ID")[col]
                      .transform(
                          lambda s, w=win: s.shift(1)
                            .rolling(w, min_periods=max(2, w // 2))
                            .mean()
                      )
                )
        return df

    # ── Private — Ratings ─────────────────────────────────────────────────────
    def _offensive_defensive_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Offensive Rating  = Runs per Plate Appearance (R / PA)  × 100
        Defensive Rating  = Opponent Runs per PA (OPP_R / PA)   × 100
        Net Rating        = Off - Def

        All calculated from rolling averages so they remain look-ahead free.
        """
        for win in self.TEAM_ROLL_WINDOWS:
            pa_col = f"PA_R{win}"
            r_col  = f"R_R{win}"
            or_col = f"OPP_R_R{win}"
            if pa_col not in df.columns:
                continue
            df[f"OFF_RTG_R{win}"] = (
                df[r_col].fillna(0) / df[pa_col].clip(lower=1) * 100
            )
            df[f"DEF_RTG_R{win}"] = (
                df[or_col].fillna(0) / df[pa_col].clip(lower=1) * 100
            )
            df[f"NET_RTG_R{win}"] = df[f"OFF_RTG_R{win}"] - df[f"DEF_RTG_R{win}"]
        return df

    # ── Private — Park Factors ────────────────────────────────────────────────
    def _apply_park_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Attach PARK_FACTOR (float centred on 1.0) for the home team's ballpark
        and derive PARK_ADJ_OFF_RTG for each rolling window.

        The adjustment multiplies the home team's raw offensive rating by the
        park factor and the away team's offensive rating by the *inverse* of the
        park factor — runs are harder (or easier) to score for *both* teams.
        """
        if "TEAM_ABV" not in df.columns:
            return df

        df["PARK_FACTOR"] = df.apply(
            lambda row: PARK_FACTORS.get(
                row["TEAM_ABV"] if row.get("HOME", 0) == 1 else row.get("OPP_TEAM_ABV", ""),
                1.0,
            ),
            axis=1,
        )

        for win in self.TEAM_ROLL_WINDOWS:
            off_col = f"OFF_RTG_R{win}"
            if off_col not in df.columns:
                continue
            # Scale team's own offensive output by the park factor.
            df[f"PARK_ADJ_OFF_RTG_R{win}"] = df[off_col] * df["PARK_FACTOR"]
        return df

    # ── Private — ePPA (expected runs per PA) ────────────────────────────────
    def _eppa(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Expected Runs Per PA (ePPA) — Poisson-style run-expectation proxy.

        Analogous to the NBA's ePPP:
          ePPA = √( OFF_RTG_team  ×  DEF_RTG_opponent )

        The geometric mean blends the team's run-scoring ability against the
        specific opponent's run-prevention to produce a matchup-adjusted λ.
        We use the park-adjusted offensive rating where available.
        """
        for win in self.TEAM_ROLL_WINDOWS:
            off_col = f"PARK_ADJ_OFF_RTG_R{win}" if f"PARK_ADJ_OFF_RTG_R{win}" in df.columns \
                      else f"OFF_RTG_R{win}"
            def_col = f"DEF_RTG_R{win}"
            if off_col not in df.columns or def_col not in df.columns:
                continue
            df[f"ePPA_R{win}"] = np.sqrt(
                df[off_col].clip(lower=0.001) * df[def_col].clip(lower=0.001)
            )
        return df

    # ── Private — Pitcher Handedness ─────────────────────────────────────────
    def _encode_pitcher_handedness(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode starting pitcher handedness as a binary integer feature.
          SP_HAND: 0 = LHP, 1 = RHP  (as supplied by data_loader)

        If SP_HAND is absent (e.g. older seasons with missing data), the column
        is filled with 1 (RHP majority) rather than being dropped, so the
        feature matrix remains rectangular.
        """
        if "SP_HAND" not in df.columns:
            log.warning(
                "SP_HAND column absent — defaulting to 1 (RHP) for all rows.  "
                "Populate SP_HAND in MLB_data_loader for handedness splits."
            )
            df["SP_HAND"] = 1
        else:
            # Data loader stores strings "LHP", "RHP", "UNK" — map to 0/1 before
            # casting.  "UNK" and any other unexpected value defaults to 1 (RHP,
            # the majority handedness) to match the absent-column fallback above.
            hand_map = {"LHP": 0, "RHP": 1}
            df["SP_HAND"] = (
                df["SP_HAND"]
                .map(hand_map)           # "LHP"→0, "RHP"→1, everything else→NaN
                .fillna(1)               # "UNK", NaN, already-numeric → 1
                .astype(int)
            )
        return df

    # ── Private — Targets ─────────────────────────────────────────────────────
    def _spread_and_totals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive regression targets from actual game outcomes.

        RAW_PROJ_SPREAD = actual run margin  (home team perspective)
        RAW_PROJ_TOTAL  = actual combined run total
        """
        df["RAW_PROJ_SPREAD"] = df.get("MARGIN",  df["R"] - df.get("OPP_R", 0))
        df["RAW_PROJ_TOTAL"]  = df.get("GAME_TOTAL", df["R"] + df.get("OPP_R", 0))
        return df

    # ── Private — Feature detection ───────────────────────────────────────────
    @staticmethod
    def _detect_feature_cols(df: pd.DataFrame) -> list[str]:
        """Auto-detect all numeric engineered columns to use as model inputs."""
        exclude = {
            "GAME_ID", "TEAM_ID", "OPP_TEAM_ID", "SP_ID", "OPP_SP_ID",
            "SEASON", "WIN", "WL", "R", "OPP_R", "MARGIN",
            "GAME_TOTAL", "RAW_PROJ_SPREAD", "RAW_PROJ_TOTAL",
        }
        return [
            c for c in df.columns
            if c not in exclude
            and df[c].dtype in [np.float64, np.int64, float, int]
            and not df[c].isna().all()
        ]