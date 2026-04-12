"""
backtest.py — Walk-Forward Backtesting Module
Evaluates InsideLine model accuracy by simulating historical predictions using
only data available at each prediction date (no look-ahead).

Usage
-----
Projection-only mode (model accuracy vs actuals):
    python backtest.py --sport MLB --start 2024-04-01 --end 2024-10-01

Scanner mode (grade-based record, requires historical odds CSV):
    python backtest.py --sport MLB --start 2024-04-01 --end 2024-10-01 \\
        --historical-odds odds.csv --output-csv results.csv

Parameter sweep (tests 16 combinations of market_blend × edge_shrink):
    python backtest.py --sport MLB --start 2024-04-01 --end 2024-10-01 \\
        --historical-odds odds.csv --sweep-params

Historical Odds CSV schema (for scanner mode):
    game_date, home_abv, away_abv, home_run_line, market_total, home_ml
    2024-04-15, NYY, BOS, -1.5, 8.5, -140

Retrain Frequencies:
    daily   — retrain before every single game date (slowest, most accurate)
    weekly  — retrain once per week (default, good balance)
    monthly — retrain once per month (fastest)
"""

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from itertools import product
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    sport:               str            = "MLB"
    start_date:          str            = "2024-04-01"
    end_date:            str            = "2024-10-01"
    warmup:              int            = 200
    retrain_freq:        str            = "weekly"    # daily | weekly | monthly
    market_blend_factor: float          = -1.0        # -1 = use default
    edge_shrink_factor:  float          = -1.0        # -1 = use default
    min_confidence:      float          = -1.0        # -1 = use default
    historical_odds_csv: Optional[str]  = None
    output_csv:          Optional[str]  = None
    force_refresh:       bool           = False


# Parameter grid for --sweep-params
SWEEP_GRID = {
    "market_blend_factor": [0.25, 0.35, 0.50, 0.60],
    "edge_shrink_factor":  [0.45, 0.55, 0.65, 0.75],
}


# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestResults:
    config:              BacktestConfig
    predictions:         list[dict]     = field(default_factory=list)
    total_games:         int            = 0
    # Projection accuracy (always computed)
    mae_runs:            float          = 0.0
    mae_total:           float          = 0.0
    mae_spread:          float          = 0.0
    win_accuracy:        float          = 0.0
    # Calibration
    calibration_deciles: list[dict]     = field(default_factory=list)
    # Scanner mode (only when historical_odds_csv supplied)
    plays:               list[dict]     = field(default_factory=list)
    records_by_grade:    dict           = field(default_factory=dict)
    records_by_type:     dict           = field(default_factory=dict)
    roi_by_grade:        dict           = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Main backtester class
# ─────────────────────────────────────────────────────────────────────────────

class WalkForwardBacktester:
    """
    Walk-forward backtesting engine.

    Design
    ------
    1. Load all historical data once (using cached DataLoader).
    2. Build features on the full dataset (shift(1) in rolling features
       guarantees no look-ahead bias at row level).
    3. For each retraining window:
       - Train models on all games BEFORE the window start.
       - Predict games WITHIN the window.
       - Compare predictions to actuals.
    """

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config
        self._sport = config.sport.upper()
        if self._sport not in ("MLB", "NBA"):
            raise ValueError(f"Unknown sport '{config.sport}'. Use MLB or NBA.")

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self) -> BacktestResults:
        results = BacktestResults(config=self.config)

        log.info(f"{'='*60}")
        log.info(f"  Walk-Forward Backtest: {self._sport}")
        log.info(f"  Date range : {self.config.start_date} → {self.config.end_date}")
        log.info(f"  Retrain    : {self.config.retrain_freq}")
        log.info(f"  Warmup     : {self.config.warmup} games")
        log.info(f"{'='*60}")

        # Load and build features once
        df_all = self._load_and_build()
        if df_all is None or df_all.empty:
            log.error("No data loaded — aborting backtest.")
            return results

        df_all["GAME_DATE"] = pd.to_datetime(df_all["GAME_DATE"])
        start = pd.Timestamp(self.config.start_date)
        end   = pd.Timestamp(self.config.end_date)

        # Load historical odds if provided
        odds_df = self._load_odds()

        # Walk-forward loop
        all_preds = []
        retrain_boundaries = self._retrain_boundaries(df_all, start, end)

        for i, (window_start, window_end) in enumerate(retrain_boundaries):
            train_df = df_all[df_all["GAME_DATE"] < window_start]
            if len(train_df) < self.config.warmup:
                log.info(f"Skipping window starting {window_start.date()} "
                         f"— only {len(train_df)} training rows (warmup={self.config.warmup})")
                continue

            log.info(f"[{i+1}/{len(retrain_boundaries)}] "
                     f"Training on {len(train_df):,} rows, "
                     f"predicting {window_start.date()} → {window_end.date()}")

            trainer, engine = self._retrain(train_df)
            if trainer is None:
                log.warning(f"Training failed for window {window_start.date()} — skipping.")
                continue

            window_df = df_all[
                (df_all["GAME_DATE"] >= window_start) &
                (df_all["GAME_DATE"] <= window_end)
            ]

            for pred_date, day_df in window_df.groupby("GAME_DATE"):
                day_preds = self._predict_day(pred_date, day_df, train_df, trainer, engine, odds_df)
                all_preds.extend(day_preds)

        results.predictions = all_preds
        results.total_games = len(set(p["game_id"] for p in all_preds))

        if all_preds:
            results = self._evaluate(results)

        return results

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_and_build(self):
        """Load data and build features for the target sport."""
        try:
            if self._sport == "MLB":
                from MLB.MLB_data_loader import MLBDataLoader
                from MLB.MLB_features    import FeatureEngine
                loader = MLBDataLoader(force_refresh=self.config.force_refresh)
            else:
                from NBA.data_loader import DataLoader as NBADataLoader
                from NBA.features    import FeatureEngine
                loader = NBADataLoader(force_refresh=self.config.force_refresh)

            log.info("Loading historical data …")
            df_raw = loader.load()

            log.info("Building feature matrix …")
            engine = FeatureEngine()
            df = engine.build(df_raw)

            # Store engine on self for later use
            self._engine = engine
            return df
        except Exception as exc:
            log.error(f"Data load/build failed: {exc}", exc_info=True)
            return None

    def _load_odds(self) -> Optional[pd.DataFrame]:
        if not self.config.historical_odds_csv:
            return None
        try:
            odds = pd.read_csv(self.config.historical_odds_csv)
            odds["game_date"] = pd.to_datetime(odds["game_date"])
            odds["home_abv"]  = odds["home_abv"].str.upper().str.strip()
            odds["away_abv"]  = odds["away_abv"].str.upper().str.strip()
            log.info(f"Loaded {len(odds):,} historical odds rows from {self.config.historical_odds_csv}")
            return odds
        except Exception as exc:
            log.warning(f"Could not load odds CSV: {exc}")
            return None

    # ── Retraining schedule ───────────────────────────────────────────────────

    def _retrain_boundaries(
        self,
        df:    pd.DataFrame,
        start: pd.Timestamp,
        end:   pd.Timestamp,
    ) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """Return (window_start, window_end) pairs for the retrain frequency."""
        freq = self.config.retrain_freq
        boundaries = []
        cur = start
        while cur <= end:
            if freq == "daily":
                nxt = cur + pd.Timedelta(days=1)
            elif freq == "monthly":
                # Jump to first of next month
                if cur.month == 12:
                    nxt = pd.Timestamp(cur.year + 1, 1, 1)
                else:
                    nxt = pd.Timestamp(cur.year, cur.month + 1, 1)
            else:  # weekly (default)
                nxt = cur + pd.Timedelta(days=7)
            boundaries.append((cur, min(nxt - pd.Timedelta(days=1), end)))
            cur = nxt
        return boundaries

    # ── Model training ────────────────────────────────────────────────────────

    def _retrain(self, train_df: pd.DataFrame):
        """Train models on train_df. Returns (trainer, engine) or (None, None)."""
        try:
            engine = self._engine
            X, y_score, y_win = engine.get_feature_matrix(train_df)

            if self._sport == "MLB":
                from MLB.MLB_models import ModelTrainer
            else:
                from NBA.models import ModelTrainer

            trainer = ModelTrainer()
            trainer.train(X, y_score, y_win)
            return trainer, engine
        except Exception as exc:
            log.error(f"Retrain failed: {exc}", exc_info=True)
            return None, None

    # ── Per-day prediction ────────────────────────────────────────────────────

    def _predict_day(
        self,
        pred_date: pd.Timestamp,
        day_df:    pd.DataFrame,
        train_df:  pd.DataFrame,
        trainer,
        engine,
        odds_df:   Optional[pd.DataFrame],
    ) -> list[dict]:
        """Generate predictions for all games on pred_date."""
        preds = []
        engine = self._engine

        # Group by game
        for game_id, game_rows in day_df.groupby("GAME_ID"):
            home_rows = game_rows[game_rows["HOME"] == 1]
            away_rows = game_rows[game_rows["HOME"] == 0]

            if home_rows.empty or away_rows.empty:
                continue

            home_row = home_rows.iloc[0]
            away_row = away_rows.iloc[0]

            home_abv = str(home_row["TEAM_ABV"])
            away_abv = str(away_row["TEAM_ABV"])

            # Build feature vectors from the training set's most recent snapshot
            home_feat = train_df[train_df["TEAM_ABV"] == home_abv]
            away_feat = train_df[train_df["TEAM_ABV"] == away_abv]

            if home_feat.empty or away_feat.empty:
                continue

            home_X = home_feat.sort_values("GAME_DATE").iloc[[-1]][engine.feature_cols]
            away_X = away_feat.sort_values("GAME_DATE").iloc[[-1]][engine.feature_cols]

            try:
                if self._sport == "MLB":
                    home_pred = float(trainer.predict_runs(home_X)[0])
                    away_pred = float(trainer.predict_runs(away_X)[0])
                else:
                    home_pred = float(trainer.predict_pts(home_X)[0])
                    away_pred = float(trainer.predict_pts(away_X)[0])

                home_win_prob = float(trainer.predict_win_prob(home_X)[0])
            except Exception as exc:
                log.warning(f"Inference failed for {home_abv} vs {away_abv}: {exc}")
                continue

            actual_home = int(home_row["R" if self._sport == "MLB" else "PTS"])
            actual_away = int(away_row["R" if self._sport == "MLB" else "PTS"])
            actual_win  = int(home_row["WIN"])

            pred = {
                "game_id":        game_id,
                "game_date":      pred_date.date(),
                "home_abv":       home_abv,
                "away_abv":       away_abv,
                "pred_home":      round(home_pred, 2),
                "pred_away":      round(away_pred, 2),
                "pred_total":     round(home_pred + away_pred, 2),
                "pred_spread":    round(home_pred - away_pred, 2),
                "home_win_prob":  round(home_win_prob, 4),
                "actual_home":    actual_home,
                "actual_away":    actual_away,
                "actual_total":   actual_home + actual_away,
                "actual_spread":  actual_home - actual_away,
                "actual_win":     actual_win,
                "plays":          [],
            }

            # Scanner mode: add plays if odds are available
            if odds_df is not None:
                game_date_str = pred_date.date()
                match = odds_df[
                    (odds_df["game_date"].dt.date == game_date_str) &
                    (odds_df["home_abv"] == home_abv) &
                    (odds_df["away_abv"] == away_abv)
                ]
                if not match.empty:
                    odds_row = match.iloc[0]
                    plays = self._run_scanner(pred, odds_row)
                    pred["plays"] = plays

            preds.append(pred)

        return preds

    def _run_scanner(self, pred: dict, odds_row: pd.Series) -> list[dict]:
        """Run the value scanner against historical odds and return plays."""
        try:
            market_run_line = float(odds_row.get("home_run_line", -1.5))
            market_total    = float(odds_row.get("market_total",  9.0))
            market_ml_home  = int(odds_row.get("home_ml",         -110))

            config = self.config
            scanner_kwargs = {}
            if config.market_blend_factor >= 0:
                scanner_kwargs["market_blend_factor"] = config.market_blend_factor
            if config.edge_shrink_factor >= 0:
                if self._sport == "MLB":
                    scanner_kwargs["edge_shrink_model"] = config.edge_shrink_factor
                else:
                    scanner_kwargs["edge_shrink_factor"] = config.edge_shrink_factor
            if config.min_confidence >= 0:
                scanner_kwargs["edge_total_min"]  = config.min_confidence
                scanner_kwargs["edge_spread_min"] = config.min_confidence

            if self._sport == "MLB":
                from MLB.MLB_scanner import ValueScanner
                scanner = ValueScanner(**scanner_kwargs)

                # Build ePPA dicts — use predicted runs as proxy when rolling ePPA unavailable
                home_eppas = {5: pred["pred_home"], 10: pred["pred_home"], 20: pred["pred_home"]}
                away_eppas = {5: pred["pred_away"], 10: pred["pred_away"], 20: pred["pred_away"]}

                result = scanner.scan(
                    home_pred_runs  = pred["pred_home"],
                    away_pred_runs  = pred["pred_away"],
                    home_win_prob   = pred["home_win_prob"],
                    home_eppas      = home_eppas,
                    away_eppas      = away_eppas,
                    home_pa         = 36.0,
                    away_pa         = 36.0,
                    market_run_line = market_run_line,
                    market_total    = market_total,
                    market_ml_home  = market_ml_home,
                )
            else:
                from NBA.scanner import ValueScanner
                scanner = ValueScanner(**scanner_kwargs)

                home_eppps = {5: pred["pred_home"] / 100, 10: pred["pred_home"] / 100, 20: pred["pred_home"] / 100}
                away_eppps = {5: pred["pred_away"] / 100, 10: pred["pred_away"] / 100, 20: pred["pred_away"] / 100}

                market_spread  = float(odds_row.get("home_run_line", -3.0))
                market_ml_away = int(odds_row.get("away_ml", 110))

                result = scanner.scan(
                    home_pred_pts  = pred["pred_home"],
                    away_pred_pts  = pred["pred_away"],
                    home_win_prob  = pred["home_win_prob"],
                    home_eppps     = home_eppps,
                    away_eppps     = away_eppps,
                    home_pace      = 100.0,
                    away_pace      = 100.0,
                    market_spread  = market_spread,
                    market_total   = market_total,
                    market_ml_home = market_ml_home,
                    market_ml_away = market_ml_away,
                )

            plays = result.get("plays", [])

            # Tag each play with outcome
            for play in plays:
                play["actual_win"]    = pred["actual_win"]
                play["actual_spread"] = pred["actual_spread"]
                play["actual_total"]  = pred["actual_total"]
                play["game_id"]       = pred["game_id"]
                play["game_date"]     = pred["game_date"]
                play["home_abv"]      = pred["home_abv"]
                play["away_abv"]      = pred["away_abv"]
                play["outcome"]       = self._resolve_outcome(play)

            return plays
        except Exception as exc:
            log.warning(f"Scanner failed for {pred['home_abv']} vs {pred['away_abv']}: {exc}")
            return []

    @staticmethod
    def _resolve_outcome(play: dict) -> str:
        """Determine WIN/LOSS/PUSH for a play given actual game results."""
        side         = play.get("side", "")
        bet_type     = play.get("type", "")
        actual_spread = play.get("actual_spread", 0)
        actual_total  = play.get("actual_total", 0)
        actual_win    = play.get("actual_win", 0)

        if bet_type == "MONEYLINE":
            if side == "HOME":
                return "WIN" if actual_win == 1 else "LOSS"
            else:
                return "WIN" if actual_win == 0 else "LOSS"

        elif bet_type == "SPREAD":
            # MLB: fixed run line ±1.5
            if side == "HOME":
                covered = actual_spread > 1.5
            else:
                covered = actual_spread < -1.5
            if abs(actual_spread) == 1.5:
                return "PUSH"
            return "WIN" if covered else "LOSS"

        elif bet_type == "TOTAL":
            market_line = play.get("market_line", 0)
            if actual_total == market_line:
                return "PUSH"
            if side == "OVER":
                return "WIN" if actual_total > market_line else "LOSS"
            else:
                return "WIN" if actual_total < market_line else "LOSS"

        return "UNKNOWN"

    # ── Evaluation ────────────────────────────────────────────────────────────

    def _evaluate(self, results: BacktestResults) -> BacktestResults:
        preds = results.predictions
        if not preds:
            return results

        actual_home  = np.array([p["actual_home"]  for p in preds])
        pred_home    = np.array([p["pred_home"]     for p in preds])
        actual_total = np.array([p["actual_total"]  for p in preds])
        pred_total   = np.array([p["pred_total"]    for p in preds])
        actual_spread= np.array([p["actual_spread"] for p in preds])
        pred_spread  = np.array([p["pred_spread"]   for p in preds])
        actual_win   = np.array([p["actual_win"]    for p in preds])
        home_win_prob= np.array([p["home_win_prob"] for p in preds])

        results.mae_runs   = float(np.mean(np.abs(actual_home  - pred_home)))
        results.mae_total  = float(np.mean(np.abs(actual_total - pred_total)))
        results.mae_spread = float(np.mean(np.abs(actual_spread - pred_spread)))
        results.win_accuracy = float(np.mean((pred_spread > 0) == (actual_win == 1)))

        # Calibration: split into 10 decile bins by predicted win prob
        calibration = []
        bin_edges = np.linspace(0.0, 1.0, 11)
        for i in range(10):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (home_win_prob >= lo) & (home_win_prob < hi)
            if mask.sum() == 0:
                continue
            calibration.append({
                "bin":           f"{lo:.1f}–{hi:.1f}",
                "n":             int(mask.sum()),
                "pred_win_rate": float(home_win_prob[mask].mean()),
                "actual_win_rate": float(actual_win[mask].mean()),
            })
        results.calibration_deciles = calibration

        # Scanner mode metrics
        all_plays = [p for pred in preds for p in pred.get("plays", [])]
        if all_plays:
            results.plays = all_plays
            results.records_by_grade = self._record_by_key(all_plays, "grade")
            results.records_by_type  = self._record_by_key(all_plays, "type")
            results.roi_by_grade     = self._roi_by_grade(all_plays)

        return results

    @staticmethod
    def _record_by_key(plays: list[dict], key: str) -> dict:
        records = {}
        for play in plays:
            k = play.get(key, "?")
            if k not in records:
                records[k] = {"W": 0, "L": 0, "P": 0}
            outcome = play.get("outcome", "UNKNOWN")
            if outcome == "WIN":
                records[k]["W"] += 1
            elif outcome == "LOSS":
                records[k]["L"] += 1
            elif outcome == "PUSH":
                records[k]["P"] += 1
        return records

    @staticmethod
    def _roi_by_grade(plays: list[dict]) -> dict:
        roi = {}
        for play in plays:
            grade   = play.get("grade", "?")
            kelly   = float(play.get("kelly_$", 0))
            outcome = play.get("outcome", "UNKNOWN")
            if grade not in roi:
                roi[grade] = {"wagered": 0.0, "returned": 0.0}
            if outcome == "WIN":
                roi[grade]["wagered"]  += kelly
                roi[grade]["returned"] += kelly * 1.909  # ~-110 juice assumed
            elif outcome == "LOSS":
                roi[grade]["wagered"]  += kelly
                roi[grade]["returned"] += 0.0
            elif outcome == "PUSH":
                roi[grade]["wagered"]  += kelly
                roi[grade]["returned"] += kelly
        result = {}
        for g, vals in roi.items():
            if vals["wagered"] > 0:
                result[g] = round((vals["returned"] - vals["wagered"]) / vals["wagered"], 4)
            else:
                result[g] = 0.0
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Output rendering
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(results: BacktestResults) -> None:
    config = results.config
    sport  = config.sport.upper()
    unit   = "runs" if sport == "MLB" else "pts"

    print()
    print("═" * 62)
    print(f"  InsideLine Backtest Results — {sport}")
    print(f"  {config.start_date} → {config.end_date}  |  "
          f"warmup={config.warmup}  |  retrain={config.retrain_freq}")
    print("═" * 62)

    print(f"\n  Games predicted : {results.total_games:,}")
    print(f"  MAE ({unit}/game) : {results.mae_runs:.3f}")
    print(f"  MAE (total)     : {results.mae_total:.3f}")
    print(f"  MAE (spread)    : {results.mae_spread:.3f}")
    print(f"  Win direction   : {results.win_accuracy:.1%}")

    if results.calibration_deciles:
        print(f"\n  Calibration (predicted → actual win rate):")
        print(f"  {'Bin':>10}  {'N':>6}  {'Pred':>8}  {'Actual':>8}  {'Δ':>8}")
        print("  " + "-" * 46)
        for row in results.calibration_deciles:
            delta = row["actual_win_rate"] - row["pred_win_rate"]
            print(f"  {row['bin']:>10}  {row['n']:>6,}  "
                  f"{row['pred_win_rate']:>8.1%}  "
                  f"{row['actual_win_rate']:>8.1%}  "
                  f"{delta:>+8.1%}")

    if results.records_by_grade:
        print(f"\n  Record by Grade:")
        print(f"  {'Grade':>6}  {'W':>5}  {'L':>5}  {'P':>5}  {'Win%':>8}  {'ROI':>8}")
        print("  " + "-" * 44)
        for grade in sorted(results.records_by_grade.keys()):
            rec = results.records_by_grade[grade]
            w, l, p = rec["W"], rec["L"], rec["P"]
            total = w + l
            win_pct = w / total if total else 0.0
            roi = results.roi_by_grade.get(grade, 0.0)
            print(f"  {grade:>6}  {w:>5}  {l:>5}  {p:>5}  "
                  f"{win_pct:>8.1%}  {roi:>+8.1%}")

    if results.records_by_type:
        print(f"\n  Record by Bet Type:")
        print(f"  {'Type':>10}  {'W':>5}  {'L':>5}  {'P':>5}  {'Win%':>8}")
        print("  " + "-" * 40)
        for btype in sorted(results.records_by_type.keys()):
            rec = results.records_by_type[btype]
            w, l, p = rec["W"], rec["L"], rec["P"]
            total = w + l
            win_pct = w / total if total else 0.0
            print(f"  {btype:>10}  {w:>5}  {l:>5}  {p:>5}  {win_pct:>8.1%}")

    print()
    print("═" * 62)


def _print_sweep_table(sweep_results: list[dict]) -> None:
    print()
    print("═" * 70)
    print("  Parameter Sweep Results")
    print("═" * 70)
    print(f"  {'Blend':>6}  {'Shrink':>6}  "
          f"{'MAE':>6}  {'WinAcc':>8}  {'Plays':>6}  {'ROI-A':>8}  {'ROI-B':>8}")
    print("  " + "-" * 58)
    for row in sorted(sweep_results, key=lambda r: r.get("roi_a", -99), reverse=True):
        print(f"  {row['blend']:>6.2f}  {row['shrink']:>6.2f}  "
              f"{row['mae']:>6.3f}  {row['win_acc']:>8.1%}  "
              f"{row['n_plays']:>6}  "
              f"{row['roi_a']:>+8.1%}  {row['roi_b']:>+8.1%}")
    print("═" * 70)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> tuple[BacktestConfig, bool]:
    parser = argparse.ArgumentParser(
        description="InsideLine Walk-Forward Backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sport",            default="MLB",   choices=["MLB", "NBA"])
    parser.add_argument("--start",            default="2024-04-01",  dest="start_date")
    parser.add_argument("--end",              default="2024-10-01",  dest="end_date")
    parser.add_argument("--warmup",           default=200,     type=int)
    parser.add_argument("--retrain-freq",     default="weekly",
                        choices=["daily", "weekly", "monthly"], dest="retrain_freq")
    parser.add_argument("--historical-odds",  default=None,    dest="historical_odds_csv")
    parser.add_argument("--output-csv",       default=None,    dest="output_csv")
    parser.add_argument("--market-blend",     default=-1.0,    type=float, dest="market_blend_factor")
    parser.add_argument("--edge-shrink",      default=-1.0,    type=float, dest="edge_shrink_factor")
    parser.add_argument("--min-confidence",   default=-1.0,    type=float, dest="min_confidence")
    parser.add_argument("--force-refresh",    action="store_true", dest="force_refresh")
    parser.add_argument("--sweep-params",     action="store_true", dest="sweep_params")

    args = parser.parse_args()
    config = BacktestConfig(
        sport               = args.sport,
        start_date          = args.start_date,
        end_date            = args.end_date,
        warmup              = args.warmup,
        retrain_freq        = args.retrain_freq,
        market_blend_factor = args.market_blend_factor,
        edge_shrink_factor  = args.edge_shrink_factor,
        min_confidence      = args.min_confidence,
        historical_odds_csv = args.historical_odds_csv,
        output_csv          = args.output_csv,
        force_refresh       = args.force_refresh,
    )
    return config, args.sweep_params


def _save_csv(results: BacktestResults, path: str) -> None:
    rows = []
    for pred in results.predictions:
        base = {k: v for k, v in pred.items() if k != "plays"}
        plays = pred.get("plays", [])
        if plays:
            for play in plays:
                rows.append({**base, **play})
        else:
            rows.append(base)
    pd.DataFrame(rows).to_csv(path, index=False)
    log.info(f"Results saved to {path}")


def main() -> None:
    config, sweep_params = _parse_args()

    if sweep_params:
        if not config.historical_odds_csv:
            log.warning("--sweep-params works best with --historical-odds; "
                        "running projection-only sweep.")

        # Build shared data once by running a dummy backtester
        backtester = WalkForwardBacktester(config)
        df_all = backtester._load_and_build()
        if df_all is None:
            sys.exit(1)

        sweep_results = []
        blends  = SWEEP_GRID["market_blend_factor"]
        shrinks = SWEEP_GRID["edge_shrink_factor"]
        total   = len(blends) * len(shrinks)

        for i, (blend, shrink) in enumerate(product(blends, shrinks), 1):
            log.info(f"Sweep [{i}/{total}]: market_blend={blend}  edge_shrink={shrink}")
            sweep_config = BacktestConfig(
                sport               = config.sport,
                start_date          = config.start_date,
                end_date            = config.end_date,
                warmup              = config.warmup,
                retrain_freq        = config.retrain_freq,
                market_blend_factor = blend,
                edge_shrink_factor  = shrink,
                historical_odds_csv = config.historical_odds_csv,
                force_refresh       = False,
            )
            bt = WalkForwardBacktester(sweep_config)
            # Reuse already-loaded engine
            bt._engine = backtester._engine

            # Re-run walk-forward on already-built df_all
            results = BacktestResults(config=sweep_config)
            df_all_copy = df_all.copy()
            df_all_copy["GAME_DATE"] = pd.to_datetime(df_all_copy["GAME_DATE"])
            start = pd.Timestamp(config.start_date)
            end   = pd.Timestamp(config.end_date)
            odds_df = bt._load_odds()
            all_preds = []
            for window_start, window_end in bt._retrain_boundaries(df_all_copy, start, end):
                train_df = df_all_copy[df_all_copy["GAME_DATE"] < window_start]
                if len(train_df) < config.warmup:
                    continue
                trainer, engine = bt._retrain(train_df)
                if trainer is None:
                    continue
                window_df = df_all_copy[
                    (df_all_copy["GAME_DATE"] >= window_start) &
                    (df_all_copy["GAME_DATE"] <= window_end)
                ]
                for pred_date, day_df in window_df.groupby("GAME_DATE"):
                    all_preds.extend(bt._predict_day(pred_date, day_df, train_df,
                                                     trainer, engine, odds_df))
            results.predictions = all_preds
            results.total_games = len(set(p["game_id"] for p in all_preds))
            if all_preds:
                results = bt._evaluate(results)

            all_plays = [p for pred in all_preds for p in pred.get("plays", [])]
            roi_a = results.roi_by_grade.get("A", 0.0)
            roi_b = results.roi_by_grade.get("B", 0.0)
            sweep_results.append({
                "blend":   blend,
                "shrink":  shrink,
                "mae":     results.mae_runs,
                "win_acc": results.win_accuracy,
                "n_plays": len(all_plays),
                "roi_a":   roi_a,
                "roi_b":   roi_b,
            })

        _print_sweep_table(sweep_results)
        return

    # Single run
    backtester = WalkForwardBacktester(config)
    results    = backtester.run()
    _print_summary(results)

    if config.output_csv:
        _save_csv(results, config.output_csv)


if __name__ == "__main__":
    main()
