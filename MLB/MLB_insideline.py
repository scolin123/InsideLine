"""
MLB_insideLine.py — MODULE 0: OddsClient  +  Pipeline orchestration  +  Supabase storage

  OddsClient          : HTTP client for The-Odds-API v4 baseball_mlb endpoint
  PipelineArtifacts   : Shared, pre-trained state bundle (DataFrame + engine + trainer)
  build_pipeline_artifacts(): Expensive one-time data load + model training
  run_pipeline()      : Per-game inference — uses PipelineArtifacts to skip re-training
  init_supabase()     : Initialise Supabase client from environment credentials
  save_to_supabase()  : Persist flagged plays to the `mlb_projections` table

MLB-specific changes vs. NBA version
──────────────────────────────────────
• OddsClient._ENDPOINT points to baseball_mlb.
• GameOdds.home_run_line (fixed ±1.5) replaces the floating home_spread.
  home_spread is retained as a property alias returning home_run_line so
  that downstream scanner code remains uniform.
• run_pipeline() extracts ePPA (expected runs per plate appearance) instead
  of ePPP, and uses home_pa / away_pa (plate appearances) for the Poisson
  projection.
• Supabase table name is `mlb_projections` to keep NBA and MLB rows separate.
"""
import os
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Optional

import numpy as np
import requests
from supabase import create_client, Client as SupabaseClient

from .MLB_config import (
    MLB_ODDS_API_BASE_URL,
    ODDS_API_KEY,
    ODDS_QUOTA_MIN,
    ODDS_REGIONS,
    ODDS_MARKETS,
    ODDS_FORMAT,
    ODDS_TEAM_NAME_MAP,
    SUPABASE_URL,
    SUPABASE_KEY,
)
from .MLB_data_loader import DataLoader
from .MLB_features    import FeatureEngine
from .MLB_models      import ModelTrainer
from .MLB_scanner     import ValueScanner, print_projection

log = logging.getLogger(__name__)

# Supabase table for MLB projections (separate from NBA `projections` table)
_MLB_SUPABASE_TABLE = "mlb_projections"


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 0 — ODDS CLIENT  (MLB)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class GameOdds:
    """
    Structured container for a single MLB game's best available lines.

    MLB market conventions
    ──────────────────────
    • Moneyline  — highest (least negative) American ML for each side.
    • Run Line   — fixed at ±1.5 runs.  home_run_line = −1.5 if home is the
                   favourite, +1.5 if the home team is the underdog.
    • Total      — consensus mid-point of the best Over and Under lines.

    Note: home_spread is a property alias for home_run_line so the scanner
    can treat run-line and spread plays identically.
    """
    home_abv:        str
    away_abv:        str
    home_ml:         Optional[int]   = None
    away_ml:         Optional[int]   = None
    home_run_line:   Optional[float] = None   # always ±1.5 when present
    run_line_juice:  Optional[int]   = None
    total_over:      Optional[float] = None
    total_under:     Optional[float] = None
    over_juice:      Optional[int]   = None
    under_juice:     Optional[int]   = None
    bookmakers_used: list[str]       = field(default_factory=list)
    commence_time:   Optional[str]   = None   # ISO 8601 UTC from Odds API

    # ── Convenience aliases ───────────────────────────────────────────────────
    @property
    def home_spread(self) -> Optional[float]:
        """Alias: run line = spread in this context."""
        return self.home_run_line

    @property
    def spread_juice(self) -> Optional[int]:
        """Alias: run-line juice = spread juice in this context."""
        return self.run_line_juice

    @property
    def consensus_total(self) -> Optional[float]:
        """Mid-point of best Over and best Under lines."""
        if self.total_over is not None and self.total_under is not None:
            return round((self.total_over + self.total_under) / 2, 1)
        return self.total_over or self.total_under


class OddsClient:
    """
    Thin HTTP client for The-Odds-API v4 (baseball_mlb endpoint).

    Responsibilities
    ────────────────
    1. Fire a single GET /v4/sports/baseball_mlb/odds request.
    2. Guard the monthly quota — abort if ≤ ODDS_QUOTA_MIN requests remain.
    3. Parse the JSON response into a list of GameOdds dataclasses.
    4. Expose a .find() method to locate a specific home/away matchup.
    """

    _ENDPOINT = MLB_ODDS_API_BASE_URL   # https://api.the-odds-api.com/v4/sports/baseball_mlb/odds

    def __init__(
        self,
        api_key:         str = ODDS_API_KEY,
        quota_min:       int = ODDS_QUOTA_MIN,
        request_timeout: int = 10,
    ) -> None:
        self.api_key           = os.environ.get("ODDS_API_KEY", api_key)
        self.quota_min         = quota_min
        self.timeout           = request_timeout
        self._quota_remaining: Optional[int]  = None
        self._quota_used:      Optional[int]  = None
        self._games:           list[GameOdds] = []

    # ── Public ────────────────────────────────────────────────────────────────
    def fetch(self) -> list[GameOdds]:
        """Fetch live MLB odds and return a list of GameOdds."""
        if not self.api_key or self.api_key == "YOUR_ODDS_API_KEY_HERE":
            raise RuntimeError(
                "OddsClient: No API key set. Pass api_key= or set the "
                "ODDS_API_KEY environment variable."
            )

        self._pre_flight_quota_check()

        params = {
            "apiKey":     self.api_key,
            "regions":    ODDS_REGIONS,
            "markets":    ODDS_MARKETS,
            "oddsFormat": ODDS_FORMAT,
        }

        log.info("OddsClient → fetching live MLB lines from The-Odds-API …")
        try:
            resp = requests.get(
                self._ENDPOINT, params=params, timeout=self.timeout
            )
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"OddsClient: request timed out after {self.timeout}s."
            )
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(f"OddsClient: connection error — {exc}")

        self._parse_quota_headers(resp.headers)
        self._check_response_status(resp)

        raw_games: list[dict] = resp.json()
        self._games = [self._parse_game(g) for g in raw_games if g]
        self._games = [g for g in self._games if g.home_abv and g.away_abv]

        log.info(
            f"OddsClient → {len(self._games)} games parsed. "
            f"Quota: {self._quota_remaining} requests remaining "
            f"({self._quota_used} used this cycle)."
        )
        return self._games

    def find(self, home_abv: str, away_abv: str) -> Optional[GameOdds]:
        """Locate the GameOdds entry matching the requested matchup."""
        h = home_abv.upper().strip()
        a = away_abv.upper().strip()
        for game in self._games:
            if game.home_abv.upper() == h and game.away_abv.upper() == a:
                return game
        return None

    @property
    def quota_remaining(self) -> Optional[int]:
        return self._quota_remaining

    @property
    def quota_used(self) -> Optional[int]:
        return self._quota_used

    # ── Private helpers ───────────────────────────────────────────────────────
    def _pre_flight_quota_check(self) -> None:
        if (
            self._quota_remaining is not None
            and self._quota_remaining <= self.quota_min
        ):
            raise RuntimeError(
                f"OddsClient: only {self._quota_remaining} API requests "
                f"remaining (minimum threshold: {self.quota_min}). "
                "Aborting to preserve monthly quota."
            )

    def _parse_quota_headers(
        self, headers: requests.structures.CaseInsensitiveDict
    ) -> None:
        try:
            self._quota_remaining = int(headers.get("x-requests-remaining", -1))
            self._quota_used      = int(headers.get("x-requests-used",      -1))
        except (TypeError, ValueError):
            log.warning("OddsClient: could not parse quota headers.")

        if (
            self._quota_remaining is not None
            and 0 <= self._quota_remaining <= self.quota_min
        ):
            log.warning(
                f"⚠  OddsClient: quota critically low — "
                f"{self._quota_remaining} requests remaining!"
            )

    @staticmethod
    def _check_response_status(resp: requests.Response) -> None:
        if resp.status_code == 200:
            return
        status_messages = {
            401: "Invalid or missing API key.",
            422: "Unprocessable request — check query parameters.",
            429: "Rate limit exceeded — too many requests per minute.",
        }
        detail = status_messages.get(
            resp.status_code,
            f"Unexpected HTTP {resp.status_code}.",
        )
        raise RuntimeError(
            f"OddsClient: {detail}  Body: {resp.text[:300]}"
        )

    def _parse_game(self, raw: dict) -> GameOdds:
        """Convert one raw game dict from The-Odds-API JSON into a GameOdds."""
        home_full = raw.get("home_team", "")
        away_full = raw.get("away_team", "")
        home_abv  = ODDS_TEAM_NAME_MAP.get(home_full, "")
        away_abv  = ODDS_TEAM_NAME_MAP.get(away_full, "")

        if not home_abv or not away_abv:
            log.debug(
                f"OddsClient: unmapped team name(s): '{home_full}' / '{away_full}'. "
                "Update ODDS_TEAM_NAME_MAP in MLB_config.py if this is a new franchise."
            )

        game = GameOdds(home_abv=home_abv, away_abv=away_abv)

        raw_ct = raw.get("commence_time", "")
        if raw_ct:
            try:
                dt_utc = datetime.fromisoformat(raw_ct.replace("Z", "+00:00"))
                dt_et  = dt_utc.astimezone(ZoneInfo("America/New_York"))
                day    = dt_et.day
                if 11 <= (day % 100) <= 13:
                    suffix = "th"
                else:
                    suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
                hour   = dt_et.hour % 12 or 12
                minute = dt_et.strftime("%M")
                game.commence_time = f"{dt_et.strftime('%B')} {day}{suffix} @ {hour}:{minute}"
            except Exception:
                game.commence_time = raw_ct

        home_mls:       list[int]   = []
        away_mls:       list[int]   = []
        home_run_lines: list[float] = []
        run_line_juices: list[int]  = []
        overs:          list[float] = []
        unders:         list[float] = []
        over_juices:    list[int]   = []
        under_juices:   list[int]   = []

        for book in raw.get("bookmakers", []):
            bk_key = book.get("key", "unknown")
            game.bookmakers_used.append(bk_key)

            for market in book.get("markets", []):
                mkey     = market.get("key")
                outcomes = market.get("outcomes", [])

                if mkey == "h2h":
                    self._collect_h2h(outcomes, home_full, home_mls, away_mls)
                elif mkey == "spreads":
                    # MLB spread market is the run line (always ±1.5)
                    self._collect_spreads(
                        outcomes, home_full, home_run_lines, run_line_juices
                    )
                elif mkey == "totals":
                    self._collect_totals(
                        outcomes, overs, unders, over_juices, under_juices
                    )

        # ── Best-line aggregation ─────────────────────────────────────────────
        game.home_ml = max(home_mls) if home_mls else None
        game.away_ml = max(away_mls) if away_mls else None

        if home_run_lines:
            # Run line is always ±1.5; pick the one closest to 0 for best juice.
            idx                = int(np.argmin(np.abs(home_run_lines)))
            game.home_run_line = home_run_lines[idx]
            game.run_line_juice = run_line_juices[idx] if run_line_juices else None

        if overs:
            idx             = int(np.argmax(overs))
            game.total_over = overs[idx]
            game.over_juice = over_juices[idx] if over_juices else None

        if unders:
            idx              = int(np.argmin(unders))
            game.total_under = unders[idx]
            game.under_juice = under_juices[idx] if under_juices else None

        return game

    # ── Market collectors (identical signatures to NBA version) ───────────────
    @staticmethod
    def _collect_h2h(
        outcomes:  list[dict],
        home_full: str,
        home_mls:  list[int],
        away_mls:  list[int],
    ) -> None:
        for o in outcomes:
            price = o.get("price")
            if price is None:
                continue
            if o.get("name") == home_full:
                home_mls.append(int(price))
            else:
                away_mls.append(int(price))

    @staticmethod
    def _collect_spreads(
        outcomes:       list[dict],
        home_full:      str,
        home_run_lines: list[float],
        run_line_juices: list[int],
    ) -> None:
        """Collect run-line (spread) outcomes for the home team."""
        for o in outcomes:
            if o.get("name") == home_full:
                point = o.get("point")
                price = o.get("price")
                if point is not None:
                    home_run_lines.append(float(point))
                if price is not None:
                    run_line_juices.append(int(price))

    @staticmethod
    def _collect_totals(
        outcomes:     list[dict],
        overs:        list[float],
        unders:       list[float],
        over_juices:  list[int],
        under_juices: list[int],
    ) -> None:
        for o in outcomes:
            point = o.get("point")
            price = o.get("price")
            if point is None:
                continue
            name = o.get("name", "").lower()
            if name == "over":
                overs.append(float(point))
                over_juices.append(int(price) if price is not None else -110)
            elif name == "under":
                unders.append(float(point))
                under_juices.append(int(price) if price is not None else -110)


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE ARTIFACTS  (shared, pre-trained state)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class PipelineArtifacts:
    """
    Immutable bundle produced by build_pipeline_artifacts().
    Passed into run_pipeline() so that the expensive work — pulling multiple
    seasons of game logs, engineering rolling features, fitting XGBoost +
    LightGBM — is done *once* per process, not once per game.
    """
    df:         "object"        # pd.DataFrame
    engine:     FeatureEngine
    trainer:    ModelTrainer
    trained_at: datetime        = field(default_factory=datetime.utcnow)
    cv_results: dict            = field(default_factory=dict)


def build_pipeline_artifacts(force_refresh: bool = False) -> PipelineArtifacts:
    """
    Execute the expensive, game-agnostic pipeline steps once:
      1. DataLoader   — fetch multiple seasons of MLB game logs
      2. FeatureEngine — build rolling / park-adjusted feature matrix
      3. ModelTrainer  — fit XGBoost (runs) + LightGBM (win) with TimeSeriesSplit CV

    Parameters
    ----------
    force_refresh : bool
        When True, bypass the local parquet cache and re-fetch all seasons
        from the MLB Stats API.  Defaults to False (use cache when available).
    """
    t0 = time.perf_counter()
    log.info("═" * 60)
    log.info("  build_pipeline_artifacts() [MLB] — START")
    log.info("═" * 60)

    log.info("[1/3] DataLoader: fetching MLB game logs …")
    loader = DataLoader(force_refresh=force_refresh)
    df_raw = loader.load()

    log.info("[2/3] FeatureEngine: building MLB feature matrix …")
    engine = FeatureEngine()
    df     = engine.build(df_raw)

    log.info("[3/3] ModelTrainer: fitting XGBoost + LightGBM …")
    X, y_runs, y_win = engine.get_feature_matrix(df)
    trainer = ModelTrainer()
    trainer.train(X, y_runs, y_win)

    elapsed = time.perf_counter() - t0
    log.info(f"  build_pipeline_artifacts() [MLB] — DONE in {elapsed:.1f}s")
    log.info(
        f"  XGB CV-MAE : {trainer.cv_results.get('xgb_mae', '?'):.3f} runs  |  "
        f"LGB CV-Acc : {trainer.cv_results.get('lgb_acc', '?'):.3f}  |  "
        f"LGB CV-LogLoss : {trainer.cv_results.get('lgb_ll', '?'):.4f}"
    )
    log.info("═" * 60)

    return PipelineArtifacts(
        df         = df,
        engine     = engine,
        trainer    = trainer,
        trained_at = datetime.utcnow(),
        cv_results = trainer.cv_results,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  FULL PIPELINE RUNNER  (MLB)
# ══════════════════════════════════════════════════════════════════════════════
def run_pipeline(
    home_team_abv:  str,
    away_team_abv:  str,
    market_spread:  Optional[float]             = None,  # home run line (−1.5 / +1.5)
    market_total:   Optional[float]             = None,
    market_ml_home: Optional[int]               = None,
    market_ml_away: Optional[int]               = None,  # away ML — required alongside market_ml_home for devigged edge
    bankroll:       float                       = 1000.0,
    live_odds:      bool                        = False,
    odds_api_key:   Optional[str]               = None,
    artifacts:      Optional[PipelineArtifacts] = None,
) -> dict:
    """
    Per-game MLB inference pipeline.

    When ``artifacts`` is supplied (recommended batch path), skips data loading
    and model training entirely — typically < 1 second per game.

    When ``artifacts`` is None (single-game / legacy path), builds the
    artifacts in-place from scratch.
    """

    # ── Step 0 ── Resolve live odds (single-game mode only) ───────────────────
    if live_odds:
        log.info(
            f"live_odds=True → calling OddsClient for "
            f"{away_team_abv} @ {home_team_abv} …"
        )
        client_kwargs: dict = {}
        if odds_api_key:
            client_kwargs["api_key"] = odds_api_key
        try:
            client = OddsClient(**client_kwargs)
            client.fetch()
            game_odds = client.find(home_team_abv, away_team_abv)

            if game_odds is None:
                log.warning(
                    f"OddsClient: no live line found for "
                    f"{away_team_abv} @ {home_team_abv}.  "
                    "Falling back to manually supplied market lines (if any)."
                )
            else:
                market_spread  = game_odds.home_run_line
                market_total   = game_odds.consensus_total
                market_ml_home = game_odds.home_ml
                market_ml_away = game_odds.away_ml
                log.info(
                    f"Live lines  →  Run Line: {market_spread}  "
                    f"Total: {market_total}  ML(home): {market_ml_home:+d}  "
                    f"ML(away): {market_ml_away:+d}  "
                    f"Books: {', '.join(game_odds.bookmakers_used[:4])} …"
                )
        except RuntimeError as exc:
            log.error(
                f"OddsClient failed: {exc}  → proceeding without live lines."
            )

    # ── Step 1-3 ── Build artifacts (only if not pre-supplied) ───────────────
    if artifacts is None:
        log.info(
            "run_pipeline: no PipelineArtifacts supplied — "
            "building from scratch (slow path).  "
            "For batch runs call build_pipeline_artifacts() once, then pass "
            "artifacts= to every run_pipeline() call."
        )
        artifacts = build_pipeline_artifacts()

    df      = artifacts.df
    engine  = artifacts.engine
    trainer = artifacts.trainer

    # ── Step 4 ── Extract latest team snapshots ───────────────────────────────
    home_df = df[df["TEAM_ABV"] == home_team_abv].sort_values("GAME_DATE").tail(1)
    away_df = df[df["TEAM_ABV"] == away_team_abv].sort_values("GAME_DATE").tail(1)

    if home_df.empty or away_df.empty:
        missing = [
            t for t, d in [(home_team_abv, home_df), (away_team_abv, away_df)]
            if d.empty
        ]
        raise ValueError(
            f"No recent feature data found for: {missing}.  "
            "Verify the abbreviation(s) exist in BALLPARK_COORDS / ODDS_TEAM_NAME_MAP."
        )

    home_X = home_df[engine.feature_cols]
    away_X = away_df[engine.feature_cols]

    # ── Step 5 ── Model inference ─────────────────────────────────────────────
    home_pred_runs = float(trainer.predict_runs(home_X)[0])
    away_pred_runs = float(trainer.predict_runs(away_X)[0])
    home_win_prob  = float(trainer.predict_win_prob(home_X)[0])

    # ── Step 6 ── Collect ePPAs & plate appearances ───────────────────────────
    home_eppas: dict = {
        w: float(home_df[f"ePPA_R{w}"].values[0])
        for w in FeatureEngine.TEAM_ROLL_WINDOWS
        if f"ePPA_R{w}" in home_df.columns
    }
    away_eppas: dict = {
        w: float(away_df[f"ePPA_R{w}"].values[0])
        for w in FeatureEngine.TEAM_ROLL_WINDOWS
        if f"ePPA_R{w}" in away_df.columns
    }
    home_pa = (
        float(home_df["PA_R10"].values[0])
        if "PA_R10" in home_df.columns else 36.0
    )
    away_pa = (
        float(away_df["PA_R10"].values[0])
        if "PA_R10" in away_df.columns else 36.0
    )

    # ── Step 7 ── Scan for value ──────────────────────────────────────────────
    scanner = ValueScanner()
    result  = scanner.scan(
        home_pred_runs  = home_pred_runs,
        away_pred_runs  = away_pred_runs,
        home_win_prob   = home_win_prob,
        home_eppas      = home_eppas,
        away_eppas      = away_eppas,
        home_pa         = home_pa,
        away_pa         = away_pa,
        market_run_line = market_spread,
        market_total    = market_total,
        market_ml_home  = market_ml_home,
        market_ml_away  = market_ml_away,
        bankroll        = bankroll,
    )

    # ── Step 8 ── Print individual projection card ────────────────────────────
    print_projection(home_team_abv, away_team_abv, result, bankroll)

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 5 — SUPABASE STORAGE  (MLB)
# ══════════════════════════════════════════════════════════════════════════════
def init_supabase() -> Optional[SupabaseClient]:
    """
    Initialise and return a Supabase client from environment credentials.
    Returns None (with a warning) when either credential is absent.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        log.warning(
            "Supabase credentials missing — SUPABASE_URL and/or SUPABASE_KEY "
            "not set in .env.  Cloud persistence will be skipped this run."
        )
        return None
    try:
        client: SupabaseClient = create_client(SUPABASE_URL, SUPABASE_KEY)
        log.info("Supabase client initialised successfully.")
        return client
    except Exception as exc:
        log.warning(
            f"Supabase init failed: {exc}  — cloud persistence disabled."
        )
        return None


def save_to_supabase(
    sb_client:  Optional[SupabaseClient],
    matchup:    str,
    result:     dict,
    game_odds:  Optional[GameOdds] = None,
) -> None:
    """
    Persist every flagged value play from a single MLB game to the
    ``mlb_projections`` table in Supabase.
    No-op when sb_client is None or plays list is empty.
    """
    if sb_client is None:
        return

    plays: list[dict] = result.get("plays", [])
    if not plays:
        log.debug(
            f"save_to_supabase: no plays for {matchup} — skipping insert."
        )
        return

    win_prob_home: float = result.get("win_prob_home", 0.0)

    def _resolve_market_lines(
        play_type: str,
        play_side: str,
    ) -> tuple[Optional[float], Optional[int]]:
        if game_odds is None:
            return None, None
        if play_type == "SPREAD":
            return game_odds.home_run_line, game_odds.run_line_juice
        if play_type == "TOTAL":
            juice = (
                game_odds.over_juice if play_side == "OVER" else game_odds.under_juice
            )
            if juice is None:
                juice = -110
            return game_odds.consensus_total, juice
        if play_type == "MONEYLINE":
            market_odds = (
                game_odds.home_ml if play_side == "HOME" else game_odds.away_ml
            )
            return None, market_odds
        log.warning(
            f"save_to_supabase: unrecognised play type '{play_type}' — stored as None."
        )
        return None, None

    rows: list[dict] = []
    for play in plays:
        market_line, market_odds = _resolve_market_lines(
            play["type"], play["side"]
        )
        rows.append({
            "matchup":       matchup,
            "type":          play["type"],
            "side":          play["side"],
            "edge":          float(play["edge"]),
            "ev":            float(play.get("ev", 0.0)),
            "grade":         play.get("grade", "C"),
            "kelly_bet":     float(play.get("kelly_$", 0.0)),
            "win_prob_home": float(win_prob_home),
            "market_line":   float(market_line)  if market_line  is not None else None,
            "market_odds":   int(market_odds)    if market_odds  is not None else None,
            "sport":         "MLB",
        })

    try:
        sb_client.table(_MLB_SUPABASE_TABLE).insert(rows).execute()
        log.info(
            f"  ✦ Supabase ← {len(rows)} row(s) inserted for {matchup}  "
            f"(grade breakdown: "
            + ", ".join(
                f"{g}×{sum(1 for r in rows if r['grade'] == g)}"
                for g in ("A", "B", "C")
                if any(r["grade"] == g for r in rows)
            )
            + ")"
        )
    except Exception as exc:
        log.warning(
            f"  ⚠  Supabase insert failed for {matchup}: {exc}  "
            "— play data retained in memory; cloud persistence skipped for this game."
        )