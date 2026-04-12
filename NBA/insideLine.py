"""
insideLine.py — MODULE 0: OddsClient  +  Pipeline orchestration  +  Supabase storage

  OddsClient       : HTTP client for The-Odds-API v4 (quota guard, best-line aggregation)
  PipelineArtifacts: Shared, pre-trained state bundle (DataFrame + engine + trainer)
  build_pipeline_artifacts(): Expensive one-time data load + model training
  run_pipeline()   : Per-game inference — uses PipelineArtifacts to skip re-training
  init_supabase()  : Initialise Supabase client from environment credentials
  save_to_supabase(): Persist flagged plays to the `projections` table
"""
import os
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import requests
from supabase import create_client, Client as SupabaseClient

from .config import (
    ODDS_API_BASE_URL, ODDS_API_KEY, ODDS_QUOTA_MIN,
    ODDS_REGIONS, ODDS_MARKETS, ODDS_FORMAT,
    ODDS_TEAM_NAME_MAP,
    SUPABASE_URL, SUPABASE_KEY,
)
from .data_loader import DataLoader
from .features import FeatureEngine
from .models import ModelTrainer
from .scanner import ValueScanner, print_projection

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 0 — ODDS CLIENT
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class GameOdds:
    """
    Structured container for a single game's best available lines.

    "Best" is defined per-market from the bettor's perspective:
      • Moneyline  — highest (least negative / most positive) American ML for
                     each side, i.e. the most generous payout offered.
      • Spread     — home-team spread closest to 0 (smallest number of points
                     given up), which is the sharpest / tightest line.
      • Total      — highest Over line and lowest Under line are both stored;
                     we surface the consensus mid-point as proj_total.
                     over_juice / under_juice carry the American-odds price
                     paired with those best lines for CLV tracking.
    """
    home_abv:        str
    away_abv:        str
    home_ml:         Optional[int]   = None
    away_ml:         Optional[int]   = None
    home_spread:     Optional[float] = None
    spread_juice:    Optional[int]   = None
    total_over:      Optional[float] = None
    total_under:     Optional[float] = None
    over_juice:      Optional[int]   = None
    under_juice:     Optional[int]   = None
    bookmakers_used: list[str]       = field(default_factory=list)

    @property
    def consensus_total(self) -> Optional[float]:
        """Mid-point of best Over and best Under lines."""
        if self.total_over is not None and self.total_under is not None:
            return round((self.total_over + self.total_under) / 2, 1)
        return self.total_over or self.total_under


class OddsClient:
    """
    Thin HTTP client for The-Odds-API v4.

    Responsibilities
    ────────────────
    1. Fire a single GET /v4/sports/basketball_nba/odds request.
    2. Guard the monthly quota — abort if ≤ ODDS_QUOTA_MIN requests remain.
    3. Parse the JSON response into a list of GameOdds dataclasses.
    4. Expose a .find() method to locate a specific home/away matchup.
    """

    _ENDPOINT = ODDS_API_BASE_URL

    def __init__(
        self,
        api_key:         str = ODDS_API_KEY,
        quota_min:       int = ODDS_QUOTA_MIN,
        request_timeout: int = 10,
    ):
        self.api_key          = os.environ.get("ODDS_API_KEY", api_key)
        self.quota_min        = quota_min
        self.timeout          = request_timeout
        self._quota_remaining: Optional[int]    = None
        self._quota_used:      Optional[int]    = None
        self._games:           list[GameOdds]   = []

    # ── Public ────────────────────────────────────────────────────────────────
    def fetch(self) -> list[GameOdds]:
        """Fetch live NBA odds and return a list of GameOdds."""
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

        log.info("OddsClient → fetching live NBA lines from The-Odds-API …")
        try:
            resp = requests.get(self._ENDPOINT, params=params, timeout=self.timeout)
        except requests.exceptions.Timeout:
            raise RuntimeError(f"OddsClient: request timed out after {self.timeout}s.")
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

    def _parse_quota_headers(self, headers: requests.structures.CaseInsensitiveDict) -> None:
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
        raise RuntimeError(f"OddsClient: {detail}  Body: {resp.text[:300]}")

    def _parse_game(self, raw: dict) -> GameOdds:
        """Convert one raw game dict from The-Odds-API JSON into a GameOdds."""
        home_full = raw.get("home_team", "")
        away_full = raw.get("away_team", "")
        home_abv  = ODDS_TEAM_NAME_MAP.get(home_full, "")
        away_abv  = ODDS_TEAM_NAME_MAP.get(away_full, "")

        if not home_abv or not away_abv:
            log.debug(
                f"OddsClient: unmapped team name(s): '{home_full}' / '{away_full}'. "
                "Update ODDS_TEAM_NAME_MAP if this is a new franchise."
            )

        game = GameOdds(home_abv=home_abv, away_abv=away_abv)

        home_mls:      list[int]   = []
        away_mls:      list[int]   = []
        home_spreads:  list[float] = []
        spread_juices: list[int]   = []
        overs:         list[float] = []
        unders:        list[float] = []
        over_juices:   list[int]   = []
        under_juices:  list[int]   = []

        for book in raw.get("bookmakers", []):
            bk_key = book.get("key", "unknown")
            game.bookmakers_used.append(bk_key)

            for market in book.get("markets", []):
                mkey     = market.get("key")
                outcomes = market.get("outcomes", [])

                if mkey == "h2h":
                    self._collect_h2h(outcomes, home_full, home_mls, away_mls)
                elif mkey == "spreads":
                    self._collect_spreads(outcomes, home_full, home_spreads, spread_juices)
                elif mkey == "totals":
                    self._collect_totals(outcomes, overs, unders, over_juices, under_juices)

        # ── Best-line aggregation ─────────────────────────────────────────────
        game.home_ml = max(home_mls) if home_mls else None
        game.away_ml = max(away_mls) if away_mls else None

        if home_spreads:
            idx               = int(np.argmin(np.abs(home_spreads)))
            game.home_spread  = home_spreads[idx]
            game.spread_juice = spread_juices[idx] if spread_juices else None

        if overs:
            idx             = int(np.argmax(overs))
            game.total_over = overs[idx]
            game.over_juice = over_juices[idx] if over_juices else None

        if unders:
            idx              = int(np.argmin(unders))
            game.total_under = unders[idx]
            game.under_juice = under_juices[idx] if under_juices else None

        return game

    # ── Market collectors ─────────────────────────────────────────────────────
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
        outcomes:      list[dict],
        home_full:     str,
        home_spreads:  list[float],
        spread_juices: list[int],
    ) -> None:
        for o in outcomes:
            if o.get("name") == home_full:
                point = o.get("point")
                price = o.get("price")
                if point is not None:
                    home_spreads.append(float(point))
                if price is not None:
                    spread_juices.append(int(price))

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
    Passed into run_pipeline so that the expensive work — pulling 4 seasons
    of game logs, engineering rolling features, fitting XGBoost + LightGBM —
    is done *once* per process, not once per game.
    """
    df:         "object"    # pd.DataFrame — typed as object to avoid circular imports
    engine:     FeatureEngine
    trainer:    ModelTrainer
    trained_at: datetime    = field(default_factory=datetime.utcnow)
    cv_results: dict        = field(default_factory=dict)


def build_pipeline_artifacts(force_refresh: bool = False) -> PipelineArtifacts:
    """
    Execute the expensive, game-agnostic pipeline steps once:
      1. DataLoader  — fetch 4 seasons of game logs from nba_api
      2. FeatureEngine — build rolling / opponent-adjusted feature matrix
      3. ModelTrainer  — fit XGBoost (pts) + LightGBM (win) with TimeSeriesSplit CV

    Parameters
    ----------
    force_refresh : bool
        When True, bypass the local parquet cache and re-fetch all seasons
        from nba_api.  Defaults to False (use cache when available).
    """
    t0 = time.perf_counter()
    log.info("═" * 60)
    log.info("  build_pipeline_artifacts() — START")
    log.info("═" * 60)

    log.info("[1/3] DataLoader: fetching game logs …")
    loader = DataLoader(force_refresh=force_refresh)
    df_raw = loader.load()

    log.info("[2/3] FeatureEngine: building feature matrix …")
    engine = FeatureEngine()
    df     = engine.build(df_raw)

    log.info("[3/3] ModelTrainer: fitting XGBoost + LightGBM …")
    X, y_pts, y_win = engine.get_feature_matrix(df)
    trainer = ModelTrainer()
    trainer.train(X, y_pts, y_win)

    elapsed = time.perf_counter() - t0
    log.info(f"  build_pipeline_artifacts() — DONE in {elapsed:.1f}s")
    log.info(
        f"  XGB CV-MAE: {trainer.cv_results.get('xgb_mae', '?'):.3f} pts  |  "
        f"LGB CV-Acc: {trainer.cv_results.get('lgb_acc', '?'):.3f}  |  "
        f"LGB CV-LogLoss: {trainer.cv_results.get('lgb_ll', '?'):.4f}"
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
#  FULL PIPELINE RUNNER
# ══════════════════════════════════════════════════════════════════════════════
def run_pipeline(
    home_team_abv:  str,
    away_team_abv:  str,
    market_spread:  Optional[float]             = None,
    market_total:   Optional[float]             = None,
    market_ml_home: Optional[int]               = None,
    market_ml_away: Optional[int]               = None,
    bankroll:       float                       = 1000.0,
    live_odds:      bool                        = False,
    odds_api_key:   Optional[str]               = None,
    artifacts:      Optional[PipelineArtifacts] = None,
    retrain:        bool                        = True,
) -> dict:
    """
    Per-game inference pipeline.

    When ``artifacts`` is supplied (recommended batch path), skips data loading
    and model training entirely — typically < 1 second per game.

    When ``artifacts`` is None (single-game / legacy path), builds the
    artifacts in-place from scratch (~3–8 min).
    """

    # ── Step 0 ── Resolve live odds (single-game mode only) ───────────────────
    if live_odds:
        log.info(f"live_odds=True → calling OddsClient for {away_team_abv} @ {home_team_abv} …")
        client_kwargs: dict = {}
        if odds_api_key:
            client_kwargs["api_key"] = odds_api_key
        try:
            client = OddsClient(**client_kwargs)
            client.fetch()
            game_odds = client.find(home_team_abv, away_team_abv)

            if game_odds is None:
                log.warning(
                    f"OddsClient: no live line found for {away_team_abv} @ {home_team_abv}. "
                    "Falling back to manually supplied market lines (if any)."
                )
            else:
                market_spread  = game_odds.home_spread
                market_total   = game_odds.consensus_total
                market_ml_home = game_odds.home_ml
                market_ml_away = game_odds.away_ml

                home_ml_str = f"{market_ml_home:+d}" if market_ml_home is not None else "None"
                away_ml_str = f"{market_ml_away:+d}" if market_ml_away is not None else "None"

                log.info(
                    f"Live lines  →  Spread: {market_spread:+}  "
                    f"Total: {market_total}  ML(home): {home_ml_str}  ML(away): {away_ml_str}  "
                    f"Books: {', '.join(game_odds.bookmakers_used[:4])} …"
                )
        except RuntimeError as exc:
            log.error(f"OddsClient failed: {exc}  → proceeding without live lines.")

    # ── Step 1-3 ── Build artifacts (only if not pre-supplied) ───────────────
    if artifacts is None:
        log.info(
            "run_pipeline: no PipelineArtifacts supplied — "
            "building from scratch (slow path). "
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
        missing = [t for t, d in [(home_team_abv, home_df), (away_team_abv, away_df)] if d.empty]
        raise ValueError(
            f"No recent feature data found for: {missing}. "
            "Verify the abbreviation(s) exist in ARENA_COORDS / ODDS_TEAM_NAME_MAP."
        )

    home_X = home_df[engine.feature_cols]
    away_X = away_df[engine.feature_cols]

    # ── Step 5 ── Model inference ─────────────────────────────────────────────
    home_pred_pts = float(trainer.predict_pts(home_X)[0])
    away_pred_pts = float(trainer.predict_pts(away_X)[0])
    home_win_prob = float(trainer.predict_win_prob(home_X)[0])

    # ── Step 6 ── Collect ePPPs & pace ────────────────────────────────────────
    home_eppps = {
        w: float(home_df[f"ePPP_R{w}"].values[0])
        for w in FeatureEngine.ROLL_WINDOWS
        if f"ePPP_R{w}" in home_df.columns
    }
    away_eppps = {
        w: float(away_df[f"ePPP_R{w}"].values[0])
        for w in FeatureEngine.ROLL_WINDOWS
        if f"ePPP_R{w}" in away_df.columns
    }
    home_pace = float(home_df["POSS_R10"].values[0]) if "POSS_R10" in home_df.columns else 100.0
    away_pace = float(away_df["POSS_R10"].values[0]) if "POSS_R10" in away_df.columns else 100.0

    # ── Step 7 ── Scan for value ──────────────────────────────────────────────
    scanner = ValueScanner()
    result  = scanner.scan(
        home_pred_pts  = home_pred_pts,
        away_pred_pts  = away_pred_pts,
        home_win_prob  = home_win_prob,
        home_eppps     = home_eppps,
        away_eppps     = away_eppps,
        home_pace      = home_pace,
        away_pace      = away_pace,
        market_spread  = market_spread,
        market_total   = market_total,
        market_ml_home = market_ml_home,
        market_ml_away = market_ml_away,
        bankroll       = bankroll,
    )

    # ── Step 8 ── Print individual projection card ────────────────────────────
    print_projection(home_team_abv, away_team_abv, result, bankroll)

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 5 — SUPABASE STORAGE
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
        log.warning(f"Supabase init failed: {exc}  — cloud persistence disabled.")
        return None


def save_to_supabase(
    sb_client:  Optional[SupabaseClient],
    matchup:    str,
    result:     dict,
    game_odds:  Optional[GameOdds] = None,
) -> None:
    """
    Persist every flagged value play from a single game to the ``projections``
    table in Supabase.  No-op when sb_client is None or plays list is empty.
    """
    if sb_client is None:
        return

    plays: list[dict] = result.get("plays", [])
    if not plays:
        log.debug(f"save_to_supabase: no plays for {matchup} — skipping insert.")
        return

    win_prob_home: float = result.get("win_prob_home", 0.0)

    def _resolve_market_lines(
        play_type: str,
        play_side: str,
    ) -> tuple[Optional[float], Optional[int]]:
        if game_odds is None:
            return None, None
        if play_type == "SPREAD":
            return game_odds.home_spread, game_odds.spread_juice
        if play_type == "TOTAL":
            juice = (
                game_odds.over_juice  if play_side == "OVER"  else game_odds.under_juice
            )
            if juice is None:
                juice = -110
            return game_odds.consensus_total, juice
        if play_type == "MONEYLINE":
            market_odds = (
                game_odds.home_ml if play_side == "HOME" else game_odds.away_ml
            )
            return None, market_odds
        log.warning(f"save_to_supabase: unrecognised play type '{play_type}' — stored as None.")
        return None, None

    rows: list[dict] = []
    for play in plays:
        market_line, market_odds = _resolve_market_lines(play["type"], play["side"])
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
        })

    try:
        sb_client.table("projections").insert(rows).execute()
        log.info(
            f"  ✦ Supabase ← {len(rows)} row(s) inserted for {matchup}  "
            f"(grade breakdown: "
            + ", ".join(
                f"{g}×{sum(1 for r in rows if r['grade'] == g)}"
                for g in ('A', 'B', 'C')
                if any(r['grade'] == g for r in rows)
            )
            + ")"
        )
    except Exception as exc:
        log.warning(
            f"  ⚠  Supabase insert failed for {matchup}: {exc}  "
            "— play data retained in memory; cloud persistence skipped for this game."
        )
