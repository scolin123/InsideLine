"""
MLB_scanner.py — MODULE 4: ValueScanner (MLB)
Consumes model outputs → calculates Fair Lines → compares to Market Lines
→ flags +EV plays → sizes bets via Kelly Criterion.

Also contains ANSI colour helpers and the print_projection pretty-printer.

MLB SANITY GUARDS
─────────────────
All seven guards from the NBA version are preserved; the numeric constants
are re-calibrated for baseball's much narrower run distribution.

[Guard 1] Probability-weighted grading:
          • Grade A Floor   : win_prob ≥ 52 %
          • Absolute Floor  : win_prob < 35 % → forced Grade C
          Re-calibrated edge thresholds for runs (not points):
            Grade A: |edge| ≥ 1.0 run  (was 12.5 pts in NBA)
            Grade B: |edge| ≥ 0.4 run  (was  4.0 pts in NBA)

[Guard 2] Edge normalisation: raw edge hard-capped at MAX_EFFECTIVE_EDGE
          (2.5 runs) before CDF probability calculations, then further
          compressed via S-curve (tanh).  In baseball a 2-run edge is
          genuinely enormous; capping at 2.5 prevents model outliers from
          inflating Kelly stakes.
          Kelly blend: 15 % model win_prob / 85 % normalised-edge-derived
          prob — identical blend to NBA for consistency.

[Guard 3] Separate sigma for totals: both LEAGUE_SIGMA and TOTALS_SIGMA
          are set to MLB_LEAGUE_SIGMA (4.2).  MLB run distributions are
          nearly symmetric so a single sigma works well for both markets.

[Guard 4] Dynamic grade-dependent Kelly ceiling:
          • Grade A: Max 5.0 % of bankroll per bet.
          • Grade B: Max 2.5 % of bankroll per bet.

[Guard 5] Run Line edge signage:
          MLB run line is fixed at ±1.5 runs (unlike NBA where the spread
          floats).  spread_edge = proj_spread − market_cover_threshold.
            proj_spread = home_pred_runs − away_pred_runs
            market_cover_threshold = −market_run_line  (convention: home
            favourite carries −1.5, underdog carries +1.5)
          Example: proj=+2.8, mkt=−1.5 → edge = 2.8 − 1.5 = +1.3 ✓

[Guard 6] Slate-level bankroll cap: 25 % of bankroll total exposure.

[Guard 7] Favourite Safety Valve: suppress Moneyline plays when market
          implies one team has > 85 % win probability.
"""
import logging
import math
from typing import Optional

from scipy.stats import norm

from .MLB_config import (
    LEAGUE_SIGMA,
    GARBAGE_SPREAD_THR,
    GARBAGE_ADJUST_PCT,
    EDGE_TOTAL_MIN,
    EDGE_SPREAD_MIN,
    KELLY_FRACTION,
    MARKET_BLEND_FACTOR_SPREAD,
    MARKET_BLEND_FACTOR_TOTAL,
    PROJ_SPREAD_MAX,
    PROJ_TOTAL_MIN,
    PROJ_TOTAL_MAX,
    WIN_PROB_SHRINK_TO_50,
)

log = logging.getLogger(__name__)

# ── Sanity-guard constants (MLB-calibrated) ───────────────────────────────────
MLB_LEAGUE_SIGMA: float       = 4.2    # historical scoring-margin σ in runs
TOTALS_SIGMA:     float       = 4.2    # [Guard 3] combined-runs σ (same for MLB)
MAX_EFFECTIVE_EDGE: float     = 2.5    # [Guard 2] hard cap (runs) before S-curve
MAX_KELLY_PCT: float          = 0.05   # [Guard 4] Grade A ceiling: 5 % of bankroll
GRADE_B_MAX_KELLY_PCT: float  = 0.025  # [Guard 4] Grade B ceiling: 2.5 % of bankroll
SLATE_KELLY_CAP: float        = 0.25   # [Guard 6] max aggregate risk per slate

# [Guard 1] Dual-gate grade thresholds
# --- Grade A (standard: both win_prob AND EV must clear) ---
# Raised significantly — targeting 2-3 A-tier MLB bets/night.
# MLB model MAE ~2.5 runs, so edges below that threshold are indistinguishable
# from noise. A-tier must clear the noise floor with meaningful margin.
GRADE_A_MIN_WIN_PROB: float        = 0.66   # ML / run-line win_prob floor  (was 0.58)
GRADE_A_MIN_WIN_PROB_TOTAL: float  = 0.65   # totals win_prob floor          (was 0.57)
GRADE_A_MIN_EV_ML: float           = 8.00   # $/100 EV floor — moneyline    (was 4.00)
GRADE_A_MIN_EV_SPREAD: float       = 7.00   # $/100 EV floor — run line     (was 3.50)
GRADE_A_MIN_EV_TOTAL: float        = 6.00   # $/100 EV floor — total        (was 3.00)
GRADE_A_ML_EDGE: float             = 0.15   # probability edge floor (15 %)  (was 0.07)
GRADE_A_SPREAD_EDGE: float         = 2.00   # run-line edge floor (runs)     (was 1.50)
GRADE_A_TOTAL_EDGE: float          = 2.50   # totals edge floor (runs)       (was 2.00)
# --- Grade B (both win_prob AND EV must clear, lower bars) ---
# Shifted up slightly to absorb former borderline-A bets.
GRADE_B_MIN_WIN_PROB: float        = 0.55   # (was 0.53)
GRADE_B_MIN_EV: float              = 2.50   # $/100  (was 1.50)
GRADE_B_ML_EDGE: float             = 0.07   # 7 % probability edge floor     (was 0.04)
GRADE_B_SPREAD_EDGE: float         = 1.00   # run-line edge floor (runs)     (was 0.75)
GRADE_B_TOTAL_EDGE: float          = 1.75   # totals edge floor (runs)       (was 1.50)
# --- Grade C (watch list — positive EV, above absolute floor, kelly = 0) ---
GRADE_C_MAX_WIN_PROB: float        = 0.35   # absolute floor; below this → forced C
# --- Edge credibility ceiling (diagnostic: edges above this are likely model noise) ---
MAX_CREDIBLE_EDGE: float           = 3.0    # [Guard 1] raw run/total edge > 3 → forced C
MAX_CREDIBLE_ML_EDGE: float        = 0.20   # [Guard 1] ML prob edge > 20 % → forced C
# --- UNDER-specific win_prob floor (unders are directionally fragile) ---
GRADE_A_UNDER_MIN_WIN_PROB: float  = 0.70   # A-tier UNDER requires 70 % win prob  (was 0.62)
# --- A-dog: underdog ML special track ---
GRADE_ADOG_MARKET_PROB_MAX: float  = 0.45   # team must be a market underdog (≤ 45 % implied)
GRADE_ADOG_MIN_WIN_PROB: float     = 0.40   # model must still see a real shot
GRADE_ADOG_MIN_ML_EDGE: float      = 0.12   # probability edge over market (12 %)   (was 0.08)
GRADE_ADOG_MIN_EV: float           = 7.00   # $/100 (underdogs need higher EV)       (was 5.00)
GRADE_ADOG_MAX_KELLY_PCT: float    = 0.03   # 3 % bankroll cap (vs 5 % for standard A)

# [Guard 7] Moneyline safety valve threshold
ML_FAVORITE_SAFETY_THRESHOLD: float = 0.85

# Win-probability blend: model weight / CDF weight
MARKET_BLEND_FACTOR: float = 0.60   # model win_prob weight in blend
# Kelly blend: model win_prob weight / normalised-edge-derived prob weight
EDGE_SHRINK_MODEL:   float = 0.15   # model win_prob weight in Kelly sizing

# MLB run line is always fixed at 1.5
MLB_RUN_LINE: float           = 1.5


# ══════════════════════════════════════════════════════════════════════════════
#  ANSI COLOUR HELPERS
# ══════════════════════════════════════════════════════════════════════════════
class _C:
    """ANSI escape codes for terminal colouring."""
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    RESET  = "\033[0m"


def _colorize(text: str, code: str) -> str:
    """Wrap *text* in an ANSI colour code + reset."""
    return f"{code}{text}{_C.RESET}"


def devig_two_way_probs(home_ml: int, away_ml: int) -> tuple:
    """
    Remove the bookmaker's vig from a two-sided moneyline market and return
    fair (normalised) win probabilities for home and away.

    Method: Additive — divide each raw implied prob by their sum so they
    sum to exactly 1.0.

    Returns
    -------
    (fair_home_prob, fair_away_prob)  — each in [0, 1], sum == 1.0
    """
    def _implied(ml: int) -> float:
        return abs(ml) / (abs(ml) + 100.0) if ml < 0 else 100.0 / (ml + 100.0)

    raw_home = _implied(home_ml)
    raw_away = _implied(away_ml)
    total    = raw_home + raw_away
    if total <= 0:
        return 0.5, 0.5
    return raw_home / total, raw_away / total


# ══════════════════════════════════════════════════════════════════════════════
#  VALUE SCANNER  (MLB)
# ══════════════════════════════════════════════════════════════════════════════
class ValueScanner:
    """
    Consumes MLB model outputs → calculates Fair Lines → compares to Market Lines
    → flags +EV plays → sizes bets via Kelly Criterion.

    Core maths (MLB)
    ─────────────────
    1. Projected Total  = ePPA_home × PA_home + ePPA_away × PA_away
       (park-factor already baked into ePPA from FeatureEngine)

    2. Projected Spread = predicted_runs_home − predicted_runs_away
       (positive = home leads; negative = away leads)

    3. P(home wins outright)  = Φ( proj_spread / σ )

    4. Run Line  — fixed at ±1.5:
       market_run_line convention: −1.5 = home favourite
       home_cover_threshold = −market_run_line = +1.5
       run_line_edge = proj_spread − home_cover_threshold
                     = proj_spread − (−market_run_line)
                     = proj_spread + market_run_line

       Example: proj=+2.8, mkt=−1.5 → edge = 2.8 + (−1.5) = +1.3 ✓

    5. Fair Moneyline → American odds  (same formula as NBA version)

    6. Kelly  f* = (b·p − q) / b,  fractional Kelly = f* × kelly_fraction
    """

    def __init__(
        self,
        sigma:               float = MLB_LEAGUE_SIGMA,
        garbage_thr:         float = GARBAGE_SPREAD_THR,
        garbage_pct:         float = GARBAGE_ADJUST_PCT,
        edge_total_min:      float = EDGE_TOTAL_MIN,
        edge_spread_min:     float = EDGE_SPREAD_MIN,
        kelly_fraction:      float = KELLY_FRACTION,
        market_blend_factor: float = MARKET_BLEND_FACTOR,
        edge_shrink_model:   float = EDGE_SHRINK_MODEL,
    ) -> None:
        self.sigma               = sigma
        self.garbage_thr         = garbage_thr
        self.garbage_pct         = garbage_pct
        self.edge_total_min      = edge_total_min
        self.edge_spread_min     = edge_spread_min
        self.kelly_fraction      = kelly_fraction
        self.market_blend_factor = market_blend_factor
        self.edge_shrink_model   = edge_shrink_model

    # ── Main entry point ──────────────────────────────────────────────────────
    def scan(
        self,
        home_pred_runs:   float,
        away_pred_runs:   float,
        home_win_prob:    float,           # LightGBM straight-up win prob (home)
        home_eppas:       dict,            # {window: ePPA}
        away_eppas:       dict,
        home_pa:          float,           # rolling plate appearances per game
        away_pa:          float,
        market_run_line:  Optional[float] = None,  # home run line (−1.5 or +1.5)
        market_total:     Optional[float] = None,
        market_ml_home:   Optional[int]   = None,
        market_ml_away:   Optional[int]   = None,
        bankroll:         float           = 1000.0,
        run_line_juice:   int             = -110,
        total_juice:      int             = -110,
    ) -> dict:
        """
        Compute fair lines and return a full projection dict.

        Parameters
        ──────────
        home/away_pred_runs : XGBoost run predictions for each team
        home_win_prob       : LightGBM win probability (home perspective)
        home/away_eppas     : {window: ePPA} from FeatureEngine
        home/away_pa        : rolling plate appearances per game
        market_run_line     : home run line (convention: −1.5 = home fav)
        market_total        : consensus over/under total
        market_ml_home      : American ML for the home team
        bankroll            : current bankroll in $ for Kelly sizing
        run_line_juice      : American-odds price on run-line bets (default −110)
        total_juice         : American-odds price on totals bets   (default −110)
        """

        # ── 1. Projected Run Line ─────────────────────────────────────────────
        raw_proj_spread = round(home_pred_runs - away_pred_runs, 2)

        # ── [Guard 10] Projection clamping ────────────────────────────────────
        # Hard-cap raw XGBoost outputs before any downstream calculation.
        # Anything outside ±PROJ_SPREAD_MAX is almost certainly model noise.
        if abs(raw_proj_spread) > PROJ_SPREAD_MAX:
            log.debug(
                "[Guard 10] proj_spread clamped: raw=%.2f → ±%.1f",
                raw_proj_spread, PROJ_SPREAD_MAX,
            )
        proj_spread = round(
            math.copysign(min(abs(raw_proj_spread), PROJ_SPREAD_MAX), raw_proj_spread), 2
        )

        # ── 2. Poisson total ─────────────────────────────────────────────────
        raw_proj_total = self._poisson_total(home_eppas, away_eppas, home_pa, away_pa)

        if not (PROJ_TOTAL_MIN <= raw_proj_total <= PROJ_TOTAL_MAX):
            log.debug(
                "[Guard 10] proj_total clamped: raw=%.2f → [%.1f, %.1f]",
                raw_proj_total, PROJ_TOTAL_MIN, PROJ_TOTAL_MAX,
            )
        proj_total = round(
            max(PROJ_TOTAL_MIN, min(raw_proj_total, PROJ_TOTAL_MAX)), 2
        )

        # ── 3. Blowout adjustment ─────────────────────────────────────────────
        garbage_adj = abs(proj_spread) > self.garbage_thr
        proj_total  = round(
            proj_total * (1 - self.garbage_pct) if garbage_adj else proj_total, 2
        )

        # ── 3.5. Market-line blending [Guard 14] ──────────────────────────────
        # Pull the effective spread/total used for edges and win-probability
        # calculations toward the market consensus.  The raw proj_spread and
        # proj_total are preserved for display (PROJ column) so the full model
        # disagreement remains visible.
        #
        # market_run_line convention: -1.5 = home fav
        #   → market_implied_spread = -market_run_line = +1.5
        if market_run_line is not None:
            market_implied_spread = -market_run_line
            eff_spread = round(
                proj_spread * (1.0 - MARKET_BLEND_FACTOR_SPREAD)
                + market_implied_spread * MARKET_BLEND_FACTOR_SPREAD,
                2,
            )
            log.debug(
                "[Guard 14] Spread blend: proj=%.2f  mkt_implied=%.2f  eff=%.2f",
                proj_spread, market_implied_spread, eff_spread,
            )
        else:
            eff_spread = proj_spread    # no market line → use raw projection

        if market_total is not None:
            eff_total = round(
                proj_total * (1.0 - MARKET_BLEND_FACTOR_TOTAL)
                + market_total * MARKET_BLEND_FACTOR_TOTAL,
                2,
            )
            log.debug(
                "[Guard 14] Total blend: proj=%.2f  mkt=%.2f  eff=%.2f",
                proj_total, market_total, eff_total,
            )
        else:
            eff_total = proj_total      # no market line → use raw projection

        # ── 4. Blended + calibrated home win probability [Guard 8] ───────────
        # CDF uses eff_spread (market-blended) so the win probability is also
        # pulled toward consensus.  After blending model and CDF, shrink the
        # result toward 50 % to prevent extreme model artefacts from producing
        # unrealistically large moneyline edges.
        cdf_win_prob   = self._cdf_win_prob(eff_spread)
        blend_win_prob = (
            self.market_blend_factor * home_win_prob
            + (1.0 - self.market_blend_factor) * cdf_win_prob
        )
        blend_win_prob = round(
            0.5 + (blend_win_prob - 0.5) * WIN_PROB_SHRINK_TO_50, 4
        )

        fair_ml_home = self._prob_to_american_ml(blend_win_prob)
        fair_ml_away = self._prob_to_american_ml(1.0 - blend_win_prob)

        result: dict = {
            "proj_spread":    proj_spread,
            "proj_total":     proj_total,
            "garbage_adj":    garbage_adj,
            "win_prob_home":  blend_win_prob,
            "win_prob_away":  round(1.0 - blend_win_prob, 4),
            "fair_ml_home":   fair_ml_home,
            "fair_ml_away":   fair_ml_away,
            "plays":          [],
        }

        # ── 5a. Run Line edge ─────────────────────────────────────────────────
        if market_run_line is not None:
            # [Guard 5] Run line is fixed at ±1.5.
            # market_run_line = −1.5 (home favourite) or +1.5 (home underdog).
            # Home covers when actual_margin > home_cover_threshold.
            # home_cover_threshold = −market_run_line
            # run_line_edge = proj_spread − (−market_run_line)
            #               = proj_spread + market_run_line
            run_line_edge = round(eff_spread + market_run_line, 2)
            result["spread_edge"] = run_line_edge

            if abs(run_line_edge) >= self.edge_spread_min:
                side = "HOME" if run_line_edge > 0 else "AWAY"

                # [Guard 2] Cap edge before CDF.
                capped_rl_edge = math.copysign(
                    min(abs(run_line_edge), MAX_EFFECTIVE_EDGE), run_line_edge
                )

                cover_prob_home = float(norm.cdf(capped_rl_edge / self.sigma))
                rl_win_prob     = cover_prob_home if side == "HOME" else (1.0 - cover_prob_home)

                ev    = self._raw_ev(rl_win_prob, run_line_juice)
                grade = self.grade_play("SPREAD", run_line_edge, ev, rl_win_prob)
                kelly = (
                    self._kelly(rl_win_prob, run_line_juice, bankroll,
                                raw_edge=run_line_edge, grade=grade)
                    if grade in ("A", "B") else 0.0
                )
                result["plays"].append({
                    "type":        "SPREAD",
                    "side":        side,
                    "edge":        run_line_edge,
                    "ev":          round(ev, 2),
                    "grade":       grade,
                    "kelly_$":     kelly,
                    "win_prob":    round(rl_win_prob, 4),
                    "odds":        run_line_juice,
                    "market_line": market_run_line,
                    "proj_line":   proj_spread,
                })

        # ── 5b. Totals edge ───────────────────────────────────────────────────
        if market_total is not None:
            total_edge = round(eff_total - market_total, 2)
            result["total_edge"] = total_edge

            if abs(total_edge) >= self.edge_total_min:
                side = "OVER" if total_edge > 0 else "UNDER"

                capped_total_edge = math.copysign(
                    min(abs(total_edge), MAX_EFFECTIVE_EDGE), total_edge
                )

                # [Guard 3] TOTALS_SIGMA = 4.2 (same as LEAGUE_SIGMA for MLB).
                over_prob      = float(norm.cdf(capped_total_edge / TOTALS_SIGMA))
                total_win_prob = over_prob if side == "OVER" else (1.0 - over_prob)

                ev    = self._raw_ev(total_win_prob, total_juice)
                grade = self.grade_play("TOTAL", total_edge, ev, total_win_prob, bet_side=side)
                kelly = (
                    self._kelly(total_win_prob, total_juice, bankroll,
                                raw_edge=total_edge, grade=grade)
                    if grade in ("A", "B") else 0.0
                )
                result["plays"].append({
                    "type":        "TOTAL",
                    "side":        side,
                    "edge":        total_edge,
                    "ev":          round(ev, 2),
                    "grade":       grade,
                    "kelly_$":     kelly,
                    "win_prob":    round(total_win_prob, 4),
                    "odds":        total_juice,
                    "market_line": market_total,
                    "proj_line":   proj_total,
                })

        # ── 5c. Moneyline edge ────────────────────────────────────────────────
        # Requires BOTH sides to properly remove the bookmaker's vig before
        # computing edge.  _flip_ml() is no longer used for edge detection —
        # it was comparing model prob against a vigged market prob, causing
        # side-selection errors and inflated away EV.
        if market_ml_home is not None and market_ml_away is not None:
            fair_mkt_home, fair_mkt_away = devig_two_way_probs(
                market_ml_home, market_ml_away
            )

            ml_edge_home = round(blend_win_prob - fair_mkt_home, 4)
            ml_edge_away = round((1.0 - blend_win_prob) - fair_mkt_away, 4)

            # Pick the side with the larger positive edge (mirrors NBA logic)
            if abs(ml_edge_home) >= abs(ml_edge_away):
                ml_edge = ml_edge_home
                side    = "HOME"
                ml_prob = blend_win_prob
                ml_odds = market_ml_home
                market_prob_for_side = fair_mkt_home
            else:
                ml_edge = ml_edge_away
                side    = "AWAY"
                ml_prob = round(1.0 - blend_win_prob, 4)
                ml_odds = market_ml_away
                market_prob_for_side = fair_mkt_away

            result["ml_edge"] = ml_edge

            if ml_edge > 0 and abs(ml_edge) >= 0.03:
                # [Guard 7] Suppress ML plays on near-certain favourites.
                fav_prob = max(fair_mkt_home, fair_mkt_away)
                if fav_prob > ML_FAVORITE_SAFETY_THRESHOLD:
                    log.debug(
                        "[Guard 7] Moneyline suppressed: fav_prob=%.3f > %.2f  "
                        "(%s @ %s)",
                        fav_prob, ML_FAVORITE_SAFETY_THRESHOLD, side, ml_odds,
                    )
                else:
                    ev    = self._raw_ev(ml_prob, ml_odds)
                    grade = self.grade_play(
                        "MONEYLINE", ml_edge, ev, ml_prob,
                        market_prob=market_prob_for_side,
                    )
                    kelly = (
                        self._kelly(ml_prob, ml_odds, bankroll, grade=grade)
                        if grade in ("A", "A-dog", "B") else 0.0
                    )
                    result["plays"].append({
                        "type":      "MONEYLINE",
                        "side":      side,
                        "edge":      ml_edge,
                        "ev":        round(ev, 2),
                        "grade":     grade,
                        "kelly_$":   kelly,
                        "win_prob":  round(ml_prob, 4),
                        "odds":      ml_odds,
                        "market_line": None,
                        "proj_line":   None,
                    })

        elif market_ml_home is not None:
            log.info(
                "ML betting skipped: market_ml_away not provided. "
                "Supply both sides for proper vig removal."
            )
            result["ml_skipped_reason"] = "market_ml_away missing; both sides required"

        # ── [Guard 6] Slate-level bankroll cap ────────────────────────────────
        plays_with_kelly = [p for p in result["plays"] if p["kelly_$"] > 0]
        if plays_with_kelly:
            total_kelly = sum(p["kelly_$"] for p in plays_with_kelly)
            max_slate   = SLATE_KELLY_CAP * bankroll
            if total_kelly > max_slate:
                scale_factor = max_slate / total_kelly
                log.warning(
                    "Slate Kelly cap triggered: total=%.2f > max=%.2f "
                    "(%.1f%% of $%.0f bankroll). Scaling all bets by %.4f.",
                    total_kelly, max_slate,
                    SLATE_KELLY_CAP * 100, bankroll, scale_factor,
                )
                for play in result["plays"]:
                    play["kelly_$"] = round(play["kelly_$"] * scale_factor, 2)

        return result

    # ── Maths helpers ─────────────────────────────────────────────────────────
    def _poisson_total(
        self,
        home_eppas: dict,
        away_eppas: dict,
        home_pa:    float,
        away_pa:    float,
    ) -> float:
        """
        Project game total (combined runs) using a Poisson-weighted model.

        Expected runs = blended ePPA × plate appearances.
        Rolling-window weights: 5-game = 40 %, 10-game = 35 %, 20-game = 25 %.
        """
        weights      = {5: 0.40, 10: 0.35, 20: 0.25}
        home_blended = sum(weights.get(w, 0) * v for w, v in home_eppas.items())
        away_blended = sum(weights.get(w, 0) * v for w, v in away_eppas.items())
        # Use each team's actual PA separately — not an average applied to both.
        return round((home_blended * home_pa + away_blended * away_pa) / 100.0, 2)

    def _cdf_win_prob(self, spread: float) -> float:
        """P(home wins outright) = Φ(proj_spread / σ). Positive spread → P > 0.5 ✓"""
        return float(norm.cdf(spread / self.sigma))

    @staticmethod
    def _prob_to_american_ml(p: float) -> int:
        """Convert a decimal win probability to American moneyline odds."""
        p = min(max(p, 0.001), 0.999)
        if p >= 0.5:
            return round(-100.0 * p / (1.0 - p))
        return round(100.0 * (1.0 - p) / p)

    @staticmethod
    def _american_ml_to_prob(ml: int) -> float:
        """Convert American moneyline to implied probability (raw, vig retained)."""
        if ml < 0:
            return abs(ml) / (abs(ml) + 100.0)
        return 100.0 / (ml + 100.0)

    @staticmethod
    def _flip_ml(ml: int) -> int:
        """Approximate the opposite side's American ML."""
        p   = abs(ml) / (abs(ml) + 100.0) if ml < 0 else 100.0 / (ml + 100.0)
        opp = 1.0 - p
        return ValueScanner._prob_to_american_ml(opp)   # type: ignore[attr-defined]

    # ── [Guard 2] Edge normalisation ─────────────────────────────────────────
    @staticmethod
    def _normalise_edge(raw_edge: float, cap: float = MAX_EFFECTIVE_EDGE) -> float:
        """
        Apply a tanh S-curve to compress extreme raw run edges.

        normalised = cap × tanh(raw_edge / cap)

        Key reference points (cap = 2.5 runs):
          raw = 0.5 runs →  normalised ≈ 0.47 runs  (minimal compression)
          raw = 1.0 run  →  normalised ≈ 0.85 runs
          raw = 1.5 runs →  normalised ≈ 1.14 runs
          raw = 2.0 runs →  normalised ≈ 1.38 runs
          raw = 3.0 runs →  normalised ≈ 1.71 runs  (heavy compression)
          raw = 5.0 runs →  normalised ≈ 2.16 runs
        """
        return cap * math.tanh(raw_edge / cap)

    # ── EV & Grading ─────────────────────────────────────────────────────────
    @staticmethod
    def _raw_ev(win_prob: float, market_ml: int, stake: float = 100.0) -> float:
        """
        Expected value of a flat $stake wager at market_ml odds.

        EV = (win_prob × profit_if_win) − (loss_prob × stake)
        """
        if market_ml > 0:
            profit_if_win = stake * market_ml / 100.0
        else:
            profit_if_win = stake * 100.0 / abs(market_ml)
        return (win_prob * profit_if_win) - ((1.0 - win_prob) * stake)

    @staticmethod
    def grade_play(
        bet_type:    str,
        edge:        float,
        ev:          float,
        win_prob:    float,
        market_prob: Optional[float] = None,
        bet_side:    Optional[str]   = None,
    ) -> str:
        """
        Assign a letter grade to a flagged play.

        [Guard 1] Dual-gate grading — BOTH win_prob AND EV must clear
        ────────────────────────────────────────────────────────────────
        Absolute Floor  : win_prob < 35 % → forced Grade C (no bet)
        EV < 0          : forced Grade C
        Credibility Gate: |raw_edge| > 3.0 runs → forced Grade C (model noise)

        Grade A  (High Confidence + High Value) — dual-gate AND
          MONEYLINE  : |edge| ≥ 7 %  AND  win_prob ≥ 58 %  AND  EV ≥ $4.00
          RUN LINE   : |edge| ≥ 1.5 runs AND win_prob ≥ 58 % AND EV ≥ $3.50
          TOTAL OVER : |edge| ≥ 2.0 runs AND win_prob ≥ 57 % AND EV ≥ $3.00
          TOTAL UNDER: |edge| ≥ 2.0 runs AND win_prob ≥ 62 % AND EV ≥ $3.00
            (UNDER requires higher win_prob — one big inning destroys the bet)

        Grade A-dog  (Underdog special track — ML only)
          Applies when market implies ≤ 45 % win probability for the bet side.
          Conditions: win_prob ≥ 40 %  AND  |ml_edge| ≥ 8 %  AND  EV ≥ $5.00
          Kelly cap is 3 % (vs 5 % for standard A) to offset higher variance.

        Grade B  (Solid Value — dual-gate AND)
          MONEYLINE  : |edge| ≥ 4 %  AND  win_prob ≥ 53 %  AND  EV ≥ $1.50
          RUN LINE   : |edge| ≥ 0.75 runs AND win_prob ≥ 53 % AND EV ≥ $1.50
          TOTAL      : |edge| ≥ 1.5 runs  AND win_prob ≥ 53 % AND EV ≥ $1.50

        Grade C  (Watch list — positive EV, no Kelly allocation)
          Positive EV, above absolute floor, but below B thresholds.
        """
        # ── Absolute floor ────────────────────────────────────────────────────
        if win_prob < GRADE_C_MAX_WIN_PROB:
            log.debug(
                "Grade forced to C: win_prob=%.3f < absolute floor %.2f",
                win_prob, GRADE_C_MAX_WIN_PROB,
            )
            return "C"

        if ev < 0:
            return "C"

        abs_edge = abs(edge)

        # ── [Guard 1] Edge credibility gates ──────────────────────────────────
        # Run-line / total: edges above 3 runs are almost certainly ePPA noise.
        if abs_edge > MAX_CREDIBLE_EDGE and bet_type != "MONEYLINE":
            log.debug(
                "Grade forced to C: |edge|=%.2f runs exceeds run/total ceiling %.1f",
                abs_edge, MAX_CREDIBLE_EDGE,
            )
            return "C"

        # Moneyline: probability edges above 20 % are implausible even after
        # shrinkage — flag as noise (market + model can't disagree by >20 % on
        # a well-calibrated system).
        if bet_type == "MONEYLINE" and abs_edge > MAX_CREDIBLE_ML_EDGE:
            log.debug(
                "ML grade forced to C: |edge|=%.3f exceeds ML ceiling %.2f",
                abs_edge, MAX_CREDIBLE_ML_EDGE,
            )
            return "C"

        if bet_type == "MONEYLINE":
            # A-dog track: underdog with large model edge over market
            if (
                market_prob is not None
                and market_prob <= GRADE_ADOG_MARKET_PROB_MAX
                and win_prob >= GRADE_ADOG_MIN_WIN_PROB
                and abs_edge >= GRADE_ADOG_MIN_ML_EDGE
                and ev >= GRADE_ADOG_MIN_EV
            ):
                return "A-dog"
            # Standard Grade A: dual-gate
            if (
                abs_edge >= GRADE_A_ML_EDGE
                and win_prob >= GRADE_A_MIN_WIN_PROB
                and ev >= GRADE_A_MIN_EV_ML
            ):
                return "A"
            # Grade B: dual-gate (lower bars)
            if (
                abs_edge >= GRADE_B_ML_EDGE
                and win_prob >= GRADE_B_MIN_WIN_PROB
                and ev >= GRADE_B_MIN_EV
            ):
                return "B"
            return "C"

        elif bet_type == "SPREAD":
            if (
                abs_edge >= GRADE_A_SPREAD_EDGE
                and win_prob >= GRADE_A_MIN_WIN_PROB
                and ev >= GRADE_A_MIN_EV_SPREAD
            ):
                return "A"
            if (
                abs_edge >= GRADE_B_SPREAD_EDGE
                and win_prob >= GRADE_B_MIN_WIN_PROB
                and ev >= GRADE_B_MIN_EV
            ):
                return "B"
            return "C"

        else:  # TOTAL
            # UNDER requires a higher win_prob floor than OVER: one big inning
            # destroys an under regardless of overall projection accuracy.
            a_total_win_prob = (
                GRADE_A_UNDER_MIN_WIN_PROB
                if bet_side == "UNDER"
                else GRADE_A_MIN_WIN_PROB_TOTAL
            )
            if (
                abs_edge >= GRADE_A_TOTAL_EDGE
                and win_prob >= a_total_win_prob
                and ev >= GRADE_A_MIN_EV_TOTAL
            ):
                return "A"
            if (
                abs_edge >= GRADE_B_TOTAL_EDGE
                and win_prob >= GRADE_B_MIN_WIN_PROB
                and ev >= GRADE_B_MIN_EV
            ):
                return "B"
            return "C"

    def _kelly(
        self,
        win_prob:   float,
        market_ml:  int,
        bankroll:   float,
        raw_edge:   Optional[float] = None,
        grade:      str             = "A",
    ) -> float:
        """
        Fractional Kelly bet size in dollars.

        [Guard 2] If raw_edge (in runs) is supplied, it is compressed via
        _normalise_edge() before deriving the effective win probability used in
        Kelly.

        [Guard 4] Grade-dependent ceiling:
          Grade A → MAX_KELLY_PCT (5 %) of bankroll
          Grade B → GRADE_B_MAX_KELLY_PCT (2.5 %) of bankroll
        """
        if market_ml > 0:
            decimal_odds = 1.0 + market_ml / 100.0
        else:
            decimal_odds = 1.0 + 100.0 / abs(market_ml)

        b = decimal_odds - 1.0
        if b <= 0:
            return 0.0

        if raw_edge is not None:
            norm_edge   = self._normalise_edge(abs(raw_edge))
            effective_p = float(norm.cdf(norm_edge / self.sigma))
            # model win_prob / normalised-edge-derived prob blend
            p = self.edge_shrink_model * win_prob + (1.0 - self.edge_shrink_model) * effective_p
        else:
            p = win_prob

        q = 1.0 - p

        full_kelly       = max((b * p - q) / b, 0.0)
        fractional_kelly = full_kelly * self.kelly_fraction
        raw_bet          = fractional_kelly * bankroll

        # [Guard 2 — Soft-Cap] Half-Kelly dampener
        raw_bet = raw_bet * 0.5

        # [Guard 4] Grade-dependent ceiling
        if grade == "B":
            max_bet = GRADE_B_MAX_KELLY_PCT * bankroll
        elif grade == "A-dog":
            max_bet = GRADE_ADOG_MAX_KELLY_PCT * bankroll
        else:
            max_bet = MAX_KELLY_PCT * bankroll
        capped_bet = min(raw_bet, max_bet)

        if raw_bet > max_bet:
            log.debug(
                "Kelly cap triggered (Grade %s): raw=%.2f capped=%.2f "
                "(%.1f%% of $%.0f bankroll)",
                grade, raw_bet, capped_bet, (max_bet / bankroll) * 100, bankroll,
            )

        return round(capped_bet, 2)


# ══════════════════════════════════════════════════════════════════════════════
#  PRETTY PRINTER
# ══════════════════════════════════════════════════════════════════════════════
def print_projection(
    home_team: str,
    away_team: str,
    result:    dict,
    bankroll:  float = 1000.0,
) -> None:
    """Render a formatted MLB projection card to stdout."""
    sep = "═" * 60
    print(f"\n{sep}")
    print(f"  MLB PROJECTION  ·  {away_team}  @  {home_team}")
    print(sep)
    spread_str = (
        f"{home_team} {result['proj_spread']:+.1f}"
        if result["proj_spread"] != 0
        else "Pick'em"
    )
    print(f"  Projected Run Line : {spread_str}")
    print(
        f"  Projected Total    : {result['proj_total']:.1f}"
        + ("  ← blowout adj." if result["garbage_adj"] else "")
    )
    print(f"  Win Prob (Home)    : {result['win_prob_home'] * 100:.1f}%")
    print(f"  Win Prob (Away)    : {result['win_prob_away'] * 100:.1f}%")
    print(f"  Fair ML  (Home)    : {result['fair_ml_home']:+d}")
    print(f"  Fair ML  (Away)    : {result['fair_ml_away']:+d}")

    if "spread_edge" in result:
        print(f"\n  Run Line Edge      : {result['spread_edge']:+.2f} runs")
    if "total_edge" in result:
        print(f"  Total  Edge        : {result['total_edge']:+.2f} runs")
    if "ml_edge" in result:
        print(f"  ML     Edge        : {result['ml_edge']:+.4f} (prob)")

    if result["plays"]:
        print(f"\n  ── VALUE PLAYS (bankroll ${bankroll:,.0f}) ──")

        total_kelly = sum(p["kelly_$"] for p in result["plays"])
        if total_kelly >= SLATE_KELLY_CAP * bankroll * 0.999:
            print(
                f"  ⚠  Slate cap active — total risk ≤ "
                f"{SLATE_KELLY_CAP * 100:.0f}% of bankroll "
                f"(${SLATE_KELLY_CAP * bankroll:,.0f})"
            )

        _GRADE_COLOR = {"A": _C.GREEN, "A-dog": _C.GREEN, "B": _C.YELLOW, "C": _C.RED}
        for play in result["plays"]:
            grade       = play.get("grade", "?")
            grade_color = _GRADE_COLOR.get(grade, _C.RESET)
            grade_str   = _colorize(f"[{grade}]", grade_color)
            kelly_note  = (
                f"  Kelly bet=${play['kelly_$']:,.2f}"
                if grade in ("A", "A-dog", "B")
                else "  Kelly bet=—  (watch list, no sizing)"
            )
            win_prob_pct = play.get("win_prob", 0.0) * 100

            if play["type"] == "MONEYLINE":
                edge_display = f"{play['edge'] * 100:+.2f}%"
            else:
                edge_display = f"{play['edge']:+.2f} runs"

            odds_val = play.get("odds")
            odds_str = f"  odds={odds_val:+d}" if odds_val is not None else ""

            print(
                f"  ✦ {grade_str} {play['type']:<10} {play['side']:<5}"
                f"  edge={edge_display}"
                f"  prob={win_prob_pct:.1f}%"
                f"  EV=${play.get('ev', 0):+.2f}"
                f"{odds_str}"
                f"{kelly_note}"
            )
    else:
        print("\n  No +EV plays flagged vs. market lines.")
    print(sep + "\n")