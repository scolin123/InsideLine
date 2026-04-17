"""
scanner.py — MODULE 4: ValueScanner
Consumes model outputs → calculates Fair Lines → compares to Market Lines
→ flags +EV plays → sizes bets via Kelly Criterion.

Also contains ANSI colour helpers and the print_projection pretty-printer.

SANITY GUARDS (v9 — Market-Blended, Correlation-Aware, Confidence-Gated)
──────────────────────────────────────────────────────────────────────────
[Guard 1] Probability-weighted grading (TIGHTENED in v8, held in v9):
          grade_play() now accepts win_prob as a required argument.
          • Grade A Floor   : win_prob ≥ 58.5 % for spread/total, 60 % for ML.
          • Grade B+ Floor  : near-A plays that clear B+ thresholds AND are
                              within configurable closeness of A thresholds
                              (BOTH edge AND prob must be close — AND, not OR).
          • Grade C Floor   : win_prob < 35 % → rejected entirely (None).
          • Sub-C Floor     : plays below C-tier minimums → rejected, not displayed.

[Guard 11] Projection fragility / confidence penalty (v8):
          compute_projection_confidence() scores each play 0→1 by measuring
          how stable its ePPP inputs are across rolling windows (5/10/20 g).
          apply_fragility_discount() then:
            • Shrinks calibrated_edge   by (confidence)
            • Shrinks win_prob further toward 0.50 by (1 − confidence) × 0.20
              (scale raised from 0.10 in v8 to 0.20 in v9)
          This runs BEFORE grade_play(), so fragile plays cannot reach A/B+.

[Guard 12] Dynamic sigma for spread probability (v8):
          Spread cover-probability is computed with an effective sigma that
          widens when confidence is low:
            σ_eff = LEAGUE_SIGMA × (1 + (1−confidence) × SIGMA_FRAGILITY_BOOST)

[Guard 13] Extra totals conservatism (v8):
          Totals use a further-widened sigma when pace and ePPP inputs are
          noisy.  Overs additionally receive a small haircut via
          TOTALS_OVER_FRAGILITY_PENALTY.

[Guard 14] Market-line blending (NEW in v9):
          Before computing edge or win probability, the effective spread/total
          used in all maths is blended TOWARD the market line:
            eff_spread = proj_spread × (1−MARKET_BLEND_FACTOR)
                       + market_implied_spread × MARKET_BLEND_FACTOR
          MARKET_BLEND_FACTOR = 0.25 (25% pull toward market).
          This directly reduces leverage from overconfident projections.
          The display still shows the raw model projection.

[Guard 15] Confidence gating for bettability (NEW in v9):
          Plays below MIN_CONFIDENCE_FOR_BET (0.55) are display-only regardless
          of grade.  The second bet per game requires MIN_CONFIDENCE_FOR_SECOND_BET
          (0.70).  ML plays require ML_MIN_CONFIDENCE (0.60).

[Guard 16] Confidence-scaled Kelly sizing (NEW in v9):
          After the grade cap, bet size is multiplied by:
            max(CONFIDENCE_KELLY_SCALE_FLOOR, confidence)
          At confidence=0.55: bet is 55% of grade cap.
          At confidence=1.00: no reduction.

[Guard 17] Same-game correlation control (NEW in v9):
          A helper _bets_are_correlated() detects when a second game play
          shares directional game-script dependency with the first bet
          (e.g., SPREAD HOME + ML HOME; SPREAD HOME + OVER).
          Correlated second bets require CORRELATED_SECOND_BET_MIN_GRADE (B+).

[Guard 2] Two-stage edge shrinkage before display and Kelly:
          raw_edge is multiplied by EDGE_SHRINK_FACTOR (0.65) to produce
          calibrated_edge.  calibrated_edge is then hard-capped at
          SPREAD_DISPLAY_EDGE_CAP / TOTAL_DISPLAY_EDGE_CAP.

[Guard 3] Separate sigma for totals: TOTALS_SIGMA = 16.0 vs.
          LEAGUE_SIGMA (12.5) for spreads.

[Guard 4] Dynamic grade-dependent Kelly ceiling:
          • Grade A:  Max 4.0 % of bankroll per bet.
          • Grade B+: Max 3.0 % of bankroll per bet.
          • Grade B:  Max 2.0 % of bankroll per bet.
          • Grade C:  No Kelly sizing (display-only).

[Guard 5] Edge signage verified: spread_edge = proj_spread + market_spread.

[Guard 6] Slate-level bankroll cap (Anti-Ruin guard):
          SLATE_KELLY_CAP = 20 % (applied only to bettable plays).

[Guard 7] Moneyline requires BOTH sides (home + away market lines).
          Vig is removed via devig_two_way_probs() before computing ML edge.
          Tighter ML thresholds: min edge 3 % (C-tier gate), Grade A 8 %,
          Grade B 5.5 %, Grade C 3 %.

[Guard 8] Win-probability calibration:
          blend = 0.50 × model_prob + 0.50 × cdf_prob (equal weight).
          CDF input uses eff_spread (market-blended) × CDF_SPREAD_SHRINK.
          Then pulled toward 50 % via WIN_PROB_SHRINK_TO_50 = 0.85.

[Guard 9] Kelly uses calibrated win probability directly.
          USE_FLAT_BET_SIZING flag available for backtests.

[Guard 10] Projection clamping: proj_spread clamped to ±30 pts,
           proj_total to [150, 280] pts.

TWO-LAYER PIPELINE (v6+)
─────────────────────────
Layer 1: Generate all valid display candidates (A/B/C that meet minimum
         C-tier quality).  Every play gets is_bettable=False initially.

Layer 2: Select the bettable subset.
         • Only BET_GRADES (A, B+, B by default) are considered.
         • [Guard 15] Confidence gate applied before selection.
         • At most MAX_BETS_PER_GAME bettable plays per game.
         • [Guard 17] Correlated second bets require B+.
         • Slate-level cap: apply_slate_kelly_cap() for multi-game slates.
         • Selected plays are marked is_bettable=True and receive
           confidence-scaled Kelly sizing [Guard 16].
"""
import logging
import math
from typing import List, Optional, Tuple

from scipy.stats import norm

from .config import (
    LEAGUE_SIGMA,
    GARBAGE_SPREAD_THR,
    GARBAGE_ADJUST_PCT,
    EDGE_TOTAL_MIN,
    EDGE_SPREAD_MIN,
    KELLY_FRACTION,
)

log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
#  DISPLAY & BETTING MODE CONFIGURATION
#  These flags control which plays are shown and which are flagged for wagering.
#  Edit here to adjust scanner behaviour without touching core thresholds.
# ══════════════════════════════════════════════════════════════════════════════

# -- Display toggles -----------------------------------------------------------
SHOW_C_TIER: bool               = True   # include C-tier plays in output list
BET_ON_C_TIER: bool             = False  # C-tier is NEVER bettable by default

# -- Volume limits (per-game) --------------------------------------------------
MAX_BETS_PER_GAME: int          = 2      # max bettable plays from a single game
MAX_DISPLAY_PLAYS_PER_GAME: int = 3      # max plays surfaced for frontend display

# -- Slate-level betting cap ---------------------------------------------------
MAX_BETTABLE_PLAYS_PER_SLATE: int = 5   # used by apply_slate_limits() helper

# -- Grade sets ----------------------------------------------------------------
# BET_GRADES: which grades are eligible to be marked is_bettable=True
BET_GRADES: set                 = {"A", "B+", "B"}
# DISPLAY_GRADES: which grades appear in result["plays"] at all
DISPLAY_GRADES: set             = {"A", "B+", "B", "C"}

# ── Overconfidence-hardening constants ───────────────────────────────────────

# [Guard 2] Edge shrinkage & display caps
EDGE_SHRINK_FACTOR: float       = 0.65   # raw_edge × this = calibrated_edge
SPREAD_DISPLAY_EDGE_CAP: float  = 10.0   # hard ceiling for calibrated spread edge (pts)
TOTAL_DISPLAY_EDGE_CAP: float   = 12.0   # hard ceiling for calibrated total edge (pts)

# [Guard 3] Wider sigma for totals
TOTALS_SIGMA: float             = 16.0

# [Guard 4] Kelly size ceilings (tightened vs v4)
MAX_KELLY_PCT: float            = 0.04   # Grade A:  4 % of bankroll
GRADE_BPLUS_MAX_KELLY_PCT: float= 0.03   # Grade B+: 3 % of bankroll
GRADE_B_MAX_KELLY_PCT: float    = 0.02   # Grade B:  2 % of bankroll

# [Guard 6] Slate bankroll cap (applied to bettable plays only)
SLATE_KELLY_CAP: float          = 0.20

# ── Grade thresholds (A / B+ / B / C) ────────────────────────────────────────
# [Guard 1] Structured threshold config — easy to tune without touching logic.
#
# Each tier requires BOTH calibrated_edge (abs) AND win_prob to qualify.
# Tiers are evaluated top-down: A → B+ → B → C.
# B+ only fires when A is missed AND the is_close_to_a() check passes.

# [Guard 1 / v8] Tightened grade thresholds — A and B grades are now harder to reach.
# Rationale: the old floors allowed projection-driven plays that looked strong on paper
# but regularly missed due to noisy ePPP / pace inputs.  Each tier's edge AND win_prob
# requirements have been raised so only genuinely robust spots qualify.
#
# Spread A:  7.0→8.0 pts, 56%→58.5%  |  Total A:  8.0→9.5 pts, 56%→58.5%
# Spread B+: 6.2→7.0 pts, 55%→57%    |  Total B+: 7.0→8.5 pts, 55%→57%
# Spread B:  5.0→5.5 pts, 54%→55.5%  |  Total B:  6.0→7.0 pts, 54%→55.5%
# (C-tier gates are unchanged — they are display-only filters, not betting gates.)
GRADE_THRESHOLDS: dict = {
    "spread": {
        # A raised well above model noise floor (~10 pt MAE) — targets 0-2 bets/night
        "A":      {"edge": 11.0, "win_prob": 0.630},   # was 8.0 / 0.585
        # B+ absorbs what used to be borderline A (8-11 pt range)
        "B_PLUS": {"edge": 8.5,  "win_prob": 0.600},   # was 7.0 / 0.570
        "B":      {"edge": 5.5,  "win_prob": 0.555},   # unchanged
        "C":      {"edge": 3.0,  "win_prob": 0.515},   # unchanged
    },
    "total": {
        # Totals are more projection-driven → A threshold set higher than spread
        "A":      {"edge": 13.0, "win_prob": 0.630},   # was 9.5 / 0.585
        # B+ absorbs what used to be borderline A (9.5-13 pt range)
        "B_PLUS": {"edge": 10.5, "win_prob": 0.600},   # was 8.5 / 0.570
        "B":      {"edge": 7.0,  "win_prob": 0.555},   # unchanged
        "C":      {"edge": 3.5,  "win_prob": 0.515},   # unchanged
    },
    "moneyline": {
        # ML A requires a very large probability edge — model at 54% can't
        # reliably produce edges below 15%; anything smaller is likely noise
        "A":      {"edge": 0.15, "win_prob": 0.650},   # was 0.08 / 0.60
        # B+ absorbs the 10-15% ML edge range
        "B_PLUS": {"edge": 0.10, "win_prob": 0.620},   # was 0.07 / 0.580
        "B":      {"edge": 0.055,"win_prob": 0.565},   # unchanged
        "C":      {"edge": 0.03, "win_prob": 0.52},    # unchanged
    },
}

# ── B+ "closeness to A" config ───────────────────────────────────────────────
# A play is B+ only if it passes B+ thresholds AND is within one of these
# distances of the A threshold (on either edge OR win_prob).
# This prevents ordinary B plays from being relabelled as B+.

BPLUS_EDGE_CLOSE_SPREAD: float  = 2.5    # within 2.5 pts of A edge (was 1.0 — gap is now 11-8.5=2.5)
BPLUS_EDGE_CLOSE_TOTAL: float   = 2.5    # within 2.5 pts of A edge (was 1.0 — gap is now 13-10.5=2.5)
BPLUS_EDGE_CLOSE_ML: float      = 0.05   # within 5% of A edge    (was 0.01 — gap is now 15-10=5%)
BPLUS_PROB_CLOSE: float         = 0.03   # within 0.03 of A win_prob (was 0.01)
BPLUS_ML_PROB_CLOSE: float      = 0.03   # within 0.03 of A win_prob (was 0.015)

# ── Post-diagnostic guards ────────────────────────────────────────────────────
# [Guard 1-ext] Large market spread cap: if the market already prices a team at
# ±13+ pts, covering that number is far less reliable than covering a normal
# spread.  Plays with |market_spread| > this threshold are capped at grade B.
LARGE_MARKET_SPREAD_THR: float  = 13.0   # absolute market spread (pts)

# [Guard 1-ext] UNDER-specific win_prob floor for A-tier totals.
# NBA unders are fragile — pace spikes and garbage-time offense blow them up.
# Require 63 % vs the standard 58.5 % OVER floor to reach A.
GRADE_A_UNDER_WIN_PROB: float   = 0.68   # was 0.63 — UNDER A requires 68% (very selective)

# ── Convenience aliases (keep callers that reference old flat constants) ──────
# A-tier
GRADE_A_SPREAD_EDGE: float      = GRADE_THRESHOLDS["spread"]["A"]["edge"]
GRADE_A_TOTAL_EDGE: float       = GRADE_THRESHOLDS["total"]["A"]["edge"]
GRADE_A_SPREAD_WIN_PROB: float  = GRADE_THRESHOLDS["spread"]["A"]["win_prob"]
GRADE_A_ML_EDGE: float          = GRADE_THRESHOLDS["moneyline"]["A"]["edge"]
GRADE_A_ML_WIN_PROB: float      = GRADE_THRESHOLDS["moneyline"]["A"]["win_prob"]

# B-tier
GRADE_B_SPREAD_EDGE: float      = GRADE_THRESHOLDS["spread"]["B"]["edge"]
GRADE_B_TOTAL_EDGE: float       = GRADE_THRESHOLDS["total"]["B"]["edge"]
GRADE_B_SPREAD_WIN_PROB: float  = GRADE_THRESHOLDS["spread"]["B"]["win_prob"]
GRADE_B_ML_EDGE: float          = GRADE_THRESHOLDS["moneyline"]["B"]["edge"]
GRADE_B_ML_WIN_PROB: float      = GRADE_THRESHOLDS["moneyline"]["B"]["win_prob"]

# C-tier (minimum quality gate to surface at all)
GRADE_C_SPREAD_EDGE: float      = GRADE_THRESHOLDS["spread"]["C"]["edge"]
GRADE_C_TOTAL_EDGE: float       = GRADE_THRESHOLDS["total"]["C"]["edge"]
GRADE_C_SPREAD_WIN_PROB: float  = GRADE_THRESHOLDS["spread"]["C"]["win_prob"]
GRADE_C_TOTAL_WIN_PROB: float   = GRADE_THRESHOLDS["total"]["C"]["win_prob"]
GRADE_C_ML_EDGE: float          = GRADE_THRESHOLDS["moneyline"]["C"]["edge"]
GRADE_C_ML_WIN_PROB: float      = GRADE_THRESHOLDS["moneyline"]["C"]["win_prob"]

# Absolute floor — below this win_prob, reject entirely (not even C)
GRADE_C_MAX_WIN_PROB: float     = 0.35

# [Guard 7] Moneyline thresholds (conservative)
ML_MIN_EDGE: float              = GRADE_C_ML_EDGE   # C-tier gate; grade_play() enforces tiers
ML_GRADE_A_EDGE: float          = GRADE_A_ML_EDGE
ML_GRADE_B_EDGE: float          = GRADE_B_ML_EDGE
ML_GRADE_A_WIN_PROB: float      = GRADE_A_ML_WIN_PROB
ML_GRADE_B_WIN_PROB: float      = GRADE_B_ML_WIN_PROB
ML_UNDERDOG_EXTRA_EDGE: float   = 0.02   # extra buffer required for plus-money bets
ML_UNDERDOG_MIN_PROB: float     = 0.50   # plus-money side must still clear 50 %

# Favorite safety valve (suppress extreme heavy-favorite ML bets)
ML_FAVORITE_SAFETY_THRESHOLD: float = 0.85

# [Guard 8] Win-probability calibration
WIN_PROB_BLEND_MODEL: float     = 0.50   # weight on LightGBM model win prob
WIN_PROB_BLEND_CDF: float       = 0.50   # weight on CDF spread win prob
WIN_PROB_SHRINK_TO_50: float    = 0.85   # shrink blended prob toward 50 %
CDF_SPREAD_SHRINK: float        = 0.85   # shrink spread fed into CDF (conservative)

# [Guard 9] Flat-bet override (set True for backtests)
USE_FLAT_BET_SIZING: bool       = False
FLAT_BET_AMOUNT: float          = 100.0  # $ flat bet when USE_FLAT_BET_SIZING=True

# Projection clamp limits [Guard 10]
PROJ_SPREAD_MAX: float          = 30.0   # abs clamp on projected spread
PROJ_TOTAL_MIN: float           = 150.0
PROJ_TOTAL_MAX: float           = 280.0

# ── [Guard 11] Fragility / confidence system constants (NEW in v8) ────────────
#
# compute_projection_confidence() returns a score in [0, 1]:
#   1.0 = inputs are fully consistent across all windows → no penalty
#   0.0 = extreme disagreement across windows → maximum penalty
#
# The confidence score is used in apply_fragility_discount() to:
#   • Shrink calibrated_edge multiplicatively (edge × confidence)
#   • Pull win_prob further toward 0.50:
#       adj_prob = 0.5 + (prob − 0.5) × (FRAGILITY_PROB_SHRINK_BASE
#                                         + (1−confidence) × FRAGILITY_PROB_SHRINK_SCALE)
#
# Tune FRAGILITY_*_SCALE to control how aggressively shaky projections are penalised.

# Maximum relative spread (std / mean) in ePPP across windows before confidence = 0.
FRAGILITY_EPPPP_CV_MAX: float   = 0.15   # coefficient of variation cap for ePPP inputs

# Shrink base: even a perfectly confident play is shrunk by this (combines with Guard 8)
FRAGILITY_PROB_SHRINK_BASE: float  = 1.0    # no extra shrink at confidence=1.0
# Scale: at confidence=0 the prob is pulled this much extra toward 0.50.
# v9: raised from 0.10 → 0.20 so that low-confidence projections are materially
# less aggressive (at conf=0, win_prob moves an extra 10% toward 50% vs. 5% before).
FRAGILITY_PROB_SHRINK_SCALE: float = 0.20   # e.g. at conf=0, prob moves 20% extra toward 0.5

# [Guard 12] Dynamic sigma: how much to widen σ when confidence is low.
# σ_eff = LEAGUE_SIGMA × (1 + (1 − confidence) × SIGMA_FRAGILITY_BOOST)
# e.g. at confidence=0.5 and boost=0.4: σ_eff = 12.5 × 1.2 = 15.0
SIGMA_FRAGILITY_BOOST: float    = 0.40   # spread sigma boost factor at zero confidence

# [Guard 13] Totals-specific conservatism.
# Totals sigma is additionally widened by fragility on top of Guard 3.
TOTALS_SIGMA_FRAGILITY_BOOST: float = 0.50  # wider than spread boost — totals are noisier
# Overs haircut: calibrated_edge for overs is further reduced by this factor when
# confidence < TOTALS_OVER_FRAGILITY_THRESHOLD (catches fragile pace-driven overs).
TOTALS_OVER_FRAGILITY_PENALTY: float     = 0.08   # 8% extra edge shrink on shaky overs
TOTALS_OVER_FRAGILITY_THRESHOLD: float   = 0.75   # apply over penalty below this confidence

# ── [Guard 14] Market-line blending (NEW in v9) ───────────────────────────────
#
# Before computing edge or win probability, we blend the raw projected
# spread/total TOWARD the market line by MARKET_BLEND_FACTOR.
#
# Effect: at factor=0.25, the effective projection is 75% model / 25% market.
# This has two benefits:
#   1. The raw edge shrinks by (1 - MARKET_BLEND_FACTOR) — direct conservatism.
#   2. The CDF win probability also becomes more conservative, because it is
#      computed from the blended spread, not the raw model spread.
#   3. ML edge (derived from calib_win_prob) inherits the same conservatism.
#
# The displayed proj_spread/proj_total remain the raw model projections so
# that users can still see how much the model disagrees with the market.
#
# Tune: 0.20 = mild humility  |  0.30 = strong humility
# When market lines are unavailable, the effective projection is unchanged.
MARKET_BLEND_FACTOR: float          = 0.25   # 25% pull toward market line

# ── [Guard 15] Confidence gating for bettable selection (NEW in v9) ──────────
#
# Plays below MIN_CONFIDENCE_FOR_BET are display-only regardless of grade.
# The second bet per game additionally requires MIN_CONFIDENCE_FOR_SECOND_BET
# so that correlated plays from uncertain projections are suppressed.
# ML plays require ML_MIN_CONFIDENCE because their edge is entirely model-derived.
#
# These thresholds complement the existing fragility discount (Guard 11):
# fragility already reduces edge/win_prob, but a noisy play can still squeak
# above the grade floor after shrinkage.  These gates add a hard floor.
MIN_CONFIDENCE_FOR_BET: float           = 0.55   # absolute minimum to be bettable
MIN_CONFIDENCE_FOR_SECOND_BET: float    = 0.70   # second game bet requires higher confidence
ML_MIN_CONFIDENCE: float                = 0.60   # ML plays need cleaner inputs

# ── [Guard 16] Confidence-weighted Kelly scaling (NEW in v9) ─────────────────
#
# After computing the grade-capped Kelly amount, the bet size is multiplied
# by max(CONFIDENCE_KELLY_SCALE_FLOOR, confidence).
# At confidence=0.55 (near the betting floor), sizing is at most 55% of cap.
# At confidence=1.0, no reduction.
#
# This creates a smooth sizing ramp that is completely independent from the
# hard grade caps — both apply simultaneously.
CONFIDENCE_KELLY_SCALE_FLOOR: float     = 0.40   # minimum fraction of cap even at conf=0

# ── [Guard 17] Same-game correlated second-bet guard (NEW in v9) ─────────────
#
# When selecting a second bet from the same game, if it shares directional
# exposure with the first bet (e.g., HOME spread + HOME ML, or same pace
# direction on spread + total), require at least CORRELATED_SECOND_BET_MIN_GRADE.
# This reduces the risk of doubling up on a projection that is wrong in both
# markets simultaneously.
#
# Correlation is judged by _bets_are_correlated():
#   SPREAD HOME + ML HOME          → correlated (same game-script dependency)
#   SPREAD HOME + OVER             → potentially correlated (high-scoring favors)
#   SPREAD + UNDER                 → lower correlation (favored team slows pace)
#   ML + TOTAL (same game)         → correlated if ML and total direction agree
# Correlated second bets must be B+ or better to be selected.
CORRELATED_SECOND_BET_MIN_GRADE: str    = "B+"


# ══════════════════════════════════════════════════════════════════════════════
#  ANSI COLOR HELPERS
# ══════════════════════════════════════════════════════════════════════════════
class _C:
    """ANSI escape codes for terminal colouring."""
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"
    RESET  = "\033[0m"


def _colorize(text: str, code: str) -> str:
    return f"{code}{text}{_C.RESET}"


# ══════════════════════════════════════════════════════════════════════════════
#  MONEYLINE PROBABILITY HELPERS  (module-level, importable)
# ══════════════════════════════════════════════════════════════════════════════

def american_to_implied_prob(ml: int) -> float:
    """
    Raw implied probability from American odds (vig still included).

    Negative ML (favourite):  p =  |ml| / (|ml| + 100)
    Positive ML (underdog):   p = 100  / (ml + 100)
    """
    if ml < 0:
        return abs(ml) / (abs(ml) + 100.0)
    return 100.0 / (ml + 100.0)


def devig_two_way_probs(home_ml: int, away_ml: int) -> Tuple[float, float]:
    """
    Remove the bookmaker's vig from a two-sided market and return fair
    (normalised) win probabilities for home and away.

    Method: Additive (Shin-adjacent) — divide each raw implied prob by
    the sum of both raw implied probs so they sum to exactly 1.0.

    Returns
    -------
    (fair_home_prob, fair_away_prob)  — each in [0, 1], sum == 1.0
    """
    raw_home = american_to_implied_prob(home_ml)
    raw_away = american_to_implied_prob(away_ml)
    total    = raw_home + raw_away
    if total <= 0:
        return 0.5, 0.5
    return raw_home / total, raw_away / total


def fair_prob_to_american(prob: float) -> int:
    """Convert a vig-free decimal probability to American moneyline odds."""
    prob = min(max(prob, 0.001), 0.999)
    if prob >= 0.5:
        return round(-100.0 * prob / (1.0 - prob))
    return round(100.0 * (1.0 - prob) / prob)


def moneyline_ev(model_prob: float, market_ml: int, stake: float = 100.0) -> float:
    """
    Expected value of a flat $stake ML wager.

    EV = (model_prob × profit_if_win) – ((1 – model_prob) × stake)
    """
    if market_ml > 0:
        profit_if_win = stake * market_ml / 100.0
    else:
        profit_if_win = stake * 100.0 / abs(market_ml)
    return (model_prob * profit_if_win) - ((1.0 - model_prob) * stake)


def _play_quality_score(grade: str, calibrated_edge: float, win_prob: float,
                         bet_type: str, confidence: float = 1.0) -> float:
    """
    Composite score used for sorting and display ranking.

    FOR DISPLAY USE ONLY — not a confidence metric or bet-sizing input.

    Formula:
      grade_base       : A=100, B+=75, B=50, C=0
      edge_bonus       : calibrated_edge (abs) × 3  (spreads/totals) or
                         calibrated_edge (abs) × 400 (ML, normalises prob-space edge)
      prob_bonus       : win_prob × 10
      confidence_bonus : confidence × 15   [NEW v9]
                         Ensures a high-confidence B+ sorts above a fragile B+,
                         and a fragile A doesn't automatically top the list.
    """
    grade_base = {"A": 100, "B+": 75, "B": 50, "C": 0}.get(grade, 0)
    edge_multiplier = 400.0 if bet_type == "MONEYLINE" else 3.0
    edge_bonus = abs(calibrated_edge) * edge_multiplier
    prob_bonus = win_prob * 10.0
    # [Guard 15] Confidence bonus: high-confidence plays sort above fragile ones
    # of the same grade.  15 pts is meaningful (~half a grade step for B→B+).
    confidence_bonus = confidence * 15.0
    return round(grade_base + edge_bonus + prob_bonus + confidence_bonus, 4)


# ══════════════════════════════════════════════════════════════════════════════
#  SLATE-LEVEL HELPER  (multi-game use)
# ══════════════════════════════════════════════════════════════════════════════

def apply_slate_limits(
    all_plays: List[dict],
    bankroll: float,
    max_bettable: int = MAX_BETTABLE_PLAYS_PER_SLATE,
    slate_kelly_cap: float = SLATE_KELLY_CAP,
) -> List[dict]:
    """
    Apply slate-level betting constraints across plays from multiple games.

    Call this AFTER collecting scan() results for all games in a slate.

    Steps
    ─────
    1. Collect all is_bettable=True plays from across the slate.
    2. Sort by play_quality_score descending.
    3. Keep only the top `max_bettable` as bettable; demote the rest
       (is_bettable=False, kelly_$=0).
    4. Scale remaining bettable Kelly amounts if total exceeds the slate cap.

    Parameters
    ──────────
    all_plays       : flat list of play dicts from multiple scan() calls.
                      Each dict must have is_bettable, kelly_$, play_quality_score.
    bankroll        : current bankroll in $.
    max_bettable    : hard cap on total bettable plays for the slate.
    slate_kelly_cap : max fraction of bankroll that can be committed slate-wide.

    Returns
    ───────
    The same list with is_bettable and kelly_$ fields adjusted in-place.
    """
    bettable = sorted(
        [p for p in all_plays if p.get("is_bettable")],
        key=lambda p: p.get("play_quality_score", 0),
        reverse=True,
    )

    # Demote plays beyond the slate cap
    for play in bettable[max_bettable:]:
        play["is_bettable"] = False
        play["kelly_$"] = 0.0
        log.debug(
            "Slate cap demoted play: %s %s (quality=%.2f)",
            play.get("type"), play.get("side"), play.get("play_quality_score", 0),
        )

    # Scale remaining bettable plays to the slate Kelly cap
    final_bettable = [p for p in all_plays if p.get("is_bettable")]
    total_kelly = sum(p.get("kelly_$", 0) for p in final_bettable)
    max_slate   = slate_kelly_cap * bankroll

    if total_kelly > max_slate and total_kelly > 0:
        scale_factor = max_slate / total_kelly
        log.warning(
            "Slate Kelly cap triggered: total=%.2f > max=%.2f "
            "(%.1f%% of $%.0f bankroll). Scaling %d bets by %.4f.",
            total_kelly, max_slate, slate_kelly_cap * 100, bankroll,
            len(final_bettable), scale_factor,
        )
        for play in final_bettable:
            play["kelly_$"] = round(play["kelly_$"] * scale_factor, 2)

    return all_plays


# ══════════════════════════════════════════════════════════════════════════════
#  B+ CLOSENESS HELPER
# ══════════════════════════════════════════════════════════════════════════════

def is_close_to_a(bet_type: str, abs_edge: float, win_prob: float) -> bool:
    """
    Return True when a play is within configurable distance of A-tier thresholds.

    Used together with the B+ threshold check: a play is labelled B+ only when
    BOTH conditions hold:
      1. It passes the B+ threshold (edge ≥ B_PLUS edge AND win_prob ≥ B_PLUS prob).
      2. is_close_to_a() returns True — it is genuinely near A, not just a
         marginally-better B.

    Closeness is satisfied when EITHER:
      • abs_edge  is within <BPLUS_EDGE_CLOSE_*>  of the A edge threshold, OR
      • win_prob  is within <BPLUS_PROB_CLOSE[_ML]> of the A win_prob threshold.

    This prevents ordinary B plays from being relabelled as B+ simply because
    they clear the B+ floor by a comfortable margin.
    """
    bet_type_key = bet_type.lower()
    if bet_type_key == "moneyline":
        a_edge  = GRADE_THRESHOLDS["moneyline"]["A"]["edge"]
        a_prob  = GRADE_THRESHOLDS["moneyline"]["A"]["win_prob"]
        edge_close = abs_edge >= (a_edge - BPLUS_EDGE_CLOSE_ML)
        prob_close = win_prob >= (a_prob - BPLUS_ML_PROB_CLOSE)
    elif bet_type_key == "spread":
        a_edge  = GRADE_THRESHOLDS["spread"]["A"]["edge"]
        a_prob  = GRADE_THRESHOLDS["spread"]["A"]["win_prob"]
        edge_close = abs_edge >= (a_edge - BPLUS_EDGE_CLOSE_SPREAD)
        prob_close = win_prob >= (a_prob - BPLUS_PROB_CLOSE)
    else:  # TOTAL
        a_edge  = GRADE_THRESHOLDS["total"]["A"]["edge"]
        a_prob  = GRADE_THRESHOLDS["total"]["A"]["win_prob"]
        edge_close = abs_edge >= (a_edge - BPLUS_EDGE_CLOSE_TOTAL)
        prob_close = win_prob >= (a_prob - BPLUS_PROB_CLOSE)

    # [v9] Require BOTH edge and prob to be close to A (was OR).
    # OR logic allowed a play to be B+ if it was close on only ONE dimension,
    # producing too many "near-A" labels for plays that were merely B with a
    # strong edge OR high prob — but not both.  AND is stricter and more honest.
    return edge_close and prob_close


# ══════════════════════════════════════════════════════════════════════════════
#  [Guard 11 / 12 / 13]  FRAGILITY / CONFIDENCE HELPERS  (NEW in v8)
#
#  These functions form a reusable "fragility filter" that runs BEFORE
#  grade_play().  The pipeline is:
#
#    raw inputs → calibrate_edge → compute_projection_confidence
#              → apply_fragility_discount → grade_play
#
#  The confidence score is stored in each play dict as "confidence" so it
#  can be inspected / logged without altering the rest of the output schema.
# ══════════════════════════════════════════════════════════════════════════════

def compute_projection_confidence(
    home_eppps: dict,
    away_eppps: dict,
    home_pace:  float,
    away_pace:  float,
    *,
    is_total:   bool = False,
) -> float:
    """
    [Guard 11] Score the stability / trustworthiness of the projection inputs.

    Returns a confidence multiplier in [0.0, 1.0]:
      1.0 → all windows agree tightly; projection is robust.
      0.0 → maximum disagreement across windows; projection is fragile.

    Method
    ──────
    1. Compute the coefficient of variation (CV = std/mean) of each team's
       ePPP values across the three rolling windows (5/10/20 game).
       High CV means the recent-form signal is noisy or trending hard.

    2. For totals, also measure the relative spread between home_pace and
       away_pace (pace mismatch).  Large mismatches inflate projected totals
       in ways that may not be sustainable.

    3. Score each dimension linearly against FRAGILITY_EPPPP_CV_MAX (above the
       cap → that team scores 0 for this dimension).

    4. Return the geometric mean of the individual dimension scores so that one
       very unstable input brings the overall confidence down significantly.

    Parameters
    ──────────
    home/away_eppps : {window: ePPP} dicts from FeatureEngine (keys 5, 10, 20)
    home/away_pace  : rolling possessions-per-game for each team
    is_total        : if True, also penalise for pace mismatch
    """
    def _cv_score(eppps: dict) -> float:
        """Return 1.0 − clipped_CV, where CV = std/mean of ePPP values."""
        vals = list(eppps.values())
        if len(vals) < 2:
            return 1.0   # can't measure variance; don't penalise
        mean = sum(vals) / len(vals)
        if mean <= 0:
            return 0.5   # degenerate case — moderate penalty
        variance = sum((v - mean) ** 2 for v in vals) / len(vals)
        cv = math.sqrt(variance) / mean
        # Linearly map [0, FRAGILITY_EPPPP_CV_MAX] → [1.0, 0.0]
        return max(0.0, 1.0 - cv / FRAGILITY_EPPPP_CV_MAX)

    home_score = _cv_score(home_eppps)
    away_score = _cv_score(away_eppps)

    if is_total:
        # Pace mismatch: large differences between home and away pace inflate
        # totals projections in ways that are historically unreliable.
        avg_pace = (home_pace + away_pace) / 2.0 if (home_pace + away_pace) > 0 else 1.0
        pace_diff_pct = abs(home_pace - away_pace) / avg_pace
        # Allow up to 10 % pace mismatch before penalising; cap at 30 %
        pace_score = max(0.0, 1.0 - max(0.0, pace_diff_pct - 0.10) / 0.20)
        # Geometric mean of three dimensions (home ePPP, away ePPP, pace)
        product = home_score * away_score * pace_score
    else:
        # Geometric mean of two dimensions (home ePPP, away ePPP)
        product = home_score * away_score

    confidence = round(product ** (1.0 / (3 if is_total else 2)), 4)
    log.debug(
        "[Guard 11] Projection confidence: %.4f  (home_cv=%.4f away_cv=%.4f%s)",
        confidence, home_score, away_score,
        f" pace={pace_score:.4f}" if is_total else "",
    )
    return confidence


def apply_fragility_discount(
    calibrated_edge: float,
    win_prob:        float,
    confidence:      float,
    *,
    is_over:         bool  = False,
) -> Tuple[float, float]:
    """
    [Guard 11] Apply confidence-weighted penalties to edge and win probability.

    A confidence of 1.0 returns the inputs unchanged (beyond existing shrinkage).
    A confidence below 1.0 both compresses the edge toward 0 and pulls win_prob
    further toward 0.50, making it harder for uncertain projections to reach A/B+.

    Formula
    ───────
    adj_edge = calibrated_edge × confidence
               (+ extra over haircut if is_over and confidence < threshold)

    adj_prob = 0.5 + (win_prob − 0.5)
               × (FRAGILITY_PROB_SHRINK_BASE − (1−confidence) × FRAGILITY_PROB_SHRINK_SCALE)

    Parameters
    ──────────
    calibrated_edge : edge already shrunk and capped (calibrate_edge output)
    win_prob        : calibrated win probability (Guard 8 output)
    confidence      : from compute_projection_confidence() — in [0, 1]
    is_over         : if True and confidence is low, apply extra over haircut

    Returns
    ───────
    (adj_edge, adj_prob)  — both ready for grade_play()
    """
    # Edge compression: weaker inputs → edge shrinks toward 0.
    adj_edge = calibrated_edge * confidence

    # [Guard 13] Extra over haircut: overs are systematically overfit to high-
    # pace projections.  A low-confidence over gets an additional discount so
    # it must clear the grade threshold by a wider margin.
    if is_over and confidence < TOTALS_OVER_FRAGILITY_THRESHOLD:
        adj_edge = adj_edge * (1.0 - TOTALS_OVER_FRAGILITY_PENALTY)

    # Probability shrink: pull further toward 0.50 proportional to uncertainty.
    shrink = FRAGILITY_PROB_SHRINK_BASE - (1.0 - confidence) * FRAGILITY_PROB_SHRINK_SCALE
    shrink = max(0.0, min(1.0, shrink))   # clamp to [0, 1]
    adj_prob = 0.5 + (win_prob - 0.5) * shrink

    log.debug(
        "[Guard 11] Fragility discount: conf=%.4f  "
        "edge %.4f→%.4f  prob %.4f→%.4f  is_over=%s",
        confidence, calibrated_edge, adj_edge, win_prob, adj_prob, is_over,
    )
    return round(adj_edge, 4), round(adj_prob, 4)
class ValueScanner:
    """
    Consumes model outputs → calculates Fair Lines → compares to Market Lines
    → flags +EV plays → sizes bets via Kelly Criterion.

    Two-Layer Pipeline
    ──────────────────
    Layer 1: scan() generates ALL valid display candidates (A/B+/B/C plays
             meeting minimum C-tier quality).  No betting flags yet.

    Layer 2: scan() then calls _select_bettable_plays() to mark a limited
             subset as is_bettable=True and assign Kelly sizing.  This
             enforces per-game correlated exposure limits and grade filters.
             B+ plays are bettable by default (included in BET_GRADES).

    Core maths
    ──────────
    1. Projected Total  = ePPP_home × PACE + ePPP_away × PACE

    2. Projected Spread = predicted_pts_home - predicted_pts_away

    3. P(home wins outright)  = Φ( proj_spread × CDF_SPREAD_SHRINK / σ )
       Shrinking the spread input pulls win-probs toward 50 % [Guard 8].

    4. Blended win prob = 0.50 × model_prob + 0.50 × cdf_prob
       Then calibrated  = 0.50 + (blend – 0.50) × WIN_PROB_SHRINK_TO_50

    5. raw_edge  = proj_spread + market_spread  (spread convention)
                 = proj_total  – market_total   (totals convention)
       calibrated_edge = raw_edge × EDGE_SHRINK_FACTOR
       calibrated_edge capped at SPREAD_DISPLAY_EDGE_CAP / TOTAL_DISPLAY_EDGE_CAP

    6. ML edge = model_prob – devigged_market_prob  (vig removed first)

    7. Kelly  f* = (b·p – q) / b   where p = calibrated win_prob (direct)
       Applied only to bettable (is_bettable=True) plays.
    """

    def __init__(
        self,
        sigma:               float = LEAGUE_SIGMA,
        garbage_thr:         float = GARBAGE_SPREAD_THR,
        garbage_pct:         float = GARBAGE_ADJUST_PCT,
        edge_total_min:      float = EDGE_TOTAL_MIN,
        edge_spread_min:     float = EDGE_SPREAD_MIN,
        kelly_fraction:      float = KELLY_FRACTION,
        market_blend_factor: float = MARKET_BLEND_FACTOR,
        edge_shrink_factor:  float = EDGE_SHRINK_FACTOR,
    ):
        self.sigma               = sigma
        self.garbage_thr         = garbage_thr
        self.garbage_pct         = garbage_pct
        self.edge_total_min      = edge_total_min
        self.edge_spread_min     = edge_spread_min
        self.kelly_fraction      = kelly_fraction
        self.market_blend_factor = market_blend_factor
        self.edge_shrink_factor  = edge_shrink_factor

    # ── Main entry point ──────────────────────────────────────────────────────
    def scan(
        self,
        home_pred_pts:   float,
        away_pred_pts:   float,
        home_win_prob:   float,          # LightGBM straight-up win prob (home)
        home_eppps:      dict,           # {window: ePPP}
        away_eppps:      dict,
        home_pace:       float,
        away_pace:       float,
        market_spread:   Optional[float] = None,
        market_total:    Optional[float] = None,
        market_ml_home:  Optional[int]   = None,   # American ML, home side
        market_ml_away:  Optional[int]   = None,   # American ML, away side (required for ML bets)
        bankroll:        float           = 1000.0,
        spread_juice:    int             = -110,
        total_juice:     int             = -110,
    ) -> dict:
        """
        Compute fair lines and return a full projection dict with tiered plays.

        Parameters
        ──────────
        home/away_pred_pts  : XGBoost point predictions for each team
        home_win_prob       : LightGBM win probability (home perspective)
        home/away_eppps     : dict of {window: ePPP} from FeatureEngine
        home/away_pace      : recent rolling possessions per game
        market_ml_home      : American ML for home side (e.g. –150)
        market_ml_away      : American ML for away side (e.g. +130)
                              BOTH sides are required for any ML betting signal.
                              If only one side is supplied, ML bets are skipped.
        market_*            : optional market lines for edge calculation
        bankroll            : current bankroll in $ for Kelly sizing
        spread_juice        : American-odds price on spread bets (default –110)
        total_juice         : American-odds price on totals bets (default –110)

        Output schema (per play in result["plays"])
        ──────────────────────────────────────────
        DISPLAY fields (always populated, safe for frontend rendering):
          grade             : "A" | "B" | "C"
          market_type       : "SPREAD" | "TOTAL" | "MONEYLINE"
          type              : alias for market_type (legacy compat)
          side              : "HOME" | "AWAY" | "OVER" | "UNDER"
          raw_edge          : edge before shrinkage  (diagnostic only)
          calibrated_edge   : shrunk + capped edge   (display-safe)
          win_prob          : calibrated win probability (display)
          ev                : expected value per $100 stake
          play_quality_score: composite display-rank score (not for bet sizing)
          display_rank      : rank within this game's display list (1 = best)
          game_rank         : rank among bettable plays for this game

        BET SIZING fields (use only when is_bettable=True):
          is_bettable       : True → include in wagering; False → display only
          kelly_$           : recommended bet size (0 if not bettable)
          recommended_for_bet: alias for is_bettable (convenience)

        INFORMATIONAL fields:
          market_line       : the market line this play was evaluated against
          proj_line         : the projected line from the model
          market_prob_devigged: (ML only) devigged market probability
        """

        # ── 1. Projected Spread ───────────────────────────────────────────────
        raw_proj_spread = round(home_pred_pts - away_pred_pts, 2)

        # [Guard 10] Clamp extreme projections
        if abs(raw_proj_spread) > PROJ_SPREAD_MAX:
            log.warning(
                "[Guard 10] proj_spread clamped: raw=%.2f → ±%.1f",
                raw_proj_spread, PROJ_SPREAD_MAX,
            )
        proj_spread = round(
            math.copysign(min(abs(raw_proj_spread), PROJ_SPREAD_MAX), raw_proj_spread), 2
        )

        # ── 2. Poisson total ─────────────────────────────────────────────────
        proj_total = self._poisson_total(home_eppps, away_eppps, home_pace, away_pace)

        # [Guard 10] Clamp total projection
        if proj_total < PROJ_TOTAL_MIN or proj_total > PROJ_TOTAL_MAX:
            log.warning(
                "[Guard 10] proj_total clamped: raw=%.2f → [%.0f, %.0f]",
                proj_total, PROJ_TOTAL_MIN, PROJ_TOTAL_MAX,
            )
        proj_total = round(max(PROJ_TOTAL_MIN, min(proj_total, PROJ_TOTAL_MAX)), 2)

        # ── 3. Garbage-time adjustment ───────────────────────────────────────
        garbage_adj = abs(proj_spread) > self.garbage_thr
        proj_total  = round(
            proj_total * (1 - self.garbage_pct) if garbage_adj else proj_total, 2
        )

        # ── 3.5 Market-line blending  [Guard 14]  (NEW in v9) ─────────────────
        # Pull the effective spread/total used for edge and probability
        # calculations toward the market line.  The display still shows the
        # raw model projection (proj_spread / proj_total) so users can see the
        # full disagreement, but the maths use a blended number that is more
        # conservative.
        #
        # Market convention: market_spread is the line from the home team's
        # perspective (e.g., -3 means home is a 3-point favourite, so the
        # market's implied home margin is -market_spread = +3).
        if market_spread is not None:
            market_implied_spread = -market_spread        # what market thinks home wins by
            eff_spread = round(
                proj_spread * (1.0 - self.market_blend_factor)
                + market_implied_spread * self.market_blend_factor,
                2,
            )
            log.debug(
                "[Guard 14] Spread blending: proj=%.2f  mkt_implied=%.2f  "
                "eff=%.2f  (blend_factor=%.2f)",
                proj_spread, market_implied_spread, eff_spread, self.market_blend_factor,
            )
        else:
            eff_spread = proj_spread    # no market data → no blending

        if market_total is not None:
            eff_total = round(
                proj_total * (1.0 - self.market_blend_factor)
                + market_total * self.market_blend_factor,
                2,
            )
            log.debug(
                "[Guard 14] Total blending: proj=%.2f  mkt=%.2f  "
                "eff=%.2f  (blend_factor=%.2f)",
                proj_total, market_total, eff_total, self.market_blend_factor,
            )
        else:
            eff_total = proj_total      # no market data → no blending

        # ── 4. Blended + calibrated home win probability  [Guard 8] ──────────
        # [Guard 14] Use eff_spread (market-blended) for CDF so that the win
        # probability is also pulled toward the market consensus.
        # The 0.50/0.50 blend between model and CDF still applies; using
        # eff_spread in the CDF leg means the blend is more conservative
        # when the model and market disagree.
        shrunk_spread  = eff_spread * CDF_SPREAD_SHRINK   # was proj_spread
        cdf_win_prob   = self._cdf_win_prob(shrunk_spread)
        blend_win_prob = (
            WIN_PROB_BLEND_MODEL * home_win_prob
            + WIN_PROB_BLEND_CDF  * cdf_win_prob
        )
        # Pull blend toward 50 % — prevents model artefacts from looking certain.
        calib_win_prob = round(
            0.5 + (blend_win_prob - 0.5) * WIN_PROB_SHRINK_TO_50, 4
        )

        fair_ml_home = fair_prob_to_american(calib_win_prob)
        fair_ml_away = fair_prob_to_american(1.0 - calib_win_prob)

        result = {
            "proj_spread":        proj_spread,
            "proj_total":         proj_total,
            "garbage_adj":        garbage_adj,
            "win_prob_home":      calib_win_prob,
            "win_prob_away":      round(1.0 - calib_win_prob, 4),
            "fair_ml_home":       fair_ml_home,
            "fair_ml_away":       fair_ml_away,
            # Layer 1 output: all display candidates (A/B/C meeting C-tier gate)
            "plays":              [],
        }

        # ══════════════════════════════════════════════════════════════════════
        #  LAYER 1: GENERATE ALL DISPLAY CANDIDATES
        #  Gate = C-tier minimums.  is_bettable starts False for every play.
        #  Kelly sizing is NOT assigned here — that is Layer 2's job.
        # ══════════════════════════════════════════════════════════════════════

        # ── 5a. Spread edge ───────────────────────────────────────────────────
        if market_spread is not None:
            # [Guard 14] Use eff_spread (blended) for edge — reduces leverage from
            # overconfident model projections.  raw_edge is still logged for diagnostics.
            raw_spread_edge = round(proj_spread + market_spread, 2)          # diagnostic only
            blended_spread_edge = round(eff_spread + market_spread, 2)      # used for all maths

            # [Guard 2] Calibrate and cap the displayed/used edge
            calibrated_spread_edge = self._calibrate_edge(
                blended_spread_edge, SPREAD_DISPLAY_EDGE_CAP
            )

            result["spread_raw_edge"]        = raw_spread_edge
            result["spread_calibrated_edge"] = calibrated_spread_edge
            # Legacy key kept for backward compatibility with callers
            result["spread_edge"]            = calibrated_spread_edge

            # Gate: minimum C-tier edge (lower than old edge_spread_min)
            spread_gate = min(self.edge_spread_min, GRADE_C_SPREAD_EDGE)
            if abs(calibrated_spread_edge) >= spread_gate:
                side = "HOME" if calibrated_spread_edge > 0 else "AWAY"

                # [Guard 11] Compute fragility / confidence for spread inputs.
                # Spread confidence is driven by ePPP consistency across windows.
                spread_confidence = compute_projection_confidence(
                    home_eppps, away_eppps, home_pace, away_pace, is_total=False
                )

                # [Guard 12] Dynamic sigma: widen when confidence is low so an
                # uncertain edge doesn't map to an overconfident cover probability.
                dynamic_sigma = self.sigma * (
                    1.0 + (1.0 - spread_confidence) * SIGMA_FRAGILITY_BOOST
                )

                cover_prob_home = float(
                    norm.cdf(calibrated_spread_edge / dynamic_sigma)
                )
                spread_win_prob = (
                    cover_prob_home if side == "HOME" else (1.0 - cover_prob_home)
                )

                # [Guard 11] Apply fragility discount before grading.
                # This reduces both edge and win_prob for uncertain projections,
                # making A/B+ grades harder to reach when inputs are noisy.
                adj_spread_edge, adj_spread_prob = apply_fragility_discount(
                    calibrated_spread_edge, spread_win_prob, spread_confidence
                )

                ev    = self._raw_ev(adj_spread_prob, spread_juice)
                grade = self.grade_play(
                    "SPREAD", adj_spread_edge, ev, adj_spread_prob,
                    market_spread=market_spread,
                )

                # grade_play returns None for sub-C quality — skip entirely
                if grade is not None and grade in DISPLAY_GRADES:
                    quality = _play_quality_score(
                        grade, adj_spread_edge, adj_spread_prob, "SPREAD",
                        confidence=spread_confidence,   # [Guard 15] confidence-aware ranking
                    )
                    # is_bettable=False initially; Layer 2 will promote if eligible
                    result["plays"].append({
                        # -- display fields --
                        "type":             "SPREAD",
                        "market_type":      "SPREAD",
                        "side":             side,
                        "raw_edge":         raw_spread_edge,         # diagnostic
                        "calibrated_edge":  adj_spread_edge,         # fragility-adjusted
                        "edge":             adj_spread_edge,         # legacy alias
                        "win_prob":         round(adj_spread_prob, 4),
                        "ev":               round(ev, 2),
                        "grade":            grade,
                        "confidence":       spread_confidence,       # [Guard 11] new field
                        "play_quality_score": quality,               # display sort key
                        "display_rank":     None,   # set after Layer 2 sort
                        "game_rank":        None,   # set after Layer 2 sort
                        # -- bet sizing fields (populated by Layer 2) --
                        "is_bettable":      False,
                        "recommended_for_bet": False,
                        "kelly_$":          0.0,
                        # -- informational --
                        "odds":             spread_juice,
                        "market_line":      market_spread,
                        "proj_line":        proj_spread,
                    })

        # ── 5b. Totals edge ───────────────────────────────────────────────────
        if market_total is not None:
            # [Guard 14] Use eff_total (market-blended) for edge.
            raw_total_edge = round(proj_total - market_total, 2)        # diagnostic
            blended_total_edge = round(eff_total - market_total, 2)    # used for maths

            # [Guard 2] Calibrate and cap
            calibrated_total_edge = self._calibrate_edge(
                blended_total_edge, TOTAL_DISPLAY_EDGE_CAP
            )

            result["total_raw_edge"]        = raw_total_edge
            result["total_calibrated_edge"] = calibrated_total_edge
            result["total_edge"]            = calibrated_total_edge

            # Gate: minimum C-tier edge
            total_gate = min(self.edge_total_min, GRADE_C_TOTAL_EDGE)
            if abs(calibrated_total_edge) >= total_gate:
                side = "OVER" if calibrated_total_edge > 0 else "UNDER"

                # [Guard 11] Compute fragility / confidence for totals inputs.
                # Totals use is_total=True to also penalise pace mismatches.
                total_confidence = compute_projection_confidence(
                    home_eppps, away_eppps, home_pace, away_pace, is_total=True
                )

                # [Guard 3 / Guard 13] Wider sigma for totals, further widened
                # when confidence is low.  This prevents a fragile pace + ePPP
                # combo from producing a high-probability over/under.
                dynamic_totals_sigma = TOTALS_SIGMA * (
                    1.0 + (1.0 - total_confidence) * TOTALS_SIGMA_FRAGILITY_BOOST
                )

                # [Guard 3] Wider sigma for totals
                over_prob      = float(
                    norm.cdf(calibrated_total_edge / dynamic_totals_sigma)
                )
                total_win_prob = over_prob if side == "OVER" else (1.0 - over_prob)

                # [Guard 11 / 13] Apply fragility discount before grading.
                # is_over flag triggers extra haircut on low-confidence overs.
                adj_total_edge, adj_total_prob = apply_fragility_discount(
                    calibrated_total_edge, total_win_prob, total_confidence,
                    is_over=(side == "OVER"),
                )

                ev    = self._raw_ev(adj_total_prob, total_juice)
                grade = self.grade_play(
                    "TOTAL", adj_total_edge, ev, adj_total_prob,
                    bet_side=side,
                )

                if grade is not None and grade in DISPLAY_GRADES:
                    quality = _play_quality_score(
                        grade, adj_total_edge, adj_total_prob, "TOTAL",
                        confidence=total_confidence,   # [Guard 15] confidence-aware ranking
                    )
                    result["plays"].append({
                        # -- display fields --
                        "type":             "TOTAL",
                        "market_type":      "TOTAL",
                        "side":             side,
                        "raw_edge":         raw_total_edge,
                        "calibrated_edge":  adj_total_edge,          # fragility-adjusted
                        "edge":             adj_total_edge,
                        "win_prob":         round(adj_total_prob, 4),
                        "ev":               round(ev, 2),
                        "grade":            grade,
                        "confidence":       total_confidence,         # [Guard 11] new field
                        "play_quality_score": quality,
                        "display_rank":     None,
                        "game_rank":        None,
                        # -- bet sizing fields --
                        "is_bettable":      False,
                        "recommended_for_bet": False,
                        "kelly_$":          0.0,
                        # -- informational --
                        "odds":             total_juice,
                        "market_line":      market_total,
                        "proj_line":        proj_total,
                    })

        # ── 5c. Moneyline edge  [Guard 7] ─────────────────────────────────────
        # Requires BOTH market_ml_home AND market_ml_away.
        # _flip_ml() is NOT used to generate betting signals.
        # C-tier gate is ML_MIN_EDGE (0.03); grade_play() enforces A/B/C tiers.
        if market_ml_home is not None and market_ml_away is not None:
            # Devig the two-sided market to get fair market probs
            fair_mkt_home, fair_mkt_away = devig_two_way_probs(
                market_ml_home, market_ml_away
            )

            # ML edge = model prob – devigged market prob (vig removed)
            ml_edge_home = round(calib_win_prob - fair_mkt_home, 4)
            ml_edge_away = round((1.0 - calib_win_prob) - fair_mkt_away, 4)

            # Pick the side with the largest edge (if any qualifies)
            if abs(ml_edge_home) >= abs(ml_edge_away):
                ml_edge   = ml_edge_home
                side      = "HOME"
                ml_prob   = calib_win_prob
                ml_odds   = market_ml_home
                mkt_dvig  = fair_mkt_home
            else:
                ml_edge   = ml_edge_away
                side      = "AWAY"
                ml_prob   = round(1.0 - calib_win_prob, 4)
                ml_odds   = market_ml_away
                mkt_dvig  = fair_mkt_away

            result["ml_edge"]              = ml_edge
            result["ml_market_prob_home"]  = round(fair_mkt_home, 4)
            result["ml_market_prob_away"]  = round(fair_mkt_away, 4)

            # C-tier gate: positive edge >= ML_MIN_EDGE (0.03)
            if ml_edge > 0 and ml_edge >= ML_MIN_EDGE:
                ev = moneyline_ev(ml_prob, ml_odds)

                # Extra buffer required for plus-money (underdog) bets
                is_underdog    = ml_odds > 0
                required_edge  = (
                    ML_MIN_EDGE + ML_UNDERDOG_EXTRA_EDGE if is_underdog else ML_MIN_EDGE
                )
                required_prob  = ML_UNDERDOG_MIN_PROB if is_underdog else 0.0

                if ml_edge < required_edge or ml_prob < required_prob:
                    log.debug(
                        "[Guard 7] ML %s skipped: edge=%.4f < required=%.4f "
                        "or prob=%.4f < min=%.4f (plus_money=%s)",
                        side, ml_edge, required_edge, ml_prob, required_prob, is_underdog,
                    )
                else:
                    grade = self.grade_play("MONEYLINE", ml_edge, ev, ml_prob)

                    if grade is not None and grade in DISPLAY_GRADES:
                        # [Guard 7] Favorite safety valve — suppress extreme-heavy plays
                        fav_mkt_prob = max(fair_mkt_home, fair_mkt_away)
                        if fav_mkt_prob > ML_FAVORITE_SAFETY_THRESHOLD:
                            log.debug(
                                "[Guard 7] Favorite Safety Valve suppressed %s ML: "
                                "fav_market_prob=%.3f > %.2f (ml_edge=%.4f ev=$%+.2f)",
                                side, fav_mkt_prob, ML_FAVORITE_SAFETY_THRESHOLD,
                                ml_edge, ev,
                            )
                        else:
                            # [Guard 15] ML confidence — use spread confidence as a proxy
                            # because ML edge is derived from the same calib_win_prob.
                            # If spread confidence hasn't been computed yet (no market_spread),
                            # compute it now.
                            ml_confidence = compute_projection_confidence(
                                home_eppps, away_eppps, home_pace, away_pace, is_total=False
                            )
                            quality = _play_quality_score(
                                grade, ml_edge, ml_prob, "MONEYLINE",
                                confidence=ml_confidence,   # [Guard 15] confidence-aware
                            )
                            result["plays"].append({
                                # -- display fields --
                                "type":               "MONEYLINE",
                                "market_type":        "MONEYLINE",
                                "side":               side,
                                "raw_edge":           ml_edge,
                                "calibrated_edge":    ml_edge,
                                "edge":               ml_edge,
                                "win_prob":           round(ml_prob, 4),
                                "ev":                 round(ev, 2),
                                "grade":              grade,
                                "confidence":         ml_confidence,   # [Guard 15] new field
                                "play_quality_score": quality,
                                "display_rank":       None,
                                "game_rank":          None,
                                # -- bet sizing fields --
                                "is_bettable":        False,
                                "recommended_for_bet": False,
                                "kelly_$":            0.0,
                                # -- informational --
                                "odds":               ml_odds,
                                "market_prob_devigged": round(mkt_dvig, 4),
                                "market_line":        ml_odds,
                                "proj_line":          None,
                            })

        elif market_ml_home is not None and market_ml_away is None:
            log.info(
                "ML betting skipped: market_ml_away not provided. "
                "Supply both sides for proper vig removal."
            )
            result["ml_skipped_reason"] = "market_ml_away missing; both sides required"

        # ══════════════════════════════════════════════════════════════════════
        #  LAYER 2: SELECT BETTABLE SUBSET
        #  Enforce per-game limits and grade filters.
        #  Mark selected plays as is_bettable=True and assign Kelly sizing.
        # ══════════════════════════════════════════════════════════════════════
        result["plays"] = self._select_bettable_plays(
            result["plays"], bankroll, spread_juice, total_juice
        )

        # Apply display limits and assign display_rank / game_rank
        result["plays"] = self._finalize_display(result["plays"])

        return result

    # ── Layer 2: bettable selection ───────────────────────────────────────────
    def _select_bettable_plays(
        self,
        plays: List[dict],
        bankroll: float,
        spread_juice: int,
        total_juice: int,
    ) -> List[dict]:
        """
        Layer 2: choose which plays to mark is_bettable=True.

        Rules (v9 — confidence-gated, correlation-aware)
        ─────
        1. Only plays whose grade is in BET_GRADES are eligible.
        2. C-tier is never bettable unless BET_ON_C_TIER=True.
        3. [Guard 15] Hard confidence floor: plays below MIN_CONFIDENCE_FOR_BET
           are display-only regardless of grade.  ML plays additionally require
           ML_MIN_CONFIDENCE.
        4. Sort eligible plays by play_quality_score descending (which now
           includes a confidence bonus, so high-confidence plays rank higher).
        5. First pass: one play per distinct market category (SPREAD / TOTAL / ML).
        6. Second pass: fill remaining slots, but enforce:
           • [Guard 15] Second bet requires confidence ≥ MIN_CONFIDENCE_FOR_SECOND_BET.
           • [Guard 17] Correlated second bets (same game-script dependency) require
             grade ≥ CORRELATED_SECOND_BET_MIN_GRADE (B+).
        7. Assign Kelly sizing — confidence-scaled via [Guard 16].
        8. Apply per-game Kelly cap.

        NOTE: Slate-level cap across multiple games is handled externally by
              apply_slate_limits() — call that after collecting all game results.
        """
        if not plays:
            return plays

        _grade_order = {"A": 0, "B+": 1, "B": 2, "C": 3}

        # Sort all plays: grade order → quality score descending.
        # play_quality_score now includes a confidence bonus (Guard 15), so
        # high-confidence plays naturally float above fragile plays of the same grade.
        plays.sort(
            key=lambda p: (
                _grade_order.get(p["grade"], 4),
                -p.get("play_quality_score", 0),
            )
        )

        def _bets_are_correlated(p1: dict, p2: dict) -> bool:
            """
            [Guard 17] Return True when two plays share directional game-script
            exposure — i.e. both win or both lose on the same game outcome.

            Correlated pairs (both need the same script to play out):
              SPREAD HOME  + ML HOME     → both require home to win convincingly
              SPREAD AWAY  + ML AWAY     → both require away to win convincingly
              SPREAD HOME  + OVER        → home blowout = high total
              SPREAD AWAY  + UNDER       → away grinds out low-scoring win
              ML HOME      + OVER        → home favourite in a shootout
              ML AWAY      + UNDER       → underdog sneaks a defensive win

            Lower-correlation pairs (not flagged):
              SPREAD HOME  + UNDER       → home wins but game stays low
              SPREAD AWAY  + OVER        → away wins but game is high-scoring
            """
            t1, s1 = p1["market_type"], p1["side"]
            t2, s2 = p2["market_type"], p2["side"]
            pair = {(t1, s1), (t2, s2)}
            correlated_sets = [
                {("SPREAD", "HOME"),      ("MONEYLINE", "HOME")},
                {("SPREAD", "AWAY"),      ("MONEYLINE", "AWAY")},
                {("SPREAD", "HOME"),      ("TOTAL",     "OVER")},
                {("SPREAD", "AWAY"),      ("TOTAL",     "UNDER")},
                {("MONEYLINE", "HOME"),   ("TOTAL",     "OVER")},
                {("MONEYLINE", "AWAY"),   ("TOTAL",     "UNDER")},
            ]
            return any(pair == cs for cs in correlated_sets)

        # ── Build eligible pool (grade + confidence gates) ─────────────────────
        eligible = []
        for play in plays:
            grade = play["grade"]
            conf  = play.get("confidence", 1.0)

            if grade not in BET_GRADES:
                continue
            if grade == "C" and not BET_ON_C_TIER:
                continue

            # [Guard 15] Absolute confidence floor — below this the projection is
            # too noisy to bet regardless of what the grade says post-shrinkage.
            if conf < MIN_CONFIDENCE_FOR_BET:
                log.debug(
                    "[Guard 15] Play gated by MIN_CONFIDENCE_FOR_BET "
                    "(conf=%.3f < %.2f): %s %s grade=%s",
                    conf, MIN_CONFIDENCE_FOR_BET,
                    play["market_type"], play["side"], grade,
                )
                continue

            # [Guard 15] ML plays additionally require ML_MIN_CONFIDENCE because
            # their edge is derived entirely from the model win_prob — an extra
            # conservative gate is warranted.
            if play["market_type"] == "MONEYLINE" and conf < ML_MIN_CONFIDENCE:
                log.debug(
                    "[Guard 15] ML gated by ML_MIN_CONFIDENCE "
                    "(conf=%.3f < %.2f): %s grade=%s",
                    conf, ML_MIN_CONFIDENCE, play["side"], grade,
                )
                continue

            eligible.append(play)

        # ── First pass: one play per market category ───────────────────────────
        selected_count  = 0
        selected_plays  = []          # track chosen plays for correlation check
        seen_categories: dict = {}

        for play in eligible:
            if selected_count >= MAX_BETS_PER_GAME:
                break
            cat = play["market_type"]
            if seen_categories.get(cat, 0) < 1:
                play["is_bettable"]         = True
                play["recommended_for_bet"] = True
                seen_categories[cat]        = seen_categories.get(cat, 0) + 1
                selected_count             += 1
                selected_plays.append(play)

        # ── Second pass: fill remaining slots with tighter guards ──────────────
        # [Guard 15] The second bet requires higher confidence than the first,
        # because by definition it overlaps with a game where we already have
        # exposure.  [Guard 17] Correlated second bets require B+ or better.
        if selected_count < MAX_BETS_PER_GAME:
            for play in eligible:
                if selected_count >= MAX_BETS_PER_GAME:
                    break
                if play["is_bettable"]:
                    continue

                conf  = play.get("confidence", 1.0)
                grade = play["grade"]

                # [Guard 15] Stricter confidence required for a second bet
                if conf < MIN_CONFIDENCE_FOR_SECOND_BET:
                    log.debug(
                        "[Guard 15] Second bet gated by MIN_CONFIDENCE_FOR_SECOND_BET "
                        "(conf=%.3f < %.2f): %s %s grade=%s",
                        conf, MIN_CONFIDENCE_FOR_SECOND_BET,
                        play["market_type"], play["side"], grade,
                    )
                    continue

                # [Guard 17] Correlation check: if this play is directionally
                # correlated with any already-selected play, require
                # CORRELATED_SECOND_BET_MIN_GRADE (B+ or better).
                correlated = any(
                    _bets_are_correlated(play, sel) for sel in selected_plays
                )
                if correlated:
                    min_rank = _grade_order.get(CORRELATED_SECOND_BET_MIN_GRADE, 1)
                    if _grade_order.get(grade, 4) > min_rank:
                        log.debug(
                            "[Guard 17] Correlated second bet suppressed "
                            "(grade=%s < required %s): %s %s",
                            grade, CORRELATED_SECOND_BET_MIN_GRADE,
                            play["market_type"], play["side"],
                        )
                        continue

                play["is_bettable"]         = True
                play["recommended_for_bet"] = True
                cat = play["market_type"]
                seen_categories[cat]        = seen_categories.get(cat, 0) + 1
                selected_count             += 1
                selected_plays.append(play)

        # ── Assign Kelly to bettable plays (confidence-scaled) ─────────────────
        # [Guard 16] _kelly now accepts a confidence argument and scales the
        # final bet size down proportionally — see _kelly() for the formula.
        juice_map = {"SPREAD": spread_juice, "TOTAL": total_juice, "MONEYLINE": None}
        for play in plays:
            if not play["is_bettable"]:
                play["kelly_$"] = 0.0
                continue
            grade    = play["grade"]
            win_prob = play["win_prob"]
            conf     = play.get("confidence", 1.0)
            if play["market_type"] == "MONEYLINE":
                market_ml = play["market_line"]
            else:
                market_ml = juice_map[play["market_type"]]
            play["kelly_$"] = self._kelly(
                win_prob, market_ml, bankroll, grade=grade, confidence=conf
            )

        # Per-game Kelly cap (single-game guardrail; slate cap handled externally)
        total_kelly = sum(p["kelly_$"] for p in plays if p["is_bettable"])
        max_game    = SLATE_KELLY_CAP * bankroll
        if total_kelly > max_game and total_kelly > 0:
            scale_factor = max_game / total_kelly
            log.warning(
                "Per-game Kelly cap triggered: total=%.2f > max=%.2f. "
                "Scaling %d bettable bet(s) by %.4f.",
                total_kelly, max_game, selected_count, scale_factor,
            )
            for play in plays:
                if play["is_bettable"]:
                    play["kelly_$"] = round(play["kelly_$"] * scale_factor, 2)

        return plays

    # ── Display finalisation ──────────────────────────────────────────────────
    def _finalize_display(self, plays: List[dict]) -> List[dict]:
        """
        After bettable selection, trim the display list and assign ranks.

        Steps
        ─────
        1. Filter to DISPLAY_GRADES.
        2. If SHOW_C_TIER is False, drop C plays.
        3. Trim to MAX_DISPLAY_PLAYS_PER_GAME (keeping bettable plays first).
        4. Assign display_rank (overall, 1-indexed) and game_rank (bettable only).
        """
        _grade_order = {"A": 0, "B+": 1, "B": 2, "C": 3}

        # Filter by display grades
        display = [p for p in plays if p["grade"] in DISPLAY_GRADES]

        # Optionally hide C-tier from output
        if not SHOW_C_TIER:
            display = [p for p in display if p["grade"] != "C"]

        # Sort: bettable first within each grade tier, then by quality
        display.sort(
            key=lambda p: (
                _grade_order.get(p["grade"], 4),
                0 if p["is_bettable"] else 1,     # bettable first within grade
                -p.get("play_quality_score", 0),
            )
        )

        # Trim to display limit
        display = display[:MAX_DISPLAY_PLAYS_PER_GAME]

        # Assign display_rank (1 = highest quality for display)
        for rank, play in enumerate(display, start=1):
            play["display_rank"] = rank

        # Assign game_rank only to bettable plays within this game
        game_rank = 1
        for play in display:
            if play["is_bettable"]:
                play["game_rank"] = game_rank
                game_rank += 1
            else:
                play["game_rank"] = None

        return display

    # ── Maths helpers ─────────────────────────────────────────────────────────
    def _poisson_total(
        self,
        home_eppps: dict,
        away_eppps: dict,
        home_pace:  float,
        away_pace:  float,
    ) -> float:
        """
        Project game total using a weighted Poisson model.
        Rolling-window weights: 5-game = 40 %, 10-game = 35 %, 20-game = 25 %.
        """
        shared_pace  = (home_pace + away_pace) / 2.0
        weights      = {5: 0.40, 10: 0.35, 20: 0.25}
        home_blended = sum(weights.get(w, 0) * v for w, v in home_eppps.items())
        away_blended = sum(weights.get(w, 0) * v for w, v in away_eppps.items())
        return home_blended * shared_pace + away_blended * shared_pace

    def _cdf_win_prob(self, spread: float) -> float:
        """
        P(home wins outright) = Φ(spread / σ).
        Caller is responsible for any pre-shrinkage of the spread argument.
        """
        return float(norm.cdf(spread / self.sigma))

    def _calibrate_edge(self, raw_edge: float, cap: float) -> float:
        """
        [Guard 2] Two-stage shrinkage:
          1. Multiply by edge_shrink_factor to compress raw outliers.
          2. Hard-cap at `cap` (sign-preserving).

        Returns calibrated_edge with the same sign as raw_edge.
        """
        shrunk = raw_edge * self.edge_shrink_factor
        return round(
            math.copysign(min(abs(shrunk), cap), shrunk), 4
        )

    @staticmethod
    def _prob_to_american_ml(p: float) -> int:
        """Convert a decimal win probability to American moneyline odds."""
        return fair_prob_to_american(p)

    @staticmethod
    def _american_ml_to_prob(ml: int) -> float:
        """Convert American moneyline to raw implied probability (vig retained)."""
        return american_to_implied_prob(ml)

    @staticmethod
    def _flip_ml(ml: int) -> int:
        """
        Approximate the opposite side's American ML.
        DEPRECATED for betting signals — kept only for fair-line display.
        Do NOT use to generate ML edge or plays.
        """
        p   = american_to_implied_prob(ml)
        opp = 1.0 - p
        return fair_prob_to_american(opp)

    # ── EV & Grading ─────────────────────────────────────────────────────────
    @staticmethod
    def _raw_ev(win_prob: float, market_ml: int, stake: float = 100.0) -> float:
        """
        Expected value of a flat $stake wager at market_ml odds.
        EV = (win_prob × profit_if_win) – (loss_prob × stake)
        """
        if market_ml > 0:
            profit_if_win = stake * market_ml / 100.0
        else:
            profit_if_win = stake * 100.0 / abs(market_ml)
        return (win_prob * profit_if_win) - ((1.0 - win_prob) * stake)

    @staticmethod
    def grade_play(
        bet_type:     str,
        edge:         float,
        ev:           float,
        win_prob:     float,
        bet_side:     Optional[str]   = None,
        market_spread: Optional[float] = None,
    ) -> Optional[str]:
        """
        Assign a letter grade, or return None if the play is below C-tier quality.

        Returns None for sub-C plays — callers should skip None grades entirely.

        Grade hierarchy (top → bottom):  A  >  B+  >  B  >  C

        [Guard 1] Probability-weighted grading
        ────────────────────────────────────────
        Absolute floor  : win_prob < 35 %  → None (rejected).
        Negative EV     : ev < 0           → None (rejected).

        C-tier minimum gates (play appears in display but NOT for betting):
        ─────────────────────────────────────────────────────────────────────
          SPREAD  : calibrated_edge ≥ 3.0 pts  AND win_prob ≥ 51.5 %
          TOTAL   : calibrated_edge ≥ 3.5 pts  AND win_prob ≥ 51.5 %
          ML      : edge ≥ 3 %               AND win_prob ≥ 52.0 %
          → plays below these return None (not displayed at all)

        B-tier thresholds (bettable, lower priority):
        ─────────────────────────────────────────────
          SPREAD  : calibrated_edge ≥ 5.0 pts  AND win_prob ≥ 54 %
          TOTAL   : calibrated_edge ≥ 6.0 pts  AND win_prob ≥ 54 %
          ML      : edge ≥ 5 %               AND win_prob ≥ 55 %

        B+-tier thresholds (bettable, near-A plays):
        ─────────────────────────────────────────────
          Fires when the play passes B+ thresholds AND is_close_to_a() is True.
          SPREAD  : calibrated_edge ≥ 6.2 pts  AND win_prob ≥ 55 %
          TOTAL   : calibrated_edge ≥ 7.0 pts  AND win_prob ≥ 55 %
          ML      : edge ≥ 6 %               AND win_prob ≥ 56.5 %

        A-tier thresholds (strict, most trustworthy — always bettable):
        ─────────────────────────────────────────────────────────────────
          SPREAD  : calibrated_edge ≥ 8.0 pts  AND win_prob ≥ 58.5 % (was 7.0/56%)
          TOTAL OVER  : calibrated_edge ≥ 9.5 pts AND win_prob ≥ 58.5 %
          TOTAL UNDER : calibrated_edge ≥ 9.5 pts AND win_prob ≥ 63.0 %
            (UNDER requires higher floor — pace spikes and garbage-time offense
             regularly blow up NBA unders regardless of projection accuracy)
          ML      : edge ≥ 8 %               AND win_prob ≥ 60 %    (was 7%/58%)

        Large market spread gate:
        ─────────────────────────────────────────────────────────────────
          If |market_spread| > 13 pts, the play is capped at grade B.
          Covering 16.5 pts is far less reliable than covering 6.5 pts;
          this gate caught the MEM/MIL/UTA/WSH failures from 2025-04-11.

        IMPORTANT (v8): inputs to grade_play() should already be fragility-adjusted
        (via apply_fragility_discount()).  The thresholds have been tightened so
        that even after Guard 2 edge shrinkage, a play still needs a genuinely
        robust signal to qualify as A or B+.

        Calibration preserved: A-tier thresholds are NOT loosened from v8.
        """
        # ── Absolute floor ────────────────────────────────────────────────────
        if win_prob < GRADE_C_MAX_WIN_PROB:
            log.debug(
                "Play rejected: win_prob=%.3f < absolute floor %.2f",
                win_prob, GRADE_C_MAX_WIN_PROB,
            )
            return None

        # ── Negative EV is always rejected ───────────────────────────────────
        if ev < 0:
            return None

        abs_edge = abs(edge)

        if bet_type == "MONEYLINE":
            thr = GRADE_THRESHOLDS["moneyline"]
            # A-tier
            if abs_edge >= thr["A"]["edge"] and win_prob >= thr["A"]["win_prob"]:
                return "A"
            # B+-tier: passes B+ threshold AND is close to A
            if (abs_edge >= thr["B_PLUS"]["edge"]
                    and win_prob >= thr["B_PLUS"]["win_prob"]
                    and is_close_to_a("MONEYLINE", abs_edge, win_prob)):
                return "B+"
            # B-tier
            if abs_edge >= thr["B"]["edge"] and win_prob >= thr["B"]["win_prob"]:
                return "B"
            # C-tier gate
            if abs_edge >= thr["C"]["edge"] and win_prob >= thr["C"]["win_prob"]:
                return "C"
            return None

        elif bet_type == "SPREAD":
            thr = GRADE_THRESHOLDS["spread"]
            # [Guard 1-ext] Large market spread cap: covering 13+ pts is too
            # uncertain to warrant A/B+ — skip straight to B/C evaluation.
            if (
                market_spread is not None
                and abs(market_spread) > LARGE_MARKET_SPREAD_THR
            ):
                log.debug(
                    "Large spread cap: |market_spread|=%.1f > %.1f — capped at B",
                    abs(market_spread), LARGE_MARKET_SPREAD_THR,
                )
                if abs_edge >= thr["B"]["edge"] and win_prob >= thr["B"]["win_prob"]:
                    return "B"
                if abs_edge >= thr["C"]["edge"] and win_prob >= thr["C"]["win_prob"]:
                    return "C"
                return None
            # A-tier
            if abs_edge >= thr["A"]["edge"] and win_prob >= thr["A"]["win_prob"]:
                return "A"
            # B+-tier
            if (abs_edge >= thr["B_PLUS"]["edge"]
                    and win_prob >= thr["B_PLUS"]["win_prob"]
                    and is_close_to_a("SPREAD", abs_edge, win_prob)):
                return "B+"
            # B-tier
            if abs_edge >= thr["B"]["edge"] and win_prob >= thr["B"]["win_prob"]:
                return "B"
            # C-tier gate
            if abs_edge >= thr["C"]["edge"] and win_prob >= thr["C"]["win_prob"]:
                return "C"
            return None

        else:  # TOTAL
            thr = GRADE_THRESHOLDS["total"]
            # [Guard 1-ext] UNDER-specific A-tier win_prob floor.
            # NBA unders are fragile — pace spikes and garbage-time scoring
            # routinely blow them up regardless of projection accuracy.
            a_win_prob = (
                GRADE_A_UNDER_WIN_PROB
                if bet_side == "UNDER"
                else thr["A"]["win_prob"]
            )
            # A-tier
            if abs_edge >= thr["A"]["edge"] and win_prob >= a_win_prob:
                return "A"
            # B+-tier
            if (abs_edge >= thr["B_PLUS"]["edge"]
                    and win_prob >= thr["B_PLUS"]["win_prob"]
                    and is_close_to_a("TOTAL", abs_edge, win_prob)):
                return "B+"
            # B-tier
            if abs_edge >= thr["B"]["edge"] and win_prob >= thr["B"]["win_prob"]:
                return "B"
            # C-tier gate
            if abs_edge >= thr["C"]["edge"] and win_prob >= thr["C"]["win_prob"]:
                return "C"
            return None

    def _kelly(
        self,
        win_prob:   float,
        market_ml:  int,
        bankroll:   float,
        grade:      str   = "A",
        confidence: float = 1.0,   # [Guard 16] NEW — confidence scaling
    ) -> float:
        """
        Fractional Kelly bet size in dollars.

        FOR BET SIZING ONLY — only called for is_bettable=True plays.

        [Guard 9] Uses the final calibrated win_prob directly.
        No secondary edge-derived probability — eliminates double-counting.

        [Guard 4] Dynamic grade-dependent ceiling:
          Grade A  → MAX_KELLY_PCT           (4 %)
          Grade B+ → GRADE_BPLUS_MAX_KELLY_PCT (3 %)
          Grade B  → GRADE_B_MAX_KELLY_PCT   (2 %)
          Grade C  → 0 (not bettable, should never reach _kelly)

        [Guard 16] Confidence scaling (NEW in v9):
          After the grade cap, the bet is multiplied by:
            max(CONFIDENCE_KELLY_SCALE_FLOOR, confidence)
          This creates a smooth sizing ramp independent of the grade caps.
          Example: confidence=0.55 → bet is 55% of grade cap.
                   confidence=1.00 → bet is 100% of grade cap (no reduction).
          The floor (0.40) prevents sizing from collapsing to zero for plays
          that cleared the confidence gate but are near the floor.

        USE_FLAT_BET_SIZING override: if True, returns FLAT_BET_AMOUNT
        for Grade A/B+/B plays (useful for backtests).
        """
        if grade == "C":
            # C-tier should never reach here but guard defensively
            return 0.0

        if USE_FLAT_BET_SIZING:
            return FLAT_BET_AMOUNT if grade in ("A", "B+", "B") else 0.0

        # Decimal odds
        if market_ml > 0:
            decimal_odds = 1.0 + market_ml / 100.0
        else:
            decimal_odds = 1.0 + 100.0 / abs(market_ml)

        b = decimal_odds - 1.0
        if b <= 0:
            return 0.0

        p = win_prob
        q = 1.0 - p

        full_kelly       = max((b * p - q) / b, 0.0)
        fractional_kelly = full_kelly * self.kelly_fraction
        raw_bet          = fractional_kelly * bankroll

        # Half-Kelly dampener before the hard ceiling
        raw_bet = raw_bet * 0.5

        # [Guard 4] Grade-dependent ceiling
        if grade == "A":
            max_bet = MAX_KELLY_PCT * bankroll
        elif grade == "B+":
            max_bet = GRADE_BPLUS_MAX_KELLY_PCT * bankroll
        else:
            max_bet = GRADE_B_MAX_KELLY_PCT * bankroll
        capped_bet = min(raw_bet, max_bet)

        # [Guard 16] Confidence scaling: reduce sizing for lower-confidence plays.
        # Plays near the confidence betting floor get proportionally smaller bets.
        # This is a separate axis from the grade cap — both apply simultaneously.
        conf_scale = max(CONFIDENCE_KELLY_SCALE_FLOOR, confidence)
        scaled_bet = capped_bet * conf_scale

        if raw_bet > max_bet or conf_scale < 1.0:
            log.debug(
                "Kelly sizing: Grade=%s conf=%.2f scale=%.2f "
                "raw=%.2f grade_cap=%.2f conf_scaled=%.2f",
                grade, confidence, conf_scale,
                raw_bet, capped_bet, scaled_bet,
            )

        return round(scaled_bet, 2)


# ══════════════════════════════════════════════════════════════════════════════
#  PRETTY PRINTER
# ══════════════════════════════════════════════════════════════════════════════
def print_projection(
    home_team: str,
    away_team: str,
    result:    dict,
    bankroll:  float = 1000.0,
) -> None:
    """
    Render a formatted projection card to stdout.

    Display note: plays are shown in full (A/B/C), but the terminal output
    distinguishes bettable plays (✦) from display-only plays (◇).
    Only bettable plays show Kelly bet sizing.
    """
    sep = "═" * 60
    print(f"\n{sep}")
    print(f"  NBA PROJECTION  ·  {away_team}  @  {home_team}")
    print(sep)
    spread_str = (
        f"{home_team} {result['proj_spread']:+.1f}"
        if result["proj_spread"] != 0
        else "Pick'em"
    )
    print(f"  Projected Spread : {spread_str}")
    print(f"  Projected Total  : {result['proj_total']:.1f}"
          + ("  ← garbage-time adj." if result["garbage_adj"] else ""))
    print(f"  Win Prob (Home)  : {result['win_prob_home'] * 100:.1f}%")
    print(f"  Win Prob (Away)  : {result['win_prob_away'] * 100:.1f}%")
    print(f"  Fair ML  (Home)  : {result['fair_ml_home']:+d}")
    print(f"  Fair ML  (Away)  : {result['fair_ml_away']:+d}")

    # Show calibrated edges only — raw edges are diagnostic, not displayed
    if "spread_calibrated_edge" in result:
        raw  = result.get("spread_raw_edge", "n/a")
        cal  = result["spread_calibrated_edge"]
        print(f"\n  Spread Edge      : {cal:+.2f} pts  (raw={raw:+.2f}, fragility-adj)")
    if "total_calibrated_edge" in result:
        raw  = result.get("total_raw_edge", "n/a")
        cal  = result["total_calibrated_edge"]
        print(f"  Total  Edge      : {cal:+.2f} pts  (raw={raw:+.2f}, fragility-adj)")
    if "ml_edge" in result:
        print(f"  ML     Edge      : {result['ml_edge']:+.4f} (devigged prob)")
        if "ml_market_prob_home" in result:
            print(
                f"  ML Mkt Prob      : home={result['ml_market_prob_home']:.3f}"
                f"  away={result['ml_market_prob_away']:.3f}  (vig removed)"
            )
    if "ml_skipped_reason" in result:
        print(f"  ML Skipped       : {result['ml_skipped_reason']}")

    if result["plays"]:
        bettable_plays = [p for p in result["plays"] if p["is_bettable"]]
        display_only   = [p for p in result["plays"] if not p["is_bettable"]]

        total_kelly = sum(p["kelly_$"] for p in bettable_plays)
        print(f"\n  ── PLAYS  (bankroll ${bankroll:,.0f})  "
              f"[{len(bettable_plays)} bettable · {len(display_only)} display-only] ──")

        if total_kelly > 0 and total_kelly >= SLATE_KELLY_CAP * bankroll * 0.999:
            print(
                f"  ⚠  Game cap active — total risk ≤ "
                f"{SLATE_KELLY_CAP * 100:.0f}% of bankroll "
                f"(${SLATE_KELLY_CAP * bankroll:,.0f})"
            )

        _GRADE_COLOR = {"A": _C.GREEN, "B+": _C.CYAN, "B": _C.YELLOW, "C": _C.RED}

        for play in result["plays"]:
            grade        = play.get("grade", "?")
            grade_color  = _GRADE_COLOR.get(grade, _C.RESET)
            grade_str    = _colorize(f"[{grade}]", grade_color)
            win_prob_pct = play.get("win_prob", 0.0) * 100
            is_bet       = play.get("is_bettable", False)
            marker       = "✦" if is_bet else "◇"

            # Kelly note: only bettable plays show sizing
            if is_bet:
                kelly_note = f"  Kelly bet=${play['kelly_$']:,.2f}"
            elif grade == "C":
                kelly_note = "  [C-tier · display only · no bet]"
            else:
                kelly_note = "  [display only · not selected this game]"

            if play["type"] == "MONEYLINE":
                edge_display = f"{play['calibrated_edge'] * 100:+.2f}%"
                mkt_dvig_str = (
                    f"  mkt_dvig={play['market_prob_devigged']:.3f}"
                    if "market_prob_devigged" in play else ""
                )
            else:
                edge_display = f"{play['calibrated_edge']:+.2f} pts"
                mkt_dvig_str = ""

            rank_str = (
                f"  rank=#{play['display_rank']}"
                if play.get("display_rank") is not None else ""
            )
            # [Guard 11] Show confidence score for spread/total plays so users
            # can see how fragile the underlying projection is.
            conf_str = (
                f"  conf={play['confidence']:.2f}"
                if play.get("confidence") is not None else ""
            )

            odds_val = play.get("odds")
            odds_str = f"  odds={odds_val:+d}" if odds_val is not None else ""

            print(
                f"  {marker} {grade_str} {play['type']:<10} {play['side']:<5}"
                f"  edge={edge_display}"
                f"  prob={win_prob_pct:.1f}%"
                f"  EV=${play.get('ev', 0):+.2f}"
                f"{odds_str}"
                f"  qscore={play.get('play_quality_score', 0):.1f}"
                f"{mkt_dvig_str}"
                f"{conf_str}"
                f"{rank_str}"
                f"{kelly_note}"
            )
    else:
        print("\n  No qualifying plays (A/B+/B/C) flagged vs. market lines.")

    # Grade legend
    print(
        f"\n  Legend: "
        f"{_colorize('[A]', _C.GREEN)} High Value  "
        f"{_colorize('[B+]', _C.CYAN)} Near-A Value  "
        f"{_colorize('[B]', _C.YELLOW)} Medium Value  "
        f"{_colorize('[C]', _C.RED)} Lean / Display Only"
    )
    print(sep + "\n")