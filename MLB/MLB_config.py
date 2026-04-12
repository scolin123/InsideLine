"""
mlb_config.py — Global configuration for the MLB Quantitative Modeling Pipeline.
Centralises all constants, API settings, logging, and lookup tables so every
other module can do a clean `from mlb_config import ...` without side-effects.
"""
from dotenv import load_dotenv
load_dotenv()

import os
import logging
import warnings

warnings.filterwarnings("ignore")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── MLB Seasons ───────────────────────────────────────────────────────────────
SEASONS: list[str] = ["2022", "2023", "2024", "2025"]

# ── API / pipeline hyper-parameters ──────────────────────────────────────────
API_SLEEP           = 0.50          # seconds between MLB Stats API calls
EDGE_TOTAL_MIN      = 1.5           # minimum edge (runs) to flag a total play
EDGE_SPREAD_MIN     = 0.75          # minimum edge (runs) to flag a run-line play
KELLY_FRACTION      = 0.25          # fractional Kelly (25 % of full Kelly)
LEAGUE_SIGMA        = 4.2           # historical scoring-margin std-dev (runs)
# Blowout adjustment parameters (mirroring NBA structure, MLB-tuned)
GARBAGE_SPREAD_THR  = 5.0           # run differential at which blowout adjustment kicks in
GARBAGE_ADJUST_PCT  = 0.08          # 8 % total reduction for projected blowouts
BULLPEN_FATIGUE_DAYS = 3            # rolling window (days) for bullpen pitch count
SP_RECENT_STARTS_SHORT = 5          # "short" trailing window for SP metrics
SP_RECENT_STARTS_LONG  = 10         # "long"  trailing window for SP metrics
FIP_CONSTANT        = 3.10          # league-average FIP constant (re-calibrate yearly)

# ── The-Odds-API configuration ────────────────────────────────────────────────
ODDS_API_BASE_URL   = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
ODDS_API_KEY: str   = os.environ.get("ODDS_API_KEY", "YOUR_ODDS_API_KEY_HERE")
ODDS_QUOTA_MIN      = 10
ODDS_REGIONS        = "us"
ODDS_MARKETS        = "h2h,spreads,totals"
ODDS_FORMAT         = "american"

# Backwards-compat alias for callers expecting MLB_ODDS_API_BASE_URL
MLB_ODDS_API_BASE_URL = ODDS_API_BASE_URL

# ── Supabase configuration ────────────────────────────────────────────────────
SUPABASE_URL: str = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY", "")

# ── MLB ballpark GPS coordinates (lat, lon) ───────────────────────────────────
# Used for travel-distance calculations (home series vs. road trips).
BALLPARK_COORDS: dict[str, tuple[float, float]] = {
    "ARI": (33.4453, -112.0667),   # Chase Field
    "ATL": (33.8908,  -84.4681),   # Truist Park
    "BAL": (39.2838,  -76.6218),   # Camden Yards
    "BOS": (42.3467,  -71.0972),   # Fenway Park
    "CHC": (41.9484,  -87.6553),   # Wrigley Field
    "CWS": (41.8300,  -87.6338),   # Guaranteed Rate Field
    "CIN": (39.0974,  -84.5082),   # Great American Ball Park
    "CLE": (41.4962,  -81.6852),   # Progressive Field
    "COL": (39.7559, -104.9942),   # Coors Field
    "DET": (42.3390,  -83.0485),   # Comerica Park
    "HOU": (29.7573,  -95.3555),   # Minute Maid Park
    "KC":  (39.0517,  -94.4803),   # Kauffman Stadium
    "LAA": (33.8003, -117.8827),   # Angel Stadium
    "LAD": (34.0739, -118.2400),   # Dodger Stadium
    "MIA": (25.7781,  -80.2197),   # loanDepot park
    "MIL": (43.0280,  -87.9712),   # American Family Field
    "MIN": (44.9817,  -93.2778),   # Target Field
    "NYM": (40.7571,  -73.8458),   # Citi Field
    "NYY": (40.8296,  -73.9262),   # Yankee Stadium
    "OAK": (37.7516, -122.2005),   # Oakland Coliseum
    "PHI": (39.9061,  -75.1665),   # Citizens Bank Park
    "PIT": (40.4469,  -80.0057),   # PNC Park
    "SD":  (32.7076, -117.1570),   # Petco Park
    "SF":  (37.7786, -122.3893),   # Oracle Park
    "SEA": (47.5914, -122.3325),   # T-Mobile Park
    "STL": (38.6226,  -90.1928),   # Busch Stadium
    "TB":  (27.7682,  -82.6534),   # Tropicana Field
    "TEX": (32.7512,  -97.0832),   # Globe Life Field
    "TOR": (43.6414,  -79.3894),   # Rogers Centre
    "WSH": (38.8730,  -77.0074),   # Nationals Park
}

# ── Park Factors ──────────────────────────────────────────────────────────────
# Single-number run-scoring park factor (100 = perfectly neutral).
# Values > 100 favour offence; < 100 favour pitching.
# Source: multi-year Statcast / FanGraphs averages — update each off-season.
PARK_FACTORS: dict[str, float] = {
    "ARI": 105.2,   # Hitter-friendly desert air
    "ATL": 100.4,
    "BAL": 103.1,
    "BOS": 104.8,   # Green Monster / short LF
    "CHC": 101.3,
    "CWS":  98.7,
    "CIN": 104.5,
    "CLE":  97.6,
    "COL": 115.8,   # Coors Field altitude — highest in MLB
    "DET":  98.2,
    "HOU":  98.9,
    "KC":   99.4,
    "LAA": 100.6,
    "LAD":  96.5,   # Pitcher-friendly
    "MIA":  94.8,   # Large foul territory, humid air
    "MIL":  99.1,
    "MIN": 101.7,
    "NYM": 100.3,
    "NYY": 103.5,   # Short porch RF
    "OAK":  97.3,
    "PHI": 102.8,
    "PIT":  98.6,
    "SD":   96.0,   # Marine layer
    "SF":   94.2,   # Oracle Park — pitcher-friendly
    "SEA":  97.8,
    "STL":  99.7,
    "TB":   98.3,
    "TEX": 104.1,
    "TOR": 101.5,
    "WSH": 100.9,
}

# ── wOBA linear weights (2024 MLB averages) ───────────────────────────────────
# Re-calibrate annually from Fangraphs / Baseball Reference.
WOBA_WEIGHTS: dict[str, float] = {
    "BB":  0.690,
    "HBP": 0.722,
    "1B":  0.888,
    "2B":  1.271,
    "3B":  1.616,
    "HR":  2.101,
}
WOBA_SCALE: float = 1.157   # wOBA-to-wRAA scale factor

# ── The-Odds-API full-name → team abbreviation map ───────────────────────────
ODDS_TEAM_NAME_MAP: dict[str, str] = {
    "Arizona Diamondbacks":    "ARI",
    "Atlanta Braves":          "ATL",
    "Baltimore Orioles":       "BAL",
    "Boston Red Sox":          "BOS",
    "Chicago Cubs":            "CHC",
    "Chicago White Sox":       "CWS",
    "Cincinnati Reds":         "CIN",
    "Cleveland Guardians":     "CLE",
    "Colorado Rockies":        "COL",
    "Detroit Tigers":          "DET",
    "Houston Astros":          "HOU",
    "Kansas City Royals":      "KC",
    "Los Angeles Angels":      "LAA",
    "Los Angeles Dodgers":     "LAD",
    "Miami Marlins":           "MIA",
    "Milwaukee Brewers":       "MIL",
    "Minnesota Twins":         "MIN",
    "New York Mets":           "NYM",
    "New York Yankees":        "NYY",
    "Oakland Athletics":       "OAK",
    "Philadelphia Phillies":   "PHI",
    "Pittsburgh Pirates":      "PIT",
    "San Diego Padres":        "SD",
    "San Francisco Giants":    "SF",
    "Seattle Mariners":        "SEA",
    "St. Louis Cardinals":     "STL",
    "Tampa Bay Rays":          "TB",
    "Texas Rangers":           "TEX",
    "Toronto Blue Jays":       "TOR",
    "Washington Nationals":    "WSH",
}

# ── MLB Stats API numeric team-id → abbreviation (stable across seasons) ─────
# Primary lookup used by the data loader; more reliable than teamName strings.
STATSAPI_TEAM_ID_MAP: dict[int, str] = {
    108: "LAA",
    109: "ARI",
    110: "BAL",
    111: "BOS",
    112: "CHC",
    113: "CIN",
    114: "CLE",
    115: "COL",
    116: "DET",
    117: "HOU",
    118: "KC",
    119: "LAD",
    120: "WSH",
    121: "NYM",
    133: "OAK",
    134: "PIT",
    135: "SD",
    136: "SEA",
    137: "SF",
    138: "STL",
    139: "TB",
    140: "TEX",
    141: "TOR",
    142: "MIN",
    143: "PHI",
    144: "ATL",
    145: "CWS",
    146: "MIA",
    147: "NYY",
    158: "MIL",
}

# ── MLB Stats API team-name → abbreviation (statsapi internal names) ──────────
STATSAPI_TEAM_ABV_MAP: dict[str, str] = {
    "D-backs":        "ARI",
    "Braves":         "ATL",
    "Orioles":        "BAL",
    "Red Sox":        "BOS",
    "Cubs":           "CHC",
    "White Sox":      "CWS",
    "Reds":           "CIN",
    "Guardians":      "CLE",
    "Rockies":        "COL",
    "Tigers":         "DET",
    "Astros":         "HOU",
    "Royals":         "KC",
    "Angels":         "LAA",
    "Dodgers":        "LAD",
    "Marlins":        "MIA",
    "Brewers":        "MIL",
    "Twins":          "MIN",
    "Mets":           "NYM",
    "Yankees":        "NYY",
    "Athletics":      "OAK",
    "Phillies":       "PHI",
    "Pirates":        "PIT",
    "Padres":         "SD",
    "Giants":         "SF",
    "Mariners":       "SEA",
    "Cardinals":      "STL",
    "Rays":           "TB",
    "Rangers":        "TEX",
    "Blue Jays":      "TOR",
    "Nationals":      "WSH",
}