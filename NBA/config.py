"""
config.py — Global configuration for the NBA Quantitative Modeling Pipeline.
Centralises all constants, API settings, logging, and lookup tables so every
other module can do a clean `from config import ...` without side-effects.
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

# ── NBA arena GPS coordinates (lat, lon) ──────────────────────────────────────
ARENA_COORDS: dict[str, tuple[float, float]] = {
    "ATL": (33.7573, -84.3963), "BOS": (42.3662, -71.0621),
    "BKN": (40.6826, -73.9754), "CHA": (35.2251, -80.8392),
    "CHI": (41.8807, -87.6742), "CLE": (41.4965, -81.6881),
    "DAL": (32.7905, -96.8103), "DEN": (39.7487, -105.0077),
    "DET": (42.3410, -83.0554), "GSW": (37.7680, -122.3877),
    "HOU": (29.7508, -95.3621), "IND": (39.7640, -86.1555),
    "LAC": (33.9425, -118.1080), "LAL": (34.0430, -118.2673),
    "MEM": (35.1382, -90.0506), "MIA": (25.7814, -80.1870),
    "MIL": (43.0450, -87.9170), "MIN": (44.9795, -93.2762),
    "NOP": (29.9490, -90.0821), "NYK": (40.7505, -73.9934),
    "OKC": (35.4634, -97.5151), "ORL": (28.5392, -81.3839),
    "PHI": (39.9012, -75.1720), "PHX": (33.4457, -112.0712),
    "POR": (45.5316, -122.6669), "SAC": (38.5802, -121.4997),
    "SAS": (29.4269, -98.4375), "TOR": (43.6435, -79.3791),
    "UTA": (40.7683, -111.9011), "WAS": (38.8981, -77.0209),
}

# ── Pipeline hyper-parameters ─────────────────────────────────────────────────
SEASONS            = ["2021-22", "2022-23", "2023-24", "2024-25"]
API_SLEEP          = 0.65          # seconds between NBA API calls
LEAGUE_SIGMA       = 12.5          # historical scoring-margin std-dev
GARBAGE_SPREAD_THR = 12.0          # spread threshold for garbage-time adjustment
GARBAGE_ADJUST_PCT = 0.035         # 3.5 % total reduction for blowouts
EDGE_TOTAL_MIN     = 2.0           # minimum edge (pts) to flag a total play
EDGE_SPREAD_MIN    = 1.5           # minimum edge (pts) to flag a spread play
KELLY_FRACTION     = 0.25          # fractional Kelly (25 % of full Kelly)

# ── The-Odds-API configuration ────────────────────────────────────────────────
ODDS_API_BASE_URL  = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
ODDS_API_KEY       = "YOUR_ODDS_API_KEY_HERE"   # ← replace or set via env var
ODDS_QUOTA_MIN     = 10             # refuse to call if fewer requests remain
ODDS_REGIONS       = "us"
ODDS_MARKETS       = "h2h,spreads,totals"
ODDS_FORMAT        = "american"

# ── Supabase configuration ────────────────────────────────────────────────────
# Both values are read from the .env file at startup via python-dotenv.
# Set them there — never hard-code credentials in source control.
#
#   SUPABASE_URL = https://<project-ref>.supabase.co
#   SUPABASE_KEY = <service-role or anon key>
#
SUPABASE_URL: str = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY", "")

# ── Full-name → 3-letter abbreviation map (The-Odds-API → nba_api) ────────────
ODDS_TEAM_NAME_MAP: dict[str, str] = {
    "Atlanta Hawks":          "ATL",
    "Boston Celtics":         "BOS",
    "Brooklyn Nets":          "BKN",
    "Charlotte Hornets":      "CHA",
    "Chicago Bulls":          "CHI",
    "Cleveland Cavaliers":    "CLE",
    "Dallas Mavericks":       "DAL",
    "Denver Nuggets":         "DEN",
    "Detroit Pistons":        "DET",
    "Golden State Warriors":  "GSW",
    "Houston Rockets":        "HOU",
    "Indiana Pacers":         "IND",
    "Los Angeles Clippers":   "LAC",
    "Los Angeles Lakers":     "LAL",
    "Memphis Grizzlies":      "MEM",
    "Miami Heat":             "MIA",
    "Milwaukee Bucks":        "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans":   "NOP",
    "New York Knicks":        "NYK",
    "Oklahoma City Thunder":  "OKC",
    "Orlando Magic":          "ORL",
    "Philadelphia 76ers":     "PHI",
    "Phoenix Suns":           "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings":       "SAC",
    "San Antonio Spurs":      "SAS",
    "Toronto Raptors":        "TOR",
    "Utah Jazz":              "UTA",
    "Washington Wizards":     "WAS",
}
