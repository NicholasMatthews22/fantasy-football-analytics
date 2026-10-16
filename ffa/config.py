"""Project-wide settings."""
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DB_PATH = DATA_DIR / "ff.db"
MODEL_DIR = ROOT / "models"
METRICS_PATH = DATA_DIR / "metrics.json"

# Seasons used for training. The NFL season starts in September,
# so before September we're still in the offseason of the prior year.
FIRST_SEASON = 2021
_today = date.today()
CURRENT_SEASON = _today.year if _today.month >= 9 else _today.year - 1

POSITIONS = ["QB", "RB", "WR", "TE"]
POS_CODE = {p: i for i, p in enumerate(POSITIONS)}

# Exponential weighting for player form: a game 4 games ago counts half as much as the latest one.
PLAYER_HALFLIFE = 4
# Defenses change more slowly, so we smooth them over a longer window.
DEFENSE_HALFLIFE = 6

# Most fantasy leagues end in Week 17.
FANTASY_LAST_WEEK = 17
