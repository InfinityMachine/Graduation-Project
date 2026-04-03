from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data.csv"

ARTIFACTS_DIR = ROOT / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
TABLES_DIR = ARTIFACTS_DIR / "tables"
MODELS_DIR = ARTIFACTS_DIR / "models"
TMP_DIR = ARTIFACTS_DIR / "tmp"

REPORTS_DIR = ROOT / "reports"
PAPER_DIR = ROOT / "paper"

TARGET_COL = "PM2.5"
ID_COL = "OBJECTID_1"
GROUP_COL = "CITY"
PROVINCE_COL = "PROVINCE"
CITY_COL = "CITY"
COUNTY_COL = "COUNTY"
CAT_COLS = [PROVINCE_COL, CITY_COL]
NUM_COLS = ["AET", "ppt", "tem", "wind", "NOX", "SO2", "fertilzier", "manure"]

RANDOM_STATE = 42


def ensure_directories() -> None:
    for path in [
        ARTIFACTS_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        MODELS_DIR,
        TMP_DIR,
        REPORTS_DIR,
        PAPER_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
