from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .settings import (
    CITY_COL,
    COUNTRY_COL,
    COUNTY_COL,
    DATA_DIR,
    ID_COL,
    LEGACY_DATA_PATH,
    NUM_COLS,
    PROVINCE_COL,
    TARGET_COL,
)


@dataclass(slots=True)
class DatasetBundle:
    frame: pd.DataFrame
    features: pd.DataFrame
    target: pd.Series


def load_dataset() -> DatasetBundle:
    frame = _load_raw_frame()
    frame = _standardize_multisource_frame(frame)
    features = frame.drop(columns=[TARGET_COL, ID_COL])
    target = frame[TARGET_COL].copy()
    return DatasetBundle(frame=frame, features=features, target=target)


def _load_raw_frame() -> pd.DataFrame:
    if DATA_DIR.exists():
        tables = [
            _load_australia(DATA_DIR / "Australia.csv"),
            _load_brazil(DATA_DIR / "Brazil.xlsx"),
            _load_china(DATA_DIR / "China.xlsx"),
            _load_eu(DATA_DIR / "EU.xlsx"),
            _load_usa(DATA_DIR / "USA.xlsx"),
        ]
        return pd.concat(tables, ignore_index=True)

    if LEGACY_DATA_PATH.exists():
        return pd.read_csv(LEGACY_DATA_PATH)

    raise FileNotFoundError("No supported dataset source was found. Expected data/ or data.csv.")


def _standardize_multisource_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required_cols = [
        ID_COL,
        COUNTRY_COL,
        PROVINCE_COL,
        CITY_COL,
        COUNTY_COL,
        *NUM_COLS,
        TARGET_COL,
    ]
    missing_cols = [col for col in required_cols if col not in frame.columns]
    if missing_cols:
        raise ValueError(f"Standardized dataset is missing required columns: {missing_cols}")

    out = frame[required_cols].copy()
    out[ID_COL] = out[ID_COL].astype(str)
    for col in [COUNTRY_COL, PROVINCE_COL, CITY_COL, COUNTY_COL]:
        out[col] = out[col].astype(str).str.strip().replace({"": "UNKNOWN"})

    for col in NUM_COLS + [TARGET_COL]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    return out


def _load_china(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path)
    rename_map = {
        "OBJECTID_1": ID_COL,
        "PM2.5": TARGET_COL,
        "AET": "AET",
        "ppt": "ppt",
        "tem": "tem",
        "wind": "wind",
        "NOX": "NOX",
        "SO2": "SO2",
        "fertilzier": "fertilzier",
        "manure": "manure",
    }
    out = frame.rename(columns=rename_map).copy()
    out[COUNTRY_COL] = "China"
    out[PROVINCE_COL] = "China::" + frame["PROVINCE"].astype(str)
    out[CITY_COL] = out[PROVINCE_COL] + "::" + frame["CITY"].astype(str)
    out[COUNTY_COL] = out[CITY_COL] + "::" + frame["COUNTY"].astype(str)
    return out


def _load_australia(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    geo_id = frame["ID"].astype(str).str.replace(".0", "", regex=False).str.zfill(9)
    out = pd.DataFrame(
        {
            ID_COL: "Australia::" + geo_id,
            COUNTRY_COL: "Australia",
            PROVINCE_COL: "Australia::" + geo_id.str[:1],
            CITY_COL: "Australia::" + geo_id.str[:1] + "::" + geo_id.str[:3],
            COUNTY_COL: "Australia::" + geo_id,
            "AET": frame["AET"],
            "ppt": frame["PPT"],
            "tem": frame["TEM"],
            "wind": frame["WIND"],
            "NOX": frame["NOX"],
            "SO2": frame["SO2density"],
            "fertilzier": frame["fertilizer"],
            "manure": frame["manure"],
            TARGET_COL: frame["PM25"],
        }
    )
    return out


def _load_brazil(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path)
    geo_id = frame.iloc[:, 0].astype(str).str.replace(".0", "", regex=False).str.zfill(7)
    out = pd.DataFrame(
        {
            ID_COL: "Brazil::" + geo_id,
            COUNTRY_COL: "Brazil",
            PROVINCE_COL: "Brazil::" + geo_id.str[:2],
            CITY_COL: "Brazil::" + geo_id.str[:2] + "::" + geo_id.str[:4],
            COUNTY_COL: "Brazil::" + geo_id,
            "AET": frame["AET"],
            "ppt": frame["ppt"],
            "tem": frame["tem"],
            "wind": frame["wind"],
            "NOX": frame["Nox"],
            "SO2": frame["SO2"],
            "fertilzier": frame["fertilizer N"],
            "manure": frame["manure N"],
            TARGET_COL: frame["pm25"],
        }
    )
    return out


def _load_eu(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path)
    geo_id = frame["VALUE"].astype(str)
    country_code = geo_id.str[:2]
    out = pd.DataFrame(
        {
            ID_COL: "EU::" + geo_id,
            COUNTRY_COL: "EU",
            PROVINCE_COL: "EU::" + country_code,
            CITY_COL: "EU::" + country_code + "::" + geo_id.str[:3],
            COUNTY_COL: "EU::" + geo_id,
            "AET": frame["AET"],
            "ppt": frame["PPT"],
            "tem": frame["TEM"],
            "wind": frame["WIND"],
            "NOX": frame["NO2density"],
            "SO2": frame["SO2density"],
            "fertilzier": frame["fertilizer"],
            "manure": frame["manureN"],
            TARGET_COL: frame["pm25"],
        }
    )
    return out


def _load_usa(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path)
    state = frame["State"].astype(str)
    county = frame["County"].astype(str)
    out = pd.DataFrame(
        {
            ID_COL: "USA::" + frame["OBJECTID"].astype(str),
            COUNTRY_COL: "USA",
            PROVINCE_COL: "USA::" + state,
            CITY_COL: "USA::" + state,
            COUNTY_COL: "USA::" + state + "::" + county,
            "AET": frame["AET"],
            "ppt": frame["PPT"],
            "tem": frame["TEM"],
            "wind": frame["WIND"],
            "NOX": frame["NOXdensity"],
            "SO2": frame["SO2density"],
            "fertilzier": frame["FERTILIZER"],
            "manure": frame["manure "],
            TARGET_COL: frame["pm25"],
        }
    )
    return out
