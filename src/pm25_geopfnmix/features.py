from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .settings import CAT_COLS, CITY_COL, NUM_COLS, PROVINCE_COL


@dataclass(slots=True)
class FeatureConfig:
    clip_quantiles: tuple[float, float] = (0.01, 0.99)
    log_cols: tuple[str, ...] = ("ppt", "NOX", "SO2", "fertilzier", "manure")
    province_zscore_cols: tuple[str, ...] = ("ppt", "NOX", "SO2", "fertilzier", "manure")
    interaction_pairs: tuple[tuple[str, str, str], ...] = (
        ("fertilzier", "wind", "fertilzier_x_wind"),
        ("manure", "ppt", "manure_x_ppt"),
        ("NOX", "tem", "NOX_x_tem"),
        ("SO2", "wind", "SO2_x_wind"),
        ("AET", "ppt", "AET_x_ppt"),
    )


@dataclass
class FeatureEngineer:
    config: FeatureConfig = field(default_factory=FeatureConfig)

    def __post_init__(self) -> None:
        self.clip_bounds_: dict[str, tuple[float, float]] = {}
        self.province_stats_: dict[str, pd.DataFrame] = {}
        self.global_stats_: dict[str, tuple[float, float]] = {}
        self.province_count_: dict[str, int] = {}
        self.city_count_: dict[str, int] = {}

    def fit(self, frame: pd.DataFrame) -> "FeatureEngineer":
        lower_q, upper_q = self.config.clip_quantiles
        for col in NUM_COLS:
            series = frame[col].astype(float)
            self.clip_bounds_[col] = (
                float(series.quantile(lower_q)),
                float(series.quantile(upper_q)),
            )

        clipped = self._apply_clip(frame.copy())
        self.province_count_ = frame[PROVINCE_COL].value_counts().to_dict()
        self.city_count_ = frame[CITY_COL].value_counts().to_dict()

        for col in self.config.province_zscore_cols:
            stats = clipped.groupby(PROVINCE_COL)[col].agg(["mean", "std"])
            stats["std"] = stats["std"].replace(0.0, 1.0).fillna(1.0)
            self.province_stats_[col] = stats
            self.global_stats_[col] = (
                float(clipped[col].mean()),
                float(clipped[col].std() or 1.0),
            )
        return self

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = self._apply_clip(frame.copy())
        present_cat_cols = [col for col in CAT_COLS if col in out.columns]
        out = out[present_cat_cols + NUM_COLS].copy()

        for col in self.config.log_cols:
            out[f"log1p_{col}"] = np.log1p(np.clip(out[col], a_min=0.0, a_max=None))

        out["NOX_plus_SO2"] = out["NOX"] + out["SO2"]
        out["NOX_div_SO2"] = out["NOX"] / (out["SO2"] + 1.0)
        out["fertilzier_plus_manure"] = out["fertilzier"] + out["manure"]
        out["agri_pressure_logsum"] = out["log1p_fertilzier"] + out["log1p_manure"]
        out["emission_pressure_logsum"] = out["log1p_NOX"] + out["log1p_SO2"]

        for left, right, new_name in self.config.interaction_pairs:
            out[new_name] = out[left] * out[right]

        for col in self.config.province_zscore_cols:
            stats = self.province_stats_[col]
            global_mean, global_std = self.global_stats_[col]
            province_mean = out[PROVINCE_COL].map(stats["mean"]).fillna(global_mean)
            province_std = out[PROVINCE_COL].map(stats["std"]).fillna(global_std)
            out[f"province_z_{col}"] = (out[col] - province_mean) / province_std

        out["province_sample_count"] = out[PROVINCE_COL].map(self.province_count_).fillna(0).astype(float)
        out["city_sample_count"] = out[CITY_COL].map(self.city_count_).fillna(0).astype(float)
        out["is_low_support_city"] = (out["city_sample_count"] <= 3).astype(float)
        return out

    def _apply_clip(self, frame: pd.DataFrame) -> pd.DataFrame:
        for col in NUM_COLS:
            lower, upper = self.clip_bounds_.get(col, (-np.inf, np.inf))
            frame[col] = frame[col].astype(float).clip(lower=lower, upper=upper)
        return frame
