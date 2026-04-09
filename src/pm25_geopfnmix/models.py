from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBRegressor

from .auth import has_tabpfn_headless_access, prime_auth_tokens, resolve_hf_token
from .features import FeatureEngineer
from .settings import CAT_COLS, CITY_COL, PROVINCE_COL, RANDOM_STATE

try:
    from tabpfn import TabPFNRegressor
except Exception:  # pragma: no cover - optional dependency path
    TabPFNRegressor = None  # type: ignore[assignment]

try:
    import torch
except Exception:  # pragma: no cover - optional dependency path
    torch = None  # type: ignore[assignment]


class BaseRawRegressor:
    def fit(self, frame: pd.DataFrame, target: pd.Series) -> "BaseRawRegressor":
        raise NotImplementedError

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


def _split_feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    cat_cols = [col for col in CAT_COLS if col in frame.columns]
    num_cols = [col for col in frame.columns if col not in cat_cols]
    return cat_cols, num_cols


class SklearnTableRegressor(BaseRawRegressor):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.feature_engineer = FeatureEngineer()
        self.pipeline: Pipeline | None = None
        self.prediction_clip_: tuple[float, float] | None = None

    def fit(self, frame: pd.DataFrame, target: pd.Series) -> "SklearnTableRegressor":
        engineered = self.feature_engineer.fit_transform(frame)
        cat_cols, num_cols = _split_feature_columns(engineered)

        if self.model_name == "ridge":
            preprocessor = ColumnTransformer(
                [
                    (
                        "num",
                        Pipeline(
                            [
                                ("imputer", SimpleImputer(strategy="median")),
                                ("scaler", StandardScaler()),
                            ]
                        ),
                        num_cols,
                    ),
                ]
            )
            estimator = Ridge(alpha=30.0)
        elif self.model_name == "rf":
            preprocessor = ColumnTransformer(
                [
                    ("num", SimpleImputer(strategy="median"), num_cols),
                    (
                        "cat",
                        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                        cat_cols,
                    ),
                ]
            )
            estimator = RandomForestRegressor(
                n_estimators=600,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        elif self.model_name == "hgbt":
            preprocessor = ColumnTransformer(
                [
                    ("num", SimpleImputer(strategy="median"), num_cols),
                    (
                        "cat",
                        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                        cat_cols,
                    ),
                ]
            )
            estimator = HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_depth=6,
                max_iter=500,
                random_state=RANDOM_STATE,
            )
        elif self.model_name == "lightgbm":
            preprocessor = ColumnTransformer(
                [
                    ("num", SimpleImputer(strategy="median"), num_cols),
                    (
                        "cat",
                        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                        cat_cols,
                    ),
                ]
            )
            estimator = LGBMRegressor(
                n_estimators=800,
                learning_rate=0.03,
                num_leaves=63,
                subsample=0.9,
                colsample_bytree=0.9,
                min_child_samples=20,
                random_state=RANDOM_STATE,
                verbose=-1,
            )
        elif self.model_name == "xgboost":
            preprocessor = ColumnTransformer(
                [
                    ("num", SimpleImputer(strategy="median"), num_cols),
                    (
                        "cat",
                        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                        cat_cols,
                    ),
                ]
            )
            estimator = XGBRegressor(
                n_estimators=800,
                learning_rate=0.03,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                objective="reg:squarederror",
                tree_method="hist",
                n_jobs=12,
                random_state=RANDOM_STATE,
            )
        else:
            raise ValueError(f"Unsupported sklearn model: {self.model_name}")

        self.pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        self.pipeline.fit(engineered, target)
        if self.model_name == "ridge":
            self.prediction_clip_ = (
                float(target.quantile(0.01)),
                float(target.quantile(0.99)),
            )
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Model has not been fitted.")
        engineered = self.feature_engineer.transform(frame)
        predictions = np.asarray(self.pipeline.predict(engineered), dtype=float)
        if self.model_name == "ridge" and self.prediction_clip_ is not None:
            low, high = self.prediction_clip_
            predictions = np.clip(predictions, low, high)
        return predictions


class CatBoostTableRegressor(BaseRawRegressor):
    def __init__(self) -> None:
        self.feature_engineer = FeatureEngineer()
        self.model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=900,
            depth=6,
            learning_rate=0.03,
            l2_leaf_reg=6.0,
            random_seed=RANDOM_STATE,
            allow_writing_files=False,
            verbose=False,
        )

    def fit(self, frame: pd.DataFrame, target: pd.Series) -> "CatBoostTableRegressor":
        engineered = self.feature_engineer.fit_transform(frame)
        self.model.fit(engineered, target, cat_features=CAT_COLS, verbose=False)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        engineered = self.feature_engineer.transform(frame)
        return np.asarray(self.model.predict(engineered), dtype=float)


class OptionalTabPFNRegressorWrapper(BaseRawRegressor):
    def __init__(self) -> None:
        if TabPFNRegressor is None:
            raise RuntimeError("tabpfn is not importable in the current environment.")
        prime_auth_tokens()
        if not resolve_hf_token():
            raise RuntimeError("HF_TOKEN is required for TabPFN model download.")
        if not has_tabpfn_headless_access():
            raise RuntimeError(
                "TabPFN headless mode requires a Prior Labs access token via TABPFN_TOKEN "
                "or a cached accepted-license token."
            )

        self.feature_engineer = FeatureEngineer()
        self.preprocessor: ColumnTransformer | None = None
        self.model: TabPFNRegressor | None = None
        self.device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

    def fit(self, frame: pd.DataFrame, target: pd.Series) -> "OptionalTabPFNRegressorWrapper":
        engineered = self.feature_engineer.fit_transform(frame)
        cat_cols, num_cols = _split_feature_columns(engineered)
        self.preprocessor = ColumnTransformer(
            [
                ("num", SimpleImputer(strategy="median"), num_cols),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "encoder",
                                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                            ),
                        ]
                    ),
                    cat_cols,
                ),
            ],
            verbose_feature_names_out=False,
        )
        train_array = self.preprocessor.fit_transform(engineered, target)
        categorical_indices = list(range(len(num_cols), len(num_cols) + len(cat_cols)))
        self.model = TabPFNRegressor(
            device=self.device,
            random_state=RANDOM_STATE,
            n_estimators=4,
            fit_mode="fit_preprocessors",
            ignore_pretraining_limits=True,
            categorical_features_indices=categorical_indices,
        )
        self.model.fit(train_array, target)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.preprocessor is None or self.model is None:
            raise RuntimeError("TabPFN wrapper has not been fitted.")
        engineered = self.feature_engineer.transform(frame)
        test_array = self.preprocessor.transform(engineered)
        return np.asarray(self.model.predict(test_array), dtype=float)


@dataclass
class HierarchicalResidualCalibrator:
    alpha_province: float = 15.0
    alpha_city: float = 8.0

    def fit(
        self,
        frame: pd.DataFrame,
        residuals: np.ndarray,
    ) -> "HierarchicalResidualCalibrator":
        residual_frame = frame[[PROVINCE_COL, CITY_COL]].copy()
        residual_frame["residual"] = np.asarray(residuals, dtype=float)
        self.global_bias_ = float(residual_frame["residual"].mean())

        province_stats = residual_frame.groupby(PROVINCE_COL)["residual"].agg(["mean", "count"])
        province_stats["effect"] = (
            province_stats["count"] / (province_stats["count"] + self.alpha_province)
        ) * (province_stats["mean"] - self.global_bias_)
        self.province_effect_ = province_stats["effect"].to_dict()
        self.province_mean_ = province_stats["mean"].to_dict()

        city_stats = (
            residual_frame.groupby([PROVINCE_COL, CITY_COL])["residual"].agg(["mean", "count"]).reset_index()
        )
        city_effect: dict[tuple[str, str], float] = {}
        for row in city_stats.itertuples(index=False):
            parent_mean = self.province_mean_[row.PROVINCE]
            city_effect[(row.PROVINCE, row.CITY)] = (
                row.count / (row.count + self.alpha_city)
            ) * (row.mean - parent_mean)
        self.city_effect_ = city_effect
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        province_term = frame[PROVINCE_COL].map(self.province_effect_).fillna(0.0).to_numpy(dtype=float)
        city_term = np.array(
            [self.city_effect_.get((prov, city), 0.0) for prov, city in zip(frame[PROVINCE_COL], frame[CITY_COL])],
            dtype=float,
        )
        return self.global_bias_ + province_term + city_term


class GeoPFNMixRegressor(BaseRawRegressor):
    def __init__(
        self,
        *,
        global_model_name: str = "rf",
        prior_model_name: str = "lightgbm",
        use_residual_branch: bool = True,
        use_prior_branch: bool = True,
        inner_splits: int = 3,
    ) -> None:
        self.global_model_name = global_model_name
        self.default_prior_model_name = prior_model_name
        self.use_residual_branch = use_residual_branch
        self.use_prior_branch = use_prior_branch
        self.inner_splits = inner_splits
        self.global_model = self._make_global_model()
        self.prior_model_name: str | None = None
        self.prior_model: BaseRawRegressor | None = None
        self.residual_calibrator: HierarchicalResidualCalibrator | None = None
        self.meta_model: Pipeline | None = None
        self.province_count_: dict[str, int] = {}
        self.city_count_: dict[str, int] = {}

    def fit(self, frame: pd.DataFrame, target: pd.Series) -> "GeoPFNMixRegressor":
        self.province_count_ = frame[PROVINCE_COL].value_counts().to_dict()
        self.city_count_ = frame[CITY_COL].value_counts().to_dict()

        self.global_model.fit(frame, target)
        global_train_pred = self.global_model.predict(frame)

        if self.use_residual_branch:
            self.residual_calibrator = HierarchicalResidualCalibrator().fit(frame, target.to_numpy() - global_train_pred)

        if self.use_prior_branch:
            self.prior_model = self._make_prior_model()
            try:
                self.prior_model.fit(frame, target)
            except Exception:
                if self.prior_model_name == "tabpfn":
                    self.prior_model_name = "lightgbm"
                    self.prior_model = SklearnTableRegressor("lightgbm")
                    self.prior_model.fit(frame, target)
                else:
                    raise

        if not self.use_prior_branch:
            return self

        splitter = GroupKFold(n_splits=self.inner_splits)
        meta_rows: list[pd.DataFrame] = []
        meta_target: list[pd.Series] = []
        groups = frame[CITY_COL]

        for inner_train_idx, inner_valid_idx in splitter.split(frame, target, groups):
            train_frame = frame.iloc[inner_train_idx].reset_index(drop=True)
            valid_frame = frame.iloc[inner_valid_idx].reset_index(drop=True)
            train_target = target.iloc[inner_train_idx].reset_index(drop=True)
            valid_target = target.iloc[inner_valid_idx].reset_index(drop=True)

            fold_global = self._make_global_model().fit(train_frame, train_target)
            fold_global_pred_train = fold_global.predict(train_frame)
            fold_global_pred_valid = fold_global.predict(valid_frame)

            fold_geo_shift = np.zeros(len(valid_frame), dtype=float)
            if self.use_residual_branch:
                fold_residual = HierarchicalResidualCalibrator().fit(
                    train_frame,
                    train_target.to_numpy() - fold_global_pred_train,
                )
                fold_geo_shift = fold_residual.predict(valid_frame)

            fold_prior = self._make_prior_model()
            try:
                fold_prior.fit(train_frame, train_target)
            except Exception:
                if self.prior_model_name == "tabpfn":
                    self.prior_model_name = "lightgbm"
                    fold_prior = SklearnTableRegressor("lightgbm")
                    fold_prior.fit(train_frame, train_target)
                else:
                    raise
            fold_prior_pred_valid = fold_prior.predict(valid_frame)

            meta_rows.append(
                self._build_meta_features(
                    frame=valid_frame,
                    global_pred=fold_global_pred_valid,
                    geo_shift=fold_geo_shift,
                    prior_pred=fold_prior_pred_valid,
                    train_frame=train_frame,
                )
            )
            meta_target.append(valid_target)

        meta_frame = pd.concat(meta_rows, ignore_index=True)
        meta_y = pd.concat(meta_target, ignore_index=True)
        self.meta_model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=3.0),
        )
        self.meta_model.fit(meta_frame, meta_y)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        global_pred = self.global_model.predict(frame)
        if not self.use_prior_branch:
            if self.use_residual_branch and self.residual_calibrator is not None:
                return global_pred + self.residual_calibrator.predict(frame)
            return global_pred

        if self.prior_model is None or self.meta_model is None:
            raise RuntimeError("GeoPFNMix was not fitted correctly.")

        geo_shift = np.zeros(len(frame), dtype=float)
        if self.use_residual_branch and self.residual_calibrator is not None:
            geo_shift = self.residual_calibrator.predict(frame)

        prior_pred = self.prior_model.predict(frame)
        meta_frame = self._build_meta_features(
            frame=frame,
            global_pred=global_pred,
            geo_shift=geo_shift,
            prior_pred=prior_pred,
            train_frame=None,
        )
        return np.asarray(self.meta_model.predict(meta_frame), dtype=float)

    def _build_meta_features(
        self,
        *,
        frame: pd.DataFrame,
        global_pred: np.ndarray,
        geo_shift: np.ndarray,
        prior_pred: np.ndarray,
        train_frame: pd.DataFrame | None,
    ) -> pd.DataFrame:
        if train_frame is not None:
            province_counts = train_frame[PROVINCE_COL].value_counts().to_dict()
            city_counts = train_frame[CITY_COL].value_counts().to_dict()
        else:
            province_counts = self.province_count_
            city_counts = self.city_count_

        province_support = frame[PROVINCE_COL].map(province_counts).fillna(0.0).astype(float)
        city_support = frame[CITY_COL].map(city_counts).fillna(0.0).astype(float)
        calibrated_pred = global_pred + geo_shift

        return pd.DataFrame(
            {
                "pred_global": global_pred,
                "pred_geo": calibrated_pred,
                "pred_prior": prior_pred,
                "geo_shift": geo_shift,
                "expert_gap": np.abs(global_pred - prior_pred),
                "province_support": province_support,
                "city_support": city_support,
                "low_support_city": (city_support <= 3).astype(float),
            }
        )

    def _make_prior_model(self) -> BaseRawRegressor:
        if self.prior_model_name is not None:
            chosen = self.prior_model_name
        else:
            chosen = self.default_prior_model_name
            if chosen == "tabpfn" and not has_tabpfn_headless_access():
                chosen = "lightgbm"
            self.prior_model_name = chosen

        if chosen == "tabpfn":
            try:
                return OptionalTabPFNRegressorWrapper()
            except RuntimeError:
                self.prior_model_name = "lightgbm"
                return SklearnTableRegressor("lightgbm")
        if chosen == "lightgbm":
            return SklearnTableRegressor("lightgbm")
        raise ValueError(f"Unsupported prior model: {chosen}")

    def _make_global_model(self) -> BaseRawRegressor:
        if self.global_model_name == "catboost":
            return CatBoostTableRegressor()
        if self.global_model_name in {"rf", "lightgbm", "xgboost", "hgbt", "ridge"}:
            return SklearnTableRegressor(self.global_model_name)
        raise ValueError(f"Unsupported global model: {self.global_model_name}")


def make_model(model_name: str) -> BaseRawRegressor:
    if model_name == "catboost":
        return CatBoostTableRegressor()
    if model_name in {"ridge", "rf", "hgbt", "lightgbm", "xgboost"}:
        return SklearnTableRegressor(model_name)
    if model_name == "geopfnmix":
        return GeoPFNMixRegressor(prior_model_name="lightgbm")
    if model_name == "geopfnmix_catboost":
        return GeoPFNMixRegressor(global_model_name="catboost", prior_model_name="lightgbm")
    if model_name == "geopfnmix_no_residual":
        return GeoPFNMixRegressor(
            prior_model_name="lightgbm",
            use_residual_branch=False,
            use_prior_branch=True,
        )
    if model_name == "geopfnmix_tabpfn":
        return GeoPFNMixRegressor(prior_model_name="tabpfn")
    if model_name == "geopfnmix_catboost_tabpfn":
        return GeoPFNMixRegressor(global_model_name="catboost", prior_model_name="tabpfn")
    if model_name == "geopfnmix_no_residual_tabpfn":
        return GeoPFNMixRegressor(
            prior_model_name="tabpfn",
            use_residual_branch=False,
            use_prior_branch=True,
        )
    if model_name == "geopfnmix_no_prior":
        return GeoPFNMixRegressor(use_residual_branch=True, use_prior_branch=False)
    if model_name == "tabpfn":
        return OptionalTabPFNRegressorWrapper()
    raise ValueError(f"Unknown model name: {model_name}")
