from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut

from .data import DatasetBundle
from .models import BaseRawRegressor
from .settings import COUNTRY_COL, GROUP_COL, PROVINCE_COL, RANDOM_STATE


def compute_metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true_array, y_pred_array))),
        "mae": float(mean_absolute_error(y_true_array, y_pred_array)),
        "r2": float(r2_score(y_true_array, y_pred_array)),
    }


@dataclass
class CVResult:
    model_name: str
    split_name: str
    fold_metrics: pd.DataFrame
    oof_frame: pd.DataFrame

    @property
    def summary(self) -> pd.DataFrame:
        summary = self.fold_metrics[["rmse", "mae", "r2"]].agg(["mean", "std"]).T.reset_index()
        return summary.rename(columns={"index": "metric"})


def evaluate_cv(
    *,
    dataset: DatasetBundle,
    model_name: str,
    model_factory: callable,
    split_name: str,
    n_splits: int = 5,
) -> CVResult:
    features = dataset.features.reset_index(drop=True)
    target = dataset.target.reset_index(drop=True)
    frame = dataset.frame.reset_index(drop=True)

    if split_name == "random_kfold":
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        split_iter = splitter.split(features, target)
    elif split_name == "group_city":
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(features, target, frame[GROUP_COL])
    elif split_name == "group_province":
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(features, target, frame[PROVINCE_COL])
    elif split_name == "group_country":
        splitter = LeaveOneGroupOut()
        split_iter = splitter.split(features, target, frame[COUNTRY_COL])
    else:
        raise ValueError(f"Unsupported split name: {split_name}")

    fold_rows: list[dict[str, float | int]] = []
    oof_rows: list[pd.DataFrame] = []

    for fold_id, (train_idx, test_idx) in enumerate(split_iter, start=1):
        train_features = features.iloc[train_idx].reset_index(drop=True)
        test_features = features.iloc[test_idx].reset_index(drop=True)
        train_target = target.iloc[train_idx].reset_index(drop=True)
        test_target = target.iloc[test_idx].reset_index(drop=True)

        model: BaseRawRegressor = model_factory()
        model.fit(train_features, train_target)
        fold_pred = model.predict(test_features)

        metrics = compute_metrics(test_target, fold_pred)
        metrics["fold"] = fold_id
        if split_name == "group_country":
            metrics["holdout_country"] = str(frame.iloc[test_idx][COUNTRY_COL].iloc[0])
        fold_rows.append(metrics)

        oof_payload = {
            "row_id": test_idx,
            "fold": fold_id,
            "y_true": test_target.to_numpy(),
            "y_pred": fold_pred,
            "abs_error": np.abs(test_target.to_numpy() - fold_pred),
        }
        if split_name == "group_country":
            oof_payload["holdout_country"] = frame.iloc[test_idx][COUNTRY_COL].astype(str).to_numpy()

        oof_rows.append(pd.DataFrame(oof_payload))

    fold_metrics = pd.DataFrame(fold_rows)
    oof_frame = pd.concat(oof_rows, ignore_index=True).sort_values("row_id").reset_index(drop=True)
    return CVResult(model_name=model_name, split_name=split_name, fold_metrics=fold_metrics, oof_frame=oof_frame)


def write_result_bundle(result: CVResult, output_prefix: str) -> None:
    fold_path = f"{output_prefix}_folds.csv"
    oof_path = f"{output_prefix}_oof.csv"
    summary_path = f"{output_prefix}_summary.json"

    result.fold_metrics.to_csv(fold_path, index=False)
    result.oof_frame.to_csv(oof_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_name": result.model_name,
                "split_name": result.split_name,
                "summary": result.summary.to_dict(orient="records"),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )


def compare_paired_errors(reference: CVResult, challenger: CVResult) -> dict[str, float]:
    merged = reference.oof_frame.merge(
        challenger.oof_frame,
        on="row_id",
        suffixes=("_ref", "_challenger"),
        how="inner",
    )
    statistic, p_value = wilcoxon(merged["abs_error_ref"], merged["abs_error_challenger"], zero_method="wilcox")
    return {
        "wilcoxon_statistic": float(statistic),
        "p_value": float(p_value),
        "n": int(len(merged)),
    }
