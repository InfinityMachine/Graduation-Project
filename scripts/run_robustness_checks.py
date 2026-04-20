from __future__ import annotations

import argparse
import warnings

import pandas as pd

from pm25_geopfnmix.data import DatasetBundle, load_dataset
from pm25_geopfnmix.evaluation import compare_paired_errors, evaluate_cv, write_result_bundle
from pm25_geopfnmix.models import make_model
from pm25_geopfnmix.settings import COUNTRY_COL, TABLES_DIR, ensure_directories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run supplementary robustness checks for cross-source generalization."
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "gpu", "AUTO", "CPU", "GPU"],
        help="Device preference for GPU-capable components.",
    )
    parser.add_argument(
        "--include-tabpfn",
        action="store_true",
        help="Also evaluate standalone TabPFN in the robustness checks.",
    )
    return parser.parse_args()


def _build_dataset_without_country(dataset: DatasetBundle) -> DatasetBundle:
    return DatasetBundle(
        frame=dataset.frame.copy(),
        features=dataset.features.drop(columns=[COUNTRY_COL], errors="ignore").copy(),
        target=dataset.target.copy(),
    )


def main() -> None:
    args = parse_args()
    device_preference = args.device.lower()
    warnings.filterwarnings("ignore")
    ensure_directories()

    dataset = load_dataset()
    dataset_no_country = _build_dataset_without_country(dataset)

    core_models = [
        "catboost",
        "geopfnmix_lite_catboost",
        "geopfnmix_lite_catboost_tabpfn",
        "geopfnmix_catboost",
        "geopfnmix_catboost_tabpfn",
    ]
    if args.include_tabpfn:
        core_models.insert(1, "tabpfn")

    print(f"[config] device_preference={device_preference}", flush=True)
    print(f"[config] models={core_models}", flush=True)

    country_holdout_rows: list[dict[str, str | float | int]] = []
    country_holdout_summary_rows: list[dict[str, str | float]] = []
    full_feature_results = {}
    no_country_results = {}
    feature_ablation_rows: list[dict[str, str | float | int]] = []
    feature_ablation_summary_rows: list[dict[str, str | float]] = []

    for model_name in core_models:
        country_holdout_result = evaluate_cv(
            dataset=dataset,
            model_name=model_name,
            model_factory=lambda model_name=model_name: make_model(
                model_name,
                device_preference=device_preference,
            ),
            split_name="group_country",
        )
        write_result_bundle(country_holdout_result, str(TABLES_DIR / f"group_country_{model_name}"))
        fold_frame = country_holdout_result.fold_metrics.copy()
        fold_frame["model_name"] = model_name
        country_holdout_rows.extend(fold_frame.to_dict(orient="records"))

        summary_frame = country_holdout_result.summary.copy()
        summary_frame["model_name"] = model_name
        country_holdout_summary_rows.extend(summary_frame.to_dict(orient="records"))
        print(f"[done] country_holdout::{model_name}", flush=True)

        full_result = evaluate_cv(
            dataset=dataset,
            model_name=model_name,
            model_factory=lambda model_name=model_name: make_model(
                model_name,
                device_preference=device_preference,
            ),
            split_name="group_city",
        )
        no_country_result = evaluate_cv(
            dataset=dataset_no_country,
            model_name=f"{model_name}_no_country",
            model_factory=lambda model_name=model_name: make_model(
                model_name,
                device_preference=device_preference,
            ),
            split_name="group_city",
        )
        full_feature_results[model_name] = full_result
        no_country_results[model_name] = no_country_result

        write_result_bundle(full_result, str(TABLES_DIR / f"group_city_with_country_{model_name}"))
        write_result_bundle(no_country_result, str(TABLES_DIR / f"group_city_no_country_{model_name}"))

        full_summary = full_result.summary.set_index("metric")["mean"]
        no_country_summary = no_country_result.summary.set_index("metric")["mean"]
        paired_stats = compare_paired_errors(full_result, no_country_result)

        feature_ablation_rows.append(
            {
                "model_name": model_name,
                "rmse_with_country": float(full_summary["rmse"]),
                "rmse_without_country": float(no_country_summary["rmse"]),
                "rmse_delta_without_minus_with": float(no_country_summary["rmse"] - full_summary["rmse"]),
                "mae_with_country": float(full_summary["mae"]),
                "mae_without_country": float(no_country_summary["mae"]),
                "mae_delta_without_minus_with": float(no_country_summary["mae"] - full_summary["mae"]),
                "r2_with_country": float(full_summary["r2"]),
                "r2_without_country": float(no_country_summary["r2"]),
                "r2_delta_without_minus_with": float(no_country_summary["r2"] - full_summary["r2"]),
                **paired_stats,
            }
        )

        for variant_name, result in [
            ("with_country", full_result),
            ("without_country", no_country_result),
        ]:
            summary_frame = result.summary.copy()
            summary_frame["model_name"] = model_name
            summary_frame["feature_variant"] = variant_name
            feature_ablation_summary_rows.extend(summary_frame.to_dict(orient="records"))
        print(f"[done] drop_country::{model_name}", flush=True)

    pd.DataFrame(country_holdout_rows).to_csv(TABLES_DIR / "country_holdout_fold_metrics.csv", index=False)
    pd.DataFrame(country_holdout_summary_rows).to_csv(TABLES_DIR / "country_holdout_summary.csv", index=False)
    pd.DataFrame(feature_ablation_rows).to_csv(TABLES_DIR / "drop_country_feature_pairs.csv", index=False)
    pd.DataFrame(feature_ablation_summary_rows).to_csv(TABLES_DIR / "drop_country_feature_summary.csv", index=False)


if __name__ == "__main__":
    main()
