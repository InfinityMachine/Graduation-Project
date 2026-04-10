from __future__ import annotations

import argparse
import json
import warnings

import pandas as pd

from pm25_geopfnmix.data import load_dataset
from pm25_geopfnmix.evaluation import compare_paired_errors, evaluate_cv, write_result_bundle
from pm25_geopfnmix.models import make_model
from pm25_geopfnmix.settings import TABLES_DIR, ensure_directories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GeoPFNMix ablation experiments.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "gpu", "AUTO", "CPU", "GPU"],
        help="Device preference for GPU-capable components.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device_preference = args.device.lower()
    warnings.filterwarnings("ignore")
    ensure_directories()
    dataset = load_dataset()
    print(f"[config] device_preference={device_preference}", flush=True)

    experiment_names = [
        "rf",
        "catboost",
        "tabpfn",
        "geopfnmix_no_prior",
        "geopfnmix_no_residual",
        "geopfnmix_lite_catboost",
        "geopfnmix_no_residual_tabpfn",
        "geopfnmix",
        "geopfnmix_tabpfn",
        "geopfnmix_catboost",
        "geopfnmix_catboost_tabpfn",
    ]

    results = {}
    summary_rows: list[dict[str, str | float]] = []

    for model_name in experiment_names:
        result = evaluate_cv(
            dataset=dataset,
            model_name=model_name,
            model_factory=lambda model_name=model_name: make_model(
                model_name,
                device_preference=device_preference,
            ),
            split_name="group_city",
        )
        results[model_name] = result
        write_result_bundle(result, str(TABLES_DIR / f"group_city_{model_name}"))
        summary_frame = result.summary.copy()
        summary_frame["model_name"] = model_name
        summary_rows.extend(summary_frame.to_dict(orient="records"))
        print(f"[done] {model_name}", flush=True)

    summary_table = pd.DataFrame(summary_rows)
    summary_table.to_csv(TABLES_DIR / "geopfnmix_ablation_summary.csv", index=False)

    comparisons = [
        ("rf", "geopfnmix_no_residual", "rf_vs_geopfnmix_lite"),
        ("rf", "geopfnmix", "rf_vs_geopfnmix_full"),
        ("geopfnmix_no_residual", "geopfnmix", "geopfnmix_lite_vs_full"),
        ("catboost", "geopfnmix_catboost", "catboost_vs_geopfnmix_catboost"),
        (
            "catboost",
            "geopfnmix_lite_catboost",
            "catboost_vs_geopfnmix_lite_catboost",
        ),
        (
            "geopfnmix_lite_catboost",
            "geopfnmix_catboost",
            "geopfnmix_lite_catboost_vs_geopfnmix_catboost",
        ),
        (
            "geopfnmix_no_residual",
            "geopfnmix_lite_catboost",
            "geopfnmix_lite_rf_vs_lite_catboost",
        ),
        (
            "catboost",
            "geopfnmix_catboost_tabpfn",
            "catboost_vs_geopfnmix_catboost_tabpfn",
        ),
        ("geopfnmix", "geopfnmix_catboost", "geopfnmix_full_vs_geopfnmix_catboost"),
        (
            "geopfnmix",
            "geopfnmix_tabpfn",
            "geopfnmix_full_lightgbm_vs_tabpfn",
        ),
        (
            "geopfnmix_no_residual_tabpfn",
            "geopfnmix_tabpfn",
            "geopfnmix_lite_tabpfn_vs_full_tabpfn",
        ),
        (
            "geopfnmix_no_residual",
            "geopfnmix_no_residual_tabpfn",
            "geopfnmix_lite_lightgbm_vs_tabpfn",
        ),
        (
            "geopfnmix_catboost",
            "geopfnmix_catboost_tabpfn",
            "geopfnmix_catboost_lightgbm_vs_tabpfn",
        ),
        (
            "tabpfn",
            "geopfnmix_no_residual_tabpfn",
            "tabpfn_vs_geopfnmix_lite_tabpfn",
        ),
        (
            "tabpfn",
            "geopfnmix_tabpfn",
            "tabpfn_vs_geopfnmix_full_tabpfn",
        ),
        (
            "tabpfn",
            "geopfnmix_catboost_tabpfn",
            "tabpfn_vs_geopfnmix_catboost_tabpfn",
        ),
    ]
    significance_rows: list[dict[str, str | float | int]] = []
    significance_payload: dict[str, dict[str, str | float | int]] = {}

    for reference_name, challenger_name, comparison_name in comparisons:
        stats = compare_paired_errors(results[reference_name], results[challenger_name])
        row = {
            "comparison": comparison_name,
            "reference_model": reference_name,
            "challenger_model": challenger_name,
            **stats,
        }
        significance_rows.append(row)
        significance_payload[comparison_name] = row

    with open(TABLES_DIR / "geopfnmix_significance.json", "w", encoding="utf-8") as handle:
        json.dump(significance_payload, handle, ensure_ascii=False, indent=2)
    pd.DataFrame(significance_rows).to_csv(TABLES_DIR / "geopfnmix_significance_pairs.csv", index=False)

    print(summary_table.to_string(index=False), flush=True)
    print(json.dumps(significance_payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
