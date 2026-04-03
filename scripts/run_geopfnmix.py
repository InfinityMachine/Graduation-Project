from __future__ import annotations

import json
import warnings

import pandas as pd

from pm25_geopfnmix.data import load_dataset
from pm25_geopfnmix.evaluation import compare_paired_errors, evaluate_cv, write_result_bundle
from pm25_geopfnmix.models import make_model
from pm25_geopfnmix.settings import TABLES_DIR, ensure_directories


def main() -> None:
    warnings.filterwarnings("ignore")
    ensure_directories()
    dataset = load_dataset()

    experiment_names = [
        "rf",
        "catboost",
        "geopfnmix_no_prior",
        "geopfnmix_no_residual",
        "geopfnmix",
    ]

    results = {}
    summary_rows: list[dict[str, str | float]] = []

    for model_name in experiment_names:
        result = evaluate_cv(
            dataset=dataset,
            model_name=model_name,
            model_factory=lambda model_name=model_name: make_model(model_name),
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

    best_baseline = results["rf"]
    final_model = results["geopfnmix"]
    significance = compare_paired_errors(best_baseline, final_model)

    with open(TABLES_DIR / "geopfnmix_significance.json", "w", encoding="utf-8") as handle:
        json.dump(significance, handle, ensure_ascii=False, indent=2)

    print(summary_table.to_string(index=False), flush=True)
    print(json.dumps(significance, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
