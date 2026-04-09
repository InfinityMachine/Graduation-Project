from __future__ import annotations

import argparse
import json
import warnings

import pandas as pd

from pm25_geopfnmix.data import load_dataset
from pm25_geopfnmix.evaluation import evaluate_cv, write_result_bundle
from pm25_geopfnmix.models import make_model
from pm25_geopfnmix.settings import TABLES_DIR, ensure_directories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline experiments.")
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
    baseline_names = ["ridge", "rf", "hgbt", "xgboost", "lightgbm", "catboost"]
    split_names = ["random_kfold", "group_city", "group_province"]

    all_rows: list[dict[str, str | float]] = []
    skipped_models: list[dict[str, str]] = []

    for split_name in split_names:
        for model_name in baseline_names + ["tabpfn"]:
            try:
                result = evaluate_cv(
                    dataset=dataset,
                    model_name=model_name,
                    model_factory=lambda model_name=model_name: make_model(
                        model_name,
                        device_preference=device_preference,
                    ),
                    split_name=split_name,
                )
            except Exception as exc:
                skipped_models.append({"model_name": model_name, "split_name": split_name, "reason": str(exc)})
                print(f"[skip] {model_name} @ {split_name}: {exc}", flush=True)
                continue

            prefix = str(TABLES_DIR / f"{split_name}_{model_name}")
            write_result_bundle(result, prefix)
            summary_frame = result.summary.copy()
            summary_frame["model_name"] = model_name
            summary_frame["split_name"] = split_name
            all_rows.extend(summary_frame.to_dict(orient="records"))
            print(f"[done] {model_name} @ {split_name}", flush=True)

    summary_table = pd.DataFrame(all_rows)
    summary_table.to_csv(TABLES_DIR / "baseline_summary.csv", index=False)
    with open(TABLES_DIR / "baseline_skips.json", "w", encoding="utf-8") as handle:
        json.dump(skipped_models, handle, ensure_ascii=False, indent=2)
    print(summary_table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
