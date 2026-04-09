from __future__ import annotations

import json

from pm25_geopfnmix.data import load_dataset
from pm25_geopfnmix.settings import COUNTRY_COL, FIGURES_DIR, TABLES_DIR, TARGET_COL, ensure_directories
from pm25_geopfnmix.visualization import (
    save_feature_vs_target,
    save_group_size_distribution,
    save_numeric_correlation,
    save_target_distribution,
)


def main() -> None:
    ensure_directories()
    dataset = load_dataset()
    frame = dataset.frame

    summary = {
        "shape": list(frame.shape),
        "missing": frame.isna().sum().to_dict(),
        "duplicate_rows": int(frame.duplicated().sum()),
        "country_count": int(frame[COUNTRY_COL].nunique()),
        "province_count": int(frame["PROVINCE"].nunique()),
        "city_count": int(frame["CITY"].nunique()),
        "target_mean": float(frame[TARGET_COL].mean()),
        "target_std": float(frame[TARGET_COL].std()),
    }

    with open(TABLES_DIR / "eda_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    frame.describe().T.to_csv(TABLES_DIR / "numeric_describe.csv")
    save_target_distribution(frame, FIGURES_DIR / "pm25_distribution.png")
    save_numeric_correlation(frame, FIGURES_DIR / "numeric_correlation.png")
    save_group_size_distribution(frame, FIGURES_DIR / "city_group_size_distribution.png")
    save_feature_vs_target(frame, FIGURES_DIR / "key_features_vs_target.png")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
