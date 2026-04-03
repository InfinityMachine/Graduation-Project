from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap

from pm25_geopfnmix.data import load_dataset
from pm25_geopfnmix.models import make_model
from pm25_geopfnmix.settings import FIGURES_DIR, TABLES_DIR, ensure_directories

sns.set_theme(style="whitegrid", context="talk")


def save_model_comparison(summary: pd.DataFrame) -> None:
    plot_frame = summary[summary["metric"] == "rmse"].copy().sort_values("mean")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=plot_frame, x="mean", y="model_name", palette="crest", ax=ax)
    ax.set_xlabel("RMSE")
    ax.set_ylabel("")
    ax.set_title("Group-City RMSE Comparison")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "group_city_model_comparison.png", dpi=200)
    plt.close(fig)


def save_prediction_scatter(frame: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.scatterplot(
        data=frame,
        x="y_true",
        y="y_pred",
        s=35,
        alpha=0.7,
        color="#346b78",
        ax=ax,
    )
    line_min = float(min(frame["y_true"].min(), frame["y_pred"].min()))
    line_max = float(max(frame["y_true"].max(), frame["y_pred"].max()))
    ax.plot([line_min, line_max], [line_min, line_max], linestyle="--", color="#aa4c4c", linewidth=2)
    ax.set_title("GeoPFNMix-Lite OOF Predictions")
    ax.set_xlabel("Observed PM2.5")
    ax.set_ylabel("Predicted PM2.5")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "geopfnmix_lite_scatter.png", dpi=200)
    plt.close(fig)


def save_province_error(frame: pd.DataFrame) -> None:
    plot_frame = (
        frame.groupby("PROVINCE", as_index=False)["abs_error"]
        .mean()
        .sort_values("abs_error", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(10, 11))
    sns.barplot(data=plot_frame, x="abs_error", y="PROVINCE", palette="mako", ax=ax)
    ax.set_xlabel("Mean Absolute Error")
    ax.set_ylabel("")
    ax.set_title("Province-Level Error of GeoPFNMix-Lite")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "province_level_mae.png", dpi=200)
    plt.close(fig)
    plot_frame.to_csv(TABLES_DIR / "province_level_mae.csv", index=False)


def save_shap_assets() -> None:
    dataset = load_dataset()
    model = make_model("rf")
    model.fit(dataset.features, dataset.target)

    engineered = model.feature_engineer.transform(dataset.features)  # type: ignore[attr-defined]
    transformed = model.pipeline.named_steps["preprocessor"].transform(engineered)  # type: ignore[attr-defined]
    regressor = model.pipeline.named_steps["model"]  # type: ignore[attr-defined]
    feature_names = model.pipeline.named_steps["preprocessor"].get_feature_names_out()  # type: ignore[attr-defined]

    sample_size = min(128, transformed.shape[0])
    transformed_sample = transformed[:sample_size]

    explainer = shap.TreeExplainer(regressor)
    shap_values = explainer.shap_values(transformed_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_frame = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": pd.DataFrame(shap_values, columns=feature_names).abs().mean().values,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    shap_frame.to_csv(TABLES_DIR / "rf_global_shap_importance.csv", index=False)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        transformed_sample,
        feature_names=feature_names,
        plot_type="bar",
        max_display=15,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rf_global_shap_bar.png", dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    warnings.filterwarnings("ignore")
    ensure_directories()

    summary = pd.read_csv(TABLES_DIR / "geopfnmix_ablation_summary.csv")
    save_model_comparison(summary)

    dataset = load_dataset()
    prediction_frame = pd.read_csv(TABLES_DIR / "group_city_geopfnmix_no_residual_oof.csv")
    prediction_frame = prediction_frame.merge(
        dataset.frame.reset_index().rename(columns={"index": "row_id"})[["row_id", "PROVINCE"]],
        on="row_id",
        how="left",
    )
    save_prediction_scatter(prediction_frame)
    save_province_error(prediction_frame)
    save_shap_assets()


if __name__ == "__main__":
    main()
