from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

from pm25_geopfnmix.data import load_dataset
from pm25_geopfnmix.evaluation import compute_metrics
from pm25_geopfnmix.models import make_model
from pm25_geopfnmix.settings import FIGURES_DIR, TABLES_DIR, ensure_directories

sns.set_theme(style="whitegrid", context="talk")

PROVINCE_LABELS = {
    "上海市": "Shanghai",
    "云南省": "Yunnan",
    "内蒙古自治区": "Inner Mongolia",
    "北京市": "Beijing",
    "吉林省": "Jilin",
    "四川省": "Sichuan",
    "天津市": "Tianjin",
    "宁夏回族自治区": "Ningxia",
    "安徽省": "Anhui",
    "山东省": "Shandong",
    "山西省": "Shanxi",
    "广东省": "Guangdong",
    "广西壮族自治区": "Guangxi",
    "新疆维吾尔自治区": "Xinjiang",
    "江苏省": "Jiangsu",
    "江西省": "Jiangxi",
    "河北省": "Hebei",
    "河南省": "Henan",
    "浙江省": "Zhejiang",
    "海南省": "Hainan",
    "湖北省": "Hubei",
    "湖南省": "Hunan",
    "甘肃省": "Gansu",
    "福建省": "Fujian",
    "西藏自治区": "Tibet",
    "贵州省": "Guizhou",
    "辽宁省": "Liaoning",
    "重庆市": "Chongqing",
    "陕西省": "Shaanxi",
    "青海省": "Qinghai",
    "黑龙江省": "Heilongjiang",
}

MODEL_LABELS = {
    "ridge": "Ridge",
    "rf": "RandomForest",
    "hgbt": "HistGBDT",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "tabpfn": "TabPFN",
    "geopfnmix_no_prior": "GeoPFNMix-no-prior",
    "geopfnmix_no_residual": "GeoPFNMix-Lite",
    "geopfnmix_lite_catboost": "GeoPFNMix-Lite-CatBoost",
    "geopfnmix_lite_catboost_tabpfn": "GeoPFNMix-Lite-CatBoost-TabPFN",
    "geopfnmix_no_residual_tabpfn": "GeoPFNMix-Lite-TabPFN",
    "geopfnmix": "GeoPFNMix",
    "geopfnmix_catboost": "GeoPFNMix-CatBoost",
    "geopfnmix_catboost_tabpfn": "GeoPFNMix-CatBoost-TabPFN",
}

SPLIT_LABELS = {
    "random_kfold": "RandomKFold",
    "group_city": "GroupKFold(CITY)",
    "group_province": "GroupKFold(PROVINCE)",
}

PREFERRED_COUNTRY_ORDER = ["Australia", "Brazil", "China", "EU", "USA"]


def _clean_feature_name(name: str) -> str:
    return name.replace("num__", "").replace("cat__", "")


def _to_province_english(series: pd.Series) -> pd.Series:
    def _translate(value: str) -> str:
        if not isinstance(value, str):
            return value
        parts = value.split("::")
        if not parts:
            return value
        last_part = parts[-1]
        translated = PROVINCE_LABELS.get(last_part, last_part)
        if len(parts) == 1:
            return translated
        return "::".join([*parts[:-1], translated])

    return series.map(_translate)


def _format_model_names(frame: pd.DataFrame, column: str = "model_name") -> pd.DataFrame:
    out = frame.copy()
    out[column] = out[column].map(MODEL_LABELS).fillna(out[column])
    return out


def _order_country_summary(summary: pd.DataFrame) -> pd.DataFrame:
    extra_countries = [name for name in summary["country"].tolist() if name not in PREFERRED_COUNTRY_ORDER]
    category_order = [name for name in PREFERRED_COUNTRY_ORDER if name in summary["country"].tolist()] + extra_countries
    summary = summary.copy()
    summary["country"] = pd.Categorical(summary["country"], categories=category_order, ordered=True)
    return summary.sort_values("country").reset_index(drop=True)


def _compute_country_metrics(
    frame: pd.DataFrame,
    *,
    model_name: str | None = None,
    model_label: str | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for country_name, group in frame.groupby("COUNTRY", sort=False):
        metrics = compute_metrics(group["y_true"], group["y_pred"])
        row: dict[str, float | int | str] = {
            "country": country_name,
            "sample_count": int(len(group)),
            **metrics,
        }
        if model_name is not None:
            row["model_name"] = model_name
        if model_label is not None:
            row["model_label"] = model_label
        rows.append(row)

    return _order_country_summary(pd.DataFrame(rows))


def _load_prediction_frame(dataset, model_name: str) -> pd.DataFrame:
    prediction_frame = pd.read_csv(TABLES_DIR / f"group_city_{model_name}_oof.csv")
    prediction_frame = prediction_frame.merge(
        dataset.frame.reset_index().rename(columns={"index": "row_id"})[["row_id", "COUNTRY", "PROVINCE"]],
        on="row_id",
        how="left",
    )
    prediction_frame["model_name"] = model_name
    prediction_frame["model_label"] = MODEL_LABELS.get(model_name, model_name)
    return prediction_frame


def save_protocol_heatmap(summary: pd.DataFrame) -> None:
    plot_frame = summary[summary["metric"] == "rmse"].copy()
    plot_frame["model_name"] = plot_frame["model_name"].map(MODEL_LABELS).fillna(plot_frame["model_name"])
    plot_frame["split_name"] = plot_frame["split_name"].map(SPLIT_LABELS).fillna(plot_frame["split_name"])
    heatmap_frame = plot_frame.pivot(index="model_name", columns="split_name", values="mean")
    heatmap_frame = heatmap_frame.reindex(
        columns=["RandomKFold", "GroupKFold(CITY)", "GroupKFold(PROVINCE)"]
    )
    fig, ax = plt.subplots(figsize=(10, 5.8))
    sns.heatmap(heatmap_frame, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
    ax.set_title("RMSE Across Evaluation Protocols")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "baseline_protocol_heatmap.png", dpi=200)
    plt.close(fig)
    heatmap_frame.reset_index().to_csv(TABLES_DIR / "baseline_protocol_rmse_matrix.csv", index=False)


def save_protocol_best_tables(summary: pd.DataFrame) -> None:
    rmse_frame = summary[summary["metric"] == "rmse"].copy()
    best_rows = rmse_frame.loc[rmse_frame.groupby("split_name")["mean"].idxmin()].copy()
    best_rows["split_label"] = best_rows["split_name"].map(SPLIT_LABELS).fillna(best_rows["split_name"])
    best_rows["model_label"] = best_rows["model_name"].map(MODEL_LABELS).fillna(best_rows["model_name"])
    best_rows = best_rows.sort_values("mean").reset_index(drop=True)

    random_best = float(best_rows.loc[best_rows["split_name"] == "random_kfold", "mean"].iloc[0])
    best_rows["rmse_increase_vs_random_best_pct"] = ((best_rows["mean"] - random_best) / random_best) * 100.0
    best_rows[
        [
            "split_name",
            "split_label",
            "model_name",
            "model_label",
            "mean",
            "std",
            "rmse_increase_vs_random_best_pct",
        ]
    ].to_csv(TABLES_DIR / "protocol_best_models.csv", index=False)

    grouped = rmse_frame.pivot(index="model_name", columns="split_name", values="mean").reset_index()
    grouped["model_label"] = grouped["model_name"].map(MODEL_LABELS).fillna(grouped["model_name"])
    grouped["city_minus_random"] = grouped["group_city"] - grouped["random_kfold"]
    grouped["province_minus_random"] = grouped["group_province"] - grouped["random_kfold"]
    grouped["city_increase_pct"] = grouped["city_minus_random"] / grouped["random_kfold"] * 100.0
    grouped["province_increase_pct"] = grouped["province_minus_random"] / grouped["random_kfold"] * 100.0
    grouped.to_csv(TABLES_DIR / "protocol_generalization_gap.csv", index=False)


def save_model_comparison(summary: pd.DataFrame) -> None:
    plot_frame = summary[summary["metric"] == "rmse"].copy().sort_values("mean")
    plot_frame["model_name"] = plot_frame["model_name"].map(MODEL_LABELS).fillna(plot_frame["model_name"])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=plot_frame, x="mean", y="model_name", hue="model_name", palette="crest", ax=ax, legend=False)
    ax.set_xlabel("RMSE")
    ax.set_ylabel("")
    ax.set_title("GroupKFold(CITY) RMSE Comparison")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "group_city_model_comparison.png", dpi=200)
    plt.close(fig)


def save_ablation_metrics(summary: pd.DataFrame) -> None:
    plot_frame = _format_model_names(summary[summary["metric"].isin(["rmse", "mae"])].copy())
    plot_frame["metric"] = plot_frame["metric"].str.upper()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for axis, metric_name in zip(axes, ["RMSE", "MAE"]):
        metric_frame = plot_frame[plot_frame["metric"] == metric_name].sort_values("mean")
        sns.barplot(
            data=metric_frame,
            x="mean",
            y="model_name",
            hue="model_name",
            palette="viridis",
            ax=axis,
            legend=False,
        )
        axis.set_title(f"Ablation Comparison by {metric_name}")
        axis.set_xlabel(metric_name)
        axis.set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "geopfnmix_ablation_metrics.png", dpi=200)
    plt.close(fig)


def save_prediction_scatter(frame: pd.DataFrame) -> None:
    model_label = frame["model_label"].iloc[0]
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
    ax.set_title(f"{model_label} OOF Predictions")
    ax.set_xlabel("Observed PM2.5")
    ax.set_ylabel("Predicted PM2.5")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "best_geopfnmix_scatter.png", dpi=200)
    plt.close(fig)


def save_province_error(frame: pd.DataFrame) -> None:
    model_label = frame["model_label"].iloc[0]
    plot_frame = (
        frame.groupby("PROVINCE", as_index=False)["abs_error"]
        .mean()
        .sort_values("abs_error", ascending=False)
    )
    plot_frame["province_en"] = _to_province_english(plot_frame["PROVINCE"])
    if len(plot_frame) > 40:
        plot_frame_for_chart = pd.concat([plot_frame.head(20), plot_frame.tail(20)], ignore_index=True)
    else:
        plot_frame_for_chart = plot_frame
    fig, ax = plt.subplots(figsize=(10, 11))
    sns.barplot(
        data=plot_frame_for_chart,
        x="abs_error",
        y="province_en",
        hue="province_en",
        palette="mako",
        ax=ax,
        legend=False,
    )
    ax.set_xlabel("Mean Absolute Error")
    ax.set_ylabel("")
    ax.set_title(f"Province/State-Level Error of {model_label}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "province_level_mae.png", dpi=200)
    plt.close(fig)
    plot_frame.rename(columns={"PROVINCE": "province_cn"}).to_csv(TABLES_DIR / "province_level_mae.csv", index=False)

    top_bottom = pd.concat([plot_frame.head(5), plot_frame.tail(5)], ignore_index=True)
    top_bottom.rename(columns={"PROVINCE": "province_cn"}).to_csv(
        TABLES_DIR / "province_level_mae_top_bottom.csv", index=False
    )


def save_error_by_target_bin(frame: pd.DataFrame) -> None:
    model_label = frame["model_label"].iloc[0]
    plot_frame = frame.copy()
    plot_frame["pm25_bin"] = pd.cut(
        plot_frame["y_true"],
        bins=[-np.inf, 20.0, 35.0, 50.0, np.inf],
        labels=["<20", "20-35", "35-50", ">=50"],
    )
    summary = (
        plot_frame.groupby("pm25_bin", observed=False)
        .agg(
            sample_count=("row_id", "count"),
            mae=("abs_error", "mean"),
            rmse=("abs_error", lambda values: float(np.sqrt(np.mean(np.square(values))))),
        )
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.barplot(data=summary, x="pm25_bin", y="mae", hue="pm25_bin", palette="flare", ax=ax, legend=False)
    ax.set_xlabel("Observed PM2.5 Bin")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title(f"{model_label} Error by PM2.5 Range")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "geopfnmix_error_by_target_bin.png", dpi=200)
    plt.close(fig)
    summary.to_csv(TABLES_DIR / "geopfnmix_error_by_target_bin.csv", index=False)


def save_country_breakdown(frame: pd.DataFrame) -> None:
    model_label = frame["model_label"].iloc[0]
    summary = _compute_country_metrics(frame)
    summary.to_csv(TABLES_DIR / "best_model_country_metrics.csv", index=False)

    plot_frame = summary.melt(
        id_vars=["country", "sample_count", "r2"],
        value_vars=["rmse", "mae"],
        var_name="metric",
        value_name="value",
    )
    plot_frame["metric"] = plot_frame["metric"].str.upper()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), sharey=True)
    for axis, metric_name in zip(axes, ["RMSE", "MAE"]):
        metric_frame = plot_frame[plot_frame["metric"] == metric_name]
        sns.barplot(
            data=metric_frame,
            x="value",
            y="country",
            hue="country",
            palette="crest",
            ax=axis,
            legend=False,
        )
        axis.set_title(f"{metric_name} by Data Source")
        axis.set_xlabel(metric_name)
        axis.set_ylabel("" if metric_name == "MAE" else "Data Source")
        for patch, (_, row) in zip(axis.patches, metric_frame.iterrows()):
            axis.text(
                patch.get_width() + 0.02,
                patch.get_y() + patch.get_height() / 2,
                f"n={int(row['sample_count'])}",
                va="center",
                fontsize=10,
            )
    fig.suptitle(f"{model_label} Performance by Data Source", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "best_model_country_metrics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_selected_models_country_breakdown(dataset, model_names: list[str]) -> None:
    ordered_model_names: list[str] = []
    for model_name in model_names:
        if model_name not in ordered_model_names:
            ordered_model_names.append(model_name)

    summary_frames: list[pd.DataFrame] = []
    model_label_order = [MODEL_LABELS.get(model_name, model_name) for model_name in ordered_model_names]

    for model_name, model_label in zip(ordered_model_names, model_label_order):
        prediction_frame = _load_prediction_frame(dataset, model_name)
        summary_frames.append(
            _compute_country_metrics(
                prediction_frame,
                model_name=model_name,
                model_label=model_label,
            )
        )

    combined = pd.concat(summary_frames, ignore_index=True)
    extra_countries = [name for name in combined["country"].tolist() if name not in PREFERRED_COUNTRY_ORDER]
    country_order = [name for name in PREFERRED_COUNTRY_ORDER if name in combined["country"].tolist()] + extra_countries
    combined["country"] = pd.Categorical(
        combined["country"],
        categories=country_order,
        ordered=True,
    )
    combined["model_label"] = pd.Categorical(
        combined["model_label"],
        categories=model_label_order,
        ordered=True,
    )
    combined = combined.sort_values(["model_label", "country"]).reset_index(drop=True)
    combined.to_csv(TABLES_DIR / "selected_models_country_metrics.csv", index=False)

    sample_counts = (
        combined.groupby("country", observed=False)["sample_count"]
        .max()
        .dropna()
        .astype(int)
        .to_dict()
    )
    country_order = [name for name in country_order if name in sample_counts]
    country_labels = country_order

    metric_specs = [
        ("rmse", "RMSE", "YlGnBu", None),
        ("mae", "MAE", "crest", None),
        ("r2", "R²", "coolwarm", 0.0),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for axis, (metric_name, metric_label, cmap, center) in zip(axes, metric_specs):
        heatmap_frame = (
            combined.pivot(index="model_label", columns="country", values=metric_name)
            .reindex(index=model_label_order)
            .reindex(columns=country_order)
        )
        sns.heatmap(
            heatmap_frame,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            center=center,
            linewidths=0.5,
            linecolor="white",
            cbar=True,
            ax=axis,
        )
        axis.set_title(f"{metric_label} by Data Source")
        axis.set_xlabel("")
        axis.set_ylabel("Model" if metric_name == "rmse" else "")
        axis.set_xticklabels(country_labels, rotation=0)
        axis.set_yticklabels(axis.get_yticklabels(), rotation=0)
        if metric_name != "rmse":
            axis.tick_params(axis="y", labelleft=False)

    fig.suptitle("Selected Models Performance by Data Source", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "selected_models_country_metrics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_shap_assets(global_expert_name: str) -> None:
    dataset = load_dataset()
    model = make_model(global_expert_name)
    model.fit(dataset.features, dataset.target)

    sample_size = min(128, dataset.features.shape[0])

    if global_expert_name == "catboost":
        engineered = model.feature_engineer.transform(dataset.features)  # type: ignore[attr-defined]
        transformed_sample = engineered.iloc[:sample_size].copy()
        regressor = model.model  # type: ignore[attr-defined]
        feature_names = transformed_sample.columns.tolist()
    else:
        engineered = model.feature_engineer.transform(dataset.features)  # type: ignore[attr-defined]
        transformed = model.pipeline.named_steps["preprocessor"].transform(engineered)  # type: ignore[attr-defined]
        transformed_sample = transformed[:sample_size]
        regressor = model.pipeline.named_steps["model"]  # type: ignore[attr-defined]
        feature_names = model.pipeline.named_steps["preprocessor"].get_feature_names_out()  # type: ignore[attr-defined]

    explainer = shap.TreeExplainer(regressor)
    shap_values = explainer.shap_values(transformed_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_frame = pd.DataFrame(
        {
            "feature": feature_names,
            "feature_clean": [_clean_feature_name(name) for name in feature_names],
            "mean_abs_shap": pd.DataFrame(shap_values, columns=feature_names).abs().mean().values,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    shap_frame.to_csv(TABLES_DIR / "best_global_expert_shap_importance.csv", index=False)
    shap_frame.head(10).to_csv(TABLES_DIR / "best_global_expert_shap_top10.csv", index=False)

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
    plt.savefig(FIGURES_DIR / "best_global_expert_shap_bar.png", dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    warnings.filterwarnings("ignore")
    ensure_directories()

    baseline_summary = pd.read_csv(TABLES_DIR / "baseline_summary.csv")
    save_protocol_heatmap(baseline_summary)
    save_protocol_best_tables(baseline_summary)

    ablation_summary = pd.read_csv(TABLES_DIR / "geopfnmix_ablation_summary.csv")
    save_model_comparison(ablation_summary)
    save_ablation_metrics(ablation_summary)

    ablation_rmse = ablation_summary[ablation_summary["metric"] == "rmse"].copy()
    geopfnmix_candidates = ablation_rmse[ablation_rmse["model_name"].str.startswith("geopfnmix")].copy()
    best_model_name = geopfnmix_candidates.sort_values("mean").iloc[0]["model_name"]
    best_model_label = MODEL_LABELS.get(best_model_name, best_model_name)
    best_global_expert = "catboost" if "catboost" in str(best_model_name) else "rf"

    dataset = load_dataset()
    prediction_frame = _load_prediction_frame(dataset, best_model_name)
    prediction_frame["model_label"] = best_model_label
    selected_model_names = geopfnmix_candidates.sort_values("mean")["model_name"].drop_duplicates().head(2).tolist()
    selected_model_names.extend(["catboost", "tabpfn"])
    save_prediction_scatter(prediction_frame)
    save_country_breakdown(prediction_frame)
    save_selected_models_country_breakdown(dataset, selected_model_names)
    save_province_error(prediction_frame)
    save_error_by_target_bin(prediction_frame)
    save_shap_assets(best_global_expert)


if __name__ == "__main__":
    main()
