from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from pm25_geopfnmix.data import load_dataset


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "artifacts" / "tables"
OUT = ROOT / "artifacts" / "figures" / "论文重绘图表"

NAVY = "#14394E"
TEAL = "#257584"
TEAL_DARK = "#1F6F8B"
TEAL_LIGHT = "#E0F1F2"
ORANGE = "#C97031"
ORANGE_LIGHT = "#FFF4EB"
BLUE_GRAY = "#8FA5B5"
LIGHT_GRAY = "#F2F5F7"
DARK = "#1E272E"
GRAY = "#5C6773"

MODEL_LABELS = {
    "rf": "RandomForest",
    "catboost": "CatBoost",
    "tabpfn": "TabPFN",
    "geopfnmix_no_prior": "GeoPFNMix-No-Prior",
    "geopfnmix_no_residual": "GeoPFNMix-Lite",
    "geopfnmix_no_residual_tabpfn": "GeoPFNMix-Lite-TabPFN",
    "geopfnmix": "GeoPFNMix",
    "geopfnmix_tabpfn": "GeoPFNMix-TabPFN",
    "geopfnmix_lite_catboost": "GeoPFNMix-Lite-CatBoost",
    "geopfnmix_catboost": "GeoPFNMix-CatBoost",
    "geopfnmix_lite_catboost_tabpfn": "GeoPFNMix-Lite-CatBoost-TabPFN",
    "geopfnmix_catboost_tabpfn": "GeoPFNMix-CatBoost-TabPFN",
}

COUNTRY_ORDER = ["Australia", "Brazil", "China", "EU", "USA"]


def configure_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "font.family": "sans-serif",
            "axes.unicode_minus": False,
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.edgecolor": "#3A3A3A",
            "axes.linewidth": 0.8,
        }
    )


def save_figure(fig: plt.Figure, filename: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{filename}.png", bbox_inches="tight", dpi=300)
    fig.savefig(OUT / f"{filename}.svg", bbox_inches="tight")
    plt.close(fig)


def metric_values(y_true: pd.Series, y_pred: pd.Series) -> tuple[float, float, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return rmse, mae, r2


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", color="#D9DEE3", linewidth=0.7, alpha=0.8)
    ax.grid(True, axis="x", color="#E6EAEE", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_box(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str, body: str, color: str) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=1.35,
        edgecolor=color,
        facecolor="white",
    )
    ax.add_patch(patch)
    ax.add_patch(Rectangle((x, y + h - 0.32), w, 0.32, color=color, lw=0))
    ax.text(x + w / 2, y + h - 0.16, title, ha="center", va="center", color="white", weight="bold", fontsize=10.5)
    line_count = body.count("\n") + 1
    body_font = 8.0 if line_count >= 3 else 9.0
    body_y = y + (h - 0.32) / 2
    ax.text(x + w / 2, body_y, body, ha="center", va="center", color=DARK, fontsize=body_font, linespacing=1.28)


def draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = NAVY, dashed: bool = False) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.25,
            color=color,
            linestyle="--" if dashed else "-",
        )
    )


def figure_01_target_distribution(frame: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    values = frame["PM2.5"].astype(float)
    sns.histplot(values, bins=42, kde=True, color=TEAL, alpha=0.45, edgecolor="white", ax=ax)
    mean_value = values.mean()
    median_value = values.median()
    ax.axvline(mean_value, color=ORANGE, linestyle="--", linewidth=1.8, label=f"均值 = {mean_value:.2f}")
    ax.axvline(median_value, color=NAVY, linestyle=":", linewidth=1.8, label=f"中位数 = {median_value:.2f}")
    ax.set_title("PM2.5 目标值分布")
    ax.set_xlabel("PM2.5 浓度")
    ax.set_ylabel("样本数")
    ax.legend(frameon=False, loc="upper right")
    style_axis(ax)
    save_figure(fig, "图1_PM25目标分布")


def figure_02_city_group_size(frame: pd.DataFrame) -> None:
    counts = frame.groupby("CITY").size()
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.hist(counts, bins=45, color=TEAL, alpha=0.78, edgecolor="white")
    ax.axvline(counts.median(), color=ORANGE, linestyle="--", linewidth=1.8, label=f"中位数 = {counts.median():.0f}")
    ax.set_title("二级地理组样本规模分布")
    ax.set_xlabel("每个二级地理组的样本数")
    ax.set_ylabel("二级地理组数量")
    ax.legend(frameon=False, loc="upper right")
    stats_text = f"组数：{len(counts)}\n最大组：{counts.max()}\n均值：{counts.mean():.1f}"
    ax.text(0.98, 0.72, stats_text, transform=ax.transAxes, ha="right", va="top", fontsize=9.5, color=DARK)
    style_axis(ax)
    save_figure(fig, "图2_二级地理组样本规模分布")


def figure_03_numeric_correlation(frame: pd.DataFrame) -> None:
    numeric_cols = ["AET", "ppt", "tem", "wind", "NOX", "SO2", "fertilzier", "manure", "PM2.5"]
    corr = frame[numeric_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "Pearson 相关系数", "shrink": 0.78},
        ax=ax,
    )
    ax.set_title("数值特征相关性热力图")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)
    save_figure(fig, "图3_数值特征相关性热力图")


def figure_04_key_features_vs_target(frame: pd.DataFrame) -> None:
    plot_source = frame.copy()
    for col in ["NOX", "SO2"]:
        plot_source[f"log1p_{col}"] = np.log1p(plot_source[col].clip(lower=0))

    features = [
        ("log1p_NOX", "log1p(NOX) 与 PM2.5", "log1p(NOX)"),
        ("log1p_SO2", "log1p(SO2) 与 PM2.5", "log1p(SO2)"),
        ("AET", "AET 与 PM2.5", "AET"),
        ("wind", "风速与 PM2.5", "风速"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2))
    for ax, (feature, title, xlabel) in zip(axes.ravel(), features):
        plot_frame = plot_source[[feature, "PM2.5"]].dropna()
        ax.scatter(plot_frame[feature], plot_frame["PM2.5"], s=12, alpha=0.22, color=TEAL, edgecolors="none")
        if plot_frame[feature].nunique() > 3:
            x = plot_frame[feature].astype(float)
            y = plot_frame["PM2.5"].astype(float)
            order = np.argsort(x)
            x_sorted = x.iloc[order].to_numpy()
            y_sorted = y.iloc[order].rolling(max(30, len(y) // 80), min_periods=10, center=True).mean().to_numpy()
            ax.plot(x_sorted, y_sorted, color=ORANGE, linewidth=1.5, alpha=0.9)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("PM2.5 浓度")
        style_axis(ax)
    fig.suptitle("关键特征与 PM2.5 的关系", fontsize=14, color=NAVY, weight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, "图4_关键特征与目标关系")


def figure_05_geopfnmix_framework() -> None:
    fig, ax = plt.subplots(figsize=(12.6, 6.4))
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    ax.text(6.3, 6.08, "GeoPFNMix 框架示意图", ha="center", va="center", fontsize=16, weight="bold", color=NAVY)
    ax.text(6.3, 5.75, "以低容量融合连接全局规律、地理层级偏差与表格先验", ha="center", va="center", fontsize=10.5, color=GRAY)

    draw_box(ax, 0.35, 4.45, 2.15, 0.85, "原始多源数据", "污染 / 气象 / 农业投入\nCOUNTRY / PROVINCE / CITY\nPM2.5 目标", TEAL)
    draw_box(ax, 0.35, 3.25, 2.15, 0.75, "统一特征工程", "缺失插补、分位裁剪\nlog1p 与交互特征", TEAL)
    draw_box(ax, 0.35, 2.05, 2.15, 0.75, "工程化特征", "统一输入矩阵 X*", TEAL)
    draw_box(ax, 0.35, 0.88, 0.98, 0.67, "分组信息", "PROVINCE\nCITY", TEAL)
    draw_box(ax, 1.52, 0.88, 0.98, 0.67, "训练目标", "y", TEAL)

    draw_box(ax, 3.35, 4.25, 2.0, 0.75, "全局专家", "RF 或 CatBoost\n输出：y_g", "#2E8B57")
    draw_box(ax, 3.35, 3.05, 2.0, 0.75, "先验专家", "LightGBM 或 TabPFN\n输出：y_p", "#B44A45")
    residual = FancyBboxPatch(
        (3.15, 1.05),
        2.45,
        1.55,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=1.3,
        linestyle="--",
        edgecolor=ORANGE,
        facecolor=ORANGE_LIGHT,
    )
    ax.add_patch(residual)
    ax.text(4.38, 2.35, "训练折内残差分支", ha="center", va="center", fontsize=10.5, weight="bold", color=ORANGE)
    ax.text(4.38, 1.82, "e = y - y_g\n按 PROVINCE / CITY 估计\n并按样本支撑度收缩", ha="center", va="center", fontsize=9.2, color=DARK)
    ax.text(4.38, 1.25, "输出：y_h", ha="center", va="center", fontsize=9.5, color=DARK)

    draw_box(ax, 6.65, 2.55, 2.05, 1.45, "元特征构造", "全局预测 y_g\n校正预测 y_h\n先验预测 y_p\n专家分歧与支撑度", "#5B4B8A")
    draw_box(ax, 9.4, 2.85, 1.7, 0.95, "低容量线性融合", "OOF 训练\n控制过拟合", NAVY)
    draw_box(ax, 9.55, 1.35, 1.45, 0.85, "最终预测", "PM2.5 估计\n分组泛化评估", "#2E8B57")

    draw_arrow(ax, (1.43, 4.45), (1.43, 4.02), TEAL)
    draw_arrow(ax, (1.43, 3.25), (1.43, 2.82), TEAL)
    draw_arrow(ax, (2.5, 2.42), (3.35, 4.62), NAVY)
    draw_arrow(ax, (2.5, 2.42), (3.35, 3.42), NAVY)
    draw_arrow(ax, (1.02, 1.55), (3.15, 1.78), NAVY, dashed=True)
    draw_arrow(ax, (2.0, 1.55), (3.15, 1.78), NAVY, dashed=True)
    draw_arrow(ax, (5.35, 4.62), (6.65, 3.55), NAVY)
    draw_arrow(ax, (5.35, 3.42), (6.65, 3.22), NAVY)
    draw_arrow(ax, (5.6, 1.75), (6.65, 2.83), ORANGE)
    draw_arrow(ax, (8.7, 3.25), (9.4, 3.32), NAVY)
    draw_arrow(ax, (10.25, 2.85), (10.25, 2.2), NAVY)
    ax.text(0.35, 0.28, "说明：残差分支仅使用训练折可见信息；验证/测试折只应用已估计的收缩校正，避免信息泄漏。", fontsize=9.2, color=GRAY)
    save_figure(fig, "图5_GeoPFNMix框架示意图")


def figure_06_protocol_heatmap() -> None:
    matrix = pd.read_csv(TABLES / "baseline_protocol_rmse_matrix.csv").set_index("model_name")
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "RMSE", "shrink": 0.82},
        ax=ax,
    )
    ax.set_title("不同评估协议下的 RMSE 对比")
    ax.set_xlabel("评估协议")
    ax.set_ylabel("模型")
    ax.tick_params(axis="x", rotation=18)
    ax.tick_params(axis="y", rotation=0)
    save_figure(fig, "图6_不同协议RMSE热力图")


def get_ablation_wide() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(TABLES / "geopfnmix_ablation_summary.csv")
    wide = summary.pivot(index="model_name", columns="metric", values="mean")
    std = summary.pivot(index="model_name", columns="metric", values="std")
    models = [m for m in wide.index if m in MODEL_LABELS]
    order = sorted(models, key=lambda m: float(wide.loc[m, "rmse"]))
    plot = pd.DataFrame(
        {
            "model_name": order,
            "model_label": [MODEL_LABELS[m] for m in order],
            "rmse": [float(wide.loc[m, "rmse"]) for m in order],
            "rmse_std": [float(std.loc[m, "rmse"]) for m in order],
            "mae": [float(wide.loc[m, "mae"]) for m in order],
            "mae_std": [float(std.loc[m, "mae"]) for m in order],
        }
    )
    return plot, wide


def model_color(label: str, best: bool = False) -> str:
    if best:
        return TEAL_DARK
    if "CatBoost" in label:
        return "#5F9EA0"
    return BLUE_GRAY


def figure_07_model_comparison() -> None:
    plot, _ = get_ablation_wide()
    fig, ax = plt.subplots(figsize=(10.8, 6.7))
    y = np.arange(len(plot))
    colors = [model_color(label, i == 0) for i, label in enumerate(plot["model_label"])]
    ax.barh(
        y,
        plot["rmse"],
        xerr=plot["rmse_std"],
        color=colors,
        edgecolor=NAVY,
        linewidth=0.45,
        capsize=2.5,
        error_kw={"elinewidth": 1.0, "ecolor": DARK},
    )
    ax.set_yticks(y)
    ax.set_yticklabels(plot["model_label"])
    ax.invert_yaxis()
    ax.set_xlabel("RMSE（越低越好）")
    ax.set_title("GeoPFNMix 家族在 GroupKFold(CITY) 下的 RMSE 对比")
    ax.set_xlim(0, max(plot["rmse"] + plot["rmse_std"]) + 0.65)
    for yi, val, std in zip(y, plot["rmse"], plot["rmse_std"]):
        ax.text(val + std + 0.06, yi, f"{val:.3f}", va="center", fontsize=8.8, color=DARK)
    style_axis(ax)
    ax.grid(True, axis="x", color="#D9DEE3", linewidth=0.7, alpha=0.8)
    ax.grid(False, axis="y")
    fig.subplots_adjust(left=0.31, right=0.98, top=0.9, bottom=0.12)
    save_figure(fig, "图7_GeoPFNMix家族RMSE对比")


def figure_08_ablation_metrics() -> None:
    plot, _ = get_ablation_wide()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.8), sharey=True)
    y = np.arange(len(plot))
    for ax, metric, std_metric, title, color in [
        (axes[0], "rmse", "rmse_std", "RMSE", TEAL),
        (axes[1], "mae", "mae_std", "MAE", ORANGE),
    ]:
        ax.barh(
            y,
            plot[metric],
            xerr=plot[std_metric],
            color=color,
            alpha=0.86,
            edgecolor=NAVY,
            linewidth=0.4,
            capsize=2.2,
            error_kw={"elinewidth": 1.0, "ecolor": DARK},
        )
        ax.set_title(title)
        ax.set_xlabel(f"{title}（越低越好）")
        ax.set_xlim(0, max(plot[metric] + plot[std_metric]) + (0.58 if metric == "rmse" else 0.44))
        for yi, val, std_value in zip(y, plot[metric], plot[std_metric]):
            ax.text(val + std_value + 0.045, yi, f"{val:.3f}", va="center", fontsize=8.2, color=DARK)
        style_axis(ax)
        ax.grid(True, axis="x", color="#D9DEE3", linewidth=0.7, alpha=0.8)
        ax.grid(False, axis="y")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(plot["model_label"])
    axes[0].invert_yaxis()
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    fig.suptitle("GeoPFNMix 家族消融实验：RMSE 与 MAE 对比", fontsize=14, color=NAVY, weight="bold")
    fig.subplots_adjust(left=0.25, right=0.98, top=0.88, bottom=0.12, wspace=0.12)
    save_figure(fig, "图8_GeoPFNMix家族双指标消融")


def figure_09_oof_scatter() -> None:
    oof = pd.read_csv(TABLES / "group_city_geopfnmix_catboost_tabpfn_oof.csv")
    rmse, mae, r2 = metric_values(oof["y_true"], oof["y_pred"])
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.scatter(oof["y_true"], oof["y_pred"], s=12, alpha=0.25, color=TEAL, edgecolors="none")
    lim_min = min(oof["y_true"].min(), oof["y_pred"].min())
    lim_max = max(oof["y_true"].max(), oof["y_pred"].max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color=ORANGE, linestyle="--", linewidth=1.8, label="理想预测线")
    ax.set_title("当前最佳模型的 OOF 预测散点图")
    ax.set_xlabel("真实 PM2.5")
    ax.set_ylabel("预测 PM2.5")
    ax.text(
        0.05,
        0.95,
        f"RMSE = {rmse:.3f}\nMAE = {mae:.3f}\nR² = {r2:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color=DARK,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#D9DEE3"},
    )
    ax.legend(frameon=False, loc="lower right")
    style_axis(ax)
    save_figure(fig, "图9_当前最佳模型OOF预测散点图")


def figure_10_error_by_target_bin() -> None:
    bins = pd.read_csv(TABLES / "geopfnmix_error_by_target_bin.csv")
    x = np.arange(len(bins))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.bar(x - width / 2, bins["mae"], width, label="MAE", color=TEAL, alpha=0.88)
    ax.bar(x + width / 2, bins["rmse"], width, label="RMSE", color=ORANGE, alpha=0.86)
    for xi, count in zip(x, bins["sample_count"]):
        ax.text(xi, max(bins["rmse"]) * 0.03, f"n={count}", ha="center", va="bottom", fontsize=8.8, color=DARK)
    ax.set_xticks(x)
    ax.set_xticklabels(bins["pm25_bin"])
    ax.set_title("当前最佳模型在不同 PM2.5 区间上的误差")
    ax.set_xlabel("PM2.5 区间")
    ax.set_ylabel("误差")
    ax.legend(frameon=False)
    style_axis(ax)
    save_figure(fig, "图10_不同PM25区间误差")


def figure_11_country_metrics() -> None:
    metrics = pd.read_csv(TABLES / "best_model_country_metrics.csv")
    metrics["country"] = pd.Categorical(metrics["country"], categories=COUNTRY_ORDER, ordered=True)
    metrics = metrics.sort_values("country")
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.6))
    specs = [("rmse", "RMSE", TEAL), ("mae", "MAE", ORANGE), ("r2", "R²", NAVY)]
    for ax, (col, title, color) in zip(axes, specs):
        ax.bar(metrics["country"].astype(str), metrics[col], color=color, alpha=0.86, edgecolor=NAVY, linewidth=0.35)
        ax.axhline(0, color="#B0B8BF", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=28)
        style_axis(ax)
        for i, val in enumerate(metrics[col]):
            offset = 0.05 if val >= 0 else -0.08
            va = "bottom" if val >= 0 else "top"
            ax.text(i, val + offset, f"{val:.2f}", ha="center", va=va, fontsize=8.2, color=DARK)
    fig.suptitle("当前最佳模型按数据源拆分后的表现", fontsize=14, color=NAVY, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    save_figure(fig, "图11_当前最佳模型按数据源拆分表现")


def figure_12_selected_models_country_heatmap() -> None:
    metrics = pd.read_csv(TABLES / "selected_models_country_metrics.csv")
    label_map = {
        "geopfnmix_catboost_tabpfn": "GeoPFNMix-CB-TabPFN",
        "geopfnmix_lite_catboost_tabpfn": "GeoPFNMix-Lite-CB-TabPFN",
        "catboost": "CatBoost",
        "tabpfn": "TabPFN",
    }
    metrics = metrics[metrics["model_name"].isin(label_map)].copy()
    metrics["model_label"] = metrics["model_name"].map(label_map)
    metrics["country"] = pd.Categorical(metrics["country"], categories=COUNTRY_ORDER, ordered=True)
    model_order = list(label_map.values())

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.2), gridspec_kw={"wspace": 0.34})
    specs = [
        ("rmse", "RMSE", "YlGnBu", "RMSE"),
        ("mae", "MAE", "YlGnBu", "MAE"),
        ("r2", "R²", "coolwarm_r", "R²"),
    ]
    for ax, (value_col, title, cmap, cbar_label) in zip(axes, specs):
        pivot = metrics.pivot(index="model_label", columns="country", values=value_col)
        pivot = pivot.reindex(index=model_order, columns=COUNTRY_ORDER)
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            linewidths=0.65,
            linecolor="white",
            cbar_kws={"label": cbar_label, "shrink": 0.78},
            ax=ax,
        )
        ax.set_title(f"{title} 按数据源对比", fontsize=12.5, color=NAVY, weight="bold", pad=10)
        ax.set_xlabel("数据源")
        ax.set_ylabel("模型" if ax is axes[0] else "")
        ax.tick_params(axis="x", rotation=28)
        ax.tick_params(axis="y", rotation=0)
        if ax is not axes[0]:
            ax.set_yticklabels([])
    fig.suptitle("四个代表模型按数据源拆分后的多指标表现", fontsize=15, color=NAVY, weight="bold", y=1.03)
    fig.tight_layout()
    save_figure(fig, "图12_代表模型按数据源拆分多指标表现")


def figure_13_province_level_mae() -> None:
    province = pd.read_csv(TABLES / "province_level_mae.csv")
    province = province.dropna(subset=["abs_error"]).copy()
    top = province.sort_values("abs_error", ascending=False).head(8)
    bottom = province.sort_values("abs_error", ascending=True).head(8).sort_values("abs_error", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), gridspec_kw={"width_ratios": [1.0, 1.25]})

    axes[0].hist(province["abs_error"], bins=28, color=TEAL, alpha=0.78, edgecolor="white")
    axes[0].axvline(province["abs_error"].median(), color=ORANGE, linestyle="--", linewidth=1.7, label=f"中位数={province['abs_error'].median():.2f}")
    axes[0].set_title("一级地理组 MAE 分布")
    axes[0].set_xlabel("MAE")
    axes[0].set_ylabel("一级地理组数量")
    axes[0].legend(frameon=False)
    style_axis(axes[0])

    combo = pd.concat([top.assign(group="误差较高"), bottom.assign(group="误差较低")])
    colors = [ORANGE if g == "误差较高" else TEAL for g in combo["group"]]
    y = np.arange(len(combo))
    axes[1].barh(y, combo["abs_error"], color=colors, alpha=0.86, edgecolor=NAVY, linewidth=0.3)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(combo["province_en"])
    axes[1].invert_yaxis()
    axes[1].set_title("典型高误差与低误差地理组")
    axes[1].set_xlabel("MAE")
    for yi, val in zip(y, combo["abs_error"]):
        axes[1].text(val + 0.12, yi, f"{val:.2f}", va="center", fontsize=8.3, color=DARK)
    style_axis(axes[1])
    axes[1].grid(True, axis="x", color="#D9DEE3", linewidth=0.7, alpha=0.8)
    axes[1].grid(False, axis="y")

    fig.suptitle("一级地理组误差分布", fontsize=14, color=NAVY, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    save_figure(fig, "图13_一级地理组误差分布")


def figure_14_shap_bar() -> None:
    shap_top = pd.read_csv(TABLES / "best_global_expert_shap_top10.csv").sort_values("mean_abs_shap", ascending=True)
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    colors = [ORANGE if feature == "COUNTRY" else TEAL for feature in shap_top["feature_clean"]]
    ax.barh(shap_top["feature_clean"], shap_top["mean_abs_shap"], color=colors, alpha=0.88, edgecolor=NAVY, linewidth=0.35)
    for yi, val in enumerate(shap_top["mean_abs_shap"]):
        ax.text(val + 0.04, yi, f"{val:.3f}", va="center", fontsize=8.6, color=DARK)
    ax.set_title("当前最佳全局专家的 SHAP 特征重要性")
    ax.set_xlabel("平均绝对 SHAP 值")
    ax.set_ylabel("特征")
    ax.set_xlim(0, shap_top["mean_abs_shap"].max() + 0.8)
    style_axis(ax)
    ax.grid(True, axis="x", color="#D9DEE3", linewidth=0.7, alpha=0.8)
    ax.grid(False, axis="y")
    save_figure(fig, "图14_SHAP特征重要性排名")


def figure_overview_contact_sheet() -> None:
    files = sorted(
        OUT.glob("图*_*.png"),
        key=lambda path: int(re.match(r"图(\d+)_", path.name).group(1)) if re.match(r"图(\d+)_", path.name) else 999,
    )
    files = [path for path in files if re.match(r"图\d+_", path.name)]
    fig, axes = plt.subplots(7, 2, figsize=(13.2, 24.5))
    for ax, path in zip(axes.ravel(), files):
        image = plt.imread(path)
        ax.imshow(image)
        ax.set_title(path.stem, fontsize=10, color=NAVY, pad=6)
        ax.axis("off")
    for ax in axes.ravel()[len(files) :]:
        ax.axis("off")
    fig.suptitle("论文图表重绘总览", fontsize=18, color=NAVY, weight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(OUT / "论文重绘图表总览.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def main() -> None:
    configure_style()
    dataset = load_dataset()
    frame = dataset.frame.copy()
    figure_01_target_distribution(frame)
    figure_02_city_group_size(frame)
    figure_03_numeric_correlation(frame)
    figure_04_key_features_vs_target(frame)
    figure_05_geopfnmix_framework()
    figure_06_protocol_heatmap()
    figure_07_model_comparison()
    figure_08_ablation_metrics()
    figure_09_oof_scatter()
    figure_10_error_by_target_bin()
    figure_11_country_metrics()
    figure_12_selected_models_country_heatmap()
    figure_13_province_level_mae()
    figure_14_shap_bar()
    figure_overview_contact_sheet()
    print(f"重绘图表已输出到：{OUT}")


if __name__ == "__main__":
    main()
