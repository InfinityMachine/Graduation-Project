from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .settings import NUM_COLS, TARGET_COL

sns.set_theme(style="whitegrid", context="talk")


def save_target_distribution(frame: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.histplot(frame[TARGET_COL], kde=True, bins=30, color="#2f6f72", ax=ax)
    ax.set_title("PM2.5 Distribution")
    ax.set_xlabel("PM2.5")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_numeric_correlation(frame: pd.DataFrame, output_path: Path) -> None:
    corr = frame[NUM_COLS + [TARGET_COL]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="RdBu_r", center=0.0, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Numeric Correlation Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_group_size_distribution(frame: pd.DataFrame, output_path: Path) -> None:
    city_sizes = frame["CITY"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.histplot(city_sizes, bins=30, color="#8f5f3f", ax=ax)
    ax.set_title("City Group Size Distribution")
    ax.set_xlabel("Samples per city")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_feature_vs_target(frame: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    chosen = ["NOX", "SO2", "fertilzier", "manure"]
    colors = ["#275d63", "#ac4d4d", "#8d6c2f", "#4e7c59"]
    for axis, feature, color in zip(axes.flat, chosen, colors):
        sns.scatterplot(data=frame, x=feature, y=TARGET_COL, s=30, alpha=0.7, color=color, ax=axis)
        axis.set_title(f"{feature} vs PM2.5")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
