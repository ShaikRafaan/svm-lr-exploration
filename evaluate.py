"""
evaluate.py
-----------
Evaluation utilities: accuracy tables, train-vs-test plots,
confusion matrices, and LR vs SVM comparison charts.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from matplotlib.patches import Patch

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

PALETTE = {"SVM": "#4C72B0", "LR": "#DD8452"}


def save_results_table(df, filename):
    path = os.path.join(RESULTS_DIR, filename)
    df.to_csv(path, index=False)
    print(f"\nResults saved -> {path}")
    print(df.to_string(index=False))
    return path


def plot_train_vs_test(df, title, filename):
    df_plot = df[df["train_acc"] != "CV"].copy()
    df_plot["train_acc"] = df_plot["train_acc"].astype(float)
    df_plot["test_acc"]  = df_plot["test_acc"].astype(float)

    if "kernel" in df_plot.columns:
        df_plot["label"] = df_plot.apply(lambda r: f"{r['kernel']} | C={r['C']}", axis=1)
    else:
        df_plot["label"] = df_plot.apply(lambda r: f"{r['penalty']} | C={r['C']}", axis=1)

    fig, ax = plt.subplots(figsize=(max(10, len(df_plot) * 0.6), 5))
    x = np.arange(len(df_plot))
    w = 0.35
    ax.bar(x - w/2, df_plot["train_acc"], w, label="Train Accuracy", color="#4C72B0", alpha=0.85)
    ax.bar(x + w/2, df_plot["test_acc"],  w, label="Test Accuracy",  color="#DD8452", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot["label"], rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_ylabel("Accuracy")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved -> {path}")


def plot_lr_vs_svm(svm_df, lr_df, filename="lr_vs_svm_comparison.png"):
    svm_best = (
        svm_df.groupby("kernel")["test_acc"].max().reset_index()
        .rename(columns={"kernel": "config", "test_acc": "best_acc"})
    )
    svm_best["model"] = "SVM"

    lr_best = (
        lr_df.groupby("penalty")["test_acc"].max().reset_index()
        .rename(columns={"penalty": "config", "test_acc": "best_acc"})
    )
    lr_best["model"] = "LR"

    combined = pd.concat([svm_best, lr_best], ignore_index=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(combined))
    colors = [PALETTE[m] for m in combined["model"]]
    bars = ax.bar(x, combined["best_acc"], color=colors, alpha=0.88, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{row['model']}\n{row['config']}" for _, row in combined.iterrows()], fontsize=9
    )
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_ylabel("Best Test Accuracy")
    ax.set_title("LR vs SVM - Best Accuracy by Configuration", fontsize=13, fontweight="bold")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.1%}", ha="center", va="bottom", fontsize=9)
    ax.legend(handles=[Patch(facecolor=PALETTE["SVM"], label="SVM"), Patch(facecolor=PALETTE["LR"], label="LR")])
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved -> {path}")


def plot_heatmap(df, row_col, col_col, title, filename):
    pivot = df.pivot_table(index=row_col, columns=col_col, values="test_acc", aggfunc="max")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", linewidths=0.5, ax=ax, vmin=0, vmax=1)
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Heatmap saved -> {path}")


def plot_confusion_matrix(y_true, y_pred, class_names, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved -> {path}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))