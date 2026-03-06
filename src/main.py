"""
main.py
-------
End-to-end runner for Task 1 (SVM parameter exploration) and
Task 2 (LR vs SVM comparison). Run this script to reproduce all results.

Usage:
    python src/main.py --dataset iris --cv 0
    python src/main.py --dataset breast_cancer --cv 10
"""

import sys
import os
import argparse

# Allow running from repo root
sys.path.insert(0, os.path.dirname(__file__))

from src.preprocess import full_pipeline
from src.train import train_svm_grid, train_lr_grid
from src.evaluate import (
    save_results_table,
    plot_train_vs_test,
    plot_lr_vs_svm,
    plot_heatmap,
    plot_confusion_matrix,
)

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def get_best_model_predictions(df_results, X_train, y_train, X_test, y_test, model_type="svm"):
    """Re-fit the best parameter combo and return predictions."""
    best = df_results.iloc[0]

    if model_type == "svm":
        model = SVC(
            kernel=best["kernel"],
            C=float(best["C"]),
            gamma=best["gamma"] if best["gamma"] != "N/A" else "scale",
            decision_function_shape="ovr",
            random_state=42
        )
    else:
        model = LogisticRegression(
            penalty=best["penalty"],
            C=float(best["C"]),
            solver=best["solver"],
            max_iter=1000,
            random_state=42
        )

    model.fit(X_train, y_train)
    return model.predict(X_test)


def run(dataset_name: str = "iris", cv_folds: int = 0):
    print("\n" + "=" * 60)
    print(f"  SVM vs LR Parameter Exploration — Dataset: {dataset_name.upper()}")
    print("=" * 60)

    # ── 1. Data pipeline ─────────────────────────────────────────
    X_train, X_test, y_train, y_test, feature_names, class_names = full_pipeline(
        dataset_name, scale_method="standard", test_size=0.2
    )
    cv = cv_folds if cv_folds > 1 else None

    # ── 2. TASK 1: SVM parameter grid ────────────────────────────
    print("\n[TASK 1] Exploring SVM parameters...")
    svm_results = train_svm_grid(X_train, y_train, X_test, y_test, cv_folds=cv)
    save_results_table(svm_results, f"svm_results_{dataset_name}.csv")
    plot_train_vs_test(svm_results, f"SVM Train vs Test Accuracy ({dataset_name})",
                       f"svm_train_vs_test_{dataset_name}.png")
    plot_heatmap(svm_results, "C", "kernel",
                 f"SVM Test Accuracy Heatmap ({dataset_name})",
                 f"svm_heatmap_{dataset_name}.png")

    # ── 3. TASK 2: LR parameter grid ─────────────────────────────
    print("\n[TASK 2] Exploring LR parameters...")
    lr_results = train_lr_grid(X_train, y_train, X_test, y_test, cv_folds=cv)
    save_results_table(lr_results, f"lr_results_{dataset_name}.csv")
    plot_train_vs_test(lr_results, f"LR Train vs Test Accuracy ({dataset_name})",
                       f"lr_train_vs_test_{dataset_name}.png")

    # ── 4. LR vs SVM comparison ───────────────────────────────────
    print("\n[COMPARISON] LR vs SVM...")
    plot_lr_vs_svm(svm_results, lr_results, f"lr_vs_svm_comparison_{dataset_name}.png")

    # ── 5. Confusion matrices for best models ─────────────────────
    if cv is None:
        svm_preds = get_best_model_predictions(svm_results, X_train, y_train, X_test, y_test, "svm")
        lr_preds  = get_best_model_predictions(lr_results,  X_train, y_train, X_test, y_test, "lr")

        plot_confusion_matrix(y_test, svm_preds, class_names,
                              f"SVM Confusion Matrix ({dataset_name})",
                              f"svm_confusion_{dataset_name}.png")
        plot_confusion_matrix(y_test, lr_preds, class_names,
                              f"LR Confusion Matrix ({dataset_name})",
                              f"lr_confusion_{dataset_name}.png")

    print("\n" + "=" * 60)
    print("  All results saved to /results/")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SVM/LR parameter exploration")
    parser.add_argument("--dataset", type=str, default="iris",
                        choices=["iris", "breast_cancer"],
                        help="Dataset to use: 'iris' or 'breast_cancer'")
    parser.add_argument("--cv", type=int, default=0,
                        help="Number of CV folds (0 = use 80/20 split, 10 = 10-fold CV)")
    args = parser.parse_args()

    run(dataset_name=args.dataset, cv_folds=args.cv)