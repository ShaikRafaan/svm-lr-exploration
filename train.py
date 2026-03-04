"""
train.py
--------
Train SVM and Logistic Regression classifiers with multiple parameter
combinations and collect results for comparison.
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from itertools import product


# ---------------------------------------------------------------------------
# SVM parameter grid
# ---------------------------------------------------------------------------
SVM_PARAM_GRID = {
    "kernel":  ["linear", "rbf", "poly"],
    "C":       [0.01, 0.1, 1, 10, 100],
    "gamma":   ["scale", "auto"],          # only used by rbf / poly
}

# ---------------------------------------------------------------------------
# LR parameter grid
# ---------------------------------------------------------------------------
LR_PARAM_GRID = {
    "penalty":   ["l2", "l1"],
    "C":         [0.01, 0.1, 1, 10, 100],
    "solver":    ["lbfgs", "liblinear"],   # liblinear supports l1
    "max_iter":  [1000],
}


def train_svm_grid(X_train, y_train, X_test, y_test,
                   param_grid: dict = None, cv_folds: int = None,
                   random_state: int = 42) -> pd.DataFrame:
    """
    Train SVM across a grid of parameters and return a results DataFrame.

    Parameters
    ----------
    cv_folds : int or None
        If None, use the provided train/test split.
        If int (e.g. 10), use stratified k-fold cross-validation on X_train.

    Returns
    -------
    pd.DataFrame with columns: kernel, C, gamma, train_acc, test_acc
    """
    if param_grid is None:
        param_grid = SVM_PARAM_GRID

    results = []

    kernels = param_grid["kernel"]
    C_values = param_grid["C"]
    gammas = param_grid["gamma"]

    total = len(kernels) * len(C_values) * len(gammas)
    print(f"\nTraining SVM — {total} combinations...")

    for kernel, C, gamma in product(kernels, C_values, gammas):
        # gamma is irrelevant for linear kernel — skip duplicates
        if kernel == "linear" and gamma != gammas[0]:
            continue

        model = SVC(kernel=kernel, C=C, gamma=gamma,
                    decision_function_shape="ovr", random_state=random_state)

        if cv_folds:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
            test_acc = cv_scores.mean()
            train_acc = None   # not computed for CV
        else:
            model.fit(X_train, y_train)
            train_acc = accuracy_score(y_train, model.predict(X_train))
            test_acc  = accuracy_score(y_test,  model.predict(X_test))

        results.append({
            "model":     "SVM",
            "kernel":    kernel,
            "C":         C,
            "gamma":     gamma if kernel != "linear" else "N/A",
            "train_acc": round(train_acc, 4) if train_acc is not None else "CV",
            "test_acc":  round(test_acc,  4),
        })

    df = pd.DataFrame(results).sort_values("test_acc", ascending=False).reset_index(drop=True)
    print(f"Best SVM  → test_acc={df['test_acc'].iloc[0]:.4f}  "
          f"(kernel={df['kernel'].iloc[0]}, C={df['C'].iloc[0]}, gamma={df['gamma'].iloc[0]})")
    return df


def train_lr_grid(X_train, y_train, X_test, y_test,
                  param_grid: dict = None, cv_folds: int = None,
                  random_state: int = 42) -> pd.DataFrame:
    """
    Train Logistic Regression across a grid of parameters.

    Returns
    -------
    pd.DataFrame with columns: penalty, C, solver, train_acc, test_acc
    """
    if param_grid is None:
        param_grid = LR_PARAM_GRID

    results = []

    penalties = param_grid["penalty"]
    C_values  = param_grid["C"]
    solvers   = param_grid["solver"]
    max_iter  = param_grid["max_iter"][0]

    total = len(penalties) * len(C_values) * len(solvers)
    print(f"\nTraining LR  — {total} combinations...")

    for penalty, C, solver in product(penalties, C_values, solvers):
        # liblinear supports l1; lbfgs does not
        if penalty == "l1" and solver == "lbfgs":
            continue
        # lbfgs does not support l1 either; skip invalid combos
        if penalty == "l2" and solver == "liblinear":
            pass  # valid

        try:
            model = LogisticRegression(
                penalty=penalty, C=C, solver=solver,
                max_iter=max_iter, random_state=random_state
            )

            if cv_folds:
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
                test_acc  = cv_scores.mean()
                train_acc = None
            else:
                model.fit(X_train, y_train)
                train_acc = accuracy_score(y_train, model.predict(X_train))
                test_acc  = accuracy_score(y_test,  model.predict(X_test))

            results.append({
                "model":     "LR",
                "penalty":   penalty,
                "C":         C,
                "solver":    solver,
                "train_acc": round(train_acc, 4) if train_acc is not None else "CV",
                "test_acc":  round(test_acc,  4),
            })

        except Exception:
            pass

    if not results:
        raise RuntimeError("No valid LR parameter combinations produced results. Check your param_grid.")

    df = pd.DataFrame(results).sort_values("test_acc", ascending=False).reset_index(drop=True)
    print(f"Best LR   → test_acc={df['test_acc'].iloc[0]:.4f}  "
          f"(penalty={df['penalty'].iloc[0]}, C={df['C'].iloc[0]}, solver={df['solver'].iloc[0]})")
    return df


if __name__ == "__main__":
    # Quick sanity check
    from preprocess import full_pipeline
    X_train, X_test, y_train, y_test, _, _ = full_pipeline("iris")

    svm_results = train_svm_grid(X_train, y_train, X_test, y_test)
    lr_results  = train_lr_grid(X_train, y_train, X_test, y_test)

    print("\nTop 5 SVM results:")
    print(svm_results.head())
    print("\nTop 5 LR results:")
    print(lr_results.head())