"""
preprocess.py
-------------
Loads and preprocesses a UCI dataset for use with SVM and LR classifiers.
Supports: Iris, Breast Cancer, Heart Disease (via sklearn or CSV).
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


def load_dataset(name: str = "iris"):
    """
    Load a dataset by name.

    Parameters
    ----------
    name : str
        One of 'iris', 'breast_cancer', or a file path to a CSV.

    Returns
    -------
    X : np.ndarray  — feature matrix
    y : np.ndarray  — label vector
    feature_names : list[str]
    class_names   : list[str]
    """
    if name == "iris":
        data = load_iris()
        return data.data, data.target, list(data.feature_names), list(data.target_names)

    elif name == "breast_cancer":
        data = load_breast_cancer()
        return data.data, data.target, list(data.feature_names), list(data.target_names)

    else:
        # Assume CSV path; last column = target
        df = pd.read_csv(name)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        feature_names = list(df.columns[:-1])
        class_names = [str(c) for c in np.unique(y)]
        return X, y, feature_names, class_names


def handle_missing(X: np.ndarray, strategy: str = "mean") -> np.ndarray:
    """
    Impute missing values using the given strategy ('mean', 'median', 'most_frequent').
    Rows with too many missing values should be dropped before calling this.
    """
    imputer = SimpleImputer(strategy=strategy)
    return imputer.fit_transform(X)


def scale_features(X_train: np.ndarray, X_test: np.ndarray, method: str = "standard"):
    """
    Scale features using StandardScaler or MinMaxScaler.

    Parameters
    ----------
    method : str — 'standard' (zero-mean, unit-variance) or 'minmax' (0–1 range)

    Returns
    -------
    X_train_scaled, X_test_scaled, scaler
    """
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}. Choose 'standard' or 'minmax'.")

    X_train_scaled = scaler.fit_transform(X_train)   # fit only on training data
    X_test_scaled = scaler.transform(X_test)          # apply same transform to test
    return X_train_scaled, X_test_scaled, scaler


def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42):
    """
    Split data into train/test sets with stratification.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def full_pipeline(dataset_name: str = "iris", scale_method: str = "standard", test_size: float = 0.2):
    """
    Convenience function: load → handle missing → split → scale.

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names, class_names
    """
    X, y, feature_names, class_names = load_dataset(dataset_name)

    # Handle missing values if any
    if np.isnan(X).any():
        X = handle_missing(X)

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)
    X_train, X_test, _ = scale_features(X_train, X_test, method=scale_method)

    print(f"Dataset      : {dataset_name}")
    print(f"Total samples: {len(X)}")
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    print(f"Classes      : {class_names}")
    print(f"Features     : {len(feature_names)}")
    return X_train, X_test, y_train, y_test, feature_names, class_names


if __name__ == "__main__":
    full_pipeline("iris")
    full_pipeline("breast_cancer")