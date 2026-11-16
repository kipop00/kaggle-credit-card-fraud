"""Data loading helpers."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
SUBMISSIONS = ROOT / "submissions"

ID_COL = "id"
TARGET = "is_fraud"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, test, sample_submission) from data/raw."""
    for name in ("train.csv", "test.csv"):
        if not (RAW / name).exists():
            raise FileNotFoundError(f"{RAW / name} missing - see README 'Setup' to download the data.")
    train = pd.read_csv(RAW / "train.csv")
    test = pd.read_csv(RAW / "test.csv")
    sample = pd.read_csv(RAW / "sample_submission.csv")
    return train, test, sample


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in (ID_COL, TARGET)]


def prepare_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cast object columns to a shared category dtype so tree models can use them."""
    cols = feature_columns(train)
    X, X_test = train[cols].copy(), test[cols].copy()
    for c in X.select_dtypes(include=["object", "string"]).columns:
        cats = pd.Categorical(pd.concat([X[c], X_test[c]])).categories
        X[c] = pd.Categorical(X[c], categories=cats)
        X_test[c] = pd.Categorical(X_test[c], categories=cats)
    return X, X_test
