"""Data loading helpers."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW, SUBMISSIONS = ROOT / "data" / "raw", ROOT / "submissions"
ID_COL, TARGET = "id", "is_fraud"


def load_data():
    """Return (train, test, sample_submission) from data/raw."""
    return tuple(pd.read_csv(RAW / f) for f in ("train.csv", "test.csv", "sample_submission.csv"))


def feature_columns(df):
    return [c for c in df.columns if c not in (ID_COL, TARGET)]


def prepare_features(train, test):
    """Cast object columns to a shared category dtype so tree models can use them."""
    cols = feature_columns(train)
    X, X_test = train[cols].copy(), test[cols].copy()
    for c in X.select_dtypes(include=["object", "string"]).columns:
        cats = pd.concat([X[c], X_test[c]]).astype("category").cat.categories
        X[c] = X[c].astype(pd.CategoricalDtype(cats))
        X_test[c] = X_test[c].astype(pd.CategoricalDtype(cats))
    return X, X_test
