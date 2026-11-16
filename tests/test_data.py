import pandas as pd

from fraud.data import feature_columns, prepare_features


def test_feature_columns_excludes_id_and_target():
    df = pd.DataFrame({"id": [1], "a": [2], "is_fraud": [0]})
    assert feature_columns(df) == ["a"]


def test_prepare_features_shares_categories():
    train = pd.DataFrame({"id": [1, 2], "m": ["x", "y"], "is_fraud": [0, 1]})
    test = pd.DataFrame({"id": [3], "m": ["z"]})
    X, X_test = prepare_features(train, test)
    assert list(X["m"].cat.categories) == list(X_test["m"].cat.categories) == ["x", "y", "z"]
