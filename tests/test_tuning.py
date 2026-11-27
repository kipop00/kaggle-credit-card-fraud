import numpy as np
import pandas as pd

from fraud.tuning import cv_auc, expand_grid, param_grid


def test_expand_grid_returns_every_combination():
    combos = expand_grid({"a": [1, 2], "b": ["x", "y", "z"]})
    assert len(combos) == 6
    assert {"a": 2, "b": "z"} in combos


def test_param_grid_is_nonempty_lists():
    grid = param_grid()
    assert grid and all(isinstance(v, list) and v for v in grid.values())


def test_cv_auc_is_between_zero_and_one():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    y = pd.Series((X["a"] + rng.normal(size=200) > 0).astype(int))
    auc = cv_auc({"max_iter": 20}, X, y, n_splits=3)
    assert 0.5 < auc <= 1.0
