"""Hyperparameter tuning for the baseline model: python -m fraud.tuning (TODO: stub)"""


def param_grid() -> dict:
    """TODO: return the HistGradientBoostingClassifier parameters to search."""
    raise NotImplementedError


def cv_auc(params: dict, n_splits: int = 5, seed: int = 42) -> float:
    """TODO: return the overall out-of-fold ROC AUC for one parameter set (see baseline.py)."""
    raise NotImplementedError


if __name__ == "__main__":
    # TODO: load data, loop over param_grid(), keep the best params by cv_auc(), print them.
    raise NotImplementedError
