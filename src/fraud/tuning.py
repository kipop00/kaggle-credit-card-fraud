"""Grid search over HistGradientBoosting parameters. Run: python -m fraud.tuning"""

from itertools import product

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from fraud.data import TARGET, load_data, prepare_features


def param_grid() -> dict:
    """Candidate values for each HistGradientBoostingClassifier parameter."""
    return {
        "learning_rate": [0.03, 0.1],
        "max_leaf_nodes": [31, 63],
        "min_samples_leaf": [20, 50],
    }


def expand_grid(grid: dict) -> list[dict]:
    """Turn {name: [values]} into a list of every combination."""
    keys = list(grid)
    return [dict(zip(keys, values)) for values in product(*grid.values())]


def cv_auc(params: dict, X, y, n_splits: int = 5, seed: int = 42) -> float:
    """Overall out-of-fold ROC AUC for one parameter set."""
    oof = np.zeros(len(X))
    folds = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    for tr, va in folds.split(X, y):
        model = HistGradientBoostingClassifier(
            categorical_features="from_dtype", early_stopping=True, random_state=seed, **params
        )
        model.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
    return roc_auc_score(y, oof)


def main():
    train, test, _ = load_data()
    X, _ = prepare_features(train, test)
    y = train[TARGET]

    best_params, best_auc = None, -np.inf
    for params in expand_grid(param_grid()):
        auc = cv_auc(params, X, y)
        print(f"{auc:.5f}  {params}")
        if auc > best_auc:
            best_params, best_auc = params, auc

    print(f"best CV AUC: {best_auc:.5f}  {best_params}")


if __name__ == "__main__":
    main()
