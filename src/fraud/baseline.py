"""Stratified 5-fold CV with HistGradientBoosting. Run: python -m fraud.baseline"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from fraud.data import ID_COL, SUBMISSIONS, TARGET, load_data, prepare_features


def main(n_splits=5, seed=42):
    train, test, _ = load_data()
    X, X_test = prepare_features(train, test)
    y = train[TARGET]
    folds = StratifiedKFold(n_splits, shuffle=True, random_state=seed)

    oof, preds = np.zeros(len(X)), np.zeros(len(X_test))
    for fold, (tr, va) in enumerate(folds.split(X, y)):
        model = HistGradientBoostingClassifier(
            categorical_features="from_dtype", early_stopping=True, random_state=seed
        )
        model.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        preds += model.predict_proba(X_test)[:, 1] / n_splits
        print(f"fold {fold}: AUC {roc_auc_score(y.iloc[va], oof[va]):.5f}")

    print(f"CV AUC: {roc_auc_score(y, oof):.5f}")
    SUBMISSIONS.mkdir(exist_ok=True)
    out = pd.DataFrame({ID_COL: test[ID_COL], TARGET: preds})
    out.to_csv(SUBMISSIONS / "baseline.csv", index=False)


if __name__ == "__main__":
    main()
