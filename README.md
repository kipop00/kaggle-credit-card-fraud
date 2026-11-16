# Kaggle Playground S6E11 — Predicting Credit Card Fraud

Pandas-focused starter for [Playground Series – Season 6, Episode 11](https://www.kaggle.com/competitions/playground-series-s6e11).

- **Goal:** predict the probability of `is_fraud` for each `id` in the test set
- **Metric:** ROC AUC
- **Deadline:** 2026-11-30 (23:59 UTC)

## Setup

```powershell
uv sync
# Kaggle API token: https://www.kaggle.com/settings -> "Create New Token"
# save it as ~/.kaggle/kaggle.json  (or set KAGGLE_API_TOKEN)
# Also accept the competition rules on the website once.
uv run kaggle competitions download -c playground-series-s6e11 -p data/raw
Expand-Archive data/raw/playground-series-s6e11.zip data/raw -Force
```

## Workflow

```powershell
uv run python -m fraud.eda          # quick pandas profile of train/test
uv run python -m fraud.baseline     # 5-fold CV baseline + writes submissions/baseline.csv
uv run kaggle competitions submit -c playground-series-s6e11 -f submissions/baseline.csv -m "baseline"
uv run jupyter lab                  # notebooks/ for exploration
```

## Layout

```
src/fraud/        data loading, features, EDA, baseline model
notebooks/        exploration (keep them thin; move reusable code to src/)
data/raw/         Kaggle downloads (gitignored)
submissions/      generated CSVs (gitignored)
tests/            pytest
```
