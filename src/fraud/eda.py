"""Quick pandas profile of train/test. Run: python -m fraud.eda"""

import pandas as pd

from fraud.data import TARGET, load_data

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 50)


def main():
    train, test, _ = load_data()
    print(f"train {train.shape} | test {test.shape}")
    print(
        pd.DataFrame(
            {"dtype": train.dtypes, "missing": train.isna().sum(), "nunique": train.nunique()}
        )
    )
    print(train[TARGET].value_counts(normalize=True))
    print(train.describe().T)
    corr = train.corr(numeric_only=True)[TARGET].drop(TARGET)
    print(corr.sort_values(key=abs, ascending=False))
    for c in train.select_dtypes(include=["object", "string"]):
        stats = train.groupby(c)[TARGET].agg(["mean", "count"])
        print(stats.sort_values("mean", ascending=False).head(10))


if __name__ == "__main__":
    main()
