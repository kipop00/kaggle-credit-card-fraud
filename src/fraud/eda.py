"""Quick pandas profile: python -m fraud.eda"""

import pandas as pd

from fraud.data import TARGET, load_data


def main() -> None:
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 50)
    train, test, _ = load_data()

    print(f"train {train.shape} | test {test.shape}")
    print("\n--- dtypes / missing / unique ---")
    print(
        pd.DataFrame(
            {"dtype": train.dtypes, "missing": train.isna().sum(), "nunique": train.nunique()}
        )
    )
    print("\n--- target balance ---")
    print(train[TARGET].value_counts(normalize=True))
    print("\n--- numeric summary ---")
    print(train.describe().T)
    print("\n--- numeric correlation with target ---")
    print(train.select_dtypes("number").corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False))
    print("\n--- fraud rate by categorical column ---")
    for c in train.select_dtypes(include=["object", "string"]).columns:
        print(f"\n{c}")
        print(train.groupby(c)[TARGET].agg(["mean", "count"]).sort_values("mean", ascending=False).head(10))


if __name__ == "__main__":
    main()
