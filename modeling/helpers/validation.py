"""Train/test split helpers.

For panel data (country x year) we may want to split by year rather than
randomly — splitting by year tests temporal generalization. Default is
random split; pass by_year=True for a temporal holdout.
"""
import pandas as pd
from sklearn.model_selection import train_test_split


def train_test_split_panel(df: pd.DataFrame, target: str, predictors: list,
                           test_size: float = 0.2, random_state: int = 42,
                           by_year: bool = False, holdout_year: int = None):
    """Split into train/test. If by_year=True, hold out rows from holdout_year forward."""
    df = df.dropna(subset=[target] + predictors)

    if by_year:
        cutoff = holdout_year if holdout_year is not None else int(df['year'].quantile(0.8))
        train_df = df[df['year'] < cutoff]
        test_df = df[df['year'] >= cutoff]
        print(f"Temporal split at year {cutoff}: {len(train_df):,} train / {len(test_df):,} test")
        return train_df[predictors], test_df[predictors], train_df[target], test_df[target]

    X, y = df[predictors], df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Random split: {len(X_train):,} train / {len(X_test):,} test")
    return X_train, X_test, y_train, y_test
