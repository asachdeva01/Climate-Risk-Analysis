"""
Feature Engineering Entry Point

One-hot encodes categorical columns so they're usable in OLS/Ridge/Lasso.
Numeric transforms (log, polynomial) and interactions are available as helpers
but are only applied when the EDA tells us they're needed — no default transforms.
"""
import pandas as pd

from feature_engineering.helpers.encodings import one_hot_encode


def encode_for_regression(df: pd.DataFrame, categorical_cols: list,
                          drop_first: bool = True) -> pd.DataFrame:
    """Apply one-hot encoding to categorical columns for regression modeling."""
    return one_hot_encode(df, categorical_cols, drop_first=drop_first)
