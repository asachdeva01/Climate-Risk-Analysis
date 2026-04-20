"""Categorical and numeric encodings — dataset-agnostic."""
import pandas as pd


def one_hot_encode(df: pd.DataFrame, columns: list,
                   drop_first: bool = True, dtype: type = int) -> pd.DataFrame:
    """One-hot encode the given categorical columns.

    Returns a new dataframe with the original columns replaced by dummy columns
    named '{col}_{level}'. drop_first=True drops one level per column to avoid
    the dummy trap in OLS.
    """
    if not columns:
        return df
    dummies = pd.get_dummies(df[columns], columns=columns,
                             drop_first=drop_first, dtype=dtype)
    return pd.concat([df.drop(columns=columns), dummies], axis=1)
