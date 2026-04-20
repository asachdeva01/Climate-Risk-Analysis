"""Categorical and temporal encodings (country dummies, year trend)."""
import pandas as pd


def one_hot_country(df: pd.DataFrame, drop_first: bool = True) -> pd.DataFrame:
    """One-hot encode the country column. drop_first=True to avoid the dummy trap."""
    dummies = pd.get_dummies(df['country'], prefix='country', drop_first=drop_first, dtype=int)
    return pd.concat([df, dummies], axis=1)


def add_year_trend(df: pd.DataFrame, base_year: int = None) -> pd.DataFrame:
    """Add a numeric year-trend feature centered on the earliest year (or base_year)."""
    base = base_year if base_year is not None else int(df['year'].min())
    df['year_trend'] = df['year'] - base
    return df
