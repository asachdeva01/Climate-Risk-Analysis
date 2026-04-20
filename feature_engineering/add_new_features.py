"""
Feature Engineering Entry Point

Composes derived features onto the cleaned climate dataset.

- `engineer_features` applies numeric-only transforms (log, polynomial, interactions).
  Currently a no-op — the EDA (uniform predictors, no curvature) gave no reason to transform.
- `engineer_panel_features` adds country one-hot dummies and a year trend — unlocking
  the country and time signal that became usable once the dataset was aggregated to one
  row per (country, year).
"""
import pandas as pd

from feature_engineering.helpers.encodings import one_hot_country, add_year_trend


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply numeric feature engineering steps.

    Currently a no-op: EDA found no predictor needing transformation.
    Kept as an explicit entry point so future transforms can be added here.
    """
    return df


def engineer_panel_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add country one-hot dummies and a year trend feature.

    Returns the dataframe with new columns: `year_trend`, `country_<name>` dummies
    (first country dropped to avoid the dummy trap).
    """
    df = add_year_trend(df)
    df = one_hot_country(df, drop_first=True)
    return df


def panel_predictor_columns(df: pd.DataFrame, base_predictors: list) -> list:
    """Return base_predictors + year_trend + all country_* dummy columns in df."""
    country_cols = sorted(c for c in df.columns if c.startswith('country_'))
    return base_predictors + ['year_trend'] + country_cols
