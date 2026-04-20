"""Column type inference — dataset-agnostic.

Replaces the previous hardcoded NUMERIC_COLUMNS list. Works on any dataframe
by inspecting dtypes, with an option to exclude identifier columns (like
the target or row-id columns).
"""
import pandas as pd
import numpy as np


def infer_column_types(df: pd.DataFrame, exclude: list = None) -> dict:
    """Classify columns as numeric vs categorical based on pandas dtype.

    Returns {'numeric': [...], 'categorical': [...]}.
    Pass exclude=[target, ...] to omit identifier or response columns.
    """
    exclude = set(exclude or [])
    numeric, categorical = [], []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
        else:
            categorical.append(col)
    return {'numeric': numeric, 'categorical': categorical}


def coerce_numeric(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Coerce the given columns to numeric, logging any failures as NaN."""
    for col in columns:
        before_na = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors='coerce')
        after_na = df[col].isna().sum()
        if after_na > before_na:
            print(f"  {col}: {after_na - before_na} values failed numeric conversion")
    return df
