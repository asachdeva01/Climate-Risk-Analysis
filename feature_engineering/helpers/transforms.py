"""Numeric transformations for predictors (log, polynomial, standardization)."""
import numpy as np
import pandas as pd


def add_log_transforms(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Add log(1 + x) transformed columns for right-skewed predictors."""
    for col in columns:
        df[f"log_{col}"] = np.log1p(df[col])
    return df


def add_polynomial_terms(df: pd.DataFrame, columns: list, degree: int = 2) -> pd.DataFrame:
    """Add polynomial terms (x^2, x^3, ...) for predictors with curvature."""
    for col in columns:
        for d in range(2, degree + 1):
            df[f"{col}_pow{d}"] = df[col] ** d
    return df


def standardize(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Z-score standardize columns in place. Useful before Ridge/Lasso."""
    for col in columns:
        mu, sigma = df[col].mean(), df[col].std()
        df[f"{col}_z"] = (df[col] - mu) / sigma
    return df
