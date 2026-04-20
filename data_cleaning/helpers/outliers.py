"""IQR-based outlier detection for predictor columns.

Flags outliers by default rather than removing them — regression diagnostics later
may tell us whether to drop, transform, or keep them.
"""
import pandas as pd


def iqr_bounds(series: pd.Series, multiplier: float = 1.5) -> tuple:
    """Calculate IQR-based lower and upper bounds."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - multiplier * iqr, q3 + multiplier * iqr


def flag_outliers(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> pd.DataFrame:
    """Add a boolean '{column}_outlier' flag. Does not remove any rows."""
    lower, upper = iqr_bounds(df[column].dropna(), multiplier)
    flag_col = f"{column}_outlier"
    df[flag_col] = (df[column] < lower) | (df[column] > upper)

    n_outliers = df[flag_col].sum()
    print(f"{column}: {n_outliers:,} outliers flagged "
          f"({n_outliers / len(df) * 100:.1f}%) — bounds [{lower:,.2f}, {upper:,.2f}]")
    return df
