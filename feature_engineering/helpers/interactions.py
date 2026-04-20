"""Pairwise interaction term generation."""
import pandas as pd


def add_interaction_terms(df: pd.DataFrame, pairs: list) -> pd.DataFrame:
    """Create multiplicative interaction columns for each (col_a, col_b) pair.

    Pairs should be chosen based on EDA / domain intuition (e.g., heatwave_days x drought_index).
    """
    for a, b in pairs:
        df[f"{a}_x_{b}"] = df[a] * df[b]
    return df
