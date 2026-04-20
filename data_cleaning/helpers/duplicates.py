"""Duplicate row handling for panel (country x year) data."""
import pandas as pd


def drop_duplicate_rows(df: pd.DataFrame, subset: list = None) -> pd.DataFrame:
    """Drop duplicate rows. For panel data pass subset=['country', 'year']."""
    before = len(df)
    df = df.drop_duplicates(subset=subset).reset_index(drop=True)
    dropped = before - len(df)
    key = "+".join(subset) if subset else "all columns"
    print(f"Dropped {dropped:,} duplicate rows on [{key}] — {before:,} -> {len(df):,}")
    return df
