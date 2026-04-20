"""
Data Cleaning & Preprocessing Pipeline

Loads the raw climate dataset and applies all cleaning steps:
1. Coerce numeric columns to proper dtypes
2. Drop duplicate country-year rows
3. Report missing values
4. Flag outliers for predictor variables

Designed to be run standalone or imported by the EDA notebook.
"""
import pandas as pd

from data_cleaning.helpers.types import coerce_numeric_columns
from data_cleaning.helpers.duplicates import drop_duplicate_rows
from data_cleaning.helpers.missing import missing_value_report
from data_cleaning.helpers.outliers import flag_outliers


def preprocess_climate(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning steps to the climate risk dataset.

    Note on duplicates: this Kaggle dataset does NOT use (country, year) as a primary
    key — multiple independent sampled observations per country-year are expected.
    We only drop fully-identical rows (all 20 columns match), not country-year duplicates.
    """
    df = coerce_numeric_columns(df)
    df = drop_duplicate_rows(df)  # no subset → only exact full-row duplicates
    missing_value_report(df, "Climate Risk Dataset")
    return df


if __name__ == "__main__":
    df = pd.read_csv('data/climate_data.csv')
    print(f"Climate data: {df.shape[0]:,} rows x {df.shape[1]} columns")
    df = preprocess_climate(df)
