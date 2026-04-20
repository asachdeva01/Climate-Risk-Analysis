"""
Data Cleaning & Preprocessing Pipeline

Generic cleaning for any dataframe given a target column:
1. Infer numeric vs categorical columns
2. Drop fully-duplicate rows
3. Report missing values (and drop rows missing the target)
4. Optionally flag outliers on numeric predictors

Designed to be run standalone or imported by the EDA notebook and the model pipeline.
"""
import argparse
import pandas as pd

from data_cleaning.helpers.types import infer_column_types, coerce_numeric
from data_cleaning.helpers.duplicates import drop_duplicate_rows
from data_cleaning.helpers.missing import missing_value_report, drop_missing_target


def preprocess(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Apply dataset-agnostic cleaning steps.

    - Drops exact-duplicate rows
    - Reports missing values per column
    - Drops rows with a missing target value
    """
    types = infer_column_types(df, exclude=[target])
    df = coerce_numeric(df, types['numeric'])
    df = drop_duplicate_rows(df)
    missing_value_report(df, dataset_name="Input")
    df = drop_missing_target(df, target=target)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--target', required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns from {args.data}")
    df = preprocess(df, args.target)
    print(f"Post-cleaning shape: {df.shape}")
