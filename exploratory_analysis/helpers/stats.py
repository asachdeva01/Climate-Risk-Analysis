"""Statistical summary helpers for EDA.

Functions that compute descriptive statistics, correlations, and
multicollinearity diagnostics. Plotting is separated into visualize.py.
"""
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor


def describe_numeric(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """Extended describe(): count, mean, std, min/max, quartiles, skew, kurtosis."""
    cols = columns or df.select_dtypes(include=np.number).columns.tolist()
    summary = df[cols].describe().T
    summary['skew'] = df[cols].skew()
    summary['kurtosis'] = df[cols].kurtosis()
    summary['missing'] = df[cols].isna().sum()
    return summary


def pearson_correlations(df: pd.DataFrame, target: str, predictors: list = None) -> pd.Series:
    """Pearson correlation of each predictor with the target, sorted by |r|."""
    cols = predictors or df.select_dtypes(include=np.number).columns.drop(target).tolist()
    corr = df[cols + [target]].corr()[target].drop(target)
    return corr.reindex(corr.abs().sort_values(ascending=False).index)


def correlation_matrix(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """Full pairwise correlation matrix for the given columns."""
    cols = columns or df.select_dtypes(include=np.number).columns.tolist()
    return df[cols].corr()


def compute_vif(df: pd.DataFrame, predictors: list) -> pd.DataFrame:
    """Variance Inflation Factor for each predictor — diagnoses multicollinearity.

    Rule of thumb: VIF > 5 is moderate, > 10 is severe.
    """
    X = df[predictors].dropna().assign(_const=1.0)
    vifs = [
        variance_inflation_factor(X.values, i)
        for i in range(len(predictors))
    ]
    return pd.DataFrame({'predictor': predictors, 'VIF': vifs}).sort_values('VIF', ascending=False)
