"""Statistical summary helpers for EDA.

Numeric helpers: descriptive stats, correlation with target, correlation matrix, VIF.
Categorical helpers: group means, one-way ANOVA F-tests.
"""
import pandas as pd
import numpy as np
from scipy import stats as sp_stats
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ---------------------------------------------------------------------------
# Numeric
# ---------------------------------------------------------------------------

def describe_numeric(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """Extended describe(): count, mean, std, quartiles, skew, kurtosis, missing."""
    cols = columns or df.select_dtypes(include=np.number).columns.tolist()
    summary = df[cols].describe().T
    summary['skew'] = df[cols].skew()
    summary['kurtosis'] = df[cols].kurtosis()
    summary['missing'] = df[cols].isna().sum()
    return summary


def pearson_correlations(df: pd.DataFrame, target: str, predictors: list = None) -> pd.Series:
    """Pearson correlation of each numeric predictor with the target, sorted by |r|."""
    cols = predictors or df.select_dtypes(include=np.number).columns.drop(target).tolist()
    corr = df[cols + [target]].corr()[target].drop(target)
    return corr.reindex(corr.abs().sort_values(ascending=False).index)


def correlation_matrix(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """Full pairwise correlation matrix for the given columns."""
    cols = columns or df.select_dtypes(include=np.number).columns.tolist()
    return df[cols].corr()


def compute_vif(df: pd.DataFrame, predictors: list) -> pd.DataFrame:
    """Variance Inflation Factor for each predictor. VIF > 5 is moderate, > 10 severe."""
    X = df[predictors].dropna().assign(_const=1.0)
    vifs = [variance_inflation_factor(X.values, i) for i in range(len(predictors))]
    return pd.DataFrame({'predictor': predictors, 'VIF': vifs}).sort_values('VIF', ascending=False)


# ---------------------------------------------------------------------------
# Categorical
# ---------------------------------------------------------------------------

def group_means(df: pd.DataFrame, categorical: str, target: str) -> pd.DataFrame:
    """Mean and SD of target for each level of a categorical predictor."""
    g = df.groupby(categorical)[target].agg(['count', 'mean', 'std']).round(3)
    g['spread'] = g['mean'].max() - g['mean'].min()
    return g.sort_values('mean', ascending=False)


def anova_f_test(df: pd.DataFrame, categorical: str, target: str) -> dict:
    """One-way ANOVA F-test: do the group means differ significantly?

    A small p-value means the categorical predictor has real signal on the target.
    """
    clean = df[[categorical, target]].dropna()
    groups = [grp[target].values for _, grp in clean.groupby(categorical)]
    if len(groups) < 2:
        return {'f_stat': np.nan, 'p_value': np.nan, 'n_groups': len(groups)}
    f, p = sp_stats.f_oneway(*groups)
    return {'f_stat': float(f), 'p_value': float(p), 'n_groups': len(groups)}


def categorical_signal_report(df: pd.DataFrame, categoricals: list, target: str) -> pd.DataFrame:
    """Rank categorical predictors by ANOVA F-stat (how strongly they separate the target)."""
    rows = []
    for col in categoricals:
        test = anova_f_test(df, col, target)
        spread = group_means(df, col, target)['mean']
        rows.append({
            'predictor': col,
            'n_levels': test['n_groups'],
            'group_mean_min': spread.min() if len(spread) else np.nan,
            'group_mean_max': spread.max() if len(spread) else np.nan,
            'group_mean_range': (spread.max() - spread.min()) if len(spread) else np.nan,
            'anova_f': test['f_stat'],
            'anova_p': test['p_value'],
        })
    return pd.DataFrame(rows).sort_values('anova_f', ascending=False).reset_index(drop=True)
