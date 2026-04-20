"""Multiple Linear Regression (OLS) via statsmodels.

Uses statsmodels so we get p-values, confidence intervals, and the standard
Montgomery-style regression diagnostics (F-test, t-tests, AIC/BIC).
"""
import pandas as pd
import statsmodels.api as sm


def fit_mlr(X: pd.DataFrame, y: pd.Series):
    """Fit OLS regression and return the fitted statsmodels results object."""
    X_with_const = sm.add_constant(X, has_constant='add')
    model = sm.OLS(y, X_with_const, missing='drop').fit()
    return model
