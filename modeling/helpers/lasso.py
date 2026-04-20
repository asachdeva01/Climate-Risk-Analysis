"""Lasso regression (L1 regularization) via scikit-learn.

Provides sparse variable selection as a comparison to stepwise OLS.
Inputs should be standardized before calling.
"""
import pandas as pd
from sklearn.linear_model import LassoCV


def fit_lasso(X: pd.DataFrame, y: pd.Series, alphas=None, cv: int = 5):
    """Fit Lasso with cross-validated alpha selection."""
    model = LassoCV(alphas=alphas, cv=cv, max_iter=10000)
    model.fit(X, y)
    return model
