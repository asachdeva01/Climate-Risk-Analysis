"""Lasso regression (L1) via scikit-learn.

Uses a Pipeline so the numeric scale differences across predictors
are standardized before the L1 penalty is applied. Provides sparse
variable selection as a comparison to stepwise OLS.
"""
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV


def fit_lasso(X: pd.DataFrame, y: pd.Series, alphas=None, cv: int = 5):
    """Fit Lasso with cross-validated alpha on standardized inputs."""
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lasso', LassoCV(alphas=alphas, cv=cv, max_iter=20000)),
    ])
    pipe.fit(X, y)
    return pipe
