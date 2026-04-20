"""Ridge regression (L2 regularization) via scikit-learn.

Used to compare against OLS when multicollinearity is present. Inputs should be
standardized before calling (see feature_engineering.helpers.transforms.standardize).
"""
import pandas as pd
from sklearn.linear_model import RidgeCV


def fit_ridge(X: pd.DataFrame, y: pd.Series, alphas=None):
    """Fit Ridge with cross-validated alpha selection."""
    alphas = alphas if alphas is not None else [0.01, 0.1, 1.0, 10.0, 100.0]
    model = RidgeCV(alphas=alphas, store_cv_values=False)
    model.fit(X, y)
    return model
