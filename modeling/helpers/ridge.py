"""Ridge regression (L2) via scikit-learn.

Uses a Pipeline so the numeric scale differences across predictors
(e.g., Attendance ~80 vs one-hot dummies in {0,1}) get standardized
before the L2 penalty is applied. Without this, the penalty would
effectively only shrink the small-scale dummies.
"""
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV


def fit_ridge(X: pd.DataFrame, y: pd.Series, alphas=None):
    """Fit Ridge with cross-validated alpha on standardized inputs."""
    alphas = alphas if alphas is not None else [0.01, 0.1, 1.0, 10.0, 100.0]
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', RidgeCV(alphas=alphas)),
    ])
    pipe.fit(X, y)
    return pipe
