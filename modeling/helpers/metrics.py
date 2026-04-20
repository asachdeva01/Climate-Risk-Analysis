"""Train/test performance metrics for any fitted regression model.

Returns a dict suitable for JSON serialization and for building model
comparison tables later.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_metrics(model, X_train, y_train, X_test, y_test) -> dict:
    """Compute MSE/MAE/R^2 on both splits plus adjusted R^2, AIC, BIC when available."""
    X_train_pred = _prepare_design(model, X_train)
    X_test_pred = _prepare_design(model, X_test)

    y_train_hat = model.predict(X_train_pred)
    y_test_hat = model.predict(X_test_pred)

    train = _split_metrics(y_train, y_train_hat)
    test = _split_metrics(y_test, y_test_hat)

    n, p = len(y_train), X_train.shape[1]
    train['adj_r2'] = _adj_r2(train['r2'], n, p)

    metrics = {'train': train, 'test': test, 'n_train': n, 'n_test': len(y_test), 'p': p}

    # statsmodels-only
    if hasattr(model, 'aic'):
        metrics['aic'] = float(model.aic)
    if hasattr(model, 'bic'):
        metrics['bic'] = float(model.bic)
    if hasattr(model, 'fvalue'):
        metrics['f_stat'] = float(model.fvalue)
        metrics['f_pvalue'] = float(model.f_pvalue)

    return metrics


def _split_metrics(y_true, y_pred) -> dict:
    return {
        'mse': float(mean_squared_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)),
    }


def _adj_r2(r2: float, n: int, p: int) -> float:
    return float(1 - (1 - r2) * (n - 1) / (n - p - 1))


def _prepare_design(model, X: pd.DataFrame) -> pd.DataFrame:
    """statsmodels OLS expects a constant column; sklearn estimators do not."""
    if hasattr(model, 'model') and 'const' in getattr(model.model, 'exog_names', []):
        import statsmodels.api as sm
        return sm.add_constant(X, has_constant='add')
    return X
