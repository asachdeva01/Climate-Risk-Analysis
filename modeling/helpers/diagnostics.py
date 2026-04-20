"""Regression diagnostics: linearity, normality, homoscedasticity, multicollinearity.

Produces both numeric test statistics and diagnostic plots. Used after fitting
any OLS model to validate classical regression assumptions.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor


def run_diagnostics(model, X: pd.DataFrame, y: pd.Series) -> dict:
    """Run the full diagnostic battery and return a dict of test results."""
    resid = model.resid if hasattr(model, 'resid') else y - model.predict(X)
    fitted = model.fittedvalues if hasattr(model, 'fittedvalues') else model.predict(X)

    results = {
        'durbin_watson': float(durbin_watson(resid)),
        'shapiro_wilk': _shapiro(resid),
        'breusch_pagan': _breusch_pagan(model, resid),
        'vif': _vif_table(X).to_dict(orient='records'),
    }
    return results


def _shapiro(resid) -> dict:
    """Shapiro-Wilk normality test on residuals."""
    stat, p = stats.shapiro(resid)
    return {'statistic': float(stat), 'p_value': float(p)}


def _breusch_pagan(model, resid) -> dict:
    """Breusch-Pagan test for heteroscedasticity."""
    try:
        lm, lm_p, f, f_p = het_breuschpagan(resid, model.model.exog)
        return {'lm_stat': float(lm), 'lm_p': float(lm_p),
                'f_stat': float(f), 'f_p': float(f_p)}
    except Exception as e:
        return {'error': str(e)}


def _vif_table(X: pd.DataFrame) -> pd.DataFrame:
    """Variance inflation factors for each predictor."""
    X_ = X.dropna().assign(_const=1.0)
    cols = [c for c in X_.columns if c != '_const']
    vifs = [variance_inflation_factor(X_.values, X_.columns.get_loc(c)) for c in cols]
    return pd.DataFrame({'predictor': cols, 'VIF': vifs}).sort_values('VIF', ascending=False)


def plot_residual_diagnostics(model, X: pd.DataFrame, y: pd.Series):
    """Four-panel diagnostic plot: residuals vs fitted, QQ, scale-location, histogram."""
    resid = model.resid if hasattr(model, 'resid') else y - model.predict(X)
    fitted = model.fittedvalues if hasattr(model, 'fittedvalues') else model.predict(X)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].scatter(fitted, resid, alpha=0.4, s=12)
    axes[0, 0].axhline(0, color='red', linestyle='--')
    axes[0, 0].set_xlabel('Fitted')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')

    stats.probplot(resid, dist='norm', plot=axes[0, 1])
    axes[0, 1].set_title('Normal Q-Q')

    std_resid = np.sqrt(np.abs((resid - resid.mean()) / resid.std()))
    axes[1, 0].scatter(fitted, std_resid, alpha=0.4, s=12)
    axes[1, 0].set_xlabel('Fitted')
    axes[1, 0].set_ylabel(r'$\sqrt{|standardized\ residuals|}$')
    axes[1, 0].set_title('Scale-Location')

    axes[1, 1].hist(resid, bins=30, edgecolor='black')
    axes[1, 1].set_xlabel('Residual')
    axes[1, 1].set_title('Residual Histogram')

    plt.tight_layout()
    plt.show()
