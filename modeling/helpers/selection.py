"""Variable selection procedures for OLS regression.

Stepwise selection by p-value and by AIC. Intended to match the techniques
described in Montgomery's Introduction to Linear Regression Analysis.
"""
import pandas as pd
import statsmodels.api as sm


def forward_selection(X: pd.DataFrame, y: pd.Series, threshold_in: float = 0.05):
    """Forward selection: add the predictor with the lowest p-value until none < threshold_in.

    Returns the list of selected predictor names.
    """
    selected, remaining = [], list(X.columns)
    while remaining:
        pvals = {}
        for col in remaining:
            cols = selected + [col]
            model = sm.OLS(y, sm.add_constant(X[cols], has_constant='add'), missing='drop').fit()
            pvals[col] = model.pvalues[col]
        best_col = min(pvals, key=pvals.get)
        if pvals[best_col] < threshold_in:
            selected.append(best_col)
            remaining.remove(best_col)
            print(f"  + {best_col} (p={pvals[best_col]:.4f})")
        else:
            break
    return selected


def backward_elimination(X: pd.DataFrame, y: pd.Series, threshold_out: float = 0.05):
    """Backward elimination: drop the predictor with the highest p-value > threshold_out."""
    selected = list(X.columns)
    while True:
        model = sm.OLS(y, sm.add_constant(X[selected], has_constant='add'), missing='drop').fit()
        pvals = model.pvalues.drop('const', errors='ignore')
        worst_p = pvals.max()
        if worst_p > threshold_out:
            drop_col = pvals.idxmax()
            selected.remove(drop_col)
            print(f"  - {drop_col} (p={worst_p:.4f})")
        else:
            break
    return selected


def stepwise_selection(X: pd.DataFrame, y: pd.Series,
                       threshold_in: float = 0.05, threshold_out: float = 0.10):
    """Combined forward-backward stepwise selection (standard Montgomery procedure)."""
    selected = []
    while True:
        changed = False
        excluded = [c for c in X.columns if c not in selected]
        new_pvals = {}
        for col in excluded:
            cols = selected + [col]
            model = sm.OLS(y, sm.add_constant(X[cols], has_constant='add'), missing='drop').fit()
            new_pvals[col] = model.pvalues[col]
        if new_pvals and min(new_pvals.values()) < threshold_in:
            best = min(new_pvals, key=new_pvals.get)
            selected.append(best)
            changed = True
            print(f"  + {best} (p={new_pvals[best]:.4f})")

        if selected:
            model = sm.OLS(y, sm.add_constant(X[selected], has_constant='add'), missing='drop').fit()
            pvals = model.pvalues.drop('const', errors='ignore')
            if pvals.max() > threshold_out:
                drop = pvals.idxmax()
                selected.remove(drop)
                changed = True
                print(f"  - {drop} (p={pvals[drop]:.4f})")

        if not changed:
            break
    return selected
