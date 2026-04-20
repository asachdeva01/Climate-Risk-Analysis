"""Persist fitted-model artifacts to reports/.

For each model writes three files:
  reports/summary/{model_name}_summary.txt          statsmodels summary or sklearn text summary
  reports/model_outputs/{model_name}_coefficients.csv   coefficient table (with p-values for OLS)
  reports/model_outputs/{model_name}_metrics.json       structured metrics + diagnostics
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np


SUMMARY_DIR = Path("reports/summary")
OUTPUTS_DIR = Path("reports/model_outputs")


def save_model_outputs(model_name: str, model, metrics: dict, diagnostics: dict):
    """Write all three artifacts for the given fitted model."""
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    _save_summary(model_name, model)
    _save_coefficients(model_name, model)
    _save_metrics(model_name, metrics, diagnostics)
    print(f"Saved artifacts for {model_name} to reports/")


def _save_summary(model_name: str, model):
    """Dump the statsmodels .summary() text; for sklearn models dump a minimal stand-in."""
    path = SUMMARY_DIR / f"{model_name}_summary.txt"
    if hasattr(model, 'summary') and callable(getattr(model, 'summary')):
        text = str(model.summary())
    else:
        text = _sklearn_summary(model)
    path.write_text(text)


def _extract_sklearn_estimator(model):
    """If model is a sklearn Pipeline, return the final estimator; else return model."""
    if hasattr(model, 'named_steps'):
        return list(model.named_steps.values())[-1]
    return model


def _sklearn_summary(model) -> str:
    """Minimal text summary for sklearn estimators (Ridge/Lasso), including Pipeline."""
    est = _extract_sklearn_estimator(model)
    lines = [f"{type(est).__name__}"]
    if hasattr(est, 'alpha_'):
        lines.append(f"alpha selected: {est.alpha_}")
    if hasattr(est, 'intercept_'):
        lines.append(f"intercept: {est.intercept_}")
    if hasattr(est, 'coef_'):
        feat_names = getattr(model, 'feature_names_in_', None)
        if feat_names is None:
            feat_names = getattr(est, 'feature_names_in_', None)
        if feat_names is None:
            feat_names = [f"x{i}" for i in range(len(est.coef_))]
        lines.append("coefficients (on standardized inputs):")
        # Sort by |coef| descending to make the summary scannable
        pairs = sorted(zip(feat_names, est.coef_), key=lambda kv: -abs(kv[1]))
        for name, coef in pairs:
            lines.append(f"  {name:<40} {coef:+.6f}")
    return "\n".join(lines)


def _save_coefficients(model_name: str, model):
    """Save a CSV of coefficients (+ p-values / CIs when available)."""
    path = OUTPUTS_DIR / f"{model_name}_coefficients.csv"

    if hasattr(model, 'params'):  # statsmodels
        df = pd.DataFrame({
            'predictor': model.params.index,
            'coefficient': model.params.values,
            'std_err': model.bse.values,
            't_value': model.tvalues.values,
            'p_value': model.pvalues.values,
            'ci_lower': model.conf_int()[0].values,
            'ci_upper': model.conf_int()[1].values,
        })
    else:  # sklearn (possibly Pipeline)
        est = _extract_sklearn_estimator(model)
        feat_names = getattr(model, 'feature_names_in_', None)
        if feat_names is None:
            feat_names = getattr(est, 'feature_names_in_', None)
        if feat_names is None:
            feat_names = [f"x{i}" for i in range(len(est.coef_))]
        df = pd.DataFrame({'predictor': list(feat_names), 'coefficient': est.coef_})
        if hasattr(est, 'intercept_'):
            df = pd.concat([
                pd.DataFrame({'predictor': ['intercept'], 'coefficient': [float(est.intercept_)]}),
                df,
            ], ignore_index=True)

    df.to_csv(path, index=False)


def _save_metrics(model_name: str, metrics: dict, diagnostics: dict):
    """Save metrics + diagnostics as JSON."""
    path = OUTPUTS_DIR / f"{model_name}_metrics.json"
    payload = {'metrics': metrics, 'diagnostics': diagnostics}
    path.write_text(json.dumps(payload, indent=2, default=_json_default))


def _json_default(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")
