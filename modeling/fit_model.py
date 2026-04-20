"""
Modeling Entry Point

Composes the full regression pipeline: split → fit → diagnostics → metrics → save.
One model per helper file (mlr.py, ridge.py, lasso.py). To add a new model,
import its fit function here and register it in MODEL_REGISTRY.

Flags:
    --model         one or more of: mlr, ridge, lasso
    --expanded      add country dummies + year trend to the predictor set
                    (outputs are saved with an `_expanded` suffix so results
                    can be compared side-by-side with the base model)
    --data          path to the input CSV

Usage:
    python -m modeling.fit_model --model mlr
    python -m modeling.fit_model --model mlr --expanded
    python -m modeling.fit_model --model mlr ridge lasso --expanded
"""
import argparse
import pandas as pd

from feature_engineering.add_new_features import (
    engineer_panel_features,
    panel_predictor_columns,
)
from modeling.helpers.mlr import fit_mlr
from modeling.helpers.ridge import fit_ridge
from modeling.helpers.lasso import fit_lasso
from modeling.helpers.validation import train_test_split_panel
from modeling.helpers.metrics import compute_metrics
from modeling.helpers.diagnostics import run_diagnostics
from modeling.helpers.save_outputs import save_model_outputs


MODEL_REGISTRY = {
    'mlr':   fit_mlr,
    'ridge': fit_ridge,
    'lasso': fit_lasso,
}

BASE_PREDICTORS = [
    'heatwave_days', 'drought_index', 'flood_events_count',
    'deforestation_rate', 'fossil_fuel_consumption', 'co2_concentration_ppm',
    'renewable_energy_share', 'forest_cover_percent', 'air_quality_index',
]
TARGET = 'climate_risk_index'


def run_pipeline(df: pd.DataFrame, target: str, predictors: list,
                 model_name: str, save_name: str = None):
    """Run split -> fit -> diagnostics -> metrics -> save for a single model."""
    save_name = save_name or model_name
    X_train, X_test, y_train, y_test = train_test_split_panel(df, target, predictors)

    fit_fn = MODEL_REGISTRY[model_name]
    model = fit_fn(X_train, y_train)

    diagnostics = run_diagnostics(model, X_train, y_train)
    metrics = compute_metrics(model, X_train, y_train, X_test, y_test)

    save_model_outputs(save_name, model, metrics, diagnostics)
    return model, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='+', default=['mlr'],
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument('--expanded', action='store_true',
                        help='Include country dummies + year trend in predictors')
    parser.add_argument('--data', default='data/climate_data.csv')
    args = parser.parse_args()

    df = pd.read_csv(args.data)

    if args.expanded:
        df = engineer_panel_features(df)
        predictors = panel_predictor_columns(df, BASE_PREDICTORS)
        suffix = '_expanded'
    else:
        predictors = BASE_PREDICTORS
        suffix = ''

    print(f"Predictor count: {len(predictors)} | rows: {len(df):,}")

    for name in args.model:
        save_name = f"{name}{suffix}"
        print(f"\n=== Running {save_name.upper()} ===")
        run_pipeline(df, TARGET, predictors, name, save_name=save_name)


if __name__ == "__main__":
    main()
