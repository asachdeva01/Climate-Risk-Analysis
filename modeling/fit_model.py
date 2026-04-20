"""
Modeling Entry Point — dataset-agnostic regression pipeline.

Takes any CSV + target column name, auto-infers numeric/categorical predictors,
one-hot encodes categoricals, fits MLR / Ridge / Lasso, runs diagnostics, saves
per-model artifacts, and ranks models by adjusted R².

Usage:
    python -m modeling.fit_model --data data/StudentPerformanceFactors.csv --target Exam_Score
    python -m modeling.fit_model --data data/OTHER.csv --target Y --model mlr
    python -m modeling.fit_model --data data/X.csv --target Y --exclude col1,col2
"""
import argparse
import pandas as pd

from data_cleaning.preprocess import preprocess
from data_cleaning.helpers.types import infer_column_types
from feature_engineering.add_new_features import encode_for_regression
from modeling.helpers.mlr import fit_mlr
from modeling.helpers.ridge import fit_ridge
from modeling.helpers.lasso import fit_lasso
from modeling.helpers.validation import train_test_split_df
from modeling.helpers.metrics import compute_metrics
from modeling.helpers.diagnostics import run_diagnostics
from modeling.helpers.save_outputs import save_model_outputs
from modeling.helpers.comparison import rank_models, best_model


MODEL_REGISTRY = {
    'mlr':   fit_mlr,
    'ridge': fit_ridge,
    'lasso': fit_lasso,
}


def run_pipeline(df: pd.DataFrame, target: str, predictors: list,
                 model_name: str) -> dict:
    """Run split -> fit -> diagnostics -> metrics -> save for one model."""
    X_train, X_test, y_train, y_test = train_test_split_df(df, target, predictors)
    fit_fn = MODEL_REGISTRY[model_name]
    model = fit_fn(X_train, y_train)

    diagnostics = run_diagnostics(model, X_train, y_train)
    metrics = compute_metrics(model, X_train, y_train, X_test, y_test)

    save_model_outputs(model_name, model, metrics, diagnostics)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--model', nargs='+', default=list(MODEL_REGISTRY.keys()),
                        choices=list(MODEL_REGISTRY.keys()),
                        help='Which models to fit. Default: all three.')
    parser.add_argument('--exclude', default='',
                        help='Comma-separated columns to exclude from predictors.')
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns from {args.data}")

    df = preprocess(df, target=args.target)

    exclude = set(c.strip() for c in args.exclude.split(',') if c.strip())
    exclude.add(args.target)
    types = infer_column_types(df, exclude=exclude)
    print(f"Inferred types — numeric: {len(types['numeric'])}, "
          f"categorical: {len(types['categorical'])}")

    df_encoded = encode_for_regression(df, types['categorical'])
    predictors = [c for c in df_encoded.columns if c != args.target and c not in exclude]
    print(f"Final predictor count (post-encoding): {len(predictors)}")

    results = {}
    for name in args.model:
        print(f"\n=== Fitting {name.upper()} ===")
        results[name] = run_pipeline(df_encoded, args.target, predictors, name)

    if len(results) > 1:
        print("\n=== Model Comparison (ranked by adjusted R²) ===")
        ranking = rank_models(results)
        print(ranking.to_string(index=False))
        print(f"\nBest model by adjusted R²: {best_model(ranking).upper()}")


if __name__ == "__main__":
    main()
