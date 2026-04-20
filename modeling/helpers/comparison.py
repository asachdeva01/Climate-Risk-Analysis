"""Model comparison — rank fitted models by adjusted R²."""
import pandas as pd


def rank_models(results: dict) -> pd.DataFrame:
    """Given {model_name: metrics_dict}, produce a comparison table ranked by adj R².

    Each metrics_dict should have the structure produced by modeling.helpers.metrics.compute_metrics.
    """
    rows = []
    for name, m in results.items():
        train = m.get('train', {})
        test = m.get('test', {})
        rows.append({
            'model': name,
            'train_r2': train.get('r2'),
            'adj_r2': train.get('adj_r2'),
            'test_r2': test.get('r2'),
            'train_rmse': train.get('rmse'),
            'test_rmse': test.get('rmse'),
            'aic': m.get('aic'),
            'bic': m.get('bic'),
            'n_predictors': m.get('p'),
        })
    return pd.DataFrame(rows).sort_values('adj_r2', ascending=False).reset_index(drop=True)


def best_model(ranking: pd.DataFrame) -> str:
    """Return the name of the top-ranked model."""
    return ranking.iloc[0]['model']
