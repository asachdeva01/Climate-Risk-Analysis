"""Train/test split helper — dataset-agnostic."""
import pandas as pd
from sklearn.model_selection import train_test_split


def train_test_split_df(df: pd.DataFrame, target: str, predictors: list,
                        test_size: float = 0.2, random_state: int = 42):
    """Random train/test split on the given dataframe."""
    df = df.dropna(subset=[target] + predictors)
    X, y = df[predictors], df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Split: {len(X_train):,} train / {len(X_test):,} test | {len(predictors)} predictors")
    return X_train, X_test, y_train, y_test
