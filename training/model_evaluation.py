"""
Model evaluation module: compute metrics (MAE, RMSE, R², MAPE) on a test set or full dataset.
Used to assess model quality and track improvements across experiment runs.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from typing import Any, Dict

from training.asset_rating_model.train_and_export import build_feature_matrix


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute standard regression evaluation metrics.

    Returns dict with:
    - mae: Mean Absolute Error (in EUR)
    - rmse: Root Mean Squared Error (in EUR)
    - r2: R² score (0-1, higher is better)
    - mape: Mean Absolute Percentage Error (0-100%, lower is better)  
    - median_error: Median Absolute Error
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE: percentage error (avoid division by zero)
    nonzero_mask = y_true != 0
    if nonzero_mask.sum() > 0:
        mape = mean_absolute_percentage_error(y_true[nonzero_mask], y_pred[nonzero_mask])
    else:
        mape = np.nan
    
    # Median absolute error
    median_error = np.median(np.abs(y_true - y_pred))
    
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "mape": float(mape),
        "median_error": float(median_error),
    }


def evaluate_model(
    model: Any,
    df: pd.DataFrame,
    target_name: str = "valeur_fonciere",
) -> Dict[str, float]:
    """
    Evaluate the model on a DataFrame. Uses the entire dataset.

    Args:
        model: Trained sklearn model
        df: DataFrame with features and target
        target_name: Name of the target column

    Returns:
        Dict of metrics: mae, rmse, r2, mape, median_error
    """
    X = build_feature_matrix(df)
    y_true = df[target_name].values.astype(np.float64)
    y_pred = model.predict(X)
    
    return compute_metrics(y_true, y_pred)


def evaluate_model_with_split(
    model: Any,
    df: pd.DataFrame,
    test_size: float = 0.2,
    target_name: str = "valeur_fonciere",
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate the model with train/test split.
    
    Returns dict with 'train' and 'test' each containing metrics.
    """
    from sklearn.model_selection import train_test_split
    
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state
    )
    
    df_train = df.iloc[train_idx]
    df_test = df.iloc[test_idx]
    
    train_metrics = evaluate_model(model, df_train, target_name)
    test_metrics = evaluate_model(model, df_test, target_name)
    
    return {
        "train": train_metrics,
        "test": test_metrics,
    }


def compare_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    percentiles: tuple = (5, 25, 50, 75, 95),
) -> Dict[str, Any]:
    """
    Detailed comparison statistics: percentiles of errors, residuals, etc.
    Useful for understanding model behavior across value ranges.
    
    Returns statistics like:
    - error_percentiles: prediction error distribution
    - residual_stats: min, max, mean of residuals
    - price_range: min/max of true and predicted values
    """
    errors = np.abs(y_true - y_pred)
    residuals = y_pred - y_true
    
    error_percentiles = [float(np.percentile(errors, p)) for p in percentiles]
    
    return {
        "error_percentiles": {
            f"p{p}": err for p, err in zip(percentiles, error_percentiles)
        },
        "residual_stats": {
            "min": float(np.min(residuals)),
            "max": float(np.max(residuals)),
            "mean": float(np.mean(residuals)),
            "std": float(np.std(residuals)),
        },
        "price_range": {
            "true_min": float(np.min(y_true)),
            "true_max": float(np.max(y_true)),
            "pred_min": float(np.min(y_pred)),
            "pred_max": float(np.max(y_pred)),
        },
    }
