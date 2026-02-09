import numpy as np
import pandas as pd
import warnings


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, average_precision_score
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)



def get_metrics(anomalyForecast: np.ndarray, ground_truth_labels: np.ndarray, time_series_df:pd.DataFrame, context_length: int, prediction_length: int) -> dict:
    """
    Compute evaluation metrics for the anomaly forecast.

    Args:
        anomalyForecast: Binary array of shape [num_segments] with predicted anomalies
        ground_truth_labels: Binary array of shape [num_segments] with true anomalies
        time_series_df: Original time series DataFrame (for timestamp alignment)
        context_length: Length of the context window used for forecasting
        prediction_length: Length of the prediction window

    Returns:
        Dictionary with evaluation metrics (e.g., precision, recall, F1-score)
    """
    # Speaking of shapes, anomalyForecast has shape [num_segments], while ground_truth_labels has shape [length time series]. We must elaborate a strategy to align them. 
    v3 = np.array(pd.concat([time_series_df['item_id'], pd.Series(ground_truth_labels, name='anomaly')], axis=1).groupby("item_id")['anomaly'].max().reset_index()['anomaly'].iloc[2:])

    return {
        'accuracy': float(accuracy_score(v3, anomalyForecast)),
        'precision': float(precision_score(v3, anomalyForecast, zero_division=0)),
        'recall': float(recall_score(v3, anomalyForecast, zero_division=0)),
        'f1_score': float(f1_score(v3, anomalyForecast, zero_division=0)),
        "confusion_matrix": confusion_matrix(v3, anomalyForecast).tolist(),
        "auc_pr": float(average_precision_score(v3, anomalyForecast)),
        "gt": v3.tolist(),
        "pred": anomalyForecast.tolist()
    }
