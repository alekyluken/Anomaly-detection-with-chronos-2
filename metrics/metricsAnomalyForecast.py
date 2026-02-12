import numpy as np
import pandas as pd
import warnings


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, average_precision_score
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)



def get_metrics(anomalyForecast: np.ndarray, ground_truth_labels: np.ndarray, time_series_df:pd.DataFrame) -> dict:
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
    v3 = np.array(pd.concat([time_series_df['item_id'], pd.Series(ground_truth_labels, name='anomaly')], axis=1).groupby("item_id")['anomaly'].max().reset_index()['anomaly'].iloc[2:]).astype(int)

    return {
        'accuracy': float(accuracy_score(v3, anomalyForecast)),
        'precision': float(precision_score(v3, anomalyForecast, zero_division=0)),
        'recall': float(recall_score(v3, anomalyForecast, zero_division=0)),
        'f1_score': float(f1_score(v3, anomalyForecast, zero_division=0)),
        "confusion_matrix": confusion_matrix(v3, anomalyForecast).tolist(),
        "auc_pr": float(average_precision_score(v3, anomalyForecast)),
        "gt": collapseArray(v3.tolist()),
        "pred": collapseArray(anomalyForecast.tolist())
    }


def collapseArray(array:np.ndarray) ->list[list[int, int]]:
    """This function collapses an array of values in segments of consecutive values. For example, if the input is [0, 0, 1, 1, 0, 1], the output will be [[0, 2], [1, 2], [0, 1], [1, 1]], where each sublist contains the value and the count of consecutive occurrences.
    
    Args:
        array: A numpy array of values to be collapsed.

    Returns:
        A list of lists, where each sublist contains a value and its count of consecutive occurrences.
    """
    if len(array) == 0:
        return []

    collapsed = []
    current_value = array[0]
    count = 1

    for i in range(1, len(array)):
        if array[i] == current_value:
            count += 1
        else:
            collapsed.append([current_value, count])
            current_value = array[i]
            count = 1

    # Append the last segment
    collapsed.append([current_value, count])

    return collapsed