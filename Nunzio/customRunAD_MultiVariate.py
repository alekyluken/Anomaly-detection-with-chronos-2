import torch, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import argparse


from chronos import Chronos2Pipeline

# Ensure project root is on sys.path so local packages resolve when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from ..metrics.metricsEvaluation import get_metrics
except (ImportError, ModuleNotFoundError):
    from metrics.metricsEvaluation import get_metrics

from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.exceptions import UndefinedMetricWarning

from json import dump as json_dump, load as json_load


import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

if torch.cuda.is_available():
    torch.cuda.empty_cache()


def get_pipeline(model_name: str = "amazon/chronos-2", device: str = None) -> Chronos2Pipeline:
    """Load Chronos-2 pipeline
    
    model_name (str): Pretrained model name
    device (str): Device to run the model on ("cuda", "cpu", or None for auto)

    Returns:
        model (Chronos2Pipeline): Loaded pipeline instance
    """
    return Chronos2Pipeline.from_pretrained(model_name, device_map= device if device else ("cuda" if torch.cuda.is_available() else "cpu"))


def get_timestamp(start_date: str = "2026-01-01 00:00:00", periods: int = 100, freq: str = 'min'):
    """Generate timestamps for time series
    
    Args:
        start_date(str): Starting date for timestamps
        periods(int): Number of timestamps to generate
        freq(str): Frequency of timestamps (e.g., 'min' for minutes)
            
    Returns:
        dates (pd.DatetimeIndex): Generated timestamps
    """
    return pd.date_range(start=start_date, periods=periods, freq=freq)


def prepare_data_for_chronos(dataset_path: str):
    """
    Prepare data in Chronos-2 format (DataFrame with timestamp, item_id, target columns)

    Args:
        dataset_path(str): Path to CSV file with time series data and labels
    
    Returns:
        - time_series_df: Formatted DataFrame for Chronos
        - ground_truth_labels: Anomaly labels
        - actual_future_values: Values to compare against predictions
    """
    # Read CSV
    df = pd.read_csv(dataset_path, header=0, index_col=None)
    
    # Remove label from data
    df_clean = df.drop(columns=[df.columns[-1]])
    
    # Create Chronos-compatible DataFrame
    df_chronos = pd.DataFrame()
    df_chronos['timestamp'] = get_timestamp(periods=len(df_clean))
    df_chronos['item_id'] = 0  # Single time series
    raise ValueError("Check this part carefully, it was modified recently")
    df_chronos[df.columns[0]] = df_clean[df.columns[0]].values
    
    return df_chronos, df[df.columns[-1]].values, df.columns[0]


def dimensionalityReductionViaUMAP(data: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Reduce data dimensionality using UMAP
    
    Args:
        data (np.ndarray): Input data
        n_components (int): Number of dimensions to reduce to
    
    Returns:
        reduced_data (np.ndarray): Dimensionally reduced data
    """
    pass






def main(configuration:dict, name:str)->None:
    pass

if __name__ == "__main__":
    main(None, None)