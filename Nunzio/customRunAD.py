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
    df_chronos[df.columns[0]] = df_clean[df.columns[0]].values
    
    return df_chronos, df[df.columns[-1]].values, df.columns[0]


def make_predictions_sliding_window(time_series_df: pd.DataFrame,pipeline: Chronos2Pipeline,target_col: str,context_length: int = 100,prediction_length: int = 1,step_size: int = 1,batch_size: int = 32,
                                    quantile_levels: list[float] = [0.01, 0.05, 0.1, 0.2,  0.5, 0.8, 0.9, 0.95, 0.99]):
    """
    Generate predictions using sliding window approach
    
    Args:
        time_series_df: DataFrame with columns [timestamp, item_id, target]
        pipeline: Chronos2Pipeline instance
        target_col: Name of target column
        context_length: Number of historical points for context
        prediction_length: Number of steps to forecast
        step_size: Stride of sliding window
        batch_size: Batch size for inference
    
    Returns:
        predictions_df: DataFrame with predictions and quantiles
        prediction_indices: Indices in original series corresponding to each prediction
    """
    
    predictions_list = []
    
    # Pre-calculate indices using numpy for efficiency
    indices = np.arange(context_length, len(time_series_df) - prediction_length + 1, step_size)
    
    print(f"Total prediction windows: {len(indices)}")
    
    # Convert target column to numpy array for faster access
    timestamps = time_series_df['timestamp'].values
    
    # Process in batches
    for batch_start in tqdm(range(0, len(indices), batch_size), desc="Processing prediction batches", leave=False):
        batch_indices = indices[batch_start:min(batch_start + batch_size, len(indices))]
        
        try:
            # Build contexts and futures using slicing instead of appending
            combined_contexts_list = []
            combined_futures_list = []
            
            for i, start_idx in enumerate(batch_indices):
                combined_contexts_list.append(pd.DataFrame(
                    {
                        'timestamp': timestamps[start_idx - context_length:start_idx],
                        'item_id': np.full(context_length, i, dtype=np.int32),
                        target_col: time_series_df[target_col].iloc[start_idx - context_length:start_idx].values,
                    }
                ))

                combined_futures_list.append(pd.DataFrame(
                    {
                        'timestamp': timestamps[start_idx:start_idx + prediction_length],
                        'item_id': np.full(prediction_length, i, dtype=np.int32),
                    }
                ))
            
            # Make predictions
            predictions_list.append(pipeline.predict_df(
                df=pd.concat(combined_contexts_list, ignore_index=True),
                future_df=pd.concat(combined_futures_list, ignore_index=True),
                target=target_col,
                prediction_length=prediction_length,
                quantile_levels=quantile_levels,  # Use multiple quantiles
                cross_learning=False,
                batch_size=len(batch_indices),
            ))
            
        except Exception as e:
            print(f"Error processing batch starting at index {batch_start}: {e}")

    # Build prediction_indices efficiently using numpy
    if predictions_list:
        prediction_indices = np.repeat(indices, prediction_length)
        for i in range(len(indices)):
            for j in range(1, prediction_length):
                prediction_indices[i * prediction_length + j] = indices[i] + j

        return pd.concat(predictions_list, ignore_index=True), prediction_indices
    
    return pd.DataFrame(), np.array([], dtype=np.int32)

def computeContinuousAnomalyScores(predictions_df: pd.DataFrame, actual_values: np.ndarray, thresholds_percentile: list[list[float]]):
    """
    Compute continuous anomaly scores (higher = more anomalous)
    
    Strategy: Measure how far actual_values are from the predicted quantile range
    
    Returns:
        scores: List of continuous scores for each threshold pair (higher = more anomalous)
        thresholds_info: Metadata about thresholds
    """
    scores = []
    thresholds_info = []
    
    for q_low, q_high in thresholds_percentile:
        # Get predicted quantiles
        lower_bound = predictions_df[str(q_low)].to_numpy()
        upper_bound = predictions_df[str(q_high)].to_numpy()
        
        # Calculate range (IQR - InterQuantile Range)
        iqr = np.maximum(upper_bound - lower_bound, 1e-6) + 1e-8  # Avoid division by zero
        
        # Calculate distance from range (0 if inside, positive if outside)
        distance_below = np.maximum(0, lower_bound - actual_values)  # How far below lower bound
        distance_above = np.maximum(0, actual_values - upper_bound)  # How far above upper bound

        # Normalized anomaly score (higher = more anomalous)
        # If inside range: small score based on distance from median
        # If outside range: score proportional to how far outside
        raw_score = (distance_below + distance_above) / iqr
        
        # For points inside range, add small penalty based on distance from median
        inside_mask = (actual_values >= lower_bound) & (actual_values <= upper_bound)
        median_distance = np.abs(actual_values - predictions_df['0.5'].to_numpy()) / iqr 
        raw_score[inside_mask] = median_distance[inside_mask] * 0.5  # Scale down
        
        scores.append(raw_score)
        thresholds_info.append({'q_low': q_low, 'q_high': q_high, 'range': f'{q_low}-{q_high}'})
    
    return scores, thresholds_info


def computeContinuousAnomalyScoresNaive(predictions_df: pd.DataFrame, actual_values: np.ndarray, thresholds_percentile: list[list[float]],
                                    square_distance: bool = False):
    """
    Compute continuous anomaly scores (higher = more anomalous)
    
    Strategy: Measure how far actual_values are from the median predicted value

    Args:
        predictions_df: DataFrame with predicted quantiles
        actual_values: Actual observed values
        thresholds_percentile: List of [low, high] quantile pairs
        square_distance: If True, square the distance to emphasize larger deviations
    
    Returns:
        scores: List of continuous scores for each threshold pair (higher = more anomalous)
        thresholds_info: Metadata about thresholds
    """
    scores = []
    thresholds_info = []
    
    for q_low, q_high in thresholds_percentile:
        raw_score = np.abs(actual_values - predictions_df['0.5'].to_numpy()) 
        
        if square_distance:
            raw_score = raw_score ** 2

        scores.append(raw_score)
        thresholds_info.append({'q_low': q_low, 'q_high': q_high, 'range': f'{q_low}-{q_high}'})
    
    return scores, thresholds_info


def computeDiscreteAnomalyScores(predictions_df: pd.DataFrame, actual_values: np.ndarray, thresholds_percentile: list[list[float]]):
    """
    Compute discrete/binary anomaly predictions (0 = normal, 1 = anomaly)
    
    Args:
        score_threshold: If using continuous_scores, values > threshold = anomaly
    
    Returns:
        binary_preds: List of binary arrays for each threshold pair
    """
    return [((actual_values < predictions_df[str(q_low)].to_numpy()) | (actual_values > predictions_df[str(q_high)].to_numpy())).astype(np.int8)
                            for q_low, q_high in thresholds_percentile]


def evaluate_dataset(dataset_path: str,pipeline: Chronos2Pipeline,configuration: dict):
    """
    Complete evaluation pipeline for a single dataset

    Args:
        dataset_path: Path to CSV dataset
        pipeline: Chronos2Pipeline instance
        configuration: Dictionary with evaluation parameters
            - context_length
            - prediction_length
            - step_size
            - batch_size
            - thresholds_percentile

    Returns:
        results: Dictionary with evaluation metrics
    """
    print(f"Processing: {os.path.basename(dataset_path)}")
    
    # Prepare data
    time_series_df, ground_truth_labels, target_col = prepare_data_for_chronos(dataset_path)
    
    print(f"Ground truth anomaly rate: {np.mean(ground_truth_labels):.2%}")
    
    # Make predictions
    predictions_df, prediction_indices = make_predictions_sliding_window(
        time_series_df=time_series_df,
        pipeline=pipeline,
        target_col=target_col,
        context_length=configuration.get('context_length', 100),
        prediction_length=configuration.get('prediction_length', 1),
        step_size=configuration.get('step_size', 1),
        batch_size=configuration.get('batch_size', 256),
        quantile_levels=sorted(set([t for v in configuration.get('thresholds_percentile', [[0.05, 0.95]]) for t in v] + [0.5]))
    )
    
    if not len(predictions_df):
        print("No predictions generated!")
        return None
    
    # Detect anomalies using reconstruction error
    continuosAnomalyScores, th  = computeContinuousAnomalyScoresNaive(
        predictions_df=predictions_df,
        actual_values=time_series_df[target_col].iloc[prediction_indices].values,
        thresholds_percentile=configuration.get('thresholds_percentile', [[0.05, 0.95]]),
        square_distance=configuration.get('square_distance', False)
    ) if configuration.get('use_naive', True) else computeContinuousAnomalyScores(
        predictions_df=predictions_df,
        actual_values=time_series_df[target_col].iloc[prediction_indices].values,
        thresholds_percentile=configuration.get('thresholds_percentile', [[0.05, 0.95]])
    )

    discreteAnomalyScores = computeDiscreteAnomalyScores(
        predictions_df=predictions_df,
        actual_values=time_series_df[target_col].iloc[prediction_indices].values,
        thresholds_percentile=configuration.get('thresholds_percentile', [[0.05, 0.95]])
    )

    # Calculate metrics
    return {
        'file': os.path.basename(dataset_path),
        "thresholds": th,
        'metrics':[{
            **get_metrics(score=cont, labels=ground_truth_labels[prediction_indices], pred=disc),
            'accuracy': float(accuracy_score(ground_truth_labels[prediction_indices], disc)),
            'precision': float(precision_score(ground_truth_labels[prediction_indices], disc, zero_division=0)),
            'recall': float(recall_score(ground_truth_labels[prediction_indices], disc, zero_division=0)),
            'f1_score': float(f1_score(ground_truth_labels[prediction_indices], disc, zero_division=0)),
            'confusion_matrix': confusion_matrix(ground_truth_labels[prediction_indices], disc).tolist(),
            'thresholds': t} for t, cont, disc in zip(th, continuosAnomalyScores, discreteAnomalyScores)]
        }



def main(configuration:dict, name:str)->None:
    """Main execution function"""
    # Configuration
    data_path = "./TSB-AD-U/" 
    out_initial_path = f"./results/{name}/"

    os.makedirs(out_initial_path, exist_ok=True)

    # Parameters
    pipeline = get_pipeline(device='cuda')
    print(f"Using device: {next(pipeline.model.parameters()).device}")
    
    save_path = f"results_{max([int(fname.split('_')[1].split('.')[0]) for fname in os.listdir(out_initial_path) if fname.startswith("results_") and fname.endswith(".json")] + [0]) + 1}.json"

    if os.path.exists(os.path.join(out_initial_path, save_path)):
        with open(os.path.join(out_initial_path, save_path), 'r', encoding='utf-8') as f:
            existing_results = json_load(f)
    else:
        existing_results = {}

    # Process datasets
    dataset_files = sorted(pd.read_csv("test_files_U.csv")["name"].tolist()) if configuration.get('use_restricted_dataset', True) else sorted(filter(lambda x: x.endswith('.csv'), os.listdir(data_path)))

    for filename in tqdm(dataset_files, desc="Processing datasets"):
        if filename in existing_results:
            tqdm.write(f"Skipping file: {filename}")
            continue
        
        tqdm.write(f"Evaluating file: {filename}")
        try:
            result = evaluate_dataset(
                os.path.join(data_path, filename),
                pipeline=pipeline,
                configuration=configuration,
            )
        except Exception as e:
            tqdm.write(f"Error processing file {filename}: {e}")
            continue
        
        if result is not None:
            with open(os.path.join(out_initial_path, save_path), 'w', encoding='utf-8') as f:
                existing_results[filename] = {**result, **configuration}
                json_dump(existing_results, f, indent=4)
                print(f"Results saved for {filename}")


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Run Chronos-2 Anomaly Detection on datasets")
    args.add_argument('--user', type=str, help='Username of the person running the script', 
                        choices=['Nunzio', 'Aldo', 'Sara', 'Valentino', 'Simone'], required=True)
    
    args.add_argument('--context_length', type=int, default=-1, help='Context length for Chronos-2')
    args.add_argument('--prediction_length', type=int, default=-1, help='Prediction length for Chronos-2')
    args.add_argument('--step_size', type=int, default=-1, help='Step size for sliding window')
    args.add_argument('--batch_size', type=int, default=-1, help='Batch size for predictions')
    args.add_argument('--square_distance', action='store_true', default=False, help='Use squared distance for anomaly scoring')
    args.add_argument('--use_naive', action='store_true', default=False, help='Use naive anomaly scoring method')
    args.add_argument('--thresholds', type=str, default='0.05-0.95', help='Comma-separated list of quantile thresholds (e.g., "0.05-0.95,0.1-0.9")')
    args.add_argument('--use_restricted_dataset', action='store_true', default=False, help='Use restricted dataset from test_files_U.csv')
    args.add_argument('--colab', action='store_true', default=False, help='Flag to indicate running in Google Colab environment')

    configuration = {
        'context_length': args.parse_args().context_length if args.parse_args().context_length > 0 else 100,
        'prediction_length': args.parse_args().prediction_length if args.parse_args().prediction_length > 0 else 1,
        'step_size': args.parse_args().step_size if args.parse_args().step_size > 0 else 1,
        'batch_size': args.parse_args().batch_size if args.parse_args().batch_size > 0 else 256,
        'square_distance': bool(args.parse_args().square_distance),
        'use_naive': bool(args.parse_args().use_naive),
        'thresholds_percentile': [[float(pair.split('-')[0]), float(pair.split('-')[1])] for pair in args.parse_args().thresholds.strip().split(',')] if args.parse_args().thresholds else [[0.05, 0.95]],
        'use_restricted_dataset': bool(args.parse_args().use_restricted_dataset),
        'colab': bool(args.parse_args().colab),
    }

    
    main(configuration, name=str(args.parse_args().user).strip().upper())
