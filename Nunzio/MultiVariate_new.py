import torch, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import argparse

from sklearn.linear_model import Ridge
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


def prepare_data_for_chronos(dataset_path: str, context_length: int = 100, prediction_length: int = 64):
    """
    Prepare data in Chronos-2 format (DataFrame with timestamp, item_id, target columns)

    Args:
        dataset_path(str): Path to CSV file with time series data and labels
        context_length(int): Length of context window (item_id=0)
        prediction_length(int): Length of each prediction segment (item_id=1, 2, 3, ...)
    
    Returns:
        - time_series_df: Formatted DataFrame for Chronos
        - ground_truth_labels: Anomaly labels
        - target_col: Name of the target column
    """
    # Read CSV
    df = pd.read_csv(dataset_path, header=0, index_col=None)

    # Remove label from data
    df_clean = df.drop(columns=[df.columns[-1]])
    item_ids = np.zeros(len(df_clean), dtype=np.int32)
    
    # Samples after context get sequential item_ids based on prediction_length segments
    if context_length < len(df_clean):
        for seg in range((len(df_clean) - context_length + prediction_length - 1) // prediction_length):
            item_ids[context_length + seg * prediction_length:min(context_length + (seg + 1) * prediction_length, len(df_clean))] = seg + 1  # item_id starts from 1 for prediction segments

    # Create Chronos-compatible DataFrame
    df_chronos = pd.DataFrame()
    df_chronos['timestamp'] = pd.date_range(start="2026-01-01 00:00:00", periods=len(df_clean), freq="min")
    df_chronos['item_id'] = item_ids
    df_chronos[df_clean.columns] = df_clean[df_clean.columns].values
    
    return df_chronos, df[df.columns[-1]].values, df_clean.columns.tolist()



def make_predictions_multivariate( time_series_df: pd.DataFrame, pipeline: Chronos2Pipeline, target_cols: list[str],
    context_length: int = 100, prediction_length: int = 64, quantile_levels: list[float] = [0.05, 0.5, 0.95], batch_size: int = 32,
) -> tuple[dict[str, pd.DataFrame], np.ndarray]:
    """
    Generate multivariate predictions where each segment (item_id >= 1) is predicted
    using the context_length values before it. All D columns are used as multivariate target.
    
    Args:
        time_series_df: DataFrame with columns [timestamp, item_id, col1, col2, ..., colD]
        pipeline: Chronos2Pipeline instance
        target_cols: List of target column names (D columns)
        context_length: Number of historical points for context
        prediction_length: Number of steps to forecast per segment
        quantile_levels: Quantile levels for probabilistic forecasts
        batch_size: Number of segments to process in parallel
    
    Returns:
        predictions_dict: Dict mapping each target_col to DataFrame with predictions
        prediction_indices: Array of original indices for each prediction timestep
    
    Note:
        - cross_learning=False ensures different item_ids don't influence each other
        - All D columns are predicted jointly (multivariate forecasting within each segment)
    """
    # Get unique item_ids (excluding 0 which is context-only)
    prediction_item_ids = [iid for iid in sorted(time_series_df['item_id'].unique()) if iid > 0][:-1] # Exclude last segment, since we don't have the labels for it (it would be in the future of the series)
    
    tasks, segment_start_indices = [], []  # Track where each segment starts in original data    
    for item_id in prediction_item_ids:
        segment_start = time_series_df.index[time_series_df['item_id'] == item_id].tolist()[0]
        
        tasks.append({
            "target": time_series_df.loc[segment_start - context_length:segment_start-1, target_cols].values.T.astype(np.float32)
        })
        segment_start_indices.append(segment_start)
    
    # Run predictions in batches
    # cross_learning=False: different segments don't influence each other
    # But within each task, all D columns ARE predicted jointly (multivariate)
    all_predictions = []
    
    for batch_start in tqdm(range(0, len(tasks), batch_size), desc="Predicting segments", leave=False):
        batch_tasks = tasks[batch_start:batch_start + batch_size]
        
        try:
            # predict() returns list of tensors, each of shape (D, n_quantiles, prediction_length)
            all_predictions.extend(pipeline.predict(
                    inputs=batch_tasks,
                    prediction_length=prediction_length,
                    batch_size=len(batch_tasks),
                    context_length=context_length,
                    cross_learning=False,  # Segments don't influence each other
                )
            )
        except Exception as e:
            print(f"Error in batch starting at {batch_start}: {e}")
            raise
    
    # Convert predictions to DataFrames, one per target column
    # all_predictions[i] has shape (D, n_quantiles, prediction_length)
    predictions_dict = {col: [] for col in target_cols}
    all_indices = []
    
    for seg_idx, (pred_tensor, seg_start) in enumerate(zip(all_predictions, segment_start_indices)):
        # pred_tensor shape: (D, n_quantiles, prediction_length)
        pred_np = pred_tensor.cpu().numpy() if hasattr(pred_tensor, 'cpu') else np.array(pred_tensor)
        
        # For each target column (variate)
        for d_idx, col in enumerate(target_cols):
            seg_df = pd.DataFrame({
                'item_id': prediction_item_ids[seg_idx],  # item_id for this segment
                'timestep': np.arange(seg_start, seg_start + prediction_length),
                'predictions': pred_np[d_idx, len(quantile_levels) // 2, :],  # Median as point prediction
            })
            # Add quantile columns
            for q_idx, q_level in enumerate(quantile_levels):
                seg_df[str(q_level)] = pred_np[d_idx, q_idx, :]
            
            predictions_dict[col].append(seg_df)
        
        # Track original indices
        all_indices.extend(range(seg_start, seg_start + prediction_length))
    
    return {col:pd.concat(predictions_dict[col], ignore_index=True) for col in target_cols}, np.array(all_indices, dtype=np.int32)


def computeMultiHorizonAnomalyScore(predictions_df: pd.DataFrame, actual_values: np.ndarray, prediction_indices: np.ndarray,
                                    horizons: list[int] = [1, 8, 32, 64], quantile_col: str = 'predictions'):
    """
    Compute multi-horizon anomaly scores using prediction errors across multiple forecast horizons.
    
    Args:
        predictions_df: DataFrame with predictions and quantiles
        actual_values: Ground truth values
        prediction_indices: Indices in actual_values corresponding to predictions
        target_col: Target column name
        horizons: List of forecast horizons to consider (default [32, 64])
        quantile_col: Quantile column to use (default '0.5' for median). 
                    Can be '0.5', '0.95', '0.05' etc.
    
    Returns:
        Array of anomaly scores (max error across horizons for each prediction)
    """
    all_horizon_scores = []
    cols = predictions_df.columns.tolist()
    
    for h in horizons:
        idx_to_check = prediction_indices + (h - 1)
        mask = idx_to_check < len(actual_values)
        
        if not np.any(mask):
            continue
        elif quantile_col in cols:
            error_h = np.zeros(len(prediction_indices))
            error_h[mask] = (actual_values[idx_to_check[mask]] - predictions_df[quantile_col].to_numpy()[mask])**2
            all_horizon_scores.append(error_h)
    
    if not all_horizon_scores:
        # Fallback to standard squared difference if no horizons match [cite: 344]
        return np.zeros(len(prediction_indices))
        
    return np.max(all_horizon_scores, axis=0)  # [cite: 829]

def get_top_k_jump(scores: np.ndarray, k: int) -> np.ndarray:
    """
    Select top-k anomalies based on anomaly scores, it selects dynamically the top-k, cutting where there is the bigegst jump
    in the sorted scores, to adapt to different anomaly rates across datasets.
    
    Args:
        scores: Array of anomaly scores

    Returns:
        a mask of the same shape as scores, where True indicates selcetd and False indicates normal
    """    
    if len(scores) == 0:
        return np.array([], dtype=bool)
    
    sorted_indices = np.argsort(scores)[::-1]  # Sort in descending order
    sorted_scores = scores[sorted_indices]
    
    # Compute the differences between consecutive sorted scores
    score_diffs = np.diff(sorted_scores)
    
    if len(score_diffs) == 0:
        return np.zeros_like(scores, dtype=bool)
    
    # Find the index of the largest jump in scores
    jump_index = np.argmax(score_diffs)
    
    # Select all indices up to and including the jump index
    selected_indices = sorted_indices[:jump_index + 1]
    
    # Create a boolean mask for selected anomalies
    anomaly_mask = np.zeros_like(scores, dtype=bool)
    anomaly_mask[selected_indices] = True
    
    return anomaly_mask
         

def aggregateAnomalyScores(continuousScores: dict[str, np.ndarray], aggregation_method: str = 'mean', 
                        percentile: float = 95.0, normalization_method: str = 'none',top_k_method: str = 'none') :
    """
    Aggregate anomaly scores across multiple horizons and determine thresholds
    
    Args:
        continuousScores: Dictionary of anomaly scores for each target column
        aggregation_method: How to combine scores ('mean', 'max', 'sum')
        percentile: Percentile threshold for anomaly detection
        normalization_method: How to normalize scores before aggregation ('minmax', 'zscore', 'robust', None)
        top_k_method: Method to select top-k anomalies based on score distribution (none, jump)
    Returns:
        - continuosAnomalyScores: List of aggregated anomaly scores
        - discreteAnomalyScores: List of binary anomaly predictions based on thresholds
    """
    if not continuousScores:
        return np.array([]), []
    
    # Normalize scores per column before aggregation
    normalized_scores = {}
    for col, scores in continuousScores.items():
        if normalization_method == 'minmax':
            # Min-max scaling to [0, 1]
            score_min, score_max = np.min(scores), np.max(scores)
            normalized_scores[col] = (scores - score_min) / (score_max - score_min + 1e-8)
        elif normalization_method == 'zscore':
            # Z-score normalization
            score_mean, score_std = np.mean(scores), np.std(scores)
            normalized_scores[col] = (scores - score_mean) / (score_std + 1e-8)
        elif normalization_method == 'robust':
            # Robust scaling using median and IQR
            score_median = np.median(scores)
            score_iqr = np.percentile(scores, 75) - np.percentile(scores, 25)
            normalized_scores[col] = (scores - score_median) / (score_iqr + 1e-8)
        else:
            # No normalization
            normalized_scores[col] = scores
            
    if top_k_method == 'jump':
        mask = get_top_k_jump(np.concatenate(list(normalized_scores.values())), k=None)
        normalized_scores = {col: scores[mask] for col, scores in normalized_scores.items()}
        
    
    # Stack normalized scores and aggregate
    stacked_scores = np.column_stack(list(normalized_scores.values()))
    match aggregation_method.lower():
        case 'max': aggregated_scores = np.max(stacked_scores, axis=1)
        case 'sum': aggregated_scores = np.sum(stacked_scores, axis=1)
        case _:     aggregated_scores = np.mean(stacked_scores, axis=1)
    
    return aggregated_scores, (aggregated_scores > np.percentile(aggregated_scores, percentile)).astype(np.int8)




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
    
    # Prepare data with proper item_id segmentation
    context_length = configuration.get('context_length', 100)
    horizons = configuration.get('horizons', [32, 64])
    maxH = max(horizons)
    
    time_series_df, ground_truth_labels, target_cols = prepare_data_for_chronos(
        dataset_path, 
        context_length=context_length, 
        prediction_length=maxH
    )
    
    print(f"Ground truth anomaly rate: {np.mean(ground_truth_labels):.2%}")

    # Make multivariate predictions
    # All D columns are predicted jointly, but different segments don't influence each other
    predictions_dict, prediction_indices = make_predictions_multivariate(
        time_series_df=time_series_df,
        pipeline=pipeline,
        target_cols=target_cols,
        context_length=context_length,
        prediction_length=maxH,
        quantile_levels=sorted(set([t for v in configuration.get('thresholds_percentile', [[0.05, 0.95]]) for t in v] + [0.5])),
        batch_size=configuration.get('batch_size', 32),
    )

    # Compute anomaly scores for each column
    continuousScores = {col: computeMultiHorizonAnomalyScore(
            predictions_df=predictions_dict[col],
            actual_values=time_series_df[col].values,
            prediction_indices=prediction_indices,
            horizons=horizons
        ) for col in target_cols}
    
    continuosAnomalyScores, discreteAnomalyScores = aggregateAnomalyScores(
        continuousScores=continuousScores, 
        aggregation_method=configuration.get('aggregation_method', 'mean'),
        percentile=configuration.get('percentile', 95),
        normalization_method=configuration.get('normalization_method', 'none'),
    )

    # Calculate metrics
    return {
        'file': os.path.basename(dataset_path),
        "thresholds": configuration.get('thresholds_percentile', [[0.05, 0.95]]),
        'metrics':[{
            **get_metrics(score=continuosAnomalyScores, labels=ground_truth_labels[prediction_indices], pred=discreteAnomalyScores),
            'accuracy': float(accuracy_score(ground_truth_labels[prediction_indices], discreteAnomalyScores)),
            'precision': float(precision_score(ground_truth_labels[prediction_indices], discreteAnomalyScores, zero_division=0)),
            'recall': float(recall_score(ground_truth_labels[prediction_indices], discreteAnomalyScores, zero_division=0)),
            'f1_score': float(f1_score(ground_truth_labels[prediction_indices], discreteAnomalyScores, zero_division=0)),
            'confusion_matrix': confusion_matrix(ground_truth_labels[prediction_indices], discreteAnomalyScores).tolist(),
            }]
        }



def main(configuration:dict, name:str)->None:
    """Main execution function"""
    # Configuration
    data_path = "./TSB-AD-M/" 
    # data_path = './Nunzio/provaData/multivariate/'
    out_initial_path = f"./results/{name}/MultiVariateNew/"

    os.makedirs(out_initial_path, exist_ok=True)

    # Parameters
    pipeline = get_pipeline(device='cuda')
    print(f"Using device: {next(pipeline.model.parameters()).device}")
    
    save_path = f"results_{max([int(fname.split('_')[1].split('.')[0]) for fname in os.listdir(out_initial_path) if fname.startswith('results_') and fname.endswith('.json')] + [1])}.json"
    
    if os.path.exists(os.path.join(out_initial_path, save_path)):
        with open(os.path.join(out_initial_path, save_path), 'r', encoding='utf-8') as f:
            existing_results = json_load(f)
    else:
        existing_results = {}

    # Process datasets
    dataset_files = sorted(pd.read_csv("test_files_M.csv")["name"].tolist()) if configuration.get('use_restricted_dataset', True) else sorted(filter(lambda x: x.endswith('.csv'), os.listdir(data_path)))

    if all(fname in existing_results for fname in dataset_files):
        existing_results = {}
        save_path = f"results_{int(save_path.split('_')[1].split('.')[0])+1}.json"
        print(f"All files already processed. Switching to new results file: {save_path}")
    else:
        print(f"Continuing with existing results file: {save_path}")


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
            raise e
            # continue
        
        if result is not None:
            with open(os.path.join(out_initial_path, save_path), 'w', encoding='utf-8') as f:
                existing_results[filename] = {**result, **configuration}
                json_dump(existing_results, f, indent=4)
                print(f"\nResults saved for {filename}\n")


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
    args.add_argument('--use_restricted_dataset', action='store_true', default=False, help='Use restricted dataset from test_files_M.csv')
    args.add_argument('--horizons', type=str, help='Comma-separated list of horizons for multi-horizon scoring')
    args.add_argument('--aggregation_method', type=str, default='max', help='Method to aggregate anomaly scores across horizons (mean, max, sum)')
    args.add_argument('--normalization_method', type=str, default='none', help='Method to normalize scores before aggregation (minmax, zscore, robust, none)')
    args.add_argument('--howToEvaluate_u', type=str, default='sum_CRPS', help='Method to evaluate utility for PageRank aggregation (e.g., sum_CRPS, mean_CRPS)')
    args.add_argument('--top_k_method', type=str, default='none', help='Method to select top-k anomalies based on score distribution (none, jump)')
    parsed_args = args.parse_args()

    configuration = {
        'context_length': parsed_args.context_length if parsed_args.context_length > 0 else 100,
        'prediction_length': parsed_args.prediction_length if parsed_args.prediction_length > 0 else 1,
        'step_size': parsed_args.step_size if parsed_args.step_size > 0 else 1,
        'batch_size': parsed_args.batch_size if parsed_args.batch_size > 0 else 256,
        'square_distance': bool(parsed_args.square_distance),
        'use_naive': bool(parsed_args.use_naive),
        'thresholds_percentile': [[float(pair.split('-')[0]), float(pair.split('-')[1])] for pair in parsed_args.thresholds.strip().split(',')] if parsed_args.thresholds else [[0.05, 0.95]],
        'use_restricted_dataset': bool(parsed_args.use_restricted_dataset),
        'horizons': list(map(int, parsed_args.horizons.split(','))) if parsed_args.horizons else [1, 8, 32, 64],
        'aggregation_method': parsed_args.aggregation_method,
        'normalization_method': parsed_args.normalization_method,
        'howToEvaluate_u': parsed_args.howToEvaluate_u,
        'percentile': 95.0,
        'top_k_method': parsed_args.top_k_method,
    }

    
    main(configuration, name=str(parsed_args.user).strip().upper())