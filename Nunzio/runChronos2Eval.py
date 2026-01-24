"""
Chronos-2 Anomaly Detection with Reconstruction Error
Correct implementation for time series anomaly detection using forecasting models
"""

import torch
import os
import numpy as np
import pandas as pd
from chronos import Chronos2Pipeline
from TSB_AD.evaluation.metrics import get_metrics
from time import time as getCurrentTime

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

from json import dump as json_dump, load as json_load

torch.cuda.empty_cache()
print(os.environ)


def get_pipeline(model_name: str = "amazon/chronos-2", device: str = None):
    """Load Chronos-2 pipeline"""
    return Chronos2Pipeline.from_pretrained(model_name, device_map="cuda" if device and torch.cuda.is_available() else "cpu")


def get_timestamp(start_date: str = "2026-01-01 00:00:00", periods: int = 100, freq: str = 'min'):
    """Generate timestamps for time series"""
    return pd.date_range(start=start_date, periods=periods, freq=freq)


def prepare_data_for_chronos(dataset_path: str):
    """
    Prepare data in Chronos-2 format (DataFrame with timestamp, item_id, target columns)
    
    Returns:
        - time_series_df: Formatted DataFrame for Chronos
        - ground_truth_labels: Anomaly labels
        - actual_future_values: Values to compare against predictions
    """
    # Read CSV
    df = pd.read_csv(dataset_path, header=0, index_col=None)
    
    # Remove label from data
    df_clean = df.drop(columns=[df.columns[-1]]).copy()
    
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
    
    predictions_list, prediction_indices = [], []
    
    # Prepare context-future pairs
    contexts, futures, indices = [], [], []
    
    #create sliding windows 
    idx = context_length
    while idx + prediction_length <= len(time_series_df):
        # Extract context
        contexts.append(time_series_df.iloc[idx - context_length:idx].copy())
        
        # Extract future metadata (timestamp, item_id for next step)
        futures.append(time_series_df[['timestamp', 'item_id']].iloc[idx:idx + prediction_length].copy())
        
        indices.append(idx)
        idx += step_size
    
    print(f"Total prediction windows: {len(contexts)}")
    
    # Process in batches
    batch_range = range(0, len(contexts), batch_size)
    for batch_start in tqdm(batch_range, desc="Processing prediction batches", leave=False):
        batch_end = min(batch_start + batch_size, len(contexts)) #l'ultimo batch potrebbe essere più piccolo
        batch_contexts = contexts[batch_start:batch_end]
        
        try:
            # Combine contexts with unique item_id
            combined_contexts = []
            for i, ctx in enumerate(batch_contexts):
                ctx_copy = ctx.copy()
                ctx_copy['item_id'] = i
                combined_contexts.append(ctx_copy)
            
            # Combine futures with matching item_id
            combined_futures = []
            for i, fut in enumerate(futures[batch_start:batch_end]):
                fut_copy = fut.copy()
                fut_copy['item_id'] = i
                combined_futures.append(fut_copy)
            
            # Make predictions
            pred_df = pipeline.predict_df(
                df=pd.concat(combined_contexts, ignore_index=True),
                future_df=pd.concat(combined_futures, ignore_index=True),
                target=target_col,
                prediction_length=prediction_length,
                quantile_levels=quantile_levels,  # Use multiple quantiles
                cross_learning=False,
                batch_size=len(batch_contexts),
            )
            
            predictions_list.append(pred_df)
            
            # Map each prediction row to its corresponding timestep index
            # When prediction_length > 1, each context window produces prediction_length predictions
            for start_idx in indices[batch_start:batch_end]:
                for pred_step in range(prediction_length):
                    prediction_indices.append(start_idx + pred_step)
            
            
        except Exception as e:
            print(f"Error processing batch starting at index {batch_start}: {e}")

    if predictions_list:
        return pd.concat(predictions_list, ignore_index=True), np.array(prediction_indices)
    return pd.DataFrame(), np.array(prediction_indices)


def detect_anomalies_reconstruction_error(predictions_df: pd.DataFrame,actual_values: np.ndarray, thresholds_percentile:list[list[float]] = 
                                            [[0.2, 0.8], [0.01, 0.99],[0.05, 0.95],[0.025, 0.975],[0.1, 0.9]]):
    """
    Detect anomalies using reconstruction error (prediction error)
    
    Args:
        predictions_df: DataFrame with '0.5' column (median predictions)
        actual_values: Actual observed values
        thresholds_percentile: List of [lower_percentile, upper_percentile] pairs for thresholding
        
    Returns:
        anomaly_labels: Binary array (0=normal, 1=anomaly)
        reconstruction_errors: Absolute errors
        threshold: Used threshold
    """
    return ([((actual_values < predictions_df[str(q_low)].to_numpy()) | (actual_values > predictions_df[str(q_high)].to_numpy())).astype(np.int8) 
            for q_low, q_high in thresholds_percentile], thresholds_percentile)



def evaluate_dataset(dataset_path: str,pipeline: Chronos2Pipeline,context_length: int = 100,
        thresholds_percentile: list[list[float]] = [[0.2, 0.8], [0.01, 0.99], [0.05, 0.95], [0.025, 0.975],[0.1, 0.9]],
        step_size: int = 1,batch_size: int = 32, prediction_length: int = 1):
    """
    Complete evaluation pipeline for a single dataset
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
        context_length=context_length,
        prediction_length=prediction_length,
        step_size=step_size,
        batch_size=batch_size,
        quantile_levels=[t for v in thresholds_percentile for t in v] + [0.5] 
    )
    
    if not len(predictions_df):
        print("No predictions generated!")
        return None
    
    # Detect anomalies using reconstruction error
    predictedAnomalies, th  = detect_anomalies_reconstruction_error(
        predictions_df=predictions_df,
        actual_values=time_series_df[target_col].iloc[prediction_indices].values,
        thresholds_percentile=thresholds_percentile
    )

    # Calculate metrics
    return {
        'file': os.path.basename(dataset_path),
        "thresholds": th,
        'metrics':[{
            **get_metrics(predicted, ground_truth_labels[prediction_indices]),
            'accuracy': float(accuracy_score(ground_truth_labels[prediction_indices], predicted)),
            'precision': float(precision_score(ground_truth_labels[prediction_indices], predicted, zero_division=0)),
            'recall': float(recall_score(ground_truth_labels[prediction_indices], predicted, zero_division=0)),
            'f1_score': float(f1_score(ground_truth_labels[prediction_indices], predicted, zero_division=0)),
            'confusion_matrix': confusion_matrix(ground_truth_labels[prediction_indices], predicted).tolist(),
            'thresholds': t} for t, predicted in zip(th, predictedAnomalies)]
        }


def main():
    """Main execution"""
    
    # Configuration
    data_path = "./TSB-AD-U/" #aldo
    # data_path = "./Nunzio/provaData/"
    out_initial_path = "./Nunzio/results/univariate/"
    
    os.makedirs(out_initial_path, exist_ok=True)

    # Parameters
    context_length = 100
    thresholds_percentile = [[0.2, 0.8], [0.1, 0.9], [0.05, 0.95], [0.025, 0.975], [0.01, 0.99]]
    step_size = 1 
    batch_size = 256
    prediction_length = 1

    pipeline = get_pipeline(device='cuda')
    print(f"Using device: {next(pipeline.model.parameters()).device}")
    
    save_path = f"result_con{context_length}_pred{prediction_length}_step{step_size}_batch{batch_size}.json"
    
    if os.path.exists(os.path.join(out_initial_path, save_path)):
        with open(os.path.join(out_initial_path, save_path), 'r', encoding='utf-8') as f:
            existing_results = json_load(f)
    else:
        existing_results = {}

    # Process datasets
    dataset_files = [f for f in sorted(os.listdir(data_path)) if f.endswith('.csv')]
    for filename in tqdm(dataset_files, desc="Processing datasets"):
        if filename in existing_results:
            tqdm.write(f"Skipping file: {filename}")
            continue
        
        tqdm.write(f"Evaluating file: {filename}")
        
        result = evaluate_dataset(
            os.path.join(data_path, filename),
            pipeline=pipeline,
            context_length=context_length,
            thresholds_percentile=thresholds_percentile,
            step_size=step_size,
            batch_size=batch_size,
            prediction_length=prediction_length
        )
        
        if result is not None:
            with open(os.path.join(out_initial_path, save_path), 'w', encoding='utf-8') as f:
                existing_results[filename] = {**result, 'context_length': context_length, 'prediction_length': prediction_length,
                            'step_size': step_size, "batch_size": batch_size}
                json_dump(existing_results, f, indent=4)
                print(f"Results saved for {filename}")


if __name__ == "__main__":
    main()
