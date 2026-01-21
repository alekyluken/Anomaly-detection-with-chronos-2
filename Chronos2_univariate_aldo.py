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


def get_pipeline(model_name: str = "amazon/chronos-2", device: str = None):
    """Load Chronos-2 pipeline"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return Chronos2Pipeline.from_pretrained(model_name, device_map=device)


def get_timestamp(start_date: str = "2026-01-01 00:00:00", periods: int = 100, freq: str = 'min'):
    """Generate timestamps for time series"""
    return pd.date_range(start=start_date, periods=periods, freq=freq)


def prepare_data_for_chronos(dataset_path: str, context_length: int = 100):
    """
    Prepare data in Chronos-2 format (DataFrame with timestamp, item_id, target columns)
    
    Returns:
        - time_series_df: Formatted DataFrame for Chronos
        - ground_truth_labels: Anomaly labels
        - actual_future_values: Values to compare against predictions
    """
    # Read CSV
    df = pd.read_csv(dataset_path)
    
    # Extract target column (first column) and labels (last column)
    target_col = df.columns[0]
    label_col = df.columns[-1]
    
    # Get labels
    ground_truth_labels = df[label_col].values
    
    # Remove label from data
    df_clean = df.drop(columns=[label_col]).copy()
    
    # Create Chronos-compatible DataFrame
    df_chronos = pd.DataFrame()
    df_chronos['timestamp'] = get_timestamp(periods=len(df_clean))
    df_chronos['item_id'] = 0  # Single time series
    df_chronos[target_col] = df_clean[target_col].values
    
    return df_chronos, ground_truth_labels, target_col


def make_predictions_sliding_window(time_series_df: pd.DataFrame,pipeline: Chronos2Pipeline,target_col: str,context_length: int = 100,prediction_length: int = 1,step_size: int = 1,batch_size: int = 32):
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
    prediction_indices = []
    
    # Prepare context-future pairs
    contexts = []
    futures = []
    indices = []
    
    #create sliding windows 
    idx = context_length
    while idx + prediction_length <= len(time_series_df):
        # Extract context
        context = time_series_df.iloc[idx - context_length:idx].copy()
        
        # Extract future metadata (timestamp, item_id for next step)
        future = time_series_df[['timestamp', 'item_id']].iloc[idx:idx + prediction_length].copy()
        
        contexts.append(context)
        futures.append(future)
        indices.append(idx)
        
        idx += step_size
    
    print(f"Total prediction windows: {len(contexts)}")
    
    # Process in batches
    for batch_start in range(0, len(contexts), batch_size):
        batch_end = min(batch_start + batch_size, len(contexts)) #l'ultimo batch potrebbe essere più piccolo
        batch_contexts = contexts[batch_start:batch_end]
        batch_futures = futures[batch_start:batch_end]
        batch_indices = indices[batch_start:batch_end]
        
        try:
            # Combine contexts with unique item_id
            combined_contexts = []
            for i, ctx in enumerate(batch_contexts):
                ctx_copy = ctx.copy()
                ctx_copy['item_id'] = i
                combined_contexts.append(ctx_copy)
            combined_context_df = pd.concat(combined_contexts, ignore_index=True)
            
            # Combine futures with matching item_id
            combined_futures = []
            for i, fut in enumerate(batch_futures):
                fut_copy = fut.copy()
                fut_copy['item_id'] = i
                combined_futures.append(fut_copy)
            combined_future_df = pd.concat(combined_futures, ignore_index=True)
            
            # Make predictions
            pred_df = pipeline.predict_df(
                df=combined_context_df,
                future_df=combined_future_df,
                target=target_col,
                prediction_length=prediction_length,
                quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9],  # Use multiple quantiles
                cross_learning=False,
                batch_size=len(batch_contexts),
            )
            
            predictions_list.append(pred_df)
            prediction_indices.extend(batch_indices)
            
            print(f"Batch {batch_start // batch_size + 1}: Processed {len(batch_contexts)} windows")
            
        except Exception as e:
            print(f"Error processing batch starting at index {batch_start}: {e}")

    if predictions_list:
        final_predictions = pd.concat(predictions_list, ignore_index=True)
        return final_predictions, np.array(prediction_indices)
    else:
        return pd.DataFrame(), np.array(prediction_indices)


def detect_anomalies_reconstruction_error(predictions_df: pd.DataFrame,actual_values: np.ndarray,threshold_percentile: float = 90):
    """
    Detect anomalies using reconstruction error (prediction error)
    
    Args:
        predictions_df: DataFrame with '0.5' column (median predictions)
        actual_values: Actual observed values
        threshold_percentile: Percentile for threshold (e.g., 90 = top 10% are anomalies)
    
    Returns:
        anomaly_labels: Binary array (0=normal, 1=anomaly)
        reconstruction_errors: Absolute errors
        threshold: Used threshold
    """
    
    # Get median predictions
    predicted_median = predictions_df['0.5'].values
    
    # Calculate reconstruction error (absolute difference)
    errors = np.abs(actual_values - predicted_median)
    
    # Calculate adaptive threshold
    threshold = np.percentile(errors, threshold_percentile)
    
    # Classify
    anomaly_labels = (errors > threshold).astype(int)
    
    print(f"Threshold: {threshold:.4f}")
    print(f"Anomalies detected: {np.sum(anomaly_labels)} / {len(anomaly_labels)}")
    print(f"Anomaly rate: {np.mean(anomaly_labels):.2%}")
    
    return anomaly_labels, errors, threshold


def evaluate_dataset(dataset_path: str,pipeline: Chronos2Pipeline,context_length: int = 100,threshold_percentile: float = 90,step_size: int = 1,batch_size: int = 32, prediction_length: int = 1):
    """
    Complete evaluation pipeline for a single dataset
    """
    print(f"Processing: {os.path.basename(dataset_path)}")

    
    # Prepare data
    time_series_df, ground_truth_labels, target_col = prepare_data_for_chronos(dataset_path, context_length=context_length)
    
    print(f"Data shape: {time_series_df.shape}")
    print(f"Ground truth anomaly rate: {np.mean(ground_truth_labels):.2%}")
    
    # Make predictions
    predictions_df, prediction_indices = make_predictions_sliding_window(
        time_series_df=time_series_df,
        pipeline=pipeline,
        target_col=target_col,
        context_length=context_length,
        prediction_length=prediction_length,
        step_size=step_size,
        batch_size=batch_size
    )
    
    if len(predictions_df) == 0:
        print("No predictions generated!")
        return None
    
    # Extract actual values corresponding to predictions
    actual_values = time_series_df[target_col].iloc[prediction_indices].values
    corresponding_labels = ground_truth_labels[prediction_indices]
    
    print(f"Predictions: {len(predictions_df)}")
    print(f"Actual values: {len(actual_values)}")
    
    # Detect anomalies using reconstruction error
    predicted_anomalies, errors, threshold = detect_anomalies_reconstruction_error(
        predictions_df=predictions_df,
        actual_values=actual_values,
        threshold_percentile=threshold_percentile
    )
    
    # Calculate metrics
    metrics = get_metrics(predicted_anomalies, corresponding_labels)
    
    print(f"\nMetrics:")
    print(f"  AUC-PR: {metrics.get('AUC-PR', 'N/A'):.4f}")
    print(f"  AUC-ROC: {metrics.get('AUC-ROC', 'N/A'):.4f}")
    print(f"  Standard F1: {metrics.get('Standard-F1', 'N/A'):.4f}")
    print(f"  Event-based F1: {metrics.get('Event-based-F1', 'N/A'):.4f}")
    
    return {
        'file': os.path.basename(dataset_path),
        'metrics': metrics,
        'predictions': predicted_anomalies,
        'actual_labels': corresponding_labels,
        'errors': errors,
        'threshold': threshold
    }


def main():
    """Main execution"""
    
    # Configuration
    data_path = "./TSB-AD-U/TSB-AD-U/"
    out_initial_path = "./results/univariate/"
    
    # Parameters
    context_length = 100
    threshold_percentile = 90  # Top 10% are anomalies
    step_size = 10  
    batch_size = 32
    prediction_length = 1   
    
    output_file_path = os.path.join(out_initial_path, f"result_u_Percentile{threshold_percentile}_step{step_size}_pre{prediction_length}_Context{context_length}_Batch{batch_size}.csv")
    

    pipeline = get_pipeline()
    print(f"Using device: {next(pipeline.model.parameters()).device}")
    
    # Process datasets
    results = []
    for filename in sorted(os.listdir(data_path)):  
        if not filename.endswith('.csv'):
            continue
        
        dataset_file = os.path.join(data_path, filename)
        
        try:
            result = evaluate_dataset(
                dataset_file,
                pipeline=pipeline,
                context_length=context_length,
                threshold_percentile=threshold_percentile,
                step_size=step_size,
                batch_size=batch_size,
                prediction_length=prediction_length
            )
            
            if result is not None:
                results.append(result)
                
                # Save to file
                with open(output_file_path, "a") as f:
                    f.write(f"{result['file']},{result['metrics']}\n")
                    
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    print(f"\n{'='*80}")
    print(f"Processing complete. Results saved to {output_file_path}")
    print(f"Files processed: {len(results)}")


if __name__ == "__main__":
    main()
