"""
Multivariate Anomaly Detection via Chronos-2 + SimpleQuantileAnomalyDetector
=============================================================================

Inference pipeline using SimpleQuantileAnomalyDetector instead of GNN.

Pipeline:
    1. Load raw multivariate CSV
    2. Segment into Chronos-2 format (item_id based)
    3. Generate multivariate quantile predictions via Chronos-2
    4. Feed (quantiles, values) → SimpleQuantileAnomalyDetector → binary + continuous scores
    5. Evaluate against ground truth

Key changes vs. original MultiVariateGNN.py:
    - Replaced GNN aggregation with SimpleQuantileAnomalyDetector (end-to-end)
    - No separate anomaly score computation per variable + GNN aggregation
    - Direct quantile-based classification: quantiles [B, D, T, Q] + values [B, D, T] → predictions
"""

import torch, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import argparse

from chronos import Chronos2Pipeline
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from sklearn.exceptions import UndefinedMetricWarning
from json import dump as json_dump, load as json_load

import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Ensure project root is on sys.path so local packages resolve
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

_NUNZIO_DIR = os.path.dirname(os.path.abspath(__file__))
if _NUNZIO_DIR not in sys.path:
    sys.path.insert(0, _NUNZIO_DIR)

try:
    from metrics.metricsEvaluation import get_metrics
except (ImportError, ModuleNotFoundError):
    try:
        from ..metrics.metricsEvaluation import get_metrics
    except (ImportError, ModuleNotFoundError):
        get_metrics = None
        print("Warning: metricsEvaluation not found — get_metrics unavailable")

from SimplerNN import SimpleQuantileAnomalyDetector

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════

def get_pipeline(model_name: str = "amazon/chronos-2", device: str = None) -> Chronos2Pipeline:
    """Load Chronos-2 pipeline.

    Args:
        model_name: Pretrained model name
        device: Device ('cuda', 'cpu', or None for auto)

    Returns:
        Chronos2Pipeline instance
    """
    return Chronos2Pipeline.from_pretrained(
        model_name,
        device_map=device if device else ("cuda" if torch.cuda.is_available() else "cpu"),
    )


def get_quantile_detector(
    model_path: str,
    device: str = 'cpu',
) -> tuple:
    """Load trained SimpleQuantileAnomalyDetector from checkpoint.

    Args:
        model_path: Path to saved checkpoint (.pth)
        device: Device to load model on

    Returns:
        (model, threshold): Loaded model in eval mode + optimal threshold
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_cfg = checkpoint.get('config', {})

    model = SimpleQuantileAnomalyDetector(
        in_features=model_cfg.get('in_features', 12),
        hidden_dim=model_cfg.get('hidden_dim', 64),
        kernel_size=model_cfg.get('kernel_size', 7),
        num_layers=model_cfg.get('num_layers', 3),
        num_attention_heads=model_cfg.get('num_attention_heads', 4),
        num_attention_layers=model_cfg.get('num_attention_layers', 4),
        dropout=model_cfg.get('dropout', 0.2),
    ).to(device)

    # Prefer EMA weights if available
    if 'ema_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['ema_state_dict'])
        print(f"  Loaded EMA weights from {model_path}")
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded model weights from {model_path}")
    else:
        model.load_state_dict(checkpoint)
        print(f"  Loaded raw state_dict from {model_path}")

    threshold = checkpoint.get('threshold', 0.5)
    print(f"  Threshold: {threshold:.4f}")

    return model.eval(), threshold


# ═══════════════════════════════════════════════════════════════
# DATA PREPARATION
# ═══════════════════════════════════════════════════════════════

def prepare_data_for_chronos(
    dataset_path: str,
    context_length: int = 100,
    prediction_length: int = 64,
):
    """
    Prepare data in Chronos-2 format (DataFrame with timestamp, item_id, target columns).

    Args:
        dataset_path: Path to CSV file with time series data + labels (last column)
        context_length: Length of context window (item_id=0)
        prediction_length: Length of each prediction segment (item_id=1, 2, 3, ...)

    Returns:
        time_series_df: Formatted DataFrame for Chronos
        ground_truth_labels: Anomaly labels array
        target_cols: List of target column names
    """
    df = pd.read_csv(dataset_path, header=0, index_col=None)

    # Remove label (last column) from data
    df_clean = df.drop(columns=[df.columns[-1]])
    item_ids = np.zeros(len(df_clean), dtype=np.int32)

    if context_length < len(df_clean):
        for seg in range(
            (len(df_clean) - context_length + prediction_length - 1) // prediction_length
        ):
            start = context_length + seg * prediction_length
            end = min(context_length + (seg + 1) * prediction_length, len(df_clean))
            item_ids[start:end] = seg + 1

    df_chronos = pd.DataFrame()
    df_chronos['timestamp'] = pd.date_range(
        start="2026-01-01 00:00:00", periods=len(df_clean), freq="min"
    )
    df_chronos['item_id'] = item_ids
    df_chronos[df_clean.columns] = df_clean[df_clean.columns].values

    return df_chronos, df[df.columns[-1]].values, df_clean.columns.tolist()


# ═══════════════════════════════════════════════════════════════
# CHRONOS-2 PREDICTIONS
# ═══════════════════════════════════════════════════════════════

def make_predictions_multivariate(
    time_series_df: pd.DataFrame,
    pipeline: Chronos2Pipeline,
    target_cols: list[str],
    context_length: int = 100,
    prediction_length: int = 64,
    quantile_levels: list[float] = None,
    batch_size: int = 32,
) -> tuple[dict[str, pd.DataFrame], np.ndarray]:
    """
    Generate multivariate quantile predictions via Chronos-2.

    For each segment (item_id >= 1), predicts using the context_length values before it.
    All D columns are predicted jointly (multivariate forecasting within each segment).

    Args:
        time_series_df: DataFrame [timestamp, item_id, col1, ..., colD]
        pipeline: Chronos2Pipeline instance
        target_cols: List of D target column names
        context_length: Number of historical points for context
        prediction_length: Steps to forecast per segment
        quantile_levels: Quantile levels for probabilistic forecasts
        batch_size: Segments to process in parallel

    Returns:
        predictions_dict: Dict col → DataFrame with quantile predictions
        prediction_indices: Array of original indices for prediction timesteps
    """
    if quantile_levels is None:
        quantile_levels = [0.05, 0.5, 0.95]

    prediction_item_ids = [
        iid for iid in sorted(time_series_df['item_id'].unique()) if iid > 0
    ][:-1]  # Exclude last (future)

    tasks, segment_start_indices = [], []
    for item_id in prediction_item_ids:
        segment_start = time_series_df.index[
            time_series_df['item_id'] == item_id
        ].tolist()[0]
        tasks.append({
            "target": time_series_df.loc[
                segment_start - context_length:segment_start - 1, target_cols
            ].values.T.astype(np.float32),
        })
        segment_start_indices.append(segment_start)

    all_predictions = []
    for batch_start in tqdm(
        range(0, len(tasks), batch_size), desc="Predicting segments", leave=False
    ):
        batch_tasks = tasks[batch_start:batch_start + batch_size]
        try:
            all_predictions.extend(
                pipeline.predict(
                    inputs=batch_tasks,
                    prediction_length=prediction_length,
                    batch_size=len(batch_tasks),
                    context_length=context_length,
                    cross_learning=False,
                )
            )
        except Exception as e:
            print(f"Error in batch starting at {batch_start}: {e}")
            raise

    # Convert to DataFrames
    predictions_dict = {col: [] for col in target_cols}
    all_indices = []

    for seg_idx, (pred_tensor, seg_start) in enumerate(
        zip(all_predictions, segment_start_indices)
    ):
        pred_np = (
            pred_tensor.cpu().numpy()
            if hasattr(pred_tensor, 'cpu')
            else np.array(pred_tensor)
        )

        for d_idx, col in enumerate(target_cols):
            seg_df = pd.DataFrame({
                'item_id': prediction_item_ids[seg_idx],
                'timestep': np.arange(seg_start, seg_start + prediction_length),
                'predictions': pred_np[d_idx, len(quantile_levels) // 2, :],
            })
            for q_idx, q_level in enumerate(quantile_levels):
                seg_df[str(q_level)] = pred_np[d_idx, q_idx, :]
            predictions_dict[col].append(seg_df)

        all_indices.extend(range(seg_start, seg_start + prediction_length))

    return (
        {col: pd.concat(predictions_dict[col], ignore_index=True) for col in target_cols},
        np.array(all_indices, dtype=np.int32),
    )


# ═══════════════════════════════════════════════════════════════
# ANOMALY DETECTION VIA SimpleQuantileAnomalyDetector
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def detect_anomalies_via_quantile_model(
    predictions_dict: dict[str, pd.DataFrame],
    time_series_df: pd.DataFrame,
    prediction_indices: np.ndarray,
    target_cols: list[str],
    quantile_levels: list[float],
    model: SimpleQuantileAnomalyDetector,
    device: torch.device,
    threshold: float = 0.5,
    prediction_length: int = 64,
    batch_size: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run SimpleQuantileAnomalyDetector on Chronos-2 quantile predictions.

    Reshapes per-variable prediction DataFrames into [B, D, T, Q] tensors,
    feeds them through the model, and returns per-timestep scores.

    Args:
        predictions_dict: Dict col → DataFrame with quantile columns
        time_series_df: Original time series DataFrame
        prediction_indices: Array of original indices for predictions
        target_cols: List of D target column names
        quantile_levels: List of Q quantile level floats
        model: Trained SimpleQuantileAnomalyDetector (eval mode)
        device: torch device
        threshold: Binary classification threshold
        prediction_length: Timesteps per segment
        batch_size: Segments per forward pass

    Returns:
        continuous_scores: [N_timesteps] continuous anomaly scores
        binary_predictions: [N_timesteps] binary predictions {0, 1}
    """
    model.eval()
    q_cols = [str(q) for q in sorted(quantile_levels)]
    D = len(target_cols)
    Q = len(quantile_levels)

    # Get unique segment IDs
    first_col = target_cols[0]
    segment_ids = sorted(predictions_dict[first_col]['item_id'].unique())

    # Build per-segment tensors
    all_quantiles = []   # list of [D, T, Q]
    all_values = []      # list of [D, T]
    seg_lengths = []     # actual length of each segment

    for seg_id in segment_ids:
        seg_quantiles = []
        seg_values = []

        for col in target_cols:
            pred_df = predictions_dict[col]
            seg_rows = pred_df[pred_df['item_id'] == seg_id].sort_values('timestep')
            timesteps = seg_rows['timestep'].values.astype(int)

            # Quantile values: [T, Q]
            q_vals = seg_rows[q_cols].values
            seg_quantiles.append(q_vals)

            # Actual values at those timesteps
            actual = time_series_df[col].values[timesteps]
            seg_values.append(actual)

        T_seg = seg_quantiles[0].shape[0]
        seg_lengths.append(T_seg)
        all_quantiles.append(np.stack(seg_quantiles))  # [D, T, Q]
        all_values.append(np.stack(seg_values))          # [D, T]

    # Process in batches (batch segments with same T together)
    # For simplicity, process one segment at a time to handle variable T
    all_continuous = []
    all_binary = []

    for i in tqdm(range(0, len(all_quantiles), batch_size),
                  desc="Detecting anomalies", leave=False):
        # Process one segment at a time (T may vary across segments,
        # but consecutive segments from same file have same T)
        for j in range(i, min(i + batch_size, len(all_quantiles))):
            q_tensor = torch.tensor(
                all_quantiles[j], dtype=torch.float32
            ).unsqueeze(0).to(device)  # [1, D, T, Q]
            v_tensor = torch.tensor(
                all_values[j], dtype=torch.float32
            ).unsqueeze(0).to(device)  # [1, D, T]

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                cont, binary_logits = model(q_tensor, v_tensor)

            all_continuous.append(cont.squeeze(0).squeeze(-1).cpu().numpy())  # [T]
            all_binary.append(
                (torch.sigmoid(binary_logits).squeeze(0).squeeze(-1).cpu().numpy() >= threshold).astype(int)
            )  # [T]

    # Flatten all segments to match prediction_indices
    continuous_scores = np.concatenate(all_continuous)
    binary_predictions = np.concatenate(all_binary)

    return continuous_scores, binary_predictions


# ═══════════════════════════════════════════════════════════════
# EVALUATION PIPELINE
# ═══════════════════════════════════════════════════════════════

def evaluate_dataset(
    dataset_path: str,
    pipeline: Chronos2Pipeline,
    model: SimpleQuantileAnomalyDetector,
    threshold: float,
    configuration: dict,
    device: torch.device = torch.device('cpu'),
) -> dict:
    """
    Complete evaluation pipeline for a single multivariate dataset.

    Args:
        dataset_path: Path to CSV dataset
        pipeline: Chronos2Pipeline instance
        model: Trained SimpleQuantileAnomalyDetector
        threshold: Binary classification threshold
        configuration: Dict with evaluation parameters
        device: torch device

    Returns:
        Dictionary with evaluation metrics
    """
    print(f"Processing: {os.path.basename(dataset_path)}")

    context_length = configuration.get('context_length', 100)
    prediction_length = configuration.get('prediction_length', 64)
    quantile_levels = sorted(set(
        [t for v in configuration.get('thresholds_percentile', [[0.05, 0.95]]) for t in v]
        + [0.5]
    ))

    # 1. Prepare data
    time_series_df, ground_truth_labels, target_cols = prepare_data_for_chronos(
        dataset_path,
        context_length=context_length,
        prediction_length=prediction_length,
    )
    print(f"  Ground truth anomaly rate: {np.mean(ground_truth_labels):.2%}")
    print(f"  Variables (D): {len(target_cols)}, Quantiles (Q): {len(quantile_levels)}")

    # 2. Chronos-2 multivariate predictions
    predictions_dict, prediction_indices = make_predictions_multivariate(
        time_series_df=time_series_df,
        pipeline=pipeline,
        target_cols=target_cols,
        context_length=context_length,
        prediction_length=prediction_length,
        quantile_levels=quantile_levels,
        batch_size=configuration.get('batch_size', 32),
    )

    # 3. Anomaly detection via SimpleQuantileAnomalyDetector
    continuous_scores, binary_predictions = detect_anomalies_via_quantile_model(
        predictions_dict=predictions_dict,
        time_series_df=time_series_df,
        prediction_indices=prediction_indices,
        target_cols=target_cols,
        quantile_levels=quantile_levels,
        model=model,
        device=device,
        threshold=threshold,
        prediction_length=prediction_length,
        batch_size=configuration.get('detector_batch_size', 16),
    )

    # 4. Evaluate against ground truth
    gt_aligned = ground_truth_labels[prediction_indices]

    result = {
        'file': os.path.basename(dataset_path),
        'thresholds': configuration.get('thresholds_percentile', [[0.05, 0.95]]),
        'detector_threshold': float(threshold),
        'metrics': [{
            'accuracy': float(accuracy_score(gt_aligned, binary_predictions)),
            'precision': float(precision_score(gt_aligned, binary_predictions, zero_division=0)),
            'recall': float(recall_score(gt_aligned, binary_predictions, zero_division=0)),
            'f1_score': float(f1_score(gt_aligned, binary_predictions, zero_division=0)),
            'confusion_matrix': confusion_matrix(gt_aligned, binary_predictions).tolist(),
        }],
    }

    # Add extended metrics if available
    if get_metrics is not None:
        try:
            result['metrics'][0].update(
                get_metrics(
                    score=continuous_scores,
                    labels=gt_aligned,
                    pred=binary_predictions,
                )
            )
        except Exception as e:
            print(f"  Warning: get_metrics failed: {e}")

    return result


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main(configuration: dict, name: str) -> None:
    """Main execution function."""
    data_path = "./TSB-AD-M/"
    out_initial_path = f"./results/{name}/MultiVariate/"
    os.makedirs(out_initial_path, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Chronos-2
    pipeline = get_pipeline(device=device)
    print(f"Using device: {next(pipeline.model.parameters()).device}")

    # Load SimpleQuantileAnomalyDetector
    model, threshold = get_quantile_detector(
        model_path=configuration['detector_model_path'],
        device=str(device),
    )
    model = model.to(device)
    print(f"Detector parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Results file management
    save_path = f"results_{max([int(fname.split('_')[1].split('.')[0]) for fname in os.listdir(out_initial_path) if fname.startswith('results_') and fname.endswith('.json')] + [1])}.json"

    if os.path.exists(os.path.join(out_initial_path, save_path)):
        with open(os.path.join(out_initial_path, save_path), 'r', encoding='utf-8') as f:
            existing_results = json_load(f)
    else:
        existing_results = {}

    # Dataset files
    dataset_files = (
        sorted(pd.read_csv("test_files_M.csv")["name"].tolist())
        if configuration.get('use_restricted_dataset', True)
        else sorted(filter(lambda x: x.endswith('.csv'), os.listdir(data_path)))
    )

    if all(fname in existing_results for fname in dataset_files):
        existing_results = {}
        save_path = f"results_{int(save_path.split('_')[1].split('.')[0]) + 1}.json"
        print(f"All files already processed. Switching to new results file: {save_path}")
    else:
        print(f"Continuing with existing results file: {save_path}")

    # Process datasets
    for filename in tqdm(dataset_files, desc="Processing datasets"):
        if filename in existing_results:
            tqdm.write(f"Skipping file: {filename}")
            continue

        tqdm.write(f"Evaluating file: {filename}")
        try:
            result = evaluate_dataset(
                os.path.join(data_path, filename),
                pipeline=pipeline,
                model=model,
                threshold=threshold,
                configuration=configuration,
                device=device,
            )
        except Exception as e:
            tqdm.write(f"Error processing file {filename}: {e}")
            raise e

        if result is not None:
            existing_results[filename] = {**result, **configuration}
            with open(os.path.join(out_initial_path, save_path), 'w', encoding='utf-8') as f:
                json_dump(existing_results, f, indent=4)
            print(f"\nResults saved for {filename}\n")


if __name__ == "__main__":
    args = argparse.ArgumentParser(
        description="Run Chronos-2 + SimpleQuantileAnomalyDetector on multivariate datasets"
    )
    args.add_argument(
        '--user', type=str, required=True,
        choices=['Nunzio', 'Aldo', 'Sara', 'Valentino', 'Simone'],
        help='Username of the person running the script',
    )
    args.add_argument('--context_length', type=int, default=-1)
    args.add_argument('--prediction_length', type=int, default=-1)
    args.add_argument('--batch_size', type=int, default=-1)
    args.add_argument(
        '--thresholds', type=str, default='0.05-0.95',
        help='Comma-separated quantile threshold pairs (e.g. "0.05-0.95,0.1-0.9")',
    )
    args.add_argument('--use_restricted_dataset', action='store_true', default=False)
    args.add_argument(
        '--detector_model_path', type=str,
        default='./SAVED_MODELS_SIMPLE/best_model.pth',
        help='Path to trained SimpleQuantileAnomalyDetector checkpoint',
    )
    args.add_argument('--detector_batch_size', type=int, default=16)
    parsed_args = args.parse_args()

    configuration = {
        'context_length': parsed_args.context_length if parsed_args.context_length > 0 else 100,
        'prediction_length': parsed_args.prediction_length if parsed_args.prediction_length > 0 else 64,
        'batch_size': parsed_args.batch_size if parsed_args.batch_size > 0 else 32,
        'thresholds_percentile': [
            [float(pair.split('-')[0]), float(pair.split('-')[1])]
            for pair in parsed_args.thresholds.strip().split(',')
        ] if parsed_args.thresholds else [[0.05, 0.95]],
        'use_restricted_dataset': bool(parsed_args.use_restricted_dataset),
        'detector_model_path': parsed_args.detector_model_path,
        'detector_batch_size': parsed_args.detector_batch_size,
    }

    main(configuration, name=str(parsed_args.user).strip().upper())
