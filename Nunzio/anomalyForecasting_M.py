"""
Multivariate Anomaly Forecasting with Chronos-2 + Two-Stage Detector
=====================================================================

Pipeline:
  1. Load Chronos-2 pipeline for embedding extraction
  2. Load TwoStageMultivariateDetector (Stage 1: per-series, Stage 2: set-based aggregator)
  3. For each multivariate dataset (D columns):
     a. Prepare data → item_id segmentation per column
     b. Extract Chronos-2 embeddings per column per segment → [num_segments, D, 9, 768]
     c. Run two-stage model → global anomaly score per segment
     d. Compute metrics against ground truth
"""

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
    from .KIMI_RAIKKONEN import ChronosAnomalyDetector
    from .two_stage_detector import TwoStageMultivariateDetector
except (ImportError, ModuleNotFoundError):
    from metrics.metricsAnomalyForecast import get_metrics
    from KIMI_RAIKKONEN import ChronosAnomalyDetector
    from two_stage_detector import TwoStageMultivariateDetector

from tqdm import tqdm
from json import dump as json_dump, load as json_load


if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════

def getModel(model_path: str,stage1_checkpoint: str = None,device: str = "cpu",context_length: int = 128) -> tuple:
    """Load TwoStageMultivariateDetector from a saved two-stage checkpoint directory.

    The directory must contain:
      - config.json          (architecture and training config)
      - best_model.pth       (model weights + val_metrics with optimal threshold)
      - training_history.json (per-epoch logs, used as fallback for threshold)

    Args:
        model_path: Path to the two-stage model directory (e.g. ./Saved_Models/two_stage/)
        stage1_checkpoint: Optional override for Stage 1 checkpoint path.
                           If None, uses the path stored in config.json.
        device: Device string
        context_length: Context length (controls numPatches for Stage 1)

    Returns:
        (model, threshold): Ready-to-use model in eval mode and optimal binary threshold
    """
    config = json_load(open(os.path.join(model_path, "config.json"), "r", encoding="utf-8"))
    checkpoint = torch.load(os.path.join(model_path, "best_model.pth"), map_location=device, weights_only=False)

    # ── Build Stage 1 ──
    stage1_model = ChronosAnomalyDetector(
        embed_dim=768,
        hidden_dim=config["stage1_hidden_dim"],
        num_heads=4,
        dropout=config.get("dropout", 0.15),
        numPatches=int(np.ceil(context_length / 16) + 2),
    )

    # ── Build Two-Stage model ──
    model = TwoStageMultivariateDetector(
        stage1_model=stage1_model,
        stage1_hidden_dim=config["stage1_hidden_dim"],
        stage2_hidden_dim=config["stage2_hidden_dim"],
        num_isab_layers=config.get("num_isab_layers", 1),
        num_heads=config.get("num_heads", 2),
        freeze_stage1=True,
        dropout=config.get("dropout", 0.3),
    ).to(device)

    # Load full two-stage weights (Stage 1 + Stage 2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Two-stage model loaded from {model_path}")

    if stage1_checkpoint is not None:
        # Load Stage 1 weights only (if provided)
        stage1_weights = torch.load(stage1_checkpoint, map_location=device, weights_only=True)
        model.stage1.load_state_dict(stage1_weights)
        print(f"Stage 1 weights loaded from {stage1_checkpoint}")

    # ── Extract optimal threshold ──
    # Priority 1: from checkpoint's val_metrics
    threshold = 0.5
    if "val_metrics" in checkpoint and "global_optimal_threshold" in checkpoint["val_metrics"]:
        threshold = checkpoint["val_metrics"]["global_optimal_threshold"]
        print(f"  Threshold from best checkpoint: {threshold:.4f}")
    else:
        # Priority 2: from training_history.json (best epoch)
        history_path = os.path.join(model_path, "training_history.json")
        if os.path.exists(history_path):
            history = json_load(open(history_path, "r", encoding="utf-8"))
            best_epoch = max(history, key=lambda e: e["val"].get("global_optimal_f1", 0))
            threshold = best_epoch["val"].get("global_optimal_threshold", 0.5)
            print(f"  Threshold from training history (epoch {best_epoch['epoch']}): {threshold:.4f}")

    return model, threshold


# ═══════════════════════════════════════════════════════════════
# DATA PREPARATION
# ═══════════════════════════════════════════════════════════════

def prepare_data_for_chronos(dataset_path: str, context_length: int = 100, prediction_length: int = 64):
    """
    Prepare multivariate data in Chronos-2 format.

    Args:
        dataset_path: Path to CSV file (columns: [col1, col2, ..., colD, label])
        context_length: Length of context window (item_id=0)
        prediction_length: Length of each prediction segment (item_id=1, 2, ...)

    Returns:
        time_series_df: DataFrame with [timestamp, item_id, col1, ..., colD]
        ground_truth_labels: Binary anomaly labels array
        target_cols: List of column names (excluding label)
    """
    df = pd.read_csv(dataset_path, header=0, index_col=None)

    # Last column is always the label
    df_clean = df.drop(columns=[df.columns[-1]])
    ground_truth_labels = df[df.columns[-1]].values

    # Assign item_ids: 0=context, 1..K=prediction segments
    item_ids = np.zeros(len(df_clean), dtype=np.int32)
    if context_length < len(df_clean):
        for seg in range((len(df_clean) - context_length + prediction_length - 1) // prediction_length):
            start = context_length + seg * prediction_length
            end = min(context_length + (seg + 1) * prediction_length, len(df_clean))
            item_ids[start:end] = seg + 1

    # Build Chronos-compatible DataFrame
    df_chronos = pd.DataFrame()
    df_chronos["timestamp"] = pd.date_range(start="2026-01-01 00:00:00", periods=len(df_clean), freq="min")
    df_chronos["item_id"] = item_ids
    df_chronos[df_clean.columns] = df_clean.values

    return df_chronos, ground_truth_labels, df_clean.columns.tolist()


# ═══════════════════════════════════════════════════════════════
# EMBEDDING EXTRACTION
# ═══════════════════════════════════════════════════════════════

def getMultivariateChronos2Embeddings(
    time_series_df: pd.DataFrame,
    pipeline: Chronos2Pipeline,
    target_cols: list,
    context_length: int = 100,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Extract Chronos-2 embeddings for each column and each segment.

    For each segment (item_id >= 1), we take the last `context_length` values
    from all preceding data as context for each column independently,
    then stack them across columns.

    Args:
        time_series_df: DataFrame with [timestamp, item_id, col1, ..., colD]
        pipeline: Chronos2Pipeline instance
        target_cols: List of D column names to process
        context_length: Number of historical points for context
        batch_size: Batch size for Chronos embedding extraction
        device: Device

    Returns:
        embeddings: np.ndarray of shape [num_segments, D, 9, 768]
                    where 9 = Chronos-2 token count, 768 = embed dim
    """
    past_data = time_series_df.copy().drop(columns=["timestamp"], errors="ignore")
    max_item = int(past_data["item_id"].max())

    # We cannot evaluate the last segment (no future labels to check)
    num_segments = max_item - 1
    if num_segments <= 0:
        return np.empty((0, len(target_cols), 9, 768), dtype=np.float32)

    all_embeddings = []  # will be [num_segments, D, 9, 768]

    for item in tqdm(range(1, max_item), desc="  Embedding segments", leave=False):
        # Context: all rows before this segment, last context_length rows
        context_mask = past_data["item_id"] < item
        context_data = past_data.loc[context_mask].iloc[-context_length:]

        segment_embeddings = []  # [D, 9, 768] for this segment

        for col in target_cols:
            col_values = context_data[col].values.astype(np.float32)
            inp = torch.tensor(col_values, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, T]
            emb = pipeline.embed(inputs=inp, batch_size=batch_size)[0][0]  # [9, 768]
            segment_embeddings.append(emb.cpu().numpy() if isinstance(emb, torch.Tensor) else emb)

        all_embeddings.append(np.stack(segment_embeddings))  # [D, 9, 768]

    return np.array(all_embeddings, dtype=np.float32)  # [num_segments, D, 9, 768]


# ═══════════════════════════════════════════════════════════════
# ANOMALY FORECASTING
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def generateAnomalyForecast(
    embeddings: np.ndarray,
    model: TwoStageMultivariateDetector,
    threshold: float = 0.5,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """
    Generate per-segment global anomaly predictions using the two-stage model.

    For each segment, the model receives [D, 9, 768] embeddings
    and outputs a global anomaly logit.

    Args:
        embeddings: [num_segments, D, 9, 768]
        model: TwoStageMultivariateDetector (in eval mode)
        threshold: Binary classification threshold (from training)
        device: Device

    Returns:
        predictions: [num_segments] binary anomaly predictions
    """
    model.eval()
    predictions = []

    for seg_idx in range(embeddings.shape[0]):
        global_logit = model.forward_single_sample(torch.tensor(embeddings[seg_idx], dtype=torch.float32).to(device))[0]
        predictions.append(int(torch.sigmoid(global_logit).cpu().item() > threshold))

    return np.array(predictions)


# ═══════════════════════════════════════════════════════════════
# DATASET EVALUATION
# ═══════════════════════════════════════════════════════════════

def evaluate_dataset(
    dataset_path: str,
    pipeline: Chronos2Pipeline,
    model: TwoStageMultivariateDetector,
    configuration: dict,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    Complete multivariate evaluation pipeline for a single dataset.

    Args:
        dataset_path: Path to CSV dataset (multivariate + label column)
        pipeline: Chronos2Pipeline instance
        model: TwoStageMultivariateDetector in eval mode
        configuration: Dict with context_length, prediction_length, batch_size, binary_threshold
        device: Device

    Returns:
        Dictionary with evaluation metrics
    """
    print(f"Processing: {os.path.basename(dataset_path)}")

    context_length = configuration.get("context_length", 100)
    prediction_length = configuration.get("prediction_length", 64)

    time_series_df, ground_truth_labels, target_cols = prepare_data_for_chronos(
        dataset_path,
        context_length=context_length,
        prediction_length=prediction_length,
    )

    print(f"  Columns (D): {len(target_cols)} | Ground truth anomaly rate: {np.mean(ground_truth_labels):.2%}")

    # Extract multivariate embeddings: [num_segments, D, 9, 768]
    embeddings = getMultivariateChronos2Embeddings(
        time_series_df=time_series_df,
        pipeline=pipeline,
        target_cols=target_cols,
        context_length=context_length,
        batch_size=configuration.get("batch_size", 32),
    )

    if embeddings.shape[0] == 0:
        print("  WARNING: No segments to evaluate")
        return None
    

    # Run through two-stage model
    anomaly_forecast = generateAnomalyForecast(
        embeddings=embeddings,
        model=model,
        threshold=configuration["binary_threshold"],
        device=device,
    )

    return get_metrics(anomaly_forecast, ground_truth_labels, time_series_df, context_length, prediction_length)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main(configuration: dict, name: str) -> None:
    """Main execution function for multivariate anomaly forecasting."""
    data_path = "./TSB-AD-M"
    out_initial_path = f"./results/{name}/Multivariate_Anomaly_Forecast/"

    os.makedirs(out_initial_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Chronos-2 pipeline
    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device)
    print(f"Using device: {next(pipeline.model.parameters()).device}")

    # Load two-stage model
    model, configuration["binary_threshold"] = getModel(
        model_path=configuration["model_path"],
        stage1_checkpoint=configuration.get("stage1_checkpoint"),
        device=device,
        context_length=configuration.get("context_length", 100),
    )

    # Determine results file
    existing_nums = [int(fname.split("_")[1].split(".")[0])for fname in os.listdir(out_initial_path)if fname.startswith("results_") and fname.endswith(".json")]
    save_path = f"results_{max(existing_nums + [1])}.json"

    if os.path.exists(os.path.join(out_initial_path, save_path)):
        with open(os.path.join(out_initial_path, save_path), "r", encoding="utf-8") as f:
            existing_results = json_load(f)
    else:
        existing_results = {}

    # Process datasets
    dataset_files = sorted(f for f in os.listdir(data_path) if f.endswith(".csv"))
    if all(fname in existing_results for fname in dataset_files):
        existing_results = {}
        save_path = f"results_{int(save_path.split('_')[1].split('.')[0]) + 1}.json"
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
                model=model,
                configuration=configuration,
                device=device,
            )
        except Exception as e:
            tqdm.write(f"Error processing file {filename}: {e}")
            raise e

        if result is not None:
            existing_results[filename] = {**result, **configuration}
            with open(os.path.join(out_initial_path, save_path), "w", encoding="utf-8") as f:
                json_dump(existing_results, f, indent=4)
            print(f"\nResults saved for {filename}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Chronos-2 Multivariate Anomaly Forecasting")
    parser.add_argument("--user", type=str, required=True,choices=["Nunzio", "Aldo", "Sara", "Valentino", "Simone"],help="Username of the person running the script",)
    parser.add_argument("--context_length", type=int, default=-1, help="Context length for Chronos-2")
    parser.add_argument("--prediction_length", type=int, default=-1, help="Prediction length for Chronos-2")
    parser.add_argument("--batch_size", type=int, default=-1, help="Batch size for embedding extraction")
    parser.add_argument("--model_path", type=str, default="./Saved_Models/two_stage/",help="Path to the two-stage model directory",)
    parser.add_argument("--stage1_checkpoint", type=str, default=None, help="Override Stage 1 checkpoint path (uses config.json value if not set)",)

    args = parser.parse_args()

    configuration = {
        "context_length": args.context_length if args.context_length > 0 else 100,
        "prediction_length": args.prediction_length if args.prediction_length > 0 else 64,
        "batch_size": args.batch_size if args.batch_size > 0 else 256,
        "model_path": args.model_path,
        "stage1_checkpoint": args.stage1_checkpoint,
    }

    main(configuration, name=str(args.user).strip().upper())