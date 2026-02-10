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
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.exceptions import UndefinedMetricWarning

from json import dump as json_dump, load as json_load

import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

if torch.cuda.is_available():
    torch.cuda.empty_cache()


def get_pipeline(model_name: str = "amazon/chronos-2", device: str = None) -> Chronos2Pipeline:
    """Load Chronos-2 pipeline"""
    return Chronos2Pipeline.from_pretrained(
        model_name,
        device_map=device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )


def extract_chronos_embeddings(
    time_series_df: pd.DataFrame,
    pipeline: Chronos2Pipeline,
    target_col: str,
    context_length: int = 100,
    batch_size: int = 128,
):
    """
    Extract Chronos embeddings using Chronos2Pipeline.embed().
    Compatible with Chronos 2.2.2:
    - input shape: (batch_size, n_variates=1, context_length)
    - CPU tensor, pipeline handles device internally
    """
    values = time_series_df[target_col].values
    timestamps = time_series_df['timestamp'].values

    all_embeddings = []
    valid_indices = []

    indices = np.arange(context_length, len(time_series_df))

    for batch_start in tqdm(range(0, len(indices), batch_size),
                            desc="Extracting Chronos embeddings",
                            leave=False):
        batch_indices = indices[batch_start:batch_start + batch_size]

        # 3D tensor (batch, 1, context_length)
        batch_contexts_list = []
        for idx in batch_indices:
            context = torch.tensor(
                values[idx - context_length:idx],
                dtype=torch.float32
            ).unsqueeze(0).unsqueeze(0)  # (1,1,context_length)
            batch_contexts_list.append(context)

        batch_contexts = torch.cat(batch_contexts_list, dim=0)  # (batch_size,1,context_length)

        with torch.no_grad():
            embeddings, _ = pipeline.embed(batch_contexts)

        batch_embeddings = torch.stack(embeddings)  # tensor

        # (batch, tokens, hidden)
        if batch_embeddings.dim() == 3:
            batch_embeddings = batch_embeddings.mean(dim=1)

        # (batch, 1, tokens, hidden)
        elif batch_embeddings.dim() == 4:
            batch_embeddings = batch_embeddings.mean(dim=2).squeeze(1)

        # Now it's 2D
        batch_embeddings = batch_embeddings.cpu().numpy()

        all_embeddings.append(batch_embeddings)
        valid_indices.extend(batch_indices.tolist())

    all_embeddings = np.vstack(all_embeddings)  # (num_windows, hidden_dim)

    return all_embeddings, valid_indices


def get_timestamp(start_date: str = "2026-01-01 00:00:00", periods: int = 100, freq: str = 'min'):
    """Generate timestamps for time series"""
    return pd.date_range(start=start_date, periods=periods, freq=freq)


def prepare_data_for_chronos(dataset_path: str):
    """Prepare data in Chronos-2 format"""
    df = pd.read_csv(dataset_path, header=0, index_col=None)
    df_clean = df.drop(columns=[df.columns[-1]])

    df_chronos = pd.DataFrame()
    df_chronos['timestamp'] = get_timestamp(periods=len(df_clean))
    df_chronos['item_id'] = 0
    df_chronos[df.columns[0]] = df_clean[df.columns[0]].values

    return df_chronos, df[df.columns[-1]].values, df.columns[0]


def computeEmbeddingAnomalyScore(
    time_series_df: pd.DataFrame,
    pipeline: Chronos2Pipeline,
    target_col: str,
    prediction_indices: np.ndarray,
    context_length: int = 100,
    k: int = 5
):
    """
    KNN anomaly detection on Chronos embeddings (using pipeline.embed).
    """
    embeddings, valid_indices = extract_chronos_embeddings(
        time_series_df=time_series_df,
        pipeline=pipeline,
        target_col=target_col,
        context_length=context_length
    )

    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(embeddings))).fit(embeddings)
    distances, _ = nbrs.kneighbors(embeddings)

    knn_scores = distances[:, 1:].mean(axis=1)
    knn_scores = (knn_scores - knn_scores.min()) / (knn_scores.max() - knn_scores.min() + 1e-8)

    final_scores = np.zeros(len(prediction_indices))
    for idx, valid_idx in enumerate(valid_indices):
        mask = prediction_indices == valid_idx
        if np.any(mask):
            final_scores[mask] = knn_scores[idx]

    return final_scores


def computeDiscreteAnomalyScores(continuous_scores: np.ndarray, percentile: float = 95):
    """Compute binary anomaly predictions based on percentile threshold"""
    threshold = np.percentile(continuous_scores, percentile)
    return (continuous_scores > threshold).astype(np.int8)


def evaluate_dataset(dataset_path: str, pipeline: Chronos2Pipeline, configuration: dict):
    """Complete evaluation pipeline for a single dataset"""
    print(f"Processing: {os.path.basename(dataset_path)}")

    time_series_df, ground_truth_labels, target_col = prepare_data_for_chronos(dataset_path)
    print(f"Ground truth anomaly rate: {np.mean(ground_truth_labels):.2%}")

    context_length = configuration.get('context_length', 100)
    valid_indices = np.arange(context_length, len(time_series_df))
    prediction_indices = valid_indices

    # KNN call
    knn_scores = computeEmbeddingAnomalyScore(
        time_series_df=time_series_df,
        pipeline=pipeline,
        target_col=target_col,
        prediction_indices=prediction_indices,
        context_length=context_length,
        k=configuration.get('k_neighbors', 5) 
    )

    
    knn_discrete = computeDiscreteAnomalyScores(
        knn_scores,
        percentile=configuration.get('threshold_percentile', 95) 
    )

    return {
        'file': os.path.basename(dataset_path),
        'metrics': [{
            **get_metrics(score=knn_scores,
                          labels=ground_truth_labels[prediction_indices],
                          pred=knn_discrete),
            'accuracy': float(accuracy_score(ground_truth_labels[prediction_indices], knn_discrete)),
            'precision': float(precision_score(ground_truth_labels[prediction_indices], knn_discrete, zero_division=0)),
            'recall': float(recall_score(ground_truth_labels[prediction_indices], knn_discrete, zero_division=0)),
            'f1_score': float(f1_score(ground_truth_labels[prediction_indices], knn_discrete, zero_division=0)),
            'confusion_matrix': confusion_matrix(
                ground_truth_labels[prediction_indices],
                knn_discrete
            ).tolist(),
            'method': 'knn_only'
        }]
    }


def main(configuration: dict, name: str) -> None:
    data_path = "./TSB-AD-U/"
    out_initial_path = f"./results/{name}/"
    os.makedirs(out_initial_path, exist_ok=True)

    pipeline = get_pipeline(device='cuda')
    print(f"Using device: {next(pipeline.model.parameters()).device}")

    save_path = f"results_{max([int(fname.split('_')[1].split('.')[0]) for fname in os.listdir(out_initial_path) if fname.startswith('results_') and fname.endswith('.json')] + [-1])}.json"

    if os.path.exists(os.path.join(out_initial_path, save_path)):
        with open(os.path.join(out_initial_path, save_path), 'r', encoding='utf-8') as f:
            existing_results = json_load(f)
    else:
        existing_results = {}

    dataset_files = (
        sorted(pd.read_csv("test_files_U.csv")["name"].tolist())
        if configuration.get('use_restricted_dataset', True)
        else sorted(filter(lambda x: x.endswith('.csv'), os.listdir(data_path)))
    )

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
                configuration=configuration,
            )
        except Exception as e:
            tqdm.write(f"Error processing file {filename}: {e}")
            continue

        if result is not None:
            with open(os.path.join(out_initial_path, save_path), 'w', encoding='utf-8') as f:
                existing_results[filename] = {**result, **configuration}
                json_dump(existing_results, f, indent=4)
                print(f"\nResults saved for {filename}\n")


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Run K-NN Anomaly Detection on datasets")

    args.add_argument('--user', type=str,
                      choices=['Nunzio', 'Aldo', 'Sara', 'Valentino', 'Simone', 'Senku'],
                      required=True)

    args.add_argument('--context_length', type=int, default=-1)
    args.add_argument('--batch_size', type=int, default=-1)
    args.add_argument('--k_neighbors', type=int, default=-1)  
    args.add_argument('--threshold_percentile', type=float, default=-1) 
    args.add_argument('--use_restricted_dataset', action='store_true', default=False)
    args.add_argument('--colab', action='store_true', default=False)

    parsed_args = args.parse_args()

    configuration = {
        'context_length': parsed_args.context_length if parsed_args.context_length > 0 else 100,
        'batch_size': parsed_args.batch_size if parsed_args.batch_size > 0 else 256,
        'k_neighbors': parsed_args.k_neighbors if parsed_args.k_neighbors > 0 else 5, 
        'threshold_percentile': parsed_args.threshold_percentile if parsed_args.threshold_percentile > 0 else 95, 
        'use_restricted_dataset': bool(parsed_args.use_restricted_dataset),
        'colab': bool(parsed_args.colab),
    }

    main(configuration, name=str(parsed_args.user).strip().upper())