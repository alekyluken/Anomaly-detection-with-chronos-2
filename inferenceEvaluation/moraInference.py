import torch, argparse, json
import pandas as pd, numpy as np


from chronos import Chronos2Pipeline
from itertools import product
from time import time as getCurrentTime
from tqdm import tqdm


if torch.cuda.is_available():
    torch.cuda.empty_cache()


def prepare_data_for_chronos(context_length: int = 100, D:int=1):
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

    df = pd.DataFrame(np.random.rand(context_length + 1, D), columns=[f'col{i}' for i in range(D)])  # Placeholder for actual data loading

    # faster way to create the same DataFrame without copying data multiple times
    return pd.concat([
        pd.DataFrame({
            'timestamp': list(range(context_length + 1)),
            'item_id': [0] * context_length + [1],
        }),
        df.reset_index(drop=True),
    ], axis=1), df.columns.tolist()


def evaluate_dataset(pipeline: Chronos2Pipeline, configuration: dict) -> dict:
    """
    Benchmark inference pipeline: times prediction and score computation separately.

    Args:
        pipeline: Chronos2Pipeline instance
        configuration: Dictionary with evaluation parameters

    Returns:
        Dictionary with prediction_times, 
    """
    context_length = configuration.get('context_length', 100)
    
    time_series_df, target_cols = prepare_data_for_chronos(context_length, configuration.get('D', 1))

    # ── Warmup run (exclude from timing) ──
    _ = pipeline.predict(inputs=torch.tensor(
        time_series_df[target_cols].iloc[:context_length].values.astype(np.float32).T, 
        dtype=torch.float32).unsqueeze(0),  prediction_length=1)

    # ── Loop 1: Time predictions ──
    prediction_times = np.zeros(configuration.get('n_runs', 10))
    for i in range(configuration.get('n_runs', 10)):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        prediction_times[i] = getCurrentTime()

        _ = pipeline.predict(inputs=torch.tensor(
                time_series_df[target_cols].iloc[:context_length].values.astype(np.float32).T, 
                dtype=torch.float32).unsqueeze(0),  prediction_length=1)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        prediction_times[i] = getCurrentTime() - prediction_times[i]

    return np.array(prediction_times)

def printStats(times:np.ndarray, title:str):
    stats = {
        'average': np.mean(times),
        'median': np.median(times),
        'min': np.min(times),
        'max': np.max(times),
        'std_dev': np.std(times)
    }
    
    print("\n", title)
    print(f"Average time: {stats['average']:.4f} seconds")
    print(f"Median time: {stats['median']:.4f} seconds")
    print(f"Min time: {stats['min']:.4f} seconds")
    print(f"Max time: {stats['max']:.4f} seconds")
    print(f"Standard deviation: {stats['std_dev']:.4f} seconds")
    print("-"*50, "\n")

    return stats


def main(configuration:dict)->None:
    """Main execution function — benchmarks inference times across context lengths and D."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")
    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device)

    outFile = "inference_benchmarks_pc.json"
    with open(outFile, 'w') as f:
        json.dump({}, f, indent=4)

    for context_length, D in tqdm(list(product(
        [32, 64, 128, 256, 512, 1024],  # context_length
        [1, 2, 5, 10, 20, 50, 100],     # number of variates D
    )), desc="Benchmarking"):
        print(f"\nEvaluating ctx={context_length}, D={D}")
        configuration['context_length'] = context_length
        configuration['D'] = D
        
        title = f"Context Length: {context_length}, D: {D}"

        res = printStats(evaluate_dataset(pipeline=pipeline, configuration=configuration), title=title)

        with open(outFile, 'r') as f:
            allResults = json.load(f)
        allResults[title] = res
        with open(outFile, 'w') as f:
            json.dump(allResults, f, indent=4)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Run Chronos-2 Anomaly Detection on datasets")
    args.add_argument('--batch_size', type=int, default=64, help='Batch size for predictions')
    args.add_argument('--n_runs', type=int, default=100, help='Number of timing iterations per (context_length, D) combo')
    parsed_args = args.parse_args()

    configuration = {
        'batch_size': parsed_args.batch_size if parsed_args.batch_size > 0 else 256,
        'n_runs': max(parsed_args.n_runs, 1),
    }
    
    main(configuration)