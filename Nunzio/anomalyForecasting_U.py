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
except (ImportError, ModuleNotFoundError):
    from metrics.metricsAnomalyForecast import get_metrics
    from KIMI_RAIKKONEN import ChronosAnomalyDetector

from tqdm import tqdm


from json import dump as json_dump, load as json_load



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


def getModel(model_path: str, device: str = 'cpu', context_length: int =128) -> ChronosAnomalyDetector:
    """Load GNN model from path
    
    model_path (str): Path to saved model
    device (str): Device to load the model on

    Returns:
        model (torch.nn.Module): Loaded GNN model
    """
    checkpoint = torch.load(model_path, map_location=device)
    config = json_load(open(os.path.join(os.path.dirname(model_path), 'config.json'), 'r', encoding='utf-8'))
    model = ChronosAnomalyDetector(
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        num_heads=config['num_heads'],
        dropout=config['dropout'],
        numPatches=config.get('numPatches', np.ceil(context_length/16)+2)  # Default to 9 if not specified
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])

    threshold = json_load(open(os.path.join(os.path.dirname(model_path), 'training_log.json'), 'r', encoding='utf-8'))
    return model.eval(), threshold[int(model_path.split("_")[-1].strip(".pth"))-1]['val_best_threshold']


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


def getUnivariateChronos2Enbeddings(time_series_df: pd.DataFrame, pipeline: Chronos2Pipeline, context_length: int = 100, batch_size: int = 32, device: torch.device = torch.device('cpu')) -> dict[str, pd.DataFrame]:
    """
    Get univariate embeddings from Chronos-2 for each segment (item_id >= 1) using the context_length values before it.
    Each embedding is a vector of size embed_dim representing the context information for that segment.

    Args:
        time_series_df: DataFrame with columns [timestamp, item_id, col1, col2, ..., colD]
        pipeline: Chronos2Pipeline instance
        context_length: Number of historical points for context
        batch_size: Number of segments to process in parallel
        device: Device to run the model on

    Returns:
        embeddings_dict: Dict mapping each target_col to DataFrame with embeddings for each segment
    """
    # Ricordati, ti arriva un dataframe con colonna timestamp, item_id e valore della singola serie
    pastData = time_series_df.copy().drop(columns=['timestamp'], inplace=False, errors='ignore')

    return np.array([
        pipeline.embed(inputs=torch.tensor(np.expand_dims(pastData.loc[pastData['item_id'] < item, :].iloc[-context_length:, -1:].values.T, axis=0), dtype=torch.float32, requires_grad=False), batch_size=batch_size)[0][0]
        for item in range(1, pastData['item_id'].max())]) # We cannot evaluate the last segment because we don't have the future labels to check for anomalies


@torch.no_grad()
def generateAnomalyForecast(embeddings: np.ndarray, transformer: ChronosAnomalyDetector, batch_size: int = 32, threshold: float = 0.5, device: torch.device = torch.device('cpu')) -> np.ndarray:
    """
    Generate next-segment anomaly scores using the GNN model

    Args:
        embeddings: Array of shape [num_segments, embed_dim] with segment embeddings
        transformer: ChronosAnomalyDetector instance
        batch_size: Number of segments to process in parallel
        threshold: Threshold for binary classification
        device: Device to run the model on

    Returns:
        anomaly_scores: Array of shape [num_segments] with anomaly scores for each segment
    """
    indexes, predictions = [], []

    if embeddings.shape[0] < batch_size:
        indexes.append([0, embeddings.shape[0]])
    else:
        for i in range(0, embeddings.shape[0], batch_size):
            indexes.append([i, min(i + batch_size, embeddings.shape[0])])

    for start, end in indexes:
        predictions.extend((torch.sigmoid(transformer(torch.tensor(embeddings[start:end, 0, ...], dtype=torch.float32, device=device))).cpu().numpy() > threshold).astype(int).tolist())
    return np.array(predictions)



def evaluate_dataset(dataset_path: str,pipeline: Chronos2Pipeline, transformer:ChronosAnomalyDetector, configuration: dict,
                    device: torch.device = torch.device('cpu')) -> dict:
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
    prediction_length = configuration.get('prediction_length', 64)

    time_series_df, ground_truth_labels, _ = prepare_data_for_chronos(
        dataset_path, 
        context_length=context_length, 
        prediction_length=prediction_length
    )
    
    print(f"Ground truth anomaly rate: {np.mean(ground_truth_labels):.2%}")

    embeddings = getUnivariateChronos2Enbeddings(
        time_series_df=time_series_df,
        pipeline=pipeline,
        context_length=context_length,
        batch_size=configuration.get('batch_size', 32),
        device=device
    )

    # Compute next-segment anomaly scores using the model
    anomalyForecast = generateAnomalyForecast(embeddings=embeddings,transformer=transformer, 
                        batch_size=configuration.get('batch_size', 32), threshold=configuration['binary_threshold'], device=device)
    
    return get_metrics(anomalyForecast, ground_truth_labels, time_series_df, context_length, prediction_length)



def main(configuration:dict, name:str)->None:
    """Main execution function"""
    # Configuration
    data_path = "./TSB-AD-U"
    # data_path = './Nunzio/provaData/multivariate/'
    out_initial_path = f"./results/{name}/Univariate_Anomaly_Forecast/"

    os.makedirs(out_initial_path, exist_ok=True)

    # Parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pipeline = get_pipeline(device=device)
    print(f"Using device: {next(pipeline.model.parameters()).device}")
    # A good model: "Saved_Models_Temporal\v1_32_v2_4\checkpoint_epoch_25.pth"
    model, configuration['binary_threshold'] = getModel(model_path=configuration['model_path'], device=device, context_length=configuration.get('context_length', 100))
    
    save_path = f"results_{max([int(fname.split('_')[1].split('.')[0]) for fname in os.listdir(out_initial_path) if fname.startswith('results_') and fname.endswith('.json')] + [1])}.json"
    
    if os.path.exists(os.path.join(out_initial_path, save_path)):
        with open(os.path.join(out_initial_path, save_path), 'r', encoding='utf-8') as f:
            existing_results = json_load(f)
    else:
        existing_results = {}

    # Process datasets
    dataset_files = sorted(filter(lambda x: x.endswith('.csv'), os.listdir(data_path)))

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
                transformer=model,
                configuration=configuration,
                device=device
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
    args.add_argument('--model_path', type=str, default='./Nunzio/SAVED_MODELSV3', help='Path to the pretrained GNN model')
    parsed_args = args.parse_args()

    configuration = {
        'context_length': parsed_args.context_length if parsed_args.context_length > 0 else 100,
        'prediction_length': parsed_args.prediction_length if parsed_args.prediction_length > 0 else 1,
        'step_size': parsed_args.step_size if parsed_args.step_size > 0 else 1,
        'batch_size': parsed_args.batch_size if parsed_args.batch_size > 0 else 256,
        'model_path': parsed_args.model_path,
    }

    
    main(configuration, name=str(parsed_args.user).strip().upper())