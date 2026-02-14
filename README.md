# Anomaly Detection & Forecasting with Chronos-2

This is the official project for the **Deep Natural Language Processing** course. It was made by:
- Nunzio Licalzi
- Aldo Karamani
- Valentino Vacirca
- Sara Cappelletti
- Simone Francesco Licitra

A live demo of the project is available at https://huggingface.co/spaces/Nuzz23/Chronos2AD_AF

## Overview

This project explores the use of the **Amazon Chronos-2** time series foundation model for **anomaly detection** and **anomaly forecasting** in both univariate and multivariate time series

All experiments are evaluated on the [TSB-AD Benchmark](https://thedatumorg.github.io/TSB-AD/) for time series anomaly detection.

## Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

To run the experiments, download the TSB-AD benchmark datasets from https://thedatumorg.github.io/TSB-AD/ and place the `TSB-AD-U/` (univariate) and `TSB-AD-M/` (multivariate) folders in the project root.

---

## Project Structure

```
├── univariateAnomalyDetection/   # Univariate anomaly detection (direct forecasting & KNN)
├── anomalyForecasting/           # Learned anomaly detectors (transformer, two-stage)
├── multivariateAD/               # Multivariate anomaly detection
│   ├── GNN/                      # GNN-based and quantile-based learned detectors
│   └── graph/                    # PageRank-based variable aggregation
├── metrics/                      # Evaluation metrics (point-level & range-based)
├── inferenceEvaluation/          # Inference timing benchmarks
├── resultAnalizer/               # Result analysis and comparison scripts
├── experiments/                  # Jupyter notebooks for exploratory experiments -> not used in the final report, as such should be ignored
├── realTimeDetection/            # Real-time anomaly detection notebook
├── results/                      # Experiment results (JSON) organized by contributor
├── Saved_Models*/                # Trained model checkpoints and configs
└── requirements.txt
```

---

# UNIVARIATE ANOMALY DETECTION

The code for univariate anomaly detection is located in the `univariateAnomalyDetection/` folder. Three distinct approaches are implemented:

## 1. Direct Forecasting — `customRunAD.py`

Uses Chronos-2 sliding-window probabilistic forecasts to detect anomalies. For each window, quantile predictions (e.g., 10th, 50th, 90th percentiles) are generated and anomaly scores are computed from prediction errors.

**Anomaly Scoring Methods:**
- **IQR-normalized score**: deviation from median normalized by the interquartile range
- **Naive score**: absolute distance from median prediction
- **Multi-horizon score**: maximum squared error across multiple forecast horizons

Binary predictions are obtained via percentile-based thresholding on the continuous anomaly scores.

**Usage:**
```bash
python univariateAnomalyDetection/customRunAD.py --user <name> [options]
```

**Arguments:**
| Argument | Default | Description |
|---|---|---|
| `--user` | *(required)* | Contributor name (`Nunzio`, `Aldo`, `Sara`, `Valentino`, `Simone`) |
| `--context_length` | 100 | Input sequence length for the model |
| `--prediction_length` | 1 | Output forecast horizon length |
| `--step_size` | 1 | Sliding window step size |
| `--batch_size` | 256 | Batch size for inference |
| `--square_distance` | `False` | Use squared distance for anomaly scoring |
| `--use_naive` | `False` | Use naive (absolute) scoring instead of IQR-normalized |
| `--use_multihorizon` | `False` | Enable multi-horizon anomaly scoring |
| `--horizons` | 3 | Number of forecast horizons for multi-horizon scoring |
| `--percentile` | 95 | Percentile threshold for binary classification |
| `--thresholds` | *(auto)* | Custom thresholds to evaluate |
| `--use_restricted_dataset` | `False` | Evaluate only on a restricted file list |
| `--colab` | `False` | Colab compatibility mode |

## 2. Multi-Horizon Ensemble — `customRunAnomalyScore.py`

A variant of the direct forecasting approach focused on **multi-horizon ensemble scoring**. Generates multi-step predictions and computes the maximum anomaly score across all forecast horizons for more robust detection.

**Usage:**
```bash
python univariateAnomalyDetection/customRunAnomalyScore.py --user <name> [options]
```

Arguments are the same as `customRunAD.py` (minus `--use_multihorizon`).

## 3. Latent Space KNN — `KNNAD.py`

Detects anomalies by applying **K-Nearest Neighbors** in the Chronos-2 embedding space. The algorithm:
1. Extracts Chronos-2 embeddings for sliding windows via `pipeline.embed()`
2. Mean-pools token embeddings to a single vector per window
3. Fits a KNN model on all embeddings
4. Uses average distance to K nearest neighbors as the anomaly score
5. Thresholds at a configurable percentile

**Usage:**
```bash
python univariateAnomalyDetection/KNNAD.py --user <name> [options]
```

**Arguments:**
| Argument | Default | Description |
|---|---|---|
| `--user` | *(required)* | Contributor name |
| `--context_length` | 100 | Sliding window size |
| `--batch_size` | 256 | Batch size for embedding extraction |
| `--k_neighbors` | 5 | Number of neighbors for KNN |
| `--threshold_percentile` | 95 | Percentile for anomaly thresholding |
| `--use_restricted_dataset` | `False` | Use restricted dataset |
| `--colab` | `False` | Colab compatibility mode |

---

# MULTIVARIATE ANOMALY DETECTION

## Direct Prediction + Aggregation — `multivariateAD/MultiVariate_new.py`

Multivariate anomaly detection using direct Chronos-2 forecasts without any learned model. Computes per-variable multi-horizon anomaly scores, then aggregates them across variables using configurable strategies:

- **Top-K selection**: `log(D)`, `sqrt(D)`, or dynamic (largest score gap)
- **Aggregation**: mean, max, or sum of top-K scores
- **Normalization**: min-max, z-score, or robust scaling

**Usage:**
```bash
python multivariateAD/MultiVariate_new.py --user <name> [options]
```

**Arguments:**
| Argument | Default | Description |
|---|---|---|
| `--user` | *(required)* | Contributor name |
| `--context_length` | 100 | Input sequence length |
| `--prediction_length` | 1 | Forecast horizon |
| `--batch_size` | 256 | Batch size |
| `--horizons` | 3 | Number of forecast horizons |
| `--aggregation_method` | `mean` | How to aggregate top-K scores (`mean`, `max`, `sum`) |
| `--normalization_method` | `minmax` | Score normalization (`minmax`, `zscore`, `robust`) |
| `--top_k_method` | `log` | Top-K selection (`log`, `sqrt`, `jump`) |
| `--thresholds` | *(auto)* | Custom thresholds |
| `--use_restricted_dataset` | `False` | Use restricted dataset |

## PageRank-Based Aggregation — `multivariateAD/graph/MultiVariateGraph.py`

Builds a **variable-relationship graph** using either:
- **Cosine similarity** of Chronos-2 variable embeddings (with mean/max/first/last/topk_mean aggregation) --> only partial usage in the final report, some options should be ignored
- **Granger causality** (Ridge-regression-based)  -> Not used in the final report due to time constraints, should be ignored

Then runs **PageRank** with a utility vector (sum_CRPS, surprise, or energy-based) and damping factor to weight per-variable anomaly scores.

**Usage:**
```bash
python multivariateAD/graph/MultiVariateGraph.py --user <name> [options]
```

**Arguments:**
| Argument | Default | Description |
|---|---|---|
| `--user` | *(required)* | Contributor name |
| `--context_length` | 100 | Input sequence length |
| `--prediction_length` | 1 | Forecast horizon |
| `--batch_size` | 256 | Batch size |
| `--horizons` | 3 | Number of forecast horizons |
| `--aggregation_method` | `mean` | Aggregation for PageRank weights |
| `--howToEvaluate_u` | `sum` | Utility vector method (`sum`, `surprise`, `energy`) |
| `--percentile` | 95 | Anomaly threshold percentile |
| `--beta` | 0.85 | PageRank damping factor |

---

# ANOMALY FORECASTING (Learned Detectors)

The `anomalyForecasting/` folder implements **learned anomaly classifiers** trained on Chronos-2 embeddings.

## Stage 1: Token-Aware Transformer — `anomalyDetectorModule.py`

A **token-aware transformer classifier** that operates on Chronos-2 embeddings of shape `[N, W, 768]`. The architecture:
1. Projects the 768-dim embeddings to a configurable hidden dimension
2. Adds **learned token-type embeddings** (summary, patch, forecast tokens)
3. Applies **multi-head self-attention** across the W tokens
4. Extracts summary + forecast + pooled-patch representations
5. Feeds through an MLP to produce a binary anomaly logit

**Training features**: Sigmoid Focal Loss, EMA model averaging, mixed precision (AMP), OneCycleLR scheduling, gradient accumulation/clipping, PR-curve threshold optimization.

## Stage 2: Set Transformer for Cross-Variable Aggregation — `two_stage_detector.py`

A **two-stage multivariate anomaly detector**:
- **Stage 1**: Per-series `ChronosAnomalyDetector` (above) produces features for each variable
- **Stage 2**: A **Set Transformer** (Induced Set Attention Blocks + Pooling by Multihead Attention) aggregates across variables in a **permutation-invariant** manner

The Stage 2 produces both a **global** anomaly score and **per-series** anomaly scores. Training uses a combined loss:

$$L = \alpha \cdot L_{\text{global}} + \beta \cdot L_{\text{series}} + \gamma \cdot L_{\text{consistency}}$$

## Training the Two-Stage Model — `train_two_stage.py`

Full training pipeline with:
- `MultivariateAnomalyDataset`: organizes embeddings by dataset, groups segments by `item_id`, memory-mapped loading
- `PrecomputedDataset`: uses pre-extracted Stage 1 features for faster Stage 2 training
- WeightedRandomSampler for class balancing
- Optional end-to-end finetuning with differential learning rates (lower LR for Stage 1)
- Early stopping, ReduceLROnPlateau, gradient clipping

## Univariate Inference — `anomalyForecasting_U.py`

Inference script for univariate anomaly forecasting on TSB-AD-U. Extracts Chronos-2 embeddings per segment and runs the trained `ChronosAnomalyDetector`.

**Usage:**
```bash
python anomalyForecasting/anomalyForecasting_U.py --user <name> --model_path <path> [options]
```

**Arguments:**
| Argument | Default | Description |
|---|---|---|
| `--user` | *(required)* | Contributor name |
| `--context_length` | 100 | Input sequence length |
| `--prediction_length` | 1 | Forecast horizon |
| `--step_size` | 1 | Sliding window step |
| `--batch_size` | 256 | Batch size |
| `--model_path` | *(required)* | Path to trained model checkpoint |

## Multivariate Inference — `anomalyForecasting_M.py`

Inference script for multivariate anomaly forecasting on TSB-AD-M. Extracts per-column per-segment embeddings → `[num_segments, D, 9, 768]` → runs the two-stage model.

**Usage:**
```bash
python anomalyForecasting/anomalyForecasting_M.py --user <name> --model_path <path> --stage1_checkpoint <path> [options]
```

**Arguments:**
| Argument | Default | Description |
|---|---|---|
| `--user` | *(required)* | Contributor name |
| `--context_length` | 100 | Input sequence length |
| `--prediction_length` | 1 | Forecast horizon |
| `--batch_size` | 256 | Batch size |
| `--model_path` | *(required)* | Path to two-stage model checkpoint |
| `--stage1_checkpoint` | *(required)* | Path to Stage 1 checkpoint |

---

# GNN-BASED MULTIVARIATE DETECTION

The `multivariateAD/GNN/` folder contains graph neural network approaches for multivariate anomaly detection.

## Spatio-Temporal GNN v1 — `customGNN.py`

A spatio-temporal anomaly detection GNN with:
- **TemporalProcessor**: causal dilated Conv1D per node
- **WeightedMessagePassing**: $h'_i = \text{MLP}(h_i \| \sum_j W_{ij} \cdot h_j)$
- **AdaptiveNodePooling**: attention/mean/max/sum pooling
- **DualHeadAnomalyDecoder**: continuous + binary output heads

Training (`trainGNN.py`) uses a combined loss: Focal + Ranking (margin-based score separation) + Dice + Consistency losses with SGD+Nesterov, warmup + cosine LR scheduling.

## Spatio-Temporal GNN v2 — `customGNN_v2.py`

Enhanced GNN with:
- **MultiScaleTemporalProcessor**: parallel dilated convolutions at scales 1, 2, 4, 8 with fusion
- **GraphAttentionLayer**: GAT with learned attention + edge weight integration
- **ContextAwarePooling**: cross-attention with learnable query
- **CalibratedDualHeadDecoder**: dual output with learnable temperature scaling

Training (`train_v2.py`) uses advanced techniques:
- Uncertainty-based adaptive loss weighting (learned log-variance)
- Dynamic focal gamma decay
- Asymmetric loss (higher penalty for false negatives)
- Online Hard Example Mining (OHEM)
- Distribution alignment loss
- CosineAnnealingWarmRestarts scheduler

## Lightweight Quantile Detector — `SimplerNN.py`

An alternative to GNN that uses **hand-crafted quantile features** from Chronos-2 predictions:
- **QuantileFeatureExtractor**: computes 12 interpretable features (normalized deviation, z-score, IQR, skewness, tail indicators, kurtosis proxy, etc.)
- **LightweightTemporalNet**: causal dilated Conv1D with residual connections (receptive field ~30 timesteps)
- **MultiHeadVariableAttention**: permutation-invariant attention pooling across the variable dimension D

Training (`train_simple.py`) uses a merged multi-source dataset with focal loss.

## Inference with Quantile Detector — `MultiVariateGNN.py`

End-to-end inference pipeline: raw multivariate CSV → Chronos-2 quantile predictions → `SimpleQuantileAnomalyDetector` → evaluation.

**Usage:**
```bash
python multivariateAD/GNN/MultiVariateGNN.py --user <name> --detector_model_path <path> [options]
```

## Feature Extraction — `extractFeaturesForGNN.py`

Extracts Chronos-2 embeddings and predictions from raw CSVs and saves them for downstream training. Outputs are stored in `PROCESSED_TRAIN_DATA*/` directories.

**Usage:**
```bash
python multivariateAD/GNN/extractFeaturesForGNN.py --user <name> [options]
```

---

# METRICS

The `metrics/` folder contains evaluation functions used across all experiments.

## `metricsEvaluation.py`

Comprehensive point-level and range-based anomaly detection metrics:
| Metric | Description |
|---|---|
| AUC-PR | Area Under the Precision-Recall Curve |
| AUC-ROC | Area Under the ROC Curve |
| Standard-F1 | Point-level F1 score |
| PA-F1 | Point-Adjusted F1 score |
| Event-based-F1 | Segment/event-level F1 |
| R-based-F1 | Range-based F1 |
| VUS-ROC | Volume Under Surface (sliding-window generalization of AUC-ROC) |
| VUS-PR | Volume Under Surface (sliding-window generalization of AUC-PR) |

## `metricsAnomalyForecast.py`

Segment-level evaluation metrics for the anomaly forecasting task. Aligns segment-level predictions with ground truth by grouping via `item_id`, and computes accuracy, precision, recall, F1, confusion matrix, and AUC-PR.


All the metrics are taken from the `TSB-AD` benchmark codebase, with some modifications for our specific use case (e.g., handling of per-segment predictions, support for multi-horizon forecasts, etc.). The modifications are solely used to speed up the computation via optimization and do not alter the underlying logic nor functionalities of the metrics.

---

# INFERENCE BENCHMARKING

## `inferenceEvaluation/moraInference.py`

Benchmarks Chronos-2 inference times across a grid of configurations:
- **Context lengths**: 32, 64, 128, 256, 512, 1024
- **Number of variates**: 1, 2, 5, 10, 20, 50, 100

**Usage:**
```bash
python inferenceEvaluation/moraInference.py --batch_size 256 --n_runs 5
```

---

# RESULT ANALYSIS

The `resultAnalizer/` folder provides tools for analyzing and comparing experiment results.

## `anomalyForecasting.py`

Analyzes anomaly forecasting results from JSON files with per-class breakdown and overall statistics (accuracy, precision, recall, F1, AUC-PR).

```bash
python resultAnalizer/anomalyForecasting.py --file <path_to_results.json>
```

## `inferenceAnalizer.py`

Prints inference benchmark results (median ± std for each configuration).

```bash
python resultAnalizer/inferenceAnalizer.py --file <path_to_benchmarks.json>
```

## `resultComparison.py`

Compares results across multiple experiment runs side-by-side. Builds a summary DataFrame with mean ± std per metric, highlighting best values.

```bash
python resultAnalizer/resultComparison.py --files results1.json results2.json [--restrictTo class1 class2] [--printAllResults]
```

---

# EXPERIMENTS & NOTEBOOKS

- `experiments/Prova_anomaly_detection_INS.ipynb` — Exploratory anomaly detection experiments
- `experiments/prova-chrono2-detection.ipynb` — Chronos-2 detection prototyping
- `experiments/sara_chronos2small_detection.ipynb` — Chronos-2 small model detection experiments
- `experiments/TSB_AD_chronos.ipynb` — TSB-AD benchmark experiments with Chronos
- `realTimeDetection/realTimeAnomalyDetectionMora.ipynb` — Real-time anomaly detection demo

This are experimental notebooks used for prototyping and exploration during the project. They are not directly related to the final report, but they contain useful insights and preliminary results that informed our final approaches. As such, they should be ignored in the context of the final report, but they can be a valuable resource for understanding our experimentation process.


---

# SAVED MODELS

Pre-trained model checkpoints are organized in several directories:

| Directory | Description |
|---|---|
| `Saved_Models/` | Token-aware transformer (Stage 1), hidden_dim=64, 24 epochs |
| `Saved_Models/two_stage/` | Two-stage model (Stage 1 + Set Transformer), 30 epochs |
| `Saved_Models/two_stage_v2/` | Two-stage model v2, 15 epochs |
| `SAVED_MODELS_SIMPLE/` | Simple feed-forward classifier, 40 epochs |
| `Saved_Models_Simpler/` | Simpler architecture variant, 40 epochs |
| `Saved_Models_Temporal/` | Temporal attention classifier (base), 30 epochs |
| `Saved_Models_Temporal/AGGRESSIVE/` | Aggressive training schedule variant |
| `Saved_Models_Temporal/HIGHLY_AGGRESSIVE/` | Highly aggressive training variant |
| `Saved_Models_Temporal/HIGHLY_AGGRESSIVE_AMPLIFIED/` | Amplified aggressive variant |
| `Saved_Models_Temporal/TRAINED_TSB_AD/` | Trained on TSB-AD data, hidden_dim=32 |
| `Saved_Models_Temporal/TRAINED_TSB_AD_64/` | Trained on TSB-AD data, hidden_dim=64 |
| `Saved_Models_Temporal/v1_*_v2_*/` | Ablation experiments varying hidden dimensions |

---

# RESULTS

Experiment results are organized by contributor in `results/`:

| Directory | Contents |
|---|---|
| `results/ALDO/` | Univariate + MultiVariateNew results |
| `results/NUNZIO/` | Univariate + MultiVariate (graph, PageRank) + Anomaly forecast results |
| `results/SARA/` | Univariate + MultiVariateNew results |
| `results/SIMONE/` | Univariate + MultiVariateNew results |
| `results/Valentino/` | Univariate results |
| `results/timing/` | Inference benchmarks (Colab and Mora server) |

---

