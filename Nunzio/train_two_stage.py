import torch
import numpy as np
import pandas as pd
import os
import sys
import re
import json
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    precision_recall_curve, average_precision_score
)

# Make imports work from Nunzio/ directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from two_stage_detector import (
    TwoStageMultivariateDetector,
    Stage2MultivariateDetector,
    TwoStageLoss,
    collate_multivariate_batch
)
from KIMI_RAIKKONEN import ChronosAnomalyDetector


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def get_file_stems_from_split_dir(split_dir: str) -> set:
    """
    Read a split directory (TRAIN_SPLIT or TEST_SPLIT) and return
    the set of file stems, e.g. {'0_Kaggle_labeled', '3_WADI_data_0', ...}.
    
    Files in split dirs are named {idx}_{dataset}.csv
    These map to {idx}_{dataset}_embeddings.npy in processed data.
    """
    return {f.replace('.csv', '') for f in os.listdir(split_dir) if f.endswith('.csv')}


# ═══════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════

class MultivariateAnomalyDataset(torch.utils.data.Dataset):
    """
    Dataset per anomaly detection multivariato da PROCESSED_TRAIN_DATAV2.
    
    Organizza i file per dataset name (es. WADI_data_0 con 15 serie).
    Per ogni (dataset, segment_t), campiona un timestep random da ogni
    serie producendo [D, 9, 768] input + labels.
    
    Filtra i file in base a file_filter (derivato da TRAIN_SPLIT / TEST_SPLIT).
    
    Memory: Usa memory-mapping (mmap) per evitare di caricare tutto in RAM.
    
    Data format:
        embeddings: [n_groups, n_timesteps, 9, 768] per serie
        ground_truth: binary labels [N, 1] per timestep
        predictions: v1-item_id mappa timestep → segment
    """
    
    def __init__(self, data_dir: str, file_filter: set = None, verbose: bool = True):
        """
        Args:
            data_dir: Path to PROCESSED_TRAIN_DATAV2 (shared embeddings/labels/preds)
            file_filter: Set of file stems to include, e.g. {'0_Kaggle_labeled', ...}.
                        Derived from TRAIN_SPLIT or TEST_SPLIT directory listing.
                        If None, loads all files.
            verbose: Print loading info
        """
        # 1. Discover and group files by dataset name, filtered by split
        dataset_files = defaultdict(list)
        pred_dir = os.path.join(data_dir, 'predictions')
        
        for idx, f in enumerate(sorted(os.listdir(os.path.join(data_dir, 'embeddings')))):
            if file_filter is None or f.replace("_embeddings.npy", "") in file_filter:
                dataset_files[f"{idx}_{f}"].append((idx, f))
        
        if verbose:
            print(f"Found {len(dataset_files)} files")
        
        # 2. Load metadata and labels for each dataset
        self.dataset_info = {}
        self.samples = []
        total_pos, total_neg = 0, 0
        
        for ds_name, file_list in tqdm(sorted(dataset_files.items()), desc="Processing datasets"):
            series_info = []
            min_n_groups = float('inf')
            
            for series_idx, emb_file in file_list:
                try:
                    # Get shape via mmap (no RAM usage)
                    emb = np.load(os.path.join(data_dir, 'embeddings', emb_file), mmap_mode='r')
                    n_groups = emb.shape[0]
                    min_n_groups = min(min_n_groups, n_groups)
                    
                    gt = pd.read_csv(os.path.join(data_dir, 'ground_truth_labels', emb_file.replace('embeddings.npy', 'ground_truth_labels.csv')), header=None).iloc[:-1, 0].values.astype(int)
                    
                    # Only load item_ids column (skip quantile cols for speed)
                    preds = pd.read_csv(os.path.join(pred_dir, emb_file.replace('embeddings.npy', 'predictions.csv')), usecols=[0])
                    item_ids = preds.iloc[:, 0].values.astype(int)
                    
                    # Per-segment labels (binary)
                    seg_labels = np.zeros(n_groups, dtype=np.int64)
                    for g_id in range(1, n_groups + 1):
                        mask = item_ids == g_id
                        if mask.any():
                            seg_labels[g_id - 1] = int(gt[mask].sum() >= 1)
                    
                    series_info.append({
                        'emb_path': os.path.join(data_dir, 'embeddings', emb_file),
                        'labels': seg_labels,
                        'n_groups': n_groups,
                        'n_ts': emb.shape[1],
                        'series_idx': series_idx,
                    })
                    
                except Exception as e:
                    if verbose:
                        print(f"  Error loading {emb_file}: {e}")
                    continue

            self.dataset_info[ds_name] = {
                'series_info': series_info,
                'n_segments': int(min_n_groups),
                'D': len(series_info),
            }
            
            # Use ALL segments for this split
            for seg_idx in range(int(min_n_groups)):
                # Compute global label for this segment
                global_label = 0
                for si in series_info:
                    if si['labels'][seg_idx] == 1:
                        global_label = 1
                        break
                
                self.samples.append((ds_name, seg_idx))
                if global_label:
                    total_pos += 1
                else:
                    total_neg += 1
        
        if verbose:
            total = len(self.samples)
            print(f"\nDataset summary:")
            print(f"  {total} multivariate samples from {len(self.dataset_info):,} datasets")
            print(f"  Positive (anomalous): {total_pos} ({100*total_pos/max(total,1):.1f}%)")
            print(f"  Negative (normal):    {total_neg} ({100*total_neg/max(total,1):.1f}%)")
    
    def get_sampler_weights(self):
        """Per-sample weights for WeightedRandomSampler (inverse frequency)."""
        labels = np.array([self._get_global_label(i) for i in range(len(self.samples))])
        n_pos = labels.sum()
        n = len(labels)
        weights = np.where(
            labels == 1,
            n / (2.0 * max(n_pos, 1)),
            n / (2.0 * max(n - n_pos, 1)),
        )
        return torch.from_numpy(weights).double()

    def _get_global_label(self, idx):
        """Compute global label for sample idx without loading embeddings."""
        ds_name, seg_idx = self.samples[idx]
        for sd in self.dataset_info[ds_name]['series_info']:
            if sd['labels'][seg_idx] == 1:
                return 1
        return 0

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            series_embeddings: [D, 9, 768] - one random timestep per series per segment
            global_label: 0 or 1
            series_labels: [D] binary tensor
        """
        ds_name, seg_idx = self.samples[idx]
        series_embeddings, series_labels = [], []
        
        for sd in self.dataset_info[ds_name]['series_info']:
            # Memory-mapped read of single segment
            seg_emb = np.load(sd['emb_path'], mmap_mode='r')[seg_idx]  # [n_ts, 9, 768]
            
            # Random timestep sampling (data augmentation)
            series_embeddings.append(np.array(seg_emb[np.random.randint(0, seg_emb.shape[0])], dtype=np.float32))
            series_labels.append(sd['labels'][seg_idx])
        
        global_label = torch.tensor([int(max(series_labels))], dtype=torch.float32)
        return torch.from_numpy(np.stack(series_embeddings)), global_label, torch.tensor(series_labels, dtype=torch.long)


# ═══════════════════════════════════════════════════════════════
# TRAINING & VALIDATION
# ═══════════════════════════════════════════════════════════════

def train_epoch(model, dataloader, criterion, optimizer, device, 
                use_amp: bool = True, max_grad_norm: float = 1.0):
    """Train one epoch."""
    model.train()
    scaler = torch.amp.GradScaler("cuda") if use_amp and device.type == 'cuda' else None
    
    total_loss = 0.0
    all_global_preds, all_global_labels = [], []
    loss_breakdown = {}
    
    for embeddings_list, global_labels, series_labels_list in tqdm(dataloader, desc="Train", leave=False):
        # Move to device
        embeddings_list = [emb.to(device) for emb in embeddings_list]
        global_labels = global_labels.to(device)
        series_labels_list = [sl.to(device) for sl in series_labels_list]
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                global_logits, series_logits_list, _ = model.forward_batch(embeddings_list)
                loss, loss_dict = criterion(global_logits, series_logits_list,global_labels, series_labels_list)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            global_logits, series_logits_list, _ = model.forward_batch(embeddings_list)
            loss, loss_dict = criterion(global_logits, series_logits_list,global_labels, series_labels_list)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
        
        # Accumulate
        total_loss += loss.item()
        for k in loss_dict:
            loss_breakdown[k] = loss_dict[k] + loss_breakdown.get(k, 0.0)
        
        # Global predictions
        all_global_preds.extend((torch.sigmoid(global_logits.detach()).cpu().numpy() > 0.5).astype(int))
        all_global_labels.extend(global_labels.cpu().numpy())
    
    n_batches = max(len(dataloader), 1)
    all_global_preds = np.array(all_global_preds)
    all_global_labels = np.array(all_global_labels).astype(int)
    
    return {
        'loss': float(total_loss / n_batches),
        'loss_global': float(loss_breakdown.get('global', 0.0) / n_batches),
        'loss_series': float(loss_breakdown.get('series', 0.0) / n_batches),
        'loss_consistency': float(loss_breakdown.get('consistency', 0.0) / n_batches),
        'f1': float(f1_score(all_global_labels, all_global_preds, zero_division=0)),
        'precision': float(precision_score(all_global_labels, all_global_preds, zero_division=0)),
        'recall': float(recall_score(all_global_labels, all_global_preds, zero_division=0)),
    }

@torch.no_grad()
def validate_epoch(model, dataloader, criterion, device, threshold: float = 0.5):
    """Validate and compute metrics."""
    model.eval()
    
    total_loss = 0.0
    all_global_probs, all_global_labels = [], []
    all_series_preds, all_series_labels= [], []
    
    for embeddings_list, global_labels, series_labels_list in tqdm(dataloader, desc="Valid", leave=False):
        embeddings_list, global_labels, series_labels_list = [emb.to(device) for emb in embeddings_list], global_labels.to(device), [sl.to(device) for sl in series_labels_list]        
        global_logits, series_logits_list, _ = model.forward_batch(embeddings_list)
        
        loss, _ = criterion(global_logits, series_logits_list,global_labels, series_labels_list)
        total_loss += loss.item()
        
        # Global
        all_global_probs.extend(torch.sigmoid(global_logits).cpu().numpy())
        all_global_labels.extend(global_labels.cpu().numpy())
        
        # Per-series
        for s_logits, s_labels in zip(series_logits_list, series_labels_list):
            all_series_preds.extend((torch.sigmoid(s_logits.squeeze(-1)).cpu().numpy() > threshold).astype(int))
            all_series_labels.extend(s_labels.cpu().numpy())
    
    n_batches = max(len(dataloader), 1)
    all_global_probs = np.array(all_global_probs)
    all_global_labels = np.array(all_global_labels).astype(int)
    global_preds = (all_global_probs > threshold).astype(int)
    
    all_series_preds = np.array(all_series_preds)
    all_series_labels = np.array(all_series_labels).astype(int)
    
    # Optimal threshold via PR curve (global)
    try:
        p_curve, r_curve, thresholds = precision_recall_curve(all_global_labels, all_global_probs)
        f1_curve = 2 * p_curve * r_curve / np.maximum(p_curve + r_curve, 1e-8)
        best_idx = f1_curve.argmax()
        optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        optimal_f1 = f1_curve[best_idx]
        auc_pr = average_precision_score(all_global_labels, all_global_probs)
    except Exception:
        optimal_threshold = 0.5
        optimal_f1 = 0.0
        auc_pr = 0.0
    
    return {
        'loss': float(total_loss / n_batches),
        'global_f1': float(f1_score(all_global_labels, global_preds, zero_division=0)),
        'global_precision': float(precision_score(all_global_labels, global_preds, zero_division=0)),
        'global_recall': float(recall_score(all_global_labels, global_preds, zero_division=0)),
        'global_auc_pr': float(auc_pr),
        'global_optimal_f1': float(optimal_f1),
        'global_optimal_threshold': float(optimal_threshold),
        'series_f1': float(f1_score(all_series_labels, all_series_preds, zero_division=0)),
        'series_precision': float(precision_score(all_series_labels, all_series_preds, zero_division=0)),
        'series_recall': float(recall_score(all_series_labels, all_series_preds, zero_division=0)),
    }

# ═══════════════════════════════════════════════════════════════
# FEATURE PRE-EXTRACTION (for frozen Stage 1, optional speedup)
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_stage1_features(stage1_model, data_dir, save_dir, device, batch_size=256):
    """
    Pre-extract Stage 1 features for all embedding files.
    Saves [n_groups, hidden_dim] per file (mean-pooled across timesteps).
    
    This is optional — use for faster training with frozen Stage 1.
    
    Args:
        stage1_model: ChronosAnomalyDetector
        data_dir: PROCESSED_TRAIN_DATAV2
        save_dir: Where to save extracted features
        device: torch device
        batch_size: Batch size for Stage 1 inference
    """
    stage1_model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    emb_dir = os.path.join(data_dir, 'embeddings')
    
    for fname in tqdm(sorted([f for f in os.listdir(emb_dir) if f.endswith('_embeddings.npy')]), desc="Extracting Stage 1 features"):
        out_path = os.path.join(save_dir, fname.replace('_embeddings.npy', '_features.npy'))
        if os.path.exists(out_path):
            continue
        
        try:
            emb = np.load(os.path.join(emb_dir, fname), mmap_mode='r')
            n_groups, n_ts = emb.shape[0], emb.shape[1]
            # Process all (group, timestep) pairs through Stage 1
            all_features = []
            
            for g in range(n_groups):
                seg_emb = torch.from_numpy(np.array(emb[g], dtype=np.float32)).to(device)  # [n_ts, 9, 768]
                all_features.append(np.concatenate([stage1_model(seg_emb[i:i+batch_size], returnLogits=False).cpu().numpy() for i in range(0, n_ts, batch_size)], axis=0).mean(axis=0))

            np.save(out_path, np.stack(all_features).astype(np.float32))            
        except Exception as e:
            print(f"Error extracting {fname}: {e}")


class PrecomputedDataset(torch.utils.data.Dataset):
    """
    Dataset using pre-extracted Stage 1 features (for frozen Stage 1).
    Loads [n_groups, hidden_dim] features instead of [n_groups, n_ts, 9, 768] embeddings.
    Much faster and more memory-efficient.
    """
    
    def __init__(self, features_dir, data_dir, file_filter=None,  verbose=True):
        """
        Args:
            features_dir: Directory with pre-extracted features (*_features.npy)
            data_dir: PROCESSED_TRAIN_DATAV2 (for labels)
            file_filter: Set of file stems to include (from split dir), or None for all
        """
        gt_dir = os.path.join(data_dir, 'ground_truth_labels')
        pred_dir = os.path.join(data_dir, 'predictions')
        
        # Parse feature files, filtered by split
        dataset_files = defaultdict(list)
        
        
        for idx, f in enumerate(sorted(os.listdir(os.path.join(data_dir, 'embeddings')))):
            if file_filter is None or f.replace("_embeddings.npy", "") in file_filter:
                dataset_files[f"{idx}_{f}"].append((idx, f))

        self.dataset_info = {}
        self.samples = []
        
        for ds_name, file_list in sorted(dataset_files.items()):
            series_info = []
            min_n_groups = float('inf')
            
            for series_idx, feat_file in file_list:
                try:
                    feat = np.load(os.path.join(features_dir, feat_file), mmap_mode='r')
                    n_groups = feat.shape[0]
                    min_n_groups = min(min_n_groups, n_groups)
                    
                    gt = pd.read_csv(os.path.join(gt_dir, feat_file.replace('_features.npy', '_ground_truth_labels.csv')), header=None).iloc[:-1, 0].values.astype(int)
                    
                    # Only load item_ids column (skip quantile cols for speed)
                    preds = pd.read_csv(os.path.join(pred_dir, feat_file.replace('_features.npy', '_predictions.csv')), usecols=[0])
                    
                    item_ids = preds.iloc[:, 0].values.astype(int)
                    
                    seg_labels = np.zeros(n_groups, dtype=np.int64)
                    for g_id in range(1, n_groups + 1):
                        mask = item_ids == g_id
                        if mask.any():
                            seg_labels[g_id - 1] = int(gt[mask].sum() >= 1)
                    
                    series_info.append({
                        'feat_path': os.path.join(features_dir, feat_file),
                        'labels': seg_labels,
                        'n_groups': n_groups,
                        'series_idx': series_idx,
                    })
                except Exception as e:
                    if verbose:
                        print(f"  Error: {feat_file}: {e}")
                    continue
            
            self.dataset_info[ds_name] = {
                'series_info': series_info,
                'n_segments': int(min_n_groups),
                'D': len(series_info),
            }
            
            for seg_idx in range(int(min_n_groups)):
                self.samples.append((ds_name, seg_idx))
        
        if verbose:
            print(f"PrecomputedDataset: {len(self.samples)} samples from {len(self.dataset_info)} datasets")
    
    def get_sampler_weights(self):
        """Per-sample weights for WeightedRandomSampler (inverse frequency)."""
        labels = np.array([self._get_global_label(i) for i in range(len(self.samples))])
        n_pos = labels.sum()
        n = len(labels)
        weights = np.where(
            labels == 1,
            n / (2.0 * max(n_pos, 1)),
            n / (2.0 * max(n - n_pos, 1)),
        )
        return torch.from_numpy(weights).double()

    def _get_global_label(self, idx):
        """Compute global label for sample idx without loading features."""
        ds_name, seg_idx = self.samples[idx]
        for sd in self.dataset_info[ds_name]['series_info']:
            if sd['labels'][seg_idx] == 1:
                return 1
        return 0

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Returns features [D, hidden_dim], global_label, series_labels [D]."""
        ds_name, seg_idx = self.samples[idx]
        
        features, labels = [], []        
        for sd in self.dataset_info[ds_name]['series_info']:
            features.append(np.array(np.load(sd['feat_path'], mmap_mode='r')[seg_idx], dtype=np.float32))
            labels.append(sd['labels'][seg_idx])
        
        return torch.from_numpy(np.stack(features)), int(max(labels)), torch.tensor(labels, dtype=torch.long)      # [D]

def collate_precomputed_batch(batch):
    """Collate for PrecomputedDataset (variable D)."""
    features_list, global_labels, series_labels_list = [], [], []
    
    for features, g_label, s_labels in batch:
        features_list.append(features), global_labels.append(g_label), series_labels_list.append(s_labels)
    
    return features_list, torch.tensor(global_labels, dtype=torch.float32), series_labels_list


def train_epoch_precomputed(stage2_model, dataloader, criterion, optimizer, 
                            device, use_amp=True, max_grad_norm=1.0):
    """Train Stage 2 directly on pre-extracted features."""
    stage2_model.train()
    scaler = torch.amp.GradScaler("cuda") if use_amp and device.type == 'cuda' else None
    
    total_loss = 0.0
    all_global_preds, all_global_labels = [], []
    loss_breakdown = {'global': 0.0, 'series': 0.0, 'consistency': 0.0}
    
    for features_list, global_labels, series_labels_list in tqdm(dataloader, desc="Train", leave=False):
        global_labels, series_labels_list =global_labels.to(device), [sl.to(device) for sl in series_labels_list]
        
        optimizer.zero_grad()
        
        # Process each sample independently (variable D)
        batch_global_logits, batch_series_logits = [], []
        
        for features in features_list:
            g_logit, s_logits = stage2_model(features.to(device).unsqueeze(0))
            batch_global_logits.append(g_logit.squeeze())  # scalar
            batch_series_logits.append(s_logits.squeeze(0))  # [D, 1]
        
        batch_global_logits = torch.stack(batch_global_logits)  # [B]
        
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                loss, loss_dict = criterion(batch_global_logits, batch_series_logits,global_labels, series_labels_list)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(stage2_model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, loss_dict = criterion(batch_global_logits, batch_series_logits,global_labels, series_labels_list)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(stage2_model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
        
        total_loss += loss.item()
        for k in loss_breakdown:
            loss_breakdown[k] += loss_dict[k]
        
        all_global_preds.extend((torch.sigmoid(batch_global_logits.detach()).cpu().numpy() > 0.5).astype(int))
        all_global_labels.extend(global_labels.cpu().numpy())
    
    n_batches = max(len(dataloader), 1)
    all_global_preds = np.array(all_global_preds)
    all_global_labels = np.array(all_global_labels).astype(int)
    
    return {
        'loss': float(total_loss / n_batches),
        'loss_global': float(loss_breakdown['global'] / n_batches),
        'loss_series': float(loss_breakdown['series'] / n_batches),
        'loss_consistency': float(loss_breakdown['consistency'] / n_batches),
        'f1': float(f1_score(all_global_labels, all_global_preds, zero_division=0)),
        'precision': float(precision_score(all_global_labels, all_global_preds, zero_division=0)),
        'recall': float(recall_score(all_global_labels, all_global_preds, zero_division=0)),
    }


@torch.no_grad()
def validate_epoch_precomputed(stage2_model, dataloader, criterion, device, threshold=0.5):
    """Validate Stage 2 on pre-extracted features."""
    stage2_model.eval()
    
    total_loss = 0.0
    all_global_probs, all_global_labels = [], []
    all_series_preds, all_series_labels= [], []
    
    for features_list, global_labels, series_labels_list in tqdm(dataloader, desc="Valid", leave=False):
        global_labels, series_labels_list =global_labels.to(device), [sl.to(device) for sl in series_labels_list]
        
        batch_global_logits, batch_series_logits = [], []        
        for features in features_list:
            g_logit, s_logits = stage2_model(features.to(device).unsqueeze(0))
            batch_global_logits.append(g_logit.squeeze()), batch_series_logits.append(s_logits.squeeze(0))
        
        batch_global_logits = torch.stack(batch_global_logits)
        
        loss, _ = criterion(batch_global_logits, batch_series_logits,global_labels, series_labels_list)
        total_loss += loss.item()
        
        all_global_probs.extend(torch.sigmoid(batch_global_logits).cpu().numpy())
        all_global_labels.extend(global_labels.cpu().numpy())
        
        for s_logits, s_labels in zip(batch_series_logits, series_labels_list):
            all_series_preds.extend((torch.sigmoid(s_logits.squeeze(-1)).cpu().numpy() > threshold).astype(int))
            all_series_labels.extend(s_labels.cpu().numpy())
    
    n_batches = max(len(dataloader), 1)
    all_global_probs = np.array(all_global_probs)
    all_global_labels = np.array(all_global_labels).astype(int)
    global_preds = (all_global_probs > threshold).astype(int)
    all_series_preds = np.array(all_series_preds)
    all_series_labels = np.array(all_series_labels).astype(int)
    
    try:
        auc_pr = average_precision_score(all_global_labels, all_global_probs)
        p_curve, r_curve, thresholds = precision_recall_curve(all_global_labels, all_global_probs)
        f1_curve = 2 * p_curve * r_curve / np.maximum(p_curve + r_curve, 1e-8)
        best_idx = f1_curve.argmax()
        optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        optimal_f1 = f1_curve[best_idx]
    except Exception:
        auc_pr, optimal_threshold, optimal_f1 = 0.0, 0.5, 0.0
    
    return {
        'loss': float(total_loss / n_batches),
        'global_f1': float(f1_score(all_global_labels, global_preds, zero_division=0)),
        'global_precision': float(precision_score(all_global_labels, global_preds, zero_division=0)),
        'global_recall': float(recall_score(all_global_labels, global_preds, zero_division=0)),
        'global_auc_pr': float(auc_pr),
        'global_optimal_f1': float(optimal_f1),
        'global_optimal_threshold': float(optimal_threshold),
        'series_f1': float(f1_score(all_series_labels, all_series_preds, zero_division=0)),
        'series_precision': float(precision_score(all_series_labels, all_series_preds, zero_division=0)),
        'series_recall': float(recall_score(all_series_labels, all_series_preds, zero_division=0)),
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    cfg = {
        # ── Data ──
        'data_dir':           './PROCESSED_TRAIN_DATAV2',
        'data_dir2':          './PROCESSED_TRAIN_DATAV5',
        'train_split_dir':    './TRAIN_SPLIT',
        'train_split_dir2':   './TRAIN_MULTI_CLEAN',
        'test_split_dir':     './TEST_SPLIT',
        'test_split_dir2':     './TEST_MULTI_CLEAN',
        'stage1_checkpoint':  './Saved_Models_Temporal/TRAINED_TSB_AD/best_model.pth', # add _64 for 64 model


        # ── Model ──
        'stage1_hidden_dim':  32,   # must match Stage 1 checkpoint
        'stage2_hidden_dim':  32,
        'num_isab_layers':    1,
        'num_inducing':      16,
        'num_heads':          2,
        'dropout':            0.4,

        # ── Freeze / Finetune ──
        'finetune_stage1':    True, # True = end-to-end con differential LR
        'stage1_lr_factor':   0.25,   # LR multiplier per Stage 1 se finetune
        'precompute_features': False,
        'features_dir':       None,  # default: data_dir/stage1_features

        # ── Training ──
        'num_epochs':         40,
        'batch_size':         128,
        'lr':                 5e-4,
        'weight_decay':       0.04,
        'patience':           5,
        "max_grad_norm":      2.0,
        "label_smoothing":    0.1,   # for global binary labels

        # ── Loss ──
        'loss_alpha':         2.0,   # global loss weight
        'loss_beta':          -1.0,   # series loss weight
        'loss_gamma':         1,   # consistency loss weight
        'focal_alpha':        0.55,   # focal loss alpha (pos class weight)
        'focal_gamma':        2.0,

        # ── Misc ──
        'save_dir':           './saved_models/two_stage/',
        'use_amp':            True,
        'seed':               42,
        'num_workers':        0,     # set >0 for faster data loading (may cause issues in some environments)
        "max_grad_norm":       3.0,
    }

    # Seed
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features_dir = cfg['features_dir'] or os.path.join(cfg['data_dir'], 'stage1_features')

    print(f"Device: {device}")
    print(f"Mode: {'Finetune Stage 1' if cfg['finetune_stage1'] else 'Frozen Stage 1'}")
    
    # ── Load Stage 1 model ──────────────────────────────────────
    print("\nLoading Stage 1 (ChronosAnomalyDetector)...")
    stage1_model = ChronosAnomalyDetector(embed_dim=768,hidden_dim=cfg['stage1_hidden_dim'],num_heads=4,dropout=0.15)
    
    if os.path.exists(cfg['stage1_checkpoint']):
        checkpoint = torch.load(cfg['stage1_checkpoint'], map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            stage1_model.load_state_dict(checkpoint['model_state_dict'])
        elif 'ema_state_dict' in checkpoint:
            stage1_model.load_state_dict(checkpoint['ema_state_dict'])
        else:
            stage1_model.load_state_dict(checkpoint)
        print(f"  Loaded checkpoint: {cfg['stage1_checkpoint']}")
    else:
        print(f"  WARNING: Checkpoint not found: {cfg['stage1_checkpoint']}")
        print(f"  Using randomly initialized Stage 1")
    
    # ── Pre-extract features if requested ───────────────────────
    if cfg['precompute_features'] and not cfg['finetune_stage1']:
        print(f"\nPre-extracting Stage 1 features to {features_dir}...")
        stage1_model.to(device)
        extract_stage1_features(stage1_model, cfg['data_dir'], features_dir, device)
        stage1_model.cpu()
        print("  Done!")
    
    # ── Create datasets ─────────────────────────────────────────
    print("\nLoading data...")
    
    # Build file-filter sets from split directories
    train_stems = set(get_file_stems_from_split_dir(cfg['train_split_dir']))
    test_stems =  set(get_file_stems_from_split_dir(cfg['test_split_dir']))
    print(f"  Train split: {len(train_stems)} files from {cfg['train_split_dir']}")
    print(f"  Test split:  {len(test_stems)} files from {cfg['test_split_dir']}")
    
    use_precomputed = not cfg['finetune_stage1'] and os.path.isdir(features_dir) and len(os.listdir(features_dir)) > 0
    
    if use_precomputed:
        print("  Using pre-computed Stage 1 features")
        train_dataset = PrecomputedDataset(features_dir, cfg['data_dir'], file_filter=train_stems, verbose=True)
        val_dataset = PrecomputedDataset(features_dir, cfg['data_dir'],file_filter=test_stems, verbose=True)
        collate_fn = collate_precomputed_batch
    else:
        print("  Using raw embeddings (full pipeline)")
        train_dataset = MultivariateAnomalyDataset(cfg['data_dir'], file_filter=train_stems, verbose=True)
        val_dataset = MultivariateAnomalyDataset(cfg['data_dir'], file_filter=test_stems, verbose=True)
        collate_fn = collate_multivariate_batch
    
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")

    if cfg['train_split_dir2'] is not None and os.path.isdir(cfg['train_split_dir2']):
        train_stems2 = get_file_stems_from_split_dir(cfg['train_split_dir2'])
        print(f"  Additional Train split: {len(train_stems2)} files from {cfg['train_split_dir2']}")
        
        if use_precomputed:
            extra_train_dataset = PrecomputedDataset(features_dir, cfg['data_dir2'], file_filter=train_stems2, verbose=True)
        else:
            extra_train_dataset = MultivariateAnomalyDataset(cfg['data_dir2'], file_filter=train_stems2, verbose=True)
        
        print(f"  Additional Train: {len(extra_train_dataset)} samples")
        del train_stems2

    if cfg['test_split_dir2'] is not None and os.path.isdir(cfg['test_split_dir2']):
        test_stems2 = get_file_stems_from_split_dir(cfg['test_split_dir2'])
        print(f"  Additional Test split: {len(test_stems2)} files from {cfg['test_split_dir2']}")
        
        if use_precomputed:
            extra_val_dataset = PrecomputedDataset(features_dir, cfg['data_dir2'], file_filter=test_stems2, verbose=True)
        else:
            extra_val_dataset = MultivariateAnomalyDataset(cfg['data_dir2'], file_filter=test_stems2, verbose=True)
        
        print(f"  Additional Val: {len(extra_val_dataset)} samples")
        val_dataset = torch.utils.data.ConcatDataset([val_dataset, extra_val_dataset])
        del test_stems2, extra_val_dataset

    del train_stems, test_stems
    
    train_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([train_dataset, extra_train_dataset]) if 'extra_train_dataset' in locals() else train_dataset,
                                            batch_size=cfg['batch_size'], collate_fn=collate_fn, num_workers=cfg['num_workers'], pin_memory=True,
                sampler=torch.utils.data.WeightedRandomSampler(torch.cat([train_dataset.get_sampler_weights(), extra_train_dataset.get_sampler_weights()]) 
                                                                if 'extra_train_dataset' in locals() else train_dataset.get_sampler_weights(), 
                                                        num_samples=len(train_dataset) + (len(extra_train_dataset) if 'extra_train_dataset' in locals() else 0), replacement=True))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False,collate_fn=collate_fn, num_workers=cfg['num_workers'], pin_memory=True)

    # ── Create model ────────────────────────────────────────────
    if use_precomputed:
        # Train only Stage 2 (directly on features)
        print("\nCreating Stage 2 model...")
        model = Stage2MultivariateDetector(
            feature_dim=cfg['stage1_hidden_dim'],
            hidden_dim=cfg['stage2_hidden_dim'],
            num_inducing=cfg['num_inducing'],
            num_isab_layers=cfg['num_isab_layers'],
            num_heads=cfg['num_heads'],
            dropout=cfg['dropout'],
        ).to(device)
        
        train_fn = train_epoch_precomputed
        val_fn = validate_epoch_precomputed
        
    else:
        # Full two-stage model
        print("\nCreating two-stage model...")
        model = TwoStageMultivariateDetector(
            stage1_model=stage1_model,
            stage1_hidden_dim=cfg['stage1_hidden_dim'],
            stage2_hidden_dim=cfg['stage2_hidden_dim'],
            num_isab_layers=cfg['num_isab_layers'],
            num_heads=cfg['num_heads'],
            freeze_stage1=not cfg['finetune_stage1'],
            dropout=cfg['dropout'],
        ).to(device)
        
        train_fn = train_epoch
        val_fn = validate_epoch
    
    print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} / {sum(p.numel() for p in model.parameters()):,} params")
    
    # ── Optimizer & Scheduler ───────────────────────────────────
    criterion = TwoStageLoss(alpha=cfg['loss_alpha'],beta=cfg['loss_beta'], gamma=cfg['loss_gamma'],focal_gamma=cfg['focal_gamma'], label_smoothing=cfg['label_smoothing'])
    
    if use_precomputed or not cfg['finetune_stage1']:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    else:
        # Differential LR for finetuning
        param_groups = model.get_parameter_groups(base_lr=cfg['lr'], stage1_lr_factor=cfg['stage1_lr_factor'])
        optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg['weight_decay'])
        print(f"  Stage 1 LR: {cfg['lr'] * cfg['stage1_lr_factor']:.1e}")
        print(f"  Stage 2 LR: {cfg['lr']:.1e}")
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # ── Training loop ───────────────────────────────────────────
    print(f"\n{'='*60}", "Starting training...", f"{'='*60}", sep="\n")
    
    best_f1 = 0.0
    patience_counter = 0
    history = []
    
    os.makedirs(cfg['save_dir'], exist_ok=True)
    
    # Save config alongside checkpoints
    with open(os.path.join(cfg['save_dir'], 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)
    
    for epoch in range(cfg['num_epochs']):
        print(f"\nEpoch {epoch+1}/{cfg['num_epochs']}\n", '-' * 40)
        
        # Train
        train_metrics = train_fn(model, train_loader, criterion, optimizer, device, use_amp=cfg.get('use_amp', True), max_grad_norm=cfg.get('max_grad_norm', 1.0))
        
        print(f"  Train | Loss: {train_metrics['loss']:.4f} (G:{train_metrics['loss_global']:.3f} S:{train_metrics['loss_series']:.3f} C:{train_metrics['loss_consistency']:.3f})")
        print(f"       | F1: {train_metrics['f1']:.4f} P:{train_metrics['precision']:.3f} R:{train_metrics['recall']:.3f}")
        
        # Validate
        val_metrics = val_fn(model, val_loader, criterion, device)
        
        print(f"  Val   | Loss: {val_metrics['loss']:.4f}")
        print(f"       | Global F1: {val_metrics['global_f1']:.4f} P:{val_metrics['global_precision']:.3f} R:{val_metrics['global_recall']:.3f} AUC-PR:{val_metrics.get('global_auc_pr', 0):.3f}")
        print(f"       | Series F1: {val_metrics['series_f1']:.4f} P:{val_metrics['series_precision']:.3f} R:{val_metrics['series_recall']:.3f}")
        
        if 'global_optimal_f1' in val_metrics:
            print(f"       | Optimal F1: {val_metrics['global_optimal_f1']:.4f} (threshold: {val_metrics['global_optimal_threshold']:.3f})")
        
        # Track history
        history.append({'epoch': epoch + 1,'train': train_metrics,'val': val_metrics,})
        
        # Use global_optimal_f1 for model selection if available
        target_f1 = val_metrics.get('global_optimal_f1', val_metrics['global_f1'])
        
        # Save best
        if target_f1 > best_f1:
            best_f1, patience_counter = target_f1, 0
            
            save_dict = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'val_metrics': val_metrics,
                'cfg': cfg,
                'mode': 'precomputed' if use_precomputed else ('finetune' if cfg['finetune_stage1'] else 'frozen'),
            }
            torch.save(save_dict, os.path.join(cfg['save_dir'], 'best_model.pth'))
            print(f"  >>> New best F1: {best_f1:.4f} (saved)")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{cfg['patience']})")
            
            if patience_counter >= cfg['patience']:
                print("\nEarly stopping!")
                break
        
        # Scheduler step
        scheduler.step(target_f1)
        
        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'cfg': cfg,
            }, os.path.join(cfg['save_dir'], f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Save history
    with open(os.path.join(cfg['save_dir'], 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training complete! Best F1: {best_f1:.4f}")
    print(f"Saved to: {cfg['save_dir']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
