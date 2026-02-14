"""
Training Loop per SimpleQuantileAnomalyDetector.

Redesign ispirato a KIMI_RAIKKONEN.py e train_two_stage.py:
  - Dataset unificato: MergedQuantileDataset carica tutti i file da V2+V5
  - WeightedRandomSampler per bilanciamento anomalie/normali
  - SigmoidFocalLoss (singola, pulizia, .mean() reduction)
  - OneCycleLR scheduler (warmup → peak → cosine decay)
  - EMA model per valutazione più smooth
  - Mixed precision (AMP) + gradient clipping + gradient accumulation
  - Threshold optimization via PR curve su validation
  - Early stopping su val best_f1
"""
import os, sys, copy, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import lru_cache
from collections import OrderedDict

from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score
)
from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Ensure Nunzio/ is on sys.path for sibling imports
_NUNZIO_DIR = os.path.dirname(os.path.abspath(__file__))
if _NUNZIO_DIR not in sys.path:
    sys.path.insert(0, _NUNZIO_DIR)

from SimplerNN import SimpleQuantileAnomalyDetector, SigmoidFocalLoss
from customDataLoaderV2 import CustomDataset

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════
# DATASET — Merged flat dataset from PROCESSED_TRAIN_DATAV2 + V5
# ═══════════════════════════════════════════════════════════════

class MergedQuantileDataset(torch.utils.data.Dataset):
    """
    Flat dataset that merges all quantile prediction files from multiple
    processed data directories (e.g. PROCESSED_TRAIN_DATAV2 + V5).

    Phase 1 (init): Scan all files, compute per-segment anomaly labels.
    Phase 2 (__getitem__): Lazily load data via cached CustomDataset instances.

    Each sample: (quantiles [D, T, Q], values [D, T], labels [T, 1])
    where D = num variables, T = prediction_length, Q = num quantile levels.

    Note: D may vary across files (different multivariate datasets).
    Use batch_size=1 + gradient accumulation for correct batching.
    """

    def __init__(
        self,
        sources: list[tuple[str, str]],
        skip_names: set = None,
        min_variables: int = 2,
        verbose: bool = True,
        dataset_label: str = "dataset",
    ):
        """
        Args:
            sources: list of (processed_dir, original_csv_dir) pairs.
                     e.g. [('./PROCESSED_TRAIN_DATAV2', './TRAIN_SPLIT'),
                            ('./PROCESSED_TRAIN_DATAV5', './TRAIN_MULTI_CLEAN')]
            skip_names: set of substrings to skip (e.g. {"WADI"})
            min_variables: minimum D (skip univariate to avoid BatchNorm issues)
            verbose: print loading info
            dataset_label: label for tqdm progress bar
        """
        super().__init__()
        self._sources = sources
        self._skip_names = skip_names or set()
        self._min_variables = min_variables

        # Samples: (src_idx, filename, item_id)
        self.samples = []
        self.labels = []  # per-sample binary label (1=anomalous segment)
        total_pos, total_neg = 0, 0
        skipped = 0

        for src_idx, (processed_dir, original_dir) in enumerate(sources):
            if not os.path.isdir(original_dir):
                if verbose:
                    print(f"  Directory not found: {original_dir}, skipping")
                continue

            files = sorted([f for f in os.listdir(original_dir) if f.endswith('.csv')])
            for fname in tqdm(files, desc=f"  Scanning {dataset_label} (source {src_idx})",
                              disable=not verbose, leave=False):
                # Skip blacklisted names
                if any(skip in fname.upper() for skip in self._skip_names):
                    skipped += 1
                    continue

                try:
                    base = fname.rsplit('.', 1)[0]
                    pred_path = os.path.join(processed_dir, "predictions",
                                             base + "_predictions.csv")
                    gt_path = os.path.join(processed_dir, "ground_truth_labels",
                                           base + "_ground_truth_labels.csv")

                    if not os.path.exists(pred_path) or not os.path.exists(gt_path):
                        skipped += 1
                        continue

                    # Quick scan: read only item_id column + ground truth
                    preds_first_col = pd.read_csv(pred_path, usecols=[0], header=0)
                    item_ids_col = preds_first_col.iloc[:, 0].values.astype(int)
                    gt = pd.read_csv(gt_path, header=None).iloc[:-1, 0].values.astype(int)

                    # Check number of variables (D) and quantile uniformity
                    # Read header only to count prefixes + quantile cols per var
                    header = pd.read_csv(pred_path, nrows=0).columns.tolist()
                    prefix_qcounts = {}  # prefix → number of quantile cols
                    for c in header:
                        parts = c.split('-', 1)
                        if len(parts) == 2:
                            prefix, suffix = parts
                            if suffix not in ('item_id', 'timestep'):
                                try:
                                    float(suffix)
                                    prefix_qcounts[prefix] = prefix_qcounts.get(prefix, 0) + 1
                                except ValueError:
                                    pass
                    n_vars = len(prefix_qcounts)

                    if n_vars < min_variables:
                        skipped += 1
                        continue

                    # All variables must have the same number of quantile cols
                    q_counts = list(prefix_qcounts.values())
                    if len(set(q_counts)) > 1 or min(q_counts) == 0:
                        skipped += 1
                        continue

                    # Per-segment labels
                    unique_ids = sorted(set(item_ids_col.tolist()))
                    for item_id in unique_ids:
                        mask = item_ids_col == item_id
                        label = int(gt[mask].sum() >= 1)
                        self.samples.append((src_idx, fname, item_id))
                        self.labels.append(label)
                        if label:
                            total_pos += 1
                        else:
                            total_neg += 1

                except Exception as e:
                    if verbose:
                        tqdm.write(f"    Error scanning {fname}: {e}")
                    skipped += 1
                    continue

        self.labels = np.array(self.labels, dtype=np.int64)

        if verbose:
            total = len(self.samples)
            print(f"  {dataset_label}: {total:,} samples | "
                  f"Normal: {total_neg:,} | Anomaly: {total_pos:,} "
                  f"({100 * total_pos / max(total, 1):.1f}%) | Skipped: {skipped}")

        # LRU cache for CustomDataset instances (bounded)
        self._cache = OrderedDict()
        self._cache_max = 64

    def _load_dataset(self, src_idx: int, fname: str) -> CustomDataset:
        """Lazily load and cache a CustomDataset for a file."""
        key = (src_idx, fname)
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]

        processed_dir, original_dir = self._sources[src_idx]
        ds = CustomDataset(fname, processed_dir, original_dir,
                           skipNames=set(), context_length=100)
        self._cache[key] = ds

        # Evict oldest if over limit
        while len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)

        return ds

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            quantiles: [D, T, Q]
            values:    [D, T]
            labels:    [T, 1]
        """
        src_idx, fname, item_id = self.samples[idx]
        ds = self._load_dataset(src_idx, fname)

        # Bypass CustomDataset's randomMapping — access by item_id directly
        row_indices = ds.indexes[item_id]
        raw = ds._data_values[row_indices]

        if ds.timestep_col_indices:
            timesteps = raw[:, ds.timestep_col_indices[0]].astype(int)
        else:
            timesteps = np.arange(len(row_indices))

        # Build quantiles [D, T, Q] – trim to min Q across variables
        per_var = [raw[:, qi] for qi in ds.quantile_col_indices]
        min_q = min(arr.shape[-1] for arr in per_var) if per_var else 0
        quantiles = torch.tensor(
            np.stack([arr[:, :min_q] for arr in per_var], axis=0),
            dtype=torch.float32
        )  # [D, T, Q]

        values = torch.tensor(
            ds._original_values[
                np.clip(timesteps, 0, len(ds._original_values) - 1)
            ].T,
            dtype=torch.float32
        )  # [D, T]

        labels = torch.tensor(
            ds.gt[row_indices], dtype=torch.float32
        ).unsqueeze(-1)  # [T, 1]

        return quantiles, values, labels

    def get_sampler_weights(self):
        """Per-sample weights for WeightedRandomSampler (inverse frequency)."""
        n_pos = self.labels.sum()
        n = len(self.labels)
        return torch.from_numpy(
            np.where(
                self.labels == 1,
                n / (2.0 * max(n_pos, 1)),
                n / (2.0 * max(n - n_pos, 1)),
            )
        ).double()


# ═══════════════════════════════════════════════════════════════
# TRAINER — Clean training loop (KIMI_RAIKKONEN-style)
# ═══════════════════════════════════════════════════════════════

class Trainer:
    """
    Trainer for SimpleQuantileAnomalyDetector.

    Inspired by KIMI_RAIKKONEN.Trainer:
      - SigmoidFocalLoss (single clean loss)
      - AdamW optimizer
      - OneCycleLR scheduler (warmup → peak → cosine decay)
      - EMA model for smoother evaluation
      - Mixed precision (AMP)
      - Gradient clipping + accumulation
    """

    def __init__(self, model: SimpleQuantileAnomalyDetector, device: torch.device, cfg: dict):
        self.model = model.to(device)
        self.device = device
        self.cfg = cfg

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.get('lr', 5e-4),
            weight_decay=cfg.get('weight_decay', 0.01),
            betas=(0.9, 0.999),
        )

        # Loss: single clean SigmoidFocalLoss
        self.criterion = SigmoidFocalLoss(
            alpha=cfg.get('focal_alpha', 0.6),
            gamma=cfg.get('focal_gamma', 2.0),
        )

        # OneCycleLR: warmup → peak → cosine decay
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=cfg.get('lr', 5e-4),
            total_steps=cfg['steps_per_epoch'] * cfg['num_epochs'],
            pct_start=cfg.get('warmup_fraction', 0.05),
            anneal_strategy='cos',
            div_factor=10,
            final_div_factor=100,
        )

        # Mixed precision
        self.use_amp = (device.type == 'cuda')
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        # EMA model
        self.ema_model = None
        if cfg.get('use_ema', True):
            self.ema_model = copy.deepcopy(model).to(device)
            for p in self.ema_model.parameters():
                p.requires_grad_(False)
            self.ema_decay = cfg.get('ema_decay', 0.999)

    @torch.no_grad()
    def _update_ema(self):
        if self.ema_model is None:
            return
        for ema_p, p in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_p.data.mul_(self.ema_decay).add_(p.data, alpha=1.0 - self.ema_decay)

    def train_epoch(self, train_loader):
        """Train one epoch. Handles gradient accumulation for batch_size=1."""
        self.model.train()
        total_loss = 0.0
        all_probs, all_labels = [], []
        n_steps = 0

        accum_steps = self.cfg.get('grad_accumulation', 8)
        self.optimizer.zero_grad()

        for step, (quantiles, values, gt) in enumerate(
            tqdm(train_loader, desc="  Train", leave=False)
        ):
            quantiles = quantiles.to(self.device)
            values = values.to(self.device)
            gt = gt.to(self.device)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                _, binary_logits = self.model(quantiles, values)
                loss = self.criterion(binary_logits, gt) / accum_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.cfg.get('max_grad_norm', 4.0)
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
                self._update_ema()

            total_loss += loss.item() * accum_steps
            n_steps += 1

            with torch.no_grad():
                probs = torch.sigmoid(binary_logits).cpu().numpy().flatten()
                all_probs.extend(probs)
                all_labels.extend(gt.cpu().numpy().flatten())

        # Handle leftover gradient accumulation
        if (step + 1) % accum_steps != 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.cfg.get('max_grad_norm', 4.0)
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()
            self._update_ema()

        # Global metrics
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels).astype(int)
        preds_05 = (all_probs > 0.5).astype(int)

        return {
            'loss': float(total_loss) / max(n_steps, 1),
            'f1': float(f1_score(all_labels, preds_05, zero_division=0)),
            'precision': float(precision_score(all_labels, preds_05, zero_division=0)),
            'recall': float(recall_score(all_labels, preds_05, zero_division=0)),
            'lr': float(self.optimizer.param_groups[0]['lr']),
        }

    @torch.no_grad()
    def evaluate(self, val_loader, use_ema: bool = True):
        """Evaluate model. Returns metrics + optimal threshold via PR curve."""
        model = self.ema_model if (use_ema and self.ema_model is not None) else self.model
        model.eval()

        all_probs, all_labels = [], []
        total_loss = 0.0
        n_batches = 0

        for quantiles, values, gt in tqdm(val_loader, desc="  Valid", leave=False):
            quantiles = quantiles.to(self.device)
            values = values.to(self.device)
            gt = gt.to(self.device)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                _, binary_logits = model(quantiles, values)
                loss = self.criterion(binary_logits, gt)

            total_loss += loss.item()
            n_batches += 1

            probs = torch.sigmoid(binary_logits).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(gt.cpu().numpy().flatten())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels).astype(int)

        # AUC-PR
        auc_pr = 0.0
        if all_labels.sum() > 0:
            auc_pr = float(average_precision_score(all_labels, all_probs))

        # Optimal threshold via PR curve
        best_threshold, best_f1 = 0.5, 0.0
        best_precision, best_recall = 0.0, 0.0
        if all_labels.sum() > 0:
            precisions, recalls, thresholds = precision_recall_curve(
                all_labels, all_probs
            )
            f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
            best_idx = f1s.argmax()
            best_f1 = f1s[best_idx]
            best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            best_precision = precisions[best_idx]
            best_recall = recalls[best_idx]

        return {
            'loss': float(total_loss) / max(n_batches, 1),
            'auc_pr': float(auc_pr),
            'best_f1': float(best_f1),
            'best_threshold': float(best_threshold),
            'best_precision': float(best_precision),
            'best_recall': float(best_recall),
            'f1_at_05': float(f1_score(all_labels, (all_probs > 0.5).astype(int), zero_division=0)),
        }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    try:
        cfg = {
            # ── Data sources (V2 + V5 merged) ──
            'train_sources': [
                ('./PROCESSED_TRAIN_DATAV2/', './TRAIN_SPLIT/'),
                ('./PROCESSED_TRAIN_DATAV5/', './TRAIN_MULTI_CLEAN/'),
            ],
            'val_sources': [
                ('./PROCESSED_TRAIN_DATAV2/', './TEST_SPLIT/'),
                ('./PROCESSED_TRAIN_DATAV5/', './TEST_MULTI_CLEAN/'),
            ],
            'model_save_path': './SAVED_MODELS_SIMPLE_Final/',

            # ── Model ──
            'in_features': 12,
            'hidden_dim': 32,
            'kernel_size': 7,
            'num_layers': 3,
            'num_attention_heads': 4,
            'num_attention_layers': 4,
            'dropout': 0.35,

            # ── Training ──
            'num_epochs': 40,
            'lr': 5e-4,
            'weight_decay': 0.05,
            'warmup_fraction': 0.1,
            'grad_accumulation': 8,
            'max_grad_norm': 4.0,

            # ── Loss ──
            'focal_alpha': 0.6,
            'focal_gamma': 2.0,

            # ── EMA ──
            'use_ema': True,
            'ema_decay': 0.999,

            # ── Misc ──
            'num_workers': 0,
            'patience': 5,
            'skip_names': [],
            'min_variables': 2,
            'seed': 42,
        }

        # Seed
        torch.manual_seed(cfg['seed'])
        np.random.seed(cfg['seed'])

        os.makedirs(cfg['model_save_path'], exist_ok=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        # ── Load datasets (merged V2 + V5) ──
        print("\n═══ Loading training data ═══")
        train_dataset = MergedQuantileDataset(
            sources=cfg['train_sources'],
            skip_names=set(cfg['skip_names']),
            min_variables=cfg['min_variables'],
            verbose=True,
            dataset_label="Train",
        )

        print("\n═══ Loading validation data ═══")
        val_dataset = MergedQuantileDataset(
            sources=cfg['val_sources'],
            skip_names=set(cfg['skip_names']),
            min_variables=cfg['min_variables'],
            verbose=True,
            dataset_label="Validation",
        )

        print(f"\nTrain: {len(train_dataset):,} samples")
        print(f"Val:   {len(val_dataset):,} samples")

        # ── DataLoaders ──
        # batch_size=1 because D varies across files; use grad_accumulation
        sampler_weights = train_dataset.get_sampler_weights()
        sampler = torch.utils.data.WeightedRandomSampler(
            sampler_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=cfg['num_workers'],
            pin_memory=True,
            drop_last=False,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg['num_workers'],
            pin_memory=True,
        )

        cfg['steps_per_epoch'] = len(train_loader) // cfg['grad_accumulation']
        print(f"Steps per epoch (after grad accum): {cfg['steps_per_epoch']}")

        # ── Model ──
        model = SimpleQuantileAnomalyDetector(
            in_features=cfg['in_features'],
            hidden_dim=cfg['hidden_dim'],
            kernel_size=cfg['kernel_size'],
            num_layers=cfg['num_layers'],
            num_attention_heads=cfg['num_attention_heads'],
            num_attention_layers=cfg['num_attention_layers'],
            dropout=cfg['dropout'],
        )

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {n_params:,}")

        # ── Sanity check ──
        print("\nSanity check...")
        model.to(device)
        dummy_q = torch.randn(1, 5, 64, 21, device=device).sort(dim=-1)[0]
        dummy_v = torch.randn(1, 5, 64, device=device)
        with torch.no_grad():
            cont, binary = model(dummy_q, dummy_v)
        print(f"  Input: quantiles {dummy_q.shape}, values {dummy_v.shape}")
        print(f"  Output: continuous {cont.shape}, binary {binary.shape}")
        print(f"  Initial probs: {torch.sigmoid(binary).flatten()[:5].cpu().numpy().round(4)}")
        print(f"  (Should be ~0.5 — model not biased at init)\n")

        # ── Trainer ──
        trainer = Trainer(model, device, cfg)

        # ── Save config ──
        serializable_cfg = {
            k: v for k, v in cfg.items()
            if isinstance(v, (int, float, str, bool, list))
        }
        with open(os.path.join(cfg['model_save_path'], "config.json"), 'w') as f:
            json.dump(serializable_cfg, f, indent=2)

        # ── Training loop ──
        best_f1 = 0.0
        patience_counter = 0
        history = []

        for epoch in range(cfg['num_epochs']):
            print(f"\n{'='*60}")
            print(f"  Epoch {epoch + 1}/{cfg['num_epochs']}")
            print(f"{'='*60}")

            # Train
            train_met = trainer.train_epoch(train_loader)
            print(f"  Train | Loss: {train_met['loss']:.4f}  "
                  f"F1: {train_met['f1']:.4f}  P: {train_met['precision']:.3f}  "
                  f"R: {train_met['recall']:.3f}  LR: {train_met['lr']:.2e}")

            # Validate
            val_met = trainer.evaluate(val_loader)
            print(f"  Valid | Loss: {val_met['loss']:.4f}  "
                  f"F1*: {val_met['best_f1']:.4f} (thr={val_met['best_threshold']:.3f})  "
                  f"P: {val_met['best_precision']:.3f}  R: {val_met['best_recall']:.3f}  "
                  f"AUC-PR: {val_met['auc_pr']:.4f}  F1@0.5: {val_met['f1_at_05']:.4f}")

            # Save best model
            if val_met['best_f1'] > best_f1:
                best_f1, patience_counter = val_met['best_f1'], 0

                save_dict = {
                    'model_state_dict': model.state_dict(),
                    'config': serializable_cfg,
                    'threshold': float(val_met['best_threshold']),
                    'best_f1': float(best_f1),
                    'epoch': epoch + 1,
                }
                if trainer.ema_model is not None:
                    save_dict['ema_state_dict'] = trainer.ema_model.state_dict()

                torch.save(
                    save_dict,
                    os.path.join(cfg['model_save_path'], "best_model.pth")
                )
                print(f"  >>> New best F1: {best_f1:.4f}")
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{cfg['patience']}) "
                      f"-> Best F1: {best_f1:.4f}")

                if patience_counter >= cfg['patience']:
                    print(f"\n  Early stopping after {cfg['patience']} epochs "
                          f"without improvement.")
                    break

            # Save checkpoint every epoch
            torch.save(
                {'model_state_dict': model.state_dict(), 'epoch': epoch + 1},
                os.path.join(cfg['model_save_path'],
                             f"checkpoint_epoch_{epoch + 1}.pth")
            )

            # Append to history
            history.append({
                'epoch': epoch + 1,
                **{f'train_{k}': v for k, v in train_met.items()},
                **{f'val_{k}': v for k, v in val_met.items()},
            })
            with open(os.path.join(cfg['model_save_path'], "training_log.json"), 'w') as f:
                json.dump(history, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Training complete! Best val F1: {best_f1:.4f}")
        print(f"{'='*60}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
