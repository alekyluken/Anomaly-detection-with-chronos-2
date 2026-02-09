"""
Anomaly Forecasting on Chronos-2 Embeddings
=============================================

Complete redesign addressing the key bugs that prevented learning:

DESIGN CHOICES:
  - Single concatenated dataset with WeightedRandomSampler → balanced mini-batches
  - Sigmoid focal loss (binary, per-sample, .mean() reduction) → stable gradients
  - Token-aware architecture respecting Chronos-2 structure:
      Token[0]   = summary  (global series context)   ← IMPORTANT
      Token[1:8] = patches  (local temporal patterns)
      Token[8]   = forecast (future prediction)        ← IMPORTANT
  - Threshold optimization via PR curve on validation
  - EMA model for smoother evaluation
  - Float16 storage for memory efficiency (~7GB instead of ~14GB)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import copy
from tqdm import tqdm
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score
)
from json import dump as jsonDump, load as jsonLoad

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════
# DATASET — all files in one flat dataset, memory-efficient
# ═══════════════════════════════════════════════════════════════

class EmbeddingDataset(torch.utils.data.Dataset):
    """
    Loads ALL embedding files into a single flat dataset.
    Two-phase loading: first count samples, then allocate and fill.
    Uses float16 storage to halve RAM (~7GB for 525K samples).

    Each sample: (embedding [9, 768], label {0,1}).
    """

    def __init__(self, file_list, data_dir, verbose=True):
        self.data_dir = data_dir

        # ── Phase 1: count total samples ──
        total = 0
        file_info = []  # (filename, n_groups)
        skipped = []

        for f in tqdm(file_list, desc="  Scanning", disable=not verbose, leave=False):
            try:
                emb = np.load(os.path.join(data_dir, "embeddings", f), allow_pickle=True)
                file_info.append((f, emb.shape[0]))
                total += emb.shape[0]
                del emb
            except Exception as e:
                skipped.append((f, str(e)))
                continue

        if verbose and skipped:
            print(f"  Skipped {len(skipped)} files")

        # ── Phase 2: allocate and fill ──
        W = 9  # Chronos-2 always outputs 9 tokens for this config
        E = 768
        self.sequences = np.empty((total, W, E), dtype=np.float16)
        self.labels = np.empty(total, dtype=np.int64)
        offset = 0
        n_pos = 0

        for f, n_groups in tqdm(file_info, desc="  Loading", disable=not verbose, leave=False):
            try:
                emb = np.load(os.path.join(data_dir, "embeddings", f), allow_pickle=True)
                gt = pd.read_csv(os.path.join(data_dir, "ground_truth_labels", f.replace("embeddings.npy", "ground_truth_labels.csv")), header=None).iloc[:-1, 0].values.astype(int)
                preds = pd.read_csv(os.path.join(data_dir, "predictions", f.replace("embeddings.npy", "predictions.csv")))
                item_ids = preds.iloc[:, 0].values.astype(int)

                # Fill embeddings (squeeze the channel dim 1)
                self.sequences[offset:offset + n_groups] = emb.squeeze(1).astype(np.float16)

                # Compute per-group labels
                for g_id in range(1, n_groups + 1):
                    label = int(gt[item_ids == g_id].sum() >= 1)
                    self.labels[offset + g_id - 1] = label
                    n_pos += label

                offset += n_groups
                del emb

            except Exception as e:
                if verbose:
                    print(f"    Error filling {f}: {e}")
                # Fill with zeros if error occurs mid-way
                self.sequences[offset:offset + n_groups] = 0
                self.labels[offset:offset + n_groups] = 0
                offset += n_groups

        if verbose:
            print(f"  Loaded {total:,} samples | Normal: {total - n_pos:,} | "
                  f"Anomaly: {n_pos:,} ({100 * n_pos / max(total, 1):.1f}%)")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.from_numpy(self.sequences[idx].astype(np.float32)), self.labels[idx]  # int64

    def get_sampler_weights(self):
        """Per-sample weights for WeightedRandomSampler (inverse frequency)."""
        n_pos = self.labels.sum()
        weights = np.where(self.labels == 1, len(self.labels) / (2.0 * max(n_pos, 1)), len(self.labels) / (2.0 * max(len(self.labels) - n_pos, 1)))
        return torch.from_numpy(weights).double()


# ═══════════════════════════════════════════════════════════════
# MODEL — Token-aware classifier for Chronos-2 embeddings
# ═══════════════════════════════════════════════════════════════

class ChronosAnomalyDetector(nn.Module):
    """
    Token-aware anomaly detector for Chronos-2 embeddings.

    Architecture:
      1. Project all 9 tokens: Linear(768 → H)
      2. Add learnable token-type embeddings (summary/patch/forecast)
      3. Self-attention: all tokens attend to each other (1 layer)
      4. Extract summary + forecast + pool patches
      5. MLP classifier → single logit

    Input:  [N, 9, 768]
    Output: [N] logits (pre-sigmoid)
    """

    def __init__(self, embed_dim=768, hidden_dim=128, num_heads=4, dropout=0.15, numPatches=9):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Shared projection
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Token type embeddings: 0=summary, 1=patch, 2=forecast
        self.type_embed = nn.Embedding(3, hidden_dim)

        # Register the type_ids as a buffer (not a parameter)
        try:
            numPatches = int(numPatches)
            if numPatches < 3:
                raise ValueError("numPatches must be at least 3 to accommodate summary, forecast, and at least one patch.")
        except Exception as e:
            raise ValueError(f"Invalid numPatches value: {numPatches}. Must be an integer >= 3. Error: {e}")
        type_ids = torch.ones(numPatches, dtype=torch.long)
        type_ids[0], type_ids[-1]  = 0,2   # summary, forecast
        self.register_buffer('type_ids', type_ids)

        # Self-attention: all 9 tokens attend to each other
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # Feed-forward after attention
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

        # Classifier: summary(H) + forecast(H) + patch_pool(H) = 3H → 1
        self.classifier = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Small init on final layer for stable start (logits ≈ 0 → prob ≈ 0.5)
        nn.init.xavier_uniform_(self.classifier[-1].weight, gain=0.01)
        nn.init.zeros_(self.classifier[-1].bias)

    def forward(self, x):
        """
        x: [N, 9, 768]
        Returns: [N] logits
        """
        # 1. Project
        h = self.proj(x)  # [N, W, H]

        # 2. Add token type embeddings
        h = h + self.type_embed(self.type_ids).unsqueeze(0)  # broadcast [1, W, H]

        # 3. Self-attention (pre-norm style)
        attn_out, _ = self.self_attn(h, h, h)
        h = self.attn_norm(h + attn_out)

        # 4. Feed-forward (pre-norm style)
        h = self.ffn_norm(h + self.ffn(h))

        # 5. Extract features and classify
        combined = torch.cat([h[:, 0] + h[:, -1] +h[:, 1:-1].mean(dim=1), h[:, 0] , h[:, -1], h[:, 1:-1].mean(dim=1)], dim=-1)  # [N, 4H]
        return self.classifier(combined).squeeze(-1)  # [N]


# ═══════════════════════════════════════════════════════════════
# LOSS — Clean sigmoid focal loss
# ═══════════════════════════════════════════════════════════════

class SigmoidFocalLoss(nn.Module):
    """
    Focal Loss for binary classification (sigmoid output).
    Properly handles class imbalance via alpha and hard examples via gamma.

    Unlike the previous implementation:
      - Uses .mean() reduction (stable across batch sizes)
      - No entropy penalty (was actively preventing learning!)
      - Per-sample computation (correct gradient flow)

    Args:
        alpha: weight for POSITIVE class in [0,1].
               0.5 = balanced. With balanced sampling, 0.5-0.6 is ideal.
        gamma: focusing parameter. 0 = standard BCE. 2.0 = standard choice.
    """

    def __init__(self, alpha=0.5, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        logits:  [N] raw scores (pre-sigmoid)
        targets: [N] binary {0.0, 1.0}
        """
        # Per-sample BCE (numerically stable)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # p_t = probability assigned to the TRUE class
        p_t = torch.exp(-bce)

        # Focal modulating factor: down-weight easy examples
        focal_weight = (1.0 - p_t) ** self.gamma

        # Alpha weighting: favor positive class
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        return (alpha_t * focal_weight * bce).mean()  # ← MEAN not SUM (critical fix!)


# ═══════════════════════════════════════════════════════════════
# TRAINER
# ═══════════════════════════════════════════════════════════════

class Trainer:
    def __init__(self, model, device, cfg):
        self.model = model.to(device)
        self.device = device
        self.cfg = cfg

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.get('lr', 5e-4),
            weight_decay=cfg.get('weight_decay', 0.01),
            betas=(0.9, 0.999),
        )

        # Focal loss: alpha slightly favors positives (sampling handles main balance)
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
            div_factor=10,        # start_lr = max_lr / 10
            final_div_factor=100, # end_lr = start_lr / 100
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
        self.model.train()
        total_loss = 0.0
        all_probs = []
        all_labels = []
        n_steps = 0

        accum_steps = self.cfg.get('grad_accumulation', 1)
        self.optimizer.zero_grad()

        for step, (data, labels) in enumerate(tqdm(train_loader, desc="  Train", leave=False)):
            data = data.to(self.device)                # [N, 9, 768]
            labels = labels.float().to(self.device)    # [N]

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                logits = self.model(data)              # [N]
                loss = self.criterion(logits, labels) / accum_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
                self._update_ema()

            total_loss += loss.item() * accum_steps
            n_steps += 1

            with torch.no_grad():
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.cpu().numpy())

        # Handle leftover gradient accumulation
        if (step + 1) % accum_steps != 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()
            self._update_ema()

        # Global metrics (correct way — accumulate all predictions, then compute)
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
    def evaluate(self, val_loader, use_ema=True):
        model = self.ema_model if (use_ema and self.ema_model is not None) else self.model
        model.eval()

        all_probs = []
        all_labels = []
        total_loss = 0.0
        n_batches = 0

        for data, labels in tqdm(val_loader, desc="  Valid", leave=False):
            data = data.to(self.device)
            labels = labels.float().to(self.device)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                logits = model(data)
                loss = self.criterion(logits, labels)

            total_loss += loss.item()
            n_batches += 1

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

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

        # F1 at fixed threshold 0.5 for reference
        preds_05 = (all_probs > 0.5).astype(int)
        f1_05 = f1_score(all_labels, preds_05, zero_division=0)

        return {
            'loss': float(total_loss) / max(n_batches, 1),
            'auc_pr': float(auc_pr),
            'best_f1': float(best_f1),
            'best_threshold': float(best_threshold),
            'best_precision': float(best_precision),
            'best_recall': float(best_recall),
            'f1_at_05': float(f1_05),
        }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    try:
        cfg = {
            # ── Data ──
            'train_data_path':   './TRAIN_SPLIT_UNIVARIATE_WEB_FILES/',
            'val_data_path':     './TEST_SPLIT_UNIVARIATE_WEB_FILES/',
            'processed_data_dir': './PROCESSED_TRAIN_DATAV4/',
            'model_save_path':   f'./Saved_Models_Temporal/HIGHLY_AGGRESSIVE/',

            # ── Model ──
            'embed_dim':   768,
            'hidden_dim':  32,
            'num_heads':   4,
            'dropout':     0.40,

            # ── Training ──
            'num_epochs':        30,
            'batch_size':        128,
            'lr':                3e-4,
            'weight_decay':      0.1,
            'warmup_fraction':   0.15,
            'grad_accumulation': 1,
            'focal_alpha':       0.85,   # mild positive bias (sampling handles main balance)
            'focal_gamma':       2.0,   # focus on hard examples
            'use_ema':           True,
            'ema_decay':         0.9999,

            # ── Misc ──
            'num_workers': 0,
            'patience':    4,
        }

        os.makedirs(cfg['model_save_path'], exist_ok=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        # ── File lists (same logic as embedding_Training.py) ──
        train_files = sorted(os.listdir(cfg['train_data_path']))
        val_files = sorted(os.listdir(cfg['val_data_path']))
        print(f"\nTrain files: {len(train_files)}")
        print(f"Val files:   {len(val_files)}")

        # ── Load datasets ──
        print("\n═══ Loading training data ═══")
        train_dataset = EmbeddingDataset(train_files, cfg['processed_data_dir'])

        print("\n═══ Loading validation data ═══")
        val_dataset = EmbeddingDataset(val_files, cfg['processed_data_dir'])

        # ── Create DataLoaders ──
        # WeightedRandomSampler: each batch has ~50% anomalies
        sample_weights = train_dataset.get_sampler_weights()
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, num_samples=len(train_dataset), replacement=True
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg['batch_size'],
            sampler=sampler,
            num_workers=cfg['num_workers'],
            pin_memory=True,
            drop_last=True,
            persistent_workers=(cfg['num_workers'] > 0),
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg['batch_size'] * 2,
            shuffle=False,
            num_workers=cfg['num_workers'],
            pin_memory=True,
            persistent_workers=(cfg['num_workers'] > 0),
        )

        cfg['steps_per_epoch'] = len(train_loader)

        # ── Model ──
        model = ChronosAnomalyDetector(
            embed_dim=cfg['embed_dim'],
            hidden_dim=cfg['hidden_dim'],
            num_heads=cfg['num_heads'],
            dropout=cfg['dropout'],
        )
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel parameters: {n_params:,}")
        print(f"Steps per epoch: {cfg['steps_per_epoch']}")

        # ── Sanity check: forward pass ──
        print("\nSanity check...")
        model.to(device)
        dummy = torch.randn(4, 9, 768, device=device)
        with torch.no_grad():
            out = model(dummy)
        print(f"  Input: {dummy.shape} -> Output: {out.shape}")
        print(f"  Initial logits: {out.cpu().numpy().round(4)}")
        print(f"  Initial probs:  {torch.sigmoid(out).cpu().numpy().round(4)}")
        print(f"  (Should be ~0.5 for all — model not biased at init)\n")

        # ── Trainer ──
        trainer = Trainer(model, device, cfg)

        # ── Save config ──
        config_to_save = {k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool))}
        with open(os.path.join(cfg['model_save_path'], "config.json"), 'w') as f:
            jsonDump(config_to_save, f, indent=2)

        # ── Training loop ──
        best_f1 = 0.0
        patience_counter = 0
        history = []

        for epoch in range(cfg['num_epochs']):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{cfg['num_epochs']}")
            print(f"{'='*60}")

            # Train
            train_met = trainer.train_epoch(train_loader)
            print(f"  Train | Loss: {train_met['loss']:.4f}  "
                f"F1: {train_met['f1']:.4f}  "
                f"P: {train_met['precision']:.3f}  "
                f"R: {train_met['recall']:.3f}  "
                f"LR: {train_met['lr']:.2e}")

            # Validate
            val_met = trainer.evaluate(val_loader)
            print(f"  Valid | Loss: {val_met['loss']:.4f}  "
                f"F1*: {val_met['best_f1']:.4f} (thr={val_met['best_threshold']:.3f})  "
                f"P: {val_met['best_precision']:.3f}  "
                f"R: {val_met['best_recall']:.3f}  "
                f"AUC-PR: {val_met['auc_pr']:.4f}  "
                f"F1@0.5: {val_met['f1_at_05']:.4f}")

            # Save best model
            if val_met['best_f1'] > best_f1:
                best_f1 = val_met['best_f1']
                patience_counter = 0

                save_dict = {
                    'model_state_dict': model.state_dict(),
                    'config': config_to_save,
                    'threshold': float(val_met['best_threshold']),
                    'best_f1': float(best_f1),
                    'epoch': epoch + 1,
                }
                if trainer.ema_model is not None:
                    save_dict['ema_state_dict'] = trainer.ema_model.state_dict()

                torch.save(save_dict, os.path.join(cfg['model_save_path'], "best_model.pth"))
                print(f"  * New best F1: {best_f1:.4f}")
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{cfg['patience']})")

                if patience_counter >= cfg['patience']:
                    print(f"\n  Early stopping after {cfg['patience']} epochs without improvement.")
                    break

            # Save checkpoint every epoch
            torch.save(
                {'model_state_dict': model.state_dict(), 'epoch': epoch + 1},
                os.path.join(cfg['model_save_path'], f"checkpoint_epoch_{epoch + 1}.pth")
            )

            # Append to history
            history.append({
                'epoch': epoch + 1,
                **{f'train_{k}': v for k, v in train_met.items()},
                **{f'val_{k}': v for k, v in val_met.items()},
            })
            with open(os.path.join(cfg['model_save_path'], "training_log.json"), 'w') as f:
                jsonDump(history, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Training complete! Best val F1: {best_f1:.4f}")
        print(f"{'='*60}")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
