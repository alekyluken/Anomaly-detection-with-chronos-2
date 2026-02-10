"""
Embedding Training — CNN-based anomaly detection on Chronos-2 embeddings
========================================================================

ARCHITECTURE (preserved from original):
  - customNetwork: dual-branch CNN (token branch + data branch + joint branch)
  - onlyFirstLastTokensNetwork: lightweight CNN on first/last tokens only

TRAINING INFRASTRUCTURE (redesigned after KIMI_RAIKKONEN.py):
  - Single flat EmbeddingDataset with float16 storage + WeightedRandomSampler
  - Sigmoid focal loss (binary, per-sample)
  - Trainer class with EMA, AMP, OneCycleLR, gradient accumulation
  - PR-curve threshold optimization on validation
  - Early stopping & best-model tracking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import copy
import pandas as pd
import numpy as np

from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score
)
from tqdm import tqdm
from json import load as jsonLoad, dump as jsonDump

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════
# DATASET — all files in one flat dataset, memory-efficient
# ═══════════════════════════════════════════════════════════════

class EmbeddingDataset(torch.utils.data.Dataset):
    """
    Loads ALL embedding files into a single flat dataset.
    Two-phase loading: first count samples, then allocate and fill.
    Uses float16 storage to halve RAM.

    Each sample: (embedding [W, 768], label {0,1}).
    """

    def __init__(self, file_list, data_dir, verbose=True):
        self.data_dir = data_dir

        # ── Phase 1: count total samples & detect shape ──
        total = 0
        file_info = []
        skipped = []

        for f in tqdm(file_list, desc="  Scanning", disable=not verbose, leave=False):
            try:
                emb = np.load(os.path.join(data_dir, "embeddings", f), allow_pickle=True)
                file_info.append((f, emb.shape[0]))
                total += emb.shape[0]
                if len(file_info) == 1:
                    # Detect W and E from first file (squeeze channel dim if present)
                    sample = emb[0].squeeze()
                    self._W, self._E = sample.shape[-2], sample.shape[-1]
                del emb
            except Exception as e:
                skipped.append((f, str(e)))
                continue

        if verbose and skipped:
            print(f"  Skipped {len(skipped)} files")

        # ── Phase 2: allocate and fill ──
        self.sequences = np.empty((total, self._W, self._E), dtype=np.float16)
        self.labels = np.empty(total, dtype=np.int64)
        offset = 0
        n_pos = 0

        for f, n_groups in tqdm(file_info, desc="  Loading", disable=not verbose, leave=False):
            try:
                emb = np.load(os.path.join(data_dir, "embeddings", f), allow_pickle=True)
                gt = pd.read_csv(
                    os.path.join(data_dir, "ground_truth_labels",
                                 f.replace("embeddings.npy", "ground_truth_labels.csv")),
                    header=None
                ).iloc[:-1, 0].values.astype(int)
                preds = pd.read_csv(
                    os.path.join(data_dir, "predictions",
                                 f.replace("embeddings.npy", "predictions.csv"))
                )
                item_ids = preds.iloc[:, 0].values.astype(int)

                # Fill embeddings (squeeze channel dim if present)
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
                self.sequences[offset:offset + n_groups] = 0
                self.labels[offset:offset + n_groups] = 0
                offset += n_groups

        if verbose:
            print(f"  Loaded {total:,} samples ({self._W}x{self._E}) | "
                  f"Normal: {total - n_pos:,} | Anomaly: {n_pos:,} "
                  f"({100 * n_pos / max(total, 1):.1f}%)")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.sequences[idx].astype(np.float32)),
                self.labels[idx])  # int64

    def get_sampler_weights(self):
        """Per-sample weights for WeightedRandomSampler (inverse frequency)."""
        n_pos = self.labels.sum()
        n_neg = len(self.labels) - n_pos
        weights = np.where(
            self.labels == 1,
            len(self.labels) / (2.0 * max(n_pos, 1)),
            len(self.labels) / (2.0 * max(n_neg, 1)),
        )
        return torch.from_numpy(weights).double()


class customNetwork(nn.Module):
    """
    Dual-branch CNN for Chronos-2 embeddings.
      - Token branch: processes first + last tokens (summary/forecast) via 2D convolutions
      - Data branch: processes middle tokens (patches) via 1D convolutions
      - Joint branch: merges both via 1D convolutions → single logit

    Input:  [N, W, 768]  (W tokens, 768 embedding dim)
    Output: [N] logits (pre-sigmoid)
    """

    def __init__(self, hiddenDim: int, numLayersToken: int, numLayerData: int,
                 dropout: float = 0.30, classifier: bool = False):
        super().__init__()

        if numLayerData < 1 or numLayerData > 6:
            raise ValueError("numLayerData must be between 1 and 7")

        self.__tokenBranch(hiddenDim, numLayersToken, dropout)

        self.numLayersData = numLayerData
        self.__dataBranch(hiddenDim, numLayerData, dropout)

        self.__jointBranch(hiddenDim, dropout)

        if classifier:
            self.classificationHead = nn.Sequential(
                nn.Linear(hiddenDim, hiddenDim // 2),
                nn.LayerNorm(hiddenDim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hiddenDim // 2, 1),
            )
        else:
            self.classificationHead = nn.Identity()

        self._init_weights()


    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Small init on final layer → logits ≈ 0 → prob ≈ 0.5 at start
        if isinstance(self.classificationHead, nn.Sequential):
            nn.init.xavier_uniform_(self.classificationHead[-1].weight, gain=0.01)
            nn.init.zeros_(self.classificationHead[-1].bias)

    # Token branch: INPUT DIMENSION: Nx1x2*768  --> Nx1x2x32x24 --> Nx2x32x24
    def __tokenBranch(self, hiddenDim: int, numLayers: int, dropout: float):
        self.tokenBranchInit = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=hiddenDim, kernel_size=3),
            nn.BatchNorm2d(hiddenDim),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )

        self.sequentialLayersToken = nn.ModuleList()
        for i in range(numLayers):
            self.sequentialLayersToken.append(nn.Sequential(
                nn.Conv2d(in_channels=hiddenDim, out_channels=hiddenDim, kernel_size=3, padding=1),
                nn.BatchNorm2d(hiddenDim),
                nn.LeakyReLU(),
                nn.Dropout(dropout / (i + 1))
            ))

        self.tokenBranchFinal = nn.Sequential(
            nn.Conv2d(in_channels=hiddenDim, out_channels=hiddenDim, kernel_size=3),
            nn.BatchNorm2d(hiddenDim),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )

    def __forwardTokenBranch(self, x:torch.Tensor):
        # x: Nx2x32x24 --> NxhiddenDimx28x20
        x = self.tokenBranchInit(x)
        for layer in self.sequentialLayersToken:
            x = layer(x) + x
        return self.tokenBranchFinal(x)

    # Data branch: INPUT DIMENSION: Nx1x768*W
    def __dataBranch(self, hiddenDim: int, numLayers: int, dropout: float):
        self.dataBranchInitial = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=hiddenDim, kernel_size=2, stride=2),
            nn.BatchNorm1d(hiddenDim),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )

        self.sequentialLayers = nn.ModuleList()
        for i in range(numLayers):
            self.sequentialLayers.append(nn.Sequential(
                nn.Conv1d(in_channels=hiddenDim, out_channels=hiddenDim, kernel_size=2, stride=2),
                nn.BatchNorm1d(hiddenDim),
                nn.LeakyReLU(),
                nn.Dropout(dropout / (i + 1))
            ))

        self.dataBranchFinal = nn.Sequential(
            nn.Conv1d(in_channels=hiddenDim, out_channels=hiddenDim, kernel_size=1),
            nn.BatchNorm1d(hiddenDim),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )

    def __forwardDataBranch(self, x:torch.Tensor):
        # x: Nx1xWx768 --> NxhiddenDimx384//2**numLayersData 
        N, C, W, E = x.shape

        x = torch.reshape(x, (N, C, W*E)) # Nx1x768*W

        x = self.dataBranchInitial(x)
        x2 = x.clone() # skip connection 

        N, H, _ = x.shape

        for layer in self.sequentialLayers:
            x = layer(x)
        
        xPool = x2.reshape((*x.shape, -1)).mean(dim=-1) # NxhiddenDimxL//(2**numLayersData)
        
        x = self.dataBranchFinal(x + xPool)
        return torch.reshape(x, (N, H, E//(2**(self.numLayersData+1)), -1)).mean(dim=-1) # NxhiddenDimxE//(2**(self.numLayersData+1))

    # Joint Branch
    def __jointBranch(self, hiddenDim: int, dropout: float):
        self.JConv1 = nn.Sequential(
            nn.Conv1d(in_channels=hiddenDim, out_channels=hiddenDim, kernel_size=4, dilation=3),
            nn.BatchNorm1d(hiddenDim),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )

        self.JConv2_1 = nn.Sequential(
            nn.Conv1d(in_channels=hiddenDim, out_channels=hiddenDim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hiddenDim),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )

        self.JConv2_2 = nn.Sequential(
            nn.Conv1d(in_channels=hiddenDim, out_channels=hiddenDim, kernel_size=5, padding=6, dilation=3),
            nn.BatchNorm1d(hiddenDim),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )

        self.JConv3 = nn.Sequential(
            nn.Conv1d(in_channels=hiddenDim, out_channels=hiddenDim, kernel_size=1),
            nn.BatchNorm1d(hiddenDim),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )


    def __forwardJointBranch(self, tokenFeatures:torch.Tensor, dataFeatures:torch.Tensor):
        # tokenFeatures: NxhiddenDimx28x20 --> NxhiddenDimx560
        # dataFeatures: NxhiddenDimxE//(2**(self.numLayersData+1)) --> NxhiddenDimxE//(2**(self.numLayersData+1))
        N, H, W, L = tokenFeatures.shape
        tokenFeatures = torch.reshape(tokenFeatures, (N, H, -1)) # NxhiddenDimx560

        x = self.JConv1(torch.cat((tokenFeatures, dataFeatures), dim=-1)) # NxhiddenDimx(560+E//(2**(self.numLayersData+1))-3*(4-1)-1) / 2 +1) = NxhiddenDimxY

        x1 = self.JConv2_2(x) # NxhiddenDimxY
        x2 = self.JConv2_1(x) # NxhiddenDimxY

        x = self.JConv3(x + x1 + x2) # NxhiddenDimxY 

        return x.mean(dim=-1) # NxhiddenDim
    

    # final forward function
    def forward(self, x: torch.Tensor):
        """
        x: [N, W, 768] or [N, 1, W, 768]
        Returns: [N] logits (pre-sigmoid)
        """
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # Nx1xWx768

        tokens = self.__forwardTokenBranch(
            x[:, 0, [0, -1], :].reshape((x.shape[0], 2, 32, -1))
        )  # NxHx28x20
        data = self.__forwardDataBranch(x[:, :, 1:-1, :])  # NxHxL
        joint = self.__forwardJointBranch(tokens, data)     # NxH

        out = self.classificationHead(joint)
        return out.squeeze(-1) if out.dim() > 1 else out  # [N]


class onlyFirstLastTokensNetwork(nn.Module):
    """
    Lightweight CNN classifier using only the first (summary) and last (forecast) tokens.

    Input:  [N, W, 768]  (only positions 0 and -1 are used)
    Output: [N] logits (pre-sigmoid)
    """

    def __init__(self, hiddenDim: int = 64, numInternalLayers: int = 4,
                 dropout: float = 0.30, classifier: bool = False):
        super().__init__()
        # Reshape Nx2x768 -> Nx2x32x24

        if numInternalLayers < 0 or numInternalLayers > 6:
            raise ValueError("numInternalLayers must be between 0 and 6")

        # DSConv to avoid mixing branches too early
        self.initialConv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3,
                      groups=2, padding=1, stride=2),  # Nx2x16x12
            nn.BatchNorm2d(2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),

            nn.Conv2d(in_channels=2, out_channels=hiddenDim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hiddenDim // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),

            nn.Conv2d(in_channels=hiddenDim // 2, out_channels=hiddenDim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hiddenDim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )

        self.sequentialLayers = nn.ModuleList()
        for i in range(numInternalLayers):
            self.sequentialLayers.append(nn.Sequential(
                nn.Conv2d(in_channels=hiddenDim, out_channels=hiddenDim,
                          kernel_size=3, padding=1),
                nn.BatchNorm2d(hiddenDim),
                nn.LeakyReLU(),
                nn.Dropout(dropout / (i + 1)),
            ))

        self.finalConv = nn.Sequential(
            nn.Conv2d(in_channels=hiddenDim, out_channels=hiddenDim,
                      kernel_size=3, stride=2, padding=1),  # Hx8x6
            nn.BatchNorm2d(hiddenDim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),

            nn.Conv2d(in_channels=hiddenDim, out_channels=hiddenDim,
                      kernel_size=3, stride=2, padding=1),  # Hx4x3
            nn.BatchNorm2d(hiddenDim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        if classifier:
            self.classificationHead = nn.Sequential(
                nn.Linear(hiddenDim, hiddenDim // 2),
                nn.LayerNorm(hiddenDim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hiddenDim // 2, 1),
            )
        else:
            self.classificationHead = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if isinstance(self.classificationHead, nn.Sequential):
            nn.init.xavier_uniform_(self.classificationHead[-1].weight, gain=0.01)
            nn.init.zeros_(self.classificationHead[-1].bias)

    def forward(self, x: torch.Tensor):
        """
        x: [N, W, 768] or [N, 1, W, 768]
        Returns: [N] logits (pre-sigmoid)
        """
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # Nx1xWx768

        x = x[:, 0, [0, -1], :].reshape((x.shape[0], 2, 32, -1))  # Nx2x32x24

        x = self.initialConv(x)
        for layer in self.sequentialLayers:
            x = layer(x) + x

        x = self.finalConv(x)
        x = self.gap(x)  # NxHx1x1

        out = self.classificationHead(x.view(x.size(0), -1))
        return out.squeeze(-1) if out.dim() > 1 else out  # [N]



# ═══════════════════════════════════════════════════════════════
# LOSS — Sigmoid focal loss (binary, per-sample)
# ═══════════════════════════════════════════════════════════════

class SigmoidFocalLoss(nn.Module):
    """
    Focal Loss for binary classification (sigmoid output).
    Properly handles class imbalance via alpha and hard examples via gamma.

    Uses .mean() reduction for stability across batch sizes.

    Args:
        alpha: weight for POSITIVE class in [0,1].
        gamma: focusing parameter. 0 = standard BCE. 2.0 = standard choice.
    """

    def __init__(self, alpha: float = 0.5, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  [N] raw scores (pre-sigmoid)
        targets: [N] binary {0.0, 1.0}
        """
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = torch.exp(-bce)
        focal_weight = (1.0 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        return (alpha_t * focal_weight * bce).mean()


# ═══════════════════════════════════════════════════════════════
# TRAINER — EMA, AMP, OneCycleLR, gradient accumulation
# ═══════════════════════════════════════════════════════════════

class Trainer:
    def __init__(self, model: nn.Module, device: torch.device, cfg: dict):
        self.model = model.to(device)
        self.device = device
        self.cfg = cfg

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.get('lr', 5e-4),
            weight_decay=cfg.get('weight_decay', 0.01),
            betas=(0.9, 0.999),
        )

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

    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> dict:
        self.model.train()
        total_loss = 0.0
        all_probs, all_labels = [], []
        n_steps = 0
        accum_steps = self.cfg.get('grad_accumulation', 1)
        self.optimizer.zero_grad()

        for step, (data, labels) in enumerate(tqdm(train_loader, desc="  Train", leave=False)):
            data = data.to(self.device)              # [N, W, 768]
            labels = labels.float().to(self.device)   # [N]

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                logits = self.model(data)             # [N]
                loss = self.criterion(logits, labels) / accum_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
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
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
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
    def evaluate(self, val_loader: torch.utils.data.DataLoader, use_ema: bool = True) -> dict:
        model = self.ema_model if (use_ema and self.ema_model is not None) else self.model
        model.eval()

        all_probs, all_labels = [], []
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
            precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)
            f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
            best_idx = f1s.argmax()
            best_f1 = f1s[best_idx]
            best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            best_precision = precisions[best_idx]
            best_recall = recalls[best_idx]

        # F1 at fixed threshold 0.5
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
            'train_data_path':    './TRAIN_SPLIT_UNIVARIATE_WEB_FILES/',
            'val_data_path':      './TEST_SPLIT_UNIVARIATE_WEB_FILES/',
            'processed_data_dir': './PROCESSED_TRAIN_DATAV4/',
            'model_save_path':    './Saved_Models_Simpler/',

            # ── Model ──
            'hidden_dim': 64,
            'numLayersToken': 2,
            'numLayerData': 2,
            'dropout': 0.30,

            'numInternalLayers': 2,    # solo per simplerModel
            'simplerModel': False,      # True → onlyFirstLastTokensNetwork

            # ── Training ──
            'num_epochs':        60,
            'batch_size':        64,
            'lr':                1e-3,
            'weight_decay':      0.1,
            'warmup_fraction':   0.10,
            'grad_accumulation': 1,
            'focal_alpha':       0.85,
            'focal_gamma':       2.0,
            'use_ema':           True,
            'ema_decay':         0.999,

            # ── Misc ──
            'num_workers': 0,
            'patience':    4,
        }

        os.makedirs(cfg['model_save_path'], exist_ok=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        # ── File lists ──
        train_files = sorted(os.listdir(cfg['train_data_path']))
        val_files = sorted(os.listdir(cfg['val_data_path']))
        print(f"\nTrain files: {len(train_files)}")
        print(f"Val files:   {len(val_files)}")

        # ── Load datasets ──
        print("\n══ Loading training data ══")
        train_dataset = EmbeddingDataset(train_files, cfg['processed_data_dir'])

        print("\n══ Loading validation data ══")
        val_dataset = EmbeddingDataset(val_files, cfg['processed_data_dir'])

        # ── DataLoaders with WeightedRandomSampler ──
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

        # ── Build model ──
        if cfg.get('simplerModel', False):
            model = onlyFirstLastTokensNetwork(
                hiddenDim=cfg['hidden_dim'],
                numInternalLayers=cfg.get('numInternalLayers', 4),
                dropout=cfg['dropout'],
                classifier=True,
            )
        else:
            model = customNetwork(
                hiddenDim=cfg['hidden_dim'],
                numLayersToken=cfg.get('numLayersToken', cfg.get('num_layers', 1)),
                numLayerData=cfg.get('numLayerData', cfg.get('num_layers', 1)),
                dropout=cfg['dropout'],
                classifier=True,
            )

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel: {model.__class__.__name__}  |  Parameters: {n_params:,}")
        print(f"Steps per epoch: {cfg['steps_per_epoch']}")

        # ── Sanity check ──
        print("\nSanity check...")
        model.to(device)
        dummy = torch.randn(4, train_dataset._W, train_dataset._E, device=device)
        with torch.no_grad():
            out = model(dummy)
        print(f"  Input: {dummy.shape} -> Output: {out.shape}")
        print(f"  Initial logits: {out.cpu().numpy().round(4)}")
        print(f"  Initial probs:  {torch.sigmoid(out).cpu().numpy().round(4)}")
        print(f"  (Should be ~0.5 for all — model not biased at init)\n")

        # ── Trainer ──
        trainer = Trainer(model, device, cfg)

        # ── Save config ──
        config_to_save = {k: v for k, v in cfg.items()
                          if isinstance(v, (int, float, str, bool))}
        with open(os.path.join(cfg['model_save_path'], "config.json"), 'w') as f:
            jsonDump(config_to_save, f, indent=2)

        # ── Training loop with early stopping ──
        best_f1 = 0.0
        patience_counter = 0
        history = []

        for epoch in range(cfg['num_epochs']):
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch + 1}/{cfg['num_epochs']}")
            print(f"{'=' * 60}")

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

            # Best model tracking
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

                torch.save(save_dict,
                           os.path.join(cfg['model_save_path'], "best_model.pth"))
                print(f"  ★ New best F1: {best_f1:.4f}")
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{cfg['patience']})")

                if patience_counter >= cfg['patience']:
                    print(f"\n  Early stopping after {cfg['patience']} epochs "
                          "without improvement.")
                    break

            # Checkpoint every epoch
            torch.save(
                {'model_state_dict': model.state_dict(), 'epoch': epoch + 1},
                os.path.join(cfg['model_save_path'],
                             f"checkpoint_epoch_{epoch + 1}.pth"),
            )

            # Append to history
            history.append({
                'epoch': epoch + 1,
                **{f'train_{k}': v for k, v in train_met.items()},
                **{f'val_{k}': v for k, v in val_met.items()},
            })
            with open(os.path.join(cfg['model_save_path'], "training_log.json"), 'w') as f:
                jsonDump(history, f, indent=2)

        print(f"\n{'=' * 60}")
        print(f"Training complete! Best val F1: {best_f1:.4f}")
        print(f"{'=' * 60}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

