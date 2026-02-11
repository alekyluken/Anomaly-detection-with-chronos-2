"""
TWO-STAGE MULTIVARIATE ANOMALY FORECASTING
===========================================

Architecture:
    Stage 1: Per-Series Detector (ChronosAnomalyDetector from KIMI_RAIKKONEN.py)
        Input:  [D, 9, 768] Chronos-2 embeddings per D serie
        Output: [D, hidden_dim] feature vectors (penultimate layer, returnLogits=False)
    
    Stage 2: Set-Based Aggregator (Set Transformer)
        Input:  [D, hidden_dim] feature vectors (D variabile)
        Output: 
            - Global anomaly score [1]
            - Per-series anomaly flags [D]

Challenges:
    1. D variabile (numero serie diverso per ogni sample)
    2. Permutation-invariant (ordine serie non importa)
    3. Output sia globale che per-series

Solution: Set Transformer con dual output heads

Freeze/Finetune:
    - freeze_stage1=True  → Solo Stage 2 trainabile
    - freeze_stage1=False → Sia Stage 1 che Stage 2 trainabili (end-to-end finetune)
    - stage1_lr_factor     → LR multiplier per Stage 1 (es. 0.1 = 10x più lento)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ═══════════════════════════════════════════════════════════════
# SET TRANSFORMER BUILDING BLOCKS
# ═══════════════════════════════════════════════════════════════

class MultiheadAttentionBlock(nn.Module):
    """Building block for set processing."""
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(query, key, value)
        query = self.norm1(query + attn_out)
        return self.norm2(query + self.ffn(query))


class ISAB(nn.Module):
    """Induced Set Attention Block - O(D*m) complexity instead of O(D^2)."""
    def __init__(self, dim: int, num_heads: int = 4,
                num_inducing: int = 32, dropout: float = 0.1):
        super().__init__()
        self.inducing_points = nn.Parameter(torch.randn(1, num_inducing, dim))
        
        self.mab1 = MultiheadAttentionBlock(dim, num_heads, dropout)
        self.mab2 = MultiheadAttentionBlock(dim, num_heads, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, D, dim]"""
        B = x.shape[0]
        inducing = self.inducing_points.expand(B, -1, -1)
        inducing = self.mab1(inducing, x, x)  # [B, m, dim]
        return self.mab2(x, inducing, inducing)  # [B, D, dim]


class PMA(nn.Module):
    """Pooling by Multihead Attention - aggregates set into fixed-size output."""
    def __init__(self, dim: int, num_heads: int = 4, 
                 num_seeds: int = 1, dropout: float = 0.1):
        super().__init__()
        self.seed_vectors = nn.Parameter(torch.randn(1, num_seeds, dim))
        self.mab = MultiheadAttentionBlock(dim, num_heads, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, D, dim] -> [B, num_seeds, dim]"""
        B = x.shape[0]
        seeds = self.seed_vectors.expand(B, -1, -1)
        return self.mab(seeds, x, x)


# ═══════════════════════════════════════════════════════════════
# STAGE 2: SET-BASED MULTIVARIATE AGGREGATOR
# ═══════════════════════════════════════════════════════════════

class Stage2MultivariateDetector(nn.Module):
    """
    Set-based aggregator per multivariate anomaly detection.
    
    Input:  [B, D, feature_dim] - D serie (variabile), ognuna con features da Stage 1
    Output: 
        - global_logits: [B, 1] - anomalia globale
        - series_logits: [B, D, 1] - quale serie è anomala
    
    Architecture:
        1. Input projection (feature_dim → hidden_dim)
        2. ISAB layers (permutation-invariant inter-series processing)
        3. Dual output:
            a) PMA → global pooling → global anomaly head
            b) Per-series MLP → individual series anomaly head
    """
    def __init__(
        self,
        feature_dim: int = 64,   # Output dim da Stage 1 (classifierPT1)
        hidden_dim: int = 64,
        num_isab_layers: int = 2,
        num_inducing: int = 32,
        num_heads: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Input projection (se feature_dim != hidden_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        ) if feature_dim != hidden_dim else nn.Identity()
        
        # ISAB layers (process set of series with inter-series attention)
        self.isab_layers = nn.ModuleList([
            ISAB(hidden_dim, num_heads, num_inducing, dropout)
            for _ in range(num_isab_layers)
        ])
        
        # Global pooling (all series → 1 global representation)
        self.global_pool = PMA(hidden_dim, num_heads, num_seeds=1, dropout=dropout)
        
        # Global anomaly head: [B, H] → [B, 1]
        self.global_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Per-series anomaly head: [B, D, H] → [B, D, 1]
        self.series_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, series_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            series_features: [B, D, feature_dim] - features from Stage 1
        
        Returns:
            global_logits: [B, 1] - global anomaly logit
            series_logits: [B, D, 1] - per-series anomaly logits
        """
        # Project to hidden_dim
        h = self.input_proj(series_features)  # [B, D, H]
        
        # ISAB processing (permutation-invariant inter-series attention)
        for isab in self.isab_layers:
            h = isab(h)  # [B, D, H]
        
        # Global output via PMA pooling
        global_repr = self.global_pool(h).squeeze(1)       # [B, 1, H]  ---> [B, H]
        
        # Per-series output (each series gets its own anomaly score)        
        return self.global_head(global_repr) , self.series_head(h) 


# ═══════════════════════════════════════════════════════════════
# COMBINED TWO-STAGE MODEL
# ═══════════════════════════════════════════════════════════════

class TwoStageMultivariateDetector(nn.Module):
    """
    Complete two-stage pipeline for multivariate anomaly forecasting.
    
    Stage 1: ChronosAnomalyDetector (from KIMI_RAIKKONEN.py)
        - Processes each series independently: [9, 768] → [hidden_dim] features
        - Uses returnLogits=False to get penultimate layer features
    
    Stage 2: Stage2MultivariateDetector (Set Transformer)
        - Aggregates all series features: [D, hidden_dim] → global + per-series scores
    
    Args:
        stage1_model: Pre-trained ChronosAnomalyDetector instance
        stage1_hidden_dim: Output dim of Stage 1's classifierPT1 (default 128)
        stage2_hidden_dim: Hidden dim for Stage 2 (default 64)
        num_isab_layers: Number of ISAB layers in Stage 2
        num_heads: Number of attention heads
        freeze_stage1: If True, Stage 1 weights are frozen (only Stage 2 trains)
                       If False, Stage 1 is finetuned jointly (use stage1_lr_factor)
        dropout: Dropout rate for Stage 2
    """
    def __init__(
        self,
        stage1_model: nn.Module,
        stage1_hidden_dim: int = 64,
        stage2_hidden_dim: int = 64,
        num_isab_layers: int = 2,
        num_heads: int = 4,
        freeze_stage1: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Stage 1: Per-series feature extractor
        self.stage1 = stage1_model
        self.freeze_stage1 = freeze_stage1
        
        if freeze_stage1:
            for param in self.stage1.parameters():
                param.requires_grad = False
            self.stage1.eval()
        
        # Stage 2: Multivariate aggregation
        self.stage2 = Stage2MultivariateDetector(
            feature_dim=stage1_hidden_dim,
            hidden_dim=stage2_hidden_dim,
            num_isab_layers=num_isab_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
    
    def set_freeze_stage1(self, freeze: bool):
        """Dynamically freeze/unfreeze Stage 1."""
        self.freeze_stage1 = freeze
        for param in self.stage1.parameters():
            param.requires_grad = not freeze
        if freeze:
            self.stage1.eval()
    
    def get_parameter_groups(self, base_lr: float, stage1_lr_factor: float = 0.1):
        """
        Return separate parameter groups for optimizer with differential LR.
        
        Args:
            base_lr: Base learning rate (used for Stage 2)
            stage1_lr_factor: LR multiplier for Stage 1 (e.g., 0.1 = 10x slower)
        
        Returns:
            List of param dicts for optimizer
        """
        stage1_params = [p for p in self.stage1.parameters() if p.requires_grad]
        stage2_params = list(self.stage2.parameters())
        
        groups = []
        if stage1_params:
            groups.append({'params': stage1_params, 'lr': base_lr * stage1_lr_factor})
        groups.append({'params': stage2_params, 'lr': base_lr})
        
        return groups
    
    def _extract_stage1_features(self, series_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Run Stage 1 to extract features for each series.
        
        Args:
            series_embeddings: [D, 9, 768] - D series, each with Chronos-2 tokens
        
        Returns:
            features: [D, stage1_hidden_dim]
        """
        if self.freeze_stage1:
            with torch.no_grad():
                features = self.stage1(series_embeddings, returnLogits=False)
        else:
            features = self.stage1(series_embeddings, returnLogits=False)
        return features
    
    def forward_single_sample(
        self, 
        series_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process single multivariate sample (no batch dim).
        
        Args:
            series_embeddings: [D, 9, 768] - D series, each with Chronos embeddings
        
        Returns:
            global_logit: [1] - global anomaly logit
            series_logits: [D, 1] - per-series anomaly logits
            series_features: [D, H] - intermediate features (for analysis)
        """
        # Stage 1: Extract features for each series
        series_features = self._extract_stage1_features(series_embeddings)  # [D, H]
        
        # Stage 2: Aggregate (add batch dim)
        series_features_batched = series_features.unsqueeze(0)  # [1, D, H]
        global_logit, series_logits = self.stage2(series_features_batched)
        
        # Remove batch dim
        global_logit = global_logit.squeeze(0)      # [1]
        series_logits = series_logits.squeeze(0)    # [D, 1]
        
        return global_logit, series_logits, series_features
    
    def forward_batch(
        self,
        batch_series_embeddings: list  # List[Tensor[D_i, 9, 768]]
    ) -> Tuple[torch.Tensor, list, list]:
        """
        Process batch with variable D per sample.
        
        Since D varies across samples, each sample is processed independently
        through Stage 1 and Stage 2.
        
        Args:
            batch_series_embeddings: List of [D_i, 9, 768] tensors
        
        Returns:
            global_logits: [B] - global anomaly logits
            series_logits_list: List of [D_i, 1] tensors 
            series_features_list: List of [D_i, H] tensors
        """
        global_logits = []
        series_logits_list = []
        series_features_list = []
        
        for series_emb in batch_series_embeddings:
            g_logit, s_logits, s_features = self.forward_single_sample(series_emb)
            global_logits.append(g_logit.squeeze(-1))  # [1] → scalar
            series_logits_list.append(s_logits)
            series_features_list.append(s_features)
        
        global_logits = torch.stack(global_logits)  # [B]
        
        return global_logits, series_logits_list, series_features_list
    
    def train(self, mode: bool = True):
        """Override train() to keep Stage 1 in eval mode when frozen."""
        super().train(mode)
        if self.freeze_stage1 and mode:
            self.stage1.eval()
        return self


# ═══════════════════════════════════════════════════════════════
# CUSTOM COLLATE FUNCTION (for DataLoader)
# ═══════════════════════════════════════════════════════════════

def collate_multivariate_batch(batch):
    """
    Collate function per DataLoader con D variabile.
    
    Args:
        batch: List of (series_embeddings, global_label, series_labels)
            series_embeddings: [D_i, 9, 768]
            global_label: 0/1
            series_labels: [D_i] binary
    
    Returns:
        embeddings_list: List of [D_i, 9, 768]
        global_labels: [B]
        series_labels_list: List of [D_i]
    """
    embeddings_list = []
    global_labels = []
    series_labels_list = []
    
    for series_emb, g_label, s_labels in batch:
        embeddings_list.append(series_emb)
        global_labels.append(g_label)
        series_labels_list.append(s_labels)
    
    # global_labels = torch.tensor(global_labels if len(global_labels) > 1 else [global_labels], dtype=torch.float32)

    return embeddings_list, torch.cat(global_labels, dim=0), series_labels_list


# ═══════════════════════════════════════════════════════════════
# LOSS FUNCTION
# ═══════════════════════════════════════════════════════════════

class TwoStageLoss(nn.Module):
    """
    Combined loss for two-stage model.
    
    Loss = α * global_loss + β * series_loss + γ * consistency_loss
    
    - global_loss:       Focal BCE on global anomaly prediction
    - series_loss:       Focal BCE on per-series anomaly predictions (averaged)
    - consistency_loss:  If global=1, at least one series should have high prob
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, 
                 gamma: float = 0.5, focal_gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.focal_gamma = focal_gamma
    
    def focal_bce(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Focal loss per singolo output."""
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = torch.exp(-bce)
        focal = (1 - p_t) ** self.focal_gamma * bce
        return focal.mean()
    
    def forward(
        self,
        global_logits: torch.Tensor,     # [B]
        series_logits_list: list,         # List of [D_i, 1]
        global_labels: torch.Tensor,     # [B]
        series_labels_list: list          # List of [D_i]
    ) -> Tuple[torch.Tensor, dict]:
        """
        Returns:
            total_loss: scalar
            loss_dict: breakdown per component
        """
        device = global_logits.device
        
        # Global loss
        loss_global = self.focal_bce(global_logits, global_labels)
        
        # Series loss (aggregate across all series in batch)
        series_losses = []
        for s_logits, s_labels in zip(series_logits_list, series_labels_list):
            s_logits_flat = s_logits.squeeze(-1)  # [D_i]
            s_labels_tensor = s_labels.float().to(device)
            series_losses.append(self.focal_bce(s_logits_flat, s_labels_tensor))
        
        loss_series = torch.stack(series_losses).mean() if series_losses else torch.tensor(0.0, device=device)
        
        # Consistency loss: if global=1, max(series_probs) should be high
        consistency_losses = []
        for g_label, s_logits in zip(global_labels, series_logits_list):
            if g_label.item() == 1:  # Global anomaly
                s_probs = torch.sigmoid(s_logits.squeeze(-1))  # [D_i]
                max_series_prob = s_probs.max()
                # Penalize if no series detected as anomalous
                consistency_losses.append(F.relu(0.5 - max_series_prob))
        
        loss_consistency = torch.stack(consistency_losses).mean() if consistency_losses else torch.tensor(0.0, device=device)
        
        # Total
        total_loss = (
            self.alpha * loss_global + 
            self.beta * loss_series + 
            self.gamma * loss_consistency
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'global': loss_global.item(),
            'series': loss_series.item(),
            'consistency': loss_consistency.item(),
        }
        
        return total_loss, loss_dict


# ═══════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Two-Stage Multivariate Anomaly Detector")
    print("=" * 60)
    
    from KIMI_RAIKKONEN import ChronosAnomalyDetector
    
    # Create Stage 1 model
    stage1_model = ChronosAnomalyDetector(
        embed_dim=768,
        hidden_dim=32,
        num_heads=4,
        dropout=0.30
    )
    
    # Create two-stage model (frozen Stage 1)
    model = TwoStageMultivariateDetector(
        stage1_model=stage1_model,
        stage1_hidden_dim=32,
        stage2_hidden_dim=32,
        num_isab_layers=2,
        freeze_stage1=True
    )
    
    # Test with variable D
    print("\nTest 1: D=5 series")
    series_emb_5 = torch.randn(5, 9, 768)
    g_logit, s_logits, s_features = model.forward_single_sample(series_emb_5)
    print(f"  Global logit: {g_logit.shape}")    # [1]
    print(f"  Series logits: {s_logits.shape}")  # [5, 1]
    print(f"  Series features: {s_features.shape}")  # [5, 64]
    
    print("\nTest 2: D=12 series")
    series_emb_12 = torch.randn(12, 9, 768)
    g_logit, s_logits, s_features = model.forward_single_sample(series_emb_12)
    print(f"  Global logit: {g_logit.shape}")    # [1]
    print(f"  Series logits: {s_logits.shape}")  # [12, 1]
    print(f"  Series features: {s_features.shape}")  # [12, 64]
    print("\nTest 3: Batch with variable D")
    batch = [
        torch.randn(5, 9, 768),
        torch.randn(8, 9, 768),
        torch.randn(3, 9, 768),
    ]
    g_logits, s_logits_list, s_features_list = model.forward_batch(batch)
    print(f"  Global logits: {g_logits.shape}")  # [3]
    print(f"  Series logits: {[s.shape for s in s_logits_list]}")  # [[5,1], [8,1], [3,1]]
    print(f"  Series features: {[s.shape for s in s_features_list]}")  # [[5,64], [8,64], [3,64]]  
    
    # Test loss
    print("\nTest 4: Loss function")
    criterion = TwoStageLoss(alpha=1.0, beta=1.0, gamma=0.5)
    
    global_labels = torch.tensor([1.0, 0.0, 1.0])
    series_labels_list = [
        torch.tensor([1, 0, 1, 0, 0]),          # 2/5 anomalous
        torch.tensor([0, 0, 0, 0, 0, 0, 0, 0]), # all normal
        torch.tensor([1, 1, 0]),                  # 2/3 anomalous
    ]
    
    loss, loss_dict = criterion(g_logits, s_logits_list, global_labels, series_labels_list)
    print(f"  Total loss: {loss.item():.4f}")
    print(f"  Breakdown: {loss_dict}")
    
    # Test freeze/unfreeze
    print("\nTest 5: Freeze/Unfreeze Stage 1")
    frozen_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable (frozen Stage 1): {frozen_params:,}")
    
    model.set_freeze_stage1(False)
    unfrozen_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable (unfrozen Stage 1): {unfrozen_params:,}")
    
    # Test parameter groups
    print("\nTest 6: Differential LR groups")
    param_groups = model.get_parameter_groups(base_lr=1e-3, stage1_lr_factor=0.1)
    for i, g in enumerate(param_groups):
        n_params = sum(p.numel() for p in g['params'])
        print(f"  Group {i}: {n_params:,} params, lr={g['lr']:.1e}")
    
    print("\n✓ Two-stage model OK!")
    print("✓ Handles variable D")
    print("✓ Freeze / finetune Stage 1")
    print("✓ Global + per-series anomaly output")
