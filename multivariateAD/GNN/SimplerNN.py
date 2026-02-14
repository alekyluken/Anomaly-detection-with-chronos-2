import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import numpy as np
import sys, os

# Ensure Nunzio/ is on sys.path for sibling imports
_NUNZIO_DIR = os.path.dirname(os.path.abspath(__file__))
if _NUNZIO_DIR not in sys.path:
    sys.path.insert(0, _NUNZIO_DIR)

from attention import MultiHeadVariableAttention


class QuantileFeatureExtractor(nn.Module):
    """
    Estrae feature interpretabili e compatte dai 21 quantili.
    
    Motivazione:
    - 21 quantili sono troppi e ridondanti
    - Estraiamo 8 feature statistiche chiave
    - Ogni feature ha significato preciso
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, quantiles: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            quantiles: [B, D, T, Q] - ordinati dal quantile più basso al più alto
            values: [B, D, T] - valori reali
        
        Returns:
            features: [B, D, T, 12] - feature compatte
        """
        Q = quantiles.shape[-1]
        
        # Indici dinamici per supportare Q=3 (V1) e Q=21 (V2) e qualsiasi Q
        q_01  = quantiles[..., 0]                              # estremo inferiore
        q_05  = quantiles[..., max(1, Q // 20)]                # ~5th percentile
        q_25  = quantiles[..., Q // 4]                         # ~Q1
        q_50  = quantiles[..., Q // 2]                         # ~mediana
        q_75  = quantiles[..., 3 * Q // 4]                     # ~Q3
        q_95  = quantiles[..., min(Q - 2, Q - 1 - Q // 20)]   # ~95th percentile
        q_99  = quantiles[..., -1]                             # estremo superiore
        
        # Feature 1: Distanza valore reale dalla mediana predetta
        # Se valore molto lontano → probabile anomalia
        deviation = torch.abs(values - q_50)
        
        # Feature 2: IQR (Inter-Quartile Range) - spread della distribuzione
        # IQR alto → distribuzione incerta/dispersa
        iqr = q_75 - q_25
        
        # Feature 3: Deviation normalizzata per IQR
        # Se valore fuori da 3*IQR dalla mediana → outlier classico
        normalized_deviation = deviation / (iqr + 1e-6)
        
        # Feature 4: Range totale (q_95 - q_05)
        # Range molto grande → alta incertezza
        total_range = q_99 - q_01
        
        # Feature 5: Skewness (asimmetria)
        # Distribuzione normale ha skew ≈ 0
        # Anomalie spesso hanno skew alto
        skewness = (q_75 - 2*q_50 + q_25) / (iqr + 1e-6)
        
        # Feature 6: Posizione del valore nella distribuzione
        # Se valore < q_05 o > q_95 → nelle code estreme
        below_q05 = (values < q_05).float()
        above_q95 = (values > q_95).float()
        below_q01 = (values < q_01).float()
        above_q99 = (values > q_99).float()
        in_tails = below_q01 + above_q99
        second_tails = below_q05 + above_q95
        
        # Feature 7: Concentrazione (quanto stretta è la distribuzione)
        # concentration basso → distribuzione piatta/incerta
        concentration = iqr / (total_range + 1e-6)
        
        # Feature 8: Z-score approssimato
        # Distanza in "sigma" dalla mediana
        # Assumi std ≈ IQR / 1.35 (per distribuzione normale)
        approx_std = iqr / 1.35
        z_score = deviation / (approx_std + 1e-6)
        
        # ── Feature 9: Quantile position ──
        # Fraction of quantiles that the value exceeds → [0, 1]
        # 0 = below all predictions, 1 = above all
        exceeded = (values.unsqueeze(-1) > quantiles).float()  # [B, D, T, Q]
        quantile_position = exceeded.mean(dim=-1)  # [B, D, T]

        # ── Feature 10: Tail asymmetry ──
        # (q99 - q50) / (q50 - q01): heavy upper vs lower tail
        upper_tail = q_99 - q_50
        lower_tail = q_50 - q_01
        tail_asymmetry = upper_tail / (lower_tail + 1e-6)

        # ── Feature 11: Kurtosis proxy ──
        # (q99 - q01) / (q75 - q25): tail heaviness relative to core
        # Normal distribution ≈ 2.91
        kurtosis_proxy = total_range / (iqr + 1e-6)

        # Stack: [B, D, T, 12]
        features = torch.stack([
            normalized_deviation,  # 0: quanto fuori dalla mediana (in IQR)
            z_score,              # 1: quanto fuori in "sigma"
            iqr,                  # 2: spread
            total_range,          # 3: incertezza totale
            skewness,             # 4: asimmetria
            concentration,        # 5: quanto stretta
            in_tails,             # 6: se nelle code estreme
            second_tails,         # 7: se nelle code secondarie
            deviation,            # 8: distanza assoluta
            quantile_position,    # 9: posizione nella distribuzione [0,1]
            tail_asymmetry,       # 10: asimmetria code
            kurtosis_proxy,       # 11: pesantezza code
        ], dim=-1)

        return features


# ═══════════════════════════════════════════════════════════════
# ARCHITETTURA LEGGERA
# ═══════════════════════════════════════════════════════════════

class LightweightTemporalNet(nn.Module):
    """
    Rete leggera: Conv1D + Pooling.
    
    Motivazione:
    - Non serve complicare con GRU/Transformer
    - Conv1D cattura pattern temporali locali
    - 3 layer bastano per RF ~ 30 timestep
    """
    def __init__(self, in_channels: int = 8, hidden_dim: int = 32, kernel_size: int = 7, dropout: float = 0.2, num_layers: int = 3):
        super().__init__()

        if num_layers < 1 or num_layers > 8:
            raise ValueError("num_layers must be between 1 and 8")        

        self.convIn = nn.Conv1d(in_channels, hidden_dim, kernel_size, padding=kernel_size//2)
        self.bnIn = nn.BatchNorm1d(hidden_dim)

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout/(i+1))
                )
            )

        self.atrous_convs = nn.ModuleList()
        for i in range(num_layers): 
            self.atrous_convs.append(nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, dilation=(i+1)*2, padding=(kernel_size//2)*((i+1)*2)),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout/(i+1))
                )
            )

        self.convF = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.bnF = nn.BatchNorm1d(hidden_dim)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D, T, F=8]
        Output: [B, D, T, H=32]
        """
        B, D, T, N = x.shape
        
        # Reshape: [B*D, F, T] per Conv1d
        x = x.permute(0, 1, 3, 2).reshape(B*D, N, T)

        # Conv layers
        x = F.gelu(self.bnIn(self.convIn(x)))
        
        for i in range(len(self.convs)):
            x_res = x
            x = F.gelu(self.convs[i](x)) 
            x = F.gelu(self.atrous_convs[i](x)) + x_res  


        x = F.gelu(self.bnF(self.convF(x)))   
        
        # Reshape back: [B, D, T, H]
        return x.reshape(B, D, -1, T).permute(0, 1, 3, 2)


class DualHeadDecoder(nn.Module):
    def __init__(self, hidden_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        
        self.temporalConv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Solo MLP semplice e funzionale
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout / 1.5),
        )
        
        self.continuous_head = nn.Linear(hidden_dim, 1)
        self.binary_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.permute(0, 2, 1)  # [B, H, T]
        x = self.temporalConv(x) + x  # [B, H, T]
        shared = self.shared(x.permute(0, 2, 1))  # [B, T, H] → [B, T, H]
        return self.continuous_head(shared), self.binary_head(shared)

# ═══════════════════════════════════════════════════════════════
# MODELLO COMPLETO
# ═══════════════════════════════════════════════════════════════

class SimpleQuantileAnomalyDetector(nn.Module):
    """
    Anomaly detector basato su feature engineering dai quantili Chronos-2.

    Pipeline:
    1. Feature engineering (Q quantili → 12 feature statistiche)
    2. Conv1D leggera (cattura contesto temporale)
    3. Multi-Head Attention sulle variabili, permutation-invariant (D → 1)
    4. Dual head decoder (score continuo + logit binario)

    Invarianza:
    - D (variabili): nessun positional encoding → ordine variabili irrilevante
    - T (timestep): Conv1D kernel-based, nessun positional encoding
    """
    def __init__(
        self,
        in_features: int = 12,
        hidden_dim: int = 32,
        kernel_size: int = 7,
        num_layers: int = 3,
        num_attention_heads: int = 4,
        num_attention_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()

        # 1. Feature extractor (quantili → feature interpretabili)
        self.feature_extractor = QuantileFeatureExtractor()

        # 2. Temporal processing (Conv1D per pattern locali)
        self.temporal_net = LightweightTemporalNet(
            in_channels=in_features,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_layers=num_layers,
            dropout=dropout
        )

        # 3. Multi-Head Attention across variables (permutation-invariant)
        self.var_attention = MultiHeadVariableAttention(
            hidden_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            num_layers=num_attention_layers
        )

        # 4. Decoder (score continuo + classificazione binaria)
        self.decoder = DualHeadDecoder(hidden_dim, dropout=dropout)

    def forward(self, quantiles: torch.Tensor,
                values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            quantiles: [B, D, T, Q]  – previsioni quantili per D variabili
            values:    [B, D, T]     – valori reali osservati

        Returns:
            continuous:    [B, T, 1] – anomaly score continuo
            binary_logits: [B, T, 1] – logit per classificazione binaria
        """
        # 1. Feature engineering
        features = self.feature_extractor(quantiles, values)  # [B, D, T, 12]

        # 2. Temporal
        temporal = self.temporal_net(features)  # [B, D, T, H]

        # 3. Cross-variable attention pooling
        pooled = self.var_attention(temporal)  # [B, T, H]

        # 4. Decode
        return self.decoder(pooled)


# ═══════════════════════════════════════════════════════════════
# LOSS — Clean Sigmoid Focal Loss (from KIMI_RAIKKONEN pattern)
# ═══════════════════════════════════════════════════════════════

class SigmoidFocalLoss(nn.Module):
    """
    Focal Loss for binary classification (sigmoid output).
    Properly handles class imbalance via alpha and hard examples via gamma.

    Uses .mean() reduction (stable across batch sizes).
    Per-sample computation (correct gradient flow).

    Args:
        alpha: weight for POSITIVE class in [0,1].
               0.5 = balanced. With balanced sampling, 0.5-0.6 is ideal.
        gamma: focusing parameter. 0 = standard BCE. 2.0 = standard choice.
    """
    def __init__(self, alpha: float = 0.6, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  [B, T, 1] or [N] raw scores (pre-sigmoid)
        targets: [B, T, 1] or [N] binary {0.0, 1.0}
        """
        # Per-sample BCE (numerically stable)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # p_t = probability assigned to the TRUE class
        # Focal modulating factor: down-weight easy examples
        focal_weight = (1.0 - torch.exp(-bce)) ** self.gamma

        # Alpha weighting: favor positive class
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        return (alpha_t * focal_weight * bce).mean()  # MEAN not SUM


# ═══════════════════════════════════════════════════════════════
# POST-PROCESSING MINIMO
# ═══════════════════════════════════════════════════════════════

def simple_postprocess(predictions: torch.Tensor, window: int = 3) -> torch.Tensor:
    """
    Post-processing minimale: median filter.
    
    Args:
        predictions: [B, T, 1] binary 0/1
        window: finestra (dispari)
    """
    # Pad
    padded = F.pad(predictions.squeeze(-1), (window // 2, window // 2), mode='replicate')  # [B, T+2*pad]
        
    return padded.unfold(1, window, 1).median(dim=-1)[0].unsqueeze(-1)


# ═══════════════════════════════════════════════════════════════
# CALIBRAZIONE THRESHOLD
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def find_best_threshold(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    target_recall: float = 0.90
) -> float:
    """
    Trova threshold per raggiungere target recall.
    """
    model.eval()
    
    all_probs = []
    all_labels = []
    
    for quantiles, values, labels in val_loader:
        quantiles, values = quantiles.to(device), values.to(device)
        
        _, binary_logits = model(quantiles, values)
        probs = torch.sigmoid(binary_logits).cpu().numpy().flatten()
        labels_np = labels.numpy().flatten()
        
        all_probs.extend(probs)
        all_labels.extend(labels_np)
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Score anomalie
    pos_probs = all_probs[all_labels == 1]
    
    if len(pos_probs) == 0:
        return 0.5
    
    # Threshold che cattura target_recall
    threshold = np.percentile(pos_probs, (1 - target_recall) * 100)
    
    return float(np.clip(threshold, 0.01, 0.99))


# ═══════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test con dimensioni diverse
    for B, D, T, Q in [(4, 11, 64, 21), (2, 5, 128, 3), (8, 20, 32, 21)]:
        print(f"\n{'='*50}")
        print(f"Test B={B}, D={D}, T={T}, Q={Q}")
        print('='*50)

        quantiles = torch.randn(B, D, T, Q).sort(dim=-1)[0]
        values = torch.randn(B, D, T)

        model = SimpleQuantileAnomalyDetector(
            hidden_dim=32,
            kernel_size=7,
            num_layers=3,
            num_attention_heads=4,
            num_attention_layers=2
        )

        continuous, binary = model(quantiles, values)

        print(f"Input quantiles:   {quantiles.shape}")
        print(f"Input values:      {values.shape}")
        print(f"Continuous scores: {continuous.shape}, range [{continuous.min():.3f}, {continuous.max():.3f}]")
        print(f"Binary logits:     {binary.shape}, range [{binary.min():.3f}, {binary.max():.3f}]")

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters:  {total_params:,}")

    # Test features
    extractor = QuantileFeatureExtractor()
    features = extractor(quantiles, values)
    print(f"\nFeatures estratte: {features.shape} (expected 12)")

    # Test loss
    labels = torch.randint(0, 2, (B, T, 1)).float()
    criterion = SigmoidFocalLoss(alpha=0.97, gamma=2.0)
    loss = criterion(binary, labels)
    print(f"Focal loss: {loss.item():.4f}")
