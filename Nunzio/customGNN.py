import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

# ═══════════════════════════════════════════════════════════════
# MECHANISM 1: TEMPORAL PROCESSING (B, T, D variabili)
# ═══════════════════════════════════════════════════════════════

class TemporalProcessor(nn.Module):
    """
    Processes temporal features for each node independently.
    Uses causal Conv1D to be invariant to the absolute position of anomalies in time.

    Input:  [B, T, D] - batch, timesteps, features/nodes
    Output: [B, T, D, H] - aggiunge hidden dimension per ogni nodo
    """
    def __init__(self, hidden_dim: int = 32, kernel_size: int = 7, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Initial projection: 1 → H (proietta ogni singolo valore scalare)
        self.input_proj = nn.Linear(1, hidden_dim)
        
        # Stack of causal (dilated) Conv1D layers - invariant to absolute position
        # Each layer sees only past/present, not future
        # Opera su [B*D, H, T]
        self.conv_layers = nn.ModuleList()
        # Calcola num_groups che divide hidden_dim (minimo 1, massimo 8)
        num_groups = min(8, hidden_dim)
        while hidden_dim % num_groups != 0:
            num_groups -= 1
        
        for i in range(num_layers):
            dilation = 2 ** i  # 1, 2, 4, ... for exponential receptive field
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, dilation=dilation, padding=(kernel_size - 1) * dilation),
                nn.GroupNorm(num_groups, hidden_dim),  # GroupNorm instead of BatchNorm (works with batch=1)
                nn.GELU(),
                nn.Dropout(dropout)
            ))
        
        # Final layer norm for stability
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D] → [B, D, T, H]
        """
        B, T, D = x.shape
        
        # [B, T, D] → [B, T, D, 1] → [B, T, D, H]
        h = self.input_proj(x.unsqueeze(-1))  # [B, T, D, H]
        
        # Reshape per Conv1D: [B, T, D, H] → [B*D, H, T]
        h = h.permute(0, 2, 3, 1).reshape(B * D, self.hidden_dim, T)
        
        # Apply causal convolutions
        for conv in self.conv_layers:
            h = conv(h)[..., :T] + h  # Residual connection, crop to original T

        # Reshape back: [B*D, H, T] → [B, D, T, H] → [B, D, T, H]
        h = h.reshape(B, D, self.hidden_dim, T).permute(0, 1, 3, 2)  # [B, D, T, H]
        
        return self.norm(h)


# ═══════════════════════════════════════════════════════════════
# MECHANISM 2: SPATIAL AGGREGATION (Message Passing with weights [B, D, D])
# ═══════════════════════════════════════════════════════════════

class WeightedMessagePassing(nn.Module):
    """
    Message passing layer for spatial aggregation across nodes.
    
    Equation: h'_i = MLP(h_i || Σ_j W_ij * h_j)
    
    Input:  h [B, D, T, H], W [B, D, D]
    Output: [B, D, T, H]
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()

        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, h: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        h: [B, D, T, H] - node embeddings
        W: [B, D, D] - edge weights (will be row-normalized internally)
        
        Output: [B, D, T, H]
        """
        B, D, T, H = h.shape
        
        # Row-normalize W
        W = torch.where((W.diagonal(dim1=1, dim2=2).sum(dim=1) == 0).view(B,1,1), W + torch.eye(D, device=W.device).unsqueeze(0), W)  # se riga zero, usa identità

        N = W.sum(dim=-1, keepdim=True)  # [B, D, 1]
        W = W / torch.where(N == 0, torch.ones_like(N), N)  # [B, D, D]
        W = torch.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)

        # Reshape per batch matmul: [B, D, T, H] → [B, T, D, H] 
        # Message aggregation: W @ h per ogni timestep
        # Reshape: [B, T, D, H] → [B*T, D, H]
        # Il problema è qui, da tutti np.nan come moltiplicazione, le shape sono W_expanded: [B*T, D, D] e h_flat: [B*T, D, H]
        h_agg = torch.bmm(W.unsqueeze(1).expand(B, T, D, D).reshape(B * T, D, D), h.permute(0, 2, 1, 3).reshape(B * T, D, H))  # [B*T, D, H]
        h_agg = h_agg.reshape(B, T, D, H).permute(0, 2, 1, 3)  # [B, D, T, H]
        
        # Update with residual
        return self.norm(h + self.update_mlp(torch.cat([h, h_agg], dim=-1)))


# ═══════════════════════════════════════════════════════════════
# MECHANISM 3: NODE POOLING (D nodi → grafo, D variabile)
# ═══════════════════════════════════════════════════════════════

class AdaptiveNodePooling(nn.Module):
    """
    Aggrega D nodi in una rappresentazione globale, invariante a D.
    Usa attention per pesare i nodi in base alla loro "anomaly-relevance".
    
    Input:  [B, D, T, H]
    Output: [B, T, H]
    """
    def __init__(self, hidden_dim: int, pooling_type: Literal['attention', 'mean', 'max', 'sum'] = 'attention'):
        super().__init__()
        self.pooling_type = pooling_type
        
        if pooling_type == 'attention':
            self.attn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1)
            )
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: [B, D, T, H]
        Output: [B, T, H]
        """
        match self.pooling_type:
            case 'mean': return h.mean(dim=1)  # [B, T, H]
            case 'max': return h.max(dim=1)[0]  # [B, T, H]
            case 'sum': return h.sum(dim=1)  # [B, T, H]
            case _:  # attention
                attn_weights = F.softmax(self.attn(h), dim=1)  # [B, D, T, 1]
                return (h * attn_weights).sum(dim=1)  # [B, T, H]


# ═══════════════════════════════════════════════════════════════
# MECHANISM 4: ANOMALY SCORE DECODER (output continuo)
# ═══════════════════════════════════════════════════════════════

# class AnomalyScoreDecoder(nn.Module):
#     """
#     Produce anomaly score continuo [0, ∞) o logit per ogni timestep.
#     NON usa sigmoid - output raw per massima flessibilità.
    
#     Input:  [B, T, H]
#     Output: [B, T, 1] o [B, T] a seconda di squeeze_output
#     """
#     def __init__(self, hidden_dim: int, output_type: Literal['raw', 'positive', 'logit'] = 'raw', dropout: float = 0.1):
#         super().__init__()
#         self.output_type = output_type
        
#         self.mlp = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim // 2, 1)
#         )
        
#     def forward(self, h: torch.Tensor) -> torch.Tensor:
#         """
#         h: [B, T, H]
#         Output: [B, T, 1] - anomaly score
#         """
#         score = self.mlp(h)  # [B, T, 1]
        
#         # Clamp per stabilità numerica (evita overflow in sigmoid/softplus)
#         # score = score.clamp(-20, 20)
        
#         if self.output_type == 'positive':
#             return F.softplus(score) 
#         return score 

# SOSTITUISCI AnomalyScoreDecoder con:

class DualHeadAnomalyDecoder(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        """Decoder con due teste: una per score continuo, una per classificazione binaria."""
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.continuous_head = nn.Linear(hidden_dim, 1)
        self.binary_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, h):
        shared = self.shared(h)
        return self.continuous_head(shared), self.binary_head(shared)


# ═══════════════════════════════════════════════════════════════
# MODELLO COMPLETO
# ═══════════════════════════════════════════════════════════════

class SpatioTemporalAnomalyGNN(nn.Module):
    """
    GNN per anomaly detection su serie temporali multivariate.
    
    Pipeline:
    [B, T, D] residui → TemporalProc → [B, D, T, H] 
                      → GNN layers  → [B, D, T, H] 
                      → NodePool    → [B, T, H]
                      → Decoder     → [B, T, 1] anomaly score
    
    Proprietà:
    - B: batch size variabile
    - T variabile: Conv1D causali senza positional encoding
    - D variabile: Pooling permutation-invariant
    - Tempo-invariante: le anomalie vengono riconosciute indipendentemente da quando occorrono
    """
    def __init__( self, hidden_dim: int = 64, num_gnn_layers: int = 2, num_temporal_layers: int = 3, kernel_size: int = 7, 
            dropout: float = 0.1, pooling_type: Literal['attention', 'mean', 'max', 'sum'] = 'attention', 
            output_type: Literal['raw', 'positive', 'logit'] = 'raw'
    ):
        super().__init__()
        
        # 1. Temporal processing (per-node, causal)
        self.temporal_encoder = TemporalProcessor(
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_layers=num_temporal_layers,
            dropout=dropout
        )
        
        # 2. Spatial GNN layers (message passing tra nodi)
        self.gnn_layers = nn.ModuleList([
            WeightedMessagePassing(hidden_dim, dropout)
            for _ in range(num_gnn_layers)
        ])
        
        # 3. Node pooling (D → 1)
        self.pooling = AdaptiveNodePooling(hidden_dim, pooling_type)
        
        # 4. Anomaly score decoder
        self.decoder = DualHeadAnomalyDecoder(hidden_dim, dropout)
        
    def forward(self, node_features: torch.Tensor, edge_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_features: [B, T, D] - residui per ogni serie temporale
            edge_weights: [B, D, D] - matrice di similarità/connessione tra nodi
        
        Returns:
            scores: [B, T, 1] - anomaly score continuo per ogni timestep
        """
        
        # 1. Encode temporal patterns per ogni nodo
        h = self.temporal_encoder(node_features)  # [B, D, T, H]
        
        # 2. Message passing tra nodi (propaga informazione spaziale)
        for gnn in self.gnn_layers:
            h = gnn(h, edge_weights)  # [B, D, T, H]
        
        # 3. Pool nodes e decode
        h = self.pooling(h)  # [B, T, H]
        return self.decoder(h)  # [B, T, 1], [B, T, 1] (continuous scores, binary logits)
    
    @torch.no_grad()
    def predict(self, node_features: torch.Tensor, edge_weights: torch.Tensor, 
                score_type: Literal['softplus', 'sigmoid', 'exp'] = 'softplus') -> torch.Tensor:
        """
        Inference mode: ritorna anomaly scores in [0, +∞) o [0, 1].
        Usa questo metodo per anomaly forecasting senza ri-trainare.
        
        Args:
            node_features: [B, T, D] - residui per ogni serie temporale
            edge_weights: [B, D, D] - matrice di similarità/connessione tra nodi
            score_type: 
                - 'softplus': [0, +∞) - smooth, score alto = più anomalo (CONSIGLIATO)
                - 'sigmoid': [0, 1] - probabilità di anomalia
                - 'exp': [0, +∞) - più aggressivo per outlier estremi
        
        Returns:
            scores: [B, T, 1] - anomaly scores dove più alto = più anomalo
        """
        self.eval()
        logits = self.forward(node_features, edge_weights)  # [B, T, 1]
        
        match score_type:
            case 'sigmoid':
                return torch.sigmoid(logits)  # [0, 1] probabilità
            case 'exp':
                return torch.exp(logits.clamp(max=10))  # [0, +∞) aggressivo, clamped per stabilità
            case _:  # softplus (default)
                return F.softplus(logits)  # [0, +∞) smooth

# ═══════════════════════════════════════════════════════════════
# FUNZIONI DI LOSS PER ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════

def focal_loss(pred: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, alpha: float = 0.25) -> torch.Tensor:
    """
    Focal Loss per gestire class imbalance (poche anomalie).
    """
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt = torch.exp(-bce)
    focal = alpha * (1 - pt) ** gamma * bce
    return focal.mean()


def ranking_loss(scores: torch.Tensor, labels: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """
    Margin ranking loss: anomalie devono avere score > normali + margin.
    Utile quando vuoi uno score continuo ben separato.
    """
    anomaly_mask = labels.squeeze() == 1
    normal_mask = labels.squeeze() == 0
    
    if not anomaly_mask.any() or not normal_mask.any():
        return torch.tensor(0.0, device=scores.device)
    
    anomaly_scores = scores[anomaly_mask]
    normal_scores = scores[normal_mask]
    
    # Ogni anomalia vs ogni normale
    loss = F.relu(margin - anomaly_scores.unsqueeze(1) + normal_scores.unsqueeze(0))
    return loss.mean()


# ═══════════════════════════════════════════════════════════════
# ESEMPIO D'USO
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test con dimensioni variabili (B, T, D)
    for B, T, D in [(8, 64, 11), (4, 100, 5), (16, 50, 20)]:
        print(f"\n{'='*50}")
        print(f"Test con B={B} batch, T={T} timesteps, D={D} features")
        print('='*50)
        
        # Dati simulati
        node_features = torch.randn(B, T, D)  # [B, T, D] residui
        edge_weights = torch.rand(B, D, D)  # [B, D, D] similarità
        edge_weights = (edge_weights + edge_weights.transpose(-1, -2)) / 2  # simmetrizza
        
        # Modello
        model = SpatioTemporalAnomalyGNN(
            hidden_dim=32,
            num_gnn_layers=2,
            num_temporal_layers=3,
            pooling_type='attention',
            output_type='raw'  # logits per BCE loss
        )
        
        # Forward
        scores = model(node_features, edge_weights)
        print(f"Input shape: {node_features.shape}")  # [B, T, D]
        print(f"Output shape: {scores.shape}")  # [B, T, 1]
        print(f"Score range: [{scores.min().item():.3f}, {scores.max().item():.3f}]")
        
        # Training example
        labels = torch.zeros(B, T, 1)
        labels[:, T//3:T//3+5, :] = 1  # anomalia nella finestra centrale
        
        # Loss combinata
        loss_bce = F.binary_cross_entropy_with_logits(scores, labels)
        loss_rank = ranking_loss(scores.view(-1, 1), labels.view(-1, 1))
        loss = loss_bce + 0.5 * loss_rank
        
        print(f"Loss: {loss.item():.4f}")
    
    print("\n✅ Modello funziona con B, T e D variabili!")

    cfg = {
        'train_data_path': "./TRAIN_SPLIT",
        "data_path_custom": "./PROCESSED_TRAIN_DATA",
        'val_data_path': "./TEST_SPLIT",
        'model_save_path': "./SAVED_MODELS",

        'num_epochs': 30,

        'batch_size': 16,
        'lr': 0.01,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'nesterov': True,

        'gnn_hidden_dim': 128,
        'gnn_num_layers': 5,
        'gnn_num_temporal_layers': 5,
        'gnn_dropout': 0.3,

        'warmup_fraction': 0.15,
        
        # Loss parameters per class imbalance (~1-2% anomalie) - STABILI
        'focal_alpha': 0.9,        # peso classe positiva (0.9 = 9x importanza)
        'focal_gamma': 2.0,        # focusing: 2.0 è stabile, >2.5 rischia NaN
        'ranking_margin': 1.0,     # margine minimo score(anomalia) - score(normale)
        'ranking_weight': 0.5,     # peso ranking loss nella loss totale
        'dice_weight': 0.5,        # peso dice loss (ottimizza F1-like)
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpatioTemporalAnomalyGNN(hidden_dim=cfg.get('gnn_hidden_dim', 64), num_gnn_layers=cfg.get('gnn_num_layers', 3), num_temporal_layers=cfg.get('gnn_num_temporal_layers', 3), dropout=cfg.get('gnn_dropout', 0.3)).to(device)
    

    print(sum([param.numel() for layer in model.gnn_layers for param in layer.parameters() if param.requires_grad]))