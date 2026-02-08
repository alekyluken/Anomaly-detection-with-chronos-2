import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

# ═══════════════════════════════════════════════════════════════
# MIGLIORAMENTO 1: Temporal Processing con Multi-Scale Features
# ═══════════════════════════════════════════════════════════════

class MultiScaleTemporalProcessor(nn.Module):
    """
    Cattura pattern temporali a scale diverse (short/medium/long term).
    Usa dilated convolutions parallele + residual connections.
    """
    def __init__(self, hidden_dim: int = 64, kernel_size: int = 7, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Initial projection
        self.input_proj = nn.Linear(1, hidden_dim)
        
        # Multi-scale branches (diversi receptive fields)
        self.branches = nn.ModuleList()
        scales = [1, 2, 4, 8]  # dilation rates
        
        for scale in scales:
            branch = nn.ModuleList()
            num_groups = min(8, hidden_dim)
            while hidden_dim % num_groups != 0:
                num_groups -= 1
                
            for i in range(num_layers):
                dilation = scale * (2 ** i)
                padding = (kernel_size - 1) * dilation
                branch.append(nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size, 
                             dilation=dilation, padding=padding),
                    nn.GroupNorm(num_groups, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout)
                ))
            self.branches.append(branch)
        
        # Fusion layer (combina le scale)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * len(scales), hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D] → [B, D, T, H]"""
        B, T, D = x.shape
        
        # Project: [B, T, D, 1] → [B, T, D, H]
        h = self.input_proj(x.unsqueeze(-1))
        
        # Reshape per Conv1D: [B*D, H, T]
        h = h.permute(0, 2, 3, 1).reshape(B * D, self.hidden_dim, T)
        
        # Multi-scale processing
        branch_outputs = []
        for branch in self.branches:
            h_branch = h
            for conv in branch:
                h_branch = conv(h_branch)[..., :T] + h_branch  # residual
            branch_outputs.append(h_branch)
        
        # Concatenate scales: [B*D, H*num_scales, T]
        h_multi = torch.cat(branch_outputs, dim=1)
        
        # Fusion: [B*D, H*num_scales, T] → [B*D, H, T]
        h_multi = h_multi.permute(0, 2, 1)  # [B*D, T, H*num_scales]
        h_fused = self.fusion(h_multi)  # [B*D, T, H]
        h_fused = h_fused.permute(0, 2, 1)  # [B*D, H, T]
        
        # Reshape back: [B, D, T, H]
        h_out = h_fused.reshape(B, D, self.hidden_dim, T).permute(0, 1, 3, 2)
        
        return self.norm(h_out)


# ═══════════════════════════════════════════════════════════════
# MIGLIORAMENTO 2: Graph Attention Networks (GAT) invece di semplice message passing
# ═══════════════════════════════════════════════════════════════

class GraphAttentionLayer(nn.Module):
    """
    GAT layer: apprende dinamicamente l'importanza di ogni vicino.
    Più robusto del semplice weight-based message passing.
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Multi-head attention
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Edge feature integration
        self.edge_proj = nn.Linear(1, num_heads)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        h: [B, D, T, H]
        W: [B, D, D] - edge weights
        """
        B, D, T, H = h.shape
        
        # Reshape per attention: [B, D, T, H] → [B*T, D, H]
        h_flat = h.permute(0, 2, 1, 3).reshape(B * T, D, H)
        
        # Multi-head projections
        Q = self.q_proj(h_flat).view(B * T, D, self.num_heads, self.head_dim)
        K = self.k_proj(h_flat).view(B * T, D, self.num_heads, self.head_dim)
        V = self.v_proj(h_flat).view(B * T, D, self.num_heads, self.head_dim)
        
        # Compute attention scores: [B*T, num_heads, D, D]
        scores = torch.einsum('bqhd,bkhd->bhqk', Q, K) / (self.head_dim ** 0.5)
        
        # Integrate edge weights: [B, D, D] → [B*T, num_heads, D, D]
        edge_bias = self.edge_proj(W.unsqueeze(-1))  # [B, D, D, num_heads]
        edge_bias = edge_bias.permute(0, 3, 1, 2)  # [B, num_heads, D, D]
        edge_bias = edge_bias.unsqueeze(1).expand(B, T, -1, -1, -1).reshape(B * T, self.num_heads, D, D)
        
        scores = scores + edge_bias
        
        # Softmax + dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention: [B*T, num_heads, D, head_dim]
        out = torch.einsum('bhqk,bkhd->bqhd', attn, V)
        out = out.reshape(B * T, D, H)
        
        # Output projection + residual
        out = self.out_proj(out)
        h_flat = self.norm(h_flat + out)
        
        # FFN + residual
        h_flat = self.norm2(h_flat + self.ffn(h_flat))
        
        # Reshape back: [B, D, T, H]
        return h_flat.view(B, T, D, H).permute(0, 2, 1, 3)


# ═══════════════════════════════════════════════════════════════
# MIGLIORAMENTO 3: Attention-based Pooling con Context
# ═══════════════════════════════════════════════════════════════

class ContextAwarePooling(nn.Module):
    """
    Pooling che considera il contesto temporale globale.
    Usa self-attention per pesare i nodi in base al loro ruolo nell'anomalia.
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Cross-attention: temporal queries, node keys/values
        self.temporal_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.node_k = nn.Linear(hidden_dim, hidden_dim)
        self.node_v = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, D, T, H] → [B, T, H]"""
        B, D, T, H = h.shape
        
        # Reshape: [B, D, T, H] → [B, T, D, H]
        h = h.permute(0, 2, 1, 3)
        
        # Expand query per ogni batch e timestep
        q = self.temporal_query.expand(B, T, -1)  # [B, T, H]
        
        # Node keys/values: [B, T, D, H]
        k = self.node_k(h)
        v = self.node_v(h)
        
        # Multi-head attention
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, T, head_dim]
        k = k.view(B, T, D, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)  # [B, num_heads, T, D, head_dim]
        v = v.view(B, T, D, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        
        # Attention scores: [B, num_heads, T, D]
        scores = torch.einsum('bhtd,bhtnd->bhtn', q, k) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention: [B, num_heads, T, head_dim]
        out = torch.einsum('bhtn,bhtnd->bhtd', attn, v)
        
        # Reshape: [B, T, H]
        out = out.transpose(1, 2).reshape(B, T, H)
        
        return self.norm(out)


# ═══════════════════════════════════════════════════════════════
# MIGLIORAMENTO 4: Dual-Head Decoder con Calibration
# ═══════════════════════════════════════════════════════════════

class CalibratedDualHeadDecoder(nn.Module):
    """
    Decoder con:
    1. Continuous anomaly score head
    2. Binary classification head
    3. Temperature scaling per calibration
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        
        # Shared backbone
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Dual heads
        self.continuous_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        self.binary_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Temperature parameter per calibration (learnable)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """h: [B, T, H] → (continuous_scores, binary_logits)"""
        shared = self.shared(h)
        
        continuous = self.continuous_head(shared)
        binary = self.binary_head(shared) / self.temperature.abs().clamp(min=0.1, max=5.0)
        
        return continuous, binary


# ═══════════════════════════════════════════════════════════════
# MODELLO COMPLETO V2
# ═══════════════════════════════════════════════════════════════

class SpatioTemporalAnomalyGNN_V2(nn.Module):
    """
    Versione migliorata con:
    - Multi-scale temporal processing
    - Graph Attention Networks
    - Context-aware pooling
    - Calibrated dual-head decoder
    """
    def __init__(
        self, 
        hidden_dim: int = 128,
        num_gnn_layers: int = 4,
        num_temporal_layers: int = 3,
        num_heads: int = 4,
        kernel_size: int = 7,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # 1. Multi-scale temporal encoder
        self.temporal_encoder = MultiScaleTemporalProcessor(
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_layers=num_temporal_layers,
            dropout=dropout
        )
        
        # 2. Graph Attention layers
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_gnn_layers)
        ])
        
        # 3. Context-aware pooling
        self.pooling = ContextAwarePooling(hidden_dim, num_heads, dropout)
        
        # 4. Calibrated decoder
        self.decoder = CalibratedDualHeadDecoder(hidden_dim, dropout)
        
    def forward(self, node_features: torch.Tensor, edge_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_features: [B, T, D]
            edge_weights: [B, D, D]
        
        Returns:
            continuous_scores: [B, T, 1]
            binary_logits: [B, T, 1]
        """
        # 1. Temporal encoding
        h = self.temporal_encoder(node_features)  # [B, D, T, H]
        
        # 2. Graph attention
        for gat in self.gat_layers:
            h = gat(h, edge_weights)  # [B, D, T, H]
        
        # 3. Pooling
        h = self.pooling(h)  # [B, T, H]
        
        # 4. Decode
        return self.decoder(h)  # (continuous, binary)


# ═══════════════════════════════════════════════════════════════
# ESEMPIO D'USO
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test
    B, T, D = 8, 64, 11
    x = torch.randn(B, T, D).to(device)
    W = torch.rand(B, D, D).to(device)
    W = (W + W.transpose(-1, -2)) / 2
    
    model = SpatioTemporalAnomalyGNN_V2(
        hidden_dim=128,
        num_gnn_layers=4,
        num_temporal_layers=3,
        num_heads=4,
        dropout=0.2
    ).to(device)
    
    continuous, binary = model(x, W)
    print(f"Input: {x.shape}")
    print(f"Continuous scores: {continuous.shape}, range [{continuous.min():.3f}, {continuous.max():.3f}]")
    print(f"Binary logits: {binary.shape}, range [{binary.min():.3f}, {binary.max():.3f}]")
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
