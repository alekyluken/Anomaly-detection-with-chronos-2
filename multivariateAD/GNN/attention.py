"""
Multi-Head Attention module for cross-variable aggregation.

Replaces SimpleVariablePooling with a learnable, permutation-invariant mechanism.
Uses Multi-Head Self-Attention across the variable dimension D, then
attention-weighted pooling to aggregate D → 1.

Key invariance properties:
- D (variables): NO positional encoding → reordering variables doesn't change output
- T (timesteps): each timestep processed independently → reordering timesteps
  doesn't change individual outputs (temporal patterns already encoded by Conv1D upstream)
"""
import torch
import torch.nn as nn


class MultiHeadVariableAttention(nn.Module):
    """
    Multi-Head Self-Attention across the variable dimension D,
    followed by attention-weighted pooling (D → 1).

    Input:  [B, D, T, H]
    Output: [B, T, H]

    Architecture:
        1. Self-attention layers across D (variables communicate)
        2. Learnable query cross-attends to all D embeddings (pooling)

    No positional encoding → permutation invariant on both D and T.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1,
                num_layers: int = 2):
        super().__init__()
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # ── Self-attention + FFN layers (pre-norm transformer) ──
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'attn': nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                ),
                'attn_norm': nn.LayerNorm(hidden_dim),
                'ffn': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.Dropout(dropout)
                ),
                'ffn_norm': nn.LayerNorm(hidden_dim)
            }))

        # ── Attention pooling: learnable "anomaly query" cross-attends to D ──
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.pool_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D, T, H] – per-variable temporal embeddings

        Returns:
            [B, T, H] – aggregated representation
        """
        B, D, T, H = x.shape

        # Reshape: treat each timestep independently, attend across D
        # [B, D, T, H] → [B*T, D, H]
        x = x.permute(0, 2, 1, 3).reshape(B * T, D, H)

        # Self-attention layers (pre-norm)
        for layer in self.layers:
            # Self-attention across D
            x_norm = layer['attn_norm'](x)
            attn_out, _ = layer['attn'](x_norm, x_norm, x_norm)
            x = x + attn_out

            # Feed-forward
            x = x + layer['ffn'](layer['ffn_norm'](x))

        # Attention pooling: learnable query → aggregate D → 1
        query = self.pool_query.expand(B * T, -1, -1)      # [B*T, 1, H]
        pooled, _ = self.pool_attn(query, x, x)             # [B*T, 1, H]
        pooled = self.pool_norm(pooled.squeeze(1))           # [B*T, H]

        # [B*T, H] → [B, T, H]
        return pooled.reshape(B, T, H)


# ═══════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    for B, D, T, H in [(4, 11, 64, 32), (2, 5, 128, 64), (8, 20, 32, 32)]:
        print(f"Test B={B}, D={D}, T={T}, H={H}")
        x = torch.randn(B, D, T, H)

        attn = MultiHeadVariableAttention(hidden_dim=H, num_heads=4, dropout=0.2, num_layers=2)
        attn.eval()  # disable dropout for deterministic invariance check

        out = attn(x)
        assert out.shape == (B, T, H), f"Expected ({B},{T},{H}), got {out.shape}"

        # Verify permutation invariance on D
        perm = torch.randperm(D)
        out_perm = attn(x[:, perm, :, :])
        diff = (out - out_perm).abs().max().item()
        print(f"  Output: {out.shape}, D-perm invariance diff: {diff:.6f}")

        # Verify independence on T ordering
        perm_t = torch.randperm(T)
        out_t = attn(x[:, :, perm_t, :])
        out_t_unperm = out_t[:, torch.argsort(perm_t), :]
        diff_t = (out - out_t_unperm).abs().max().item()
        print(f"  T-independence diff: {diff_t:.6f}")

        params = sum(p.numel() for p in attn.parameters() if p.requires_grad)
        print(f"  Parameters: {params:,}\n")

    print("All tests passed!")
