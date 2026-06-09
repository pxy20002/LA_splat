import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    """Self-Attention + MLP, each with residual connection and LayerNorm."""

    def __init__(self, dim: int = 640, num_heads: int = 10, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True, bias=True,
        )  # uses FlashAttention via PyTorch SDPA backend
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, dim]  or  [N, dim]
        needs_squeeze = x.dim() == 2
        if needs_squeeze:
            x = x.unsqueeze(0)  # [1, N, dim]

        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))

        if needs_squeeze:
            x = x.squeeze(0)
        return x


class GaussianDecoder(nn.Module):
    """
    AnchorSplat Gaussian Decoder (Eq.8: {F_j, A_j} → Attention → MLP → Gaussians).

    Input:  anchor_features [N, D_in] + anchors [N, 3]
    Output: dict of Gaussian attributes per anchor, each anchor → 4 Gaussians.

    Architecture (matching the paper):
      - Input concatenation:  [feature, anchor_xyz]  (D_in + 3 = 67)
      - Input projection:     Linear(D_in+3 → 640)
      - 16 × TransformerBlock (640 dim, 10 heads, mlp_ratio=4)
      - 2 × single-layer MLP  (640 → 640, ReLU)
      - 5 parallel Linear heads:
          delta_μ  [N, 4, 3]    center offset
          opacity  [N, 4, 1]    α  (sigmoid)
          scale    [N, 4, 3]    s  (exp)
          rotation [N, 4, 4]    r  (L2-normalized quaternion)
          sh       [N, 4, 3]    SH DC = RGB (sigmoid)

    Parameters: ~84M  (matches paper Table: 84M).
    """

    def __init__(self, in_dim: int = 64, hidden_dim: int = 640,
                 num_blocks: int = 16, num_heads: int = 10,
                 gaussians_per_anchor: int = 4, pos_dim: int = 3):
        super().__init__()
        self.gaussians_per_anchor = gaussians_per_anchor
        self.pos_dim = pos_dim

        # Input projection: feature + 3D anchor position (Eq.8: {F_j, A_j})
        self.input_proj = nn.Linear(in_dim + pos_dim, hidden_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, mlp_ratio=4.0)
            for _ in range(num_blocks)
        ])

        # Post-transformer MLPs (2 single-layer blocks as per paper)
        self.mlp1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.mlp2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

        # Prediction heads (single-layer, no activation)
        G = gaussians_per_anchor
        self.head_delta_mu = nn.Linear(hidden_dim, G * 3)    # center offset
        self.head_opacity  = nn.Linear(hidden_dim, G * 1)    # α
        self.head_scale    = nn.Linear(hidden_dim, G * 3)    # scale
        self.head_rotation = nn.Linear(hidden_dim, G * 4)    # quaternion
        self.head_sh       = nn.Linear(hidden_dim, G * 3)    # SH DC (RGB)

        # Constrain delta_mu to stay close to anchors (paper: 10/128 ≈ 0.078)
        self.max_offset = 0.1

        # Initialize scale bias so initial scales are reasonable (~0.05)
        nn.init.constant_(self.head_scale.bias, -3.0)  # exp(-3) ≈ 0.05

    def forward(self, x: torch.Tensor, anchors: torch.Tensor = None) -> dict:
        """
        Args:
            x:       [N, D_in] anchor features (or [B, N, D_in]).
            anchors: [N, 3] 3D anchor positions (or [B, N, 3]). Eq.8: {F_j, A_j}.

        Returns:
            dict with keys:
              delta_mu  [*B, N, 4, 3]
              opacity   [*B, N, 4, 1]   (sigmoid-activated)
              scale     [*B, N, 4, 3]   (exp-activated)
              rotation  [*B, N, 4, 4]   (L2-normalized)
              sh        [*B, N, 4, 3]   (raw, degree=0)
        """
        G = self.gaussians_per_anchor
        N = x.shape[-2]

        has_batch = x.dim() == 3
        if not has_batch:
            x = x.unsqueeze(0)  # [1, N, D_in]
        if anchors is not None:
            if anchors.dim() == 2:
                anchors = anchors.unsqueeze(0)  # [1, N, 3]

        B = x.shape[0]

        if anchors is not None and anchors.shape[0] == 1 and B > 1:
            anchors = anchors.expand(B, -1, -1)  # [B, N, 3]

        # Concatenate anchor position (Eq.8: {F_j, A_j})
        if anchors is not None:
            x = torch.cat([x, anchors], dim=-1)  # [B, N, D_in + 3]

        # Project
        x = self.input_proj(x)  # [B, N, 640]

        # Transformer
        for blk in self.blocks:
            x = blk(x)  # maintains shape

        # MLPs
        x = self.mlp1(x)
        x = self.mlp2(x)

        # Heads
        delta_mu = self.head_delta_mu(x)   # [B, N, G*3]
        opacity  = self.head_opacity(x)    # [B, N, G*1]
        scale    = self.head_scale(x)      # [B, N, G*3]
        rotation = self.head_rotation(x)   # [B, N, G*4]
        sh       = self.head_sh(x)         # [B, N, G*3]

        # Reshape and activate
        delta_mu = torch.tanh(delta_mu.reshape(B, N, G, 3)) * self.max_offset
        opacity  = torch.sigmoid(opacity.reshape(B, N, G, 1))
        scale    = torch.exp(scale.reshape(B, N, G, 3)).clamp(max=0.5)  # prevent explosion
        rotation = F.normalize(rotation.reshape(B, N, G, 4), dim=-1)
        sh       = torch.sigmoid(sh.reshape(B, N, G, 3))  # SH degree=0 = RGB, constrain to [0,1]

        if not has_batch:
            delta_mu = delta_mu.squeeze(0)
            opacity  = opacity.squeeze(0)
            scale    = scale.squeeze(0)
            rotation = rotation.squeeze(0)
            sh       = sh.squeeze(0)

        return {
            "delta_mu": delta_mu,
            "opacity":  opacity,
            "scale":    scale,
            "rotation": rotation,
            "sh":       sh,
        }
