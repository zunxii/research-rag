import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveFusion(nn.Module):
    """
    Adaptive residual fusion for multimodal embeddings.
    Trained while encoders are frozen.
    """

    def __init__(self, dim: int = 512, hidden: int = 256):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(2 * dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor):
        x = torch.cat([image_emb, text_emb], dim=-1)
        g = self.gate(x)

        fused = image_emb + g * text_emb
        fused = self.norm(fused)
        fused = F.normalize(fused, dim=-1)

        return fused
