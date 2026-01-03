import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFusion(nn.Module):
    """
    Gated multimodal fusion.
    Learns how much to trust image vs text.
    """

    def __init__(self, dim: int = 512):
        super().__init__()

        # modality projections
        self.proj_img = nn.Linear(dim, dim)
        self.proj_txt = nn.Linear(dim, dim)

        # gating network
        self.gate = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, img_emb, txt_emb):
        """
        img_emb: (N, D)
        txt_emb: (N, D)
        """

        img_p = self.proj_img(img_emb)
        txt_p = self.proj_txt(txt_emb)

        concat = torch.cat([img_p, txt_p], dim=-1)
        alpha = self.gate(concat)  # (N, 1)

        fused = alpha * img_p + (1 - alpha) * txt_p
        fused = self.norm(fused)
        fused = F.normalize(fused, dim=-1)

        return fused
