import torch
from core.fusion.adaptive_fusion import AdaptiveFusion

fusion = AdaptiveFusion()

v = torch.randn(1, 512)
t = torch.randn(1, 512)

z = fusion(v, t)

print("Fused shape:", z.shape)
print("Norm:", z.norm(dim=-1).item())
