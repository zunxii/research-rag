import torch
import torch.nn.functional as F
from typing import Dict

def modality_consistency_constraint(
    img_emb: torch.Tensor,
    txt_emb: torch.Tensor
) -> Dict:
    """
    Measures how consistent image and text embeddings are.
    Returns cosine similarity + interpretation band.
    """

    img_emb = F.normalize(img_emb, dim=-1)
    txt_emb = F.normalize(txt_emb, dim=-1)

    sim = float((img_emb * txt_emb).sum().item())

    if sim >= 0.75:
        level = "high"
    elif sim >= 0.45:
        level = "moderate"
    else:
        level = "low"

    return {
        "cosine_similarity": round(sim, 4),
        "consistency_level": level,
    }
