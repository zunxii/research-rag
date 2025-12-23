# core/reasoning/constraints/modality_consistency.py
import numpy as np

def modality_consistency_constraint(
    img_emb: np.ndarray,
    txt_emb: np.ndarray
):
    # cosine similarity
    cross_sim = float(
        np.dot(img_emb, txt_emb) /
        (np.linalg.norm(img_emb) * np.linalg.norm(txt_emb) + 1e-9)
    )

    return {
        "consistent": cross_sim >= 0.6,
        "cross_similarity": round(cross_sim, 4)
    }
