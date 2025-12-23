# core/reasoning/constraints/base.py
from typing import Dict, Any
import numpy as np

Constraint = Dict[str, Any]

def normalize_scores(scores: np.ndarray) -> np.ndarray:
    total = scores.sum()
    return scores / total if total > 0 else scores

def safe_entropy(probs: np.ndarray) -> float:
    eps = 1e-9
    p = probs + eps
    return float(-(p * np.log(p)).sum())
