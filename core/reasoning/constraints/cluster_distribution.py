# core/reasoning/constraints/cluster_distribution.py
from collections import Counter
import numpy as np
from .base import safe_entropy

def cluster_distribution_constraint(retrieved_metadata: list):
    labels = [m["diagnosis_label"] for m in retrieved_metadata]
    counts = Counter(labels)

    total = sum(counts.values())
    distribution = {
        k: round(v / total, 4) for k, v in counts.items()
    }

    probs = np.array(list(distribution.values()))
    entropy = safe_entropy(probs)

    return {
        "distribution": distribution,
        "entropy": round(entropy, 4),
        "is_concentrated": max(probs) >= 0.5
    }
