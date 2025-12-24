from collections import Counter
from typing import Dict

def cluster_distribution_constraint(retrieved_metadata: list) -> Dict:
    """
    Returns soft distribution over diagnosis clusters.
    Example: { "edema": 0.7, "foot swelling": 0.3 }
    """

    labels = [m["diagnosis_label"] for m in retrieved_metadata]
    counts = Counter(labels)

    total = sum(counts.values()) or 1

    distribution = {
        label: round(count / total, 4)
        for label, count in counts.items()
    }

    return {
        "distribution": distribution,
        "num_clusters": len(distribution),
    }
