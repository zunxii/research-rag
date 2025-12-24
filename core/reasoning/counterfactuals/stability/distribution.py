from collections import Counter

def cluster_distribution(retrieved: list) -> dict:
    labels = [r["diagnosis_label"] for r in retrieved]
    counts = Counter(labels)
    total = sum(counts.values()) + 1e-9

    return {
        "distribution": {k: round(v / total, 4) for k, v in counts.items()},
        "num_clusters": len(counts),
    }
