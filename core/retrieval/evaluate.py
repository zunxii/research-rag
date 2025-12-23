import numpy as np
from collections import defaultdict


def evaluate_retrieval(results, query_label):
    """
    results: list of retrieved metadata entries (excluding self)
    query_label: ground truth diagnosis label
    """

    hits = [1 if r["diagnosis_label"] == query_label else 0 for r in results]

    metrics = {}

    for k in [1, 5, 10]:
        topk = hits[:k]
        metrics[f"R@{k}"] = int(any(topk))
        metrics[f"P@{k}"] = sum(topk) / k

    # MRR
    mrr = 0.0
    for i, h in enumerate(hits):
        if h == 1:
            mrr = 1.0 / (i + 1)
            break

    metrics["MRR"] = mrr
    return metrics
