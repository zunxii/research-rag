"""
Retrieval metrics computation
"""
import numpy as np
from typing import List, Dict


class MetricsCalculator:
    """Calculates retrieval metrics"""
    
    def compute_all_metrics(self, retrieved: List[Dict], 
                           target_label: str) -> Dict:
        """Compute all retrieval metrics"""
        hits = [1 if r["diagnosis_label"] == target_label else 0 
                for r in retrieved]
        
        return {
            **self._recall_precision(hits),
            "MRR": self._mrr(hits),
            "MAP": self._map(hits),
            **self._ndcg(hits)
        }
    
    def _recall_precision(self, hits: List[int]) -> Dict:
        """Recall and Precision at K"""
        metrics = {}
        for k in [1, 5, 10, 20]:
            if k <= len(hits):
                topk = hits[:k]
                metrics[f"R@{k}"] = 1.0 if sum(topk) > 0 else 0.0
                metrics[f"P@{k}"] = sum(topk) / k
        return metrics
    
    def _mrr(self, hits: List[int]) -> float:
        """Mean Reciprocal Rank"""
        for i, h in enumerate(hits):
            if h == 1:
                return 1.0 / (i + 1)
        return 0.0
    
    def _map(self, hits: List[int]) -> float:
        """Mean Average Precision"""
        relevant_count = 0
        precision_sum = 0.0
        for i, h in enumerate(hits):
            if h == 1:
                relevant_count += 1
                precision_sum += relevant_count / (i + 1)
        
        total_relevant = sum(hits)
        return precision_sum / total_relevant if total_relevant > 0 else 0.0
    
    def _ndcg(self, hits: List[int]) -> Dict:
        """Normalized Discounted Cumulative Gain"""
        metrics = {}
        for k in [5, 10]:
            if k <= len(hits):
                dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits[:k]))
                ideal = sorted(hits[:k], reverse=True)
                idcg = sum(h / np.log2(i + 2) for i, h in enumerate(ideal))
                metrics[f"NDCG@{k}"] = dcg / idcg if idcg > 0 else 0.0
        return metrics

