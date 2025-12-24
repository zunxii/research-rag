"""
Retrieval system tests
"""

import pytest
import numpy as np
import faiss
from pathlib import Path

from core.retrieval.retriever import KBRetriever
from core.retrieval.evaluate import evaluate_retrieval


@pytest.mark.skipif(
    not Path("outputs/kb/kb_smoke").exists(),
    reason="KB not available"
)
class TestKBRetriever:
    """Test retrieval functionality."""
    
    def test_retriever_initialization(self, kb_dir):
        """Test retriever can be initialized."""
        retriever = KBRetriever(str(kb_dir))
        
        assert retriever.index is not None
        assert retriever.metadata is not None
        assert retriever.embeddings is not None
    
    def test_retrieval_shape(self, kb_dir):
        """Test retrieval returns correct shapes."""
        retriever = KBRetriever(str(kb_dir))
        
        query_emb = np.random.randn(1, 512).astype("float32")
        scores, indices = retriever.search(query_emb, top_k=3)
        
        assert scores.shape == (1, 3)
        assert indices.shape == (1, 3)
    
    def test_retrieval_scores_valid(self, kb_dir):
        """Test retrieval scores are in valid range."""
        retriever = KBRetriever(str(kb_dir))
        
        query_emb = np.random.randn(1, 512).astype("float32")
        scores, indices = retriever.search(query_emb, top_k=3)
        
        assert np.all(scores >= -1.0)
        assert np.all(scores <= 1.0)
    
    def test_retrieval_indices_valid(self, kb_dir):
        """Test retrieval indices are valid."""
        retriever = KBRetriever(str(kb_dir))
        
        query_emb = np.random.randn(1, 512).astype("float32")
        scores, indices = retriever.search(query_emb, top_k=3)
        
        assert np.all(indices >= 0)
        assert np.all(indices < retriever.index.ntotal)


class TestRetrievalEvaluation:
    """Test retrieval evaluation metrics."""
    
    def test_evaluation_metrics(self):
        """Test evaluation produces correct metrics."""
        # Mock retrieved results
        results = [
            {"diagnosis_label": "edema"},
            {"diagnosis_label": "cyanosis"},
            {"diagnosis_label": "edema"},
        ]
        
        metrics = evaluate_retrieval(results, "edema")
        
        assert "R@1" in metrics
        assert "R@5" in metrics
        assert "P@1" in metrics
        assert "MRR" in metrics
    
    def test_perfect_recall(self):
        """Test perfect recall case."""
        results = [{"diagnosis_label": "edema"}] * 10
        metrics = evaluate_retrieval(results, "edema")
        
        assert metrics["R@1"] == 1
        assert metrics["R@5"] == 1
        assert metrics["P@1"] == 1.0
    
    def test_zero_recall(self):
        """Test zero recall case."""
        results = [{"diagnosis_label": "cyanosis"}] * 10
        metrics = evaluate_retrieval(results, "edema")
        
        assert metrics["R@1"] == 0
        assert metrics["P@1"] == 0.0
