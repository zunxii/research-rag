"""
Embedding quality tests
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import torch
from PIL import Image

from core.embeddings.biomedclip import BioMedCLIPEncoder


class EmbeddingQualityTester:
    """Tests embedding quality"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.encoder = BioMedCLIPEncoder(device=device)
    
    def test(self):
        """Run all embedding quality tests"""
        return {
            "dimension": self._test_dimension(),
            "normalization_check": self._test_normalization(),
            "deterministic_check": self._test_deterministic(),
            "no_nans": self._test_no_nans()
        }
    
    def _test_dimension(self):
        """Test embedding dimensions"""
        text = "test query"
        emb = self.encoder.encode_text(text)
        return {
            "shape": list(emb.shape),
            "expected": [512],
            "correct": emb.shape == (512,)
        }
    
    def _test_normalization(self):
        """Test L2 normalization"""
        text = "test query"
        emb = self.encoder.encode_text(text)
        norm = torch.norm(emb).item()
        return {
            "norm": norm,
            "expected": 1.0,
            "correct": abs(norm - 1.0) < 1e-3
        }
    
    def _test_deterministic(self):
        """Test deterministic encoding"""
        text = "test query"
        emb1 = self.encoder.encode_text(text)
        emb2 = self.encoder.encode_text(text)
        return {
            "max_diff": torch.max(torch.abs(emb1 - emb2)).item(),
            "correct": torch.allclose(emb1, emb2, atol=1e-6)
        }
    
    def _test_no_nans(self):
        """Test for NaN values"""
        text = "test query"
        emb = self.encoder.encode_text(text)
        return {
            "has_nans": torch.isnan(emb).any().item(),
            "correct": not torch.isnan(emb).any()
        }

