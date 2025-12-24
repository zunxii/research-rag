"""
Fusion module tests
"""

import pytest
from pathlib import Path
import torch
from core.fusion.adaptive_fusion import AdaptiveFusion
from core.embeddings.fusion import GatedFusion


class TestAdaptiveFusion:
    """Test adaptive fusion module."""
    
    def test_fusion_initialization(self):
        """Test fusion can be initialized."""
        fusion = AdaptiveFusion(dim=512)
        assert fusion is not None
    
    def test_fusion_forward_shape(self):
        """Test fusion output has correct shape."""
        fusion = AdaptiveFusion(dim=512)
        
        img_emb = torch.randn(2, 512)
        txt_emb = torch.randn(2, 512)
        
        fused = fusion(img_emb, txt_emb)
        
        assert fused.shape == (2, 512)
    
    def test_fusion_output_normalized(self):
        """Test fusion output is normalized."""
        fusion = AdaptiveFusion(dim=512)
        
        img_emb = torch.randn(2, 512)
        txt_emb = torch.randn(2, 512)
        
        fused = fusion(img_emb, txt_emb)
        
        norms = torch.norm(fused, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3)
    
    def test_fusion_no_nans(self):
        """Test fusion doesn't produce NaNs."""
        fusion = AdaptiveFusion(dim=512)
        
        img_emb = torch.randn(2, 512)
        txt_emb = torch.randn(2, 512)
        
        fused = fusion(img_emb, txt_emb)
        
        assert not torch.isnan(fused).any()
    
    def test_fusion_deterministic(self):
        """Test fusion is deterministic."""
        fusion = AdaptiveFusion(dim=512)
        fusion.eval()
        
        img_emb = torch.randn(2, 512)
        txt_emb = torch.randn(2, 512)
        
        fused1 = fusion(img_emb, txt_emb)
        fused2 = fusion(img_emb, txt_emb)
        
        assert torch.allclose(fused1, fused2)


class TestGatedFusion:
    """Test gated fusion module."""
    
    def test_gated_fusion_works(self):
        """Test gated fusion basic functionality."""
        fusion = GatedFusion(dim=512)
        
        img_emb = torch.randn(2, 512)
        txt_emb = torch.randn(2, 512)
        
        fused = fusion(img_emb, txt_emb)
        
        assert fused.shape == (2, 512)
        assert not torch.isnan(fused).any()


@pytest.mark.skipif(
    not Path("outputs/models/trained_fusion/fusion.pt").exists(),
    reason="Trained fusion not available"
)
class TestTrainedFusion:
    """Test trained fusion model."""
    
    def test_trained_fusion_loads(self, device, trained_fusion_path):
        """Test trained fusion can be loaded."""
        fusion = AdaptiveFusion().to(device)
        fusion.load_state_dict(torch.load(trained_fusion_path, map_location=device))
        fusion.eval()
        
        assert fusion is not None
    
    def test_trained_fusion_inference(self, device, trained_fusion_path):
        """Test trained fusion produces valid output."""
        fusion = AdaptiveFusion().to(device)
        fusion.load_state_dict(torch.load(trained_fusion_path, map_location=device))
        fusion.eval()
        
        img_emb = torch.randn(1, 512).to(device)
        txt_emb = torch.randn(1, 512).to(device)
        
        with torch.no_grad():
            fused = fusion(img_emb, txt_emb)
        
        assert fused.shape == (1, 512)
        assert not torch.isnan(fused).any()
