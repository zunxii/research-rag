"""
End-to-end integration tests
"""

import pytest
import torch
from pathlib import Path
from PIL import Image

from core.embeddings.biomedclip import BioMedCLIPEncoder
from core.fusion.adaptive_fusion import AdaptiveFusion
from core.retrieval.retriever import KBRetriever


@pytest.mark.skipif(
    not (Path("outputs/kb/kb_smoke").exists() and 
         Path("outputs/models/trained_fusion/fusion.pt").exists()),
    reason="Required components not available"
)
class TestEndToEndPipeline:
    """Test complete inference pipeline."""
    
    def test_full_inference_pipeline(self, device, test_image, test_text):
        """Test complete inference from query to retrieval."""
        # Load models
        encoder = BioMedCLIPEncoder(device=device)
        fusion = AdaptiveFusion().to(device)
        fusion.load_state_dict(
            torch.load(
                "outputs/models/trained_fusion/fusion.pt",
                map_location=device
            )
        )
        fusion.eval()
        
        # Load retriever
        retriever = KBRetriever("outputs/kb/kb_smoke")
        
        # Encode query
        with torch.no_grad():
            img_emb = encoder.encode_image(test_image).unsqueeze(0)
            txt_emb = encoder.encode_text(test_text).unsqueeze(0)
            query_emb = fusion(img_emb, txt_emb)
        
        # Retrieve
        query_np = query_emb.cpu().numpy()
        scores, indices = retriever.search(query_np, top_k=3)
        
        # Validate
        assert scores.shape == (1, 3)
        assert all(0 <= idx < len(retriever.metadata) for idx in indices[0])
        
        # Check metadata
        for idx in indices[0]:
            case = retriever.metadata[idx]
            assert "diagnosis_label" in case
            assert "image_path" in case