"""
Comprehensive embedding tests
"""

import pytest
from pathlib import Path
import torch
import torch.nn.functional as F
from core.embeddings.biomedclip import BioMedCLIPEncoder


class TestBioMedCLIPEncoder:
    """Test BioMedCLIP encoder functionality."""
    
    def test_encoder_initialization(self, device):
        """Test encoder can be initialized."""
        encoder = BioMedCLIPEncoder(device=device)
        assert encoder.device == device
        assert encoder.dim == 512
    
    def test_image_encoding_shape(self, device, test_image):
        """Test image embedding has correct shape."""
        encoder = BioMedCLIPEncoder(device=device)
        emb = encoder.encode_image(test_image)
        
        assert emb.shape == (512,)
        assert isinstance(emb, torch.Tensor)
    
    def test_text_encoding_shape(self, device, test_text):
        """Test text embedding has correct shape."""
        encoder = BioMedCLIPEncoder(device=device)
        emb = encoder.encode_text(test_text)
        
        assert emb.shape == (512,)
        assert isinstance(emb, torch.Tensor)
    
    def test_embeddings_normalized(self, device, test_image, test_text):
        """Test embeddings are L2 normalized."""
        encoder = BioMedCLIPEncoder(device=device)
        
        img_emb = encoder.encode_image(test_image)
        txt_emb = encoder.encode_text(test_text)
        
        img_norm = torch.norm(img_emb).item()
        txt_norm = torch.norm(txt_emb).item()
        
        assert abs(img_norm - 1.0) < 1e-3, f"Image norm {img_norm} not close to 1.0"
        assert abs(txt_norm - 1.0) < 1e-3, f"Text norm {txt_norm} not close to 1.0"
    
    def test_batch_encoding(self, device, test_image, test_text):
        """Test batch encoding works."""
        encoder = BioMedCLIPEncoder(device=device)
        
        img_batch = encoder.encode_image_batch([test_image, test_image])
        txt_batch = encoder.encode_text_batch([test_text, test_text])
        
        assert img_batch.shape == (2, 512)
        assert txt_batch.shape == (2, 512)
    
    def test_no_nans_in_embeddings(self, device, test_image, test_text):
        """Test embeddings don't contain NaN values."""
        encoder = BioMedCLIPEncoder(device=device)
        
        img_emb = encoder.encode_image(test_image)
        txt_emb = encoder.encode_text(test_text)
        
        assert not torch.isnan(img_emb).any()
        assert not torch.isnan(txt_emb).any()
    
    def test_deterministic_encoding(self, device, test_image):
        """Test encoding is deterministic."""
        encoder = BioMedCLIPEncoder(device=device)
        
        emb1 = encoder.encode_image(test_image)
        emb2 = encoder.encode_image(test_image)
        
        assert torch.allclose(emb1, emb2, atol=1e-6)
    
    def test_reasonable_similarity(self, device, test_image, test_text):
        """Test image-text similarity is in reasonable range."""
        encoder = BioMedCLIPEncoder(device=device)
        
        img_emb = encoder.encode_image(test_image)
        txt_emb = encoder.encode_text(test_text)
        
        sim = F.cosine_similarity(img_emb, txt_emb, dim=0).item()
        
        assert -1.0 <= sim <= 1.0, f"Similarity {sim} out of valid range"


@pytest.mark.skipif(
    not Path("outputs/models/trained_lora").exists(),
    reason="Trained LoRA not available"
)
class TestLoRAEncoder:
    """Test LoRA-adapted encoder."""
    
    def test_lora_encoder_loads(self, device, trained_lora_path):
        """Test LoRA encoder can be loaded."""
        encoder = BioMedCLIPEncoder(device=device, lora_path=trained_lora_path)
        assert encoder.dim == 512
    
    def test_lora_encoding_works(self, device, trained_lora_path, test_image):
        """Test LoRA encoder produces valid embeddings."""
        encoder = BioMedCLIPEncoder(device=device, lora_path=trained_lora_path)
        emb = encoder.encode_image(test_image)
        
        assert emb.shape == (512,)
        assert not torch.isnan(emb).any()

