"""
Knowledge base builder tests
"""

import pytest
from pathlib import Path
import json
import numpy as np
import faiss

from core.kb.builder import KBBuilder
from core.kb.text_processor import TextProcessor
from core.kb.image_loader import ImageLoader


class TestTextProcessor:
    """Test text processing functionality."""
    
    def test_text_cleaning(self):
        """Test text cleaning works."""
        processor = TextProcessor()
        
        text = "  SWOLLEN   tonsils  \n\n with  FEVER  "
        cleaned = processor.clean_text(text)
        
        assert cleaned == "swollen tonsils with fever"
    
    def test_text_combination(self):
        """Test combining context and description."""
        processor = TextProcessor()
        
        context = "Patient presents with"
        description = "swollen lymph nodes"
        
        combined = processor.combine_text(context, description)
        
        assert "patient presents with" in combined
        assert "swollen lymph nodes" in combined
    
    def test_anatomy_extraction(self):
        """Test anatomy extraction."""
        processor = TextProcessor()
        
        text = "swollen tonsils with fever"
        anatomy = processor.extract_anatomy(text)
        
        assert "raw_mentions" in anatomy
        assert "semantic_types" in anatomy
        assert isinstance(anatomy["raw_mentions"], list)


class TestImageLoader:
    """Test image loading functionality."""
    
    def test_image_loader_initialization(self):
        """Test image loader can be initialized."""
        loader = ImageLoader(image_root="data/images")
        assert loader.image_root == Path("data/images")
    
    @pytest.mark.skipif(
        not Path("data/images/edema_Image_1.jpg").exists(),
        reason="Test image not available"
    )
    def test_image_loading(self):
        """Test image can be loaded."""
        loader = ImageLoader(image_root="data/images")
        img = loader.load("edema_Image_1.jpg")
        
        assert img is not None
        assert img.mode == "RGB"


@pytest.mark.skipif(
    not Path("outputs/kb/kb_smoke").exists(),
    reason="Smoke test KB not available"
)
class TestKBStructure:
    """Test KB structure and contents."""
    
    def test_kb_files_exist(self, kb_dir):
        """Test all KB files exist."""
        assert (kb_dir / "embeddings.npy").exists()
        assert (kb_dir / "index.faiss").exists()
        assert (kb_dir / "metadata.json").exists()
        assert (kb_dir / "kb_config.json").exists()
        assert (kb_dir / "build_report.json").exists()
    
    def test_kb_embeddings_shape(self, kb_dir):
        """Test embeddings have correct shape."""
        embeddings = np.load(kb_dir / "embeddings.npy")
        
        assert len(embeddings.shape) == 2
        assert embeddings.shape[1] == 512
    
    def test_kb_index_consistency(self, kb_dir):
        """Test FAISS index consistency."""
        index = faiss.read_index(str(kb_dir / "index.faiss"))
        embeddings = np.load(kb_dir / "embeddings.npy")
        
        assert index.ntotal == len(embeddings)
    
    def test_kb_metadata_structure(self, kb_dir):
        """Test metadata has correct structure."""
        with open(kb_dir / "metadata.json") as f:
            metadata = json.load(f)
        
        assert isinstance(metadata, list)
        
        if len(metadata) > 0:
            entry = metadata[0]
            assert "case_id" in entry
            assert "diagnosis_label" in entry
            assert "image_path" in entry
            assert "clinical_text" in entry
            assert "embedding_id" in entry
    
    def test_kb_config_structure(self, kb_dir):
        """Test KB config has required fields."""
        with open(kb_dir / "kb_config.json") as f:
            config = json.load(f)
        
        assert "num_entries" in config
        assert "embedding_dim" in config
        assert config["embedding_dim"] == 512
