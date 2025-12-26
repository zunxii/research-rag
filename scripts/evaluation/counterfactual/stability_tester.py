"""
Stability testing logic
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import json
import torch
import faiss

from core.embeddings.biomedclip import BioMedCLIPEncoder
from core.fusion.adaptive_fusion import AdaptiveFusion
from core.kb.image_loader import ImageLoader
from core.reasoning.counterfactuals.stability.runner import StabilityRunner
from core.reasoning.counterfactuals.stability.retrieval import StabilityRetriever


class StabilityTester:
    """Tests counterfactual stability"""
    
    def __init__(self, kb_dir: str, device: str = "cpu"):
        self.device = device
        
        # Load KB
        self.index = faiss.read_index(f"{kb_dir}/index.faiss")
        with open(f"{kb_dir}/metadata.json") as f:
            self.metadata = json.load(f)
        
        # Load models
        self.encoder = BioMedCLIPEncoder(device=device)
        self.image_loader = ImageLoader()
        
        # Load fusion
        fusion_path = Path("outputs/models/trained_fusion/fusion.pt")
        self.fusion = AdaptiveFusion().to(device)
        self.fusion.load_state_dict(
            torch.load(fusion_path, map_location=device)
        )
        self.fusion.eval()
        
        # Setup stability runner
        retriever = StabilityRetriever(self.index, self.metadata)
        self.runner = StabilityRunner(retriever, self.fusion)
    
    def get_metadata(self):
        """Get metadata for sampling"""
        return self.metadata
    
    def test_sample(self, idx: int):
        """Test stability for a single sample"""
        entry = self.metadata[idx]
        
        # Encode
        img = self.image_loader.load(entry["image_path"])
        
        with torch.no_grad():
            img_emb = self.encoder.encode_image(img).unsqueeze(0)
            txt_emb = self.encoder.encode_text(
                entry["clinical_text"]["combined"]
            ).unsqueeze(0)
        
        # Run stability analysis
        stability_output = self.runner.run(img_emb, txt_emb)
        
        return {
            "case_id": entry["case_id"],
            "diagnosis": entry["diagnosis_label"],
            "stability": stability_output["stability"],
            "baseline_distribution": stability_output["baseline"]
        }
