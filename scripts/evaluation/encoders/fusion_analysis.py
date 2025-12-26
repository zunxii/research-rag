"""
Fusion model analysis
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import torch
import numpy as np

from core.fusion.adaptive_fusion import AdaptiveFusion


class FusionAnalyzer:
    """Analyzes fusion model behavior"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        
        # Load trained fusion if available
        fusion_path = Path("outputs/models/trained_fusion/fusion.pt")
        if fusion_path.exists():
            self.fusion = AdaptiveFusion().to(device)
            self.fusion.load_state_dict(
                torch.load(fusion_path, map_location=device)
            )
            self.fusion.eval()
            self.has_fusion = True
        else:
            self.fusion = None
            self.has_fusion = False
    
    def analyze(self):
        """Run fusion analysis"""
        if not self.has_fusion:
            return {"status": "fusion_model_not_found"}
        
        return {
            "gate_statistics": self._analyze_gate_behavior(),
            "output_normalized": self._test_output_normalization(),
            "modality_sensitivity": self._test_modality_sensitivity()
        }
    
    def _analyze_gate_behavior(self):
        """Analyze gating mechanism"""
        gates = []
        
        with torch.no_grad():
            for _ in range(100):
                img_emb = torch.randn(1, 512).to(self.device)
                txt_emb = torch.randn(1, 512).to(self.device)
                
                # Get gate value
                concat = torch.cat([img_emb, txt_emb], dim=-1)
                gate = self.fusion.gate(concat).item()
                gates.append(gate)
        
        return {
            "mean": float(np.mean(gates)),
            "std": float(np.std(gates)),
            "min": float(np.min(gates)),
            "max": float(np.max(gates))
        }
    
    def _test_output_normalization(self):
        """Test if output is normalized"""
        with torch.no_grad():
            img_emb = torch.randn(1, 512).to(self.device)
            txt_emb = torch.randn(1, 512).to(self.device)
            
            fused = self.fusion(img_emb, txt_emb)
            norm = torch.norm(fused).item()
        
        return {
            "norm": norm,
            "correct": abs(norm - 1.0) < 1e-3
        }
    
    def _test_modality_sensitivity(self):
        """Test sensitivity to each modality"""
        results = {}
        
        with torch.no_grad():
            # Baseline
            img_emb = torch.randn(1, 512).to(self.device)
            txt_emb = torch.randn(1, 512).to(self.device)
            baseline = self.fusion(img_emb, txt_emb)
            
            # Zero out image
            no_img = self.fusion(torch.zeros_like(img_emb), txt_emb)
            results["no_image_similarity"] = \
                torch.cosine_similarity(baseline, no_img, dim=-1).item()
            
            # Zero out text
            no_txt = self.fusion(img_emb, torch.zeros_like(txt_emb))
            results["no_text_similarity"] = \
                torch.cosine_similarity(baseline, no_txt, dim=-1).item()
        
        return results
