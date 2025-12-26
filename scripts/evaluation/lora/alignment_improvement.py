"""
Alignment improvement analysis
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import torch
import torch.nn.functional as F

from core.embeddings.biomedclip import BioMedCLIPEncoder
from core.kb.image_loader import ImageLoader


class AlignmentAnalyzer:
    """Analyzes alignment improvement from LoRA"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        
        self.base_encoder = BioMedCLIPEncoder(device=device)
        self.lora_encoder = BioMedCLIPEncoder(
            device=device,
            lora_path="outputs/models/trained_lora"
        )
        self.image_loader = ImageLoader()
    
    def analyze(self):
        """Analyze alignment improvement"""
        # Test pairs
        pairs = [
            ("data/images/edema_Image_1.jpg", 
             "swelling noted around joint"),
            ("data/images/cyanosis_Image_1.jpg",
             "bluish discoloration observed")
        ]
        
        base_sims = []
        lora_sims = []
        
        with torch.no_grad():
            for img_path, text in pairs:
                if not Path(img_path).exists():
                    continue
                
                img = self.image_loader.load(img_path)
                
                # Base encoder
                base_img = self.base_encoder.encode_image(img)
                base_txt = self.base_encoder.encode_text(text)
                base_sim = F.cosine_similarity(
                    base_img, base_txt, dim=0
                ).item()
                base_sims.append(base_sim)
                
                # LoRA encoder
                lora_img = self.lora_encoder.encode_image(img)
                lora_txt = self.lora_encoder.encode_text(text)
                lora_sim = F.cosine_similarity(
                    lora_img, lora_txt, dim=0
                ).item()
                lora_sims.append(lora_sim)
        
        if not base_sims:
            return {"status": "no_test_data"}
        
        base_avg = sum(base_sims) / len(base_sims)
        lora_avg = sum(lora_sims) / len(lora_sims)
        
        return {
            "base_similarity": base_avg,
            "lora_similarity": lora_avg,
            "improvement": lora_avg - base_avg,
            "improvement_pct": ((lora_avg - base_avg) / base_avg) * 100,
            "per_pair": [
                {"base": b, "lora": l, "improvement": l - b}
                for b, l in zip(base_sims, lora_sims)
            ]
        }
