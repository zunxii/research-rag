"""
Base vs LoRA embedding comparison
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import torch
import torch.nn.functional as F

from core.embeddings.biomedclip import BioMedCLIPEncoder


class EmbeddingComparator:
    """Compares base and LoRA embeddings"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        
        # Load both encoders
        self.base_encoder = BioMedCLIPEncoder(device=device)
        self.lora_encoder = BioMedCLIPEncoder(
            device=device,
            lora_path="outputs/models/trained_lora"
        )
    
    def compare(self):
        """Compare embeddings from both encoders"""
        test_queries = [
            "swelling around the ankle",
            "bluish discoloration of fingertips",
            "redness and inflammation"
        ]
        
        distances = []
        similarities = []
        
        with torch.no_grad():
            for text in test_queries:
                base_emb = self.base_encoder.encode_text(text)
                lora_emb = self.lora_encoder.encode_text(text)
                
                # L2 distance
                dist = torch.norm(base_emb - lora_emb).item()
                distances.append(dist)
                
                # Cosine similarity
                sim = F.cosine_similarity(
                    base_emb, lora_emb, dim=0
                ).item()
                similarities.append(sim)
        
        return {
            "distance_l2": sum(distances) / len(distances),
            "cosine_similarity": sum(similarities) / len(similarities),
            "per_query": [
                {"distance": d, "similarity": s}
                for d, s in zip(distances, similarities)
            ]
        }
