"""
Mode-specific evaluation (text, image, fusion)
"""
import sys
from pathlib import Path
from typing import Dict

from scripts.evaluation.retrieval.query_dataset import ExternalQueryDataset
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import torch
from tqdm import tqdm
from collections import defaultdict

from core.embeddings.biomedclip import BioMedCLIPEncoder
from core.fusion.adaptive_fusion import AdaptiveFusion
from core.kb.image_loader import ImageLoader
from core.retrieval.retriever import KBRetriever
from scripts.evaluation.retrieval.metrics import MetricsCalculator


class ModeEvaluator:
    """Evaluates different retrieval modes"""
    
    def __init__(self, kb_dir: str, query_csv, device: str = "cpu"):
        self.device = device
        self.retriever = KBRetriever(kb_dir)
        self.kb_metadata = self.retriever.metadata
        self.query_dataset = ExternalQueryDataset(query_csv)

        
        # Load models
        self.encoder = BioMedCLIPEncoder(device=device)
        self.image_loader = ImageLoader()
        
        # Load fusion if available
        fusion_path = Path("outputs/models/trained_fusion/fusion.pt")
        if fusion_path.exists():
            self.fusion = AdaptiveFusion().to(device)
            self.fusion.load_state_dict(
                torch.load(fusion_path, map_location=device)
            )
            self.fusion.eval()
        else:
            self.fusion = None
        
        self.metrics_calc = MetricsCalculator()
    
    def evaluate_mode(self, mode: str, top_k: int = 20):
        all_metrics = defaultdict(list)

        for query in tqdm(self.query_dataset, desc=f"Evaluating {mode}"):
            gt_label = query["label"]

            query_emb = self._encode_query_external(query, mode)
            if query_emb is None:
                continue

            query_np = query_emb.cpu().numpy().reshape(1, -1)
            _, indices = self.retriever.search(query_np, top_k)

            retrieved = [self.kb_metadata[idx] for idx in indices]

            metrics = self.metrics_calc.compute_all_metrics(
                retrieved, gt_label
            )

            for k, v in metrics.items():
                all_metrics[k].append(v)

        return {
            "metrics": {k: sum(v)/len(v) for k,v in all_metrics.items()},
            "num_queries": len(self.query_dataset)
        }
    
    def _encode_query(self, entry: Dict, mode: str):
        """Encode query based on mode"""
        with torch.no_grad():
            if mode == "text":
                return self.encoder.encode_text(
                    entry["clinical_text"]["combined"]
                )
            
            elif mode == "image":
                img = self.image_loader.load(entry["image_path"])
                return self.encoder.encode_image(img)
            
            elif mode == "fusion":
                if self.fusion is None:
                    return None
                img = self.image_loader.load(entry["image_path"])
                img_emb = self.encoder.encode_image(img).unsqueeze(0)
                txt_emb = self.encoder.encode_text(
                    entry["clinical_text"]["combined"]
                ).unsqueeze(0)
                return self.fusion(img_emb, txt_emb).squeeze(0)
        
        return None
    def _encode_query_external(self, query, mode):
        with torch.no_grad():
            if mode == "text":
                return self.encoder.encode_text(query["text"])

            elif mode == "image":
                img = self.image_loader.load(query["image_path"])
                return self.encoder.encode_image(img)

            elif mode == "fusion":
                if self.fusion is None:
                    return None
                img = self.image_loader.load(query["image_path"])
                img_emb = self.encoder.encode_image(img).unsqueeze(0)
                txt_emb = self.encoder.encode_text(query["text"]).unsqueeze(0)
                return self.fusion(img_emb, txt_emb).squeeze(0)

