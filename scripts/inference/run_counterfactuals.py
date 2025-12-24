"""
Counterfactual reasoning
"""

import json
import torch
import faiss
import numpy as np
from PIL import Image
import sys
from pathlib import Path
import argparse

# Fix imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.embeddings.biomedclip import BioMedCLIPEncoder
from core.fusion.adaptive_fusion import AdaptiveFusion
from core.reasoning.counterfactuals.stability.retrieval import StabilityRetriever
from core.reasoning.counterfactuals.stability.runner import StabilityRunner
from core.reasoning.counterfactuals.diagnostics.scorer import CounterfactualScorer
from configs.inference_config import INFERENCE_CONFIG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-text", required=True)
    parser.add_argument("--query-image", required=True)
    parser.add_argument("--kb-dir", default=INFERENCE_CONFIG["kb_dir"])
    args = parser.parse_args()

    KB_DIR = args.kb_dir
    QUERY_TEXT = args.query_text
    QUERY_IMAGE_PATH = args.query_image
    DEVICE = INFERENCE_CONFIG["device"]

    print("  Loading KB...")
    index = faiss.read_index(f"{KB_DIR}/index.faiss")
    with open(f"{KB_DIR}/metadata.json") as f:
        metadata = json.load(f)

    print("  Loading encoders + trained fusion...")
    encoder = BioMedCLIPEncoder(
        device=DEVICE,
        lora_path=INFERENCE_CONFIG.get("lora_path")
    )

    fusion = AdaptiveFusion().to(DEVICE)
    fusion.load_state_dict(
        torch.load(
            INFERENCE_CONFIG["fusion_path"],
            map_location=DEVICE
        )
    )
    fusion.eval()

    retriever = StabilityRetriever(index, metadata)

    # ---------------- Encode query (original) ----------------
    print("\n✓ Encoding query...")

    img = Image.open(QUERY_IMAGE_PATH).convert("RGB")

    with torch.no_grad():
        img_emb = encoder.encode_image(img).unsqueeze(0)
        txt_emb = encoder.encode_text(QUERY_TEXT).unsqueeze(0)

    # ---------------- Level 1: Stability (original) ----------------
    print("\n✓ Running Level 1: Counterfactual Stability...")
    stability_runner = StabilityRunner(retriever, fusion)
    stability_output = stability_runner.run(img_emb, txt_emb)

    print("\n=== STABILITY OUTPUT ===")
    print(json.dumps(stability_output, indent=2))

    # ---------------- Level 2: Diagnostic Scoring (original) ----------------
    print("\n✓ Running Level 2: Diagnostic Scoring...")
    scorer = CounterfactualScorer()
    scored = scorer.score(stability_output)

    print("\n=== DIAGNOSTIC SCORES ===")
    for s in scored:
        print(s)


if __name__ == "__main__":
    main()
