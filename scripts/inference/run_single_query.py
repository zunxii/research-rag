"""
Single query inference - EXACT original logic
"""

import json
import numpy as np
from pathlib import Path
import faiss
import torch
import sys
import argparse

# Fix imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.embeddings.biomedclip import BioMedCLIPEncoder
from core.fusion.adaptive_fusion import AdaptiveFusion
from core.kb.image_loader import ImageLoader
from configs.inference_config import INFERENCE_CONFIG


# -------------------------
# Load KB (original)
# -------------------------
def load_kb(kb_dir: str):
    kb_dir = Path(kb_dir)

    embeddings = np.load(kb_dir / "embeddings.npy")
    index = faiss.read_index(str(kb_dir / "index.faiss"))

    with open(kb_dir / "metadata.json") as f:
        metadata = json.load(f)

    assert index.ntotal == len(metadata)
    return index, metadata


# -------------------------
# Main (original)
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-text", required=True)
    parser.add_argument("--query-image", default=None)
    parser.add_argument("--kb-dir", default=INFERENCE_CONFIG["kb_dir"])
    parser.add_argument("--top-k", type=int, default=INFERENCE_CONFIG["top_k"])
    args = parser.parse_args()

    print("Loading KB...")
    index, metadata = load_kb(args.kb_dir)

    print("Loading encoder (LoRA) + trained fusion...")
    encoder = BioMedCLIPEncoder(
        device=INFERENCE_CONFIG["device"],
        lora_path=INFERENCE_CONFIG.get("lora_path")
    )

    fusion = AdaptiveFusion().to(INFERENCE_CONFIG["device"])
    fusion.load_state_dict(
        torch.load(
            INFERENCE_CONFIG["fusion_path"],
            map_location=INFERENCE_CONFIG["device"]
        )
    )
    fusion.eval()

    image_loader = ImageLoader()

    # -------------------------
    # Encode query (original)
    # -------------------------
    print("\nEncoding query...")

    with torch.no_grad():
        txt_emb = encoder.encode_text(args.query_text).unsqueeze(0)

        if args.query_image is not None:
            img = image_loader.load(args.query_image)
            img_emb = encoder.encode_image(img).unsqueeze(0)
        else:
            # text-only query → zero image vector
            img_emb = torch.zeros_like(txt_emb)

        query_emb = fusion(img_emb, txt_emb)

    query_np = query_emb.cpu().numpy().astype("float32")
    faiss.normalize_L2(query_np)

    # -------------------------
    # Search (original)
    # -------------------------
    scores, indices = index.search(query_np, args.top_k)

    print("\n=== RETRIEVED CASES ===\n")

    for rank, (idx, score) in enumerate(
        zip(indices[0], scores[0]), start=1
    ):
        case = metadata[idx]

        print(f"Rank {rank}")
        print(f"  Score: {score:.4f}")
        print(f"  Diagnosis: {case['diagnosis_label']}")
        print(f"  Image: {case['image_path']}")
        print(f"  Text: {case['clinical_text']['combined'][:160]}")
        print("-" * 60)


if __name__ == "__main__":
    main()