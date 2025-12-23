import json
import numpy as np
from pathlib import Path
import faiss
import torch

from core.embeddings.biomedclip import BioMedCLIPEncoder
from core.fusion.adaptive_fusion import AdaptiveFusion
from core.kb.image_loader import ImageLoader


# -------------------------
# CONFIG
# -------------------------
KB_DIR = "kb_final_v2"          # MUST match new KB
DEVICE = "cpu"
TOP_K = 5

QUERY_TEXT = (
    "Since my car wreck a little over a week ago where I suffered a head injury, my husband has noticed a really unpleasant smell coming from my mouth. I am 64 and quite a bit overweight. I still have a massive, fluid-filled bump on my head from the impact. I’ve been switching between ice packs and heat to try and get the swelling to go down, but I'm worried because this weird breath isn't normal for me. I've included a photo of the lump for your review"
)

QUERY_IMAGE = "data/images/edema_Image_2.jpg"  # set None for text-only


# -------------------------
# Load KB
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
# Main
# -------------------------
def main():
    print("Loading KB...")
    index, metadata = load_kb(KB_DIR)

    print("Loading encoder (LoRA) + trained fusion...")
    encoder = BioMedCLIPEncoder(
        device=DEVICE,
        lora_path=None
    )

    fusion = AdaptiveFusion().to(DEVICE)
    fusion.load_state_dict(
        torch.load("trained_fusion/fusion.pt", map_location=DEVICE)
    )
    fusion.eval()

    image_loader = ImageLoader()

    # -------------------------
    # Encode query
    # -------------------------
    print("\nEncoding query...")

    with torch.no_grad():
        txt_emb = encoder.encode_text(QUERY_TEXT).unsqueeze(0)

        if QUERY_IMAGE is not None:
            img = image_loader.load(QUERY_IMAGE)
            img_emb = encoder.encode_image(img).unsqueeze(0)
        else:
            # text-only query → zero image vector
            img_emb = torch.zeros_like(txt_emb)

        query_emb = fusion(img_emb, txt_emb)

    query_np = query_emb.cpu().numpy().astype("float32")
    faiss.normalize_L2(query_np)

    # -------------------------
    # Search
    # -------------------------
    scores, indices = index.search(query_np, TOP_K)

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
