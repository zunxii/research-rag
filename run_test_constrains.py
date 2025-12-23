# run_test_constraints_v2.py
import json
import numpy as np
import torch
import faiss
from pathlib import Path
from PIL import Image

from core.embeddings.biomedclip import BioMedCLIPEncoder
from core.fusion.adaptive_fusion import AdaptiveFusion
from core.reasoning.constraints.extractor import ConstraintExtractor
from core.kb.image_loader import ImageLoader

KB_DIR = "kb_final_v2"
QUERY_TEXT = """Since my car wreck a little over a week ago where I suffered a head injury, my husband has noticed a really unpleasant smell coming from my mouth. I am 64 and quite a bit overweight. I still have a massive, fluid-filled bump on my head from the impact. I’ve been switching between ice packs and heat to try and get the swelling to go down, but I'm worried because this weird breath isn't normal for me. I've included a photo of the lump for your review"""
QUERY_IMAGE = "data/images/edema_Image_2.jpg"
TOP_K = 10

def main():
    print("🚀 Loading KB...")
    index = faiss.read_index(f"{KB_DIR}/index.faiss")
    metadata = json.load(open(f"{KB_DIR}/metadata.json"))

    print("🚀 Loading encoders + trained fusion...")
    encoder = BioMedCLIPEncoder(
        device="cpu",
        lora_path="trained_lora"
    )

    fusion = AdaptiveFusion()
    fusion.load_state_dict(torch.load("trained_fusion/fusion.pt"))
    fusion.eval()

    image_loader = ImageLoader()

    print("\n🧠 Encoding query...")
    with torch.no_grad():
        txt_emb = encoder.encode_text(QUERY_TEXT).unsqueeze(0)
        img = image_loader.load(QUERY_IMAGE)
        img_emb = encoder.encode_image(img).unsqueeze(0)

        fused = fusion(img_emb, txt_emb)

    q = fused.cpu().numpy().astype("float32")
    faiss.normalize_L2(q)

    print("\n🔍 Retrieving...")
    scores, idxs = index.search(q, TOP_K)

    retrieved = []
    for s, i in zip(scores[0], idxs[0]):
        m = metadata[i]
        retrieved.append({
            "diagnosis_label": m["diagnosis_label"],
            "image_path": m["image_path"],
            "case_id": m["case_id"],
            "score": float(s)
        })

    print("\n📐 Extracting constraints...")
    extractor = ConstraintExtractor()

    constraints = extractor.extract(
        retrieved_metadata=retrieved,
        img_emb=img_emb.squeeze(0).cpu().numpy(),
        txt_emb=txt_emb.squeeze(0).cpu().numpy(),
        centroid_distances={},   # optional for now
        query_distance=float(scores[0][0]),
        percentile_95=float(scores[0][0])
    )

    print(json.dumps(constraints, indent=2))


if __name__ == "__main__":
    main()
