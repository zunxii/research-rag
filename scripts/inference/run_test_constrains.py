import json
import torch
import faiss
import numpy as np
from PIL import Image

from core.embeddings.biomedclip import BioMedCLIPEncoder
from core.fusion.adaptive_fusion import AdaptiveFusion
from core.reasoning.constraints.extractor import ConstraintExtractor

# ---------------- CONFIG ----------------
KB_DIR = "kb_final_v2"
QUERY_TEXT = """Since my car wreck a little over a week ago where I suffered a head injury, my husband has noticed a really unpleasant smell coming from my mouth. I am 64 and quite a bit overweight. I still have a massive, fluid-filled bump on my head from the impact. I’ve been switching between ice packs and heat to try and get the swelling to go down, but I'm worried because this weird breath isn't normal for me. I've included a photo of the lump for your review."""
QUERY_IMAGE = "data/images/edema_Image_2.jpg"
TOP_K = 10
DEVICE = "cpu"

def main():
    print(" Loading KB...")
    index = faiss.read_index(f"{KB_DIR}/index.faiss")
    metadata = json.load(open(f"{KB_DIR}/metadata.json"))

    print(" Loading encoders + trained fusion...")
    encoder = BioMedCLIPEncoder(device=DEVICE, lora_path="trained_lora")

    fusion = AdaptiveFusion().to(DEVICE)
    fusion.load_state_dict(torch.load("trained_fusion/fusion.pt", map_location=DEVICE))
    fusion.eval()

    print("\n Encoding query...")
    txt_emb = encoder.encode_text(QUERY_TEXT).unsqueeze(0)
    img = Image.open(QUERY_IMAGE).convert("RGB")
    img_emb = encoder.encode_image(img).unsqueeze(0)

    with torch.no_grad():
        fused = fusion(img_emb, txt_emb)

    q = fused.cpu().numpy().astype("float32")
    faiss.normalize_L2(q)

    print("\n Retrieving...")
    scores, idxs = index.search(q, TOP_K)

    retrieved = []
    for s, i in zip(scores[0], idxs[0]):
        m = metadata[i]
        retrieved.append({
            "score": float(s),
            **m
        })

    print("\n Extracting constraints...")
    extractor = ConstraintExtractor()

    # dummy centroid distances (replace later with real centroids)
    centroid_distances = {
        m["diagnosis_label"]: float(np.random.rand())
        for m in retrieved
    }

    constraints = extractor.extract(
        retrieved_metadata=retrieved,
        img_emb=img_emb,
        txt_emb=txt_emb,
        centroid_distances=centroid_distances,
        query_distance=float(scores[0][0]),
        percentile_95=float(np.percentile(scores[0], 95)),
    )

    print("\n CONSTRAINT OUTPUT")
    print(json.dumps(constraints, indent=2))

if __name__ == "__main__":
    main()
