import json
import torch
import faiss
import numpy as np
from PIL import Image

from core.embeddings.biomedclip import BioMedCLIPEncoder
from core.fusion.adaptive_fusion import AdaptiveFusion

from core.reasoning.counterfactuals.stability.retrieval import StabilityRetriever
from core.reasoning.counterfactuals.stability.runner import StabilityRunner
from core.reasoning.counterfactuals.diagnostics.scorer import CounterfactualScorer


# ---------------- CONFIG ----------------
KB_DIR = "kb_final_v2"
DEVICE = "cpu"

QUERY_TEXT = (
    "Since my car wreck a little over a week ago where I suffered a head injury, "
    "my husband has noticed a really unpleasant smell coming from my mouth. "
    "I am 64 and quite a bit overweight. I still have a massive, fluid-filled bump "
    "on my head from the impact."
)

QUERY_IMAGE_PATH = "data/images/edema_Image_2.jpg"


def main():
    print("  Loading KB...")
    index = faiss.read_index(f"{KB_DIR}/index.faiss")
    with open(f"{KB_DIR}/metadata.json") as f:
        metadata = json.load(f)

    print("  Loading encoders + trained fusion...")
    encoder = BioMedCLIPEncoder(
        device=DEVICE,
        lora_path="trained_lora"
    )

    fusion = AdaptiveFusion().to(DEVICE)
    fusion.load_state_dict(
        torch.load("trained_fusion/fusion.pt", map_location=DEVICE)
    )
    fusion.eval()

    retriever = StabilityRetriever(index, metadata)

    # ---------------- Encode query ----------------
    print("\ Encoding query...")

    img = Image.open(QUERY_IMAGE_PATH).convert("RGB")

    with torch.no_grad():
        img_emb = encoder.encode_image(img).unsqueeze(0)
        txt_emb = encoder.encode_text(QUERY_TEXT).unsqueeze(0)

    # ---------------- Level 1: Stability ----------------
    print("\n Running Level 1: Counterfactual Stability...")
    stability_runner = StabilityRunner(retriever, fusion)
    stability_output = stability_runner.run(img_emb, txt_emb)

    print("\n=== STABILITY OUTPUT ===")
    print(json.dumps(stability_output, indent=2))

    # ---------------- Level 2: Diagnostic Scoring ----------------
    print("\n Running Level 2: Diagnostic Scoring...")
    scorer = CounterfactualScorer()
    scored = scorer.score(stability_output)

    print("\n=== DIAGNOSTIC SCORES ===")
    for s in scored:
        print(s)


if __name__ == "__main__":
    main()
