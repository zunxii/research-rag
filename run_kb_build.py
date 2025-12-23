import torch

from core.kb.builder import KBBuilder
from core.embeddings.biomedclip import BioMedCLIPEncoder
from core.fusion.adaptive_fusion import AdaptiveFusion


# --------------------------------------------------
# Load trained fusion
# --------------------------------------------------
fusion = AdaptiveFusion()
fusion.load_state_dict(
    torch.load("trained_fusion/fusion.pt", map_location="cpu")
)
fusion.eval()  # CRITICAL: deterministic embeddings


# --------------------------------------------------
# Load encoders WITH LoRA
# --------------------------------------------------
image_encoder = BioMedCLIPEncoder(
    device="cpu",
    lora_path="trained_lora"
)

text_encoder = BioMedCLIPEncoder(
    device="cpu",
    lora_path="trained_lora"
)


# --------------------------------------------------
# Build KB
# --------------------------------------------------
builder = KBBuilder(
    image_encoder=image_encoder,
    text_encoder=text_encoder,
    fusion_model=fusion,
    output_dir="kb_final_v2",     # NEW KB (do not overwrite old)
    image_root="data/images",
    device="cpu",
)

builder.build("clipsyntel.csv")
