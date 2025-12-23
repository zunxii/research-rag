from core.kb.builder import KBBuilder
from core.embeddings.biomedclip import BioMedCLIPEncoder
from core.embeddings.fusion import GatedFusion

builder = KBBuilder(
    image_encoder=BioMedCLIPEncoder(device="cpu"),
    text_encoder=BioMedCLIPEncoder(device="cpu"),  # same encoder, text side used
    fusion_model=GatedFusion(dim=512),
    output_dir="kb_smoke",
    device="cpu",
)

builder.build("test.csv")
