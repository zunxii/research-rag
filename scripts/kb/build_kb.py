"""
Main KB builder - EXACT original logic
Builds kb_final_v2 with trained models
"""

import torch
import sys
from pathlib import Path

# Fix imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.kb.builder import KBBuilder
from core.embeddings.biomedclip import BioMedCLIPEncoder
from core.fusion.adaptive_fusion import AdaptiveFusion
from configs.kb_config import KB_BUILD_CONFIG


def main():
    # --------------------------------------------------
    # Load trained fusion (original)
    # --------------------------------------------------
    fusion = AdaptiveFusion()
    fusion.load_state_dict(
        torch.load(
            KB_BUILD_CONFIG["fusion_path"],
            map_location=KB_BUILD_CONFIG["device"]
        )
    )
    fusion.eval()  # CRITICAL: deterministic embeddings


    # --------------------------------------------------
    # Load encoders WITH LoRA (original)
    # --------------------------------------------------
    image_encoder = BioMedCLIPEncoder(
        device=KB_BUILD_CONFIG["device"],
        lora_path=KB_BUILD_CONFIG["lora_path"]
    )

    text_encoder = BioMedCLIPEncoder(
        device=KB_BUILD_CONFIG["device"],
        lora_path=KB_BUILD_CONFIG["lora_path"]
    )


    # --------------------------------------------------
    # Build KB (original)
    # --------------------------------------------------
    builder = KBBuilder(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        fusion_model=fusion,
        output_dir=KB_BUILD_CONFIG["output_dir"],
        image_root=KB_BUILD_CONFIG["image_root"],
        device=KB_BUILD_CONFIG["device"],
    )

    builder.build(KB_BUILD_CONFIG["csv_path"])


if __name__ == "__main__":
    main()
