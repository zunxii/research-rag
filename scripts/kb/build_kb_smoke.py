"""
Smoke test KB builder - EXACT original logic
Uses test.csv for quick validation
"""

import sys
from pathlib import Path

# Fix imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.kb.builder import KBBuilder
from core.embeddings.biomedclip import BioMedCLIPEncoder
from core.embeddings.fusion import GatedFusion
from configs.kb_config import KB_SMOKE_CONFIG


def main():
    builder = KBBuilder(
        image_encoder=BioMedCLIPEncoder(device=KB_SMOKE_CONFIG["device"]),
        text_encoder=BioMedCLIPEncoder(device=KB_SMOKE_CONFIG["device"]),
        fusion_model=GatedFusion(dim=512),
        output_dir=KB_SMOKE_CONFIG["output_dir"],
        image_root=KB_SMOKE_CONFIG["image_root"],
        device=KB_SMOKE_CONFIG["device"],
    )

    builder.build(KB_SMOKE_CONFIG["csv_path"])


if __name__ == "__main__":
    main()
