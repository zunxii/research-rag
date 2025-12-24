"""Configuration for KB building."""

KB_BUILD_CONFIG = {
    "csv_path": "data/raw/clipsyntel.csv",
    "image_root": "data/images",
    "output_dir": "outputs/kb/kb_final_v2",
    "device": "cpu",
    "lora_path": "outputs/models/trained_lora",
    "fusion_path": "outputs/models/trained_fusion/fusion.pt",
}

KB_SMOKE_CONFIG = {
    "csv_path": "data/raw/test.csv",
    "image_root": "data/images",
    "output_dir": "outputs/kb/kb_smoke",
    "device": "cpu",
}