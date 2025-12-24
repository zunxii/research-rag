"""Configuration for training scripts."""

# LoRA Training Config
LORA_CONFIG = {
    "csv_path": "data/raw/clipsyntel.csv",
    "image_root": "data/images",
    "output_dir": "outputs/models/trained_lora",
    "device": "cpu",
    "batch_size": 1,
    "accum_steps": 8,
    "epochs": 1,
    "lr": 5e-5,
    "temperature": 0.07,
}

# Fusion Training Config
FUSION_CONFIG = {
    "csv_path": "data/raw/clipsyntel.csv",
    "image_root": "data/images",
    "output_path": "outputs/models/trained_fusion/fusion.pt",
    "device": "cpu",
    "batch_size": 2,
    "epochs": 3,
    "lr": 1e-4,
    "temperature": 0.07,
}