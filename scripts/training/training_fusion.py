"""
Fusion training script - EXACT original logic
Only imports fixed for proper package structure
"""

import os
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image

# Fixed imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.embeddings.biomedclip import BioMedCLIPEncoder
from core.fusion.adaptive_fusion import AdaptiveFusion
from configs.training_config import FUSION_CONFIG

# ---------------- CONFIG (original) ----------------
CSV_PATH = FUSION_CONFIG["csv_path"]
IMAGE_ROOT = FUSION_CONFIG["image_root"]
OUTPUT_PATH = FUSION_CONFIG["output_path"]

DEVICE = FUSION_CONFIG["device"]
BATCH_SIZE = FUSION_CONFIG["batch_size"]
EPOCHS = FUSION_CONFIG["epochs"]
LR = FUSION_CONFIG["lr"]
TEMPERATURE = FUSION_CONFIG["temperature"]

os.makedirs(Path(OUTPUT_PATH).parent, exist_ok=True)

# ---------------- DATASET (original) ----------------
class ClipSyntelFusionDataset(Dataset):
    def __init__(self, csv_path, image_root):
        self.samples = []
        self.image_root = Path(image_root)

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row.get("image_path", "").strip()
                text = f"{row.get('context','')} {row.get('description','')}".strip()

                if not img_name or len(text) < 10:
                    continue

                img_path = self.image_root / img_name
                if img_path.exists():
                    self.samples.append((img_path, text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# ---------------- CUSTOM COLLATE (original) ----------------
def fusion_collate_fn(batch):
    image_paths, texts = zip(*batch)
    return list(image_paths), list(texts)

# ---------------- LOSS (original) ----------------
def contrastive_loss(a, b, temperature):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)

    logits = a @ b.T / temperature
    labels = torch.arange(len(a), device=a.device)

    loss_ab = F.cross_entropy(logits, labels)
    loss_ba = F.cross_entropy(logits.T, labels)

    return (loss_ab + loss_ba) / 2

# ---------------- TRAIN (original) ----------------
def main():
    print("Loading frozen encoders...")
    encoder = BioMedCLIPEncoder(device=DEVICE)

    # Freeze encoder completely (original)
    encoder.model.eval()
    for p in encoder.model.parameters():
        p.requires_grad = False

    print("Initializing fusion...")
    fusion = AdaptiveFusion().to(DEVICE)

    optimizer = torch.optim.AdamW(fusion.parameters(), lr=LR)

    dataset = ClipSyntelFusionDataset(CSV_PATH, IMAGE_ROOT)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=fusion_collate_fn,
        num_workers=0
    )

    print(f"Training samples: {len(dataset)}")

    for epoch in range(EPOCHS):
        fusion.train()
        total_loss = 0.0

        for image_paths, texts in tqdm(
            loader,
            desc=f"Epoch {epoch+1}/{EPOCHS}"
        ):
            # Load images safely (original)
            images = [
                Image.open(p).convert("RGB")
                for p in image_paths
            ]

            with torch.no_grad():
                img_emb = encoder.encode_image_batch(images)
                txt_emb = encoder.encode_text_batch(texts)

            # Single fusion (original)
            fused = fusion(img_emb, txt_emb)

            # Align fused → text space (original)
            loss = contrastive_loss(fused, txt_emb, TEMPERATURE)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss / len(loader):.4f}")

    torch.save(fusion.state_dict(), OUTPUT_PATH)
    print(f"\n✓ Fusion model saved to → {OUTPUT_PATH}")

if __name__ == "__main__":
    main()