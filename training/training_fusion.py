import os
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image

from core.embeddings.biomedclip import BioMedCLIPEncoder
from core.fusion.adaptive_fusion import AdaptiveFusion

# ---------------- CONFIG ----------------
CSV_PATH = "clipsyntel.csv"
IMAGE_ROOT = "data/images"
OUTPUT_PATH = "trained_fusion/fusion.pt"

DEVICE = "cpu"
BATCH_SIZE = 2          # CPU-safe
EPOCHS = 3
LR = 1e-4
TEMPERATURE = 0.07

os.makedirs("trained_fusion", exist_ok=True)

# ---------------- DATASET ----------------
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
        # IMPORTANT: return raw objects, collate_fn will handle them
        return self.samples[idx]

# ---------------- CUSTOM COLLATE (CRITICAL FIX) ----------------
def fusion_collate_fn(batch):
    """
    batch: List[(Path, str)]
    returns: (List[Path], List[str])
    """
    image_paths, texts = zip(*batch)
    return list(image_paths), list(texts)

# ---------------- LOSS ----------------
def contrastive_loss(a, b, temperature):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)

    logits = a @ b.T / temperature
    labels = torch.arange(len(a), device=a.device)

    loss_ab = F.cross_entropy(logits, labels)
    loss_ba = F.cross_entropy(logits.T, labels)

    return (loss_ab + loss_ba) / 2

# ---------------- TRAIN ----------------
def main():
    print("Loading frozen encoders...")
    encoder = BioMedCLIPEncoder(device=DEVICE)

    # Freeze encoder completely
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
        collate_fn=fusion_collate_fn,  # ✅ FIX
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
            # Load images safely
            images = [
                Image.open(p).convert("RGB")
                for p in image_paths
            ]

            with torch.no_grad():
                img_emb = encoder.encode_image_batch(images)
                txt_emb = encoder.encode_text_batch(texts)

            # 🔑 Single fusion (correct)
            fused = fusion(img_emb, txt_emb)

            # 🔑 Align fused → text space
            loss = contrastive_loss(fused, txt_emb, TEMPERATURE)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss / len(loader):.4f}")

    torch.save(fusion.state_dict(), OUTPUT_PATH)
    print(f"\n✅ Fusion model saved to → {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
