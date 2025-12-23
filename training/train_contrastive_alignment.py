import os
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from peft import LoraConfig, get_peft_model
import open_clip
from PIL import Image

# -------------------------
# CPU SAFETY
# -------------------------
torch.set_num_threads(2)

# -------------------------
# Config
# -------------------------
CSV_PATH = "clipsyntel.csv"
IMAGE_ROOT = "data/images"
OUTPUT_DIR = "trained_lora"

DEVICE = "cpu"            # CPU only
BATCH_SIZE = 1            # MUST be 1 on CPU
ACCUM_STEPS = 8           # gradient accumulation
EPOCHS = 1                # start with 1
LR = 5e-5
TEMPERATURE = 0.07

MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Dataset
# -------------------------
class ClipSyntelDataset(Dataset):
    def __init__(self, csv_path: str, image_root: str, preprocess):
        self.samples = []
        self.image_root = Path(image_root)
        self.preprocess = preprocess

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row.get("image_path", "").strip()
                text = f"{row.get('context','')} {row.get('description','')}".strip()

                if not img_name or len(text) < 5:
                    continue

                img_path = self.image_root / img_name
                if not img_path.exists():
                    continue

                self.samples.append((img_path, text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, text = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.preprocess(image)
        return image, text


# -------------------------
# CLIP Contrastive Loss
# -------------------------
def clip_contrastive_loss(image_emb, text_emb, temperature):
    image_emb = F.normalize(image_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)

    logits = image_emb @ text_emb.T / temperature
    labels = torch.arange(len(image_emb), device=image_emb.device)

    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)

    return (loss_i + loss_t) / 2


# -------------------------
# Main Training
# -------------------------
def main():
    print("Loading BioMedCLIP...")

    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    model.to(DEVICE)
    model.train()

    # -------------------------
    # Inject LoRA (OFFICIAL, ROBUST)
    # -------------------------
    print("Injecting LoRA adapters (official all-linear mode)...")

    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",     # 🔑 CRITICAL FIX
        task_type="FEATURE_EXTRACTION"
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # -------------------------
    # Data
    # -------------------------
    dataset = ClipSyntelDataset(CSV_PATH, IMAGE_ROOT, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    print(f"Training samples: {len(dataset)}")

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, (images, texts) in enumerate(
            tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        ):
            images = images.to(DEVICE)
            tokens = tokenizer(list(texts)).to(DEVICE)

            image_emb = model.encode_image(images)
            text_emb = model.encode_text(tokens)

            loss = clip_contrastive_loss(
                image_emb,
                text_emb,
                TEMPERATURE
            )

            loss = loss / ACCUM_STEPS
            loss.backward()

            if (step + 1) % ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()

        avg = epoch_loss / len(loader)
        print(f"Epoch {epoch+1} | Avg Loss: {avg:.4f}")

    # -------------------------
    # Save LoRA adapters
    # -------------------------
    print("Saving LoRA adapters...")
    model.save_pretrained(OUTPUT_DIR)

    print("Training complete.")


if __name__ == "__main__":
    main()
