"""
LoRA training script - FIXED VERSION
Critical fixes:
1. Batch size increased to 16 (minimum for contrastive loss)
2. Target modules dynamically extracted from actual model architecture
3. More training epochs (10 instead of 1)
"""

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

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from configs.training_config import LORA_CONFIG

# -------------------------
# FIXED CONFIG
# -------------------------
CSV_PATH = LORA_CONFIG["csv_path"]
IMAGE_ROOT = LORA_CONFIG["image_root"]
OUTPUT_DIR = LORA_CONFIG["output_dir"]

DEVICE = LORA_CONFIG["device"]

# FIX #2: Increase batch size for proper contrastive loss
BATCH_SIZE = 16  # CHANGED FROM 1 (critical fix)
ACCUM_STEPS = 2  # CHANGED FROM 8 (adjust for larger batch)

# FIX #4: More epochs
EPOCHS = 10  # CHANGED FROM 1

LR = LORA_CONFIG["lr"]
TEMPERATURE = LORA_CONFIG["temperature"]

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
    """
    CLIP-style contrastive loss.
    
    IMPORTANT: This requires batch_size > 1 to work properly!
    With batch_size=1, the logits matrix is 1x1 which is degenerate.
    """
    image_emb = F.normalize(image_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)

    # This creates a batch_size x batch_size matrix
    logits = image_emb @ text_emb.T / temperature
    labels = torch.arange(len(image_emb), device=image_emb.device)

    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)

    return (loss_i + loss_t) / 2


# -------------------------
# Model Inspection
# -------------------------
def inspect_model_layers(model):
    """Helper to inspect what layers we can target."""
    print("\n=== Inspecting Model Architecture ===")
    attention_layers = []
    
    for name, module in model.named_modules():
        if any(x in name.lower() for x in ['attn', 'attention']):
            print(f"Found: {name} -> {type(module).__name__}")
            attention_layers.append(name)
    
    return attention_layers


# -------------------------
# FIX #3: Dynamically extract target modules
# -------------------------
def get_target_modules_for_openclip(model):
    """
    FIX #3: Dynamically extract correct target modules from the actual model.
    
    Based on BioMedCLIP architecture:
    - Visual: visual.trunk.blocks.*.attn.qkv and attn.proj
    - Text: text.transformer.encoder.layer.*.attention.{self.query, self.key, self.value, output.dense}
    """
    target_modules = []
    
    for name, module in model.named_modules():
        # Visual attention layers (QKV and projection)
        if 'visual.trunk.blocks' in name and name.endswith(('.attn.qkv', '.attn.proj')):
            target_modules.append(name)
        # Text attention layers (Q, K, V, and output)
        elif 'text.transformer.encoder.layer' in name and any(
            name.endswith(suffix) for suffix in [
                '.attention.self.query',
                '.attention.self.key', 
                '.attention.self.value',
                '.attention.output.dense'
            ]
        ):
            target_modules.append(name)
    
    return target_modules


# -------------------------
# Main Training
# -------------------------
def main():
    print("="*70)
    print("FIXED LoRA TRAINING")
    print("="*70)
    print(f"Batch size: {BATCH_SIZE} (FIXED from 1)")
    print(f"Epochs: {EPOCHS} (FIXED from 1)")
    print(f"Gradient accumulation: {ACCUM_STEPS}")
    print()

    print("Loading BioMedCLIP...")
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    model.to(DEVICE)
    model.train()

    # Inspect model to verify layer names
    attention_layers = inspect_model_layers(model)
    
    if not attention_layers:
        print("WARNING: No attention layers found. Check model architecture.")

    # -------------------------
    # FIX #3: Dynamically extract correct target modules
    # -------------------------
    print("\nExtracting target modules from model architecture...")
    
    target_modules = get_target_modules_for_openclip(model)
    
    print(f"\nFound {len(target_modules)} attention layers to adapt")
    print(f"Sample modules:")
    for i, mod in enumerate(target_modules[:5]):
        print(f"  {i+1}. {mod}")
    print(f"  ... and {len(target_modules) - 5} more")
    
    if not target_modules:
        raise ValueError(
            "Could not find any target modules! "
            "Check the get_target_modules_for_openclip() function."
        )
    
    # -------------------------
    # Apply LoRA with correct modules
    # -------------------------
    print("\nInjecting LoRA adapters...")
    
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=target_modules,  # FIXED - using actual layer names
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
        batch_size=BATCH_SIZE,  # FIXED
        shuffle=True,
        num_workers=0
    )

    print(f"\nTraining samples: {len(dataset)}")
    print(f"Batches per epoch: {len(loader)}")
    print(f"Effective batch size: {BATCH_SIZE * ACCUM_STEPS}")

    # -------------------------
    # Training loop with validation
    # -------------------------
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for step, (images, texts) in enumerate(pbar):
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

            epoch_loss += loss.item() * ACCUM_STEPS
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item() * ACCUM_STEPS:.4f}'})

        avg_loss = epoch_loss / len(loader)
        print(f"\nEpoch {epoch+1}/{EPOCHS} | Avg Loss: {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"New best loss: {best_loss:.4f} - Saving checkpoint...")
            model.save_pretrained(OUTPUT_DIR)

    # -------------------------
    # Save final model
    # -------------------------
    print("\n" + "="*70)
    print("Saving final LoRA adapters...")
    model.save_pretrained(OUTPUT_DIR)
    
    # Save training info
    training_info = {
        "final_loss": avg_loss,
        "best_loss": best_loss,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "num_target_modules": len(target_modules),
        "sample_target_modules": target_modules[:10]
    }
    
    import json
    with open(Path(OUTPUT_DIR) / "training_info.json", 'w') as f:
        json.dump(training_info, f, indent=2)

    print(f"✓ Training complete!")
    print(f"✓ Models saved to: {OUTPUT_DIR}")
    print(f"✓ Final loss: {avg_loss:.4f}")
    print(f"✓ Best loss: {best_loss:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()