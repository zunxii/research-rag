"""
Inspect OpenCLIP architecture to find correct target modules
Run this ONCE before training
"""

import open_clip

MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

print("Loading BioMedCLIP to inspect architecture...\n")
model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME)

print("="*70)
print("VISUAL ENCODER (Image) Modules:")
print("="*70)
for name, module in model.visual.named_modules():
    if any(x in name for x in ['attn', 'mlp', 'proj']):
        print(f"{name:60s} {type(module).__name__}")

print("\n" + "="*70)
print("TEXT ENCODER Modules:")
print("="*70)
for name, module in model.transformer.named_modules():
    if any(x in name for x in ['attn', 'mlp', 'proj']):
        print(f"{name:60s} {type(module).__name__}")

print("\n" + "="*70)
print("Recommended target_modules for LoRA:")
print("="*70)

# Find all linear projection layers in attention
targets = set()
for name, module in model.named_modules():
    if 'attn' in name and any(x in name for x in ['in_proj', 'out_proj', 'q_proj', 'k_proj', 'v_proj']):
        # Extract the pattern
        parts = name.split('.')
        pattern = '.'.join(parts[:-1]) + '.*.' + parts[-1]
        targets.add(parts[-1])  # Just the layer name

print("Target these modules:", list(targets))