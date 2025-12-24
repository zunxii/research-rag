import torch

def remove_text(img_emb, txt_emb):
    return img_emb, torch.zeros_like(txt_emb)

def remove_image(img_emb, txt_emb):
    return torch.zeros_like(img_emb), txt_emb

def add_noise(emb, scale=0.05):
    return emb + scale * torch.randn_like(emb)
