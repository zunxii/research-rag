from PIL import Image
import torch
from core.embeddings.biomedclip import BioMedCLIPEncoder

def main():
    encoder = BioMedCLIPEncoder(device="cpu")

    img = Image.open("data/images/edema_Image_1.jpg").convert("RGB")
    txt = "Swelling around the ankle with pitting"

    img_emb = encoder.encode_image(img)
    txt_emb = encoder.encode_text(txt)

    print("Image embedding shape:", img_emb.shape)
    print("Text embedding shape:", txt_emb.shape)

    # Norm check
    print("Image norm:", torch.norm(img_emb).item())
    print("Text norm:", torch.norm(txt_emb).item())

    # Cosine similarity sanity
    sim = torch.dot(img_emb, txt_emb).item()
    print("Image–Text cosine similarity:", sim)

    assert img_emb.shape == (512,)
    assert txt_emb.shape == (512,)
    assert abs(torch.norm(img_emb).item() - 1.0) < 1e-3
    assert abs(torch.norm(txt_emb).item() - 1.0) < 1e-3

    print(" Encoder sanity check PASSED")

if __name__ == "__main__":
    main()
