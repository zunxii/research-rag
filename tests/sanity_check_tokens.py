from PIL import Image
from core.embeddings.biomedclip import BioMedCLIPEncoder
import torch.nn.functional as F

encoder = BioMedCLIPEncoder()

img = Image.open("data/images/edema_Image_1.jpg").convert("RGB")
text = "bluish discoloration of distal fingertips"

img_emb = encoder.encode_image(img)
txt_emb = encoder.encode_text(text)

print(img_emb.shape)   # (512,)
print(txt_emb.shape)   # (512,)
print("img norm:", img_emb.norm().item())
print("txt norm:", txt_emb.norm().item())
print("cosine:", F.cosine_similarity(img_emb, txt_emb, dim=0).item())
