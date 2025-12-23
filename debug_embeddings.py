import numpy as np
from core.embeddings.biomedclip import BioMedCLIPEncoder
from core.embeddings.fusion import GatedFusion
from core.kb.image_loader import ImageLoader

encoder = BioMedCLIPEncoder(device="cpu")
fusion = GatedFusion(dim=encoder.dim)
loader = ImageLoader(image_root="data/images")

img1 = loader.load("cyanosis_Image_1.jpg")
img2 = loader.load("edema_Image_1.jpg")

e1 = encoder.encode_image(img1).numpy()
e2 = encoder.encode_image(img2).numpy()

print("Image cosine similarity:",
      np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
