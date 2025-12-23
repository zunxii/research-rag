from PIL import Image
from pathlib import Path


class ImageLoader:
    def __init__(self, image_root: str = "data/images"):
        self.image_root = Path(image_root)

    def load(self, image_path: str):
        """
        image_path may be:
        - filename only (e.g. xxx.jpg)
        - relative path
        - absolute path
        """

        p = Path(image_path)

        # Case 1: absolute or relative path exists
        if p.exists():
            final_path = p

        # Case 2: filename only → prepend image_root
        else:
            final_path = self.image_root / p
            if not final_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

        return Image.open(final_path).convert("RGB")
