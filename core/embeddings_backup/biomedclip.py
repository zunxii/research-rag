import torch
import torch.nn.functional as F
import open_clip
from typing import List
from PIL import Image
from pathlib import Path


class BioMedCLIPEncoder:
    """
    Production-grade BioMedCLIP encoder (OpenCLIP-based).
    
    FIXED: Proper LoRA loading for OpenCLIP models
    """

    def __init__(
        self,
        device: str = "cpu",
        model_name: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        lora_path: str | None = None,
    ):
        self.device = device

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # FIX: Proper LoRA loading for OpenCLIP
        if lora_path is not None:
            self._load_lora_adapters(lora_path)

        self.model.to(self.device)
        self.model.eval()

        self.dim = 512  # fixed by BioMedCLIP architecture

    def _load_lora_adapters(self, lora_path: str):
        """
        FIXED: Properly load LoRA adapters for OpenCLIP models.
        
        The original code tried to use PeftModel.from_pretrained() which only
        works for HuggingFace models. We need to:
        1. Re-create the PEFT model structure
        2. Load the saved adapter weights
        """
        from peft import LoraConfig, get_peft_model
        
        lora_path = Path(lora_path)
        
        # Load the LoRA config that was saved during training
        config_path = lora_path / "adapter_config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"LoRA config not found at {config_path}. "
                f"Make sure you trained LoRA first with train_lora.py"
            )
        
        # Load config
        lora_config = LoraConfig.from_pretrained(str(lora_path))
        
        # Apply PEFT to recreate the same structure as training
        self.model = get_peft_model(self.model, lora_config)
        
        # Load the actual adapter weights
        adapter_path = lora_path / "adapter_model.bin"
        if adapter_path.exists():
            adapter_weights = torch.load(adapter_path, map_location=self.device)
            # Load weights (strict=False allows loading only LoRA params)
            self.model.load_state_dict(adapter_weights, strict=False)
        else:
            # Try safetensors format
            try:
                from safetensors.torch import load_file
                adapter_path = lora_path / "adapter_model.safetensors"
                adapter_weights = load_file(str(adapter_path))
                self.model.load_state_dict(adapter_weights, strict=False)
            except:
                raise FileNotFoundError(
                    f"Could not find adapter weights in {lora_path}"
                )

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------
    @staticmethod
    def _l2norm(x: torch.Tensor):
        return F.normalize(x, dim=-1)

    # --------------------------------------------------
    # IMAGE EMBEDDINGS
    # --------------------------------------------------
    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """
        Returns:
            Tensor of shape (512,)
        """
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        emb = self.model.encode_image(image_tensor)
        return self._l2norm(emb).squeeze(0)

    @torch.no_grad()
    def encode_image_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Returns:
            Tensor of shape (B, 512)
        """
        image_tensors = torch.stack(
            [self.preprocess(img) for img in images]
        ).to(self.device)

        emb = self.model.encode_image(image_tensors)
        return self._l2norm(emb)

    # --------------------------------------------------
    # TEXT EMBEDDINGS
    # --------------------------------------------------
    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        """
        Returns:
            Tensor of shape (512,)
        """
        tokens = self.tokenizer([text]).to(self.device)
        emb = self.model.encode_text(tokens)
        return self._l2norm(emb).squeeze(0)

    @torch.no_grad()
    def encode_text_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Returns:
            Tensor of shape (B, 512)
        """
        tokens = self.tokenizer(texts).to(self.device)
        emb = self.model.encode_text(tokens)
        return self._l2norm(emb)