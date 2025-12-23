import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    LoRA adapter for linear layers.
    Freezes base weight, learns low-rank update.
    """

    def __init__(self, base_layer: nn.Linear, rank: int = 8):
        super().__init__()
        self.base = base_layer
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        self.lora_A = nn.Linear(base_layer.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base_layer.out_features, bias=False)

        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x))
