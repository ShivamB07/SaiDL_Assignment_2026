import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, original_layer, rank=8, alpha=16):
        super().__init__()
        self.original = original_layer
        
        self.rank = rank
        self.scaling = alpha / rank
        
        # A: (in_features x rank), B: (rank x out_features)
        self.lora_A = nn.Parameter(torch.rand(original_layer.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, original_layer.out_features))
        
        nn.init.normal_(self.lora_A, std=1 / self.rank)
        nn.init.zeros_(self.lora_B) # Zero init ensures identity at start

    def forward(self, x):
        base_out = self.original(x)
        lora_A = self.lora_A.to(dtype=x.dtype, device=x.device)
        lora_B = self.lora_B.to(dtype=x.dtype, device=x.device)
        lora_out = (x @ lora_A @ lora_B) * self.scaling
        return base_out + lora_out