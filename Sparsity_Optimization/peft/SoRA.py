import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SoRALinear(nn.Module):
    def __init__(self, original_layer, init_rank=12, alpha=16):
        super().__init__()
        self.original = original_layer
        
        self.scaling = alpha / init_rank
        
        self.lora_A = nn.Parameter(torch.randn(original_layer.in_features, init_rank))
        self.lora_B = nn.Parameter(torch.zeros(init_rank, original_layer.out_features))
        self.gate = nn.Parameter(torch.ones(init_rank)) # Sparse Gate
        
        nn.init.normal_(self.lora_A, std=1 / init_rank)

    def forward(self, x):
        lora_A = self.lora_A.to(dtype=x.dtype, device=x.device)
        lora_B = self.lora_B.to(dtype=x.dtype, device=x.device)
        gate = self.gate.to(dtype=x.dtype, device=x.device)
        adapt = (lora_A * gate) @ lora_B
        return self.original(x) + (x @ adapt) * self.scaling
        
    def proximal_step(self, threshold):
        with torch.no_grad():
            self.gate.copy_(torch.sign(self.gate) * torch.clamp(torch.abs(self.gate) - threshold, min=0.0))