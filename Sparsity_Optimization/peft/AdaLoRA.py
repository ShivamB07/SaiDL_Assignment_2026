import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AdaLoRALinear(nn.Module):
    def __init__(self, original_layer, init_rank=12, alpha=16):
        super().__init__()
        self.original = original_layer
        
        self.scaling = alpha / init_rank
        
        self.lora_P = nn.Parameter(torch.randn(original_layer.in_features, init_rank))
        self.lora_Q = nn.Parameter(torch.randn(init_rank, original_layer.out_features))
        self.lora_E = nn.Parameter(torch.ones(init_rank)) # Diagonal Lambda
        
        nn.init.orthogonal_(self.lora_P)
        nn.init.orthogonal_(self.lora_Q)

    def forward(self, x):
        # Delta W = P * diag(E) * Q
        lora_P = self.lora_P.to(dtype=x.dtype, device=x.device)
        lora_Q = self.lora_Q.to(dtype=x.dtype, device=x.device)
        lora_E = self.lora_E.to(dtype=x.dtype, device=x.device)
        adapt = (lora_P * lora_E) @ lora_Q
        return self.original(x) + (x @ adapt) * self.scaling
        
    def get_orthogonality_penalty(self):
        # Required for the loss function to keep P and Q orthogonal
        I = torch.eye(self.lora_P.shape[1], device=self.lora_P.device)
        penalty_P = torch.norm(self.lora_P.T @ self.lora_P - I, p='fro')
        penalty_Q = torch.norm(self.lora_Q @ self.lora_Q.T - I, p='fro')
        return penalty_P + penalty_Q


