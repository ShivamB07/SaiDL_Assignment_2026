import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(self, n_embed, head_size, dropout_prob, block_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False) 
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) 
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed, num_heads, head_size, dropout_prob, block_size):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(
                n_embed,
                head_size,
                dropout_prob,
                block_size
            )
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x, attn):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
