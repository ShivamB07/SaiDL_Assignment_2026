import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention.MHA import MultiHeadAttention
from attention.sliding_window import SlidingwindowAttention
from attention.MQA import MultiQueryAttention
from attention.GQA import GroupedQueryAttention


class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout_prob):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(p=dropout_prob),
        )

    def forward(self, x):
        return self.network(x)


class Block(nn.Module):
    def __init__(self, head_size, num_head, n_embed, dropout_prob, block_size, 
                    attention_type: str = "mha", sliding_window_mask: bool = False, 
                    window_size: int = 512, positional_encoding: str = "learnable", group_size: int = 2,
                    use_local_conv: bool = False, ngram_size: int = 3, base = 10000):
        super().__init__()
        self.attention_type = attention_type
        self.sliding_window_mask = sliding_window_mask
        self.block_size = block_size
        self.window_size = window_size
        self.base = base
        self.positional_encoding = positional_encoding
        self.group_size = group_size
        self.use_local_conv = use_local_conv
        self.ngram_size = ngram_size
        
        # select attention implementation
        if attention_type == "mha":
            self.attn = MultiHeadAttention(
                n_embed=n_embed,
                num_heads=num_head,
                block_size=block_size,
                dropout_prob=dropout_prob,
                positional_encoding=self.positional_encoding,
                use_local_conv=self.use_local_conv,
                ngram_size=self.ngram_size,
                base = self.base
            )
        elif attention_type == "mqa":
            self.attn = MultiQueryAttention(
                n_embed=n_embed,
                num_heads=num_head,
                head_size=head_size,
                block_size=block_size,
                positional_encoding=self.positional_encoding,
                use_local_conv=self.use_local_conv,
                ngram_size=self.ngram_size,
                base = self.base
            )
        elif attention_type == "gqa":
            self.attn = GroupedQueryAttention(
                n_embed=n_embed,
                num_heads=num_head,
                head_size=head_size,
                block_size=block_size,
                group_size=self.group_size,
                positional_encoding=self.positional_encoding,
                use_local_conv=self.use_local_conv,
                ngram_size=self.ngram_size,
                base = self.base
            )
        else:
            raise ValueError(f"unknown attention type: {attention_type}")

        self.ffw = FeedForward(n_embed, dropout_prob)
        self.l1 = nn.LayerNorm(n_embed)
        self.l2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # compute mask if requested
        attention_mask = None
        if self.sliding_window_mask:
            attention_mask = SlidingwindowAttention(x.shape[1], self.window_size, device=x.device)
        else:
            attention_mask = SlidingwindowAttention(x.shape[1], self.block_size, device=x.device)

        x = x + self.attn(self.l1(x), attention_mask)

        x = x + self.ffw(self.l2(x))
        return x
