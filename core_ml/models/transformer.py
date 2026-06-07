import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Block


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_embed,
        block_size,
        num_layers,
        num_head,
        dropout_prob,
        attention_type: str = "mha",
        sliding_window_mask: bool = False,
        window_size: int = 512,
        positional_encoding: str = "learnable",
        group_size: int = 2,
        use_local_conv: bool = False,
        ngram_size: int = 3,
        base: int = 10000
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.block_size = block_size
        self.num_head = num_head
        self.head_size = n_embed // num_head
        self.num_layers = num_layers
        self.window_size = window_size
        self.dropout_prob = dropout_prob
        self.attention_type = attention_type
        self.base = base    
        self.sliding_window_mask = sliding_window_mask
        self.window_size = window_size
        self.positional_encoding = positional_encoding
        self.group_size = group_size
        self.use_local_conv = use_local_conv
        self.ngram_size = ngram_size
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        
        # Only create position embeddings if not using ALiBi
        if positional_encoding == "learnable":
            self.position_embedding_table = nn.Embedding(block_size, n_embed)
        
        self.blocks = nn.Sequential(
            *[Block(
                head_size=self.head_size,
                num_head=self.num_head,
                n_embed=self.n_embed,
                dropout_prob=self.dropout_prob,
                block_size=self.block_size,
                attention_type=self.attention_type,
                sliding_window_mask=self.sliding_window_mask,
                window_size=self.window_size,
                positional_encoding=self.positional_encoding,
                group_size=self.group_size,
                use_local_conv=self.use_local_conv,
                ngram_size=self.ngram_size,
                base=self.base
            ) for _ in range(self.num_layers)]
        )
        self.ln_f = nn.LayerNorm(n_embed)
        self.lf = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embed = self.token_embedding_table(idx)
        
        # Apply positional encoding only if not ALiBi (ALiBi is applied in attention)
        if self.positional_encoding == "learnable":
            position_embed = self.position_embedding_table(torch.arange(T, device=idx.device))
            x = token_embed + position_embed
        else:
            x = token_embed
        
        
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lf(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
