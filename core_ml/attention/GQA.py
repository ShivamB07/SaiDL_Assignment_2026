import torch
import torch.nn as nn
import torch.nn.functional as F
from attention.RoPE import RotaryEmbedding

class GroupedQueryAttention(nn.Module):

    def __init__(self, n_embed, num_heads, head_size=None, block_size=None, positional_encoding="sinusoidal", group_size=2,
                 ngram_size=3,base=10000,use_local_conv=False):
        super().__init__()
        assert num_heads % group_size == 0, "num_heads must be divisible by group_size"
        self.num_heads = num_heads
        self.group_size = group_size
        self.num_groups = num_heads // group_size
        self.head_size = head_size if head_size is not None else (n_embed // num_heads)
        self.n_embed = n_embed
        self.positional_encoding = positional_encoding  
        self.block_size = block_size
        self.ngram_size = ngram_size
        self.base = base
        self.use_local_conv = use_local_conv

        if self.use_local_conv:
            self.local_conv = nn.Conv1d(
                in_channels=n_embed, 
                out_channels=n_embed, 
                kernel_size=ngram_size, 
                padding=0,
                groups=1,
                bias=True
            )
        # queries are per-head
        self.query = nn.Linear(n_embed, num_heads * self.head_size, bias=False)
        # keys/values are per-group
        self.key = nn.Linear(n_embed, self.num_groups * self.head_size, bias=False)
        self.value = nn.Linear(n_embed, self.num_groups * self.head_size, bias=False)

        if positional_encoding == "rope":
            self.rotary_emb = RotaryEmbedding(self.head_size, base=self.base)

        if self.positional_encoding == "alibi":
            slopes = [1.0 / (2.0 ** (i / num_heads)) for i in range(num_heads)]
            self.register_buffer('slopes', torch.tensor(slopes, dtype=torch.float32))

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape

        if self.use_local_conv:
           
            x_conv = x.transpose(1, 2)
            x_conv = F.pad(x_conv, (self.ngram_size - 1, 0))
            x_conv = F.silu(self.local_conv(x_conv))
            x_conv = x_conv.transpose(1, 2)
            x = x + x_conv

        q = self.query(x)  # (B,T,num_heads*head_size)
        k = self.key(x)    # (B,T,num_groups*head_size)
        v = self.value(x)  # (B,T,num_groups*head_size)

        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)  # (B,num_heads,T,head_size)
        k = k.view(B, T, self.num_groups, self.head_size).transpose(1, 2)  # (B,num_groups,T,head_size)
        v = v.view(B, T, self.num_groups, self.head_size).transpose(1, 2)
        
        if self.positional_encoding == "rope":
            cos, sin = self.rotary_emb(q, seq_len=T)
            q, k = self.rotary_emb.apply_rotary_pos_emb(q, k, cos, sin)

        # expand k/v from groups -> heads by repeating each group `group_size` times
        k_exp = k.repeat_interleave(self.group_size, dim=1)  # (B, num_heads, T, head_size)
        v_exp = v.repeat_interleave(self.group_size, dim=1)

        wei = torch.matmul(q, k_exp.transpose(-1, -2))  # (B,num_heads,T,T)

        # ALiBi
        if self.positional_encoding == "alibi":
            row = torch.arange(T, device=x.device).unsqueeze(0)  # [1, T]
            matrix = row.repeat(T, 1)  # [T, T]
            albi_matrix = matrix - row.transpose(0, 1)  # [T, T]
            albi_bias = self.slopes.view(self.num_heads, 1, 1) * albi_matrix.unsqueeze(0)
            wei = wei + albi_bias

        if attention_mask is not None:
            wei = wei + attention_mask

        wei = F.softmax(wei / (self.head_size ** 0.5), dim=-1)
        out = torch.matmul(wei, v_exp)
        out = out.transpose(1, 2)  # (B,T,num_heads,head_size)
        out = out.contiguous().view(B, T, -1)

        return out
