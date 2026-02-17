import torch
import torch.nn as nn

class MaskedSelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        h, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        x = self.ln1(x + self.dropout(h))
        h = self.ffn(x)
        return self.ln2(x + self.dropout(h))


class AttnEncoder(nn.Module):
    def __init__(self, dim, num_layers=3, heads=8, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            MaskedSelfAttention(dim, heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

