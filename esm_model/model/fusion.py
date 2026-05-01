import torch
from torch import nn


class CrossAttentionFusion(nn.Module):
    def __init__(self, d_esm=256, d_struct=17, d_out=256, n_heads=4,
                 dropout=0.2, ffn_mult=2):
        super().__init__()
        self.struct_up = nn.Linear(d_struct, d_out)

        self.s2e_attn = nn.MultiheadAttention(
            d_out, n_heads, dropout=dropout, batch_first=True
        )
        self.e2s_attn = nn.MultiheadAttention(
            d_out, n_heads, dropout=dropout, batch_first=True
        )

        self.ln_s1 = nn.LayerNorm(d_out)
        self.ln_e1 = nn.LayerNorm(d_out)

        self.ffn_s = nn.Sequential(
            nn.Linear(d_out, d_out * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_out * ffn_mult, d_out),
        )
        self.ffn_e = nn.Sequential(
            nn.Linear(d_out, d_out * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_out * ffn_mult, d_out),
        )

        self.ln_s2 = nn.LayerNorm(d_out)
        self.ln_e2 = nn.LayerNorm(d_out)

        self.merge = nn.Linear(2 * d_out, d_out)
        self.ln_out = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, esm, struct, mask=None):
        struct_h = self.struct_up(struct)
        # MHA's key_padding_mask treats True as "ignore"; PyG's to_dense_batch
        # mask uses True for valid residues. Invert before passing in.
        kpm = ~mask if mask is not None else None

        s2e, _ = self.s2e_attn(struct_h, esm, esm,
                               key_padding_mask=kpm, need_weights=False)
        struct_h = self.ln_s1(struct_h + s2e)
        struct_h = self.ln_s2(struct_h + self.ffn_s(struct_h))

        e2s, _ = self.e2s_attn(esm, struct_h, struct_h,
                               key_padding_mask=kpm, need_weights=False)
        esm_h = self.ln_e1(esm + e2s)
        esm_h = self.ln_e2(esm_h + self.ffn_e(esm_h))

        merged = self.merge(torch.cat([esm_h, struct_h], dim=-1))
        merged = self.ln_out(self.drop(merged))
        if mask is not None:
            merged = merged * mask.unsqueeze(-1).to(merged.dtype)
        return merged
