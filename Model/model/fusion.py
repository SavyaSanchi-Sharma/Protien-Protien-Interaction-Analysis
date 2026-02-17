import torch
from torch import nn
class GatedFusion(nn.Module):
    def __init__(self, d_esm=256, d_struct=36, d_out=256, dropout=0.25):
        super().__init__()
        self.d_esm = d_esm
        self.d_struct = d_struct

        # gate uses both esm and structure
        self.gate = nn.Sequential(
            nn.Linear(d_esm + d_struct, d_struct),
            nn.Sigmoid()
        )

        # fusion projection
        self.fuse = nn.Sequential(
            nn.Linear(d_esm + d_struct, d_out),
            nn.GELU(),
            nn.LayerNorm(d_out),
            nn.Dropout(dropout),
            nn.Linear(d_out, d_out),
            nn.GELU(),
            nn.LayerNorm(d_out),
        )

    def forward(self, esm_proj, struct):
        """
        esm_proj: [N, d_esm]
        struct:   [N, d_struct]
        """
        g = self.gate(torch.cat([esm_proj, struct], dim=-1))  # [N, d_struct]
        struct_g = g * struct                                 # [N, d_struct]

        x = torch.cat([esm_proj, struct_g], dim=-1)
        return self.fuse(x)   # [N, d_out]
