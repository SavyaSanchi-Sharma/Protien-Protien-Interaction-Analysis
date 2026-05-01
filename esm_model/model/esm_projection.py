import torch
import torch.nn as nn


class MultiLayerProjection(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.in_dim = in_dim
        # ESM-2's intermediate layers have ~100x larger magnitudes than the
        # final layer (no internal normalisation), so a naive scalar mix would
        # let them dominate. Per-layer LayerNorm puts every layer on the same
        # scale before the optimiser chooses weights.
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(in_dim) for _ in range(num_layers)]
        )
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))
        self.layer_scale = nn.Parameter(torch.tensor(1.0))
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        if x.shape[-2] != self.num_layers or x.shape[-1] != self.in_dim:
            raise RuntimeError(
                f"MultiLayerProjection expected (..., {self.num_layers}, "
                f"{self.in_dim}); got {tuple(x.shape)}"
            )
        normed = torch.stack(
            [self.layer_norms[L](x[..., L, :]) for L in range(self.num_layers)],
            dim=-2,
        )
        w = torch.softmax(self.layer_weights, dim=0)
        view_shape = (1,) * (normed.ndim - 2) + (self.num_layers, 1)
        mixed = (normed * w.view(*view_shape)).sum(dim=-2) * self.layer_scale
        return self.proj(mixed)
