import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class DeepGCNLayer(nn.Module):
    def __init__(self, hidden, dropout=0.2):
        super().__init__()
        self.conv = GCNConv(hidden, hidden, add_self_loops=True, normalize=True)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None):
        out = self.conv(x, edge_index, edge_weight=edge_weight)
        out = F.gelu(out)
        out = self.norm(out)
        out = self.dropout(out)
        return out

class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden=256, num_layers=6, alpha=0.1, dropout=0.2):
        super().__init__()
        self.alpha = alpha

        self.input = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout)
        )

        self.layers = nn.ModuleList([
            DeepGCNLayer(hidden, dropout=dropout) for _ in range(num_layers)
        ])

    def forward(self, x,edge_index, edge_weight=None):
        x = self.input(x)       # [N, hidden]
        h0 = x  # Use projected features as residual base
        for layer in self.layers:
            h = layer(x, edge_index, edge_weight=edge_weight)
            h = (1 - self.alpha) * h + self.alpha * h0
            x = x + h
        return x
