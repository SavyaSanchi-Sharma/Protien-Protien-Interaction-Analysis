import torch.nn as nn

class ProtBertProjection(nn.Module):
    def __init__(self, in_dim=1024, hidden_dim=512, out_dim=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        return self.net(x)
