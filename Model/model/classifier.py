import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, dim=1024, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        return self.net(x) # logits
