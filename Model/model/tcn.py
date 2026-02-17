import torch
import torch.nn as nn
import torch.nn.functional as F

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=dilation, dilation=dilation)

        self.norm1 = nn.LayerNorm(out_ch)
        self.norm2 = nn.LayerNorm(out_ch)

        self.dropout = nn.Dropout(dropout)

        self.res = None
        if in_ch != out_ch:
            self.res = nn.Linear(in_ch, out_ch)

    def forward(self, x):
        y = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        y = F.gelu(y)
        y = self.norm1(y)
        y = self.dropout(y)

        y = self.conv2(y.transpose(1, 2)).transpose(1, 2)
        y = F.gelu(y)
        y = self.norm2(y)
        y = self.dropout(y)

        if self.res is not None:
            x = self.res(x)

        return x + y


class ResidualTCN(nn.Module):
    def __init__(self, in_dim, channels, dropout):
        super().__init__()
        blocks = []
        for i in range(len(channels)):
            in_ch = in_dim if i == 0 else channels[i-1]
            out_ch = channels[i]
            blocks.append(
                TCNBlock(in_ch, out_ch, dilation=2**i, dropout=dropout)
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class BiTCN(nn.Module):
    def __init__(self, in_dim=256, channels=(64,128,256,512), dropout=0.1):
        super().__init__()
        self.f = ResidualTCN(in_dim, channels, dropout)
        self.b = ResidualTCN(in_dim, channels, dropout)
        self.norm = nn.LayerNorm(channels[-1] * 2)

    def forward(self, x):
        xf = self.f(x)
        xb = self.b(torch.flip(x, dims=[1]))
        xb = torch.flip(xb, dims=[1])
        return self.norm(torch.cat([xf, xb], dim=-1))
