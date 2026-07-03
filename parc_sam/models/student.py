from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), ConvNormAct(in_ch, out_ch, dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, skip_ch, 1)
        self.conv = ConvNormAct(skip_ch * 2, out_ch, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(self.proj(x), size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class PARCStudent(nn.Module):
    """Single deployable student; SAM is training-only."""

    def __init__(self, in_channels: int, num_classes: int, base_channels: int = 32, dropout: float = 0.1, feature_dim: int = 128):
        super().__init__()
        c = int(base_channels)
        self.stem = ConvNormAct(in_channels, c, dropout * 0.5)
        self.down1 = Down(c, c * 2, dropout)
        self.down2 = Down(c * 2, c * 4, dropout)
        self.down3 = Down(c * 4, c * 8, dropout)
        self.context = nn.Sequential(
            nn.Conv2d(c * 8, c * 8, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(c * 8),
            nn.SiLU(inplace=True),
            nn.Conv2d(c * 8, c * 8, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(c * 8),
            nn.SiLU(inplace=True),
        )
        self.up2 = Up(c * 8, c * 4, c * 4, dropout)
        self.up1 = Up(c * 4, c * 2, c * 2, dropout)
        self.up0 = Up(c * 2, c, c, dropout * 0.5)
        self.feature_head = nn.Sequential(nn.Conv2d(c, feature_dim, 1), nn.BatchNorm2d(feature_dim), nn.SiLU(inplace=True))
        self.classifier = nn.Conv2d(feature_dim, num_classes, 1)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        x0 = self.stem(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.context(self.down3(x2))
        y = self.up2(x3, x2)
        y = self.up1(y, x1)
        y = self.up0(y, x0)
        features = self.feature_head(y)
        logits = self.classifier(features)
        if return_features:
            return {"logits": logits, "features": features, "pyramid": [x0, x1, x2, x3]}
        return logits

