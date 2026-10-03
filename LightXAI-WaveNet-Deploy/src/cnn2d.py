"""
cnn2d.py - CNN 2D Branch (adaptasi dari cnn3d.py)
Untuk dataset CT 2D / X-ray 2D
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN2D(nn.Module):
    def __init__(
        self,
        in_channels=1,
        base_channels=32,
        img_size=128,
        output_dim=128
    ):
        super(CNN2D, self).__init__()
        
        # Block 1: 128x128 -> 64x64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # Block 2: 64x64 -> 32x32
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # Block 3: 32x32 -> 16x16
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # Block 4: 16x16 -> 8x8
        self.conv4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
        )
        
        # Global Average Pooling + FC
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(base_channels * 8, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # x: (B, 1, H, W)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)              # (B, C, 1, 1)
        x = x.view(x.size(0), -1)    # (B, C)
        x = self.fc(x)               # (B, output_dim)
        return x


if __name__ == "__main__":
    model = CNN2D(in_channels=1, base_channels=32, img_size=128, output_dim=128)
    dummy = torch.randn(2, 1, 128, 128)
    out = model(dummy)
    print(f"CNN2D output shape: {out.shape}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")