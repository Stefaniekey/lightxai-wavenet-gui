"""
cnn3d.py - 3D CNN Branch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN3D(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, depth=8, img_size=128, output_dim=128):
        super(CNN3D, self).__init__()
        
        self.encoder = nn.Sequential(
            self._conv_block(in_channels, base_channels),
            self._conv_block(base_channels, base_channels * 2),
            self._conv_block(base_channels * 2, base_channels * 4),
        )
        
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(base_channels * 4, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(inplace=True)
        )
        
        self.depth = depth
        self.img_size = img_size
        self.output_dim = output_dim
        
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        features = self.fc(x)
        return features


if __name__ == "__main__":
    model = CNN3D(in_channels=1, base_channels=32, depth=8, img_size=128, output_dim=128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    dummy_input = torch.randn(2, 1, 8, 128, 128).to(device)
    print(f"Input shape: {dummy_input.shape}")
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\n✅ CNN3D berhasil!")