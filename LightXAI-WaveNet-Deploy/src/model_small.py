"""
model_small.py - LightXAI-WaveNet Versi Kecil (Sederhana)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from cnn3d import CNN3D
from vit3d import ViT3D
from cross_attention import CrossAttentionFusion


class LightXAIWaveNetSmall(nn.Module):
    def __init__(
        self,
        num_classes=5,
        img_size=128,
        patch_size=16,
        depth_dim=8,
        cnn_base_channels=32,
        vit_embed_dim=256,
        vit_depth=4,
        cross_attention_heads=4,
        dropout=0.1
    ):
        super(LightXAIWaveNetSmall, self).__init__()
        
        self.cnn_branch = CNN3D(
            in_channels=1,
            base_channels=cnn_base_channels,
            depth=depth_dim,
            img_size=img_size,
            output_dim=128
        )
        
        self.vit_branch = ViT3D(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=1,
            embed_dim=vit_embed_dim,
            depth=vit_depth,
            depth_dim=depth_dim,
            output_dim=128
        )
        
        self.cross_attention = CrossAttentionFusion(
            feature_dim=128,
            num_heads=cross_attention_heads,
            dropout=dropout
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        cnn_features = self.cnn_branch(x)
        vit_features = self.vit_branch(x)
        fused_features = self.cross_attention(cnn_features, vit_features)
        logits = self.classifier(fused_features)
        
        return logits, {
            'cnn': cnn_features,
            'vit': vit_features,
            'fused': fused_features
        }


if __name__ == "__main__":
    model = LightXAIWaveNetSmall(
        num_classes=5,
        img_size=128,
        patch_size=16,
        depth_dim=8,
        cnn_base_channels=32,
        vit_embed_dim=256,
        vit_depth=4,
        cross_attention_heads=4,
        dropout=0.1
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    dummy_input = torch.randn(2, 1, 8, 128, 128).to(device)
    logits, features = model(dummy_input)
    
    print(f"Logits shape: {logits.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    print("\n✅ LightXAIWaveNetSmall berhasil!")