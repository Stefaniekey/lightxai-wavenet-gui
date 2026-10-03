"""
model_2d.py - LightXAI-WaveNet 2D (adaptasi dari model_small.py)
Untuk dataset CT 2D / X-ray 2D
"""

import torch
import torch.nn as nn

from cnn2d import CNN2D
from vit2d import ViT2D
from cross_attention import CrossAttentionFusion


class LightXAIWaveNet2D(nn.Module):
    def __init__(
        self,
        num_classes=3,
        img_size=128,
        patch_size=16,
        cnn_base_channels=32,
        vit_embed_dim=256,
        vit_depth=4,
        vit_num_heads=8,
        cross_attention_heads=4,
        dropout=0.1
    ):
        super(LightXAIWaveNet2D, self).__init__()
        
        self.cnn_branch = CNN2D(
            in_channels=1,
            base_channels=cnn_base_channels,
            img_size=img_size,
            output_dim=128
        )
        
        self.vit_branch = ViT2D(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=1,
            embed_dim=vit_embed_dim,
            depth=vit_depth,
            num_heads=vit_num_heads,
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
    model = LightXAIWaveNet2D(num_classes=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    dummy = torch.randn(2, 1, 128, 128).to(device)
    logits, features = model(dummy)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
    print("\n✅ LightXAIWaveNet2D berhasil!")