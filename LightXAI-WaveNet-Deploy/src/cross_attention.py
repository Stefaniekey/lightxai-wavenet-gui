"""
cross_attention.py - Cross-Attention Fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, feature_dim=128, num_heads=4, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(self, query, key, value):
        B = query.shape[0]
        
        Q = self.q_proj(query).reshape(B, self.num_heads, self.head_dim)
        K = self.k_proj(key).reshape(B, self.num_heads, self.head_dim)
        V = self.v_proj(value).reshape(B, self.num_heads, self.head_dim)
        
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ V).reshape(B, self.feature_dim)
        out = self.out_proj(out)
        out = self.dropout(out)
        out = self.norm(query + out)
        
        return out


class CrossAttentionFusion(nn.Module):
    def __init__(self, feature_dim=128, num_heads=4, dropout=0.1):
        super(CrossAttentionFusion, self).__init__()
        self.cnn_to_vit = CrossAttention(feature_dim, num_heads, dropout)
        self.vit_to_cnn = CrossAttention(feature_dim, num_heads, dropout)
        
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim)
        )
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(self, cnn_features, vit_features):
        cnn_attended = self.cnn_to_vit(cnn_features, vit_features, vit_features)
        vit_attended = self.vit_to_cnn(vit_features, cnn_features, cnn_features)
        
        combined = torch.cat([cnn_attended, vit_attended], dim=1)
        fused = self.fusion(combined)
        fused = self.norm(fused)
        
        return fused


if __name__ == "__main__":
    model = CrossAttentionFusion(feature_dim=128, num_heads=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    cnn_features = torch.randn(4, 128).to(device)
    vit_features = torch.randn(4, 128).to(device)
    
    output = model(cnn_features, vit_features)
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\n✅ CrossAttentionFusion berhasil!")