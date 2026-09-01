"""
vit3d.py - 3D Vision Transformer (SEDERHANA - PASTI JALAN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class PatchEmbed3D(nn.Module):
    def __init__(self, img_size=128, patch_size=16, in_channels=1, embed_dim=256):
        super(PatchEmbed3D, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        self.num_patches_per_dim = img_size // patch_size
        self.num_patches_spatial = self.num_patches_per_dim ** 2
        
        self.proj = nn.Conv3d(
            in_channels, 
            embed_dim, 
            kernel_size=(1, patch_size, patch_size), 
            stride=(1, patch_size, patch_size)
        )
        
    def forward(self, x):
        B, C, D, H, W = x.shape
        x = self.proj(x)
        x = rearrange(x, 'b e d h w -> b (d h w) e')
        return x


class SimpleAttention3D(nn.Module):
    """
    Attention sederhana tanpa multi-head (untuk menghindari error dimensi)
    """
    def __init__(self, embed_dim=256, dropout=0.1):
        super(SimpleAttention3D, self).__init__()
        self.embed_dim = embed_dim
        self.scale = embed_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, C)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = attn @ v
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlockSimple(nn.Module):
    def __init__(self, embed_dim=256, mlp_ratio=4, dropout=0.1):
        super(TransformerBlockSimple, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = SimpleAttention3D(embed_dim, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT3D(nn.Module):
    def __init__(
        self,
        img_size=128,
        patch_size=16,
        in_channels=1,
        embed_dim=256,
        depth=4,
        mlp_ratio=4,
        dropout=0.1,
        depth_dim=8,
        output_dim=128
    ):
        super(ViT3D, self).__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        num_patches_per_dim = img_size // patch_size
        num_patches_spatial = num_patches_per_dim ** 2
        self.num_patches = num_patches_spatial * depth_dim
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlockSimple(embed_dim, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
            nn.GELU()
        )
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x):
        B = x.shape[0]
        
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        cls_output = x[:, 0]
        features = self.fc(cls_output)
        
        return features


if __name__ == "__main__":
    model = ViT3D(
        img_size=128,
        patch_size=16,
        in_channels=1,
        embed_dim=256,
        depth=4,
        depth_dim=8,
        output_dim=128
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    dummy_input = torch.randn(2, 1, 8, 128, 128).to(device)
    print(f"Input shape: {dummy_input.shape}")
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\n✅ ViT3D berhasil!")