import torch
import torch.nn as nn
import torch.nn.functional as F


class SegViTAttention(nn.Module):
    def __init__(self, dim, num_heads, attn_drop, proj_drop):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_token):
        # x_token: (B, N, C)
        B, N, C = x_token.shape
        qkv = self.qkv(x_token).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class SegViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, dropout, attn_drop, proj_drop):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SegViTAttention(dim, num_heads, attn_drop, proj_drop)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x_token):
        x_token = self.norm1(x_token + self.attn(x_token))
        x_token = self.norm2(x_token + self.mlp(x_token))
        return x_token
    

def get_gn_groups(channels, target_per=16):
    groups = channels // target_per
    groups = max(1, groups)
    while channels % groups != 0:
        groups -= 1
    return groups


class ConvPosEnc(nn.Module):
    def __init__(self, dim):
        g = get_gn_groups(dim)
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.GroupNorm(g, dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.GroupNorm(g, dim)
        )
        
    def forward(self, x, H, W):
        B, N, C = x.shape
        feat = x.transpose(1, 2).view(B, C, H, W)
        feat = self.proj(feat)
        return feat.flatten(2).transpose(1, 2)

class SegViTBottleneck(nn.Module):
    def __init__(self, in_channel: int, num_heads: int = 2, num_layers: int = 2, patch_size: int = 4, mlp_ratio=4, dropout=0.15, attn_drop=0.15, proj_drop=0.1) -> None:
        super().__init__()
        embed_dim = in_channel
        self.embed_dim = embed_dim
        g_emb = get_gn_groups(embed_dim)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channel, embed_dim, patch_size, patch_size, bias=False),
            nn.GroupNorm(g_emb, embed_dim),
            nn.ReLU(inplace=True)
        )
        self.pos_embed = ConvPosEnc(embed_dim)
        self.transformer = nn.Sequential(*[
            SegViTBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout, attn_drop=attn_drop, proj_drop=proj_drop)
            for _ in range(num_layers)
        ])
        self.up = nn.Upsample(scale_factor=patch_size, mode='bilinear') # 4→16
        self.res_conv = nn.Sequential(
            nn.Conv2d(in_channel, embed_dim, kernel_size=1, bias=False),
            nn.GroupNorm(g_emb, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = self.res_conv(x)

        x = self.patch_embed(x)
        _, _, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_embed(x, Hp, Wp)
        x = self.transformer(x)
        x = x.transpose(1, 2).reshape(B, self.embed_dim, Hp, Wp)
        x = self.up(x)
        return x + residual
