import torch
import torch.nn as nn

# -------------------------------
# Multihead Self Attention
# -------------------------------
class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, L, D = x.shape
        q = self.q(x).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(x).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 3, 1)
        v = self.v(x).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k) * self.scale  # (B,H,L,L)
        if attn_mask is not None:
            attn = attn.masked_fill(~attn_mask, float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # (B,H,L,d)
        out = out.transpose(1, 2).reshape(B, L, -1)  # (B,L,D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# -------------------------------
# Feed Forward Network
# -------------------------------
class FFNBlock(nn.Module):
    def __init__(self, d_model: int, mlp_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, d_model)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# -------------------------------
# Encoder Block (LN → MSA → Skip → LN → FFN → Skip)
# -------------------------------
class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, mlp_dim: int,
                 attn_drop: float = 0.0, proj_drop: float = 0.0, mlp_drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiheadSelfAttention(d_model, num_heads, attn_drop, proj_drop)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = FFNBlock(d_model, mlp_dim, mlp_drop)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


# -------------------------------
# Encoder (L blocks stacked)
# -------------------------------
class Encoder(nn.Module):
    def __init__(self, d_model: int, num_layers: int, num_heads: int, mlp_dim: int,
                 attn_drop: float = 0.0, proj_drop: float = 0.0, mlp_drop: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, mlp_dim, attn_drop, proj_drop, mlp_drop)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):  # x: (B, N+1, D)
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        x = self.norm(x)
        cls = x[:, 0]  # [CLS]만 반환
        return cls


# -------------------------------
# Patch Embedding
# -------------------------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, pos_drop=0.0):
        super().__init__()
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(pos_drop)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):  # x: (B,3,H,W)
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, N, D)
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.pos_drop(x)
        return x


# -------------------------------
# Head
# -------------------------------
class MLPHead(nn.Module):
    def __init__(self, in_dim, hid_dim, num_classes, drop=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.act = nn.Tanh()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hid_dim, num_classes)

    def forward(self, x):  # x: CLS (B,D)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


# -------------------------------
# 전체 Vision Transformer
# -------------------------------
class ViTClassifier(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12, mlp_dim=3072,
                 num_classes=1000, attn_drop=0.0, proj_drop=0.0, mlp_drop=0.0, pos_drop=0.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, pos_drop)
        self.encoder = Encoder(embed_dim, depth, num_heads, mlp_dim,
                               attn_drop, proj_drop, mlp_drop)
        self.head = MLPHead(embed_dim, embed_dim, num_classes, drop=0.1)

    def forward(self, x, attn_mask=None):
        tokens = self.patch_embed(x)           # (B,N+1,D)
        cls = self.encoder(tokens, attn_mask)  # (B,D)
        logits = self.head(cls)                # (B,num_classes)
        return logits

