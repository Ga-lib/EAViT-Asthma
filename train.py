import torch
import torch.nn as nn
from einops import rearrange
import timm

class EAViT(nn.Module):
    def __init__(self, input_size=(224, 224), patch_size=(16, 16), num_classes=5,
                 embed_dim=128, depth=4, num_heads=4, dropout=0.1):
        super().__init__()

        # Patch Embedding
        self.patch_embed = nn.Conv2d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        num_patches = (input_size[0] // patch_size[0]) * (input_size[1] // patch_size[1])

        # Positional Embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Classification Head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: [B, 1, H, W] -> Patch Embedding -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, embed_dim, H/ps, W/ps]
        x = rearrange(x, 'b c h w -> b (h w) c')

        # Add CLS token
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]

        # Transformer
        x = self.encoder(x)

        # Use CLS token for prediction
        x = self.norm(x[:, 0])
        return self.head(x)