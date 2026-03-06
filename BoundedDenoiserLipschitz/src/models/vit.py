import torch
import torch.nn as nn
import torch.nn.functional as F


from models.attention import AttentionMLP


def _choose_groupnorm_vit(num_channels: int, max_groups: int =8) -> int:
    for g in range(min(max_groups, num_channels), 0, -1):
        if num_channels % g == 0:
            return g
    return 1

class ConvGNact2D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        g = _choose_groupnorm_vit(num_channels=out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g, out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout < 0 else nn.Identity(),
        )
    def forward(self, x: torch.Tensor) ->  torch.Tensor:
        return self.block(x)

class FusionBlock2D(nn.Module):
    """
    upsampled feature + skip > refined feature
    """
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            ConvGNact2D(in_channels, out_channels, dropout=dropout),
            ConvGNact2D(out_channels, out_channels, dropout=dropout),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        print(f'before concat: {x.shape}')
        x = torch.cat([x, skip], dim=1)
        print(f'after concat: {x.shape}')
        return self.block(x)


class HierarchicalEncoder2D(nn.Module):
    def __init__(self,
                 in_channels,
                 embed_dim,
                 depth,
                 sr_ratio,
                 num_heads=None,
                 patch_size=4):
        super().__init__()
        assert len(embed_dim) == len(depth) == len(sr_ratio)
        self.num_stages = len(embed_dim)
        if num_heads is None:
            num_heads = [max(1, d // 64) for d in embed_dim]

        self.patch_embed = ViT2DPatchEmbed(in_channels, embed_dim[0], patch_size=patch_size)

        self.stages = nn.ModuleList()
        self.dows = nn.ModuleList()

        for i in range(self.num_stages):
            dim = embed_dim[i]
            hd = num_heads[i]
            sr = sr_ratio[i]
            dth = depth[i]

            blocks = nn.ModuleList()

            embed = ViT2DPatchEmbed(
                in_channels if i==0 else embed_dim[i-1],
                embed_dim[i],
                kernel=[7,3,3,3][i],
                stride=[4,2,2,2][i]
            )
            blocks = nn.ModuleList([
                AttentionMLP(embed_dim[i], 1<<i, sr_ratio[i])
                for _ in range(depth[i])
            ])

    def forward(self, x):

        tok, grid, feat = self.patch_embed
        feats = []
        return feats

class MLPDecoder(nn.Module):
    """
    features: [f1, f2, f3, f4] (low to high)
    decode : high to low and output logits
    """

    def __init__(self, embed_dim, num_classes, dropout=0.0):
        super().__init__()
        # embed_dim_rev = [c4, c3, c2, c1]
        c4, c3, c2, c1 = embed_dim

        self.up43 = FusionBlock2D(c4 + c3, c3, dropout=dropout)
        self.up32 = FusionBlock2D(c3 + c2, c2, dropout=dropout)
        self.up21 = FusionBlock2D(c2 + c1, c1, dropout=dropout)

        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, feats, out_size):
        f1, f2, f3, f4 = feats


        x = F.interpolate(f4, size=f3.shape[-2:], mode='bilinear', align_corners=False)
        x = self.up43(x, f3)

        x = F.interpolate(x, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        x = x + self.up32(x, f2)

        x = F.interpolate(x, size=f1.shape[-2:], mode='bilinear', align_corners=False)
        x = x + self.up21(x, f1)
        x = x + self.ref21(x)

        logits = self.head(x)

        if out_size is not None:
            logits = F.interpolate(logits, size=out_size, mode='bilinear', align_corners=False)

        return logits

class ViT2DPatchEmbed(nn.Module):
    """
    transforms 2D medical volumes into sequences of tokens
    u = f(x, y)
    """

    def __init__(self, patch_size=4, in_channels=1, embed_dim=768, kernel=None, stride=None):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels=in_channels, emded_dim=embed_dim, kernel_size=kernel, stride=stride, bias=False)

    def forward(self, x):
        # [B, dim, Ll, Ww]
        x = self.proj(x)
        B, C, L, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)

        return tokens, (L, W), x


class PatchMerging2D(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, feat):
        # feat : [B, C, L, W]
        feat = self.conv(feat)
        print(f'feat.shape: {feat.shape}') # [B, out, L/2, W/2]
        B, C, L, W = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        tokens = self.norm(tokens)

        return tokens, (L, W), feat


class Light2DVit(nn.Module):
    def __init__(self,
                 in_channels=4,
                 num_classes=3,
                 embed_dim=(48, 96, 192, 384),
                 depths=(2, 2, 2, 2),
                 sr_ratios=(4, 2, 1, 1),
                 block_type='sr',
                 ode_mode='strang',
                 ode_steps_attn=2,
                 ode_steps_mlp=1,
                 ode_steps_fric=1,
                 use_friction=True,
                 friction_position='mid',
                 patch_size=4):
        super().__init__()
        self.encoder = HierarchicalEncoder2D(
            in_channels=in_channels,
            embed_dim=list(embed_dim),
            depth=list(depths),
            sr_ratio=list(sr_ratios),
            mlp_ratio=4.0,
            dropout=0.0,
            attn_drop=0.0,
            block_type=block_type,
            ode_mode=ode_mode,
            ode_steps_attn=ode_steps_attn,
            ode_steps_mlp=ode_steps_mlp,
            ode_steps_fric=ode_steps_fric,
            use_friction=use_friction,
            friction_position=friction_position,
            patch_size=patch_size
        )
        self.decoder = MLPDecoder(list(embed_dim)[::-1], num_classes=num_classes)

    def forward(self, x):
        feats = self.encoder(x)
        out = self.decoder(feats)

        return out