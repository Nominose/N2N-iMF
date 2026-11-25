"""2D conditional U-Net, adapted from Phil Wang. See THIRD_PARTY_LICENSES."""

import itertools
import math

import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn


def Upsample2D(dim, dim_out):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, dim_out, 3, padding=1),
    )


def Downsample2D(dim, dim_out):
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, dim_out, 1),
    )


class RMSNorm2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * x.shape[1] ** 0.5


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.conv(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x


class ResnetBlock2D(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if time_emb_dim is not None
            else None
        )
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class LinearAttention2D(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** (-0.5)
        self.heads = heads
        hidden_dim = dim_head * heads
        self.norm = RMSNorm2D(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), RMSNorm2D(dim))

    def forward(self, x):
        _b, _c, h, w = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = (
            rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads) for t in qkv
        )
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention2D(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.norm = RMSNorm2D(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        _b, _c, h, w = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = (
            rearrange(t, "b (h c) x y -> b h (x y) c", h=self.heads) for t in qkv
        )
        scores = torch.einsum("bhid,bhjd->bhij", q, k) * (q.shape[-1] ** -0.5)
        out = torch.einsum("bhij,bhjd->bhid", scores.softmax(dim=-1), v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class IntervalEmbedding(nn.Module):
    def __init__(self, width):
        super().__init__()
        output = width * 4
        self.original_time_mlp = nn.Sequential(
            SinusoidalPosEmb(width),
            nn.Linear(width, output),
            nn.GELU(),
            nn.Linear(output, output),
        )
        interval_width = max(width, 32)
        self.interval_embed = nn.Sequential(
            SinusoidalPosEmb(interval_width),
            nn.Linear(interval_width, output),
            nn.GELU(),
            nn.Linear(output, output),
        )

    def forward(self, start, end):
        return self.original_time_mlp(999 * end) + self.interval_embed(
            999 * (end - start)
        )


class ConditionalUNet(nn.Module):
    def __init__(self, channels, width=64):
        super().__init__()
        self.init_conv = nn.Conv2d(1 + channels, width, 5, padding=2)
        self.time_mlp = IntervalEmbedding(width)
        time_width = width * 4
        dims = [width, width, 2 * width, 4 * width, 8 * width]
        stages = list(itertools.pairwise(dims))
        attention = [nn.Identity, nn.Identity, LinearAttention2D, Attention2D]

        def block(inputs, outputs):
            return ResnetBlock2D(inputs, outputs, time_emb_dim=time_width)

        self.downs = nn.ModuleList()
        for i, (a, b) in enumerate(stages):
            attn = attention[i](a) if i >= 2 else nn.Identity()
            down = Downsample2D(a, b) if i < 3 else nn.Conv2d(a, b, 3, padding=1)
            self.downs.append(nn.ModuleList([block(a, a), block(a, a), attn, down]))

        self.mid_block1 = block(dims[-1], dims[-1])
        self.mid_attn = Attention2D(dims[-1])
        self.mid_block2 = block(dims[-1], dims[-1])
        self.ups = nn.ModuleList()
        for i in reversed(range(len(stages))):
            a, b = stages[i]
            attn = attention[i](b) if i >= 2 else nn.Identity()
            up = Upsample2D(b, a) if i > 0 else nn.Conv2d(b, a, 5, padding=2)
            self.ups.append(nn.ModuleList([block(a + b, b), block(a + b, b), attn, up]))
        self.final_res_block = block(2 * width, width)
        self.final_conv = nn.Conv2d(width, 1, 1)
        self.final_conv_v = nn.Conv2d(width, 1, 1)

    def forward(self, image, condition, start, end):
        initial = self.init_conv(torch.cat((image, condition), dim=1))
        emb = self.time_mlp(start, end)
        x = initial
        skips = []
        for first, second, attention, down in self.downs:
            x = first(x, emb)
            skips.append(x)
            x = second(x, emb)
            x = attention(x) + x
            skips.append(x)
            x = down(x)
        x = self.mid_block1(x, emb)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, emb)
        for first, second, attention, up in self.ups:
            x = first(torch.cat((x, skips.pop()), dim=1), emb)
            x = second(torch.cat((x, skips.pop()), dim=1), emb)
            x = up(attention(x) + x)
        features = self.final_res_block(torch.cat((x, initial), dim=1), emb)
        return self.final_conv(features), self.final_conv_v(features)
