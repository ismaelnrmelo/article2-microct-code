"""Residual U-Net with bottleneck attention for the accepted checkpoint.

For 96 by 96 inputs, encoder feature maps are 96 by 96 by 64,
48 by 48 by 128, 24 by 24 by 128, and 12 by 12 by 128.
"""

import torch
from torch import nn
from torch.nn import functional as Fn

class Res(nn.Module):

    def __init__(self, cin, cout, cemb, drop=0.1):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.e = nn.Linear(cemb, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.dp = nn.Dropout(drop)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.sk = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb):
        h = self.c1(Fn.silu(self.n1(x)))
        h = h + self.e(emb)[:, :, None, None]
        h = self.c2(self.dp(Fn.silu(self.n2(h))))
        return h + self.sk(x)


class Attn(nn.Module):

    def __init__(self, ch, cabecas=4):
        super().__init__()
        self.n = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.o = nn.Conv2d(ch, ch, 1)
        self.h = cabecas

    def forward(self, x):
        (b, c, hh, ww) = x.shape
        (q, k, v) = self.qkv(self.n(x)).chunk(3, dim=1)

        def r(t):
            return t.reshape(b, self.h, c // self.h, hh * ww).transpose(2, 3)
        a = Fn.scaled_dot_product_attention(r(q), r(k), r(v))
        a = a.transpose(2, 3).reshape(b, c, hh, ww)
        return x + self.o(a)


class UNet4(nn.Module):

    def __init__(self, base=64, cemb=256):
        super().__init__()
        (c1, c2, c3, c4) = (base, base * 2, base * 2, base * 2)
        self.emb = nn.Sequential(nn.Linear(64, cemb), nn.SiLU(), nn.Linear(cemb, cemb))
        self.entra = nn.Conv2d(1, c1, 3, padding=1)
        self.d11 = Res(c1, c1, cemb)
        self.d12 = Res(c1, c1, cemb)
        self.dw1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.d21 = Res(c2, c2, cemb)
        self.d22 = Res(c2, c2, cemb)
        self.dw2 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)
        self.d31 = Res(c3, c3, cemb)
        self.d32 = Res(c3, c3, cemb)
        self.dw3 = nn.Conv2d(c3, c4, 3, stride=2, padding=1)
        self.m1 = Res(c4, c4, cemb)
        self.at = Attn(c4)
        self.m2 = Res(c4, c4, cemb)
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.u31 = Res(c3 * 2, c3, cemb)
        self.u32 = Res(c3, c3, cemb)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.u21 = Res(c2 * 2, c2, cemb)
        self.u22 = Res(c2, c2, cemb)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.u11 = Res(c1 * 2, c1, cemb)
        self.u12 = Res(c1, c1, cemb)
        self.sai_n = nn.GroupNorm(8, c1)
        self.sai = nn.Conv2d(c1, 1, 3, padding=1)
        nn.init.zeros_(self.sai.weight)
        nn.init.zeros_(self.sai.bias)
        self.register_buffer('freqs', torch.exp(torch.linspace(0, 8, 32)))

    def forward(self, x, c_noise):
        f = c_noise[:, None] * self.freqs[None, :]
        emb = self.emb(torch.cat([torch.sin(f), torch.cos(f)], dim=1))
        h1 = self.d12(self.d11(self.entra(x), emb), emb)
        h2 = self.d22(self.d21(self.dw1(h1), emb), emb)
        h3 = self.d32(self.d31(self.dw2(h2), emb), emb)
        m = self.m2(self.at(self.m1(self.dw3(h3), emb)), emb)
        u3 = self.u32(self.u31(torch.cat([self.up3(m), h3], 1), emb), emb)
        u2 = self.u22(self.u21(torch.cat([self.up2(u3), h2], 1), emb), emb)
        u1 = self.u12(self.u11(torch.cat([self.up1(u2), h1], 1), emb), emb)
        return self.sai(Fn.silu(self.sai_n(u1)))
