# Copyright (c) 2023, salesforce.com, inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Adapted from LAVIS eva_vit.py, based on EVA, BEiT, timm, DeiT and DINO.
# See LICENSE and SOURCES for notices and the original source locations.
"""Only the EVA-CLIP ViT-g/14 feature encoder used by BLIP2 pretrain."""

import torch
from timm.models.layers import Mlp, PatchEmbed
from torch import nn
from torch.nn import functional as F


WIDTH = 1408
HEADS = 16


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH, bias=False)
        self.q_bias = nn.Parameter(torch.empty(WIDTH))
        self.v_bias = nn.Parameter(torch.empty(WIDTH))
        self.proj = nn.Linear(WIDTH, WIDTH)

    def forward(self, x):
        batch, tokens, _ = x.shape
        bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias), self.v_bias))
        qkv = F.linear(x, self.qkv.weight, bias)
        q, k, v = qkv.reshape(batch, tokens, 3, HEADS, -1).permute(2, 0, 3, 1, 4).unbind(0)
        # Scale q *before* the matmul, as in EVA (matters under autocast).
        q = q * ((WIDTH // HEADS) ** -0.5)
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(batch, tokens, WIDTH))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(WIDTH, eps=1e-6)
        self.attn = Attention()
        self.norm2 = nn.LayerNorm(WIDTH, eps=1e-6)
        self.mlp = Mlp(WIDTH, hidden_features=6144)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class EVAViTG(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=224, patch_size=14, embed_dim=WIDTH)
        self.cls_token = nn.Parameter(torch.empty(1, 1, WIDTH))
        self.pos_embed = nn.Parameter(torch.empty(1, 1 + self.patch_embed.num_patches, WIDTH))
        # LAVIS uses the penultimate features: no block 39, final norm or head.
        self.blocks = nn.ModuleList(Block() for _ in range(39))

    def forward(self, image):
        x = self.patch_embed(image)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return x
