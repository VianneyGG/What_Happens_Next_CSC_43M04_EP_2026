"""
UniFormer-B adapted for T=4 frames, trained from scratch.

Hybrid architecture: local windowed attention in early stages (large spatial
maps) switches to global full attention in late stages (small spatial maps).
The local inductive bias in early stages replaces the need for pretrained
weights — the model doesn't need to learn from scratch that nearby pixels
are related.

Forward:
    Input:  (B, T, C, H, W)
    Permute: (B, C, T, H, W)  — UniFormer uses channels-first 5D internally
    Stage 1: PatchEmbed(3→64, spatial 224→56) + 5× local blocks
    Stage 2: PatchEmbed(64→128, spatial 56→28) + 8× local blocks
    Stage 3: PatchEmbed(128→320, spatial 28→14, temporal 4→2) + 20× global blocks
    Stage 4: PatchEmbed(320→512, spatial 14→7, temporal 2→1) + 7× global blocks
    Global average pool → LayerNorm → Linear(512, num_classes)
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Stochastic depth (drop path)
# ---------------------------------------------------------------------------

def _drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor = torch.floor(random_tensor + keep_prob)
    return x / keep_prob * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _drop_path(x, self.drop_prob, self.training)


# ---------------------------------------------------------------------------
# Feed-forward network (shared by local and global blocks)
# ---------------------------------------------------------------------------

class FFN(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# ---------------------------------------------------------------------------
# 3D window partition / reverse (for local windowed attention)
# ---------------------------------------------------------------------------

def _window_partition(x: torch.Tensor, wt: int, wh: int, ww: int) -> torch.Tensor:
    """
    x: (B, C, T, H, W)
    returns: (B * num_windows, wt*wh*ww, C)
    """
    B, C, T, H, W = x.shape
    x = x.permute(0, 2, 3, 4, 1)                          # (B, T, H, W, C)
    x = x.view(B, T // wt, wt, H // wh, wh, W // ww, ww, C)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()   # (B, nT, nH, nW, wt, wh, ww, C)
    x = x.view(-1, wt * wh * ww, C)                       # (B*nT*nH*nW, wt*wh*ww, C)
    return x


def _window_reverse(
    windows: torch.Tensor, wt: int, wh: int, ww: int, B: int, T: int, H: int, W: int
) -> torch.Tensor:
    """
    windows: (B * num_windows, wt*wh*ww, C)
    returns: (B, C, T, H, W)
    """
    C = windows.shape[-1]
    nT, nH, nW = T // wt, H // wh, W // ww
    x = windows.view(B, nT, nH, nW, wt, wh, ww, C)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()   # (B, nT, wt, nH, wh, nW, ww, C)
    x = x.view(B, T, H, W, C).permute(0, 4, 1, 2, 3)     # (B, C, T, H, W)
    return x


# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------

class WindowedMHSA(nn.Module):
    """Multi-head self-attention within non-overlapping 3D local windows."""

    def __init__(self, dim: int, num_heads: int, window_size: Tuple[int, int, int]) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.wt, self.wh, self.ww = window_size
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        windows = _window_partition(x, self.wt, self.wh, self.ww)  # (Bw, N, C)
        Bw, N, _ = windows.shape

        qkv = self.qkv(windows).reshape(Bw, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each (Bw, heads, N, head_dim)

        out = F.scaled_dot_product_attention(q, k, v)     # (Bw, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(Bw, N, C)
        out = self.proj(out)

        return _window_reverse(out, self.wt, self.wh, self.ww, B, T, H, W)


class GlobalMHSA(nn.Module):
    """Standard ViT-style multi-head self-attention over the full flattened sequence."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each (B, heads, N, head_dim)

        out = F.scaled_dot_product_attention(q, k, v)     # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


# ---------------------------------------------------------------------------
# UniFormer blocks
# ---------------------------------------------------------------------------

class LocalUniFormerBlock(nn.Module):
    """
    Local block for stages 1-2.

    Attention uses BN3d on the 5D feature map (keeps spatial structure).
    FFN uses LN on the flattened sequence (standard transformer FFN).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        drop_path_rate: float,
        window_size: Tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm3d(dim)
        self.attn = WindowedMHSA(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FFN(dim, mlp_ratio)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        x = x + self.drop_path(self.attn(self.norm1(x)))

        B, C, T, H, W = x.shape
        flat = x.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
        flat = flat + self.drop_path(self.ffn(self.norm2(flat)))
        return flat.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)


class GlobalUniFormerBlock(nn.Module):
    """
    Global block for stages 3-4.

    Full sequence attention — feasible because spatial maps are small
    (392 tokens in stage 3, 49 in stage 4).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        drop_path_rate: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = GlobalMHSA(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FFN(dim, mlp_ratio)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        flat = x.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
        flat = flat + self.drop_path(self.attn(self.norm1(flat)))
        flat = flat + self.drop_path(self.ffn(self.norm2(flat)))
        return flat.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)


# ---------------------------------------------------------------------------
# Patch embedding (stem between stages)
# ---------------------------------------------------------------------------

class PatchEmbed3D(nn.Module):
    """Non-overlapping 3D patch embedding: Conv3d + BN3d + GELU."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: Tuple[int, int, int],
        stride: Tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=0)
        self.bn = nn.BatchNorm3d(out_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class UniFormerB(nn.Module):
    """
    UniFormer-B for action anticipation with T=4 frames, trained from scratch.

    dims:      channel depth per stage   [64, 128, 320, 512]
    depths:    number of blocks per stage [5, 8, 20, 7]
    num_heads: attention heads per stage  [2, 4, 10, 16]
    window_size: 3D local window for stages 1-2, default [4, 7, 7]
      — covers all T=4 frames (full temporal) in a 7×7 spatial window.
    """

    def __init__(
        self,
        num_classes: int,
        depths: List[int] = None,
        dims: List[int] = None,
        num_heads: List[int] = None,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.3,
        window_size: List[int] = None,
    ) -> None:
        super().__init__()

        if depths is None:
            depths = [5, 8, 20, 7]
        if dims is None:
            dims = [64, 128, 320, 512]
        if num_heads is None:
            num_heads = [2, 4, 10, 16]
        if window_size is None:
            window_size = [4, 7, 7]

        window = tuple(window_size)

        # Linearly increasing drop path rates across all blocks
        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # --- Patch embeddings (stride=kernel → non-overlapping patches) ---
        # Spatial: 224 → 56 → 28 → 14 → 7
        # Temporal: 4 → 4 → 4 → 2 → 1
        self.patch_embed1 = PatchEmbed3D(3,       dims[0], kernel_size=(1, 4, 4), stride=(1, 4, 4))
        self.patch_embed2 = PatchEmbed3D(dims[0], dims[1], kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.patch_embed3 = PatchEmbed3D(dims[1], dims[2], kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.patch_embed4 = PatchEmbed3D(dims[2], dims[3], kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # --- Stage 1: local, (B, 64, 4, 56, 56) ---
        idx = 0
        self.stage1 = nn.Sequential(*[
            LocalUniFormerBlock(dims[0], num_heads[0], mlp_ratio, dpr[idx + i], window)
            for i in range(depths[0])
        ])
        idx += depths[0]

        # --- Stage 2: local, (B, 128, 4, 28, 28) ---
        self.stage2 = nn.Sequential(*[
            LocalUniFormerBlock(dims[1], num_heads[1], mlp_ratio, dpr[idx + i], window)
            for i in range(depths[1])
        ])
        idx += depths[1]

        # --- Stage 3: global, (B, 320, 2, 14, 14) = 392 tokens ---
        self.stage3 = nn.Sequential(*[
            GlobalUniFormerBlock(dims[2], num_heads[2], mlp_ratio, dpr[idx + i])
            for i in range(depths[2])
        ])
        idx += depths[2]

        # --- Stage 4: global, (B, 512, 1, 7, 7) = 49 tokens ---
        self.stage4 = nn.Sequential(*[
            GlobalUniFormerBlock(dims[3], num_heads[3], mlp_ratio, dpr[idx + i])
            for i in range(depths[3])
        ])

        self.norm = nn.LayerNorm(dims[3])
        self.head = nn.Linear(dims[3], num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C, H, W)
        returns logits: (B, num_classes)
        """
        x = x.permute(0, 2, 1, 3, 4)   # (B, C, T, H, W) — channels-first for Conv3d

        x = self.patch_embed1(x)         # (B, 64,  4,  56, 56)
        x = self.stage1(x)

        x = self.patch_embed2(x)         # (B, 128, 4,  28, 28)
        x = self.stage2(x)

        x = self.patch_embed3(x)         # (B, 320, 2,  14, 14)
        x = self.stage3(x)

        x = self.patch_embed4(x)         # (B, 512, 1,   7,  7)
        x = self.stage4(x)

        x = x.flatten(2).mean(dim=2)     # global avg pool → (B, 512)
        x = self.norm(x)
        return self.head(x)              # (B, num_classes)
