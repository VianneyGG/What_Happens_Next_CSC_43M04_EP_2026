"""
TSM (Temporal Shift Module) wrapping ResNet-50.

Temporal reasoning at zero extra parameters: before each residual block,
1/8 of channels are shifted to the previous frame, 1/8 to the next frame,
and the remaining 6/8 stay in place. The shifted feature map is then processed
by a standard 2D residual conv, giving implicit temporal context.

Forward:
    Input:  (B, T, C, H, W)
    Reshape: (B*T, C, H, W)
    TSM shift before each residual block (on the merged batch)
    ResNet-50 backbone (all 2D conv)  -> (B*T, 2048)
    Reshape: (B, T, 2048)
    Temporal average pool             -> (B, 2048)
    Dropout + Linear                  -> (B, num_classes)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class TemporalShift(nn.Module):
    """Wraps a residual block; shifts channels along the time axis before forwarding."""

    def __init__(self, net: nn.Module, n_segment: int = 4, fold_div: int = 8) -> None:
        super().__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._shift(x, self.n_segment, self.fold_div)
        return self.net(x)

    @staticmethod
    def _shift(x: torch.Tensor, n_segment: int, fold_div: int) -> torch.Tensor:
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        out = torch.zeros_like(x)

        # frame i receives features from frame i-1 (backward shift)
        out[:, 1:, :fold] = x[:, :-1, :fold]
        # frame i receives features from frame i+1 (forward shift)
        out[:, :-1, fold : 2 * fold] = x[:, 1:, fold : 2 * fold]
        # remaining channels are unchanged
        out[:, :, 2 * fold :] = x[:, :, 2 * fold :]

        return out.view(nt, c, h, w)


def _wrap_stage(stage: nn.Sequential, n_segment: int) -> nn.Sequential:
    """Wrap every residual block in a ResNet stage with TemporalShift."""
    return nn.Sequential(*[
        TemporalShift(block, n_segment=n_segment, fold_div=8)
        for block in stage.children()
    ])


class TSMResNet50(nn.Module):
    def __init__(self, num_classes: int, n_segment: int = 4, dropout: float = 0.5) -> None:
        super().__init__()
        self.n_segment = n_segment

        backbone = models.resnet50(weights=None)

        # Stem — no temporal wrapping needed here
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        # Residual stages wrapped with TSM
        self.layer1 = _wrap_stage(backbone.layer1, n_segment)
        self.layer2 = _wrap_stage(backbone.layer2, n_segment)
        self.layer3 = _wrap_stage(backbone.layer3, n_segment)
        self.layer4 = _wrap_stage(backbone.layer4, n_segment)

        self.avgpool = backbone.avgpool  # (B*T, 2048, 1, 1)

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C, H, W)
        returns logits: (B, num_classes)
        """
        B, T, C, H, W = x.shape
        assert T == self.n_segment, (
            f"TSM expects n_segment={self.n_segment} frames, got T={T}. "
            "Keep dataset.num_frames == model.n_segment."
        )

        x = x.view(B * T, C, H, W)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)        # (B*T, 2048, 1, 1)
        x = torch.flatten(x, 1)   # (B*T, 2048)

        x = x.view(B, T, -1).mean(dim=1)  # temporal avg pool -> (B, 2048)

        return self.head(x)
