"""Multi-scale 1D encoder that emits local signal tokens."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _feature_norm(kind: str, channels: int) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm1d(channels)
    if kind == "group":
        groups = next(
            value for value in (8, 4, 2, 1) if channels % value == 0
        )
        return nn.GroupNorm(groups, channels)
    raise ValueError("normalization must be 'batch' or 'group'.")


class _ResidualDilatedTokenBlock(nn.Module):
    """Mix pooled signal tokens over progressively longer time spans."""

    def __init__(self, token_dim: int, dilation: int, dropout: float) -> None:
        super().__init__()
        if dilation <= 0:
            raise ValueError("dilation must be positive.")
        self.norm = nn.LayerNorm(token_dim)
        self.depthwise = nn.Conv1d(
            token_dim,
            token_dim,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=token_dim,
            bias=False,
        )
        self.channel_mixer = nn.Sequential(
            nn.Conv1d(token_dim, token_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # A small initial residual scale preserves the stable CNN solution at
        # the start of continual training while allowing longer-range mixing.
        self.residual_scale = nn.Parameter(torch.tensor(1.0e-2))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        features = self.norm(tokens).transpose(1, 2)
        features = self.depthwise(features)
        features = self.channel_mixer(features).transpose(1, 2)
        return tokens + self.residual_scale * features


class MultiScaleTokenEncoder(nn.Module):
    """Encode vibration windows into a stable number of local tokens.

    Each convolution branch observes a different temporal receptive field.
    Adaptive pooling fixes the token count, allowing datasets with different
    sampling rates or window lengths to share downstream semantic modules.
    """

    def __init__(
        self,
        input_channels: int = 1,
        token_dim: int = 256,
        branch_dim: int = 32,
        num_tokens: int = 32,
        kernels: tuple[int, ...] = (7, 15, 31),
        temporal_dilations: tuple[int, ...] = (),
        temporal_dropout: float = 0.0,
        normalization: str = "batch",
    ) -> None:
        super().__init__()
        if input_channels <= 0 or token_dim <= 0 or branch_dim <= 0 or num_tokens <= 0:
            raise ValueError("Encoder dimensions must be positive.")
        if not kernels or any(kernel <= 0 or kernel % 2 == 0 for kernel in kernels):
            raise ValueError("kernels must contain positive odd values.")
        if any(dilation <= 0 for dilation in temporal_dilations):
            raise ValueError("temporal_dilations must contain positive integers.")
        if not 0.0 <= temporal_dropout < 1.0:
            raise ValueError("temporal_dropout must be in [0, 1).")
        if normalization not in {"batch", "group"}:
            raise ValueError("normalization must be 'batch' or 'group'.")
        self.input_channels = int(input_channels)
        self.token_dim = int(token_dim)
        self.num_tokens = int(num_tokens)

        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        input_channels,
                        branch_dim,
                        kernel_size=kernel,
                        stride=4,
                        padding=kernel // 2,
                        bias=False,
                    ),
                    _feature_norm(normalization, branch_dim),
                    nn.GELU(),
                    nn.Conv1d(
                        branch_dim,
                        branch_dim,
                        kernel_size=5,
                        stride=2,
                        padding=2,
                        bias=False,
                    ),
                    _feature_norm(normalization, branch_dim),
                    nn.GELU(),
                )
                for kernel in kernels
            ]
        )
        merged_dim = branch_dim * len(kernels)
        self.token_projection = nn.Sequential(
            nn.Conv1d(merged_dim, token_dim, kernel_size=1, bias=False),
            _feature_norm(normalization, token_dim),
            nn.GELU(),
        )
        self.temporal_mixer = nn.ModuleList(
            [
                _ResidualDilatedTokenBlock(
                    token_dim,
                    dilation=dilation,
                    dropout=temporal_dropout,
                )
                for dilation in temporal_dilations
            ]
        )
        self.token_norm = nn.LayerNorm(token_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"Expected [B,L] or [B,C,L], got {tuple(x.shape)}")
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} channels, got {int(x.shape[1])}"
            )
        features = torch.cat([branch(x) for branch in self.branches], dim=1)
        features = self.token_projection(features)
        features = F.adaptive_avg_pool1d(features, self.num_tokens)
        tokens = features.transpose(1, 2)
        for block in self.temporal_mixer:
            tokens = block(tokens)
        return self.token_norm(tokens)
