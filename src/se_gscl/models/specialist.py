"""Fault/condition decoupled lightweight signal specialist."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .multiscale_encoder import MultiScaleTokenEncoder


@dataclass
class SpecialistOutput:
    signal_tokens: torch.Tensor
    fault_tokens: torch.Tensor
    condition_tokens: torch.Tensor
    fault_embedding: torch.Tensor
    condition_embedding: torch.Tensor
    domain_logits: torch.Tensor | None
    condition_values: torch.Tensor | None


class _TokenBranch(nn.Module):
    def __init__(self, token_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim * 2),
            nn.GELU(),
            nn.Linear(token_dim * 2, token_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.net(tokens)


class SEGSCLSpecialist(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        token_dim: int = 256,
        branch_dim: int = 32,
        num_tokens: int = 32,
        kernels: tuple[int, ...] = (7, 15, 31),
        num_domains: int | None = None,
        condition_dim: int = 0,
    ) -> None:
        super().__init__()
        self.encoder = MultiScaleTokenEncoder(
            input_channels=input_channels,
            token_dim=token_dim,
            branch_dim=branch_dim,
            num_tokens=num_tokens,
            kernels=kernels,
        )
        self.fault_branch = _TokenBranch(token_dim)
        self.condition_branch = _TokenBranch(token_dim)
        self.domain_head = (
            nn.Linear(token_dim, int(num_domains))
            if num_domains is not None and num_domains > 0
            else None
        )
        self.condition_head = (
            nn.Linear(token_dim, int(condition_dim)) if condition_dim > 0 else None
        )

    def forward(self, x: torch.Tensor) -> SpecialistOutput:
        signal_tokens = self.encoder(x)
        fault_tokens = self.fault_branch(signal_tokens)
        condition_tokens = self.condition_branch(signal_tokens)
        fault_embedding = F.normalize(fault_tokens.mean(dim=1), dim=-1)
        condition_embedding = F.normalize(condition_tokens.mean(dim=1), dim=-1)
        return SpecialistOutput(
            signal_tokens=signal_tokens,
            fault_tokens=fault_tokens,
            condition_tokens=condition_tokens,
            fault_embedding=fault_embedding,
            condition_embedding=condition_embedding,
            domain_logits=(
                None if self.domain_head is None else self.domain_head(condition_embedding)
            ),
            condition_values=(
                None
                if self.condition_head is None
                else self.condition_head(condition_embedding)
            ),
        )
