from __future__ import annotations

import torch
import torch.nn as nn


class DualHazardHead(nn.Module):
    def __init__(self, hidden_dim: int = 512, sigma_dim: int = 4, trunk_dim: int = 256, num_bins: int = 20):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim + sigma_dim, trunk_dim),
            nn.GELU(),
        )
        self.inst = nn.Linear(trunk_dim, num_bins)
        self.group = nn.Linear(trunk_dim, num_bins)

    def forward(self, h_t: torch.Tensor, sigma_agg: torch.Tensor):
        z = self.trunk(torch.cat([h_t, sigma_agg], dim=-1))
        return self.inst(z), self.group(z)
