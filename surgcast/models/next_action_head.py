from __future__ import annotations

import torch
import torch.nn as nn


class NextActionHead(nn.Module):
    """Delta-state post-change prediction head.

    Input: [h_t(512); a_t(64)] = 576-d
    Trunk: Linear(576, 256) -> GELU
    Instrument branch: delta_add, delta_remove (each Linear(256, 6))
    Phase branch: Linear(256, 7)
    Triplet-group branch: Linear(256, G)
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        action_dim: int = 64,
        trunk_dim: int = 256,
        instrument_dim: int = 6,
        phase_dim: int = 7,
        group_dim: int = 18,
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, trunk_dim),
            nn.GELU(),
        )
        self.delta_add = nn.Linear(trunk_dim, instrument_dim)
        self.delta_remove = nn.Linear(trunk_dim, instrument_dim)
        self.phase_next = nn.Linear(trunk_dim, phase_dim)
        self.group_next = nn.Linear(trunk_dim, group_dim)

    def forward(
        self,
        h_t: torch.Tensor,
        a_t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            h_t: [B, T, 512] encoder hidden states
            a_t: [B, T, 64] action token

        Returns:
            Dict with keys: delta_add [B,T,6], delta_remove [B,T,6],
                           phase_next [B,T,7], group_next [B,T,G]
        """
        # Concat [h_t; a_t] -> trunk -> GELU
        x = torch.cat([h_t, a_t], dim=-1)  # [B, T, 576]
        z = self.trunk(x)  # [B, T, 256]

        return {
            "delta_add": self.delta_add(z),       # [B, T, 6]
            "delta_remove": self.delta_remove(z), # [B, T, 6]
            "phase_next": self.phase_next(z),     # [B, T, 7]
            "group_next": self.group_next(z),     # [B, T, G]
        }
