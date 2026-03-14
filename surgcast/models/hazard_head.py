from __future__ import annotations

import torch
import torch.nn as nn


class StateAgeEncoder(nn.Module):
    """Encodes temporal age features for phase-gated hazard head.

    Input: [age_inst, age_phase, stable_run_length] -> 16-d embedding.
    """

    def __init__(self, input_dim: int = 3, embed_dim: int = 16):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, age_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            age_features: [B, T, 3] — [age_inst, age_phase, stable_run_length]

        Returns:
            age_embed: [B, T, 16]
        """
        return self.proj(age_features)  # [B, T, 16]


class DualHazardHead(nn.Module):
    """Phase-gated dual hazard head.

    Input: [h_t(512); a_t(64); d_t(2); age_embed(16)] = 594-d
    Shared trunk: Linear(594, 256) -> GELU
    2 base heads + 7x2 phase-specific residual experts.
    Soft routing: softmax(Linear(512, 7)(h_t)).
    Final: sigma(z_base + sum_p w_p * r_p) for both inst and group.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        action_dim: int = 64,
        source_dim: int = 2,
        age_dim: int = 16,
        trunk_dim: int = 256,
        num_bins: int = 20,
        num_phases: int = 7,
    ):
        super().__init__()
        input_dim = hidden_dim + action_dim + source_dim + age_dim  # 594
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, trunk_dim),
            nn.GELU(),
        )

        # Base heads
        self.inst_base = nn.Linear(trunk_dim, num_bins)
        self.group_base = nn.Linear(trunk_dim, num_bins)

        # Phase-specific residual experts: 7 phases x 2 heads (inst, group)
        self.inst_experts = nn.ModuleList([nn.Linear(trunk_dim, num_bins) for _ in range(num_phases)])
        self.group_experts = nn.ModuleList([nn.Linear(trunk_dim, num_bins) for _ in range(num_phases)])

        # Soft routing from encoder hidden state
        self.phase_router = nn.Linear(hidden_dim, num_phases)

    def forward(
        self,
        h_t: torch.Tensor,
        a_t: torch.Tensor,
        d_t: torch.Tensor,
        age_embed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_t: [B, T, 512] encoder hidden states
            a_t: [B, T, 64] action token
            d_t: [B, T, 2] source embedding
            age_embed: [B, T, 16] state age embedding

        Returns:
            hazard_inst: [B, T, 20] logits
            hazard_group: [B, T, 20] logits
        """
        # Concat inputs: [h_t(512); a_t(64); d_t(2); age_embed(16)] -> [B, T, 594]
        x = torch.cat([h_t, a_t, d_t, age_embed], dim=-1)

        # Shared trunk -> GELU -> z [B, T, 256]
        z = self.trunk(x)

        # Base predictions
        inst_base = self.inst_base(z)    # [B, T, 20]
        group_base = self.group_base(z)  # [B, T, 20]

        # Phase routing weights: softmax over 7 phases
        phase_weights = torch.softmax(self.phase_router(h_t), dim=-1)  # [B, T, 7]

        # Weighted sum of expert residuals
        inst_residual = torch.zeros_like(inst_base)   # [B, T, 20]
        group_residual = torch.zeros_like(group_base)  # [B, T, 20]

        for p in range(len(self.inst_experts)):
            w_p = phase_weights[..., p].unsqueeze(-1)  # [B, T, 1]
            inst_residual = inst_residual + w_p * self.inst_experts[p](z)
            group_residual = group_residual + w_p * self.group_experts[p](z)

        # Final: base + weighted residual
        hazard_inst = inst_base + inst_residual    # [B, T, 20]
        hazard_group = group_base + group_residual  # [B, T, 20]

        return hazard_inst, hazard_group
