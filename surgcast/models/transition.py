from __future__ import annotations

import torch
import torch.nn as nn


class HorizonConditionedTransition(nn.Module):
    def __init__(self, hidden_dim: int = 512, horizon_embed_dim: int = 64, horizons=(1,3,5,10)):
        super().__init__()
        self.horizons = list(horizons)
        self.horizon_to_idx = {h: i for i, h in enumerate(self.horizons)}
        self.embed = nn.Embedding(len(self.horizons), horizon_embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + horizon_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )

    def forward(self, h_t: torch.Tensor, horizon: int):
        # h_t: [B, T, hidden_dim] or [B, hidden_dim]
        idx = torch.full((h_t.size(0),), self.horizon_to_idx[horizon], device=h_t.device, dtype=torch.long)
        e = self.embed(idx)  # [B, horizon_embed_dim]
        if h_t.ndim == 3:
            e = e.unsqueeze(1).expand(-1, h_t.size(1), -1)  # [B, T, horizon_embed_dim]
        out = self.mlp(torch.cat([h_t, e], dim=-1))
        pred_state, log_var = out.chunk(2, dim=-1)
        return pred_state, log_var
