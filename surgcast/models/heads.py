from __future__ import annotations

import torch
import torch.nn as nn


class MultiTaskHeads(nn.Module):
    def __init__(self, hidden_dim: int = 512, group_dim: int = 18, instrument_dim: int = 6, phase_dim: int = 7, cvs_ordinal_dim: int = 6, anatomy_dim: int = 5):
        super().__init__()
        self.triplet_group = nn.Linear(hidden_dim, group_dim)
        self.instrument = nn.Linear(hidden_dim, instrument_dim)
        self.phase = nn.Linear(hidden_dim, phase_dim)
        self.cvs = nn.Linear(hidden_dim + 2, cvs_ordinal_dim)
        self.anatomy = nn.Linear(hidden_dim, anatomy_dim)

    def forward(self, h, source_embed=None):
        out = {
            "triplet_group": self.triplet_group(h),
            "instrument": self.instrument(h),
            "phase": self.phase(h),
            "anatomy": self.anatomy(h),
        }
        if source_embed is not None:
            out["cvs"] = self.cvs(torch.cat([h, source_embed], dim=-1))
        return out
