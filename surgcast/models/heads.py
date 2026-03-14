from __future__ import annotations

import torch
import torch.nn as nn


class MultiTaskHeads(nn.Module):
    """Multi-task heads with anatomy-injected CVS.

    CVS head: Linear(519, 6) with anatomy injection [h_t(512); s_src(2); sg(anat)(5)].
    2 ordinal thresholds per CVS criterion: P(>=1), P(>=2).
    Anatomy head: Linear(512, 5).
    All other heads same signature.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        group_dim: int = 18,
        instrument_dim: int = 6,
        phase_dim: int = 7,
        anatomy_dim: int = 5,
        source_dim: int = 2,
        cvs_ordinal_dim: int = 6,
    ):
        super().__init__()
        self.triplet_group = nn.Linear(hidden_dim, group_dim)
        self.instrument = nn.Linear(hidden_dim, instrument_dim)
        self.phase = nn.Linear(hidden_dim, phase_dim)
        self.anatomy = nn.Linear(hidden_dim, anatomy_dim)

        # CVS with anatomy injection: [h_t; s_src; sg(anat)] = 512+2+5 = 519
        self.cvs = nn.Linear(hidden_dim + source_dim + anatomy_dim, cvs_ordinal_dim)

    def forward(
        self,
        h: torch.Tensor,
        source_embed: torch.Tensor | None = None,
        anatomy_sg: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            h: [B, T, 512] encoder hidden states
            source_embed: [B, T, 2] dataset source embedding
            anatomy_sg: [B, T, 5] stop-gradient anatomy predictions

        Returns:
            Dict with keys: triplet_group, instrument, phase, anatomy, [cvs]
        """
        out = {
            "triplet_group": self.triplet_group(h),
            "instrument": self.instrument(h),
            "phase": self.phase(h),
            "anatomy": self.anatomy(h),
        }

        # CVS with anatomy injection (stop-gradient anatomy predictions)
        if source_embed is not None:
            if anatomy_sg is None:
                # Compute stop-gradient anatomy from own predictions
                anatomy_sg = torch.sigmoid(out["anatomy"]).detach()  # [B, T, 5]
            cvs_input = torch.cat([h, source_embed, anatomy_sg], dim=-1)  # [B, T, 519]
            out["cvs"] = self.cvs(cvs_input)

        return out
