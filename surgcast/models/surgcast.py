from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .temporal_encoder import CausalTemporalTransformer
from .transition import HorizonConditionedTransition
from .heads import MultiTaskHeads
from .hazard_head import DualHazardHead


class SurgCastModel(nn.Module):
    """Top-level SurgCast model.

    Pipeline: features -> CausalTemporalTransformer -> HorizonConditionedTransition (x4)
              -> MultiTaskHeads + DualHazardHead (with sigma_agg from transition variance).
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        encoder_layers: int = 6,
        encoder_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 64,
        horizons: tuple = (1, 3, 5, 10),
        horizon_embed_dim: int = 64,
        group_dim: int = 18,
        instrument_dim: int = 6,
        phase_dim: int = 7,
        cvs_ordinal_dim: int = 6,
        anatomy_dim: int = 5,
        hazard_trunk_dim: int = 256,
        hazard_num_bins: int = 20,
    ):
        super().__init__()
        self.horizons = list(horizons)
        sigma_dim = len(self.horizons)  # 4

        self.encoder = CausalTemporalTransformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            layers=encoder_layers,
            heads=encoder_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

        self.transition = HorizonConditionedTransition(
            hidden_dim=hidden_dim,
            horizon_embed_dim=horizon_embed_dim,
            horizons=tuple(self.horizons),
        )

        self.heads = MultiTaskHeads(
            hidden_dim=hidden_dim,
            group_dim=group_dim,
            instrument_dim=instrument_dim,
            phase_dim=phase_dim,
            cvs_ordinal_dim=cvs_ordinal_dim,
            anatomy_dim=anatomy_dim,
        )

        self.hazard_head = DualHazardHead(
            hidden_dim=hidden_dim,
            sigma_dim=sigma_dim,
            trunk_dim=hazard_trunk_dim,
            num_bins=hazard_num_bins,
        )

    def forward(
        self,
        features: torch.Tensor,
        source_embed: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # features: [B, T, input_dim]
        h = self.encoder(features)  # [B, T, hidden_dim]

        # Transition for each horizon -> collect predicted states and log-variances
        transition_outputs = {}
        log_vars: List[torch.Tensor] = []
        for horizon in self.horizons:
            pred_state, log_var = self.transition(h, horizon)
            transition_outputs[f"transition_{horizon}s"] = pred_state
            log_vars.append(log_var)  # each [B, T, hidden_dim]

        # sigma_agg: per-horizon uncertainty summary -> [B, T, num_horizons]
        sigma_agg = torch.stack(
            [lv.exp().mean(dim=-1).sqrt() for lv in log_vars], dim=-1
        )

        # Task heads on encoder output
        task_out = self.heads(h, source_embed=source_embed)

        # Hazard heads on encoder output + sigma_agg
        hazard_inst, hazard_group = self.hazard_head(h, sigma_agg)

        out = {**task_out, **transition_outputs}
        out["hazard_inst"] = hazard_inst
        out["hazard_group"] = hazard_group
        out["sigma_agg"] = sigma_agg
        return out
