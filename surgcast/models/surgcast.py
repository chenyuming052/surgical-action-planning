from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .temporal_encoder import CausalTemporalTransformer
from .heads import MultiTaskHeads
from .hazard_head import DualHazardHead, StateAgeEncoder
from .action_encoder import ActionTokenEncoder
from .next_action_head import NextActionHead
from .event_dyn import EventDyn, ActionConditionedTransition


class SurgCastModel(nn.Module):
    """Event-conditioned SurgCast model.

    Wires: CausalTemporalTransformer -> ActionTokenEncoder -> NextActionHead
           -> DualHazardHead -> EventDyn -> MultiTaskHeads

    dynamics_version param switches between "A" (fixed-horizon) and "B" (event-conditioned).
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        encoder_layers: int = 6,
        encoder_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 64,
        # Action encoder
        instrument_dim: int = 6,
        phase_dim: int = 7,
        triplet_vocab_size: int = 100,
        action_dim: int = 64,
        # Heads
        group_dim: int = 18,
        anatomy_dim: int = 5,
        cvs_ordinal_dim: int = 6,
        # Hazard
        hazard_trunk_dim: int = 256,
        hazard_num_bins: int = 20,
        num_phases: int = 7,
        # Dynamics
        dynamics_version: str = "B",
        horizons: tuple = (1, 3, 5, 10),
        horizon_embed_dim: int = 64,
    ):
        super().__init__()
        self.dynamics_version = dynamics_version

        self.encoder = CausalTemporalTransformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            layers=encoder_layers,
            heads=encoder_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

        self.action_encoder = ActionTokenEncoder(
            instrument_dim=instrument_dim,
            phase_dim=phase_dim,
            triplet_vocab_size=triplet_vocab_size,
            token_dim=action_dim,
        )

        self.next_action_head = NextActionHead(
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            trunk_dim=hazard_trunk_dim,
            instrument_dim=instrument_dim,
            phase_dim=phase_dim,
            group_dim=group_dim,
        )

        self.state_age_encoder = StateAgeEncoder()

        self.hazard_head = DualHazardHead(
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            trunk_dim=hazard_trunk_dim,
            num_bins=hazard_num_bins,
            num_phases=num_phases,
        )

        self.heads = MultiTaskHeads(
            hidden_dim=hidden_dim,
            group_dim=group_dim,
            instrument_dim=instrument_dim,
            phase_dim=phase_dim,
            anatomy_dim=anatomy_dim,
            cvs_ordinal_dim=cvs_ordinal_dim,
        )

        # Dynamics module: Version B (event-conditioned) or A (fixed-horizon)
        if dynamics_version == "B":
            self.event_dyn = EventDyn(
                hidden_dim=hidden_dim,
                num_bins=hazard_num_bins,
            )
        else:
            self.action_transition = ActionConditionedTransition(
                hidden_dim=hidden_dim,
                action_dim=action_dim,
                horizon_embed_dim=horizon_embed_dim,
                horizons=horizons,
            )

    def forward(
        self,
        features: torch.Tensor,
        source_embed: Optional[torch.Tensor] = None,
        age_features: Optional[torch.Tensor] = None,
        instrument_labels: Optional[torch.Tensor] = None,
        phase_labels: Optional[torch.Tensor] = None,
        triplet_indices: Optional[torch.Tensor] = None,
        triplet_mask: Optional[torch.Tensor] = None,
        has_action_labels: Optional[torch.Tensor] = None,
        rho: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [B, T, input_dim] frozen backbone features
            source_embed: [B, T, 2] dataset source embedding
            age_features: [B, T, 3] state age features
            instrument_labels: [B, T, 6] GT instrument labels (teacher forcing)
            phase_labels: [B, T, 7] GT phase labels (teacher forcing)
            triplet_indices: [B, T, max_triplets] active triplet indices
            triplet_mask: [B, T, max_triplets] valid triplet mask
            has_action_labels: [B] boolean mask for videos with action labels
            rho: teacher forcing ratio

        Returns:
            Dict of output tensors
        """
        B, T, _ = features.shape
        device = features.device

        # 1. Encode features
        h = self.encoder(features)  # [B, T, 512]

        # 2. Task heads (need anatomy for stop-grad injection into CVS)
        task_out = self.heads(h, source_embed=source_embed)  # computes anatomy_sg internally

        # 3. Build action token via teacher forcing
        # Use predicted instrument/phase from task heads
        pred_inst = task_out["instrument"]  # [B, T, 6]
        pred_phase = task_out["phase"]      # [B, T, 7]

        # For missing GT, fall back to predicted (with sigmoid/softmax to get probabilities)
        inst_gt = instrument_labels if instrument_labels is not None else torch.sigmoid(pred_inst).detach()
        phase_gt = phase_labels if phase_labels is not None else torch.softmax(pred_phase, dim=-1).detach()

        # Defaults for triplet inputs if not provided
        if triplet_indices is None:
            triplet_indices = torch.zeros(B, T, 1, dtype=torch.long, device=device)
        if triplet_mask is None:
            triplet_mask = torch.zeros(B, T, 1, dtype=torch.bool, device=device)
        if has_action_labels is None:
            has_action_labels = torch.ones(B, dtype=torch.bool, device=device)

        action_token = self.action_encoder(
            instrument_labels=inst_gt,
            phase_labels=phase_gt,
            triplet_indices=triplet_indices,
            triplet_mask=triplet_mask,
            has_action_labels=has_action_labels,
            predicted_instrument=torch.sigmoid(pred_inst).detach(),
            predicted_phase=torch.softmax(pred_phase, dim=-1).detach(),
            rho=rho,
        )  # [B, T, 64]

        # 4. Next action head
        next_out = self.next_action_head(h, action_token)

        # 5. Age embedding
        if age_features is not None:
            age_embed = self.state_age_encoder(age_features)  # [B, T, 16]
        else:
            age_embed = torch.zeros(B, T, 16, device=device)

        # 6. Source embed default
        if source_embed is None:
            source_embed = torch.zeros(B, T, 2, device=device)

        # 7. Hazard head
        hazard_inst, hazard_group = self.hazard_head(
            h, action_token, source_embed, age_embed
        )  # each [B, T, 20]

        # 8. Dynamics
        dyn_outputs: Dict[str, torch.Tensor] = {}
        if self.dynamics_version == "B":
            # Get predicted bin from hazard_inst via argmax
            tau_bin = hazard_inst.argmax(dim=-1)  # [B, T]
            mu_plus, log_var = self.event_dyn(h, tau_bin)
            dyn_outputs["mu_plus"] = mu_plus
            dyn_outputs["log_var"] = log_var
        else:
            # Version A: fixed-horizon transitions
            for horizon in self.action_transition.horizons:
                pred_state, log_var = self.action_transition(h, action_token, horizon)
                dyn_outputs[f"transition_{horizon}s"] = pred_state
                dyn_outputs[f"log_var_{horizon}s"] = log_var

        # 9. Collect all outputs
        out: Dict[str, torch.Tensor] = {**task_out, **dyn_outputs}
        out["hazard_inst"] = hazard_inst
        out["hazard_group"] = hazard_group
        out["action_token"] = action_token
        out.update(next_out)  # Bug fix: no prefix, keys are delta_add etc.

        return out
