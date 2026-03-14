from __future__ import annotations

from .surgcast import SurgCastModel
from .temporal_encoder import CausalTemporalTransformer
from .heads import MultiTaskHeads
from .hazard_head import DualHazardHead, StateAgeEncoder
from .backbone import BackboneSpec, DINOV3_VITB16, LEMONFM
from .prior import StructuredPrior
from .action_encoder import ActionTokenEncoder
from .next_action_head import NextActionHead
from .event_dyn import EventDyn, ActionConditionedTransition


def build_model(config: dict) -> SurgCastModel:
    """Construct a SurgCast model from a merged config dict.

    Reads the 'model' section of the config to determine kwargs.
    dynamics_version (default "B") selects event-conditioned vs fixed-horizon dynamics.

    Args:
        config: Full merged config dict (must contain model-related keys
                under 'backbone', 'encoder', 'heads', 'hazard', etc.).

    Returns:
        Instantiated SurgCastModel.
    """
    backbone = config.get("backbone", {})
    encoder = config.get("encoder", {})
    transition = config.get("transition", {})
    heads = config.get("heads", {})
    hazard = config.get("hazard", {})
    ae = config.get("action_encoder", {})
    pgh = config.get("phase_gated_hazard", {})

    return SurgCastModel(
        input_dim=backbone.get("feature_dim", 768),
        hidden_dim=encoder.get("input_proj_dim", 512),
        encoder_layers=encoder.get("transformer_layers", 6),
        encoder_heads=encoder.get("num_heads", 8),
        dropout=encoder.get("dropout", 0.1),
        instrument_dim=heads.get("instrument_dim", 6),
        phase_dim=heads.get("phase_dim", 7),
        triplet_vocab_size=ae.get("triplet_vocab_size", 100),
        action_dim=ae.get("token_dim", 64),
        group_dim=heads.get("triplet_group_dim", 18),
        anatomy_dim=heads.get("anatomy_dim", 5),
        cvs_ordinal_dim=heads.get("cvs_ordinal_dim", 6),
        hazard_trunk_dim=hazard.get("shared_hidden_dim", 256),
        hazard_num_bins=hazard.get("num_bins", 20),
        num_phases=pgh.get("num_phases", 7),
        dynamics_version=config.get("dynamics_version", "B"),
        horizons=tuple(transition.get("horizons_sec", [1, 3, 5, 10])),
        horizon_embed_dim=transition.get("horizon_embed_dim", 64),
    )
