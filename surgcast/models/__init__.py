from __future__ import annotations

# V1 core modules
from .surgcast import SurgCastModel, SurgCastModelV2
from .temporal_encoder import CausalTemporalTransformer
from .transition import HorizonConditionedTransition
from .heads import MultiTaskHeads
from .hazard_head import DualHazardHead
from .backbone import BackboneSpec, DINOV3_VITB16, LEMONFM
from .prior import StructuredPrior

# V2 extension modules
from .heads import MultiTaskHeadsV2
from .hazard_head import StateAgeEncoder, PhaseGatedDualHazardHead
from .action_encoder import ActionTokenEncoder
from .next_action_head import NextActionHead
from .event_dyn import EventDyn, ActionConditionedTransition


def build_model(config: dict) -> SurgCastModel | SurgCastModelV2:
    """Construct a SurgCast model from a merged config dict.

    Reads the 'model' section of the config to determine version and kwargs.
    Defaults to V1 unless config contains version: "v2".

    Args:
        config: Full merged config dict (must contain model-related keys
                under 'backbone', 'encoder', 'transition', 'heads', 'hazard',
                and optionally V2 keys).

    Returns:
        Instantiated model.
    """
    version = config.get("version", "v1")

    backbone = config.get("backbone", {})
    encoder = config.get("encoder", {})
    transition = config.get("transition", {})
    heads = config.get("heads", {})
    hazard = config.get("hazard", {})

    common_kwargs = dict(
        input_dim=backbone.get("feature_dim", 768),
        hidden_dim=encoder.get("input_proj_dim", 512),
        encoder_layers=encoder.get("transformer_layers", 6),
        encoder_heads=encoder.get("num_heads", 8),
        dropout=encoder.get("dropout", 0.1),
    )

    if version == "v2":
        ae = config.get("action_encoder", {})
        ed = config.get("event_dyn", {})
        nah = config.get("next_action_head", {})
        pgh = config.get("phase_gated_hazard", {})

        return SurgCastModelV2(
            **common_kwargs,
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
    else:
        return SurgCastModel(
            **common_kwargs,
            horizons=tuple(transition.get("horizons_sec", [1, 3, 5, 10])),
            horizon_embed_dim=transition.get("horizon_embed_dim", 64),
            group_dim=heads.get("triplet_group_dim", 18),
            instrument_dim=heads.get("instrument_dim", 6),
            phase_dim=heads.get("phase_dim", 7),
            cvs_ordinal_dim=heads.get("cvs_ordinal_dim", 6),
            anatomy_dim=heads.get("anatomy_dim", 5),
            hazard_trunk_dim=hazard.get("shared_hidden_dim", 256),
            hazard_num_bins=hazard.get("num_bins", 20),
        )
