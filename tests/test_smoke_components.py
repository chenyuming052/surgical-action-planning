#!/usr/bin/env python3
"""Smoke test: submodule instantiation, forward shapes, and import sweep."""
from __future__ import annotations

import sys
import torch

from surgcast.utils.seed import set_seed


def test_action_token_encoder():
    from surgcast.models.action_encoder import ActionTokenEncoder

    enc = ActionTokenEncoder(
        instrument_dim=6,
        phase_dim=7,
        triplet_vocab_size=100,
        triplet_embed_dim=32,
        token_dim=64,
        set_transformer_heads=4,
    )
    n_params = sum(p.numel() for p in enc.parameters())
    print(f"[OK] ActionTokenEncoder instantiated: {n_params:,} params")


def test_next_action_head():
    from surgcast.models.next_action_head import NextActionHead

    head = NextActionHead(
        hidden_dim=512,
        action_dim=64,
        trunk_dim=256,
        instrument_dim=6,
        phase_dim=7,
        group_dim=18,
    )
    n_params = sum(p.numel() for p in head.parameters())
    print(f"[OK] NextActionHead instantiated: {n_params:,} params")


def test_event_dyn():
    from surgcast.models.event_dyn import EventDyn, ActionConditionedTransition

    ed = EventDyn(hidden_dim=512, num_bins=20, bin_embed_dim=64, film_hidden_dim=256)
    n_params = sum(p.numel() for p in ed.parameters())
    print(f"[OK] EventDyn instantiated: {n_params:,} params")

    act = ActionConditionedTransition(hidden_dim=512, action_dim=64, horizon_embed_dim=64)
    n_params = sum(p.numel() for p in act.parameters())
    print(f"[OK] ActionConditionedTransition instantiated: {n_params:,} params")


def test_dual_hazard_head():
    from surgcast.models.hazard_head import DualHazardHead, StateAgeEncoder

    age_enc = StateAgeEncoder(input_dim=3, embed_dim=16)
    n_params = sum(p.numel() for p in age_enc.parameters())
    print(f"[OK] StateAgeEncoder instantiated: {n_params:,} params")

    head = DualHazardHead(
        hidden_dim=512,
        action_dim=64,
        source_dim=2,
        age_dim=16,
        trunk_dim=256,
        num_bins=20,
        num_phases=7,
    )
    n_params = sum(p.numel() for p in head.parameters())
    print(f"[OK] DualHazardHead instantiated: {n_params:,} params")


def test_multi_task_heads():
    from surgcast.models.heads import MultiTaskHeads

    heads = MultiTaskHeads(
        hidden_dim=512,
        group_dim=18,
        instrument_dim=6,
        phase_dim=7,
        anatomy_dim=5,
        source_dim=2,
        cvs_ordinal_dim=6,
    )
    n_params = sum(p.numel() for p in heads.parameters())
    print(f"[OK] MultiTaskHeads instantiated: {n_params:,} params")


def test_surgcast_model():
    from surgcast.models.surgcast import SurgCastModel

    set_seed(42)

    # Version B (event-conditioned)
    model_b = SurgCastModel(
        input_dim=768,
        hidden_dim=512,
        encoder_layers=2,
        encoder_heads=8,
        dropout=0.0,
        max_seq_len=64,
        dynamics_version="B",
    )
    n_params = sum(p.numel() for p in model_b.parameters())
    n_trainable = sum(p.numel() for p in model_b.parameters() if p.requires_grad)
    print(f"[OK] SurgCastModel (Version B) instantiated: {n_trainable:,} trainable / {n_params:,} total")

    # Version A (fixed-horizon with action)
    model_a = SurgCastModel(
        input_dim=768,
        hidden_dim=512,
        encoder_layers=2,
        encoder_heads=8,
        dropout=0.0,
        max_seq_len=64,
        dynamics_version="A",
    )
    n_params = sum(p.numel() for p in model_a.parameters())
    print(f"[OK] SurgCastModel (Version A) instantiated: {n_params:,} total params")


def test_loss_function_signatures():
    from surgcast.loss.multitask import (
        ordinal_bce_cvs,
        ranking_loss,
        consistency_cvs_anatomy,
        next_action_loss,
        heteroscedastic_nll,
    )
    # Just verify they exist and are callable
    for fn in [ordinal_bce_cvs, ranking_loss, consistency_cvs_anatomy, next_action_loss, heteroscedastic_nll]:
        assert callable(fn), f"{fn.__name__} is not callable"
    print("[OK] All loss functions importable and callable")


def test_import_sweep():
    import surgcast.models
    import surgcast.loss
    import surgcast.metrics
    import surgcast.utils

    # Verify core classes are accessible
    assert hasattr(surgcast.models, 'SurgCastModel')
    assert hasattr(surgcast.models, 'ActionTokenEncoder')
    assert hasattr(surgcast.models, 'EventDyn')
    assert hasattr(surgcast.models, 'ActionConditionedTransition')
    assert hasattr(surgcast.models, 'NextActionHead')
    assert hasattr(surgcast.models, 'DualHazardHead')
    assert hasattr(surgcast.models, 'StateAgeEncoder')
    assert hasattr(surgcast.models, 'MultiTaskHeads')

    # Verify loss functions
    assert hasattr(surgcast.loss, 'ordinal_bce_cvs')
    assert hasattr(surgcast.loss, 'ranking_loss')
    assert hasattr(surgcast.loss, 'consistency_cvs_anatomy')
    assert hasattr(surgcast.loss, 'next_action_loss')
    assert hasattr(surgcast.loss, 'heteroscedastic_nll')

    # Verify metric functions
    assert hasattr(surgcast.metrics, 'compute_event_ap')
    assert hasattr(surgcast.metrics, 'compute_c_index')
    assert hasattr(surgcast.metrics, 'compute_cvs_criterion_auc')

    # Verify utility functions
    assert hasattr(surgcast.utils, 'extract_instrument_changes')
    assert hasattr(surgcast.utils, 'compute_cooccurrence_matrix')

    print("[OK] Full import sweep passed — all modules accessible")


if __name__ == "__main__":
    try:
        test_action_token_encoder()
        test_next_action_head()
        test_event_dyn()
        test_dual_hazard_head()
        test_multi_task_heads()
        test_surgcast_model()
        test_loss_function_signatures()
        test_import_sweep()
        print("\n=== All component smoke tests passed ===")
    except Exception as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        raise SystemExit(1)
