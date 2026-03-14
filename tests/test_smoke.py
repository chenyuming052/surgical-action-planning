#!/usr/bin/env python3
"""Smoke test: synthetic forward + backward pass through SurgCastModel."""
from __future__ import annotations

import sys
import torch

from surgcast.models.surgcast import SurgCastModel
from surgcast.loss.hazard_loss import discrete_time_hazard_nll
from surgcast.loss.multitask import masked_bce, masked_ce
from surgcast.utils.seed import set_seed


def test_model_forward_backward():
    set_seed(42)
    B, T, D = 4, 16, 768

    model = SurgCastModel(
        input_dim=D,
        hidden_dim=512,
        encoder_layers=2,  # fewer layers for speed
        encoder_heads=8,
        dropout=0.0,
        max_seq_len=64,
        dynamics_version="B",
    )

    features = torch.randn(B, T, D)
    out = model(features)

    # Check output shapes
    assert out["triplet_group"].shape == (B, T, 18), f"triplet_group: {out['triplet_group'].shape}"
    assert out["instrument"].shape == (B, T, 6), f"instrument: {out['instrument'].shape}"
    assert out["phase"].shape == (B, T, 7), f"phase: {out['phase'].shape}"
    assert out["anatomy"].shape == (B, T, 5), f"anatomy: {out['anatomy'].shape}"
    assert out["hazard_inst"].shape == (B, T, 20), f"hazard_inst: {out['hazard_inst'].shape}"
    assert out["hazard_group"].shape == (B, T, 20), f"hazard_group: {out['hazard_group'].shape}"
    assert out["action_token"].shape == (B, T, 64), f"action_token: {out['action_token'].shape}"
    assert out["mu_plus"].shape == (B, T, 512), f"mu_plus: {out['mu_plus'].shape}"
    assert out["log_var"].shape == (B, T, 512), f"log_var: {out['log_var'].shape}"
    assert out["delta_add"].shape == (B, T, 6), f"delta_add: {out['delta_add'].shape}"
    assert out["delta_remove"].shape == (B, T, 6), f"delta_remove: {out['delta_remove'].shape}"
    assert out["phase_next"].shape == (B, T, 7), f"phase_next: {out['phase_next'].shape}"
    assert out["group_next"].shape == (B, T, 18), f"group_next: {out['group_next'].shape}"

    # Test with source_embed for CVS head
    source_embed = torch.randn(B, T, 2)
    out_cvs = model(features, source_embed=source_embed)
    assert "cvs" in out_cvs
    assert out_cvs["cvs"].shape == (B, T, 6), f"cvs: {out_cvs['cvs'].shape}"

    print("[OK] Forward shapes verified")

    # Compute losses
    BT = B * T
    phase_targets = torch.randint(0, 7, (BT,))
    phase_mask = torch.ones(BT)
    phase_logits = out["phase"].reshape(BT, 7)
    loss_phase = masked_ce(phase_logits, phase_targets, phase_mask)

    inst_targets = torch.randint(0, 2, (BT, 6)).float()
    inst_mask = torch.ones(BT, 6)
    inst_logits = out["instrument"].reshape(BT, 6)
    loss_inst = masked_bce(inst_logits, inst_targets, inst_mask)

    hazard_bins = torch.randint(0, 20, (BT,))
    censored = torch.zeros(BT, dtype=torch.bool)
    hazard_logits = out["hazard_inst"].reshape(BT, 20)
    loss_hazard = discrete_time_hazard_nll(hazard_logits, hazard_bins, censored)

    total = loss_phase + loss_inst + loss_hazard
    total.backward()

    # Check gradients exist
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for _ in model.parameters())
    assert grad_count > 0, "No gradients computed"
    print(f"[OK] Backward pass: {grad_count}/{total_params} params have gradients")

    # Check model size
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[OK] Model: {n_trainable:,} trainable / {n_params:,} total parameters")

    return True


def test_hazard_loss_vectorized():
    """Verify vectorized hazard loss matches loop-based reference."""
    set_seed(42)
    B, K = 32, 20
    logits = torch.randn(B, K)
    target_bin = torch.randint(0, K, (B,))
    censored = torch.rand(B) > 0.7

    # Reference loop implementation
    hazards = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    ref_loss = torch.zeros(B)
    for i in range(B):
        k = int(target_bin[i].item())
        if bool(censored[i]):
            ref_loss[i] = -(torch.log(1 - hazards[i])).sum()
        else:
            if k > 0:
                ref_loss[i] = -(torch.log(1 - hazards[i, :k]).sum() + torch.log(hazards[i, k]))
            else:
                ref_loss[i] = -torch.log(hazards[i, k])
    ref = ref_loss.mean()

    # Vectorized
    vec = discrete_time_hazard_nll(logits, target_bin, censored)

    assert torch.allclose(ref, vec, atol=1e-5), f"Hazard loss mismatch: ref={ref.item():.6f}, vec={vec.item():.6f}"
    print(f"[OK] Hazard loss: ref={ref.item():.6f}, vec={vec.item():.6f}")


def test_masked_ce_with_ignore():
    """Verify masked_ce handles -1 targets safely."""
    logits = torch.randn(8, 7)
    targets = torch.tensor([0, 1, -1, 3, -1, 5, 6, 2])
    mask = torch.tensor([1, 1, 0, 1, 0, 1, 1, 1], dtype=torch.float32)
    loss = masked_ce(logits, targets, mask)
    assert torch.isfinite(loss), f"masked_ce produced non-finite loss: {loss}"
    print(f"[OK] masked_ce with -1 targets: loss={loss.item():.6f}")


if __name__ == "__main__":
    try:
        test_hazard_loss_vectorized()
        test_masked_ce_with_ignore()
        test_model_forward_backward()
        print("\n=== All smoke tests passed ===")
    except Exception as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        raise SystemExit(1)
