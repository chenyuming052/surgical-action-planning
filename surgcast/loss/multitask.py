from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_bce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, pos_weight=None) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none', pos_weight=pos_weight)
    loss = loss * mask.float()
    denom = mask.float().sum().clamp_min(1.0)
    return loss.sum() / denom


def masked_ce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    num_classes = logits.size(-1)
    safe_targets = targets.long().clamp(0, num_classes - 1)
    loss = F.cross_entropy(logits, safe_targets, reduction='none')
    loss = loss * mask.float()
    denom = mask.float().sum().clamp_min(1.0)
    return loss.sum() / denom


def ordinal_bce_cvs(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """CVS ordinal regression loss.

    Treats each CVS criterion as ordinal: P(>=1), P(>=2).
    Uses binary cross-entropy on cumulative probabilities.

    Args:
        logits: [B, T, 6] raw logits (3 criteria x 2 thresholds)
        targets: [B, T, 6] ordinal binary targets
        mask: [B, T] visibility mask

    Returns:
        Scalar ordinal BCE loss.
    """
    # [B, T, 6] element-wise BCE
    loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
    # Expand mask [B, T] -> [B, T, 1] and broadcast over the 6 ordinal channels
    loss = loss * mask.unsqueeze(-1).float()
    denom = mask.float().sum().clamp_min(1.0) * logits.size(-1)
    return loss.sum() / denom


def ranking_loss(
    predicted_ttc: torch.Tensor,
    true_ttc: torch.Tensor,
    censored: torch.Tensor,
) -> torch.Tensor:
    """Pairwise C-index ranking loss for TTC prediction.

    Constructs valid concordant pairs (respecting censoring) and computes
    differentiable ranking loss.

    Args:
        predicted_ttc: [B] predicted time-to-change
        true_ttc: [B] ground truth time-to-change
        censored: [B] boolean, True if right-censored

    Returns:
        Scalar ranking loss.
    """
    margin = 1.0
    uncensored_mask = ~censored  # [B]

    # Pairwise differences: [B, B]
    # diff_true[i, j] = true_ttc[i] - true_ttc[j]
    diff_true = true_ttc.unsqueeze(1) - true_ttc.unsqueeze(0)
    # diff_pred[i, j] = predicted_ttc[i] - predicted_ttc[j]
    diff_pred = predicted_ttc.unsqueeze(1) - predicted_ttc.unsqueeze(0)

    # Valid pair (i, j): true_ttc[i] < true_ttc[j] AND i is uncensored
    # We want pred[i] < pred[j], so penalise when pred[i] - pred[j] + margin > 0
    valid_mask = (diff_true < 0) & uncensored_mask.unsqueeze(1)

    pair_losses = torch.relu(diff_pred + margin) * valid_mask.float()
    num_valid = valid_mask.float().sum().clamp_min(1.0)
    return pair_losses.sum() / num_valid


def consistency_cvs_anatomy(
    cvs_c1_prob: torch.Tensor,
    duct_prob: torch.Tensor,
    artery_prob: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Safety consistency loss between CVS criterion 1 and anatomy predictions.

    Enforces that CVS C1 (two structures identified) is consistent with
    cystic duct and cystic artery visibility predictions.

    Args:
        cvs_c1_prob: [B, T] CVS criterion 1 probability
        duct_prob: [B, T] cystic duct detection probability
        artery_prob: [B, T] cystic artery detection probability
        mask: [B, T] visibility mask

    Returns:
        Scalar consistency loss.
    """
    # Soft consistency: C1 should approximate min(duct, artery)
    target_consistency = torch.min(duct_prob, artery_prob)
    loss = (cvs_c1_prob - target_consistency).pow(2) * mask.float()
    denom = mask.float().sum().clamp_min(1.0)
    return loss.sum() / denom


def next_action_loss(
    delta_add_logits: torch.Tensor,
    delta_remove_logits: torch.Tensor,
    phase_next_logits: torch.Tensor,
    group_next_logits: torch.Tensor,
    target_delta_add: torch.Tensor,
    target_delta_remove: torch.Tensor,
    target_phase_next: torch.Tensor,
    target_group_next: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Combined loss for NextActionHead delta-state predictions.

    Sum of masked BCE for delta_add/delta_remove + masked CE for phase_next/group_next.
    Only computed at frames with upcoming change events (mask).

    Args:
        delta_add_logits: [B, T, 6]
        delta_remove_logits: [B, T, 6]
        phase_next_logits: [B, T, 7]
        group_next_logits: [B, T, G]
        target_delta_add: [B, T, 6]
        target_delta_remove: [B, T, 6]
        target_phase_next: [B, T] integer class
        target_group_next: [B, T] integer class
        mask: [B, T] frames with valid change targets

    Returns:
        Scalar next-action loss.
    """
    # Expand mask [B, T] -> [B, T, 1] for BCE losses on [B, T, C] tensors
    mask_expanded_add = mask.unsqueeze(-1).expand_as(delta_add_logits)
    l_add = masked_bce(delta_add_logits, target_delta_add, mask_expanded_add)
    mask_expanded_rm = mask.unsqueeze(-1).expand_as(delta_remove_logits)
    l_remove = masked_bce(delta_remove_logits, target_delta_remove, mask_expanded_rm)
    # CE for phase and group: flatten [B, T, C] -> [B*T, C] since masked_ce expects 2D logits
    BT = mask.numel()
    flat_mask = mask.reshape(BT)
    l_phase = masked_ce(
        phase_next_logits.reshape(BT, -1), target_phase_next.reshape(BT), flat_mask,
    )
    l_group = masked_ce(
        group_next_logits.reshape(BT, -1), target_group_next.reshape(BT), flat_mask,
    )
    return l_add + l_remove + l_phase + l_group


def heteroscedastic_nll(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Heteroscedastic Gaussian NLL for dynamics loss.

    L = 0.5 * [log_var + (mu - target)^2 / exp(log_var)], masked.

    Args:
        mu: [B, T, D] predicted mean
        log_var: [B, T, D] predicted log-variance
        target: [B, T, D] target values
        mask: [B, T] validity mask

    Returns:
        Scalar heteroscedastic NLL loss.
    """
    # L = 0.5 * [log_var + (mu - target)^2 / exp(log_var)]
    precision = torch.exp(-log_var)
    loss = 0.5 * (log_var + (mu - target).pow(2) * precision)
    # Expand mask [B, T] -> [B, T, 1] for broadcasting over D
    loss = loss * mask.unsqueeze(-1).float()
    denom = mask.float().sum().clamp_min(1.0) * mu.size(-1)
    return loss.sum() / denom
