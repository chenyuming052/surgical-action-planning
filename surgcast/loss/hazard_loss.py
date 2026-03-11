from __future__ import annotations

import torch


def discrete_time_hazard_nll(logits: torch.Tensor, target_bin: torch.Tensor, censored: torch.Tensor) -> torch.Tensor:
    """Vectorized discrete-time hazard NLL.

    Args:
        logits: [B, K] raw logits for K hazard bins.
        target_bin: [B] integer bin indices in [0, K-1].
        censored: [B] boolean, True if observation is right-censored.

    Returns:
        Scalar mean NLL.
    """
    hazards = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    B, K = hazards.shape

    # Build a mask of bins before the target: [B, K] where mask[i,k] = (k < target_bin[i])
    bin_idx = torch.arange(K, device=logits.device).unsqueeze(0)  # [1, K]
    target_expanded = target_bin.unsqueeze(1)  # [B, 1]
    before_mask = bin_idx < target_expanded  # [B, K]

    # Survival contribution: sum of log(1 - h_k) for k < target_bin
    log_survival = (torch.log(1 - hazards) * before_mask.float()).sum(dim=1)  # [B]

    # Event contribution: log(h_{target_bin}) at the target bin (only for uncensored)
    target_clamped = target_bin.clamp(0, K - 1).long()
    log_event = torch.log(hazards.gather(1, target_clamped.unsqueeze(1)).squeeze(1))  # [B]

    # For censored: loss = -sum log(1-h_k) for all bins
    # For uncensored: loss = -(sum log(1-h_k) for k<target + log(h_target))
    censored_survival = torch.log(1 - hazards).sum(dim=1)  # [B]

    loss = torch.where(
        censored,
        -censored_survival,
        -(log_survival + log_event),
    )
    return loss.mean()
