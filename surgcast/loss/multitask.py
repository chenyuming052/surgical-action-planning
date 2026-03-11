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
