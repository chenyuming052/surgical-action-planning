from __future__ import annotations

from typing import Callable, Dict

import numpy as np


def compute_event_ap(
    hazard_logits: np.ndarray,
    true_ttc: np.ndarray,
    censored: np.ndarray,
    horizon_sec: float,
) -> float:
    """Event-AP: average precision for change detection at a given horizon.

    Args:
        hazard_logits: [N, K] hazard logits over K bins
        true_ttc: [N] ground truth time-to-change in seconds
        censored: [N] boolean, True if right-censored
        horizon_sec: evaluation horizon in seconds (e.g. 5.0)

    Returns:
        Event-AP score (float).
    """
    from sklearn.metrics import average_precision_score

    N, K = hazard_logits.shape
    # Convert hazard logits to hazard probabilities
    h = 1.0 / (1.0 + np.exp(-hazard_logits))  # sigmoid, [N, K]
    # Survival function: S(t) = prod_{k=0}^{t} (1 - h_k)
    survival = np.cumprod(1.0 - h, axis=1)  # [N, K]
    # P(event within horizon) = 1 - S(horizon)
    # Find the bin index corresponding to horizon_sec
    # We use the last bin as the horizon proxy (logits span the full time range)
    # A more precise approach: horizon maps to a specific bin, but without bin_edges
    # we use all K bins and interpret the last survival value
    # P(event <= horizon) = 1 - S(K) when horizon covers all bins
    # For a specific horizon, we'd need bin_edges. Here we use all bins.
    p_event = 1.0 - survival[:, -1]  # [N]

    # Binary label: event occurs within horizon
    label = (true_ttc <= horizon_sec).astype(np.float64)

    # Filter: keep uncensored samples OR censored with ttc > horizon
    # Censored with ttc <= horizon are ambiguous (event status unknown within horizon)
    valid = (~censored) | (true_ttc > horizon_sec)
    if valid.sum() < 2:
        return float("nan")

    p_event = p_event[valid]
    label = label[valid]

    # Need both classes present
    if label.sum() == 0 or label.sum() == len(label):
        return float("nan")

    return float(average_precision_score(label, p_event))


def compute_event_auroc(
    hazard_logits: np.ndarray,
    true_ttc: np.ndarray,
    censored: np.ndarray,
    horizon_sec: float,
) -> float:
    """Event-AUROC: area under ROC for change detection at a given horizon.

    Args:
        hazard_logits: [N, K] hazard logits over K bins
        true_ttc: [N] ground truth time-to-change in seconds
        censored: [N] boolean
        horizon_sec: evaluation horizon in seconds

    Returns:
        Event-AUROC score (float).
    """
    from sklearn.metrics import roc_auc_score

    N, K = hazard_logits.shape
    h = 1.0 / (1.0 + np.exp(-hazard_logits))  # sigmoid
    survival = np.cumprod(1.0 - h, axis=1)  # [N, K]
    p_event = 1.0 - survival[:, -1]  # [N]

    label = (true_ttc <= horizon_sec).astype(np.float64)

    # Filter: uncensored OR censored with ttc > horizon
    valid = (~censored) | (true_ttc > horizon_sec)
    if valid.sum() < 2:
        return float("nan")

    p_event = p_event[valid]
    label = label[valid]

    if label.sum() == 0 or label.sum() == len(label):
        return float("nan")

    return float(roc_auc_score(label, p_event))


def compute_post_change_map(
    predicted: np.ndarray,
    true: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Post-change mAP: mean AP of instrument predictions at first change point.

    Args:
        predicted: [N, C] predicted instrument probabilities at change points
        true: [N, C] ground truth instrument binary labels
        mask: [N] validity mask

    Returns:
        Post-change mAP score (float).
    """
    from sklearn.metrics import average_precision_score

    mask = mask.astype(bool)
    if mask.sum() == 0:
        return float("nan")

    predicted = predicted[mask]
    true = true[mask]
    n_classes = true.shape[1]

    aps = []
    for c in range(n_classes):
        y_true_c = true[:, c]
        y_pred_c = predicted[:, c]
        # Skip classes with no positive or all positive
        if y_true_c.sum() == 0 or y_true_c.sum() == len(y_true_c):
            continue
        aps.append(average_precision_score(y_true_c, y_pred_c))

    if len(aps) == 0:
        return float("nan")

    return float(np.mean(aps))


def compute_dense_map(
    predictions: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Dense mAP: mean AP across all frames.

    Args:
        predictions: [N, C] predicted probabilities
        targets: [N, C] binary ground truth
        mask: [N] validity mask

    Returns:
        Dense mAP score (float).
    """
    from sklearn.metrics import average_precision_score

    mask = mask.astype(bool)
    if mask.sum() == 0:
        return float("nan")

    predictions = predictions[mask]
    targets = targets[mask]
    n_classes = targets.shape[1]

    aps = []
    for c in range(n_classes):
        y_true_c = targets[:, c]
        y_pred_c = predictions[:, c]
        if y_true_c.sum() == 0 or y_true_c.sum() == len(y_true_c):
            continue
        aps.append(average_precision_score(y_true_c, y_pred_c))

    if len(aps) == 0:
        return float("nan")

    return float(np.mean(aps))


def compute_change_conditioned_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    is_change_frame: np.ndarray,
    metric_fn: Callable,
) -> Dict[str, float]:
    """Compute metrics conditioned on change/non-change frames.

    Reports All/Change-only/Non-change-only splits.

    Args:
        predictions: [N, ...] model predictions
        targets: [N, ...] ground truth
        is_change_frame: [N] boolean mask for change frames
        metric_fn: callable(predictions, targets) -> float

    Returns:
        Dict with keys 'all', 'change_only', 'non_change_only'.
    """
    is_change = is_change_frame.astype(bool)
    results: Dict[str, float] = {}

    # All frames
    if len(predictions) > 0:
        results["all"] = float(metric_fn(predictions, targets))
    else:
        results["all"] = float("nan")

    # Change-only frames
    if is_change.sum() > 0:
        results["change_only"] = float(
            metric_fn(predictions[is_change], targets[is_change])
        )
    else:
        results["change_only"] = float("nan")

    # Non-change-only frames
    non_change = ~is_change
    if non_change.sum() > 0:
        results["non_change_only"] = float(
            metric_fn(predictions[non_change], targets[non_change])
        )
    else:
        results["non_change_only"] = float("nan")

    return results
