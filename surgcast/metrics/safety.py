from __future__ import annotations

from typing import Dict

import numpy as np


def compute_cvs_criterion_auc(
    predicted: np.ndarray,
    true: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, float]:
    """Per-criterion AUC for CVS predictions.

    Args:
        predicted: [N, 6] predicted CVS ordinal probabilities (3 criteria x 2 thresholds)
        true: [N, 6] ground truth ordinal targets
        mask: [N] visibility mask

    Returns:
        Dict mapping criterion name -> AUC score.
    """
    from sklearn.metrics import roc_auc_score

    mask = mask.astype(bool)
    if mask.sum() < 2:
        return {name: float("nan") for name in [
            "C1_t1", "C1_t2", "C2_t1", "C2_t2", "C3_t1", "C3_t2"
        ]}

    predicted = predicted[mask]
    true = true[mask]

    col_names = ["C1_t1", "C1_t2", "C2_t1", "C2_t2", "C3_t1", "C3_t2"]
    results: Dict[str, float] = {}

    for i, name in enumerate(col_names):
        y_true_i = true[:, i]
        y_pred_i = predicted[:, i]
        # Need both classes for AUC
        if y_true_i.sum() == 0 or y_true_i.sum() == len(y_true_i):
            results[name] = float("nan")
        else:
            results[name] = float(roc_auc_score(y_true_i, y_pred_i))

    return results


def compute_cvs_mae(
    predicted: np.ndarray,
    true: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Mean absolute error for CVS score predictions.

    Args:
        predicted: [N, 3] predicted CVS scores (3 criteria)
        true: [N, 3] ground truth CVS scores
        mask: [N] visibility mask

    Returns:
        CVS MAE (float).
    """
    mask = mask.astype(bool)
    if mask.sum() == 0:
        return float("nan")

    predicted = predicted[mask]
    true = true[mask]
    mae = np.mean(np.abs(predicted - true))
    return float(mae)


def compute_clipping_detection_rate(
    hazard_logits: np.ndarray,
    clipping_events: np.ndarray,
    window_sec: float,
) -> float:
    """Detection rate: fraction of clipping events preceded by hazard alert.

    Args:
        hazard_logits: [N, K] hazard logits over K bins
        clipping_events: [M] frame indices of clipping events
        window_sec: look-back window in seconds

    Returns:
        Detection rate (float in [0, 1]).
    """
    if len(clipping_events) == 0:
        return float("nan")

    N, K = hazard_logits.shape
    # Compute P(event) at each frame
    h = 1.0 / (1.0 + np.exp(-hazard_logits))  # sigmoid
    survival = np.cumprod(1.0 - h, axis=1)  # [N, K]
    p_event = 1.0 - survival[:, -1]  # [N]

    # Approximate fps=1 (bin edges in seconds)
    window_frames = int(window_sec)
    detected = 0
    for frame in clipping_events:
        start = max(0, int(frame) - window_frames)
        end = int(frame) + 1  # inclusive of the frame itself
        end = min(end, N)
        start = min(start, end)
        if start < end and np.any(p_event[start:end] > 0.5):
            detected += 1

    return float(detected / len(clipping_events))


def compute_clipping_false_alarm_rate(
    hazard_logits: np.ndarray,
    clipping_events: np.ndarray,
    total_frames: int,
    threshold: float,
) -> float:
    """False alarm rate for clipping detection.

    Args:
        hazard_logits: [N, K] hazard logits
        clipping_events: [M] frame indices of clipping events
        total_frames: total number of frames
        threshold: hazard probability threshold for alarm

    Returns:
        False alarm rate (float).
    """
    N, K = hazard_logits.shape
    # Compute P(event) at each frame
    h = 1.0 / (1.0 + np.exp(-hazard_logits))  # sigmoid
    survival = np.cumprod(1.0 - h, axis=1)  # [N, K]
    p_event = 1.0 - survival[:, -1]  # [N]

    # Treat clipping_events as exact event frames
    event_set = set(int(f) for f in clipping_events)

    # Count alarms at non-event frames
    alarm_frames = np.where(p_event > threshold)[0]
    false_alarms = sum(1 for f in alarm_frames if f not in event_set)

    # Total non-event frames
    total_non_event = total_frames - len(event_set)
    if total_non_event <= 0:
        return float("nan")

    return float(false_alarms / total_non_event)


def compute_cvs_mae_at_clipping(
    predicted: np.ndarray,
    true: np.ndarray,
    clipping_frames: np.ndarray,
    window_sec: float,
) -> float:
    """CVS MAE evaluated within a window around clipping events.

    Args:
        predicted: [N, 3] predicted CVS scores
        true: [N, 3] ground truth CVS scores
        clipping_frames: [M] frame indices of clipping events
        window_sec: window size in seconds around clipping

    Returns:
        CVS MAE at clipping (float).
    """
    if len(clipping_frames) == 0:
        return float("nan")

    N = predicted.shape[0]
    # Approximate fps=1
    window_frames = int(window_sec)

    # Collect frame indices within window_sec of any clipping event
    relevant = set()
    for frame in clipping_frames:
        start = max(0, int(frame) - window_frames)
        end = min(N, int(frame) + window_frames + 1)
        for f in range(start, end):
            relevant.add(f)

    if len(relevant) == 0:
        return float("nan")

    idx = np.array(sorted(relevant))
    mae = np.mean(np.abs(predicted[idx] - true[idx]))
    return float(mae)
