from __future__ import annotations

import numpy as np


def compute_ttc_mae(
    hazard_logits: np.ndarray,
    true_ttc: np.ndarray,
    censored: np.ndarray,
    bin_edges: np.ndarray,
) -> float:
    """TTC MAE: mean absolute error of predicted time-to-change.

    Args:
        hazard_logits: [N, K] hazard logits over K bins
        true_ttc: [N] ground truth time-to-change in seconds
        censored: [N] boolean, True if right-censored
        bin_edges: [K+1] bin edge values in seconds

    Returns:
        MAE in seconds (float).
    """
    uncensored = ~censored.astype(bool)
    if uncensored.sum() == 0:
        return float("nan")

    predicted_ttc = compute_expected_ttc(hazard_logits, bin_edges)
    mae = np.mean(np.abs(predicted_ttc[uncensored] - true_ttc[uncensored]))
    return float(mae)


def compute_expected_ttc(
    hazard_logits: np.ndarray,
    bin_edges: np.ndarray,
) -> np.ndarray:
    """Compute expected TTC from discrete hazard distribution.

    Args:
        hazard_logits: [N, K] hazard logits over K bins
        bin_edges: [K+1] bin edge values in seconds

    Returns:
        expected_ttc: [N] expected time-to-change in seconds.
    """
    N, K = hazard_logits.shape
    # h_k = sigmoid(logits)
    h = 1.0 / (1.0 + np.exp(-hazard_logits))  # [N, K]

    # S_k = cumulative product of (1 - h_k)
    # S_{k-1}: survival just before bin k. S_0 = 1.
    one_minus_h = 1.0 - h  # [N, K]
    # S_{k-1} for k=0..K-1: [1, S_0, S_0*S_1, ..., S_0*...*S_{K-2}]
    survival_prev = np.ones((N, K), dtype=np.float64)
    if K > 1:
        survival_prev[:, 1:] = np.cumprod(one_minus_h[:, :-1], axis=1)

    # P(event in bin k) = S_{k-1} * h_k
    p_event_k = survival_prev * h  # [N, K]

    # bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0  # [K]

    # E[TTC] = sum_k bin_center_k * P(event in bin k)
    expected = np.sum(p_event_k * bin_centers[np.newaxis, :], axis=1)  # [N]

    return expected


def compute_c_index(
    predicted_ttc: np.ndarray,
    true_ttc: np.ndarray,
    censored: np.ndarray,
) -> float:
    """Concordance index for TTC predictions.

    Args:
        predicted_ttc: [N] predicted time-to-change
        true_ttc: [N] ground truth time-to-change
        censored: [N] boolean

    Returns:
        C-index (float in [0, 1]).
    """
    censored = censored.astype(bool)
    N = len(predicted_ttc)
    concordant = 0.0
    total = 0.0

    for i in range(N):
        if censored[i]:
            continue
        for j in range(N):
            if i == j:
                continue
            # Valid pair: true_ttc[i] < true_ttc[j] and i is uncensored
            if true_ttc[i] < true_ttc[j]:
                total += 1.0
                if predicted_ttc[i] < predicted_ttc[j]:
                    concordant += 1.0
                elif predicted_ttc[i] == predicted_ttc[j]:
                    concordant += 0.5

    if total == 0.0:
        return float("nan")

    return float(concordant / total)


def compute_brier_score(
    hazard_logits: np.ndarray,
    true_ttc: np.ndarray,
    eval_time_sec: float,
    bin_edges: np.ndarray,
) -> float:
    """Brier score for TTC at a specific evaluation time.

    Args:
        hazard_logits: [N, K] hazard logits over K bins
        true_ttc: [N] ground truth time-to-change
        eval_time_sec: evaluation time point in seconds
        bin_edges: [K+1] bin edge values

    Returns:
        Brier score (float).
    """
    N, K = hazard_logits.shape
    h = 1.0 / (1.0 + np.exp(-hazard_logits))  # sigmoid [N, K]
    survival = np.cumprod(1.0 - h, axis=1)  # [N, K]

    # Find the bin index for eval_time_sec
    # Use the last bin whose left edge <= eval_time_sec
    bin_idx = np.searchsorted(bin_edges[1:], eval_time_sec, side="left")
    bin_idx = min(bin_idx, K - 1)

    # P(event <= eval_time) = 1 - S(bin_idx)
    p_event = 1.0 - survival[:, bin_idx]  # [N]

    # Binary outcome: true_ttc <= eval_time
    outcome = (true_ttc <= eval_time_sec).astype(np.float64)

    brier = np.mean((p_event - outcome) ** 2)
    return float(brier)


def compute_hazard_calibration(
    hazard_logits: np.ndarray,
    true_ttc: np.ndarray,
    censored: np.ndarray,
) -> float:
    """Calibration metric for hazard predictions.

    Args:
        hazard_logits: [N, K] hazard logits over K bins
        true_ttc: [N] ground truth time-to-change
        censored: [N] boolean

    Returns:
        Calibration error (float).
    """
    censored = censored.astype(bool)
    N, K = hazard_logits.shape

    # Compute overall predicted risk = 1 - S(K) (probability of event within full horizon)
    h = 1.0 / (1.0 + np.exp(-hazard_logits))  # sigmoid
    survival = np.cumprod(1.0 - h, axis=1)  # [N, K]
    predicted_risk = 1.0 - survival[:, -1]  # [N]

    # Observed event: uncensored samples had an event
    observed_event = (~censored).astype(np.float64)

    # Bin by predicted risk percentile (10 bins)
    n_bins = 10
    if N < n_bins:
        n_bins = max(N, 1)

    # Sort by predicted risk and split into bins
    sorted_indices = np.argsort(predicted_risk)
    bin_size = N // n_bins
    remainder = N % n_bins

    calibration_errors = []
    start = 0
    for b in range(n_bins):
        end = start + bin_size + (1 if b < remainder else 0)
        if end <= start:
            continue
        idx = sorted_indices[start:end]
        mean_pred = np.mean(predicted_risk[idx])
        mean_obs = np.mean(observed_event[idx])
        calibration_errors.append(abs(mean_pred - mean_obs))
        start = end

    if len(calibration_errors) == 0:
        return float("nan")

    return float(np.mean(calibration_errors))
