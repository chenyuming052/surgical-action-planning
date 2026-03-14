from __future__ import annotations

from typing import List, Tuple

import numpy as np


def extract_instrument_changes(
    labels: np.ndarray,
    min_gap: int = 1,
) -> List[int]:
    """Extract frame indices where the instrument set changes.

    Args:
        labels: [T, 6] binary instrument labels per frame
        min_gap: minimum gap in frames between consecutive changes

    Returns:
        List of frame indices where instrument set changes.
    """
    if labels.shape[0] < 2:
        return []
    # Compare consecutive frames: any element differs means a change
    diffs = np.any(labels[1:] != labels[:-1], axis=1)  # [T-1]
    change_indices = (np.where(diffs)[0] + 1).tolist()  # +1 because diff[i] corresponds to frame i+1
    return debounce_changes(change_indices, min_gap)


def extract_group_changes(
    labels: np.ndarray,
    min_gap: int = 1,
) -> List[int]:
    """Extract frame indices where the triplet group changes.

    Args:
        labels: [T, G] binary triplet-group labels per frame
        min_gap: minimum gap in frames between consecutive changes

    Returns:
        List of frame indices where triplet group changes.
    """
    if labels.shape[0] < 2:
        return []
    diffs = np.any(labels[1:] != labels[:-1], axis=1)  # [T-1]
    change_indices = (np.where(diffs)[0] + 1).tolist()
    return debounce_changes(change_indices, min_gap)


def extract_phase_changes(
    labels: np.ndarray,
) -> List[int]:
    """Extract frame indices where the surgical phase changes.

    Args:
        labels: [T] integer phase labels per frame

    Returns:
        List of frame indices where phase changes.
    """
    if labels.shape[0] < 2:
        return []
    diffs = labels[1:] != labels[:-1]  # [T-1]
    change_indices = (np.where(diffs)[0] + 1).tolist()
    return change_indices


def compute_ttc_targets(
    change_points: List[int],
    total_frames: int,
    bin_edges: np.ndarray,
    max_horizon: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute TTC targets and censoring indicators from change points.

    Args:
        change_points: list of frame indices where changes occur
        total_frames: total number of frames in the video
        bin_edges: [K+1] hazard bin edges in seconds
        max_horizon: maximum TTC horizon in seconds

    Returns:
        target_bins: [T] integer bin indices for each frame
        censored: [T] boolean, True if no change within max_horizon
    """
    K = len(bin_edges) - 1  # number of bins
    target_bins = np.zeros(total_frames, dtype=np.int64)
    censored = np.ones(total_frames, dtype=bool)

    # Convert change_points to a sorted numpy array for efficient lookup
    cp_arr = np.array(change_points, dtype=np.int64)

    for t in range(total_frames):
        # Find the next change point strictly after frame t
        idx = np.searchsorted(cp_arr, t, side='right')
        if idx < len(cp_arr):
            ttc = float(cp_arr[idx] - t)  # frames, assuming 1 fps = seconds
            if ttc <= max_horizon:
                # Bin the TTC: find which bin it falls into
                bin_idx = int(np.searchsorted(bin_edges[1:], ttc))
                bin_idx = min(bin_idx, K - 1)  # clamp to [0, K-1]
                target_bins[t] = bin_idx
                censored[t] = False
                continue
        # No change within max_horizon: censored
        target_bins[t] = K - 1  # assign to last bin (will be masked by censored flag)
        censored[t] = True

    return target_bins, censored


def debounce_changes(
    change_points: List[int],
    min_gap: int,
) -> List[int]:
    """Remove change points that are too close together.

    Args:
        change_points: sorted list of frame indices
        min_gap: minimum gap in frames between consecutive changes

    Returns:
        Filtered list of change points.
    """
    if not change_points:
        return []
    kept = [change_points[0]]
    for cp in change_points[1:]:
        if cp - kept[-1] >= min_gap:
            kept.append(cp)
    return kept
