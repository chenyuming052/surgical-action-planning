"""Shared preprocessing utilities for NPZ generation.

Common patterns used across all preprocess scripts:
- Registry loading and filtering
- NPZ output writing with standard keys
- Label array construction and validation
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def load_registry_for_dataset(
    registry_path: str | Path,
    dataset_name: str,
) -> List[Dict[str, Any]]:
    """Load registry and filter to a specific dataset.

    Args:
        registry_path: Path to registry.json
        dataset_name: One of "cholect50", "cholec80", "cholec80_cvs", "endoscapes", "heichole"

    Returns:
        List of registry entries for the specified dataset.
    """
    with open(registry_path) as f:
        registry = json.load(f)
    return [r for r in registry if r.get("dataset") == dataset_name]


def save_npz(
    output_dir: str | Path,
    canonical_id: str,
    arrays: Dict[str, np.ndarray],
    overwrite: bool = False,
) -> Path:
    """Save label arrays as an NPZ file following the dataset contract.

    Args:
        output_dir: Directory to save NPZ files
        canonical_id: Video canonical ID (becomes filename)
        arrays: Dict of array name -> numpy array
        overwrite: Whether to overwrite existing files

    Returns:
        Path to the saved NPZ file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{canonical_id}.npz"

    if out_path.exists() and not overwrite:
        return out_path

    np.savez(out_path, **arrays)
    return out_path


def build_phase_array(
    phase_labels: List[int],
    total_frames: int,
    fps_ratio: float = 1.0,
) -> np.ndarray:
    """Build phase label array at target FPS.

    Args:
        phase_labels: Per-frame phase labels at source FPS
        total_frames: Expected number of frames at target FPS
        fps_ratio: source_fps / target_fps

    Returns:
        phase: [T] int array of phase labels
    """
    if fps_ratio == 1.0:
        return np.array(phase_labels[:total_frames], dtype=np.int64)
    # Subsample
    indices = np.round(np.arange(total_frames) * fps_ratio).astype(int)
    indices = np.clip(indices, 0, len(phase_labels) - 1)
    return np.array(phase_labels, dtype=np.int64)[indices]


def build_binary_array(
    labels: List[List[int]] | np.ndarray,
    num_classes: int,
    total_frames: int,
) -> np.ndarray:
    """Build binary multi-label array.

    Args:
        labels: Per-frame list of active class indices, or [T, C] array
        num_classes: Number of classes
        total_frames: Number of frames

    Returns:
        [T, C] float32 binary array
    """
    if isinstance(labels, np.ndarray) and labels.ndim == 2:
        return labels[:total_frames].astype(np.float32)
    arr = np.zeros((total_frames, num_classes), dtype=np.float32)
    for t, active in enumerate(labels[:total_frames]):
        for cls_idx in active:
            if 0 <= cls_idx < num_classes:
                arr[t, cls_idx] = 1.0
    return arr


def build_visibility_mask(
    observed_frames: Optional[np.ndarray],
    total_frames: int,
    num_classes: Optional[int] = None,
) -> np.ndarray:
    """Build visibility mask.

    Args:
        observed_frames: Boolean array of observed frames, or None for all-observed
        total_frames: Number of frames
        num_classes: If provided, creates [T, C] mask; otherwise [T] mask

    Returns:
        Float32 mask array (1.0 = observed, 0.0 = missing)
    """
    if observed_frames is None:
        shape = (total_frames,) if num_classes is None else (total_frames, num_classes)
        return np.ones(shape, dtype=np.float32)
    mask = observed_frames[:total_frames].astype(np.float32)
    if num_classes is not None and mask.ndim == 1:
        mask = np.broadcast_to(mask[:, None], (total_frames, num_classes)).copy()
    return mask


def validate_npz(npz_path: str | Path, expected_keys: List[str]) -> bool:
    """Validate an NPZ file has expected keys and consistent lengths.

    Args:
        npz_path: Path to NPZ file
        expected_keys: List of expected array keys

    Returns:
        True if valid
    """
    data = np.load(npz_path, allow_pickle=False)
    missing = set(expected_keys) - set(data.files)
    if missing:
        print(f"  Missing keys: {missing}")
        return False

    # Check consistent temporal dimension
    lengths = {}
    for k in data.files:
        lengths[k] = data[k].shape[0]
    unique_lengths = set(lengths.values())
    if len(unique_lengths) > 1:
        print(f"  Inconsistent lengths: {lengths}")
        return False

    return True
