from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


# Label keys that are multi-label (sigmoid) vs single-label (softmax)
MULTI_LABEL_KEYS = {"triplet_group", "instrument", "anatomy"}
SINGLE_LABEL_KEYS = {"phase"}


@dataclass
class SequenceSample:
    features: torch.Tensor
    labels: Dict[str, torch.Tensor]
    visibility_masks: Dict[str, torch.Tensor]
    meta: Dict[str, Any]


def collate_fn(batch: List[SequenceSample]) -> Dict[str, Any]:
    """Collate a list of SequenceSamples into a batched dict."""
    features = torch.stack([s.features for s in batch])

    # Collect all label keys across samples
    all_label_keys = set()
    all_mask_keys = set()
    for s in batch:
        all_label_keys.update(s.labels.keys())
        all_mask_keys.update(s.visibility_masks.keys())

    # Find a reference tensor for each key from the first sample that has it
    label_refs = {}
    for key in sorted(all_label_keys):
        for s in batch:
            if key in s.labels:
                label_refs[key] = s.labels[key]
                break

    labels = {}
    for key in sorted(all_label_keys):
        ref = label_refs[key]
        labels[key] = torch.stack([s.labels.get(key, torch.zeros_like(ref)) for s in batch])

    mask_refs = {}
    for key in sorted(all_mask_keys):
        for s in batch:
            if key in s.visibility_masks:
                mask_refs[key] = s.visibility_masks[key]
                break

    masks = {}
    for key in sorted(all_mask_keys):
        ref = mask_refs[key]
        masks[key] = torch.stack([s.visibility_masks.get(key, torch.zeros_like(ref)) for s in batch])

    meta = [s.meta for s in batch]

    return {
        "features": features,
        "labels": labels,
        "masks": masks,
        "meta": meta,
    }


class SequenceDataset(Dataset):
    """Read HDF5 features + NPZ labels and emit fixed-length windows.

    Each sample in `samples` is a registry row dict with at minimum:
        - canonical_id: str
        - coverage_group: str

    NPZ files are expected at `npz_root / {canonical_id}.npz` with arrays:
        - phase: [T] int
        - triplet_group: [T, G] float
        - instrument: [T, I] float
        - anatomy: [T, A] float
        - mask_phase, mask_triplet_group, mask_instrument, mask_anatomy: [T] or [T, C] float

    HDF5 features at `feature_h5` with datasets keyed by canonical_id: [T, D].
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        feature_h5: str,
        npz_root: str,
        seq_len: int = 16,
        stride: int = 8,
        cache_npz: bool = True,
    ):
        self.samples = {s["canonical_id"]: s for s in samples}
        self.feature_h5 = feature_h5
        self.npz_root = Path(npz_root)
        self.seq_len = seq_len
        self.stride = stride
        self._h5_handle: Optional[h5py.File] = None

        # NPZ cache: vid -> {array_name: np.ndarray}
        self._cache_npz = cache_npz
        self._npz_cache: Dict[str, Dict[str, np.ndarray]] = {}

        # Build (video_id, start_frame) index
        self.index: List[tuple] = []
        self._video_lengths: Dict[str, int] = {}
        for s in samples:
            vid = s["canonical_id"]
            npz_path = self.npz_root / f"{vid}.npz"
            if not npz_path.exists():
                continue
            data = np.load(npz_path, allow_pickle=False)
            # Cache NPZ arrays in memory to avoid repeated disk I/O
            if cache_npz:
                self._npz_cache[vid] = {k: data[k].copy() for k in data.files}
            # Use phase length as reference if available, else first array
            if "phase" in data:
                T = len(data["phase"])
            else:
                first_key = list(data.keys())[0]
                T = data[first_key].shape[0]
            self._video_lengths[vid] = T
            for start in range(0, max(1, T - seq_len + 1), stride):
                self.index.append((vid, start))

    @property
    def h5(self) -> h5py.File:
        """Lazy open HDF5 per-worker for DataLoader compatibility."""
        if self._h5_handle is None:
            self._h5_handle = h5py.File(self.feature_h5, "r")
        return self._h5_handle

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> SequenceSample:
        vid, start = self.index[idx]
        end = min(start + self.seq_len, self._video_lengths[vid])
        actual_len = end - start

        # Load features from HDF5
        feats = torch.from_numpy(self.h5[vid][start:end]).float()
        # Pad if shorter than seq_len
        if actual_len < self.seq_len:
            pad = torch.zeros(self.seq_len - actual_len, feats.size(-1))
            feats = torch.cat([feats, pad], dim=0)

        # Load labels from NPZ (cached or from disk)
        if self._cache_npz and vid in self._npz_cache:
            data = self._npz_cache[vid]
        else:
            npz_path = self.npz_root / f"{vid}.npz"
            data = dict(np.load(npz_path, allow_pickle=False))

        labels = {}
        masks = {}

        for key in ["phase", "triplet_group", "instrument", "anatomy"]:
            if key in data:
                arr = data[key][start:end]
                t = torch.from_numpy(arr.copy())
                if actual_len < self.seq_len:
                    if t.ndim == 1:
                        pad_t = torch.zeros(self.seq_len - actual_len, dtype=t.dtype)
                    else:
                        pad_t = torch.zeros(self.seq_len - actual_len, t.size(-1), dtype=t.dtype)
                    t = torch.cat([t, pad_t], dim=0)
                labels[key] = t.float() if key != "phase" else t.long()

            mask_key = f"mask_{key}"
            if mask_key in data:
                m = torch.from_numpy(data[mask_key][start:end].copy()).float()
                if actual_len < self.seq_len:
                    if m.ndim == 1:
                        pad_m = torch.zeros(self.seq_len - actual_len)
                    else:
                        pad_m = torch.zeros(self.seq_len - actual_len, m.size(-1))
                    m = torch.cat([m, pad_m], dim=0)
                masks[key] = m
            elif key in data:
                # Default: all observed for actual frames, zero for padding
                if key in SINGLE_LABEL_KEYS:
                    m = torch.ones(actual_len)
                else:
                    m = torch.ones(actual_len, labels[key].size(-1)) if labels[key].ndim > 1 else torch.ones(actual_len)
                if actual_len < self.seq_len:
                    if m.ndim == 1:
                        pad_m = torch.zeros(self.seq_len - actual_len)
                    else:
                        pad_m = torch.zeros(self.seq_len - actual_len, m.size(-1))
                    m = torch.cat([m, pad_m], dim=0)
                masks[key] = m

        # Hazard targets
        for hkey in ["hazard_inst_bin", "hazard_group_bin", "hazard_inst_censored", "hazard_group_censored"]:
            if hkey in data:
                arr = data[hkey][start:end]
                t = torch.from_numpy(arr.copy())
                if actual_len < self.seq_len:
                    pad_t = torch.zeros(self.seq_len - actual_len, dtype=t.dtype)
                    t = torch.cat([t, pad_t], dim=0)
                labels[hkey] = t

        # V2 labels: CVS ordinal targets
        if "cvs" in data:
            arr = data["cvs"][start:end]
            t = torch.from_numpy(arr.copy()).float()
            if actual_len < self.seq_len:
                pad_t = torch.zeros(self.seq_len - actual_len, t.size(-1), dtype=t.dtype) if t.ndim > 1 else torch.zeros(self.seq_len - actual_len, dtype=t.dtype)
                t = torch.cat([t, pad_t], dim=0)
            labels["cvs"] = t
            mask_key = "mask_cvs"
            if mask_key in data:
                m = torch.from_numpy(data[mask_key][start:end].copy()).float()
                if actual_len < self.seq_len:
                    pad_m = torch.zeros(self.seq_len - actual_len) if m.ndim == 1 else torch.zeros(self.seq_len - actual_len, m.size(-1))
                    m = torch.cat([m, pad_m], dim=0)
                masks["cvs"] = m
            else:
                masks["cvs"] = torch.cat([torch.ones(actual_len), torch.zeros(self.seq_len - actual_len)]) if actual_len < self.seq_len else torch.ones(actual_len)

        # V2 labels: triplet indices for action encoder
        for tkey in ["triplet_indices", "triplet_mask"]:
            if tkey in data:
                arr = data[tkey][start:end]
                t = torch.from_numpy(arr.copy())
                if actual_len < self.seq_len:
                    if t.ndim == 1:
                        pad_t = torch.zeros(self.seq_len - actual_len, dtype=t.dtype)
                    else:
                        pad_t = torch.zeros(self.seq_len - actual_len, t.size(-1), dtype=t.dtype)
                    t = torch.cat([t, pad_t], dim=0)
                labels[tkey] = t

        # V2 labels: next-action targets and change flags
        for nkey in ["target_delta_add", "target_delta_remove", "target_phase_next",
                     "target_group_next", "target_state", "change_flag", "true_ttc",
                     "ttc_censored", "age_inst", "age_phase", "stable_run_length"]:
            if nkey in data:
                arr = data[nkey][start:end]
                t = torch.from_numpy(arr.copy())
                if actual_len < self.seq_len:
                    if t.ndim == 1:
                        pad_t = torch.zeros(self.seq_len - actual_len, dtype=t.dtype)
                    else:
                        pad_t = torch.zeros(self.seq_len - actual_len, t.size(-1), dtype=t.dtype)
                    t = torch.cat([t, pad_t], dim=0)
                labels[nkey] = t.float() if t.dtype in (torch.float64,) else t

        meta = {
            "canonical_id": vid,
            "start_frame": start,
            "actual_len": actual_len,
            "coverage_group": self.samples[vid].get("coverage_group", ""),
        }

        return SequenceSample(
            features=feats,
            labels=labels,
            visibility_masks=masks,
            meta=meta,
        )
