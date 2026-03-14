#!/usr/bin/env python3
from __future__ import annotations

"""Evaluate one checkpoint on tiered benchmarks.

Tier 1: core action-change on CholecT50-test
Tier 2a: CVS safety on Endoscapes-test + Cholec80-test
Tier 2b: CVS-at-clipping on G2 test subset
Tier 3: phase
Tier 4: instrument
Tier 5: anatomy

Usage:
    python scripts/evaluate.py \
        --checkpoint outputs/surgcast_full_v1/checkpoints/best.pt \
        --config-eval configs/eval/default.yaml \
        --config-data configs/data/default.yaml \
        --registry data/registry.json \
        --features-root /yuming/data/surgcast/features \
        --npz-root /yuming/data/surgcast/npz \
        --out-dir outputs/surgcast_full_v1/eval
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from surgcast.utils import set_seed
from surgcast.utils.config import load_config
from surgcast.utils.io import save_json
from surgcast.models import build_model
from surgcast.datasets import (
    SequenceDataset,
    collate_fn,
    load_registry,
    filter_by_split,
)
from surgcast.metrics import (
    compute_event_ap,
    compute_event_auroc,
    compute_post_change_map,
    compute_dense_map,
    compute_change_conditioned_metrics,
    compute_ttc_mae,
    compute_expected_ttc,
    compute_c_index,
    compute_brier_score,
    compute_hazard_calibration,
    compute_cvs_criterion_auc,
    compute_cvs_mae,
    compute_clipping_detection_rate,
    compute_clipping_false_alarm_rate,
    compute_cvs_mae_at_clipping,
)


def collect_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Run model on all batches and collect predictions + labels."""
    model.eval()
    all_outputs: Dict[str, List[np.ndarray]] = {}
    all_labels: Dict[str, List[np.ndarray]] = {}
    all_masks: Dict[str, List[np.ndarray]] = {}

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device, non_blocking=True)
            outputs = model(features)

            for k, v in outputs.items():
                if isinstance(v, torch.Tensor):
                    all_outputs.setdefault(k, []).append(v.cpu().numpy())
            for k, v in batch["labels"].items():
                all_labels.setdefault(k, []).append(v.numpy())
            for k, v in batch["masks"].items():
                all_masks.setdefault(k, []).append(v.numpy())

    results = {}
    for k, v_list in all_outputs.items():
        results[f"pred_{k}"] = np.concatenate(v_list, axis=0)
    for k, v_list in all_labels.items():
        results[f"label_{k}"] = np.concatenate(v_list, axis=0)
    for k, v_list in all_masks.items():
        results[f"mask_{k}"] = np.concatenate(v_list, axis=0)

    return results


def _flatten_bt(arr: np.ndarray) -> np.ndarray:
    """Reshape [B, T, ...] to [B*T, ...]."""
    if arr.ndim >= 3:
        return arr.reshape(-1, *arr.shape[2:])
    return arr.reshape(-1)


def evaluate_tier1(data: Dict[str, np.ndarray], bin_edges: np.ndarray) -> Dict[str, float]:
    """Tier 1: Core action-change metrics."""
    metrics = {}

    for event_type in ["inst", "group"]:
        pred_key = f"pred_hazard_{event_type}"
        label_key = f"label_hazard_{event_type}_bin"
        cens_key = f"label_hazard_{event_type}_censored"

        if pred_key not in data or label_key not in data:
            continue

        hazard_logits = _flatten_bt(data[pred_key])
        true_bins = _flatten_bt(data[label_key]).astype(int)
        censored = _flatten_bt(data.get(cens_key, np.zeros_like(true_bins, dtype=bool))).astype(bool)

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        true_ttc = bin_centers[np.clip(true_bins, 0, len(bin_centers) - 1)]

        metrics[f"event_ap_{event_type}_5s"] = compute_event_ap(hazard_logits, true_ttc, censored, 5.0)
        metrics[f"event_auroc_{event_type}_5s"] = compute_event_auroc(hazard_logits, true_ttc, censored, 5.0)
        metrics[f"ttc_{event_type}_mae"] = compute_ttc_mae(hazard_logits, true_ttc, censored, bin_edges)

        predicted_ttc = compute_expected_ttc(hazard_logits, bin_edges)
        metrics[f"c_index_{event_type}"] = compute_c_index(predicted_ttc, true_ttc, censored)
        metrics[f"brier_{event_type}_at_5s"] = compute_brier_score(hazard_logits, true_ttc, 5.0, bin_edges)
        metrics[f"calibration_{event_type}"] = compute_hazard_calibration(hazard_logits, true_ttc, censored)

    return metrics


def evaluate_tier2a(data: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Tier 2a: CVS state metrics."""
    metrics = {}

    if "pred_cvs" not in data or "label_cvs" not in data:
        return metrics

    pred = _flatten_bt(data["pred_cvs"])
    true = _flatten_bt(data["label_cvs"])
    mask = _flatten_bt(data["mask_cvs"]) if "mask_cvs" in data else np.ones(pred.shape[0])

    auc_dict = compute_cvs_criterion_auc(pred, true, mask)
    metrics.update({f"cvs_{k}": v for k, v in auc_dict.items()})
    metrics["cvs_auc_mean"] = float(np.mean(list(auc_dict.values()))) if auc_dict else 0.0

    # Convert ordinal logits to scores for MAE
    pred_prob = 1.0 / (1.0 + np.exp(-pred))
    pred_scores = np.stack([pred_prob[:, 0] + pred_prob[:, 1],
                            pred_prob[:, 2] + pred_prob[:, 3],
                            pred_prob[:, 4] + pred_prob[:, 5]], axis=-1)
    true_scores = np.stack([true[:, 0] + true[:, 1],
                            true[:, 2] + true[:, 3],
                            true[:, 4] + true[:, 5]], axis=-1)
    metrics["cvs_mae"] = compute_cvs_mae(pred_scores, true_scores, mask)

    return metrics


def evaluate_tier3(data: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Tier 3: Phase accuracy."""
    metrics = {}

    if "pred_phase" not in data or "label_phase" not in data:
        return metrics

    pred = _flatten_bt(data["pred_phase"])
    true = _flatten_bt(data["label_phase"]).astype(int)
    mask = _flatten_bt(data["mask_phase"]) if "mask_phase" in data else np.ones(pred.shape[0])

    pred_class = pred.argmax(axis=-1)
    valid = mask > 0
    if valid.sum() > 0:
        metrics["phase_acc"] = float((pred_class[valid] == true[valid]).mean())

    return metrics


def evaluate_tier4(data: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Tier 4: Instrument prediction metrics."""
    metrics = {}

    if "pred_instrument" not in data or "label_instrument" not in data:
        return metrics

    pred = _flatten_bt(data["pred_instrument"])
    true = _flatten_bt(data["label_instrument"])
    mask = _flatten_bt(data["mask_instrument"]) if "mask_instrument" in data else np.ones(pred.shape[0])

    pred_prob = 1.0 / (1.0 + np.exp(-pred))
    metrics["instrument_dense_map"] = compute_dense_map(pred_prob, true, mask)

    return metrics


def evaluate_tier5(data: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Tier 5: Anatomy prediction metrics."""
    metrics = {}

    if "pred_anatomy" not in data or "label_anatomy" not in data:
        return metrics

    pred = _flatten_bt(data["pred_anatomy"])
    true = _flatten_bt(data["label_anatomy"])
    mask = _flatten_bt(data["mask_anatomy"]) if "mask_anatomy" in data else np.ones(pred.shape[0])

    pred_prob = 1.0 / (1.0 + np.exp(-pred))
    metrics["anatomy_dense_map"] = compute_dense_map(pred_prob, true, mask)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate SurgCast model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config-eval", required=True)
    parser.add_argument("--config-data", default="configs/data/default.yaml")
    parser.add_argument("--registry", required=True)
    parser.add_argument("--features-root", required=True)
    parser.add_argument("--npz-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint config
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})

    eval_cfg = load_config(args.config_eval)
    data_cfg = load_config(args.config_data)

    seed = config.get("seed", data_cfg.get("seed", 3407))
    set_seed(seed)

    # Build model
    model = build_model(config)
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(ckpt["model"])
    model = model.to(device)

    # Build dataset
    registry = load_registry(args.registry)
    test_samples = filter_by_split(registry, args.split)

    features_root = Path(args.features_root)
    if features_root.is_dir():
        h5_files = list(features_root.glob("*.h5")) + list(features_root.glob("*.hdf5"))
        if not h5_files:
            raise FileNotFoundError(f"No HDF5 files found in {features_root}")
        feature_h5 = str(h5_files[0])
    else:
        feature_h5 = str(features_root)

    seq_len = data_cfg.get("sequence_length", config.get("sequence_length", 16))
    stride = data_cfg.get("window_stride", config.get("window_stride", 8))
    test_ds = SequenceDataset(test_samples, feature_h5, args.npz_root, seq_len, stride)

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
    )

    if len(test_ds) == 0:
        print(f"No test samples found for split '{args.split}'", file=sys.stderr)
        return

    print(f"Evaluating on {len(test_ds)} windows from {len(test_samples)} videos", file=sys.stderr)

    # Collect predictions
    data = collect_predictions(model, test_loader, device)

    # Get bin edges
    hazard_cfg = data_cfg.get("hazard", config.get("hazard", {}))
    bin_edges = np.array(hazard_cfg.get("bin_edges_sec", list(range(21))), dtype=np.float64)

    # Evaluate each tier
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    tier_results = {}

    tier1 = evaluate_tier1(data, bin_edges)
    all_metrics.update(tier1)
    tier_results["tier1_action_change"] = tier1

    tier2a = evaluate_tier2a(data)
    all_metrics.update(tier2a)
    tier_results["tier2a_cvs_state"] = tier2a

    tier3 = evaluate_tier3(data)
    all_metrics.update(tier3)
    tier_results["tier3_phase"] = tier3

    tier4 = evaluate_tier4(data)
    all_metrics.update(tier4)
    tier_results["tier4_instrument"] = tier4

    tier5 = evaluate_tier5(data)
    all_metrics.update(tier5)
    tier_results["tier5_anatomy"] = tier5

    # Save results
    save_json(all_metrics, out_dir / "eval_results.json")
    save_json(tier_results, out_dir / "eval_results_by_tier.json")

    # Print summary
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Evaluation complete. {len(all_metrics)} metrics computed.", file=sys.stderr)
    for tier_name, tier_metrics in tier_results.items():
        if tier_metrics:
            print(f"\n  {tier_name}:", file=sys.stderr)
            for k, v in sorted(tier_metrics.items()):
                print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}", file=sys.stderr)
    print(f"\nResults saved to: {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
