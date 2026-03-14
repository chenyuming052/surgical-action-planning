#!/usr/bin/env python3
from __future__ import annotations

"""Train baseline or full SurgCast.

Example stages:
- cholec_only
- plus_phase
- plus_tool_presence
- plus_cvs
- plus_endoscapes
- plus_masking
- plus_transition
- plus_static_prior
- full

Usage:
    python scripts/train.py \
        --config-data configs/data/default.yaml \
        --config-model configs/model/default.yaml \
        --config-train configs/train/default.yaml \
        --stage full \
        --registry data/registry.json \
        --features-root /yuming/data/surgcast/features \
        --npz-root /yuming/data/surgcast/npz \
        --run-name "surgcast_full_v1"

    # With experiment override and CLI overrides:
    python scripts/train.py \
        --config-data configs/data/default.yaml \
        --config-model configs/model/default.yaml \
        --config-train configs/train/default.yaml configs/train/local_debug.yaml \
        --experiment configs/experiment/cholec_only.yaml \
        --stage cholec_only \
        --registry data/registry.json \
        --features-root /yuming/data/surgcast/features \
        --npz-root /yuming/data/surgcast/npz \
        --run-name "debug_cholec" \
        --override loss.lambda_hazard=0.5 trainer.batch_size=8
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader

from surgcast.utils import set_seed
from surgcast.utils.config import load_config, deep_merge, parse_overrides
from surgcast.models import build_model
from surgcast.datasets import (
    SequenceDataset,
    CoverageAwareSampler,
    collate_fn,
    load_registry,
    filter_by_split,
)
from surgcast.loss import masked_bce, masked_ce, discrete_time_hazard_nll
from surgcast.training import Trainer, TrainingLogger


def setup_distributed():
    """Initialize distributed training if launched via torchrun."""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}"), True
    return torch.device("cuda" if torch.cuda.is_available() else "cpu"), False


def is_main_process():
    """Check if this is the main process (rank 0)."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def build_loss_fn(config: dict):
    """Return a loss function that computes total loss from model outputs."""
    loss_cfg = config.get("loss", {})
    lambda_align = loss_cfg.get("lambda_align", 0.5)
    lambda_hazard = loss_cfg.get("lambda_hazard", 1.0)
    eta_group = loss_cfg.get("eta_group", 1.0)

    def loss_fn(outputs, labels, masks):
        losses = {}

        # Multi-task alignment losses
        if "triplet_group" in outputs and "triplet_group" in labels:
            mask = masks.get("triplet_group", torch.ones_like(labels["triplet_group"]))
            losses["triplet_group"] = masked_bce(
                outputs["triplet_group"], labels["triplet_group"], mask,
            ) * lambda_align

        if "instrument" in outputs and "instrument" in labels:
            mask = masks.get("instrument", torch.ones_like(labels["instrument"]))
            losses["instrument"] = masked_bce(
                outputs["instrument"], labels["instrument"], mask,
            ) * lambda_align

        if "phase" in outputs and "phase" in labels:
            mask = masks.get("phase", torch.ones(labels["phase"].shape[0], labels["phase"].shape[1],
                                                  device=labels["phase"].device))
            losses["phase"] = masked_ce(
                outputs["phase"], labels["phase"], mask,
            ) * lambda_align

        if "anatomy" in outputs and "anatomy" in labels:
            mask = masks.get("anatomy", torch.ones_like(labels["anatomy"]))
            losses["anatomy"] = masked_bce(
                outputs["anatomy"], labels["anatomy"], mask,
            ) * lambda_align

        # Hazard losses
        if "hazard_inst" in outputs and "hazard_inst_bin" in labels:
            B, T, K = outputs["hazard_inst"].shape
            censored = labels.get(
                "hazard_inst_censored",
                torch.zeros(B, T, dtype=torch.bool, device=outputs["hazard_inst"].device),
            )
            losses["hazard_inst"] = discrete_time_hazard_nll(
                outputs["hazard_inst"].reshape(-1, K),
                labels["hazard_inst_bin"].reshape(-1).long(),
                censored.reshape(-1).bool(),
            ) * lambda_hazard

        if "hazard_group" in outputs and "hazard_group_bin" in labels:
            B, T, K = outputs["hazard_group"].shape
            censored = labels.get(
                "hazard_group_censored",
                torch.zeros(B, T, dtype=torch.bool, device=outputs["hazard_group"].device),
            )
            losses["hazard_group"] = discrete_time_hazard_nll(
                outputs["hazard_group"].reshape(-1, K),
                labels["hazard_group_bin"].reshape(-1).long(),
                censored.reshape(-1).bool(),
            ) * lambda_hazard * eta_group

        # Total
        total = sum(losses.values()) if losses else torch.tensor(0.0)
        losses["total"] = total
        return losses

    return loss_fn


def build_loss_fn_v2(config: dict):
    """Return a loss function for V2 model that includes event-conditioned losses."""
    from surgcast.loss import (
        ordinal_bce_cvs,
        ranking_loss,
        consistency_cvs_anatomy,
        next_action_loss,
        heteroscedastic_nll,
    )

    loss_cfg = config.get("loss", {})
    lambda_align = loss_cfg.get("lambda_align", 0.5)
    lambda_hazard = loss_cfg.get("lambda_hazard", 1.0)
    eta_group = loss_cfg.get("eta_group", 1.0)
    lambda_dyn = loss_cfg.get("lambda_dyn", 0.5)
    lambda_next = loss_cfg.get("lambda_next", 0.3)
    lambda_rank = loss_cfg.get("lambda_rank", 0.1)
    lambda_consist = loss_cfg.get("lambda_consist", 0.1)

    def loss_fn(outputs, labels, masks):
        losses = {}

        # V1 multi-task alignment losses
        if "triplet_group" in outputs and "triplet_group" in labels:
            mask = masks.get("triplet_group", torch.ones_like(labels["triplet_group"]))
            losses["triplet_group"] = masked_bce(
                outputs["triplet_group"], labels["triplet_group"], mask,
            ) * lambda_align

        if "instrument" in outputs and "instrument" in labels:
            mask = masks.get("instrument", torch.ones_like(labels["instrument"]))
            losses["instrument"] = masked_bce(
                outputs["instrument"], labels["instrument"], mask,
            ) * lambda_align

        if "phase" in outputs and "phase" in labels:
            mask = masks.get("phase", torch.ones(labels["phase"].shape[0], labels["phase"].shape[1],
                                                  device=labels["phase"].device))
            losses["phase"] = masked_ce(
                outputs["phase"], labels["phase"], mask,
            ) * lambda_align

        if "anatomy" in outputs and "anatomy" in labels:
            mask = masks.get("anatomy", torch.ones_like(labels["anatomy"]))
            losses["anatomy"] = masked_bce(
                outputs["anatomy"], labels["anatomy"], mask,
            ) * lambda_align

        # Hazard losses
        if "hazard_inst" in outputs and "hazard_inst_bin" in labels:
            B, T, K = outputs["hazard_inst"].shape
            censored = labels.get(
                "hazard_inst_censored",
                torch.zeros(B, T, dtype=torch.bool, device=outputs["hazard_inst"].device),
            )
            losses["hazard_inst"] = discrete_time_hazard_nll(
                outputs["hazard_inst"].reshape(-1, K),
                labels["hazard_inst_bin"].reshape(-1).long(),
                censored.reshape(-1).bool(),
            ) * lambda_hazard

        if "hazard_group" in outputs and "hazard_group_bin" in labels:
            B, T, K = outputs["hazard_group"].shape
            censored = labels.get(
                "hazard_group_censored",
                torch.zeros(B, T, dtype=torch.bool, device=outputs["hazard_group"].device),
            )
            losses["hazard_group"] = discrete_time_hazard_nll(
                outputs["hazard_group"].reshape(-1, K),
                labels["hazard_group_bin"].reshape(-1).long(),
                censored.reshape(-1).bool(),
            ) * lambda_hazard * eta_group

        # V2 losses: dynamics
        if lambda_dyn > 0 and "mu_plus" in outputs and "target_state" in labels:
            mask = masks.get("dynamics", torch.ones(labels["target_state"].shape[:2],
                                                    device=labels["target_state"].device))
            losses["dynamics"] = heteroscedastic_nll(
                outputs["mu_plus"], outputs["log_var"],
                labels["target_state"], mask,
            ) * lambda_dyn

        # V2 losses: next action
        if lambda_next > 0 and "delta_add" in outputs and "target_delta_add" in labels:
            mask = masks.get("next_action", torch.ones(labels["target_delta_add"].shape[:2],
                                                       device=labels["target_delta_add"].device))
            losses["next_action"] = next_action_loss(
                outputs["delta_add"], outputs["delta_remove"],
                outputs["phase_next"], outputs["group_next"],
                labels["target_delta_add"], labels["target_delta_remove"],
                labels["target_phase_next"], labels["target_group_next"],
                mask,
            ) * lambda_next

        # V2 losses: ranking
        if lambda_rank > 0 and "predicted_ttc" in outputs and "true_ttc" in labels:
            censored = labels.get("ttc_censored",
                                  torch.zeros_like(labels["true_ttc"], dtype=torch.bool))
            losses["ranking"] = ranking_loss(
                outputs["predicted_ttc"], labels["true_ttc"], censored,
            ) * lambda_rank

        # V2 losses: CVS ordinal
        if lambda_align > 0 and "cvs" in outputs and "cvs" in labels:
            mask = masks.get("cvs", torch.ones(labels["cvs"].shape[:2],
                                               device=labels["cvs"].device))
            losses["cvs"] = ordinal_bce_cvs(
                outputs["cvs"], labels["cvs"], mask,
            ) * lambda_align

        # V2 losses: consistency
        if lambda_consist > 0 and "cvs_c1_prob" in outputs and "anatomy" in outputs:
            mask = masks.get("cvs", torch.ones(outputs["cvs_c1_prob"].shape,
                                               device=outputs["cvs_c1_prob"].device))
            losses["consistency"] = consistency_cvs_anatomy(
                outputs["cvs_c1_prob"],
                outputs.get("duct_prob", torch.zeros_like(outputs["cvs_c1_prob"])),
                outputs.get("artery_prob", torch.zeros_like(outputs["cvs_c1_prob"])),
                mask,
            ) * lambda_consist

        # Total
        total = sum(losses.values()) if losses else torch.tensor(0.0)
        losses["total"] = total
        return losses

    return loss_fn


def build_optimizer(model: nn.Module, config: dict):
    """Build optimizer and scheduler from config."""
    opt_cfg = config.get("optimizer", {})
    sch_cfg = config.get("scheduler", {})

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg.get("lr", 1e-4),
        weight_decay=opt_cfg.get("weight_decay", 0.05),
    )

    max_epochs = config.get("trainer", {}).get("max_epochs", 100)
    min_lr = sch_cfg.get("min_lr", 1e-6)
    warmup_epochs = sch_cfg.get("warmup_epochs", 5)

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(max_epochs - warmup_epochs, 1), eta_min=min_lr,
    )

    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, total_iters=warmup_epochs,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = cosine_scheduler

    return optimizer, scheduler


def build_data_loaders(config: dict, registry_path: str, features_root: str, npz_root: str):
    """Build train and val DataLoaders."""
    data_cfg = config.get("data", {})
    trainer_cfg = config.get("trainer", {})
    seq_len = data_cfg.get("sequence_length", 16)
    stride = data_cfg.get("window_stride", 8)
    batch_size = trainer_cfg.get("batch_size", 64)
    num_workers = trainer_cfg.get("num_workers", 8)

    registry = load_registry(registry_path)
    train_samples = filter_by_split(registry, "train")
    val_samples = filter_by_split(registry, "val")

    # Resolve feature HDF5 — accept directory (pick first .h5) or direct file
    features_root = Path(features_root)
    if features_root.is_dir():
        h5_files = list(features_root.glob("*.h5")) + list(features_root.glob("*.hdf5"))
        if not h5_files:
            raise FileNotFoundError(f"No HDF5 files found in {features_root}")
        feature_h5 = str(h5_files[0])
    else:
        feature_h5 = str(features_root)

    train_ds = SequenceDataset(train_samples, feature_h5, npz_root, seq_len, stride)
    val_ds = SequenceDataset(val_samples, feature_h5, npz_root, seq_len, stride)

    # Coverage-aware sampling
    group_probs = data_cfg.get("coverage_groups", {})
    group_to_indices = {}
    for i, (vid, start) in enumerate(train_ds.index):
        meta = train_ds.samples[vid]
        g = meta.get("coverage_group", "G7")
        group_to_indices.setdefault(g, []).append(i)

    sampler = CoverageAwareSampler(
        group_to_indices, group_probs, num_samples=len(train_ds),
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        collate_fn=collate_fn, num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
        pin_memory=True,
    ) if len(val_ds) > 0 else None

    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Train SurgCast model")
    parser.add_argument("--config-data", required=True, nargs="+",
                        help="Data config YAML(s)")
    parser.add_argument("--config-model", required=True, nargs="+",
                        help="Model config YAML(s)")
    parser.add_argument("--config-train", required=True, nargs="+",
                        help="Training config YAML(s)")
    parser.add_argument("--experiment", default=None,
                        help="Experiment override YAML")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--features-root", required=True)
    parser.add_argument("--npz-root", required=True)
    parser.add_argument("--priors-dir", default="")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--override", nargs="*", default=[],
                        help="CLI overrides, e.g. loss.lambda_hazard=0.5")
    args = parser.parse_args()

    # Load and merge configs
    data_cfg = load_config(*args.config_data)
    model_cfg = load_config(*args.config_model)
    train_cfg = load_config(*args.config_train)

    config = deep_merge(deep_merge(data_cfg, model_cfg), train_cfg)

    # Apply experiment override
    if args.experiment:
        exp_cfg = load_config(args.experiment)
        config = deep_merge(config, exp_cfg)

    # Apply CLI overrides
    if args.override:
        cli_overrides = parse_overrides(args.override)
        config = deep_merge(config, cli_overrides)

    # Seed
    seed = config.get("seed", 3407)
    set_seed(seed)

    # Output directory
    output_root = os.environ.get("SURGCAST_OUTPUT", "outputs")
    output_dir = Path(output_root) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device + distributed setup
    device, distributed = setup_distributed()

    # Build components
    model = build_model(config)
    model = model.to(device)

    if is_main_process():
        print(f"Model: {model.__class__.__name__}, "
              f"params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
              file=sys.stderr)

    optimizer, scheduler = build_optimizer(model, config)
    version = config.get("version", "v1")
    if version == "v2":
        loss_fn = build_loss_fn_v2(config)
    else:
        loss_fn = build_loss_fn(config)

    train_loader, val_loader = build_data_loaders(
        config, args.registry, args.features_root, args.npz_root,
    )

    # Logger
    logger = TrainingLogger(
        run_name=args.run_name,
        output_dir=output_dir,
        wandb_project=os.environ.get("WANDB_PROJECT", "surgcast"),
        wandb_entity=os.environ.get("WANDB_ENTITY"),
        config=config,
        use_wandb=os.environ.get("WANDB_API_KEY", "") != "",
    )

    # Trainer
    trainer_cfg = config.get("trainer", {})
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        logger=logger,
        output_dir=output_dir,
        device=device,
        precision=trainer_cfg.get("precision", "bf16-mixed"),
        teacher_forcing_config=config.get("teacher_forcing"),
    )

    max_epochs = trainer_cfg.get("max_epochs", 100)
    trainer.fit(max_epochs=max_epochs, resume_from=args.resume)

    if is_main_process():
        print(f"Training complete. Best val loss: {trainer.best_metric:.4f}", file=sys.stderr)
        print(f"Outputs saved to: {output_dir}", file=sys.stderr)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
