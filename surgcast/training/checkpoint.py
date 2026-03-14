from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def _get_git_hash() -> str:
    """Return current git short hash, or 'unknown' if not in a git repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    global_step: int,
    best_metric: float,
    config: dict,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a training checkpoint with full reproducibility metadata.

    Saves:
        - model state dict (unwraps DDP if needed)
        - optimizer state dict
        - scheduler state dict (if provided)
        - epoch, global_step, best_metric
        - full merged config
        - git hash
        - any extra metadata
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Unwrap DDP
    model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

    ckpt = {
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "config": config,
        "git_hash": _get_git_hash(),
    }
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()
    if extra:
        ckpt["extra"] = extra

    torch.save(ckpt, path)

    # Also save config as readable YAML alongside checkpoint
    config_path = path.parent / "config.yaml"
    if not config_path.exists():
        import yaml
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """Load a training checkpoint.

    Args:
        path: Path to checkpoint file.
        model: Model to load state into (handles DDP unwrapping).
        optimizer: Optimizer to load state into (optional).
        scheduler: Scheduler to load state into (optional).
        map_location: Device mapping for torch.load.

    Returns:
        Dict with 'epoch', 'global_step', 'best_metric', 'config', 'git_hash'.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    # Load model (handle DDP)
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(ckpt["model"])

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])

    return {
        "epoch": ckpt["epoch"],
        "global_step": ckpt["global_step"],
        "best_metric": ckpt.get("best_metric", float("inf")),
        "config": ckpt.get("config", {}),
        "git_hash": ckpt.get("git_hash", "unknown"),
    }
