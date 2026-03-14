from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .checkpoint import save_checkpoint, load_checkpoint
from .logger import TrainingLogger


def _is_distributed() -> bool:
    """Check if running in a distributed setting (via torchrun)."""
    return dist.is_available() and dist.is_initialized()


def _get_rank() -> int:
    return dist.get_rank() if _is_distributed() else 0


def _is_main_process() -> bool:
    return _get_rank() == 0


class Trainer:
    """Minimal training loop for SurgCast.

    Handles: train/val epochs, mixed precision, DDP, checkpointing,
    teacher forcing schedule, and logging.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        loss_fn: callable,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: dict,
        logger: TrainingLogger,
        output_dir: str | Path,
        device: torch.device = torch.device("cuda"),
        precision: str = "bf16-mixed",
        grad_accum_steps: int = 1,
        max_grad_norm: float = 1.0,
        checkpoint_every: int = 5,
        val_every: int = 1,
        teacher_forcing_config: Optional[dict] = None,
    ):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger
        self.output_dir = Path(output_dir)
        self.device = device
        self.grad_accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm
        self.checkpoint_every = checkpoint_every
        self.val_every = val_every
        self.distributed = _is_distributed()

        # Wrap model with DDP if distributed
        if self.distributed:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.model = DDP(model, device_ids=[local_rank])
        else:
            self.model = model

        # Mixed precision
        self.use_amp = precision in ("bf16-mixed", "fp16-mixed")
        amp_dtype = torch.bfloat16 if precision == "bf16-mixed" else torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16-mixed"))
        self.autocast_ctx = lambda: torch.amp.autocast("cuda", dtype=amp_dtype, enabled=self.use_amp)

        # Teacher forcing schedule
        self.tf_config = teacher_forcing_config or {}
        self.rho_init = self.tf_config.get("rho_init", 1.0)
        self.rho_final = self.tf_config.get("rho_final", 0.0)
        self.tf_schedule = self.tf_config.get("schedule", "linear")

        # State
        self.epoch = 0
        self.global_step = 0
        self.best_metric = float("inf")

        # Checkpoint dirs (main process only)
        if _is_main_process():
            (self.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    def _get_rho(self, epoch: int, max_epochs: int) -> float:
        """Compute teacher forcing ratio for current epoch."""
        if max_epochs <= 1:
            return self.rho_init
        progress = epoch / (max_epochs - 1)
        if self.tf_schedule == "cosine":
            import math
            return self.rho_final + 0.5 * (self.rho_init - self.rho_final) * (1 + math.cos(math.pi * progress))
        # linear
        return self.rho_init + (self.rho_final - self.rho_init) * progress

    def train_epoch(self, max_epochs: int) -> Dict[str, float]:
        """Run one training epoch. Returns dict of average metrics."""
        self.model.train()
        rho = self._get_rho(self.epoch, max_epochs)
        total_loss = 0.0
        loss_components: Dict[str, float] = {}
        num_batches = 0

        # Set epoch for DistributedSampler
        if self.distributed and hasattr(self.train_loader, "sampler"):
            sampler = self.train_loader.sampler
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(self.epoch)

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            features = batch["features"].to(self.device, non_blocking=True)
            labels = {k: v.to(self.device, non_blocking=True) for k, v in batch["labels"].items()}
            masks = {k: v.to(self.device, non_blocking=True) for k, v in batch["masks"].items()}

            with self.autocast_ctx():
                outputs = self.model(features)
                loss_dict = self.loss_fn(outputs, labels, masks)
                loss = loss_dict["total"] / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.global_step += 1

            total_loss += loss_dict["total"].item()
            for k, v in loss_dict.items():
                if k != "total":
                    loss_components[k] = loss_components.get(k, 0.0) + v.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        metrics = {"train/loss": avg_loss, "train/rho": rho, "train/epoch": self.epoch}
        for k, v in loss_components.items():
            metrics[f"train/{k}"] = v / max(num_batches, 1)

        return metrics

    @torch.no_grad()
    def val_epoch(self) -> Dict[str, float]:
        """Run one validation epoch. Returns dict of average metrics."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        loss_components: Dict[str, float] = {}
        num_batches = 0

        for batch in self.val_loader:
            features = batch["features"].to(self.device, non_blocking=True)
            labels = {k: v.to(self.device, non_blocking=True) for k, v in batch["labels"].items()}
            masks = {k: v.to(self.device, non_blocking=True) for k, v in batch["masks"].items()}

            with self.autocast_ctx():
                outputs = self.model(features)
                loss_dict = self.loss_fn(outputs, labels, masks)

            total_loss += loss_dict["total"].item()
            for k, v in loss_dict.items():
                if k != "total":
                    loss_components[k] = loss_components.get(k, 0.0) + v.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        metrics = {"val/loss": avg_loss, "val/epoch": self.epoch}
        for k, v in loss_components.items():
            metrics[f"val/{k}"] = v / max(num_batches, 1)

        return metrics

    def fit(self, max_epochs: int, resume_from: Optional[str] = None) -> None:
        """Main training loop.

        Args:
            max_epochs: Total number of epochs.
            resume_from: Path to checkpoint to resume from.
        """
        if resume_from:
            info = load_checkpoint(
                resume_from, self.model, self.optimizer, self.scheduler,
                map_location=str(self.device),
            )
            self.epoch = info["epoch"] + 1
            self.global_step = info["global_step"]
            self.best_metric = info["best_metric"]
            print(f"Resumed from epoch {info['epoch']}, step {self.global_step}", file=sys.stderr)

        for epoch in range(self.epoch, max_epochs):
            self.epoch = epoch
            t0 = time.time()

            # Train
            train_metrics = self.train_epoch(max_epochs)
            train_time = time.time() - t0

            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()

            train_metrics["train/time_s"] = train_time
            train_metrics["train/lr"] = self.optimizer.param_groups[0]["lr"]
            self.logger.log(train_metrics, step=self.global_step)

            # Validate
            val_metrics = {}
            if self.val_loader is not None and (epoch + 1) % self.val_every == 0:
                val_metrics = self.val_epoch()
                self.logger.log(val_metrics, step=self.global_step)

                # Track best
                val_loss = val_metrics.get("val/loss", float("inf"))
                if val_loss < self.best_metric:
                    self.best_metric = val_loss
                    if _is_main_process():
                        save_checkpoint(
                            self.output_dir / "checkpoints" / "best.pt",
                            self.model, self.optimizer, self.scheduler,
                            epoch, self.global_step, self.best_metric, self.config,
                        )

            # Periodic checkpoint (main process only)
            if (epoch + 1) % self.checkpoint_every == 0 and _is_main_process():
                save_checkpoint(
                    self.output_dir / "checkpoints" / f"epoch_{epoch:04d}.pt",
                    self.model, self.optimizer, self.scheduler,
                    epoch, self.global_step, self.best_metric, self.config,
                )

            # Print summary
            summary = f"Epoch {epoch}/{max_epochs-1} | train_loss={train_metrics['train/loss']:.4f}"
            if val_metrics:
                summary += f" | val_loss={val_metrics['val/loss']:.4f}"
            summary += f" | lr={train_metrics['train/lr']:.2e} | {train_time:.1f}s"
            print(summary, file=sys.stderr)

        # Save final checkpoint (main process only)
        if _is_main_process():
            save_checkpoint(
                self.output_dir / "checkpoints" / "last.pt",
                self.model, self.optimizer, self.scheduler,
                self.epoch, self.global_step, self.best_metric, self.config,
            )
            self.logger.log_summary({"best_val_loss": self.best_metric, "final_epoch": self.epoch})
            self.logger.finish()
