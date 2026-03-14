from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional


class TrainingLogger:
    """Lightweight training logger with W&B and JSON file fallback.

    Usage:
        logger = TrainingLogger(run_name="exp_01", output_dir="outputs/exp_01",
                                wandb_project="surgcast", config=cfg)
        logger.log({"train/loss": 0.5, "epoch": 1})
        logger.log_summary({"best_val_metric": 0.42})
        logger.finish()
    """

    def __init__(
        self,
        run_name: str,
        output_dir: str | Path,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        config: Optional[dict] = None,
        use_wandb: bool = True,
    ):
        self.run_name = run_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._wandb_run = None
        self._log_file = open(self.output_dir / "train_log.jsonl", "a")

        if use_wandb:
            self._init_wandb(wandb_project, wandb_entity, config)

    def _init_wandb(
        self,
        project: Optional[str],
        entity: Optional[str],
        config: Optional[dict],
    ) -> None:
        try:
            import wandb
            self._wandb_run = wandb.init(
                project=project or "surgcast",
                entity=entity,
                name=self.run_name,
                config=config or {},
                reinit=True,
            )
        except ImportError:
            print("wandb not installed, falling back to JSON logging", file=sys.stderr)
        except Exception as e:
            print(f"wandb init failed ({e}), falling back to JSON logging", file=sys.stderr)

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to W&B and JSON file."""
        # JSON file (always)
        record = dict(metrics)
        if step is not None:
            record["_step"] = step
        self._log_file.write(json.dumps(record) + "\n")
        self._log_file.flush()

        # W&B
        if self._wandb_run is not None:
            import wandb
            wandb.log(metrics, step=step)

    def log_summary(self, metrics: Dict[str, Any]) -> None:
        """Log summary metrics (e.g., best validation score)."""
        summary_path = self.output_dir / "summary.json"
        existing = {}
        if summary_path.exists():
            with open(summary_path) as f:
                existing = json.load(f)
        existing.update(metrics)
        with open(summary_path, "w") as f:
            json.dump(existing, f, indent=2)

        if self._wandb_run is not None:
            import wandb
            for k, v in metrics.items():
                wandb.run.summary[k] = v

    def finish(self) -> None:
        """Close logger resources."""
        self._log_file.close()
        if self._wandb_run is not None:
            import wandb
            wandb.finish()

    def __del__(self):
        if hasattr(self, "_log_file") and not self._log_file.closed:
            self._log_file.close()
