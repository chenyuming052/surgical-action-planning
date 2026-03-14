from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Any, List


class StructuredPrior:
    """Structured prior — optional regularizer.

    Three-layer design: static -> context-modulated -> coverage dropout.
    Supports categorical KL (phase) and Bernoulli KL (multi-label tasks).
    Evidence gating: w_evidence = min(1.0, count / N_sufficient).
    Optional sigma-gating: alpha(sigma_t), beta(sigma_t) with 4 learnable params.
    """

    def __init__(self, static_prior_path: str | Path, evidence_weight_path: str | Path):
        with open(static_prior_path, 'rb') as f:
            self.static = pickle.load(f)
        with open(evidence_weight_path, 'rb') as f:
            self.weights = pickle.load(f)

    def compute_prior_loss(
        self,
        predicted_logits: Dict[str, Any],
        tasks: List[str],
        coverage_group: str,
        sigma_t: Any = None,
    ) -> Any:
        """Three-layer prior loss: static -> context-modulated -> coverage dropout.

        Args:
            predicted_logits: dict mapping task name -> logits
            tasks: list of task names to compute KL for
            coverage_group: one of 'G1'..'G7'
            sigma_t: optional uncertainty for sigma-gating

        Returns:
            Scalar prior loss
        """
        raise NotImplementedError("StructuredPrior.compute_prior_loss")

    def categorical_kl(self, predicted_logits: Any, task: str) -> Any:
        """KL divergence for categorical tasks (phase).

        Args:
            predicted_logits: [B, T, C] logits
            task: task name (e.g., 'phase')

        Returns:
            Scalar KL divergence
        """
        raise NotImplementedError("StructuredPrior.categorical_kl")

    def bernoulli_kl(self, predicted_logits: Any, task: str) -> Any:
        """KL divergence for factorized Bernoulli tasks (instrument, triplet_group, anatomy).

        Args:
            predicted_logits: [B, T, D] logits
            task: task name

        Returns:
            Scalar KL divergence
        """
        raise NotImplementedError("StructuredPrior.bernoulli_kl")

    def evidence_gating(self, task: str, coverage_group: str, n_sufficient: int = 100) -> float:
        """Evidence gating weight: w = min(1.0, count / N_sufficient).

        Args:
            task: task name
            coverage_group: one of 'G1'..'G7'
            n_sufficient: threshold for full evidence weight

        Returns:
            Weight in [0, 1]
        """
        raise NotImplementedError("StructuredPrior.evidence_gating")

    def sigma_gating(self, sigma_t: Any) -> Any:
        """Optional sigma-gating: alpha(sigma_t), beta(sigma_t) with 4 learnable params.

        Args:
            sigma_t: [B, T, 4] per-horizon uncertainty

        Returns:
            alpha, beta scaling factors
        """
        raise NotImplementedError("StructuredPrior.sigma_gating")
