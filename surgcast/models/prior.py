from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Any


class StructuredPrior:
    def __init__(self, static_prior_path: str | Path, evidence_weight_path: str | Path):
        with open(static_prior_path, 'rb') as f:
            self.static = pickle.load(f)
        with open(evidence_weight_path, 'rb') as f:
            self.weights = pickle.load(f)

    # TODO: implement categorical prior for phase and factorized Bernoulli prior for multi-label tasks.
