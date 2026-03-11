from __future__ import annotations

import random
from typing import Dict, Iterator, List

from torch.utils.data import Sampler


class CoverageAwareSampler(Sampler[int]):
    """Weighted sampling by coverage group (G1-G7).

    At each draw, first sample a group according to group_probs,
    then uniformly sample an index from that group.
    """

    def __init__(self, group_to_indices: Dict[str, List[int]], group_probs: Dict[str, float], num_samples: int):
        self.group_to_indices = {g: list(indices) for g, indices in group_to_indices.items() if indices}
        # Normalize probs to only include groups that have data
        active_groups = set(self.group_to_indices.keys())
        raw = {g: p for g, p in group_probs.items() if g in active_groups}
        total = sum(raw.values())
        self.group_probs = {g: p / total for g, p in raw.items()} if total > 0 else {}
        self.num_samples = num_samples
        self._groups = list(self.group_probs.keys())
        self._weights = [self.group_probs[g] for g in self._groups]

    def __iter__(self) -> Iterator[int]:
        for _ in range(self.num_samples):
            # Sample a group
            group = random.choices(self._groups, weights=self._weights, k=1)[0]
            # Sample an index within that group
            idx = random.choice(self.group_to_indices[group])
            yield idx

    def __len__(self) -> int:
        return self.num_samples
