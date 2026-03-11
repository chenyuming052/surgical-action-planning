from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import numpy as np


def load_npz(path: str | Path) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}
