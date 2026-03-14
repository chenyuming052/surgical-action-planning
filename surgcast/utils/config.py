from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

from .io import load_yaml


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict.

    - Dict values are merged recursively.
    - All other types are replaced by the override value.
    """
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def parse_overrides(override_strs: List[str]) -> dict:
    """Parse CLI override strings like 'loss.lambda_hazard=0.5' into a nested dict.

    Supports dotted keys for nesting and basic type inference:
      - 'true'/'false' -> bool
      - integers -> int
      - floats -> float
      - everything else -> str
    """
    result: Dict[str, Any] = {}
    for s in override_strs:
        if "=" not in s:
            raise ValueError(f"Override must be key=value, got: {s!r}")
        key, val_str = s.split("=", 1)
        val = _infer_type(val_str)
        parts = key.split(".")
        d = result
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = val
    return result


def _infer_type(s: str) -> Any:
    """Infer Python type from a string value."""
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    if s.lower() in ("none", "null"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    # Handle lists like '[1,2,3]'
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [_infer_type(x.strip()) for x in inner.split(",")]
    return s


def load_config(
    *yaml_paths: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> dict:
    """Load and merge multiple YAML configs, then apply overrides.

    Args:
        *yaml_paths: Paths to YAML files, merged left-to-right
            (later files override earlier ones).
        overrides: Dict of overrides applied last (e.g., from CLI).

    Returns:
        Merged config dict.
    """
    merged: dict = {}
    for path in yaml_paths:
        cfg = load_yaml(path)
        if cfg is not None:
            merged = deep_merge(merged, cfg)
    if overrides:
        merged = deep_merge(merged, overrides)
    return merged
