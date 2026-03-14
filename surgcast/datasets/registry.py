from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def load_registry(path: str | Path) -> List[Dict[str, Any]]:
    """Load registry JSON, handling both envelope and flat-list formats.

    build_registry.py outputs {schema_version, records: {vid: {...}, ...}}.
    This function normalises to a flat list of record dicts.
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    # Envelope format: {records: {vid: {...}, ...}, ...}
    if isinstance(data, dict) and "records" in data:
        return list(data["records"].values())

    # Bare dict of vid -> record (unlikely but defensive)
    if isinstance(data, dict):
        return list(data.values())

    return data


def filter_by_split(rows: List[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    return [r for r in rows if r.get('split') == split]
