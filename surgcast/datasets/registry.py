from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def load_registry(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def filter_by_split(rows: List[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    return [r for r in rows if r.get('split') == split]
