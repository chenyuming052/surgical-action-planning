#!/usr/bin/env python3
from __future__ import annotations

"""Build a leakage-safe canonical registry for SurgCast.

This script merges Cholec80, CholecT50, Cholec80-CVS, and Endoscapes at the
physical recording level and writes a single registry.json.

Authoritative rules implemented here:
1) Canonical unit is the physical recording, keyed by canonical_id (VIDxx).
2) Overlaps are resolved before split assignment.
3) Cholec80-CVS is an annotation layer on top of Cholec80 and inherits the
   Cholec80 canonical_id and split.
4) If a CAMMA combined split manifest is provided/found, it is the source of
   truth for split assignment.
5) If no manifest is available, a deterministic fallback split is generated
   from the proposal quotas.

Important note about a proposal-level inconsistency:
- The proposal says the official Endoscapes test split has zero overlap with
  Cholec80/CholecT50, but the proposal fallback quota table also assigns one
  G3 (CholecT50+Endoscapes) video to test.
- This script therefore treats the CAMMA combined split manifest as the only
  authoritative split source. The fallback split follows the proposal quota
  table exactly and does not try to reconcile that inconsistency on its own.
"""

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence, Set, Tuple

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
TEXT_SUFFIXES = {".json", ".csv", ".txt", ".tsv", ".xlsx", ".xls"}

EXPECTED_GROUP_COUNTS = {
    "G1": 3,
    "G2": 42,
    "G3": 3,
    "G4": 3,
    "G5": 2,
    "G6": 32,
    "G7": 192,
}

# Last-resort fallback quotas from the proposal's estimated quota table.
# Used ONLY when the three CAMMA split files (CholecT50_splits.json,
# Endoscapes_splits.json, Cholec80_splits.json) are unavailable.
# The authoritative split assignment is the CAMMA combined split strategy
# which produces 168/48/61 (train/val/test), not 191/41/45.
FALLBACK_SPLIT_QUOTAS = {
    "G1": {"train": 2, "val": 1, "test": 0},
    "G2": {"train": 28, "val": 6, "test": 8},
    "G3": {"train": 2, "val": 0, "test": 1},
    "G4": {"train": 2, "val": 1, "test": 0},
    "G5": {"train": 1, "val": 0, "test": 1},
    "G6": {"train": 22, "val": 5, "test": 5},
    "G7": {"train": 134, "val": 28, "test": 30},
}

# Deterministic fallback anchors, matching the proposal where explicitly stated.
FALLBACK_PREFERRED_TEST = {
    "G3": ["VID110", "VID103", "VID96"],  # proposal: one of these goes to test
    "G5": ["VID111"],  # proposal: VID111 goes to test
}

FALLBACK_PREFERRED_VAL = {
    "G1": ["VID70", "VID68", "VID66"],
    "G4": ["VID72", "VID71", "VID67"],
}

PHASE_LABEL = "phase"
TRIPLET_LABEL = "triplet"
INSTRUMENT_LABEL = "instrument"
VERB_LABEL = "verb"
TARGET_LABEL = "target"
TOOL_PRESENCE_LABEL = "tool_presence"
CVS_CHOLEC80_LABEL = "cvs_cholec80"
CVS_ENDOSCAPES_LABEL = "cvs_endoscapes"
ANATOMY_BBOX_LABEL = "anatomy_bbox"


@dataclass
class VideoProbe:
    dataset_video_id: str
    canonical_id: Optional[str] = None
    frame_count: Optional[int] = None
    frames_dir: Optional[str] = None
    label_paths: List[str] = field(default_factory=list)
    extra_paths: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))


@dataclass
class RegistryRow:
    canonical_id: str
    split: Optional[str] = None
    split_source: Optional[str] = None
    coverage_group: Optional[str] = None
    in_cholec80: bool = False
    in_cholect50: bool = False
    in_endoscapes: bool = False
    has_cholec80_cvs: bool = False
    cholec80_tool_presence: bool = False
    has_endoscapes_cvs: bool = False
    has_endoscapes_bbox: bool = False
    labels_available: List[str] = field(default_factory=list)
    frame_counts: Dict[str, int] = field(default_factory=dict)
    source_ids: Dict[str, Any] = field(default_factory=dict)
    file_paths: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "canonical_id": self.canonical_id,
            "split": self.split,
            "split_source": self.split_source,
            "coverage_group": self.coverage_group,
            "in_cholec80": self.in_cholec80,
            "in_cholect50": self.in_cholect50,
            "in_endoscapes": self.in_endoscapes,
            "has_cholec80_cvs": self.has_cholec80_cvs,
            "cholec80_tool_presence": self.cholec80_tool_presence,
            "has_endoscapes_cvs": self.has_endoscapes_cvs,
            "has_endoscapes_bbox": self.has_endoscapes_bbox,
            "labels_available": self.labels_available,
            "frame_counts": self.frame_counts,
            "source_ids": self.source_ids,
            "file_paths": self.file_paths,
            "notes": self.notes,
        }


def normalize_vid(n: int) -> str:
    return f"VID{n:02d}" if n < 100 else f"VID{n}"


def extract_vid(text: str) -> Optional[str]:
    if text is None:
        return None
    s = str(text)
    for pattern in (r"\bVID\s*0*([0-9]{1,4})\b", r"\bvideo\s*0*([0-9]{1,4})\b"):
        m = re.search(pattern, s, flags=re.IGNORECASE)
        if m:
            return normalize_vid(int(m.group(1)))
    return None


def find_first(paths: Iterable[Path], predicate) -> Optional[Path]:
    for p in paths:
        if predicate(p):
            return p
    return None


def stable_key(seed: int, value: str) -> str:
    return hashlib.sha1(f"{seed}:{value}".encode("utf-8")).hexdigest()


def stable_shuffle(ids: Sequence[str], seed: int) -> List[str]:
    return sorted(ids, key=lambda x: stable_key(seed, x))


def json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=False)
        f.write("\n")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def choose_best_frames_dir(counter: Counter) -> Optional[str]:
    if not counter:
        return None
    # Pick the directory with the most image files, then lexical tie-break.
    best_dir, _ = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return best_dir


def discover_cholec80(root: Path) -> Dict[str, VideoProbe]:
    probes: Dict[str, VideoProbe] = {}
    frame_counts: Dict[str, int] = defaultdict(int)
    frame_dirs: Dict[str, Counter] = defaultdict(Counter)
    label_paths: Dict[str, List[str]] = defaultdict(list)
    extra_paths: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

    for path in root.rglob("*"):
        path_str = str(path)
        vid = None
        for part in (path.name, *[p.name for p in path.parents[:4]]):
            vid = extract_vid(part)
            if vid is not None:
                break
        if vid is None:
            continue

        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            frame_counts[vid] += 1
            frame_dirs[vid][str(path.parent)] += 1
        elif path.is_file():
            lower = path.name.lower()
            if "phase" in lower:
                label_paths[vid].append(path_str)
            if "tool" in lower:
                extra_paths[vid]["tool_presence_files"].append(path_str)
            if path.suffix.lower() in {".txt", ".csv"}:
                extra_paths[vid]["misc_label_files"].append(path_str)

    all_vids: Set[str] = set(frame_counts) | set(label_paths) | set(extra_paths)
    for vid in sorted(all_vids, key=_vid_sort_key):
        probes[vid] = VideoProbe(
            dataset_video_id=vid.replace("VID", "video"),
            canonical_id=vid,
            frame_count=frame_counts.get(vid),
            frames_dir=choose_best_frames_dir(frame_dirs.get(vid, Counter())),
            label_paths=sorted(label_paths.get(vid, [])),
            extra_paths={k: sorted(v) for k, v in extra_paths.get(vid, {}).items()},
        )
    return probes


def discover_cholect50(root: Path) -> Dict[str, VideoProbe]:
    probes: Dict[str, VideoProbe] = {}
    frame_counts: Dict[str, int] = defaultdict(int)
    frame_dirs: Dict[str, Counter] = defaultdict(Counter)
    label_paths: Dict[str, List[str]] = defaultdict(list)
    extra_paths: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

    for path in root.rglob("*"):
        vid = None
        for part in (path.name, *[p.name for p in path.parents[:4]]):
            vid = extract_vid(part)
            if vid is not None:
                break
        if vid is None:
            continue

        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            frame_counts[vid] += 1
            frame_dirs[vid][str(path.parent)] += 1
        elif path.is_file() and path.suffix.lower() == ".json":
            label_paths[vid].append(str(path))
        elif path.is_file() and path.suffix.lower() in {".txt", ".csv"}:
            extra_paths[vid]["misc_label_files"].append(str(path))

    all_vids: Set[str] = set(frame_counts) | set(label_paths) | set(extra_paths)
    for vid in sorted(all_vids, key=_vid_sort_key):
        probes[vid] = VideoProbe(
            dataset_video_id=vid,
            canonical_id=vid,
            frame_count=frame_counts.get(vid),
            frames_dir=choose_best_frames_dir(frame_dirs.get(vid, Counter())),
            label_paths=sorted(label_paths.get(vid, [])),
            extra_paths={k: sorted(v) for k, v in extra_paths.get(vid, {}).items()},
        )
    return probes


def infer_endoscapes_public_id_from_row(row: MutableMapping[str, str]) -> Optional[str]:
    aliases = [
        "video_id",
        "vid",
        "video",
        "public_id",
        "video_idx",
        "clip_id",
        "case_id",
    ]
    lower_map = {str(k).strip().lower(): v for k, v in row.items()}
    for alias in aliases:
        if alias in lower_map and str(lower_map[alias]).strip() not in {"", "nan", "None", "none"}:
            return str(lower_map[alias]).strip()
    return None


def discover_endoscapes(root: Path, public_to_canonical: Dict[str, str]) -> Dict[str, VideoProbe]:
    probes: Dict[str, VideoProbe] = {}

    metadata_path = find_first(root.rglob("all_metadata.csv"), lambda p: True)
    # Frame counts come from metadata CSV only (authoritative, avoids double-counting).
    counts_by_public: Dict[str, int] = defaultdict(int)
    # frames_dir comes from image walking only (to find the actual directory).
    dirs_by_public: Dict[str, Counter] = defaultdict(Counter)
    # Track which public_ids we know about from all sources.
    known_public_ids: set = set(public_to_canonical.keys())

    if metadata_path is not None:
        with metadata_path.open("r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                public_id = infer_endoscapes_public_id_from_row(row)
                if public_id is None:
                    continue
                counts_by_public[public_id] += 1
                known_public_ids.add(public_id)

    # Walk only train/, val/, test/ split directories for image discovery.
    # Real Endoscapes layout: flat files named {vid}_{frame}.jpg inside split dirs.
    split_dirs = [root / split_name for split_name in ("train", "val", "test")]
    for split_dir in split_dirs:
        if not split_dir.is_dir():
            continue
        for path in split_dir.iterdir():
            if not (path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES):
                continue
            # Extract public_id from {vid}_{frame}.jpg filename pattern.
            stem = path.stem
            sep_idx = stem.find("_")
            if sep_idx > 0:
                public_id = stem[:sep_idx]
            else:
                public_id = stem
            dirs_by_public[public_id][str(path.parent)] += 1
            known_public_ids.add(public_id)
            # Only use image walk for frame count if metadata didn't provide it.
            # (We accumulate here; we'll reconcile below.)

    # Build probes for all known public_ids.
    all_public_ids = known_public_ids | set(counts_by_public.keys())
    for public_id in sorted(all_public_ids):
        canonical_id = public_to_canonical.get(public_id)
        # Frame count: prefer metadata CSV count; fall back to image walk count.
        frame_count = counts_by_public.get(public_id)
        if frame_count is None or frame_count == 0:
            # Sum image files found for this public_id across split dirs.
            img_count = sum(dirs_by_public.get(public_id, Counter()).values())
            frame_count = img_count if img_count > 0 else None
        probes[public_id] = VideoProbe(
            dataset_video_id=public_id,
            canonical_id=canonical_id,
            frame_count=frame_count,
            frames_dir=choose_best_frames_dir(dirs_by_public.get(public_id, Counter())),
            label_paths=[str(metadata_path)] if metadata_path is not None else [],
            extra_paths={},
        )
    return probes


def load_endoscapes_mapping(mapping_dir: Path) -> Dict[str, str]:
    mapping_files_json = list(mapping_dir.rglob("mapping_to_endoscapes.json"))
    mapping_files_csv = list(mapping_dir.rglob("endoscapes_vid_id_map.csv"))

    if not mapping_files_json and not mapping_files_csv:
        raise FileNotFoundError(
            "Could not find mapping_to_endoscapes.json or endoscapes_vid_id_map.csv under mapping-dir."
        )

    # Try CAMMA composition first: JSON gives cholec_id→endo_orig_vid_id,
    # CSV gives endo_orig_vid_id→endo_public_vid_id. Compose to get
    # str(public_vid_id) → canonical_id.
    pairs: Dict[str, str] = {}
    if mapping_files_json and mapping_files_csv:
        pairs = _compose_camma_mapping(mapping_files_json[0], mapping_files_csv[0])

    # Fallback: try generic parsing if composition yielded nothing
    # (handles hypothetical pre-processed formats).
    if not pairs:
        for path in mapping_files_json:
            pairs.update(_parse_mapping_json(path))
        for path in mapping_files_csv:
            pairs.update(_parse_mapping_csv(path))

    if not pairs:
        raise RuntimeError("Endoscapes mapping files were found but no public_id -> canonical_id pairs were parsed.")
    return pairs


def _compose_camma_mapping(json_path: Path, csv_path: Path) -> Dict[str, str]:
    """Compose CAMMA mapping files to get {str(public_vid_id): canonical_id}.

    JSON format: {"cholec_id_str": endo_orig_vid_id_int, ...}
    CSV format: orig_vid_id,public_vid_id (both int, header row)

    Composition: cholec_id → endo_orig_vid_id (JSON) → endo_public_vid_id (CSV)
    Result: {str(public_vid_id): normalize_vid(cholec_id)}
    """
    # Step 1: Parse JSON → {cholec_id_int: endo_orig_vid_id_int}
    obj = json.loads(read_text(json_path))
    if not isinstance(obj, dict):
        return {}
    cholec_to_orig: Dict[int, int] = {}
    for k, v in obj.items():
        try:
            cholec_id = int(k)
            orig_vid_id = int(v)
            cholec_to_orig[cholec_id] = orig_vid_id
        except (ValueError, TypeError):
            continue

    if not cholec_to_orig:
        return {}

    # Step 2: Parse CSV → {endo_orig_vid_id_int: endo_public_vid_id_int}
    orig_to_public: Dict[int, int] = {}
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return {}
        # Normalize field names for matching
        field_map = {str(fld).strip().lower(): str(fld).strip() for fld in reader.fieldnames}
        orig_col = field_map.get("orig_vid_id")
        public_col = field_map.get("public_vid_id")
        if orig_col is None or public_col is None:
            return {}
        for row in reader:
            try:
                orig_id = int(row[orig_col])
                public_id = int(row[public_col])
                orig_to_public[orig_id] = public_id
            except (ValueError, TypeError, KeyError):
                continue

    if not orig_to_public:
        return {}

    # Step 3: Compose → {str(public_vid_id): canonical_id}
    pairs: Dict[str, str] = {}
    for cholec_id, orig_vid_id in cholec_to_orig.items():
        public_vid_id = orig_to_public.get(orig_vid_id)
        if public_vid_id is not None:
            pairs[str(public_vid_id)] = normalize_vid(cholec_id)

    return pairs


def _parse_mapping_json(path: Path) -> Dict[str, str]:
    obj = json.loads(read_text(path))
    pairs: Dict[str, str] = {}

    def visit(x: Any) -> None:
        if isinstance(x, dict):
            # Case 1: {public_id: canonical_id}
            for k, v in x.items():
                k_vid = extract_vid(k)
                v_vid = extract_vid(v) if isinstance(v, str) else None
                if k_vid is None and v_vid is not None:
                    pairs[str(k)] = v_vid
                elif k_vid is not None and isinstance(v, str) and v_vid is None:
                    pairs[str(v)] = k_vid
                elif isinstance(v, (dict, list)):
                    visit(v)
            # Case 2: explicit fields
            public_field = None
            canonical_field = None
            for key, value in x.items():
                key_l = str(key).lower()
                if key_l in {"public_id", "endoscapes_public_id", "endoscapes_id", "video_id", "vid"}:
                    public_field = str(value)
                if key_l in {"canonical_id", "cholec_id", "canonical_vid", "vid_canonical"}:
                    canonical_field = extract_vid(value) if value is not None else None
            if public_field and canonical_field:
                pairs[public_field] = canonical_field
        elif isinstance(x, list):
            for item in x:
                visit(item)

    visit(obj)
    return pairs


def _parse_mapping_csv(path: Path) -> Dict[str, str]:
    pairs: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return pairs
        fieldnames = [str(fld).strip().lower() for fld in reader.fieldnames]
        public_col = None
        canonical_col = None
        for fld in fieldnames:
            if fld in {"public_id", "endoscapes_public_id", "endoscapes_id", "video_id", "video", "vid"}:
                public_col = fld
            if fld in {"canonical_id", "cholec_id", "canonical_vid", "vid_canonical"}:
                canonical_col = fld
        for row in reader:
            row_l = {str(k).strip().lower(): v for k, v in row.items()}
            public_id = None
            canonical_id = None
            if public_col is not None:
                public_id = row_l.get(public_col)
            if canonical_col is not None:
                canonical_id = extract_vid(row_l.get(canonical_col)) if row_l.get(canonical_col) else None
            # Fallback: scan cells for a VID-like value and a non-VID value.
            if public_id is None or canonical_id is None:
                non_vid_cells: List[str] = []
                vid_cells: List[str] = []
                for val in row_l.values():
                    if val in (None, ""):
                        continue
                    val_s = str(val).strip()
                    if extract_vid(val_s) is not None:
                        vid_cells.append(val_s)
                    else:
                        non_vid_cells.append(val_s)
                if public_id is None and non_vid_cells:
                    public_id = non_vid_cells[0]
                if canonical_id is None and vid_cells:
                    canonical_id = extract_vid(vid_cells[0])
            if public_id and canonical_id:
                pairs[str(public_id).strip()] = canonical_id
    return pairs


# Individual CAMMA split files that should NOT be treated as a combined manifest.
# These are handled by load_camma_combined_split() which composes all three.
_CAMMA_INDIVIDUAL_SPLIT_FILES = {
    "cholect50_splits.json",
    "endoscapes_splits.json",
    "cholec80_splits.json",
}


def load_combined_split_manifest(mapping_dir: Path, explicit_path: Optional[Path]) -> Tuple[Dict[str, str], Optional[str]]:
    candidates: List[Path] = []
    if explicit_path is not None:
        candidates.append(explicit_path)
    candidates.extend(
        sorted(
            [
                p for p in mapping_dir.rglob("*")
                if p.is_file()
                and p.suffix.lower() in {".json", ".csv", ".txt", ".tsv"}
                and ("split" in p.name.lower() or "combined" in p.name.lower() or "assignment" in p.name.lower())
                # Exclude individual CAMMA dataset split files — they only cover
                # a single dataset and would cause assign_splits() to crash when
                # it expects coverage of all canonical IDs.
                and p.name.lower() not in _CAMMA_INDIVIDUAL_SPLIT_FILES
            ]
        )
    )

    for path in candidates:
        mapping = _parse_split_manifest(path)
        if mapping:
            return mapping, str(path)
    return {}, None


def _parse_split_manifest(path: Path) -> Dict[str, str]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        obj = json.loads(read_text(path))
        return _extract_split_pairs_from_json(obj)
    if suffix in {".csv", ".tsv", ".txt"}:
        return _extract_split_pairs_from_table(path)
    return {}


def _extract_split_pairs_from_json(obj: Any) -> Dict[str, str]:
    pairs: Dict[str, str] = {}

    # Inverted format: {"train": [id_list], "val": [...], "test": [...]}
    # List elements can be integers (bare video numbers) or strings parseable
    # by extract_vid().
    if isinstance(obj, dict):
        split_names = {"train", "val", "test"}
        if all(k in obj and isinstance(obj[k], list) for k in split_names):
            for split_name in split_names:
                for elem in obj[split_name]:
                    vid = None
                    if isinstance(elem, int):
                        vid = normalize_vid(elem)
                    else:
                        vid = extract_vid(str(elem))
                        if vid is None:
                            # Try as bare integer string
                            try:
                                vid = normalize_vid(int(elem))
                            except (ValueError, TypeError):
                                pass
                    if vid is not None:
                        pairs[vid] = split_name
            if pairs:
                return pairs

    def visit(x: Any) -> None:
        if isinstance(x, dict):
            # {"VID01": "train", ...}
            for k, v in x.items():
                maybe_vid = extract_vid(k)
                if maybe_vid is not None and isinstance(v, str) and v.lower() in {"train", "val", "test"}:
                    pairs[maybe_vid] = v.lower()
                elif isinstance(v, (dict, list)):
                    visit(v)
            # explicit fields
            vid_field = None
            split_field = None
            for k, v in x.items():
                kl = str(k).lower()
                if kl in {"canonical_id", "vid", "video_id", "id"}:
                    vid_field = extract_vid(v) if v is not None else None
                if kl in {"split", "subset", "partition"} and isinstance(v, str) and v.lower() in {"train", "val", "test"}:
                    split_field = v.lower()
            if vid_field and split_field:
                pairs[vid_field] = split_field
        elif isinstance(x, list):
            for item in x:
                visit(item)

    visit(obj)
    return pairs


def _extract_split_pairs_from_table(path: Path) -> Dict[str, str]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    pairs: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        if path.suffix.lower() == ".txt":
            lines = [line.strip() for line in f if line.strip()]
            for line in lines:
                tokens = re.split(r"[\s,]+", line)
                vid = next((extract_vid(tok) for tok in tokens if extract_vid(tok) is not None), None)
                split = next((tok.lower() for tok in tokens if tok.lower() in {"train", "val", "test"}), None)
                if vid and split:
                    pairs[vid] = split
            return pairs

        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            return pairs
        for row in reader:
            row_l = {str(k).strip().lower(): v for k, v in row.items()}
            vid = None
            split = None
            for key, value in row_l.items():
                if vid is None and value is not None:
                    vid = extract_vid(str(value))
                if split is None and isinstance(value, str) and value.lower() in {"train", "val", "test"}:
                    split = value.lower()
            if vid and split:
                pairs[vid] = split
    return pairs


def _vid_sort_key(vid: str) -> Tuple[int, str]:
    m = re.search(r"([0-9]+)$", vid)
    if m is None:
        return (sys.maxsize, vid)
    return (int(m.group(1)), vid)


def synthesize_endoscapes_canonical_id(public_id: str) -> str:
    """Synthesize a canonical ID for Endoscapes videos without a CAMMA mapping.

    This is the standard path for G7 videos: the CAMMA mapping only covers ~9
    overlapping Cholec-Endoscapes recordings.  The remaining ~192 Endoscapes-only
    videos receive ENDO_<public_id> canonical IDs.
    """
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", public_id).strip("_")
    return f"ENDO_{cleaned}"


def _parse_inverted_split_json(path: Path) -> Dict[str, List[int]]:
    """Parse ``{"train": [int,...], "val": [...], "test": [...]}`` format."""
    obj = json.loads(read_text(path))
    if not isinstance(obj, dict):
        return {}
    result: Dict[str, List[int]] = {}
    for split_name in ("train", "val", "test"):
        if split_name not in obj:
            return {}  # all three splits must be present
        ids = obj[split_name]
        if not isinstance(ids, list):
            return {}
        try:
            result[split_name] = [int(x) for x in ids]
        except (ValueError, TypeError):
            return {}
    return result


def load_camma_combined_split(
    mapping_dir: Path,
    public_to_canonical: Dict[str, str],
    allow_synthesized: bool,
) -> Tuple[Dict[str, str], Optional[str]]:
    """Compute the CAMMA combined split strategy.

    Implements the authoritative split assignment from the proposal: preserve
    CholecT50 and Endoscapes official splits, adjust Cholec80 to avoid
    conflicts.  Priority order: CT50 > Endo > C80 (adjusted).

    Requires all three split files in *mapping_dir*:
      ``CholecT50_splits.json``, ``Endoscapes_splits.json``,
      ``Cholec80_splits.json``

    Returns ``({canonical_id: split}, source_description)`` or ``({}, None)``
    if any of the required files are missing or unparseable.
    """
    ct50_path = mapping_dir / "CholecT50_splits.json"
    endo_path = mapping_dir / "Endoscapes_splits.json"
    c80_path = mapping_dir / "Cholec80_splits.json"

    if not (ct50_path.exists() and endo_path.exists() and c80_path.exists()):
        return {}, None

    ct50_splits = _parse_inverted_split_json(ct50_path)
    endo_splits = _parse_inverted_split_json(endo_path)
    c80_splits = _parse_inverted_split_json(c80_path)

    if not ct50_splits or not endo_splits or not c80_splits:
        return {}, None

    # Load orig_vid_id -> public_vid_id from endoscapes_vid_id_map.csv
    csv_files = list(mapping_dir.rglob("endoscapes_vid_id_map.csv"))
    if not csv_files:
        return {}, None

    orig_to_public: Dict[int, int] = {}
    with csv_files[0].open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is not None:
            field_map = {str(fld).strip().lower(): str(fld).strip() for fld in reader.fieldnames}
            orig_col = field_map.get("orig_vid_id")
            public_col = field_map.get("public_vid_id")
            if orig_col and public_col:
                for row in reader:
                    try:
                        orig_to_public[int(row[orig_col])] = int(row[public_col])
                    except (ValueError, TypeError, KeyError):
                        continue

    if not orig_to_public:
        return {}, None

    result: Dict[str, str] = {}

    # Priority 1: CholecT50 (highest priority — preserves official CT50 split)
    for split_name, vid_ids in ct50_splits.items():
        for vid_num in vid_ids:
            canonical_id = normalize_vid(vid_num)
            result[canonical_id] = split_name

    # Priority 2: Endoscapes (values in split file are orig_vid_ids)
    for split_name, orig_ids in endo_splits.items():
        for orig_id in orig_ids:
            public_id = orig_to_public.get(orig_id)
            if public_id is None:
                continue  # orig_id not in CSV — skip
            public_str = str(public_id)
            canonical_id = public_to_canonical.get(public_str)
            if canonical_id is None:
                if not allow_synthesized:
                    continue
                canonical_id = synthesize_endoscapes_canonical_id(public_str)
            if canonical_id not in result:  # CT50 takes priority
                result[canonical_id] = split_name

    # Priority 3: Cholec80 (lowest — only assigns G6 videos not covered above)
    for split_name, vid_ids in c80_splits.items():
        for vid_num in vid_ids:
            canonical_id = normalize_vid(vid_num)
            if canonical_id not in result:  # CT50 and Endo take priority
                result[canonical_id] = split_name

    source = f"camma_strategy:{ct50_path.name}+{endo_path.name}+{c80_path.name}"
    return result, source


def build_rows(
    cholec80: Dict[str, VideoProbe],
    cholect50: Dict[str, VideoProbe],
    endoscapes: Dict[str, VideoProbe],
    public_to_canonical: Dict[str, str],
    cvs_xlsx: Optional[Path],
    allow_synthesized_endoscapes_ids: bool,
) -> Dict[str, RegistryRow]:
    rows: Dict[str, RegistryRow] = {}

    def ensure_row(canonical_id: str) -> RegistryRow:
        if canonical_id not in rows:
            rows[canonical_id] = RegistryRow(canonical_id=canonical_id)
        return rows[canonical_id]

    # Cholec80 contributes canonical IDs directly.
    for canonical_id, probe in cholec80.items():
        row = ensure_row(canonical_id)
        row.in_cholec80 = True
        row.has_cholec80_cvs = (cvs_xlsx is not None)
        row.cholec80_tool_presence = bool(probe.extra_paths.get("tool_presence_files"))
        row.frame_counts["cholec80"] = probe.frame_count or 0
        row.source_ids["cholec80_video_id"] = probe.dataset_video_id
        if probe.frames_dir is not None:
            row.file_paths["cholec80_frames_dir"] = probe.frames_dir
        if probe.label_paths:
            row.file_paths["cholec80_phase_files"] = probe.label_paths
        if probe.extra_paths:
            for k, v in probe.extra_paths.items():
                row.file_paths[f"cholec80_{k}"] = v
        if cvs_xlsx is not None:
            row.file_paths["cholec80_cvs_xlsx"] = str(cvs_xlsx)

    # CholecT50 contributes canonical IDs directly.
    for canonical_id, probe in cholect50.items():
        row = ensure_row(canonical_id)
        row.in_cholect50 = True
        row.frame_counts["cholect50"] = probe.frame_count or 0
        row.source_ids["cholect50_video_id"] = probe.dataset_video_id
        if probe.frames_dir is not None:
            row.file_paths["cholect50_frames_dir"] = probe.frames_dir
        if probe.label_paths:
            row.file_paths["cholect50_json_files"] = probe.label_paths
        if probe.extra_paths:
            for k, v in probe.extra_paths.items():
                row.file_paths[f"cholect50_{k}"] = v

    # Endoscapes uses public_id -> canonical_id mapping.
    for public_id, probe in endoscapes.items():
        canonical_id = probe.canonical_id or public_to_canonical.get(public_id)
        if canonical_id is None:
            if not allow_synthesized_endoscapes_ids:
                raise RuntimeError(
                    f"Endoscapes public id {public_id!r} has no canonical mapping. Supply the full CAMMA mapping or use --allow-synthesized-endoscapes-ids."
                )
            canonical_id = synthesize_endoscapes_canonical_id(public_id)
        row = ensure_row(canonical_id)
        row.in_endoscapes = True
        row.has_endoscapes_cvs = True
        row.has_endoscapes_bbox = True
        row.frame_counts["endoscapes"] = probe.frame_count or 0
        public_ids = row.source_ids.setdefault("endoscapes_public_ids", [])
        if public_id not in public_ids:
            public_ids.append(public_id)
        if probe.frames_dir is not None:
            dirs = row.file_paths.setdefault("endoscapes_frames_dirs", [])
            if probe.frames_dir not in dirs:
                dirs.append(probe.frames_dir)
        if probe.label_paths:
            row.file_paths.setdefault("endoscapes_metadata_files", [])
            for p in probe.label_paths:
                if p not in row.file_paths["endoscapes_metadata_files"]:
                    row.file_paths["endoscapes_metadata_files"].append(p)

    # Finalize labels and groups.
    for row in rows.values():
        row.coverage_group = determine_coverage_group(row)
        row.labels_available = determine_labels(row)
        if row.in_cholec80 and row.in_cholect50:
            c80_n = row.frame_counts.get("cholec80")
            ct50_n = row.frame_counts.get("cholect50")
            if c80_n and ct50_n and c80_n != ct50_n:
                row.notes.append(
                    f"frame_count_mismatch: cholec80={c80_n}, cholect50={ct50_n}"
                )
    return rows


def determine_coverage_group(row: RegistryRow) -> str:
    key = (row.in_cholect50, row.in_cholec80, row.in_endoscapes)
    mapping = {
        (True, True, True): "G1",
        (True, True, False): "G2",
        (True, False, True): "G3",
        (False, True, True): "G4",
        (True, False, False): "G5",
        (False, True, False): "G6",
        (False, False, True): "G7",
    }
    if key not in mapping:
        raise ValueError(f"Invalid source membership for {row.canonical_id}: {key}")
    return mapping[key]


def determine_labels(row: RegistryRow) -> List[str]:
    labels: Set[str] = set()
    if row.in_cholect50:
        labels.update({TRIPLET_LABEL, INSTRUMENT_LABEL, VERB_LABEL, TARGET_LABEL, PHASE_LABEL})
    if row.in_cholec80:
        labels.add(PHASE_LABEL)
        if row.cholec80_tool_presence:
            labels.add(TOOL_PRESENCE_LABEL)
    if row.has_cholec80_cvs:
        labels.add(CVS_CHOLEC80_LABEL)
    if row.has_endoscapes_cvs:
        labels.add(CVS_ENDOSCAPES_LABEL)
    if row.has_endoscapes_bbox:
        labels.add(ANATOMY_BBOX_LABEL)

    canonical_order = [
        TRIPLET_LABEL,
        INSTRUMENT_LABEL,
        VERB_LABEL,
        TARGET_LABEL,
        PHASE_LABEL,
        TOOL_PRESENCE_LABEL,
        CVS_CHOLEC80_LABEL,
        CVS_ENDOSCAPES_LABEL,
        ANATOMY_BBOX_LABEL,
    ]
    return [x for x in canonical_order if x in labels]


def assign_splits(
    rows: Dict[str, RegistryRow],
    combined_manifest: Dict[str, str],
    manifest_path: Optional[str],
    seed: int,
) -> None:
    if combined_manifest:
        for canonical_id, row in rows.items():
            split = combined_manifest.get(canonical_id)
            if split is None:
                raise RuntimeError(
                    f"Combined split manifest is present but canonical_id {canonical_id} is missing from it."
                )
            if split not in {"train", "val", "test"}:
                raise RuntimeError(f"Unsupported split value for {canonical_id}: {split!r}")
            row.split = split
            row.split_source = manifest_path
        return

    # Fallback split: deterministic and quota-based.
    ids_by_group: Dict[str, List[str]] = defaultdict(list)
    for canonical_id, row in rows.items():
        ids_by_group[row.coverage_group].append(canonical_id)

    for group, ids in ids_by_group.items():
        quota = FALLBACK_SPLIT_QUOTAS[group]
        assigned = allocate_group_fallback(
            group=group,
            canonical_ids=ids,
            train_n=quota["train"],
            val_n=quota["val"],
            test_n=quota["test"],
            seed=seed,
        )
        for split, split_ids in assigned.items():
            for canonical_id in split_ids:
                rows[canonical_id].split = split
                rows[canonical_id].split_source = "proposal_fallback_quota"


def allocate_group_fallback(
    group: str,
    canonical_ids: Sequence[str],
    train_n: int,
    val_n: int,
    test_n: int,
    seed: int,
) -> Dict[str, List[str]]:
    ids = list(sorted(canonical_ids, key=_vid_sort_key))
    if len(ids) != train_n + val_n + test_n:
        raise RuntimeError(
            f"Fallback quota for {group} expects {train_n + val_n + test_n} ids, got {len(ids)}."
        )

    remaining = stable_shuffle(ids, seed=seed + int(hashlib.sha1(group.encode("utf-8")).hexdigest(), 16) % 1000)
    test_ids: List[str] = []
    val_ids: List[str] = []

    for preferred in FALLBACK_PREFERRED_TEST.get(group, []):
        if preferred in remaining and len(test_ids) < test_n:
            test_ids.append(preferred)
            remaining.remove(preferred)
    while len(test_ids) < test_n:
        test_ids.append(remaining.pop(0))

    for preferred in FALLBACK_PREFERRED_VAL.get(group, []):
        if preferred in remaining and len(val_ids) < val_n:
            val_ids.append(preferred)
            remaining.remove(preferred)
    while len(val_ids) < val_n:
        val_ids.append(remaining.pop(0))

    train_ids = list(remaining)
    if len(train_ids) != train_n:
        raise AssertionError("Internal split allocation bug.")

    return {
        "train": sorted(train_ids, key=_vid_sort_key),
        "val": sorted(val_ids, key=_vid_sort_key),
        "test": sorted(test_ids, key=_vid_sort_key),
    }


def validate_registry(
    rows: Dict[str, RegistryRow],
    strict_counts: bool,
) -> Dict[str, Any]:
    if not rows:
        raise RuntimeError("Registry is empty.")

    group_counts = Counter(row.coverage_group for row in rows.values())
    split_counts = Counter(row.split for row in rows.values())
    group_split_counts: Dict[str, Dict[str, int]] = {
        g: {"train": 0, "val": 0, "test": 0} for g in sorted(group_counts)
    }

    for row in rows.values():
        if row.split not in {"train", "val", "test"}:
            raise RuntimeError(f"Row {row.canonical_id} has invalid split {row.split!r}")
        group_split_counts[row.coverage_group][row.split] += 1
        if row.coverage_group == "G1":
            if not (row.in_cholect50 and row.in_cholec80 and row.in_endoscapes):
                raise RuntimeError(f"{row.canonical_id} mislabeled as G1")
        if row.coverage_group == "G5":
            if row.has_cholec80_cvs:
                raise RuntimeError(f"{row.canonical_id} in G5 should have no Cholec80-CVS labels")
            if row.has_endoscapes_cvs:
                raise RuntimeError(f"{row.canonical_id} in G5 should have no Endoscapes-CVS labels")
        if row.in_cholec80 and not row.has_cholec80_cvs:
            row.notes.append("no_cvs_xlsx_provided")
        if row.in_cholec80 and not row.cholec80_tool_presence:
            row.notes.append("no_tool_presence_files_found")
        if row.in_endoscapes and not (row.has_endoscapes_cvs and row.has_endoscapes_bbox):
            raise RuntimeError(f"{row.canonical_id} in Endoscapes must have CVS+bbox coverage")

    if strict_counts:
        if group_counts != EXPECTED_GROUP_COUNTS:
            raise RuntimeError(
                "Coverage-group counts do not match the proposal expectation. "
                f"Expected {EXPECTED_GROUP_COUNTS}, got {dict(group_counts)}"
            )
        if sum(group_counts.values()) != 277:
            raise RuntimeError(f"Expected 277 canonical videos, got {sum(group_counts.values())}")

    # Proposal fallback count expectations.
    fallback_counts_ok = all(
        group_split_counts[g] == FALLBACK_SPLIT_QUOTAS[g] for g in FALLBACK_SPLIT_QUOTAS if g in group_split_counts
    )

    summary = {
        "n_records": len(rows),
        "group_counts": dict(sorted(group_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "group_split_counts": {g: group_split_counts[g] for g in sorted(group_split_counts)},
        "fallback_counts_match_proposal": fallback_counts_ok,
    }
    return summary


def make_registry_payload(
    rows: Dict[str, RegistryRow],
    summary: Dict[str, Any],
    args: argparse.Namespace,
    public_to_canonical_count: int,
    split_manifest_path: Optional[str],
) -> Dict[str, Any]:
    return {
        "schema_version": "surgcast_registry_v1",
        "build_config": {
            "cholec80_root": str(args.cholec80_root),
            "cholect50_root": str(args.cholect50_root),
            "endoscapes_root": str(args.endoscapes_root),
            "mapping_dir": str(args.mapping_dir),
            "cvs_xlsx": str(args.cvs_xlsx) if args.cvs_xlsx is not None else None,
            "split_manifest": split_manifest_path,
            "strict_counts": bool(args.strict_counts),
            "allow_synthesized_endoscapes_ids": bool(args.allow_synthesized_endoscapes_ids),
            "seed": int(args.seed),
            "n_endoscapes_mapping_pairs": int(public_to_canonical_count),
        },
        "summary": summary,
        "records": {k: rows[k].to_dict() for k in sorted(rows, key=_vid_sort_key)},
    }


def write_registry_summary_csv(rows: Dict[str, RegistryRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "canonical_id",
        "split",
        "split_source",
        "coverage_group",
        "in_cholec80",
        "in_cholect50",
        "in_endoscapes",
        "has_cholec80_cvs",
        "cholec80_tool_presence",
        "has_endoscapes_cvs",
        "has_endoscapes_bbox",
        "labels_available",
        "cholec80_frames",
        "cholect50_frames",
        "endoscapes_frames",
        "endoscapes_public_ids",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for canonical_id in sorted(rows, key=_vid_sort_key):
            row = rows[canonical_id]
            writer.writerow(
                {
                    "canonical_id": row.canonical_id,
                    "split": row.split,
                    "split_source": row.split_source,
                    "coverage_group": row.coverage_group,
                    "in_cholec80": int(row.in_cholec80),
                    "in_cholect50": int(row.in_cholect50),
                    "in_endoscapes": int(row.in_endoscapes),
                    "has_cholec80_cvs": int(row.has_cholec80_cvs),
                    "cholec80_tool_presence": int(row.cholec80_tool_presence),
                    "has_endoscapes_cvs": int(row.has_endoscapes_cvs),
                    "has_endoscapes_bbox": int(row.has_endoscapes_bbox),
                    "labels_available": "|".join(row.labels_available),
                    "cholec80_frames": row.frame_counts.get("cholec80", 0),
                    "cholect50_frames": row.frame_counts.get("cholect50", 0),
                    "endoscapes_frames": row.frame_counts.get("endoscapes", 0),
                    "endoscapes_public_ids": "|".join(row.source_ids.get("endoscapes_public_ids", [])),
                    "notes": "|".join(row.notes),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cholec80-root", type=Path, required=True)
    parser.add_argument("--cholect50-root", type=Path, required=True)
    parser.add_argument("--endoscapes-root", type=Path, required=True)
    parser.add_argument("--mapping-dir", type=Path, required=True)
    parser.add_argument("--cvs-xlsx", type=Path, default=None)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True, help="Path to registry.json")
    parser.add_argument("--out-summary-csv", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--strict-counts",
        action="store_true",
        default=False,
        help="Require the proposal counts exactly: G1=3, G2=42, G3=3, G4=3, G5=2, G6=32, G7=192, total=277.",
    )
    parser.add_argument(
        "--allow-synthesized-endoscapes-ids",
        action="store_true",
        default=True,
        help="Synthesize ENDO_<public_id> canonical ids for the ~192 G7 videos "
        "that have no CAMMA mapping. This is the standard path (default: True).",
    )
    parser.add_argument(
        "--no-allow-synthesized-endoscapes-ids",
        action="store_false",
        dest="allow_synthesized_endoscapes_ids",
        help="Fail instead of synthesizing canonical ids for unmapped Endoscapes videos.",
    )
    return parser.parse_args()


def _validate_dataset_root(root: Path, name: str) -> None:
    """Ensure a dataset root directory exists and is a non-empty directory."""
    if not root.exists():
        raise FileNotFoundError(
            f"{name} root does not exist: {root}\n"
            f"Check the --{name.lower().replace(' ', '-')}-root argument."
        )
    if not root.is_dir():
        raise NotADirectoryError(f"{name} root is not a directory: {root}")


def main() -> None:
    args = parse_args()

    # --- Input validation (Fix #1 & #3) ---
    _validate_dataset_root(args.cholec80_root, "cholec80")
    _validate_dataset_root(args.cholect50_root, "cholect50")
    _validate_dataset_root(args.endoscapes_root, "endoscapes")

    if args.cvs_xlsx is not None and not args.cvs_xlsx.is_file():
        raise FileNotFoundError(
            f"--cvs-xlsx path does not exist or is not a file: {args.cvs_xlsx}"
        )

    public_to_canonical = load_endoscapes_mapping(args.mapping_dir)

    # Split source priority: CAMMA strategy > single manifest file > fallback quotas
    combined_manifest, manifest_path = load_camma_combined_split(
        args.mapping_dir, public_to_canonical, args.allow_synthesized_endoscapes_ids,
    )
    if combined_manifest:
        print(f"[INFO] Using CAMMA combined split strategy ({len(combined_manifest)} assignments)")
    else:
        combined_manifest, manifest_path = load_combined_split_manifest(args.mapping_dir, args.split_manifest)
        if combined_manifest:
            print(f"[INFO] Using single split manifest: {manifest_path}")
        else:
            print("[WARN] No CAMMA split files or manifest found — using fallback quotas (191/41/45)")

    cholec80 = discover_cholec80(args.cholec80_root)
    cholect50 = discover_cholect50(args.cholect50_root)
    endoscapes = discover_endoscapes(args.endoscapes_root, public_to_canonical)

    # Post-discovery sanity checks: ensure roots actually contained data.
    if not cholec80:
        raise RuntimeError(
            f"Cholec80 root {args.cholec80_root} exists but no videos were discovered. "
            f"Check that the directory contains video folders with frames."
        )
    if not cholect50:
        raise RuntimeError(
            f"CholecT50 root {args.cholect50_root} exists but no videos were discovered. "
            f"Check that the directory contains video folders with frames/labels."
        )
    if not endoscapes:
        raise RuntimeError(
            f"Endoscapes root {args.endoscapes_root} exists but no videos were discovered. "
            f"Check that the directory contains train/val/test split directories or all_metadata.csv."
        )
    # Warn about zero-frame Endoscapes probes — indicates the root was found
    # but the actual image data is missing or mislocated.
    zero_frame_endo = [
        pid for pid, probe in endoscapes.items()
        if probe.frame_count is None or probe.frame_count == 0
    ]
    if zero_frame_endo:
        n_zero = len(zero_frame_endo)
        n_total = len(endoscapes)
        if n_zero == n_total:
            raise RuntimeError(
                f"All {n_total} Endoscapes probes have 0 frames. "
                f"The root {args.endoscapes_root} appears to lack image data and all_metadata.csv. "
                f"Expected layout: {{root}}/train/*.jpg, {{root}}/all_metadata.csv, etc."
            )
        print(
            f"[WARN] {n_zero}/{n_total} Endoscapes videos have 0 discovered frames. "
            f"First few: {zero_frame_endo[:5]}"
        )

    # Check for metadata-only Endoscapes probes: frame_count > 0 (from
    # all_metadata.csv) but no actual image files on disk (frames_dir is None).
    # This catches the case where someone provides a root with only the CSV.
    metadata_only_endo = [
        pid for pid, probe in endoscapes.items()
        if probe.frames_dir is None and probe.frame_count is not None and probe.frame_count > 0
    ]
    if metadata_only_endo:
        n_meta_only = len(metadata_only_endo)
        n_total = len(endoscapes)
        if args.strict_counts and n_meta_only > 0:
            raise RuntimeError(
                f"{n_meta_only}/{n_total} Endoscapes probes have frame counts from "
                f"all_metadata.csv but no actual image files on disk. "
                f"Under --strict-counts this is not allowed. "
                f"First few: {metadata_only_endo[:5]}"
            )
        if n_meta_only > n_total // 2:
            raise RuntimeError(
                f"{n_meta_only}/{n_total} Endoscapes probes have frame counts from "
                f"all_metadata.csv but no actual image files on disk. "
                f"The root {args.endoscapes_root} appears to be missing most image data. "
                f"Expected layout: {{root}}/train/*.jpg, {{root}}/val/*.jpg, {{root}}/test/*.jpg"
            )
        print(
            f"[WARN] {n_meta_only}/{n_total} Endoscapes videos have metadata frame counts "
            f"but no image files found on disk. First few: {metadata_only_endo[:5]}"
        )

    rows = build_rows(
        cholec80=cholec80,
        cholect50=cholect50,
        endoscapes=endoscapes,
        public_to_canonical=public_to_canonical,
        cvs_xlsx=args.cvs_xlsx,
        allow_synthesized_endoscapes_ids=args.allow_synthesized_endoscapes_ids,
    )

    assign_splits(rows, combined_manifest=combined_manifest, manifest_path=manifest_path, seed=args.seed)
    summary = validate_registry(rows, strict_counts=args.strict_counts)

    # Surface discrepancy if CAMMA strategy produces different counts than fallback
    split_counts = summary.get("split_counts", {})
    if manifest_path and "camma_strategy:" in str(manifest_path):
        actual = (split_counts.get("train", 0), split_counts.get("val", 0), split_counts.get("test", 0))
        if actual != (191, 41, 45):
            print(
                f"[INFO] CAMMA strategy split counts {actual[0]}/{actual[1]}/{actual[2]} "
                f"differ from fallback quota estimate 191/41/45 — this is expected."
            )

    payload = make_registry_payload(
        rows=rows,
        summary=summary,
        args=args,
        public_to_canonical_count=len(public_to_canonical),
        split_manifest_path=manifest_path,
    )

    json_dump(payload, args.out)
    if args.out_summary_csv is not None:
        write_registry_summary_csv(rows, args.out_summary_csv)

    print("[OK] wrote", args.out)
    if args.out_summary_csv is not None:
        print("[OK] wrote", args.out_summary_csv)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
