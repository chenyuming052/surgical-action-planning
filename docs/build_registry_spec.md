# build_registry.py detailed implementation spec

## 1. Contract

`build_registry.py` must output a single `registry.json` that merges Cholec80,
CholecT50, Cholec80-CVS, and Endoscapes at the physical-recording level.

The canonical unit is one physical surgery video, identified by `canonical_id`
(e.g. `VID66`). The registry is the source of truth for:

- overlap resolution
- leakage-safe split assignment
- coverage-group assignment
- label availability
- per-dataset frame counts
- dataset-specific file paths

## 2. Top-level JSON layout

```json
{
  "schema_version": "surgcast_registry_v1",
  "build_config": { ... },
  "summary": {
    "n_records": 277,
    "group_counts": {"G1": 3, "G2": 42, "G3": 3, "G4": 3, "G5": 2, "G6": 32, "G7": 192},
    "split_counts": {"train": 191, "val": 41, "test": 45},
    "group_split_counts": { ... },
    "fallback_counts_match_proposal": true
  },
  "records": {
    "VID66": { ... },
    "VID67": { ... }
  }
}
```

## 3. Per-record schema

Each `records[canonical_id]` entry contains:

- `canonical_id`
- `split`: `train | val | test`
- `split_source`: `camma_manifest:<path>` or `proposal_fallback_quota`
- `coverage_group`: `G1..G7`
- `in_cholec80`
- `in_cholect50`
- `in_endoscapes`
- `has_cholec80_cvs`
- `cholec80_tool_presence`
- `has_endoscapes_cvs`
- `has_endoscapes_bbox`
- `labels_available`
- `frame_counts`: `{cholec80, cholect50, endoscapes}`
- `source_ids`: dataset-specific ids
- `file_paths`: dataset-specific roots or files
- `notes`: warnings such as frame-count mismatch across overlapping datasets

## 4. Coverage groups

Coverage groups are computed only from source membership:

- `G1`: CholecT50 + Cholec80 + Endoscapes
- `G2`: CholecT50 + Cholec80
- `G3`: CholecT50 + Endoscapes
- `G4`: Cholec80 + Endoscapes
- `G5`: CholecT50 only
- `G6`: Cholec80 only
- `G7`: Endoscapes only

Expected proposal counts:

- `G1=3`
- `G2=42`
- `G3=3`
- `G4=3`
- `G5=2`
- `G6=32`
- `G7=192`
- total `277`

## 5. Overlap logic

### 5.1 Cholec80 <-> CholecT50

Use direct `VIDxx` matching on canonical ids.

### 5.2 Cholec80 / CholecT50 <-> Endoscapes

Use:

- `mapping_to_endoscapes.json`
- `endoscapes_vid_id_map.csv`

The loader should parse both and produce `public_id -> canonical_id`.

### 5.3 Cholec80-CVS

Treat Cholec80-CVS as an annotation layer on top of Cholec80.

Rules:

- `has_cholec80_cvs` is `true` only when `--cvs-xlsx` is provided
- If `in_cholec80` and `has_cholec80_cvs` is `false`, a note `no_cvs_xlsx_provided` is added
- Cholec80-CVS inherits `canonical_id` from Cholec80
- Cholec80-CVS inherits `split` from Cholec80

## 6. Split rules

### 6.1 Priority order

0. **CAMMA combined split strategy** (authoritative): requires `CholecT50_splits.json`,
   `Endoscapes_splits.json`, and `Cholec80_splits.json` in the mapping directory.
   Preserves CT50 and Endoscapes official splits, adjusts C80 to avoid conflicts.
   Priority: CT50 > Endo > C80. Produces 168/48/61 (train/val/test).
1. If a single CAMMA combined split manifest file exists, use it.
2. Otherwise use the deterministic proposal fallback quotas.

### 6.2 Fallback quotas (last resort when CAMMA split files unavailable)

- `G1`: train 2, val 1, test 0
- `G2`: train 28, val 6, test 8
- `G3`: train 2, val 0, test 1
- `G4`: train 2, val 1, test 0
- `G5`: train 1, val 0, test 1
- `G6`: train 22, val 5, test 5
- `G7`: train 134, val 28, test 30

### 6.3 Deterministic anchors used by fallback

- `G5`: prefer `VID111` for test
- `G3`: prefer one of `VID110`, `VID103`, `VID96` for test, in that order
- `G1`: prefer `VID70`, then `VID68`, then `VID66` for val
- `G4`: prefer `VID72`, then `VID71`, then `VID67` for val

All remaining ids are filled by a stable hash-based order using the provided seed.

## 7. Validation rules

The script must fail if any of the following is violated:

- a record is in zero datasets
- a record gets an invalid split
- a `G1` record is missing one of the three datasets
- a `G5` record has any CVS label source
- an Endoscapes record does not have both CVS and bbox coverage

Soft checks (add notes instead of failing):

- a Cholec80 record without `has_cholec80_cvs` → note `no_cvs_xlsx_provided`
- a Cholec80 record without `cholec80_tool_presence` → note `no_tool_presence_files_found`

If `--strict-counts` is enabled, also fail when group counts differ from the
proposal expectation.

## 8. Known proposal inconsistency

The proposal contains both of the following statements:

- split principle text: official Endoscapes test has zero overlap with Cholec80/CholecT50
- fallback quota table: `G3` contributes one test video

Because those two statements are not jointly satisfiable without an external
combined-split manifest, the script makes the following choice:

- combined split manifest wins if available
- otherwise the proposal fallback quota table wins exactly

## 9. Example command

```bash
python scripts/build_registry.py \
  --cholec80-root /data/Cholec80 \
  --cholect50-root /data/CholecT50 \
  --endoscapes-root /data/Endoscapes2023 \
  --mapping-dir /data/camma_mapping \
  --cvs-xlsx /data/Cholec80-CVS/surgeons_annotations.xlsx \
  --out artifacts/registry.json \
  --out-summary-csv artifacts/registry_summary.csv \
  --strict-counts
```

## 10. What to inspect after the run

- `summary.group_counts`
- `summary.split_counts`
- `summary.group_split_counts`
- any `notes` values containing `frame_count_mismatch`
- whether `split_source` came from CAMMA manifest or fallback quota
