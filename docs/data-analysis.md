# Surgical Video Dataset Audit for Action Planning and Navigation

Date: 2026-03-13
Target task: action planning and navigation in image-guided surgery
Audit scope: 10 local datasets under `/yuming/data`

## Executive Summary

This document summarizes on-disk counts and official-source cross-checks for ten local surgical video datasets relevant to action planning and navigation.

1. **Cholec80** is a full-procedure workflow dataset for laparoscopic cholecystectomy, and its dense phase labels are not aligned one-to-one with the extracted 1 fps frame folders.
2. **Cholec80-CVS** adds CVS safety annotations on top of all **80 Cholec80 videos**, but the local copy is only a raw XLSX workbook; the official code links it back to Cholec80, truncates to the last 15% before clip/cut, and binarizes criterion scores.
3. **CholecT50** is a local source of next-action semantics via `<instrument, verb, target>` triplets, and the local copy matches the official **Release 2.0** description: no usable bounding boxes are present.
4. **CholecTrack20** is a full-procedure spatial tool-dynamics dataset with **35,009 annotated frames**, **65,247 detections**, and **2,624 per-video trajectories** across the three tracking perspectives.
5. **Endoscapes2023** contains **58,813** ROI JPGs, and `test/annotation_coco_vid.json` omits **228** frame entries for videos **190** and **194**.
6. **CholecInstanceSeg** and **CholecSeg8k** are spatial-supervision datasets and are not direct action-planning datasets.
7. **GraSP** provides hierarchical procedure representation annotations, and the local copy contains **13** cases from robotic prostatectomy.
8. **AutoLaparo** is present locally at **101 GB** with 21 full-length laparoscopic hysterectomy videos, 300 motion-prediction clips, and 1,800 segmentation frames.
9. **HeiChole** is a multi-task laparoscopic cholecystectomy dataset from the EndoVis 2019 challenge with **24 videos**, **1,836,369** annotated frames, and concurrent frame-level annotations for 7 surgical phases, 7 instrument categories (21 fine-grained types), 4 action types, and 5-dimensional surgical skill scores. Phase annotations are Cholec80-compatible.

## 1. Scope and Coverage

This audit covers 10 local datasets under `/yuming/data`.

- `AutoLaparo` is part of the local dataset set.
- `Cholec80-CVS` is annotation-only locally and depends on the original Cholec80 videos plus `phase_annotations/` for the official preprocessing pipeline.
- `CholecTrack20` counts in this document are dataset-wide totals unless a split is explicitly named.
- `Endoscapes2023` contains 58,813 local JPGs; the 228-frame discrepancy is confined to `test/annotation_coco_vid.json`.
- `HeiChole` is a standalone cholecystectomy dataset from 3 Heidelberg-affiliated hospitals; no video-ID overlap with any CAMMA dataset.

## 2. Verified Local Snapshot

| Dataset | Local footprint | Verified local content | Relevant tasks | Main caveat |
|---|---:|---|---|---|
| Cholec80 | 97G | 80 video dirs, 184,498 PNGs, 184,498 tool rows, 4,612,532 phase rows | full-procedure workflow modeling | phase labels require resampling/alignment to 1 fps frames |
| Cholec80-CVS | 32K | 1 XLSX, 572 annotation rows, 80 videos, 62,760 derived 5 fps labels | CVS safety gating and clip/cut readiness | local copy is raw workbook only; official code depends on Cholec80 and binarizes criterion scores |
| CholecT50 | 60G | 50 video dirs, 50 label JSONs, 100,863 PNGs | fine-grained action semantics | local Release 2.0 has no bounding boxes |
| CholecSeg8k | 3.0G | 17 videos, 8,080 annotated frames, 101 clip groups | anatomy/tissue-aware scene parsing | clip-based, not full-procedure |
| CholecInstanceSeg | 421M | 41,933 JSONs, 95 sequence folders, 85 unique video IDs, 67,085 polygons | fine tool geometry and instance masks | annotations only; image metadata fields are incomplete in many files |
| Endoscapes2023 | 6.9G | 58,813 ROI JPGs, 58,813 metadata rows | anatomy-guided navigation and safety gating | test annotation JSON misses 228 frame entries (images complete) |
| CholecTrack20 | 34G | 20 videos, 35,009 annotated frames, 65,247 detections, 20 JSONs | full-procedure tool tracking and spatial memory | verb/target labels appear only in 8 of the 14 CholecT50-overlap videos (~37% of detections) |
| GraSP | 260G | 13 cases, 3,495,415 JPGs, 8 annotation JSONs, 3,449 segmentation PNGs | hierarchical representation pretraining | local copy is partial and robotic |
| AutoLaparo | 101G | 21 videos (83,243 1fps frames), 300 motion clips, 1,800 seg frames + masks | camera-motion prediction and anatomy segmentation | laparoscopic hysterectomy, not cholecystectomy |
| HeiChole | 212G | 24 videos (SD+HD), 1,836,369 annotated frames, 4 annotation layers + skill | multi-task workflow (phase, instrument, action, skill) | Cholec80-compatible phases; official challenge split is 24 train + 9 test but only 24 training videos are publicly available; mixed 25/50 fps; extremely large video files |

## 3. Dataset-by-Dataset Audit

### 3.1 Cholec80

#### Local structure

```text
/yuming/data/cholec80/
├── frames/
├── phase_annotations/
└── tool_annotations/
```

#### Verified local facts

| Property | Value |
|---|---:|
| Video folders | 80 |
| Extracted PNG images | 184,498 |
| Tool annotation rows | 184,498 |
| Phase annotation rows | 4,612,532 |
| Mean extracted images/video | 2,306.2 |
| Median extracted images/video | 2,095 |
| Min extracted images/video | 739 |
| Max extracted images/video | 5,993 |

#### Alignment note

The local copy contains two temporal granularities:

- `frames/` and `tool_annotations/` are aligned at **1 fps**
- `phase_annotations/` are dense over the original video frame numbers

Example:

- `video01-tool.txt` frame IDs are `0, 25, 50, 75, ...`
- `video01-phase.txt` frame IDs are `0, 1, 2, 3, ...`
- `frames/video01/` contains 1,733 PNGs, exactly matching the 1 fps tool rows

Direct joining of `phase_annotations` row index `N` to extracted frame index `N` is invalid; alignment operates through the original frame IDs with explicit resampling.

#### File naming offset

Image files start at `video01_000001.png` (1-indexed), while aligned annotation frame IDs start at `0` (0-indexed). This offset affects image-label joins.

#### Phase coverage across videos

- `CalotTriangleDissection`: 80/80
- `ClippingCutting`: 80/80
- `GallbladderDissection`: 80/80
- `GallbladderPackaging`: 80/80
- `GallbladderRetraction`: 80/80
- `CleaningCoagulation`: 74/80
- `Preparation`: 71/80

#### Standard split

The official Cholec80 split (source: `Cholec80_splits.json` in the CAMMA overlap analysis repository):

| Split | Videos | Video IDs |
|---|---:|---|
| Train | 40 | 1–40 |
| Val | 8 | 41–48 |
| Test | 32 | 49–80 |

This split is needed for leakage awareness when combining Cholec80 with CVS and CholecT50 splits.

#### Task relevance

- Supports coarse temporal planning in laparoscopic cholecystectomy
- Provides dense phase supervision for phase-aware sequence models
- Lacks anatomy localization and spatial masks, which limits direct navigation supervision

### 3.2 Cholec80-CVS

#### Official scope cross-check

The Scientific Data paper and the official code release position Cholec80-CVS as a CVS-annotation layer on top of the 80 Cholec80 videos. The raw workbook stores time intervals and ordinal surgeon scores for the three Strasberg criteria:

- `two_structures`
- `cystic_plate`
- `hepatocystic_triangle`

The row-level `critical_view` flag is exactly equivalent to `Total >= 5` in the local workbook. The public baseline code does **not** train directly on this raw flag; it converts the three criteria into frame labels and later binarizes each criterion independently.

#### Local structure

```text
/yuming/data/cholec80-cvs/
└── cholec80-CVS.xlsx
```

The verified preprocessing and split logic in this section comes from the local clone of the official code repository at `/yuming/repos/CHOLEC80-CVS-PUBLIC`.

#### Verified local facts

| Property | Value |
|---|---:|
| Workbook sheets | 1 |
| Annotation rows | 572 |
| Unique videos | 80 |
| Rows with any non-zero criterion | 331 |
| Rows with `Total = 0` | 241 |
| Rows with raw `critical_view = 1` | 23 |
| Videos with any non-zero criterion | 71 |
| Videos with raw `critical_view = 1` | 16 |
| Videos with only zero rows | 9 |

Criterion-row coverage in the raw workbook:

- `two_structures`: 287 positive rows across 69 videos
- `cystic_plate`: 83 positive rows across 31 videos
- `hepatocystic_triangle`: 173 positive rows across 55 videos

#### Derived label snapshot under the official preprocessing pipeline

Applying the public preprocessing logic to the local Cholec80 phase annotations yields the following deterministic counts:

| Property | Value |
|---|---:|
| Pre-clip/cut frames at 25 fps | 2,084,872 |
| Frames retained after 85% truncation | 313,752 |
| Derived labels at 5 fps | 62,760 |
| Fixed train / val / test videos | 50 / 15 / 15 |
| Derived 5 fps frames in train / val / test | 43,850 / 9,615 / 9,295 |

Criterion balance in the derived 5 fps label set:

| Target interpretation | Positive 5 fps frames | Rate |
|---|---:|---:|
| `two_structures > 0` | 33,550 | 53.5% |
| `cystic_plate > 0` | 9,050 | 14.4% |
| `hepatocystic_triangle > 0` | 19,985 | 31.8% |
| all three criteria present after binarization | 6,245 | 10.0% |
| raw `Total >= 5` (critical view) | 2,555 | 4.1% |

The fourth row corresponds to the binary target family actually used by `colenet/cholec80csv_dataset.py`. It is materially broader than the fifth row because the loader clips score `2` to `1` and therefore treats `1+1+1` as fully positive even though the raw workbook does not mark it as `critical_view = 1`.

#### Loader caveats

- The local copy stores only the raw workbook; the official repository expects the file to be renamed to `data/surgeons_annotations.xlsx`
- The preprocessing pipeline is **not standalone**: it requires the original Cholec80 videos and `phase_annotations/`
- `get_valid_frames.py` truncates every case at the first `ClippingCutting` frame, and `annotations_2_labels.py` then discards the first **85%** of that pre-clip/cut window before sampling the remainder at **5 fps**
- `colenet/cholec80csv_dataset.py` applies `min(1, score)` to each criterion, so score `2` is collapsed to binary `1`, and the raw `critical_view` / `Total` columns are never used as direct learning targets
- The raw workbook contains **63** intervals whose end time extends beyond the first `ClippingCutting` boundary; the official script silently truncates them to the valid frame index
- The raw workbook also contains **3** malformed intervals with `final < initial`; **2** of those rows are positive (videos 06 and 48) and are silently skipped by the official labeling loop
- The public code provides both a fixed **50 / 15 / 15** split and a **16-fold** cross-validation file with 5 validation videos per fold

#### Split contrast with standard Cholec80

The CVS 50/15/15 split is **completely different** from the standard Cholec80 40/8/32 split. The CVS split intermixes video IDs across the entire 1–80 range (source: `get_training_sets.py`):

- CVS test videos: 5, 6, 7, 9, 14, 22, 24, 26, 29, 33, 35, 39, 53, 55, 61
- CVS val videos: 1, 2, 10, 16, 17, 28, 32, 46, 47, 57, 59, 63, 65, 67, 70

Cross-split leakage counts:

| Overlap | Count | Video IDs |
|---|---:|---|
| Standard Cholec80 train ∩ CVS test | 12 | 5, 6, 7, 9, 14, 22, 24, 26, 29, 33, 35, 39 |
| Standard Cholec80 test ∩ CVS train | 23 | 49–80 minus CVS val/test members |
| Standard Cholec80 test ∩ CVS val | 6 | 57, 59, 63, 65, 67, 70 |

Any pipeline that uses both Cholec80 standard splits and CVS splits must account for these overlaps.

#### Task relevance

- Provides direct CVS safety-state supervision in laparoscopic cholecystectomy
- Supports clip/cut readiness gating and safety-aware state estimation before the clipping phase
- Complements Endoscapes by focusing on CVS attainment rather than anatomy localization alone
- Does not replace full-procedure workflow supervision because it is restricted to the pre-clip/cut temporal window

### 3.3 CholecT50

#### Local structure

```text
/yuming/data/cholecT50/
├── labels/
├── videos/
├── README.md
└── label_mapping.txt
```

#### Verified local facts

| Property | Value |
|---|---:|
| Videos | 50 |
| Label JSONs | 50 |
| PNG images | 100,863 |
| Mean frames/video | 2,017.3 |
| Median frames/video | 1,970 |
| Min frames/video | 740 |
| Max frames/video | 3,946 |
| Overlap with Cholec80 | 45 videos |
| New IDs beyond Cholec80 | `VID92`, `VID96`, `VID103`, `VID110`, `VID111` |

All local image counts match each JSON's `num_frames`.

#### Annotation format

- Every frame-level annotation vector has length **15**
- Across all videos, the local copy contains **161,988** such vectors
- All bbox-like slots are `-1` in the local release

This matches the local Release 2.0 README description, which provides binary presence labels for triplets, instruments, verbs, targets, and phases, but no released bounding boxes yet.

#### Label ontology

**6 instruments:**

| ID | Instrument |
|---:|---|
| 0 | grasper |
| 1 | bipolar |
| 2 | hook |
| 3 | scissors |
| 4 | clipper |
| 5 | irrigator |

**10 verbs:**

| ID | Verb |
|---:|---|
| 0 | grasp |
| 1 | retract |
| 2 | dissect |
| 3 | coagulate |
| 4 | clip |
| 5 | cut |
| 6 | aspirate |
| 7 | irrigate |
| 8 | pack |
| 9 | null_verb |

**15 targets:**

| ID | Target |
|---:|---|
| 0 | gallbladder |
| 1 | cystic_plate |
| 2 | cystic_duct |
| 3 | cystic_artery |
| 4 | cystic_pedicle |
| 5 | blood_vessel |
| 6 | fluid |
| 7 | abdominal_wall_cavity |
| 8 | liver |
| 9 | adhesion |
| 10 | omentum |
| 11 | peritoneum |
| 12 | gut |
| 13 | specimen_bag |
| 14 | null_target |

**Triplet space:** **94 action triplets** (IDs 0–93) + **6 null triplets** (IDs 94–99, one per instrument: instrument present, null_verb, null_target) = 100 total IDs. `-1` denotes absent/unlabeled. Source: `label_mapping.txt`.

#### Annotation vector field layout

Each annotation instance is a 15-element vector. Field layout (corroborated by `docs/README-Format.md` and `files/var.png`):

| Index | Field | Domain | Note |
|---|---|---|---|
| 0 | triplet_id | 0–99 or -1 | Primary action label |
| 1 | instrument_id | 0–5 or -1 | Instrument component |
| 2 | confidence | float | 1.0 for ground truth |
| 3–6 | bbox | normalized or -1 | `[x, y, w, h]`; all -1 in Release 2.0 |
| 7 | verb_id | 0–9 or -1 | Action component |
| 8 | target_id | 0–14 or -1 | Anatomical target component |
| 9 | confidence | float | 1.0 for ground truth (mirrors field [2]) |
| 10–13 | bbox | normalized or -1 | Target `[x, y, w, h]`; all -1 in Release 2.0 |
| 14 | phase_id | 0–6 | Surgical phase label |

Critical fields for action prediction: [0], [1], [7], [8], [14].

#### Action change frequency

Definition: a "change" at a given granularity means the set of active IDs on frame `t` differs from frame `t-1`, computed over consecutive 1 fps frames across all 50 videos (100,863 frames total).

| Granularity | Total changes | Per video | Per minute |
|---|---:|---:|---:|
| Phase transitions | 289 | 5.8 | 0.17 |
| Triplet-set changes | 10,378 | 207.6 | 6.17 |
| Verb-set changes | 9,013 | 180.3 | 5.36 |
| Target-set changes | 7,912 | 158.2 | 4.71 |
| Instrument-set changes | 6,835 | 136.7 | 4.07 |
| Concurrent-instrument-count changes | 6,650 | 133.0 | 3.96 |

**Note**: "Instrument-set changes" counts frames where the set of unique active instrument IDs changes. "Concurrent-instrument-count changes" counts frames where the *number* of unique active instruments changes (a strictly weaker condition). Both are reproducible by comparing `frozenset(inst[1])` or `len(set(inst[1]))` across consecutive frames.

Phase transitions (~1 every 6 min) are too sparse for continuous prediction; triplet-set changes (~6.2/min) provide the richest signal and are the natural target granularity.

#### Multi-instance concurrency

Counting basis: 100,863 frames across 50 videos.

- Frames with >1 annotation instance (any triplet ID): **54,315** (53.9%)
- Frames with >1 **action** instance (excluding null triplets 94–99 and -1): **47,846** (47.4%)

The "post-change state" is therefore a **set** of concurrent triplets, not a single label. This affects model output head design (multi-label, not single-label) and evaluation (set-level metrics).

#### Triplet distribution

All counts exclude triplet_id = -1 (absent/unlabeled, 11,061 instances). Denominator: 150,927 labeled instances across 50 videos.

| Rank | Triplet | Instances | Share |
|---|---|---:|---:|
| 1 | T17: grasper,retract,gallbladder | 41,565 | 27.54% |
| 2 | T60: hook,dissect,gallbladder | 29,477 | 19.53% |
| 3 | T19: grasper,retract,liver | 12,975 | 8.60% |
| 4 | T58: hook,dissect,cystic_duct | 7,891 | 5.23% |

Top 3 triplets account for **55.67%** of all labeled instances. The distribution is severely long-tailed, which implies class imbalance for any state prediction task and must inform evaluation metric selection (macro F1 over accuracy) and sampling/weighting strategy.

If null triplets (94–99) are also excluded, the denominator drops to 140,494 action-only instances and the top-3 share rises to **59.80%**.

#### Phase transition patterns

Counted across all 50 CholecT50 videos (289 total transitions, 10 distinct transition types):

| From | To | Count |
|---|---|---:|
| carlot-triangle-dissection | clipping-and-cutting | 50 |
| clipping-and-cutting | gallbladder-dissection | 50 |
| preparation | carlot-triangle-dissection | 44 |
| gallbladder-dissection | gallbladder-packaging | 42 |
| gallbladder-packaging | cleaning-and-coagulation | 34 |
| cleaning-and-coagulation | gallbladder-extraction | 34 |
| gallbladder-packaging | gallbladder-extraction | 16 |
| gallbladder-dissection | cleaning-and-coagulation | 8 |
| cleaning-and-coagulation | gallbladder-packaging | 8 |
| gallbladder-extraction | cleaning-and-coagulation | 3 |

The first two transitions are universal (50/50 videos). Non-standard transitions (reverse, skip) exist in the later phases, meaning the phase sequence is not strictly linear. For a Markov prediction model, the transition matrix is sparse but not deterministic.

#### Phase ontology mismatch with Cholec80

| Cholec80 | CholecT50 |
|---|---|
| `Preparation` | `preparation` |
| `CalotTriangleDissection` | `carlot-triangle-dissection` |
| `ClippingCutting` | `clipping-and-cutting` |
| `GallbladderDissection` | `gallbladder-dissection` |
| `GallbladderPackaging` | `gallbladder-packaging` |
| `CleaningCoagulation` | `cleaning-and-coagulation` |
| `GallbladderRetraction` | `gallbladder-extraction` |

The final row is only an approximate semantic mapping. Raw string matching does not provide a valid ontology merge for these labels.

#### Task relevance

- Supports **next-action prediction**
- Provides explicit `<instrument, verb, target>` supervision
- Supplies temporal supervision for action-planning models
- Lacks spatial labels in the current local release, which limits navigation use

### 3.4 CholecSeg8k

#### Local structure

```text
/yuming/data/cholecSeg8k/
└── videoXX/
    └── videoXX_YYYYY/
        ├── frame_*_endo.png
        ├── frame_*_endo_color_mask.png
        ├── frame_*_endo_mask.png
        └── frame_*_endo_watershed_mask.png
```

#### Verified local facts

| Property | Value |
|---|---:|
| Videos | 17 |
| Annotated frames | 8,080 |
| Clip groups | 101 |
| Total files | 32,320 |
| Overlap with Cholec80 | all 17 video IDs |
| Overlap with CholecT50 | 10 video IDs |

#### Task relevance

- Provides anatomy and tissue perception supervision for navigation-related models
- Supports scene parsing, dissection-state context, and safety-aware perception
- Sparse clip-based coverage limits long-horizon planning use

### 3.5 CholecInstanceSeg

#### Local structure

```text
/yuming/data/cholecInstanceSeg/
├── cholecinstanceseg_metadata.csv
├── SYNAPSE_TABLE_QUERY_169694281.csv
├── train/
├── val/
└── test/
```

#### Verified local facts

| Property | Value |
|---|---:|
| JSON annotation files | 41,933 |
| Sequence folders | 95 |
| Unique video IDs | 85 |
| Frames from CholecT50 | 30,998 |
| Frames from CholecSeg8k | 8,080 |
| Frames from Cholec80 | 2,855 |
| Split folders | train 55 / val 18 / test 22 |
| Unique video IDs per split | train 45 / val 18 / test 22 |
| Polygon instances | 67,085 |
| Non-empty frames | 37,019 |
| Empty frames | 4,914 |

#### Important loader caveats

- The local copy contains **annotations only**; image content is resolved from the source datasets
- `Annotation_Path` in the CSV uses **Windows path separators**
- `Source_Dataset` includes the string `Cholecseg8k` instead of `CholecSeg8k`
- `imageHeight` and `imageWidth` are **not reliable global fields**
  - null in **17,930 / 41,933** JSONs
  - non-null in **24,003 / 41,933** JSONs

The local JSON set contains null `imageHeight`/`imageWidth` values in 17,930 files and non-null values in 24,003 files.

#### Instance categories actually present

- `grasper`: 40,172
- `hook`: 19,429
- `bipolar`: 2,884
- `irrigator`: 2,443
- `clipper`: 1,399
- `scissors`: 715
- `snare`: 43

#### Task relevance

- Provides precise tool-instance geometry supervision
- Supports perception branches used by downstream planning pipelines
- Does not provide direct temporal-planning supervision

### 3.6 Endoscapes2023

#### Official scope cross-check

The official repository describes:

- **201 videos**
- **58,813 ROI frames at 1 fps**
- **1,933** bounding-box annotated frames
- **493** segmentation-annotated frames

The ROI is defined during the **dissection phase and before the first clip/cut**, so the dataset emphasizes navigation and safety-related anatomy exposure rather than full-procedure coverage.

#### Local structure

```text
/yuming/data/endoscapes-2023/
├── train/ val/ test/
├── train_seg/ val_seg/ test_seg/
├── semseg/ insseg/
├── all/
├── all_metadata.csv
├── train_vids.txt / val_vids.txt / test_vids.txt
└── train_seg_vids.txt / val_seg_vids.txt / test_seg_vids.txt
```

#### Verified local facts

| Folder or file | Value |
|---|---:|
| `train/` JPGs | 36,694 |
| `val/` JPGs | 12,372 |
| `test/` JPGs | 9,747 |
| `all/` JPGs | 58,585 |
| `train_seg/` JPGs | 10,380 |
| `val_seg/` JPGs | 2,310 |
| `test_seg/` JPGs | 2,250 |
| `semseg/` PNGs | 493 |
| `all_metadata.csv` rows | 58,813 |

Selected local JSON counts:

| Subset | Images | Annotations |
|---|---:|---:|
| `train/annotation_coco.json` | 1,212 | 5,566 |
| `val/annotation_coco.json` | 409 | 1,733 |
| `test/annotation_coco.json` | 312 | 1,485 |
| `train/annotation_ds_coco.json` | 6,960 | 5,566 |
| `val/annotation_ds_coco.json` | 2,331 | 1,733 |
| `test/annotation_ds_coco.json` | 1,799 | 1,485 |
| `train/annotation_coco_vid.json` | 36,694 | 5,566 |
| `val/annotation_coco_vid.json` | 12,372 | 1,733 |
| `test/annotation_coco_vid.json` | 9,519 | 1,485 |
| `train_seg/annotation_coco.json` | 343 | 1,615 |
| `val_seg/annotation_coco.json` | 76 | 363 |
| `test_seg/annotation_coco.json` | 74 | 270 |

#### Annotation-image mismatch in test split (official release bug)

The JPG images are **complete** across all three splits: `train/` (36,694) + `val/` (12,372) + `test/` (9,747) = **58,813**, exactly matching `all_metadata.csv` and the official dataset specification. This was verified for all 201 videos individually; **no JPGs are missing from the download**.

However, `test/annotation_coco_vid.json` only indexes **9,519** images instead of the expected 9,747, omitting **228** frames from two test-split videos:

- **VID190**: 144 frames missing from annotation. The annotation retains only the 5-second DS keyframes in certain temporal windows, dropping the 4 intermediate 1fps frames between each pair. This produces 36 blocks of 4 missing frames, clustered in 5 time segments where CVS criterion C3 oscillates between 0 and non-zero at the keyframe level.
- **VID194**: 84 frames missing from annotation. A single contiguous 84-second gap (frames 18100–20175). This segment precedes the first DS keyframe (frame 20300) and has CVS = `[0.0, 0.0, 0.0]` throughout.

All 228 omitted frames have CVS = `[0.0, 0.0, 0.0]` in `all_metadata.csv`. The `train/` and `val/` annotation JSONs are fully consistent with their respective JPG counts; only `test/` is affected.

The `all/` directory in the official zip is a flat symlink-based union of the three splits' `annotation_coco_vid.json` image lists (not a listing of all on-disk images). It therefore mirrors the same 228-frame gap. For complete frame coverage, use the filesystem listing of `test/` or `all_metadata.csv` rather than the annotation JSON or `all/`.

#### Video-ID overlap with Cholec80 and CholecT50 (CAMMA official mapping)

Endoscapes uses its own public video-ID numbering (1–201) that is **not** the same as the Cholec80/CholecT50 VID## system. Overlap resolution operates through the two-step physical-recording mapping provided by the [CAMMA overlap analysis repository](https://github.com/CAMMA-public/VideoID-Overlap-Analysis-of-Cholecystectomy-Datasets): `mapping_to_endoscapes.json` (Cholec public VID → Endoscapes internal ID) + `endoscapes_vid_id_map.csv` (Endoscapes internal ID → Endoscapes public ID).

True overlaps by split:

| Endoscapes split | Overlapping dataset | Overlap count | Overlapping Cholec VID IDs | Endoscapes public ID |
|---|---|---:|---|---|
| train | Cholec80-test | 5 | 67, 68, 70, 71, 72 | 1, 2, 3, 4, 7 |
| val | Cholec80-test | 1 | 66 | 121 |
| test | Cholec80 (any) | 0 | — | — |
| train | CholecT50-train | 4 | 68, 70, 96, 110 | 2, 3, 10, 33 |
| val | CholecT50-train | 2 | 66, 103 | 121, 127 |
| test | CholecT50 (any) | 0 | — | — |

Total unique overlapping videos: **6 with Cholec80**, **6 with CholecT50**. All overlaps are confined to Endoscapes train/val; the **Endoscapes test split is clean** against both datasets.

Endoscapes public IDs (1–201) and Cholec80/CholecT50 VID identifiers occupy different namespaces and are linked through the CAMMA mapping files listed above.

#### Loader caveats

- `train_vids.txt`, `val_vids.txt`, `test_vids.txt`, and the `*_seg_vids.txt` files are stored as scientific-notation strings; parsing requires float conversion followed by int casting
- `train_seg/` in the local copy also contains `annotation_coco_vid.json` and `annotation_ds_coco.json`; each segmentation split therefore includes multiple JSON files

#### Task relevance

- Provides anatomy-and-safety supervision for navigation
- Covers gallbladder, cystic duct, cystic artery, cystic plate, HCTD, and tool-aware reasoning targets
- The temporal scope is limited to the ROI before clip/cut rather than full-procedure coverage

### 3.7 CholecTrack20

#### Official scope cross-check

The official repository describes CholecTrack20 as a **20-video** laparoscopic cholecystectomy dataset with:

- **35,009** annotated frames
- **65,247** tool detections
- **three tracking perspectives**
- **eight visual challenge labels**

#### Local structure

```text
/yuming/data/cholecTrack20/
├── Training/    # 10 videos, PNG frames + JSON
├── Validation/  # 2 videos, PNG frames + JSON
└── Testing/     # 8 videos, MP4 + JSON
```

#### Verified local facts

| Split | Videos | Frames | Detections | PNG frame dirs | MP4 files |
|---|---:|---:|---:|---:|---:|
| Training | 10 | 16,948 | 30,648 | 10 | 0 |
| Validation | 2 | 2,779 | 4,605 | 2 | 0 |
| Testing | 8 | 15,282 | 29,994 | 0 | 8 |
| Total | 20 | 35,009 | 65,247 | 12 | 8 |

#### Annotation-schema notes

Each detection record includes:

- `instrument`
- `tool_bbox`
- `phase`
- `operator`
- `intraoperative_track`
- `intracorporeal_track`
- `visibility_track`
- challenge flags

Schema details:

- `tool_bbox` uses normalized `tlwh = [top_left_x, top_left_y, box_width, box_height]`
- operator IDs are:
  - `0`: `null`
  - `1`: `main-surgeon-left-hand (MSLH)`
  - `2`: `assistant-surgeon-right-hand (ASRH)`
  - `3`: `main-surgeon-right-hand (MSRH)`
- the **8 challenge flags** are:
  - `bleeding`
  - `blurred`
  - `smoke`
  - `crowded`
  - `occluded`
  - `reflection`
  - `stainedlens`
  - `undercoverage`
- `visible` is a visibility state flag, not one of the challenge labels
- `verb` and `target` are non-`-1` in 24,122 / 65,247 detections; `triplet` is non-`-1` in 24,123 detections, of which 1,129 carry the special value `-2`. These labels appear only in 8 videos (VID06, VID23, VID25, VID92, VID96, VID103, VID110, VID111), all within the 14-video CholecT50-overlap subset; the remaining 12 videos have all three fields as `-1`

#### Trajectory counts

Per-video trajectory totals across the full dataset are:

| Perspective | Total trajectories across all 20 videos |
|---|---:|
| Intraoperative | 175 |
| Intracorporeal | 507 |
| Visibility | 1,942 |
| Sum | 2,624 |

The often-quoted counts:

- `70` intraoperative
- `247` intracorporeal
- `916` visibility

correspond to the **testing split only**, not the full dataset.

#### Frame-index caveat

JSON keys are original 25 fps frame numbers. Consecutive keys are usually separated by **25**, but larger multiples also occur because some sampled seconds are absent.

#### Video-ID overlap with other local datasets

- Overlap with **Cholec80**: **15** videos
- Overlap with **local CholecT50 copy**: **14** videos
- All 20 CholecTrack20 videos fall within the combined Cholec80 + CholecT50 video pool (15 from Cholec80, 5 from the CholecT50-only additions VID92/96/103/110/111)
- The 6 CholecTrack20 videos absent from the local CholecT50 (VID07, VID11, VID17, VID30, VID37, VID39) are all present in Cholec80

#### Training-set instrument distribution

| ID | Instrument | Detections |
|---|---|---:|
| 0 | grasper | 15,811 |
| 2 | hook | 10,392 |
| 6 | specimen-bag | 1,456 |
| 1 | bipolar | 966 |
| 4 | clipper | 733 |
| 5 | irrigator | 663 |
| 3 | scissors | 627 |

#### Task relevance

- Provides tool-state persistence and spatial-memory supervision
- Supports modeling of re-identification across occlusion and body entry/exit
- Supplies trajectory-level context beyond single-frame detection
- Provides verb/target labels in 8 of the 14 CholecT50-overlap videos (~37% of detections); the remaining 12 videos lack these labels

#### Reference benchmark results (paper-reported, test split only)

The following results are from the CholecTrack20 CVPR 2025 paper, evaluated on the **8-video testing split** only. They are not locally reproduced.

**Detection (best model: YOLOv7):** AP = 56.1% across 3 IoU thresholds

**Multi-perspective tracking (best model: Bot-SORT):**

| Perspective | HOTA | MOTA | IDF1 |
|---|---:|---:|---:|
| Intraoperative | 17.4 | 69.6 | 10.2 |
| Intracorporeal | 27.0 | 70.0 | 18.9 |
| Visibility | 44.7 | 72.0 | 41.4 |

All SOTA methods achieve < 45% HOTA, indicating significant room for improvement, especially for intraoperative and intracorporeal perspectives.

### 3.8 GraSP

#### Official scope cross-check

The official GraSP repository describes a benchmark with:

- long-term phase and step recognition
- short-term instrument segmentation and atomic action detection
- training cases numbered **CASE001-CASE021**
- testing cases numbered **CASE041-CASE053**

#### Local case coverage

The local copy is **not the full official case coverage**. It contains only:

- train-side cases: `CASE001`, `CASE002`, `CASE003`, `CASE004`, `CASE007`, `CASE014`, `CASE015`, `CASE021`
- test-side cases: `CASE041`, `CASE047`, `CASE050`, `CASE051`, `CASE053`

So the local copy covers **8 train cases + 5 test cases = 13 cases total**.

#### Verified local facts

| Property | Value |
|---|---:|
| Local case folders | 13 |
| Extracted JPG frames | 3,495,415 |
| Long-term train keyframes | 73,619 |
| Long-term test keyframes | 42,897 |
| Short-term train keyframes | 2,324 |
| Short-term test keyframes | 1,125 |
| Short-term train instances | 6,170 |
| Short-term test instances | 2,861 |
| Segmentation PNGs | 3,449 |

All eight annotation JSONs are present:

- `grasp_long-term_train.json`
- `grasp_long-term_fold1.json`
- `grasp_long-term_fold2.json`
- `grasp_long-term_test.json`
- `grasp_short-term_train.json`
- `grasp_short-term_fold1.json`
- `grasp_short-term_fold2.json`
- `grasp_short-term_test.json`

#### Short-term class distributions (local 13-case subset)

The following counts are from `grasp_short-term_train.json` in the **local 13-case partial copy**, not the official full GraSP benchmark.

Instrument instance counts:

- Monopolar Curved Scissors: 1,765
- Bipolar Forceps: 1,694
- Large Needle Driver: 896
- Suction Instrument: 816
- Prograsp Forceps: 741
- Laparoscopic Grasper: 194
- Clip Applier: 64

Action counts:

- Still: 3,574
- Hold: 2,188
- Travel: 1,653
- Push: 464
- Suction: 399
- Pull: 231
- Cauterize: 170
- Close: 169
- Open: 89
- Cut: 80
- Grasp: 75
- Release: 50
- Open Something: 18
- Other: 12

#### Local coverage note

The local copy contains 13 cases: 8 train-side cases and 5 test-side cases.

#### Task relevance

- Supports learning of hierarchical temporal abstractions
- Supports pretraining of multi-level procedure models
- Key limitations for the present target task:
  - robotic domain gap
  - partial local coverage
  - weak direct transfer to laparoscopic cholecystectomy navigation

### 3.9 AutoLaparo

#### Local structure

```text
/yuming/data/AutoLaparo/
├── Task 1 Surgical workflow recognition/    # 97 GB
│   ├── videos/                              # 21 MP4 files
│   ├── labels/                              # 21 TXT files
│   └── README.txt
├── Task 2 Laparoscope motion prediction/    # 3.6 GB
│   ├── clips/                               # 300 MP4 clips
│   ├── laparoscope_motion_label.txt         # 1 label file
│   └── README.txt
└── Task 3 Instrument and key anatomy segmentation/  # 571 MB
    ├── imgs/                                # 1,800 JPGs (1920×1080)
    ├── masks/                               # 1,800 PNG masks (1920×1080)
    └── README (update_29 Nov 2022).txt
```

Total local footprint: ~101 GB. Surgery type: laparoscopic hysterectomy. Resolution: 1920×1080, 25 fps. License: CC-BY-NC-SA 4.0.

#### Verified local facts — Task 1: Surgical Workflow Recognition

| Video | Frames | Duration | Split | Phases present |
|-------|--------|----------|-------|----------------|
| 01 | 6,388 | 106.5 min | train | 2,3,4,6,7 |
| 02 | 3,620 | 60.3 min | train | 2,3,4,5,6,7 |
| 03 | 3,000 | 50.0 min | train | 1-7 (all) |
| 04 | 2,938 | 49.0 min | train | 1-7 (all) |
| 05 | 3,220 | 53.7 min | train | 1,2,3,4,5 |
| 06 | 3,908 | 65.1 min | train | 1,2,3,4,5 |
| 07 | 1,645 | 27.4 min | train | 1,2,3,4,5 |
| 08 | 4,692 | 78.2 min | train | 1,2,3,4,6,7 |
| 09 | 5,736 | 95.6 min | train | 1,2,3,4,6,7 |
| 10 | 5,064 | 84.4 min | train | 2,3,4,5,7 |
| 11 | 4,720 | 78.7 min | val | 1,2,3,4 |
| 12 | 2,916 | 48.6 min | val | 2,3,4,6,7 |
| 13 | 2,597 | 43.3 min | val | 1,2,3,4,5 |
| 14 | 4,739 | 79.0 min | val | 1-7 (all) |
| 15 | 3,653 | 60.9 min | test | 1,2,3,4,6,7 |
| 16 | 3,612 | 60.2 min | test | 1-7 (all) |
| 17 | 4,678 | 78.0 min | test | 1-7 (all) |
| 18 | 3,546 | 59.1 min | test | 1,2,3,4,6,7 |
| 19 | 3,413 | 56.9 min | test | 1,2,3,4,6,7 |
| 20 | 4,832 | 80.5 min | test | 1-7 (all) |
| 21 | 4,326 | 72.1 min | test | 1-7 (all) |

Split totals: train 40,211 frames (670 min) / val 14,972 frames (250 min) / test 28,060 frames (468 min). Grand total: 83,243 frames = 1,387.4 minutes (paper claims 1,388 min — consistent).

Label format: tab-separated `Frame  Phase`, 1-indexed frame ID at 1 fps. Each file has a header row `Frame  Phase`.

Not all videos contain all 7 phases. Video 01 lacks P1; Videos 05/06/07/13 lack P6/P7; Video 10 lacks P1/P6; Video 11 has only P1-P4. This reflects natural surgical variation, not data errors.

#### Phase distribution (83,243 frames total)

| Phase | Name | Frames | % |
|-------|------|--------|---|
| 1 | Preparation | 3,458 | 4.2% |
| 2 | Dividing Ligament and Peritoneum | 26,337 | 31.6% |
| 3 | Dividing Uterine Vessels and Ligament | 18,248 | 21.9% |
| 4 | Transecting the Vagina | 10,703 | 12.9% |
| 5 | Specimen Removal | 1,596 | 1.9% |
| 6 | Suturing | 14,877 | 17.9% |
| 7 | Washing | 8,024 | 9.6% |

#### Verified local facts — Task 2: Laparoscope Motion Prediction

300 MP4 clips (min 11.2 MB / max 12.9 MB / mean 12.0 MB), each 10 seconds at 25 fps.

Motion label distribution (matches paper exactly):

| Label | Motion | Count |
|-------|--------|-------|
| 0 | Static | 78 |
| 1 | Up | 22 |
| 2 | Down | 45 |
| 3 | Left | 37 |
| 4 | Right | 20 |
| 5 | Zoom-in | 54 |
| 6 | Zoom-out | 44 |

Phase distribution of clips (all from P2-P4): P2=181, P3=71, P4=48.

Label format: tab-separated `Clip  Label  Phase`, 1 header row + 300 data rows.

Split: train 170 (001-170) / val 57 (171-227) / test 73 (228-300).

#### Verified local facts — Task 3: Instrument and Key Anatomy Segmentation

1,800 JPG images + 1,800 PNG masks, all at 1920×1080 RGB (original resolution). Naming: `XXXYYY.jpg` where XXX=clip ID (001-300), YYY=frame timestamp (001/025/050/075/100/125). Each clip has exactly 6 frames; 6×300 = 1,800. Image-mask correspondence is complete: every image has a matching mask, no extras.

Mask RGB encoding (all 10 values verified present):

| RGB | Class |
|-----|-------|
| (0,0,0) | background |
| (20,20,20) | tool1m (Grasping forceps manipulator) |
| (40,40,40) | tool1s (Grasping forceps shaft) |
| (60,60,60) | tool2m (LigaSure manipulator) |
| (80,80,80) | tool2s (LigaSure shaft) |
| (100,100,100) | tool3m (Dissecting forceps manipulator) |
| (120,120,120) | tool3s (Dissecting forceps shaft) |
| (140,140,140) | tool4m (Electric hook manipulator) |
| (160,160,160) | tool4s (Electric hook shaft) |
| (180,180,180) | uterus |

Class presence across masks (number of masks containing each class out of 1,800):

| Class | Masks containing | % of 1,800 |
|-------|-----------------|------------|
| tool2m (LigaSure manipulator) | 1,479 | 82.2% |
| uterus | 1,057 | 58.7% |
| tool2s (LigaSure shaft) | 1,053 | 58.5% |
| tool1m (Grasping forceps manipulator) | 586 | 32.6% |
| tool3m (Dissecting forceps manipulator) | 530 | 29.4% |
| tool1s (Grasping forceps shaft) | 449 | 24.9% |
| tool3s (Dissecting forceps shaft) | 375 | 20.8% |
| tool4s (Electric hook shaft) | 239 | 13.3% |
| tool4m (Electric hook manipulator) | 229 | 12.7% |
| **Total class presences** | **5,997** | — |

#### Annotation count note

The paper and official website report **5,936** annotation instances (split: 3,501 train / 1,127 val / 1,258 test), counted over 5 coarse instrument/anatomy categories. The local per-class-per-mask presence count is **5,997**, counted over 9 fine-grained semantic part/anatomy classes (4 instruments × shaft + manipulator, plus uterus). These two numbers use different category granularities and counting methods, so they are not directly comparable. The exact mapping between the official coarse-category count and the local fine-grained presence count would require reconstructing the official counting pipeline, which is not available.

Split: train clips 001-170 (1,020 images) / val 171-227 (342 images) / test 228-300 (438 images).

#### Loader caveats

- **Task 1 splits**: train videos 01-10 / val 11-14 / test 15-21
- **Task 2 & 3 splits**: train clips 001-170 / val 171-227 / test 228-300
- **Task 1 frame extraction** (`t1_video2frame.py:54`): samples every 25th frame (1 fps from 25 fps video), resizes and center-crops to 250×250 with black-border removal. The local videos remain at original resolution; the 250×250 images are preprocessing outputs rather than distributed source data.
- **Task 2 frame extraction** (`t2_datapre.py:23`): from each distributed 10-second clip (250 frames at 25 fps), samples only the first 125 frames at indices [1,9,17,25,33,41,50,58,66,75,83,91,100,108,116,125], yielding 16 frames covering the first 5 seconds while leaving the second half unused. Output is 250×250.
- **Task 3 images** are at original 1920×1080 resolution, unlike the 250×250 preprocessing outputs of Tasks 1 and 2.
- **Not all videos contain all 7 phases** (see per-video table above). Loader logic therefore needs phase-optional handling.
- **License**: CC-BY-NC-SA 4.0 (see README files in each task folder).

#### Data integrity summary

| Check | Result |
|-------|--------|
| Task 1: 21 videos present | ✓ |
| Task 1: 21 labels present | ✓ |
| Task 1: total frames 83,243 ≈ paper's 1,388 min | ✓ (1,387.4 min) |
| Task 2: 300 clips present | ✓ |
| Task 2: motion label distribution matches paper | ✓ exact match |
| Task 3: 1,800 images + 1,800 masks | ✓ |
| Task 3: image-mask one-to-one correspondence | ✓ |
| Task 3: each clip has exactly 6 frames | ✓ |
| Task 3: mask RGB values match documentation | ✓ |

#### Task relevance

- Provides the only local source of **explicit camera-motion labels** (7-class laparoscope motion prediction)
- Supports camera-motion pretraining and auxiliary supervision
- Provides instrument + anatomy segmentation with shaft/manipulator distinction
- Introduces a procedure-domain gap because it covers laparoscopic hysterectomy rather than cholecystectomy; this affects direct label pooling

### 3.10 HeiChole

#### Official scope cross-check

The HeiChole dataset was released as part of the **EndoVis Sub-challenge Surgical Workflow and Skill Analysis** at MICCAI 2019, organized by NCT Dresden and University Hospital Heidelberg. The companion paper (Wagner et al., Medical Image Analysis 2023) describes four concurrent recognition tasks on laparoscopic cholecystectomy videos: phase segmentation, instrument presence detection, action recognition, and surgical skill assessment. The dataset was collected from **three hospitals**: 15 videos from University Hospital Heidelberg, 15 from Salem Hospital, and 3 from Sinsheim Hospital (33 total).

The official challenge split is **24 training + 9 test** videos. Test labels were not publicly released. The local copy contains the **24 training videos** only. The 24 videos come from all three hospital sites.

Challenge platform: Synapse.org. License: CC BY-NC (post-challenge publication).

#### Local structure

```text
/yuming/data/HeiChole/
├── Videos/
│   ├── Full/                              # 24 SD MP4s + HD/ subfolder (24 HD MP4s)
│   └── Skill/                             # 48 phase-clipped MP4s (calot + dissection)
├── Annotations/
│   ├── Phase/                             # 24 CSVs + Raw_annotations/
│   ├── Instrument/                        # 24 category CSVs + 24 detailed CSVs
│   ├── Action/                            # 24 basic CSVs + 24 detailed CSVs
│   └── Skill/                             # 72 CSVs (full + calot + dissection per video)
├── Evaluation_Scripts/                    # 4 Python eval scripts
├── EndoVisWorkflow_ReadMe.pdf
├── ChallengeDesign_WorkflowSkill.pdf
├── Presentation_EndoVis_SurgicalWorkflowandSkill2019.pdf
└── SYNAPSE_METADATA_MANIFEST.tsv
```

#### Verified local facts

| Property | Value |
|---|---:|
| Full videos (SD) | 24 |
| Full videos (HD) | 24 |
| Skill phase videos (calot + dissection) | 48 |
| Total annotated frames | 1,836,369 |
| Mean frames/video | 76,515 |
| Median frames/video | 61,076 |
| Min frames/video | 31,816 (Hei-Chole4) |
| Max frames/video | 255,120 (Hei-Chole17) |
| SD video total size | 61.85 GB |
| HD video total size | 107.82 GB |
| Skill video total size | 42.17 GB |
| Total dataset size | ~212 GB |

All 24 videos have consistent frame counts across Phase, Instrument, and Action annotations (verified per-video).

#### Per-video frame counts

| Video | Frames | Phase transitions | Phases present |
|---|---:|---:|---|
| Hei-Chole1 | 54,930 | 7 | 0,1,2,3,4,5,6 |
| Hei-Chole2 | 50,913 | 6 | 0,1,2,3,4,5,6 |
| Hei-Chole3 | 71,976 | 8 | 0,1,2,3,4,5,6 |
| Hei-Chole4 | 31,816 | 6 | 0,1,2,3,4,5,6 |
| Hei-Chole5 | 78,234 | 12 | 0,1,2,3,4,5,6 |
| Hei-Chole6 | 72,211 | 6 | 0,1,2,3,4,5,6 |
| Hei-Chole7 | 104,670 | 10 | 0,1,2,3,4,5,6 |
| Hei-Chole8 | 42,791 | 10 | 0,1,2,3,4,5,6 |
| Hei-Chole9 | 67,222 | 8 | 0,1,2,3,4,5,6 |
| Hei-Chole10 | 44,746 | 7 | 0,1,2,3,4,6 |
| Hei-Chole11 | 35,312 | 9 | 0,1,2,3,4,5,6 |
| Hei-Chole12 | 44,774 | 8 | 0,1,2,3,4,5,6 |
| Hei-Chole13 | 49,447 | 6 | 0,1,2,3,4,5,6 |
| Hei-Chole14 | 53,257 | 10 | 0,1,2,3,4,5,6 |
| Hei-Chole15 | 49,489 | 7 | 0,1,2,3,4,6 |
| Hei-Chole16 | 184,700 | 10 | 0,1,2,3,4,5,6 |
| Hei-Chole17 | 255,120 | 11 | 0,1,2,3,4,5,6 |
| Hei-Chole18 | 77,620 | 6 | 0,1,2,3,4,5,6 |
| Hei-Chole19 | 94,544 | 8 | 0,1,2,3,4,5,6 |
| Hei-Chole20 | 83,048 | 6 | 0,1,2,3,4,5,6 |
| Hei-Chole21 | 32,759 | 5 | 0,1,2,3,4,6 |
| Hei-Chole22 | 35,542 | 5 | 0,1,2,3,4,6 |
| Hei-Chole23 | 91,534 | 6 | 0,1,2,3,4,5,6 |
| Hei-Chole24 | 129,714 | 7 | 0,1,2,3,4,5,6 |

Verified video resolution and frame rate (from MP4 moov atom metadata):

| Videos | Resolution | FPS | Notes |
|---|---|---:|---|
| 1–12 | 960×540 (SD) / 1920×1080 (HD) | 25 | True SD/HD pair; HD files ~5× larger |
| 13, 14, 21 | 720×576 (PAL) | 25 | SD and HD files identical (no upscaled version exists) |
| 15, 22 | 1920×1080 | 25 | Natively full-HD; SD and HD files identical |
| 16–20, 23–24 | 1920×1080 | 50 | Natively full-HD at 50 fps; SD and HD files identical |

Summary: **17 videos @25 fps, 7 videos @50 fps**. Annotations are per-frame at native fps.

#### Video file size comparison (SD vs HD)

| Video | SD (MB) | HD (MB) |
|---|---:|---:|
| Hei-Chole1 | 807.8 | 4,303.0 |
| Hei-Chole2 | 751.8 | 4,712.2 |
| Hei-Chole3 | 947.1 | 6,386.2 |
| Hei-Chole4 | 404.7 | 2,384.1 |
| Hei-Chole5 | 969.9 | 4,912.1 |
| Hei-Chole6 | 919.4 | 6,116.0 |
| Hei-Chole7 | 1,493.2 | 10,497.1 |
| Hei-Chole8 | 517.2 | 3,023.8 |
| Hei-Chole9 | 769.4 | 5,228.9 |
| Hei-Chole10 | 625.3 | 3,348.8 |
| Hei-Chole11 | 474.5 | 2,429.4 |
| Hei-Chole12 | 596.3 | 3,005.3 |
| Hei-Chole13 | 817.4 | 817.4 |
| Hei-Chole14 | 747.6 | 747.6 |
| Hei-Chole15 | 5,507.8 | 5,507.8 |
| Hei-Chole16 | 8,066.9 | 8,066.9 |
| Hei-Chole17 | 12,144.1 | 12,144.1 |
| Hei-Chole18 | 3,566.4 | 3,566.4 |
| Hei-Chole19 | 5,177.3 | 5,177.3 |
| Hei-Chole20 | 2,921.7 | 2,921.7 |
| Hei-Chole21 | 598.2 | 598.2 |
| Hei-Chole22 | 2,925.7 | 2,925.7 |
| Hei-Chole23 | 5,952.5 | 5,952.5 |
| Hei-Chole24 | 5,636.7 | 5,636.7 |

For videos 1–12 (960×540 SD / 1920×1080 HD), HD files are 4–7× larger than SD. For videos 13, 14, 21 (720×576 PAL) and 15–20, 22–24 (1920×1080 native), SD and HD files are identical in size — no separate downsampled SD variant exists for these videos.

#### Phase annotations (7 phases)

Format: `<frame #>,<phase id>` per line.

| ID | Phase | Frames | % | #Videos |
|---|---|---:|---:|---:|
| 0 | Preparation | 126,862 | 6.91% | 24/24 |
| 1 | Calot triangle dissection | 852,118 | 46.40% | 24/24 |
| 2 | Clipping and cutting | 158,519 | 8.63% | 24/24 |
| 3 | Gallbladder dissection | 327,289 | 17.82% | 24/24 |
| 4 | Gallbladder packaging | 73,798 | 4.02% | 24/24 |
| 5 | Cleaning and coagulation | 235,407 | 12.82% | 20/24 |
| 6 | Gallbladder retraction | 62,376 | 3.40% | 24/24 |

Phase 5 (Cleaning and coagulation) is absent from 4 videos: Hei-Chole10, 15, 21, 22. The dominant phase is Calot triangle dissection (46.4% of all frames).

Phase duration variability:

| Phase | Min | Max | Mean | Std | CV% |
|---|---:|---:|---:|---:|---:|
| Preparation | 2,336 | 12,740 | 5,286 | 2,325 | 44.0% |
| Calot triangle dissection | 7,065 | 145,476 | 35,505 | 31,394 | 88.4% |
| Clipping and cutting | 1,999 | 14,090 | 6,605 | 3,525 | 53.4% |
| Gallbladder dissection | 2,850 | 56,080 | 13,637 | 10,830 | 79.4% |
| Gallbladder packaging | 1,264 | 6,554 | 3,075 | 1,443 | 46.9% |
| Cleaning and coagulation | 1,570 | 27,898 | 11,770 | 7,172 | 60.9% |
| Gallbladder retraction | 28 | 9,444 | 2,599 | 2,423 | 93.2% |

The Calot triangle dissection phase has extremely high variability (CV=88.4%, range 7K–145K frames), reflecting wide differences in case complexity.

#### Phase ontology compatibility with Cholec80

The README states that HeiChole phases are **compatible with the Cholec80 dataset** (Twinanda et al., TMI 2017). Direct mapping:

| HeiChole ID | HeiChole Phase | Cholec80 Phase |
|---|---|---|
| 0 | Preparation | Preparation |
| 1 | Calot triangle dissection | CalotTriangleDissection |
| 2 | Clipping and cutting | ClippingCutting |
| 3 | Galbladder dissection | GallbladderDissection |
| 4 | Galbladder packaging | GallbladderPackaging |
| 5 | Cleaning and coagulation | CleaningCoagulation |
| 6 | Galbladder retraction | GallbladderRetraction |

This is a direct 1-to-1 semantic mapping (same 7 phases, same integer IDs). Note: HeiChole spells "Galbladder" (sic) in the README; Cholec80 uses "Gallbladder".

#### Instrument annotations (7 categories, 21 fine-grained types)

Two files per video: `_Annotation_Instrument.csv` (7-category binary) and `_Annotation_Instrument_Detailed.csv` (21-type binary + undefined shaft).

Format: `<frame #>,<cat 0 visible?>,...,<cat 20 visible?>` (22 columns for category file; 32 columns for detailed file).

**Category-level presence** (1,836,369 frames):

| ID | Category | Frames | % |
|---|---|---:|---:|
| 0 | Grasper | 1,196,122 | 65.14% |
| 1 | Clipper | 66,890 | 3.64% |
| 2 | Coagulation | 899,765 | 49.00% |
| 3 | Scissors | 46,187 | 2.52% |
| 4 | Suction-irrigation | 132,768 | 7.23% |
| 5 | Specimen bag | 180,240 | 9.82% |
| 6 | Stapler | 2,539 | 0.14% |
| 20 | Undefined instrument shaft | 26,861 | 1.46% |

Category columns 7–19 in the CSV are truly zero (reserved). Category ID 20 (the last instrument column, index 21 in the CSV) corresponds to "Undefined instrument shaft" — matching tool ID 30 in the detailed files.

**Fine-grained instrument type presence** (1,836,369 frames):

| ID | Instrument | Cat | Frames | % |
|---|---|---:|---:|---:|
| 0 | Curved atraumatic grasper | 0 | 191,777 | 10.44% |
| 1 | Toothed grasper | 0 | 60,474 | 3.29% |
| 2 | Fenestrated toothed grasper | 0 | 87,423 | 4.76% |
| 3 | Atraumatic grasper | 0 | 924,709 | 50.36% |
| 4 | Overholt | 0 | 28,737 | 1.56% |
| 5 | LigaSure | 2 | 19,956 | 1.09% |
| 6 | Electric hook | 2 | 866,223 | 47.17% |
| 7 | Scissors | 3 | 46,187 | 2.52% |
| 8 | Clip-applier (metal) | 1 | 58,818 | 3.20% |
| 9 | Clip-applier (Hem-O-Lok) | 1 | 8,072 | 0.44% |
| 10 | Swab grasper | 0 | 22,379 | 1.22% |
| 11 | Argon beamer | 2 | 13,586 | 0.74% |
| 12 | Suction-irrigation | 4 | 132,768 | 7.23% |
| 13 | Specimen bag | 5 | 180,240 | 9.82% |
| 14 | Tiger mouth forceps | 0 | 13,587 | 0.74% |
| 15 | Claw forceps | 0 | 4,506 | 0.25% |
| 16 | Atraumatic grasper short | 0 | 46,240 | 2.52% |
| 17 | Crocodile forceps | 0 | 29,494 | 1.61% |
| 18 | Flat grasper | 0 | 5,496 | 0.30% |
| 19 | Pointed forceps | 0 | 5,464 | 0.30% |
| 20 | Stapler | 6 | 2,539 | 0.14% |
| 30 | Undefined instrument shaft | 20 | 26,861 | 1.46% |

The Grasper category alone has **10 subtypes** — a much finer granularity than any CAMMA dataset. The dominant instruments are Atraumatic grasper (50.4%) and Electric hook (47.2%).

**Simultaneous instrument count** (category level):

| # Instruments | Frames | % |
|---:|---:|---:|
| 0 | 283,592 | 15.44% |
| 1 | 638,802 | 34.79% |
| 2 | 864,427 | 47.07% |
| 3 | 41,337 | 2.25% |
| 4 | 8,211 | 0.45% |

Most frames (47.1%) have exactly 2 instruments visible simultaneously.

#### Action annotations (4 types, multi-label)

Two files per video: `_Annotation_Action.csv` (4-action binary) and `_Annotation_Action_Detailed.csv` (per-hand: left/right/assistant × 4 actions = 12 columns).

Format: `<frame #>,<grasp>,<hold>,<cut>,<clip>`.

| ID | Action | Frames | % |
|---|---|---:|---:|
| 0 | Grasp | 22,315 | 1.22% |
| 1 | Hold | 1,440,974 | 78.47% |
| 2 | Cut | 5,448 | 0.30% |
| 3 | Clip | 6,883 | 0.37% |

- No action (idle): 388,483 frames (21.15%)
- Multi-action frames: 27,733 frames (1.51%)

The action label space is extremely unbalanced: Hold dominates at 78.5%, while Cut and Clip together account for less than 0.7% of frames. This coarse 4-class action vocabulary is much less expressive than CholecT50's 10-verb × 15-target triplet space.

#### Phase–instrument co-occurrence

Percentage of frames within each phase where each instrument category is present:

| Phase | Grasper | Clipper | Coagul | Scissors | Suction | SpecBag | Stapler |
|---|---:|---:|---:|---:|---:|---:|---:|
| Preparation | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Calot | 78.6% | 0.4% | 70.8% | 0.1% | 2.3% | 0.0% | 0.0% |
| Clip & Cut | 72.6% | 36.4% | 1.8% | 17.8% | 1.4% | 0.0% | 1.6% |
| GB dissection | 52.9% | 0.3% | 77.8% | 3.4% | 7.7% | 0.2% | 0.0% |
| GB packaging | 91.3% | 0.0% | 0.0% | 0.0% | 5.6% | 93.7% | 0.0% |
| Clean & Coag | 68.8% | 2.1% | 16.5% | 2.4% | 34.5% | 36.3% | 0.0% |
| GB retraction | 14.0% | 0.0% | 0.0% | 0.0% | 0.0% | 40.2% | 0.0% |

Phase–instrument patterns are clinically coherent: Calot and GB dissection use grasper + coagulation (electric hook); Clip & Cut uses clipper + scissors; GB packaging uses specimen bag + grasper; Cleaning uses suction-irrigation.

#### Skill annotations (5 dimensions, 1–5 ordinal scale)

Each video has 3 skill CSVs: full video, Calot phase, and Dissection phase. Each contains 5 comma-separated values: `depth_perception, bimanual_dexterity, efficiency, tissue_handling, case_difficulty`.

**Full-video skill scores:**

| Video | Depth | Bimanual | Efficiency | Tissue | Difficulty | Mean |
|---|---:|---:|---:|---:|---:|---:|
| Hei-Chole1 | 5 | 5 | 4 | 3 | 4 | 4.2 |
| Hei-Chole2 | 4 | 4 | 5 | 3 | 3 | 3.8 |
| Hei-Chole3 | 4 | 3 | 3 | 3 | 3 | 3.2 |
| Hei-Chole4 | 5 | 5 | 5 | 3 | 1 | 3.8 |
| Hei-Chole5 | 4 | 4 | 4 | 3 | 5 | 4.0 |
| Hei-Chole6 | 5 | 5 | 4 | 3 | 3 | 4.0 |
| Hei-Chole7 | 4 | 4 | 3 | 4 | 3 | 3.6 |
| Hei-Chole8 | 4 | 5 | 5 | 5 | 2 | 4.2 |
| Hei-Chole9 | 4 | 4 | 4 | 4 | 2 | 3.6 |
| Hei-Chole10 | 5 | 4 | 5 | 4 | 2 | 4.0 |
| Hei-Chole11 | 5 | 5 | 5 | 4 | 1 | 4.0 |
| Hei-Chole12 | 5 | 4 | 3 | 3 | 1 | 3.2 |
| Hei-Chole13 | 5 | 4 | 5 | 5 | 1 | 4.0 |
| Hei-Chole14 | 5 | 4 | 4 | 4 | 1 | 3.6 |
| Hei-Chole15 | 5 | 4 | 4 | 5 | 1 | 3.8 |
| Hei-Chole16 | 5 | 4 | 4 | 3 | 3 | 3.8 |
| Hei-Chole17 | 4 | 5 | 3 | 3 | 3 | 3.6 |
| Hei-Chole18 | 3 | 4 | 3 | 4 | 1 | 3.0 |
| Hei-Chole19 | 4 | 5 | 5 | 4 | 1 | 3.8 |
| Hei-Chole20 | 5 | 4 | 5 | 5 | 1 | 4.0 |
| Hei-Chole21 | 5 | 5 | 5 | 4 | 2 | 4.2 |
| Hei-Chole22 | 5 | 4 | 5 | 4 | 2 | 4.0 |
| Hei-Chole23 | 4 | 5 | 4 | 4 | 2 | 3.8 |
| Hei-Chole24 | 4 | 4 | 4 | 3 | 4 | 3.8 |

**Aggregate statistics:**

| Dimension | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| Depth perception | 4.50 | 0.58 | 3 | 5 |
| Bimanual dexterity | 4.33 | 0.55 | 3 | 5 |
| Efficiency | 4.21 | 0.76 | 3 | 5 |
| Tissue handling | 3.75 | 0.72 | 3 | 5 |
| Case difficulty | 2.17 | 1.14 | 1 | 5 |

Case difficulty is heavily skewed low (9/24 rated 1, only 1/24 rated 5). Skill scores for Calot phase are slightly lower than Dissection phase across Depth, Bimanual, and Efficiency (by 0.1–0.3 points), while Tissue handling is comparable.

**Score distributions (full video):**

- Depth perception: score 3=1, 4=10, 5=13
- Bimanual dexterity: score 3=1, 4=14, 5=9
- Efficiency: score 3=5, 4=9, 5=10
- Tissue handling: score 3=10, 4=10, 5=4
- Case difficulty: score 1=9, 2=6, 3=6, 4=2, 5=1

#### Evaluation scripts

Four evaluation scripts are provided:

| Script | Task | Metric | Ranking metric |
|---|---|---|---|
| `EvalPhase.py` | Phase segmentation | per-class F1, accuracy | macro F1 |
| `EvalInstrument.py` | Instrument presence | per-class F1, accuracy | macro F1 |
| `EvalAction.py` | Action recognition | per-class F1, accuracy | macro F1 |
| `EvalSkill.py` | Skill assessment | per-dimension MAE | overall MAE |

All recognition scripts use `sklearn.metrics.f1_score` with `average=None` and compute macro F1 over present classes. The skill script computes absolute error between predicted and ground-truth ordinal scores.

#### Raw annotations

The `Annotations/Phase/Raw_annotations/` directory contains **67 files** (CSV + PNG pairs per video), providing the original annotator markings before consensus.

#### Loader caveats

- **Official train/test split**: the challenge defined 24 training + 9 test videos. Only the 24 training videos and their labels are publicly available; test labels were withheld. The local copy contains the 24 training videos. Any evaluation requires defining a custom split within these 24 videos.
- **Mixed frame rates**: 17 videos are recorded at 25 fps and 7 videos (16–20, 23–24) at 50 fps. Annotations are per-frame at native fps — no 1 fps extraction. Frame counts per video must be divided by the correct fps to compute durations.
- **Extra-abdominal frames**: censored as all-white RGB(255,255,255) frames. These frames retain annotations (typically Phase 0 or Phase 6) and should be handled during training.
- **Instrument columns**: the category-level CSV has 22 columns (frame + 21 slots for categories 0–20). Columns 7–19 are reserved (all zeros). Column 20 (category ID 20) contains 26,861 non-zero frames for "Undefined instrument shaft". The detailed CSV has 32 columns (frame + 31 slots for types 0–30).
- **Video length variability**: an 8× range (31K–255K frames) means fixed-length sequence sampling will waste most frames of long videos or require many windows.

#### Data integrity summary

| Check | Result |
|---|---|
| Phase/Instrument/Action frame counts match per video | All 24 consistent |
| All 24 phase CSVs present | ✓ |
| All 24 instrument CSVs + 24 detailed CSVs present | ✓ |
| All 24 action CSVs + 24 detailed CSVs present | ✓ |
| All 72 skill CSVs present (3 per video) | ✓ |
| All 24 SD + 24 HD + 48 skill videos present | ✓ |

#### Task relevance

- Provides **Cholec80-compatible phase labels** on an independent Heidelberg cohort (24 videos from 3 hospitals), enabling cross-center phase recognition evaluation
- Adds **fine-grained instrument subtype labels** (21 types vs Cholec80's 7 binary tool presence columns) — the most detailed instrument taxonomy in the local collection
- Adds **frame-level action labels** (4 action types, multi-label, per-hand in detailed version) — bridges between Cholec80's tool-only supervision and CholecT50's triplet supervision
- Provides **surgical skill annotations** (video-level ordinal, 5 dimensions) — unique in the local collection; could support difficulty-aware training or stratified evaluation
- As a non-CAMMA dataset from a different hospital network, HeiChole is **fully independent** of all CAMMA datasets — making it suitable as an additional external test set for phase recognition and instrument detection without any video-ID overlap risk
- Key limitations: no spatial annotations (no bounding boxes or segmentation), no verb-target triplets, coarse 4-class action vocabulary, only 24 of 33 challenge videos publicly available (test labels withheld), mixed 25/50 fps requiring per-video handling, and extremely large video files

## 4. Cross-Dataset Integration Risks

### 4.1 Video overlap and leakage

Exact overlaps in the local environment (Endoscapes rows use CAMMA official video-ID mapping, not raw numeric comparison):

| Pair | Overlap |
|---|---:|
| Cholec80 vs Cholec80-CVS | 80 |
| Cholec80 vs CholecT50 | 45 |
| Cholec80 vs CholecSeg8k | 17 |
| CholecT50 vs CholecSeg8k | 10 |
| Cholec80 vs CholecInstanceSeg | 80 |
| Cholec80-CVS vs CholecT50 | 45 |
| Cholec80-CVS vs CholecSeg8k | 17 |
| Cholec80-CVS vs CholecInstanceSeg | 80 |
| CholecT50 vs CholecInstanceSeg | 50 |
| Cholec80 vs Endoscapes2023 | 6 (CAMMA ID mapping) |
| Cholec80-CVS vs Endoscapes2023 | 6 (CAMMA ID mapping) |
| CholecT50 vs Endoscapes2023 | 6 (CAMMA ID mapping) |
| Cholec80 vs CholecTrack20 | 15 |
| Cholec80-CVS vs CholecTrack20 | 15 |
| CholecT50 vs CholecTrack20 | 14 |
| HeiChole vs any CAMMA dataset | **0** |
| HeiChole vs AutoLaparo | 0 |
| HeiChole vs GraSP | 0 |

HeiChole is from Heidelberg-affiliated hospitals, entirely independent of the Strasbourg-based CAMMA datasets and all other local datasets. It is **overlap-free** and can be used as an external validation set without leakage risk.

These overlaps place leakage control at the **original video ID level** for pretraining, fine-tuning, and evaluation.

### 4.2 Ontology mismatch

- Cholec80 phases and CholecT50 phases are close but not identical
- Cholec80-CVS uses per-criterion ordinal scores `{0, 1, 2}` plus a raw `critical_view` flag defined by `Total >= 5`; the official loader instead binarizes each criterion and ignores the raw overall flag
- Cholec80 tools include `SpecimenBag`; CholecT50 instrument labels do not
- CholecInstanceSeg includes `snare`, which is absent from Cholec80 and CholecT50
- Endoscapes uses anatomy-plus-tool categories rather than triplets or dense workflow states
- GraSP labels phases, steps, instruments, and atomic actions, but in robotic prostatectomy
- HeiChole phases are Cholec80-compatible (same 7 IDs, same semantics). HeiChole instrument categories (7 types: Grasper, Clipper, Coagulation, Scissors, Suction-irrigation, Specimen bag, Stapler) overlap substantially with Cholec80's 7 binary tool columns but differ in naming and granularity: Cholec80 separates Hook and Bipolar under coagulation, whereas HeiChole merges them into "Coagulation instruments" at the category level but preserves LigaSure, Electric hook, and Argon beamer at the fine-grained level. HeiChole's 4 action types (Grasp, Hold, Cut, Clip) are a strict subset of CholecT50's 10 verbs.

A single unified label space requires an explicit ontology table.

### 4.3 Temporal mismatch

- Cholec80 extracted frames: 1 fps
- Cholec80 dense phase labels: original frame rate
- Cholec80-CVS raw annotations: time intervals in `minute:second`, later expanded to 25 fps frame labels, truncated to the final 15% of the pre-clip/cut window, then sampled to 5 fps by the official pipeline
- CholecT50 labels: 1 fps
- Endoscapes ROI frames: 1 fps within a restricted temporal window
- CholecTrack20 JSON keys: original 25 fps frame numbers sampled roughly every second
- GraSP long-term labels: 1-second intervals
- GraSP short-term labels: 35-second intervals
- HeiChole annotations: native video frame rate (25 fps for 17 videos, 50 fps for 7 videos: 16–20, 23–24); no 1 fps extraction — annotations cover every frame

Sampling alignment is a primary preprocessing requirement for this dataset collection.

### 4.4 Domain mismatch

- Cholec80, Cholec80-CVS, CholecT50, CholecSeg8k, CholecInstanceSeg, Endoscapes2023, CholecTrack20: laparoscopic cholecystectomy (Strasbourg / CAMMA)
- HeiChole: laparoscopic cholecystectomy (Heidelberg, 3 hospitals)
- AutoLaparo: laparoscopic hysterectomy
- GraSP: robot-assisted radical prostatectomy

HeiChole is the same procedure type (cholecystectomy) as the CAMMA datasets but from a different center network. This creates a natural domain-shift evaluation scenario: same procedure, different hospitals, different surgeons, different equipment. GraSP and AutoLaparo differ from the cholecystectomy datasets in procedure and platform, so their labels align more naturally with representation learning or auxiliary pretraining than with direct label pooling.

### 4.5 CAMMA split-combination reference

The [CAMMA overlap analysis repository](https://github.com/CAMMA-public/VideoID-Overlap-Analysis-of-Cholecystectomy-Datasets) documents one split-combination strategy for Cholec80 + CholecT50 + Endoscapes that preserves test-set integrity. In this configuration, Endoscapes and CholecT50 remain unchanged while Cholec80 is adjusted as follows:

**Cholec80 adjustments:**

- **Training:** remove 4 videos overlapping with CholecT50-test (6, 10, 14, 32) → 36 remain
- **Validation:** remove 1 video overlapping with CholecT50-test (42) → 7 remain
- **Test:** remove 17 videos (2 overlapping with CholecT50-val: 50, 78; 15 overlapping with CholecT50-train or Endoscapes-train: 49, 52, 56, 57, 60, 62, 65, 66, 67, 68, 70, 71, 72, 75, 79) → 15 remain

**Combined totals:** 191 train / 53 val / 65 test

This strategy was used in Walimbe et al. (MICCAI 2025, arXiv:2507.05020).

### 4.6 M2CAI overlap profile

The M2CAI challenge dataset (Strasbourg + TUM/Munich centers) is often used for workflow recognition evaluation. The CAMMA overlap analysis reports the following relationship to Cholec80:

- **Cholec80 was released to extend and replace** the M2CAI Strasbourg videos
- The overlap profile leaves the **M2CAI-Munich** (TUM) subset as the non-Strasbourg portion
- M2CAI-tool overlaps with Cholec80-test:
  - M2CAI-tool-train (videos 1–10) = Cholec80 videos 67–76
  - M2CAI-tool-test (videos 11–15) = Cholec80 videos 61, 62, 64, 65, 66
- The Strasbourg subset is fully overlapped in the CAMMA analysis context when Cholec80 is part of the dataset pool

## 5. Dataset-Task Alignment

### 5.1 Action-planning-related supervision coverage

The local dataset collection provides the following action-planning-related supervision sources:

1. **CholecT50** for triplet supervision
2. **Cholec80** for full-procedure phase priors
3. **Cholec80-CVS** for clip/cut readiness and CVS safety gating
4. **CholecTrack20** for tool-state continuity and action context
5. **GraSP** for auxiliary hierarchical pretraining signals
6. **HeiChole** for external validation of phase recognition and instrument detection (overlap-free Heidelberg cohort)

### 5.2 Navigation-related supervision coverage

The local dataset collection provides the following navigation-related supervision sources:

1. **Cholec80-CVS** for CVS safety-state supervision before clip/cut
2. **Endoscapes2023** for anatomy and safety state
3. **CholecTrack20** for tool motion and persistent tool identity
4. **CholecSeg8k** for dense scene parsing
5. **CholecInstanceSeg** for precise tool geometry
6. **AutoLaparo** for explicit camera-motion supervision

### 5.3 Integration components

1. Canonical registry keyed by original `VIDXX` or `CASEXXX` identifiers.
2. Harmonization layer for:
   - phase mapping
   - tool ontology mapping
   - temporal resampling
   - split deduplication
3. Spatial supervision sources:
   - Endoscapes2023
   - CholecSeg8k
   - CholecInstanceSeg
   - CholecTrack20
4. Temporal supervision sources:
   - CholecT50 triplets
   - Cholec80 phases
   - Cholec80-CVS pre-clip safety-state labels
   - CholecTrack20 tool trajectories as auxiliary context
5. Navigation-control-related supervision sources:
   - Cholec80-CVS CVS gate
   - Endoscapes-based anatomy state
   - CholecTrack20-based tool memory
   - AutoLaparo camera-motion pretraining

### 5.4 Remaining local supervision gaps

The current local collection contains supervision for:

- action semantics
- tool-state tracking
- anatomy/safety perception

The current local collection lacks direct supervision for:

- direct path-planning targets
- surgeon intent labels beyond tool-action triplets

The currently available supervision aligns more directly with:

- **phase-aware next-action prediction**
- plus **anatomy-aware navigation state estimation**
- plus **tool-memory modeling**

than with fully supervised autonomous navigation policy learning from supervision alone.

### 5.5 Action-change prediction task formulation

**Task definition**: given a sequence of frames up to time `t`, predict (a) how many seconds until the next action-state change, and (b) what the post-change state will be.

**Granularity trade-off**:

| Level | Changes/min | Pro | Con |
|---|---:|---|---|
| Phase | 0.17 | Clean, universal, well-studied | Too sparse for continuous prediction |
| Triplet-set | 6.17 | Richest signal, captures fine-grained action semantics | Noisy, long-tailed, multi-label |
| Instrument-set | 4.07 | Moderate frequency, visually grounded | Loses verb/target semantics |

**Primary dataset**: CholecT50 (50 videos, 1 fps, 100,863 frames, ~6.2 triplet changes/min).

**Auxiliary supervision sources**:

- Cholec80 phases for coarse temporal context
- CholecTrack20 for spatial tool state (14 overlapping videos)
- Cholec80-CVS for safety-state gating

**Key design considerations**:

1. **Multi-instance state**: 47.4% of frames have >1 action triplet → state is a set, not a single label
2. **Long-tailed distribution**: top 3 triplets cover 55.67% → macro F1 essential
3. **Null triplet handling**: null triplets (94–99) require explicit handling — are tool pickup/putdown "action changes"?
4. **Decontamination**: overlapping video IDs across all datasets is mandatory; see Section 4.1 for the full overlap matrix and Section 4.5 for a reference split-combination strategy

## 6. Sources

### Local evidence

- `/yuming/data/cholec80`
- `/yuming/data/cholec80-cvs`
- `/yuming/data/cholecT50`
- `/yuming/data/cholecSeg8k`
- `/yuming/data/cholecInstanceSeg`
- `/yuming/data/endoscapes-2023`
- `/yuming/data/cholecTrack20`
- `/yuming/data/GraSP`
- `/yuming/data/AutoLaparo`
- `/yuming/data/HeiChole`
- `/yuming/repos/CHOLEC80-CVS-PUBLIC`
- `/yuming/repos/camma_dataset_overlaps` (Cholec80 standard split, CAMMA overlap analysis)
- `/yuming/repos/cholect50/docs/README-Format.md` (annotation vector field documentation)

### Official references

- Cholec80 official repository: <https://github.com/CAMMA-public/TF-Cholec80>
- Cholec80-CVS official repository: <https://github.com/CAMMA-public/CHOLEC80-CVS-PUBLIC>
- Cholec80-CVS paper: <https://www.nature.com/articles/s41597-023-02073-7>
- CholecT50 official repository: <https://github.com/CAMMA-public/cholect50>
- Endoscapes official repository: <https://github.com/CAMMA-public/Endoscapes>
- CAMMA overlap analysis repository: <https://github.com/CAMMA-public/VideoID-Overlap-Analysis-of-Cholecystectomy-Datasets>
- CholecInstanceSeg paper: <https://doi.org/10.1038/s41597-025-05163-w>
- CholecTrack20 official repository: <https://github.com/CAMMA-public/cholectrack20>
- CholecTrack20 paper: <https://arxiv.org/abs/2312.07352>
- GraSP official repository: <https://github.com/BCV-Uniandes/GraSP>
- AutoLaparo official repository: <https://github.com/ziyiwangx/AutoLaparo>
- AutoLaparo official website: <https://autolaparo.github.io>
- AutoLaparo paper: <https://arxiv.org/abs/2208.02049>
- HeiChole paper (EndoVis 2019 challenge): Wagner et al., "Comparative validation of machine learning algorithms for surgical workflow and skill analysis with the HeiChole benchmark", Medical Image Analysis, 2023. <https://doi.org/10.1016/j.media.2023.102770>
- HeiChole challenge platform: <https://www.synapse.org/#!Synapse:syn18824884/wiki/592586>
- M2CAI challenge: <http://camma.u-strasbg.fr/m2cai2016/index.php/program-challenge/>
- Walimbe et al. (MICCAI 2025): <https://arxiv.org/abs/2507.05020>
