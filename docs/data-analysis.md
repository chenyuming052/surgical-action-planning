# Surgical Video Dataset Audit for Action Planning and Navigation

Date: 2026-03-10
Target task: action planning and navigation in image-guided surgery
Audit scope: 9 local datasets under `/yuming/data`

## Executive Summary

This document summarizes on-disk counts and official-source cross-checks for nine local surgical video datasets relevant to action planning and navigation.

1. **Cholec80** is a full-procedure workflow dataset for laparoscopic cholecystectomy, and its dense phase labels are not aligned one-to-one with the extracted 1 fps frame folders.
2. **Cholec80-CVS** adds CVS safety annotations on top of all **80 Cholec80 videos**, but the local copy is only a raw XLSX workbook; the official code links it back to Cholec80, truncates to the last 15% before clip/cut, and binarizes criterion scores.
3. **CholecT50** is a local source of next-action semantics via `<instrument, verb, target>` triplets, and the local copy matches the official **Release 2.0** description: no usable bounding boxes are present.
4. **CholecTrack20** is a full-procedure spatial tool-dynamics dataset with **35,009 annotated frames**, **65,247 detections**, and **2,624 per-video trajectories** across the three tracking perspectives.
5. **Endoscapes2023** contains **58,813** ROI JPGs, and `test/annotation_coco_vid.json` omits **228** frame entries for videos **190** and **194**.
6. **CholecInstanceSeg** and **CholecSeg8k** are spatial-supervision datasets and are not direct action-planning datasets.
7. **GraSP** provides hierarchical procedure representation annotations, and the local copy contains **13** cases from robotic prostatectomy.
8. **AutoLaparo** is present locally at **101 GB** with 21 full-length laparoscopic hysterectomy videos, 300 motion-prediction clips, and 1,800 segmentation frames.

## 1. Scope and Coverage

This audit covers 9 local datasets under `/yuming/data`.

- `AutoLaparo` is part of the local dataset set.
- `Cholec80-CVS` is annotation-only locally and depends on the original Cholec80 videos plus `phase_annotations/` for the official preprocessing pipeline.
- `CholecTrack20` counts in this document are dataset-wide totals unless a split is explicitly named.
- `Endoscapes2023` contains 58,813 local JPGs; the 228-frame discrepancy is confined to `test/annotation_coco_vid.json`.

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

These overlaps place leakage control at the **original video ID level** for pretraining, fine-tuning, and evaluation.

### 4.2 Ontology mismatch

- Cholec80 phases and CholecT50 phases are close but not identical
- Cholec80-CVS uses per-criterion ordinal scores `{0, 1, 2}` plus a raw `critical_view` flag defined by `Total >= 5`; the official loader instead binarizes each criterion and ignores the raw overall flag
- Cholec80 tools include `SpecimenBag`; CholecT50 instrument labels do not
- CholecInstanceSeg includes `snare`, which is absent from Cholec80 and CholecT50
- Endoscapes uses anatomy-plus-tool categories rather than triplets or dense workflow states
- GraSP labels phases, steps, instruments, and atomic actions, but in robotic prostatectomy

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

Sampling alignment is a primary preprocessing requirement for this dataset collection.

### 4.4 Domain mismatch

- Cholec80, Cholec80-CVS, CholecT50, CholecSeg8k, CholecInstanceSeg, Endoscapes2023, CholecTrack20: laparoscopic cholecystectomy
- AutoLaparo: laparoscopic hysterectomy
- GraSP: robot-assisted radical prostatectomy

GraSP and AutoLaparo differ from the cholecystectomy datasets in procedure and platform, so their labels align more naturally with representation learning or auxiliary pretraining than with direct label pooling.

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
- M2CAI challenge: <http://camma.u-strasbg.fr/m2cai2016/index.php/program-challenge/>
- Walimbe et al. (MICCAI 2025): <https://arxiv.org/abs/2507.05020>
