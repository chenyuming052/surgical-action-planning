# SurgCast — Surgical Action-Change Anticipation

## Project Overview

NeurIPS 2026 submission. Event-centric surgical anticipation under heterogeneous missing supervision across 4 datasets (CholecT50, Cholec80, Cholec80-CVS, Endoscapes2023), 277 videos, 7 coverage groups (G1-G7).

- Research proposal: `docs/surgcast_proposal.md` (v5.4, ~128K — read specific sections, not whole file)
- Data audit: `docs/data-analysis.md`
- NPZ schema: `docs/dataset_contract.md`
- Experiment plan: `docs/experiment_matrix.md`

## Architecture

```
surgcast/                  # Flat layout (no src/ wrapper)
├── models/                # All model code
│   ├── surgcast.py        # SurgCastModel — top-level wiring
│   ├── temporal_encoder.py  # CausalTemporalTransformer (causal masked self-attention)
│   ├── transition.py      # HorizonConditionedTransition (Δ={1,3,5,10}s)
│   ├── heads.py           # MultiTaskHeads (triplet_group, instrument, phase, anatomy, cvs)
│   ├── hazard_head.py     # DualHazardHead (shared trunk → inst + group heads)
│   ├── backbone.py        # BackboneSpec dataclass (dinov3_vitb16, lemonfm)
│   └── prior.py           # StructuredPrior (TODO: categorical + Bernoulli)
├── loss/                  # Singular (PyTorch convention)
│   ├── hazard_loss.py     # discrete_time_hazard_nll (vectorized)
│   └── multitask.py       # masked_bce, masked_ce
├── datasets/              # Data loading (not raw data — raw data lives at /yuming/data)
│   ├── sequence_dataset.py  # SequenceDataset + SequenceSample + collate_fn
│   ├── sampler.py         # CoverageAwareSampler (weighted G1-G7)
│   ├── registry.py        # load_registry, filter_by_split
│   └── npz_loader.py      # load_npz
├── metrics/               # Plural (community convention). ALL STUBS.
│   ├── change.py          # TODO: Change-mAP
│   ├── ttc.py             # TODO: TTC MAE, C-index, Brier
│   └── safety.py          # TODO: CVS AUC, CVS MAE
└── utils/
    ├── seed.py            # set_seed
    └── io.py              # load_yaml, save_json
```

## Model Pipeline

```
Input [B, T, 768] (frozen backbone features)
  → CausalTemporalTransformer → h [B, T, 512]
  → HorizonConditionedTransition (×4 horizons) → pred_state + log_var
  → σ_agg = stack(sqrt(mean(exp(log_var)))) → [B, T, 4]
  → MultiTaskHeads(h) → triplet_group, instrument, phase, anatomy, [cvs]
  → DualHazardHead(h, σ_agg) → hazard_inst, hazard_group [B, T, 20]
```

~7.7M trainable parameters (2-layer encoder for dev; 6-layer for full ~11.2M).

## Key Conventions

- **Naming**: `loss/` singular, `metrics/` `utils/` plural. Files singular (`backbone.py` not `backbones.py`). `hazard_head.py` for model, `hazard_loss.py` for loss (disambiguated).
- **Imports**: Use `from surgcast.models import SurgCastModel`, not relative across packages.
- **Config**: YAML in `configs/{data,model,train,eval}/default.yaml`. No OmegaConf yet — plain `yaml.safe_load`.
- **Data contract**: NPZ per-video labels, HDF5 features. See `docs/dataset_contract.md` for exact array shapes/dtypes.
- **Masks**: Visibility masks distinguish absent-from-dataset vs unobserved-in-frame. All losses use mask-weighted averaging.
- **Hazard bins**: K=20 non-uniform intervals. Edges in `configs/data/default.yaml`.
- **Coverage groups**: G1-G7 with weighted sampling. Probs in `configs/data/default.yaml`.

## Implementation Status

| Component | Status | Key gap |
|-----------|--------|---------|
| Models | ~95% | `StructuredPrior` is a stub |
| Losses | ~90% | Focal loss, prior KL not wired |
| Datasets | ~95% | Needs real data to test |
| Metrics | 0% | All 3 modules are stubs |
| Scripts | 1/9 done | Only `build_registry.py` complete |
| Tests | smoke only | `tests/test_smoke.py` passes |

## Scripts

Only `scripts/build_registry.py` is implemented (870 lines, merges 4 datasets → `registry.json`). All others are argparse stubs with `raise SystemExit('TODO: ...')`.

Execution order: `build_registry → preprocess_* → extract_features → build_priors → train → evaluate`. See `docs/runbook.md`.

## Engineering Constraints

- Build canonical registry BEFORE any split or feature extraction
- Cholec80-CVS: do NOT use official 85% truncation or 50/15/15 split — parse XLSX directly
- All priors computed from training split only
- Change task uses dual-event TTC: instrument-set + triplet-group
- Anatomy observation mask only on bbox-annotated frames (Endoscapes)
- Hazard loss: L_hazard = L_inst + η·L_group (η=1.0 default)
- Targets with value -1 are ignored (masked out) in cross-entropy losses

## Running

```bash
pip install -e .
python -c "from surgcast.models.surgcast import SurgCastModel; print('OK')"
python tests/test_smoke.py
```

## Deployment (K8s)

K8s Job YAMLs in `deploy/k8s/`. Apply in order:
- `0-data-transfer.yaml` — interactive pod for data transfer to PVC
- `1-git-operation.yaml` — clone repo onto PVC
- `2-train.yaml` — GPU training job (2×GPU default)

Infrastructure: PVC `yuming-pvc-2tb`, Docker image `docker.aiml.team/yuming.chen/surgical-action-planning:latest`, secrets `github-token` + `gitlab-docker-secret`.
Reference project: `/yuming/projects/temporal-perception/MedST/deploy/k8s/` for patterns.

## Editing Guidelines

- Preserve LaTeX math notation and table formatting in `surgcast_proposal.md`
- All numerical claims must be verified against `docs/data-analysis.md`
- When adding new modules: add to the relevant `__init__.py` exports
- When adding new config keys: update corresponding default YAML
- Tests go in `tests/`. Smoke test covers forward+backward; add integration tests as preprocessing is implemented.
