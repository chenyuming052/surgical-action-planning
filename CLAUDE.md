# SurgCast — Surgical Action-Change Anticipation

## Project Overview

NeurIPS 2026 submission. Event-centric surgical anticipation under heterogeneous missing supervision across 4 training datasets (CholecT50, Cholec80, Cholec80-CVS, Endoscapes2023), 277 videos, 7 coverage groups (G1-G7), plus HeiChole (24 publicly labeled videos) for external validation.

- Research proposal: `docs/surgcast-proposal.md` (v8.0, ~130K — read specific sections, not whole file)
- Data audit: `docs/data-analysis.md`
- NPZ schema: `docs/dataset-contract.md`
- Experiment plan: `docs/experiment-matrix.md`

## Architecture

```
surgcast/                  # Flat layout (no src/ wrapper)
├── models/                # All model code
│   ├── __init__.py        # Exports + build_model() factory
│   ├── surgcast.py        # SurgCastModel — top-level wiring (event-conditioned pipeline)
│   ├── temporal_encoder.py  # CausalTemporalTransformer (causal masked self-attention)
│   ├── heads.py           # MultiTaskHeads (triplet_group, instrument, phase, anatomy, cvs with anatomy injection)
│   ├── hazard_head.py     # DualHazardHead (phase-gated, residual experts) + StateAgeEncoder
│   ├── backbone.py        # BackboneSpec dataclass (dinov3_vitb16, lemonfm)
│   ├── prior.py           # StructuredPrior — optional regularizer (TODO: categorical + Bernoulli)
│   ├── action_encoder.py  # ActionTokenEncoder (coarse+fine token fusion, teacher forcing)
│   ├── event_dyn.py       # EventDyn (FiLM-conditioned dynamics) / ActionConditionedTransition (dynamics_version A)
│   └── next_action_head.py  # NextActionHead (delta-state post-change prediction)
├── training/              # Training execution layer
│   ├── __init__.py        # Exports: Trainer, save/load_checkpoint, TrainingLogger
│   ├── trainer.py         # Train/val loop, mixed precision, DDP, gradient accumulation, teacher forcing
│   ├── checkpoint.py      # Save/load with full config + git hash
│   └── logger.py          # W&B wrapper with JSON fallback
├── loss/                  # Singular (PyTorch convention)
│   ├── hazard_loss.py     # discrete_time_hazard_nll (vectorized)
│   └── multitask.py       # masked_bce, masked_ce, ordinal_bce_cvs, ranking, heteroscedastic_nll, next_action, consistency
├── datasets/              # Data loading (not raw data — raw data lives at /yuming/data)
│   ├── sequence_dataset.py  # SequenceDataset + SequenceSample + collate_fn (NPZ caching)
│   ├── sampler.py         # CoverageAwareSampler (weighted G1-G7)
│   ├── registry.py        # load_registry (envelope + flat-list compat), filter_by_split
│   └── npz_loader.py      # load_npz
├── metrics/               # Plural (community convention). 15 metrics implemented.
│   ├── change.py          # Event-AP @horizon, Event-AUROC, Post-change mAP, dense mAP, change-conditioned
│   ├── ttc.py             # TTC MAE, expected TTC, C-index, Brier score, hazard calibration
│   └── safety.py          # CVS criterion AUC, CVS MAE, clipping detection/false-alarm, CVS MAE @clipping
└── utils/
    ├── seed.py            # set_seed
    ├── io.py              # load_yaml, save_json
    ├── config.py          # load_config, deep_merge, parse_overrides
    ├── change_point.py    # extract_*_changes, compute_ttc_targets, debounce_changes
    └── triplet_clustering.py  # co-occurrence + semantic clustering for triplet groups
configs/
├── data/default.yaml      # seed, fps, hazard bins, coverage groups
├── model/default.yaml     # backbone, encoder, heads, hazard, action_encoder, event_dyn, dynamics_version
├── train/default.yaml     # optimizer, scheduler, trainer, loss weights, teacher forcing
├── train/local_debug.yaml # batch_size=4, num_workers=0, epochs=2, fp32
├── train/cluster.yaml     # K8s cluster overrides (batch_size=128, num_workers=16)
├── eval/default.yaml      # tiers, metrics list
└── experiment/            # Per-stage/ablation overrides (9 stages)
    ├── cholec_only.yaml
    ├── plus_phase.yaml
    ├── plus_tool_presence.yaml
    ├── plus_cvs.yaml
    ├── plus_endoscapes.yaml
    ├── plus_masking.yaml
    ├── plus_transition.yaml
    ├── plus_static_prior.yaml
    └── full.yaml
```

## Model Pipeline

```
Input [B, T, 768] (frozen backbone features)
  → CausalTemporalTransformer → h [B, T, 512]
  → MultiTaskHeads(h) → triplet_group, instrument, phase, anatomy, [cvs]
  → ActionTokenEncoder(y_t or ŷ_t, rho) → a_t [B, T, 64]
  → NextActionHead(h, a_t) → delta_add, delta_remove, phase_next, group_next
  → StateAgeEncoder(age_features) → age_embed [B, T, 16]
  → DualHazardHead(h, a_t, d_t, age_embed) → hazard_inst, hazard_group [B, T, 20]
  → EventDyn(h, tau_bin) → mu_plus, log_var [B, T, 512]  (dynamics_version="B")
    or ActionConditionedTransition(h, a_t, horizon) per horizon   (dynamics_version="A")
```

~8.2M trainable parameters (2-layer encoder for dev; 6-layer for full ~11.2M).
`dynamics_version` ("A" or "B") is an internal ablation switch, not a separate model.

## Key Conventions

- **Naming**: `loss/` singular, `metrics/` `utils/` plural. Files singular (`backbone.py` not `backbones.py`). `hazard_head.py` for model, `hazard_loss.py` for loss (disambiguated). Docs use hyphens (`data-analysis.md`); code/artifacts/scripts use underscores (`registry_summary.csv`).
- **Imports**: Use `from surgcast.models import SurgCastModel`, not relative across packages.
- **Config**: YAML in `configs/{data,model,train,eval}/default.yaml` + `configs/experiment/` for stage overrides. Merging via `load_config(*yamls, overrides=dict)` in `surgcast.utils.config`. CLI overrides via `--override key.subkey=value`.
- **Data contract**: NPZ per-video labels, HDF5 features. See `docs/dataset-contract.md` for exact array shapes/dtypes.
- **Masks**: Visibility masks distinguish absent-from-dataset vs unobserved-in-frame. All losses use mask-weighted averaging.
- **Hazard bins**: K=20 non-uniform intervals. Edges in `configs/data/default.yaml`.
- **Coverage groups**: G1-G7 with weighted sampling. Probs in `configs/data/default.yaml`.

## Implementation Status

| Component | Status | Key gap |
|-----------|--------|---------|
| Models | ~95% | Unified architecture with dynamics_version ablation; `StructuredPrior` still stub; `build_model()` factory in `__init__.py` |
| Training | ~95% | `Trainer` with DDP, warmup scheduler (SequentialLR), mixed precision, teacher forcing, model input extraction; `torchrun` ready |
| Losses | ~95% | All losses implemented (ordinal_bce_cvs, ranking, consistency, next_action, heteroscedastic_nll); focal + prior KL not wired |
| Datasets | ~95% | NPZ caching + label loading (CVS, triplet indices, change flags, age features); `load_registry` handles envelope format; needs real data to test |
| Metrics | ~95% | All 15 metrics implemented across 3 modules; tested with synthetic data |
| Config | ~95% | `load_config` with deep_merge + CLI overrides; 9 experiment configs; `cluster.yaml` for K8s |
| Scripts | ~40% | `build_registry.py`, `train.py`, `evaluate.py` complete; `preprocess/_common.py` added; baselines still stub |
| Tests | smoke | `test_smoke.py` (unified model forward+backward) + `test_smoke_components.py` (all submodules + imports) |
| Deploy | ~90% | K8s typo fixed; `torchrun` for DDP; `cluster.yaml` config; `Dockerfile` added |

## Scripts

```
scripts/
├── train.py                 # ★ 训练入口 (DDP + warmup + unified loss)
├── evaluate.py              # ★ 分层评估 (Tier 1-5)
├── run_local_debug.sh       # ★ 一键本地 debug（单卡, batch=4, fp32, 2 epochs）
├── run_cluster.sh           # ★ 一键集群训练（torchrun 多卡, bf16）
├── data/                    # 数据准备 pipeline（训练前执行一次）
│   ├── build_registry.py    # ★ 已实现 (1327行). 4数据集 → registry.json
│   ├── build_priors.py      # 训练集 → 先验权重 (stub)
│   ├── build_triplet_groups.py  # 共现聚类 → triplet groups (stub)
│   ├── extract_features.py  # 视频帧 → HDF5 特征 (stub)
│   └── preprocess/          # 原始标注 → NPZ (全部 stub)
│       ├── _common.py       # ★ 已实现. Shared preprocessing utilities
│       ├── cholect50.py
│       ├── cholec80.py
│       ├── cholec80_cvs.py
│       ├── endoscapes.py
│       └── heichole.py
├── paper/                   # 论文工具
│   └── generate_paper_stats.py  # registry → LaTeX 宏 (stub)
└── baselines/               # 基线实现 (全部 stub)
    ├── copy_current.py
    ├── surgfutr_style.py
    └── mml_surgadapt_style.py
```

Data preparation order: `data/build_registry → data/preprocess/* → data/extract_features → data/build_priors`.
Training: `train → evaluate`. See `docs/runbook.md`.

## Data Split (CAMMA Combined Strategy)

168 train / 48 val / 61 test. Per group:
- G1: 3/0/0, G2: 28/5/9, G3: 3/0/0, G4: 3/0/0, G5: 1/0/1, G6: 17/4/11, G7: 113/39/40

Evaluation tiers (test counts): Tier 1=10, Tier 2a=51 (Auxiliary), Tier 2b=9 (Auxiliary), Tier 3=20, Tier 4=21, Tier 5=40, Tier 6=24 (HeiChole external).
All tiers meet or exceed original proposal estimates. G3-test=0 (all in train by CAMMA).

## Engineering Constraints

- Build canonical registry BEFORE any split or feature extraction
- Cholec80-CVS: do NOT use official 85% truncation or 50/15/15 split — parse XLSX directly
- All priors computed from training split only
- Change task uses dual-event TTC: instrument-set (primary) + triplet-group (secondary)
- Anatomy observation mask only on bbox-annotated frames (Endoscapes)
- Hazard loss: L_hazard = L_inst + η·L_group (η=1.0 default)
- Targets with value -1 are ignored (masked out) in cross-entropy losses

## Running

```bash
pip install -e .
python -c "from surgcast.models.surgcast import SurgCastModel; print('OK')"
python tests/test_smoke.py
python tests/test_smoke_components.py
```

## How to Train

**Local debug** — single GPU, batch=4, fp32, 2 epochs:
```bash
./scripts/run_local_debug.sh debug_test
./scripts/run_local_debug.sh debug_cholec --experiment configs/experiment/cholec_only.yaml --stage cholec_only
./scripts/run_local_debug.sh debug_test --override loss.lambda_hazard=0.5
```

**Cluster (multi-GPU)** — torchrun, batch=128, bf16:
```bash
./scripts/run_cluster.sh surgcast_full --experiment configs/experiment/full.yaml --stage full
NGPU=4 ./scripts/run_cluster.sh surgcast_4gpu --stage full
```

**Direct train.py** (full control over config stacking):
```bash
python scripts/train.py \
    --config-data configs/data/default.yaml \
    --config-model configs/model/default.yaml \
    --config-train configs/train/default.yaml configs/train/local_debug.yaml \
    --experiment configs/experiment/cholec_only.yaml \
    --stage cholec_only \
    --registry data/registry.json \
    --features-root /yuming/data/surgcast/features \
    --npz-root /yuming/data/surgcast/npz \
    --run-name "my_run" \
    --override loss.lambda_hazard=0.5 trainer.batch_size=8
```

**Config layering**: `default.yaml` → `local_debug.yaml` or `cluster.yaml` → `experiment/*.yaml` → `--override`. Later values win.
Environment variables `FEATURES_ROOT`, `NPZ_ROOT`, `REGISTRY` override data paths in shell scripts.

## Output Directory Structure

Each training run saves to `outputs/{run_name}/`:
```
outputs/{run_name}/
├── config.yaml         # Full merged config (auto-saved at first checkpoint)
├── train_log.jsonl     # Per-step JSON metrics
├── summary.json        # Best metrics summary
└── checkpoints/
    ├── best.pt         # Best validation loss
    ├── last.pt         # Final epoch
    └── epoch_0004.pt   # Periodic checkpoints
```

## Deployment (K8s)

K8s Job YAMLs in `deploy/k8s/`. `2-train.yaml` calls `run_cluster.sh` internally.

```bash
# Setup (once)
kubectl apply -f deploy/k8s/secrets.yaml        # HF_TOKEN, WANDB_API_KEY
kubectl apply -f deploy/k8s/0-data-transfer.yaml # interactive pod → upload data to PVC
kubectl apply -f deploy/k8s/1-git-operation.yaml # clone repo onto PVC

# Train
kubectl apply -f deploy/k8s/2-train.yaml        # 2×GPU, calls run_cluster.sh
kubectl logs -f job/surgcast-train-full-gpu2      # watch progress
```

Infrastructure: PVC `yuming-pvc-2tb`, Docker image `docker.aiml.team/yuming.chen/surgcast:latest`, `Dockerfile` in repo root.
Secrets: `surgcast-secrets`, `github-token`, `gitlab-docker-secret`.

## Editing Guidelines

- Preserve LaTeX math notation and table formatting in `surgcast-proposal.md`
- All numerical claims must be verified against `docs/data-analysis.md`
- When adding new modules: add to the relevant `__init__.py` exports
- When adding new config keys: update corresponding default YAML
- Tests go in `tests/`. Smoke test covers forward+backward; add integration tests as preprocessing is implemented.
