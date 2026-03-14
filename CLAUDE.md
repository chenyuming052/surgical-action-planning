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
│   ├── surgcast.py        # SurgCastModel — top-level wiring
│   ├── temporal_encoder.py  # CausalTemporalTransformer (causal masked self-attention)
│   ├── transition.py      # HorizonConditionedTransition / Latent Rollout Module (Δ={1,3,5,10}s)
│   ├── heads.py           # MultiTaskHeads (triplet_group, instrument, phase, anatomy, cvs)
│   ├── hazard_head.py     # DualHazardHead (shared trunk → inst + group heads)
│   ├── backbone.py        # BackboneSpec dataclass (dinov3_vitb16, lemonfm)
│   ├── prior.py           # StructuredPrior — optional regularizer (TODO: categorical + Bernoulli)
│   ├── action_encoder.py  # ActionTokenEncoder (coarse+fine token fusion, teacher forcing)
│   ├── event_dyn.py       # EventDyn / ActionConditionedTransition (FiLM-conditioned dynamics)
│   └── next_action_head.py  # NextActionHead (delta-state post-change prediction)
├── training/              # Training execution layer
│   ├── __init__.py        # Exports: Trainer, save/load_checkpoint, TrainingLogger
│   ├── trainer.py         # Train/val loop, mixed precision, DDP, gradient accumulation
│   ├── checkpoint.py      # Save/load with full config + git hash
│   └── logger.py          # W&B wrapper with JSON fallback
├── loss/                  # Singular (PyTorch convention)
│   ├── hazard_loss.py     # discrete_time_hazard_nll (vectorized)
│   └── multitask.py       # masked_bce, masked_ce, ordinal_bce_cvs, ranking, heteroscedastic_nll, next_action, consistency
├── datasets/              # Data loading (not raw data — raw data lives at /yuming/data)
│   ├── sequence_dataset.py  # SequenceDataset + SequenceSample + collate_fn (NPZ caching)
│   ├── sampler.py         # CoverageAwareSampler (weighted G1-G7)
│   ├── registry.py        # load_registry, filter_by_split
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
├── model/default.yaml     # backbone, encoder, transition, heads, hazard, V2 modules
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
  → Latent Rollout Module / HorizonConditionedTransition (×4 horizons) → pred_state + log_var
  → σ_agg = stack(sqrt(mean(exp(log_var)))) → [B, T, 4]
  → MultiTaskHeads(h) → triplet_group, instrument, phase, anatomy, [cvs]
  → DualHazardHead(h, σ_agg) → hazard_inst, hazard_group [B, T, 20]

V2 additions:
  ActionTokenEncoder(y_t or ŷ_t, rho) → a_t [B, T, 64]
  EventDyn(h_t, a_t, hazard_bins) → mu_plus, log_var (FiLM-conditioned)
  NextActionHead(h_t, a_t) → delta_add, delta_remove, phase_next, group_next
  PhaseGatedDualHazardHead(h_t, σ_agg, phase_logits) → hazard_inst, hazard_group
```

~7.7M trainable parameters (2-layer encoder for dev; 6-layer for full ~11.2M).

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
| Models | ~95% | V1 + V2 forward passes implemented; `StructuredPrior` still stub; `build_model()` factory in `__init__.py` |
| Training | ~90% | `Trainer` with DDP, warmup scheduler (SequentialLR), mixed precision; `train.py` with `build_loss_fn_v2`; `torchrun` ready |
| Losses | ~95% | All V2 losses implemented (ordinal_bce_cvs, ranking, consistency, next_action, heteroscedastic_nll); focal + prior KL not wired |
| Datasets | ~95% | NPZ caching + V2 label loading (CVS, triplet indices, change flags, age features); needs real data to test |
| Metrics | ~95% | All 15 metrics implemented across 3 modules; tested with synthetic data |
| Config | ~95% | `load_config` with deep_merge + CLI overrides; 9 experiment configs; `cluster.yaml` for K8s |
| Scripts | ~40% | `build_registry.py`, `train.py`, `evaluate.py` complete; `preprocess/_common.py` added; baselines still stub |
| Tests | smoke v1+v2 | `test_smoke.py` + `test_smoke_v2.py` (V2 forward+backward verified) |
| Deploy | ~90% | K8s typo fixed; `torchrun` for DDP; `cluster.yaml` config; `Dockerfile` added |

## Scripts

```
scripts/
├── build_registry.py        # ★ 已实现 (1327行). 4数据集 → registry.json
├── build_priors.py          # 训练集 → 先验权重 (stub)
├── build_triplet_groups.py  # 共现聚类 → triplet groups (stub)
├── extract_features.py      # 视频帧 → HDF5 特征 (stub)
├── train.py                 # ★ 已实现. 训练入口 (DDP + warmup + V2 loss)
├── evaluate.py              # ★ 已实现. 分层评估 (Tier 1-5)
├── generate_paper_stats.py  # registry → LaTeX 宏 (stub)
├── preprocess/              # 原始标注 → NPZ (全部 stub)
│   ├── _common.py           # ★ 已实现. Shared preprocessing utilities
│   ├── cholect50.py
│   ├── cholec80.py
│   ├── cholec80_cvs.py
│   ├── endoscapes.py
│   └── heichole.py
└── baselines/               # 基线实现 (全部 stub)
    ├── copy_current.py
    ├── surgfutr_style.py
    └── mml_surgadapt_style.py
```

Execution order: `build_registry → preprocess/* → extract_features → build_priors → train → evaluate`. See `docs/runbook.md`.

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
python tests/test_smoke_v2.py    # V2 modules (action_encoder, event_dyn, next_action_head)

# Training (local debug)
python scripts/train.py \
    --config-data configs/data/default.yaml \
    --config-model configs/model/default.yaml \
    --config-train configs/train/default.yaml configs/train/local_debug.yaml \
    --experiment configs/experiment/cholec_only.yaml \
    --stage cholec_only \
    --registry data/registry.json \
    --features-root /yuming/data/surgcast/features \
    --npz-root /yuming/data/surgcast/npz \
    --run-name "debug_test"

# With CLI overrides
python scripts/train.py ... --override loss.lambda_hazard=0.5 trainer.batch_size=8
```

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

K8s Job YAMLs in `deploy/k8s/`. Apply in order:
- `0-data-transfer.yaml` — interactive pod for data transfer to PVC
- `1-git-operation.yaml` — clone repo onto PVC
- `2-train.yaml` — GPU training job (2×GPU default)
- `secrets.yaml.template` — template for K8s Secrets (HF_TOKEN, WANDB_API_KEY)

Secrets: `surgcast-secrets` (HF + W&B tokens via SecretRef), `github-token`, `gitlab-docker-secret`.
Infrastructure: PVC `yuming-pvc-2tb`, Docker image `docker.aiml.team/yuming.chen/surgcast:latest`, `Dockerfile` in repo root.
Reference project: `/yuming/projects/temporal-perception/MedST/deploy/k8s/` for patterns.

## Editing Guidelines

- Preserve LaTeX math notation and table formatting in `surgcast-proposal.md`
- All numerical claims must be verified against `docs/data-analysis.md`
- When adding new modules: add to the relevant `__init__.py` exports
- When adding new config keys: update corresponding default YAML
- Tests go in `tests/`. Smoke test covers forward+backward; add integration tests as preprocessing is implemented.
