# SurgCast: Event-Centric Forecasting of Surgical Action Changes under Heterogeneous Partial Supervision

**Target venue: NeurIPS 2026**
**Extension path: TMI / Nature subsidiary journal**

---

## 1. Problem Definition

Given a past surgical video context, SurgCast predicts *when* the next action change will occur and *what* instrument/action state will follow. The model is trained under heterogeneous missing supervision from multiple partially overlapping datasets (CholecT50, Cholec80, Cholec80-CVS, Endoscapes2023), with auxiliary anatomical safety assessment tasks. Cross-center generalization is validated on HeiChole (24 publicly labeled videos) as an external dataset. This framing is more robust to temporal inertia than dense anticipation, equipped with an action-conditioned observer world model for latent state forecasting, more novel than plain phase anticipation, and better aligned with audited supervision.

---

## 2. Motivation and Research Questions

### 2.1 Why Action-Change Anticipation

The surgical video understanding literature focuses on two directions: current-frame recognition (TeCNO, Trans-SVNet, Rendezvous) and future-state anticipation. Recent anticipation work has pushed in two directions: (1) SuPRA/SWAG/SurgFUTR advance anticipation from simple dense prediction toward workflow forecasting and state-change learning; (2) MML-SurgAdapt/SPML perform partial-annotation multi-task learning under multi-source settings. However, even after these advances, few works jointly study event-centric change-time prediction (rather than dense-step or implicit state-change), explicit discrete-time survival modeling for TTC, and cross-center generalization validation.

Nearly all existing anticipation methods still work within one of two paradigms: "predict the action class at future time Δ" (dense-step), or "detect whether a state change has occurred" (binary change detection).

This formulation has a fundamental flaw: in surgical video, the same action persists across many consecutive seconds. A copy-current baseline (predicting the current action as the future action) achieves high scores on dense per-second metrics. Existing metrics are heavily inflated by temporal inertia, drowning out the signal that reflects genuine predictive ability.

**Temporal inertia diagnosis at 1fps:** Under 1fps sampling, adjacent-frame state transitions are overwhelmingly identity (no-change). Data audit quantifies this: CholecT50 triplet-set changes occur only ~6.17 times/minute, instrument-set changes ~4.07 times/minute, and phase transitions are even sparser (~0.17 times/minute). This means that out of 60 adjacent frame pairs, only ~4–6 exhibit instrument-set change; the remaining ~54–56 are identity transitions. The high accuracy of dense next-step prediction is therefore inflated by temporal inertia — it is not that models genuinely "predict" the future, but that the correct answer at most timesteps is simply "no change." Furthermore, 47.4% of CholecT50 frames contain multiple concurrent action instances, confirming that action targets are inherently set-valued rather than single-class. This diagnosis directly motivates event-centric forecasting and change-conditioned evaluation.

**We pose a sharper question: action-change anticipation.** Not "what happens at time t+Δ," but:

1. **When:** How long until the next action change? (time-to-next-change, TTC)
2. **What:** What will the state be after the change? (post-change state prediction)
3. **Whether safe:** Is this change compatible with the current anatomical safety state? (safety compatibility)

These three sub-questions together constitute a complete surgical decision-support scenario — more clinically meaningful than predicting the next-second triplet, and better at distinguishing genuinely predictive models from trivial baselines.

We propose a stricter event-centric anticipation protocol that complements existing dense-step and state-change paradigms. Instrument-set change serves as the primary benchmark, eliminating the fundamental problem of temporal inertia inflating dense-step metrics; triplet-group change serves as a secondary semantic stress-test. Primary metrics are divided into three semantically clear subcategories: A1 event timing (When: TTC MAE, C-index, Brier), A2 event detection (Whether: Event-AP @horizon), and A3 post-change state prediction (What: Post-change mAP). Change-conditioned evaluation requires all dense metrics to be reported in stratified form (change vs non-change frames), ensuring the benchmark distinguishes genuinely predictive models from trivial copy-current baselines. Dense-step metrics (per-second triplet-group mAP, phase accuracy @Δ) are retained as secondary metrics for backward compatibility. The protocol applies to all baselines and is not a closed proprietary metric.

### 2.2 Why This Problem Requires Multi-Source Data

Action-change anticipation inherently requires three types of information:

- **Action semantics** (current actions, historical action patterns) → CholecT50 provides triplet annotations (instrument × verb × target)
- **Workflow context** (current surgical phase, typical action durations within each phase) → Cholec80 provides full-procedure phase annotations
- **Anatomical safety state** (whether critical structures are adequately exposed, whether safe transition conditions are met) → Cholec80-CVS provides full-procedure CVS annotations + Endoscapes2023 provides ROI-window CVS annotations and anatomy bounding boxes

**No single dataset provides all annotations simultaneously.** CholecT50 has triplets and phases but no CVS/anatomy; Cholec80 has only phases and tool presence; Cholec80-CVS adds CVS to Cholec80 but has no triplets; Endoscapes has anatomy bboxes and CVS but covers only a temporal window of each procedure.

Therefore, effective action-change anticipation requires jointly leveraging these dispersed supervisory signals. However, complex video overlaps, partially inconsistent label ontologies, and different temporal granularities across datasets preclude naive merging.

**This is the core training paradigm challenge: how to jointly train on heterogeneous missing supervision from multiple sources so that dispersed supervisory signals reinforce each other.** Structured prior regularization is one optional mechanism for addressing this challenge, but the core contribution lies in the overlap-safe multi-source protocol itself and the observer world model method.

### 2.3 Three Core Research Questions

**RQ1: Does event-centric change forecasting outperform dense fixed-horizon anticipation under strong temporal inertia?**
When evaluation is restricted to meaningful future changes (rather than all timesteps), does discrete-time hazard modeling better reflect genuine predictive ability compared to dense-step and implicit state-change methods? Validated via: Tier 1 evaluation + Copy-Current baseline + Change-conditioned evaluation.

**RQ2: Does action-conditioned observer dynamics (world model) improve event forecasting compared to action-free prediction?**
Can an observer world model — conditioning future-state predictions on observed/predicted action tokens and event structure — produce better event forecasting than action-free latent prediction? Does heterogeneous missing supervision from phase, tool, CVS, and anatomy datasets mutually reinforce each other through multi-source training, yielding better surgical state representations? Validated via: ablation A_dynamics (event-conditioned vs fixed-horizon), A_action1 (action-free vs action-conditioned), and incremental data ablation (CholecT50-only → +Cholec80 → +CVS → +Endoscapes).

**RQ3: Can the learned event representation transfer to a held-out Heidelberg cohort under cross-center and ontology shift?**
Does the representation learned from multi-source training generalize across centers? Validated via: Tier 6 HeiChole external validation (24 publicly labeled videos, zero-shot + optional few-shot).

---

## 3. Positioning Relative to Prior Work

### 3.1 Positioning Table

| Dimension | SuPRA / SWAG | MML-SurgAdapt / SPML | SurgFUTR | SurgCast (this work) |
|---|---|---|---|---|
| Prediction target | Future phase / instrument | Multi-task recognition (current frame) | Future state-change | **Action change time + post-change state set (+ auxiliary safety)** |
| Explicit change-time modeling | No | No | Partial (state-change detection) | **Yes (discrete-time hazard, survival function output)** |
| Handles heterogeneous missing labels | No (single dataset) | **Yes (partial-annotation multi-task)** | Partial (cross-dataset benchmark) | **Yes (evidence-gated structured prior)** |
| Multi-source training | No | **Yes (CholecT50+Cholec80+Endoscapes)** | Yes | **Yes (4 data sources, leakage-safe protocol)** |
| External data validation | No | No | Partial | **Yes (HeiChole 24 videos zero-shot)** |
| Event-centric evaluation | No (dense per-second) | No (per-frame recognition) | Partial | **Yes (change-point mAP + TTC + safety detection)** |
| Missing-label handling | N/A | Partial-label masking | Dataset-specific heads | **Overlap-safe multi-source protocol (+ optional prior)** |
| Action-conditioned dynamics | No | No | No | **Yes (observer world model)** |
| Temporal modeling | Dense future step | None (current frame) | Future state prediction | **Discrete-time survival analysis + action-conditioned observer dynamics** |

### 3.2 Differentiation

**vs. SuPRA / SWAG:** These works define anticipation as dense future step prediction ("what phase/instrument at time t+Δ"). We shift the prediction target from dense steps to event-centric change forecasting, complementing and validating existing methods through a stricter evaluation protocol.

**vs. MML-SurgAdapt / SPML:** MML-SurgAdapt performs partial-annotation multi-task learning on CholecT50+Cholec80+Endoscapes, but its goal is current-frame recognition, not anticipation or temporal forecasting. Our multi-source protocol serves a forecasting task and models future state trajectories via an action-conditioned observer world model.

**vs. SurgFUTR:** SurgFUTR recasts future prediction as state-change learning and establishes a cross-dataset benchmark. Our core differences are: (a) explicit discrete-time survival modeling (hazard function + survival function) for time-to-change, outputting a calibrated survival function rather than implicit state-change detection; (b) explicit post-change set prediction (predicting the post-change instrument/action state set); (c) heterogeneous partial supervision training protocol rather than single- or two-dataset settings; (d) HeiChole external dataset validation for cross-center generalization.

**vs. surgical world models:** Explicit surgical world-model works exist as of 2025–2026, but they typically pursue full interventional dynamics (action selection, counterfactual reasoning, generative video decoding). SurgCast occupies a different niche: it is an **observer world model** that conditions future-state predictions on observed/predicted action tokens, but does NOT select actions, does NOT perform counterfactual reasoning, and does NOT generate video frames. This is a precise, bounded claim — we inherit the core world-model structure (encode → dynamics → task) and "imagining the future in latent space," but restrict the scope to anticipation (forecasting) rather than control (policy). We make this distinction explicit and validate it through ablation (action-free vs action-conditioned dynamics, fixed-horizon vs event-conditioned transition).

**Summary:** This work jointly validates hazard-based event-time prediction + post-change set forecasting + overlap-safe multi-source training under a stricter event-centric setting. The core positioning is: **explicit discrete-time event modeling with post-change set prediction under heterogeneous partial supervision**. Under leakage-safe multi-source heterogeneous missing supervision, we predict the time to the next action change (discrete-time hazard TTC), the post-change action set (multi-label post-change state), and validate cross-center generalization on HeiChole (24 publicly labeled videos). Safety-related tasks (CVS state, anatomy presence) serve as auxiliary evaluation dimensions.

---

## 4. Design Rationale

This section distills the key design decisions and their justifications.

### 4.1 Why Instrument-Set Change Is the Primary Target

Instrument-set change (~4.07/min) provides a clean, well-defined signal: new instruments appearing or existing instruments being removed. Unlike triplet-group change, it does not depend on clustering definitions, eliminating reviewer concerns about artificial target construction. It is coarser than triplet-set change (~6.17/min, heavily contaminated by label flicker) but finer than phase transitions (~0.17/min, too sparse for continuous prediction).

### 4.2 Why Event-Centric Rather Than Dense-Step Anticipation

At 1fps, ~90% of adjacent frame pairs are identity transitions. Dense next-step prediction accuracy is dominated by temporal inertia — a copy-current baseline achieves high scores by simply predicting "no change." Event-centric framing restricts evaluation to meaningful state transitions, producing metrics that genuinely reflect predictive ability.

### 4.3 Why Primary Metrics Are Divided Into When / Whether / What

A single "Change-mAP" metric is ambiguous — reviewers cannot distinguish event detection, TTC ordering, and post-change state prediction at a glance. The three-way split makes each metric's evaluation target self-evident: A1 = when will it change, A2 = will it change within the horizon, A3 = what will it change to.

### 4.4 Why Phase Is Treated as a Slow Contextual Variable

Phase transitions occur at ~0.17/min (~6 per procedure), far too sparse for robust TTC evaluation. Phase serves as workflow context that informs action predictions rather than as a primary anticipation target. Phase-level TTC is reported as an appendix analysis for completeness.

### 4.5 Why an Observer World Model, Not a Full Interventional World Model

Module C implements an **observer world model**: it conditions future-state predictions on observed/predicted action tokens, models the transition dynamics in latent space, and produces calibrated uncertainty estimates that drive hazard-based event timing. However, it does NOT select actions, does NOT perform counterfactual reasoning, and does NOT generate video frames.

**Classical world model structure:** `z_t = Enc(o_t)`, `ẑ_{t+1} = Dyn(z_t, a_t)`, `r̂ = Task(ẑ)`. SurgCast maps directly onto this:
- **Enc** = DINOv3 + CausalTransformer → h_t (latent surgical state)
- **Dyn** = EventDyn(h_t, a_t, a_t^+, τ) → predicted post-change state z^+ with uncertainty σ
- **Task** = Multi-task heads + DualHazardHead (decoding from h_t and z^+)

**Three key differences from classical world models (e.g., Dreamer, IRIS):**

1. **Observer, not controller.** Dreamer learns dynamics for action selection (policy optimization via imagined rollouts). SurgCast learns dynamics for anticipation (forecasting when the next change will occur and what will follow). There is no policy, no reward engineering, no actor-critic loop — but we inherit the core principle of "imagining the future in latent space" to improve temporal representations.

2. **Event-conditioned direct jump vs single-step iteration.** Classical world models predict `z_{t+3} = f(f(f(z_t)))`, accumulating error at each step. SurgCast jumps directly to the post-change state via event conditioning — the dynamics model predicts the state at the next change point, not at every intermediate timestep. This avoids error accumulation at the cost of not modeling intermediate states, but for anticipation the endpoint (what happens after the change) matters more than the path.

3. **Heteroscedastic uncertainty as imagination confidence.** σ from the dynamics model directly encodes "how reliable this imagination is." Low σ → the model is confident about the future state, meaning the current action is likely to continue and the change is far. High σ → the future state is uncertain, meaning the state is about to fork and a change is imminent. This uncertainty signal drives hazard estimation — a connection unusual in the world model literature, where σ typically flags OOD states rather than event proximity.

### 4.6 Why Structured Prior Is Optional Rather Than Core

The structured prior leverages surgical procedure constraints (e.g., specific actions are typical in specific phases) to regularize predictions for missing labels. However, its value depends on whether the statistical support from overlap videos (31 training videos for the richest prior) is sufficient. If ablations show marginal gains (<1.5pp Change-mAP), the prior is relegated to the appendix. The core contributions — the event-centric protocol, observer world model + hazard modeling, and multi-source training — do not depend on the prior.

### 4.7 Why Safety Is Auxiliary Rather Than Co-Primary

CVS "fully achieved" (total score ≥5) is extremely rare in the data (only 23 annotation rows across 16 videos). Elevating safety to co-primary would force a binary "safe vs. unsafe" evaluation that degenerates to a trivial judgment (nearly all pre-clipping frames are "unsafe"). Instead, safety evaluation is decomposed into complementary sub-tasks (clipping event anticipation and CVS state estimation) and treated as an auxiliary evaluation category.

### 4.8 Why HeiChole Is Used Only for External Transfer Validation

HeiChole provides 24 publicly labeled videos with phase, instrument, action, and skill annotations from the Heidelberg center. It does **not** provide CVS annotations. It serves exclusively as an external validation dataset (not used during training) to test whether representations learned from multi-source training generalize across centers, with evaluation limited to phase/instrument/action transfer using intersection-only ontology metrics.

### 4.9 Why Triplet-Group Is Secondary

Triplet-group change provides finer-grained evaluation than instrument-set change — it captures within-instrument action variations (e.g., grasper-retract vs. grasper-dissect) that are clinically meaningful but invisible to instrument-set change. However, it depends on a clustering definition, making it vulnerable to concerns about artificial construction. Compositionality (the 47.4% multi-instance rate and long-tail triplet distribution motivating multi-hot BCE over CE) is the representation-level inductive bias; triplet-group is the benchmark-level secondary evaluation target. The main text does not develop a separate compositional prediction formulation.

### 4.10 Why Action Targets Are Set-Valued

47.4% of CholecT50 frames have more than one active triplet, and the triplet distribution is heavily long-tailed (top 3 triplets account for 55.67%). Actions are inherently sets, not single classes. Action-change anticipation is thus defined as set-to-set transition prediction, naturally accommodating concurrent actions.

### 4.11 The Timescale Issue

| Granularity | Changes per minute | Changes per video | Character |
|---|---|---|---|
| Phase transition | 0.17 | 5.8 | Too sparse for continuous prediction |
| Instrument-set change | 4.07 | 136.7 | Clean signal, primary target |
| Target-set change | 4.71 | 158.2 | Moderate frequency |
| Verb-set change | 5.36 | 180.3 | Higher frequency |
| Triplet-set change | 6.17 | 207.6 | Highest frequency, heavy flicker noise |

This timescale hierarchy motivates the three-layer benchmark: instrument-set change (primary, cluster-free), triplet-group change (secondary, finer-grained), and clipping event detection (safety benchmark).

### 4.12 Why State-Age Covariates for Hazard

The 16s context window limits temporal reach; instrument-set changes occur at ~15s intervals on average, so the window often captures only one change cycle. State-age covariates (elapsed time since last instrument change + stable run length) provide standard survival-analysis duration information beyond the window's reach (cf. DeepSurv). In discrete-time hazard models, duration-in-state is a standard covariate — omitting it forces the model to implicitly reconstruct elapsed time from the causal transformer's positional encoding, which is indirect and lossy.

### 4.13 Why Pairwise Ranking Loss

Discrete-time hazard NLL optimizes per-sample likelihood but does not directly optimize C-index, the primary ranking metric for survival analysis. L_rank closes this optimization-evaluation gap by explicitly encouraging correct TTC ordering between sample pairs within each batch. This is standard practice in the survival analysis literature (DeepSurv, DeepHit) and is the only loss term that directly targets the C-index metric reported in the primary evaluation (A1).

### 4.14 Why Phase-Gated Rather Than Monolithic Hazard

HeiChole phase-instrument co-occurrence analysis shows strongly phase-dependent hazard profiles: CalotTriangleDissection exhibits high instrument turnover with frequent tool exchanges, while ClippingCutting involves brief, stereotyped transitions with predictable instrument sequences. A monolithic hazard head must average across these heterogeneous profiles; phase-gated residual experts specialize per phase while sharing a common base. Soft routing (from h_t, not predicted phase logits) avoids hard phase boundary errors and circular dependency with the Phase Head.

### 4.15 Why Delta-State Rather Than Direct Post-Change Prediction

At instrument-set change points, typically only 1–2 instruments change while the remainder persist. Direct prediction must reconstruct the entire 6-d instrument set from scratch; delta-state predicts only the edit operations (add/remove), exploiting the sparsity of changes. This aligns with the event-centric framing — "what changes" rather than "what exists" — and provides interpretable output: the model explicitly identifies which instrument enters or exits. The delta-state formulation also simplifies the learning problem: predicting sparse edits (mostly zeros) is easier than predicting a full binary vector where most dimensions are copy-current.

---

## 5. Datasets and Supervision Structure

### 5.1 Dataset Hierarchy

Datasets are organized into three tiers: Core (defining primary task ground truth), Auxiliary (providing supplementary supervision), and External (cross-center generalization validation).

| Tier | Dataset | Supervision dimensions | Videos | Frames | Coverage |
|---|---|---|---|---|---|
| **Core** | CholecT50 | triplet (instrument-verb-target) + instrument + verb + target + phase | 50 | 100,863 | Full procedure |
| **Core** | Cholec80 | phase (dense) + tool presence (7-class binary) | 80 | 184,498 | Full procedure |
| **Auxiliary** | Cholec80-CVS | CVS three-criteria scores (0/1/2) | 80 | Covers Preparation + CalotTriangleDissection phases | Preparation → pre-ClippingCutting |
| **Auxiliary** | Endoscapes2023 | anatomy bbox + CVS scores | 201 | 58,813 | ROI window (dissection → first clip) |
| **External** | HeiChole | phase + instrument + action + skill | 24 (publicly labeled) | — | Full procedure (external validation only) |

**HeiChole:** The Heidelberg center's cholecystectomy video dataset. Data audit confirms 24 publicly available labeled videos, with phase, instrument, action, and skill annotations at mixed 25/50 fps. **HeiChole does not provide CVS annotations.** It is used only for phase/instrument/action transfer validation (not used during training). HeiChole's phase/instrument ontology must be mapped to a shared coarse category space, with intersection-only metrics reported. Processing must handle 25/50 fps mixed frame rates and extra-abdominal white frames.

**CholecT50 label dimensions:** Each CholecT50 frame provides 5 independent annotation dimensions: triplet (100-class multi-label), instrument (6-class multi-label), verb (10-class multi-label), target (15-class multi-label), and phase (7-class single-label). This work leverages all five dimensions.

**Cholec80-CVS:** An annotation layer added by Ríos et al. (Scientific Data 2023) to all 80 Cholec80 videos. Annotations are provided as temporal intervals (start/end frames + 0/1/2 scores for each of three criteria), covering the Preparation and Calot's Triangle Dissection phases. It is not an independent video dataset but an annotation extension of Cholec80.

**Cholec80-CVS preprocessing policy:** We do not use the official preprocessing pipeline (`annotations_2_labels.py`), which discards the first 85% of the pre-clip/cut window and samples at 5fps — this truncation serves "CVS recognition at the last moment before clipping" and is unsuitable for anticipation. We need to observe the full temporal evolution of CVS from non-achievement to achievement, so we generate 1fps frame-level labels covering the entire pre-clip/cut phase directly from the raw XLSX. Similarly, we do not adopt the Cholec80-CVS official 50/15/15 split (which has extensive cross-leakage with the CAMMA combined split); CVS annotation splits are determined entirely by canonical video IDs from the CAMMA combined strategy.

### 5.2 Coverage Structure

The four datasets have complex video overlap relationships. After data audit verification, the following 7 mutually exclusive coverage groups constitute the complete set of 277 videos:

| Group | Label configuration | Videos | Count | Share |
|---|---|---|---:|---:|
| **G1 Triple intersection** | triplet+inst+verb+phase+CVS(C80)+CVS(Endo)+bbox | VID66,68,70 | 3 | 1.1% |
| **G2 CholecT50∩Cholec80** | triplet+inst+verb+phase+CVS(C80)+tool-presence | 45 overlap minus G1's 3 | 42 | 15.2% |
| **G3 CholecT50∩Endoscapes** | triplet+inst+verb+phase+CVS(Endo)+bbox | VID96,103,110 | 3 | 1.1% |
| **G4 Cholec80∩Endoscapes** | phase+tool-presence+CVS(C80)+CVS(Endo)+bbox | VID67,71,72 | 3 | 1.1% |
| **G5 CholecT50 only** | triplet+inst+verb+phase | VID92,111 | 2 | 0.7% |
| **G6 Cholec80 only** | phase+tool-presence+CVS(C80) | 80−45−3=32 | 32 | 11.6% |
| **G7 Endoscapes only** | CVS(Endo)+bbox | 201−9=192 | 192 | 69.3% |
| **Total** | | | **277** | **100%** |

**CVS coverage:** Groups with CVS annotations: G1(3) + G2(42) + G3(3) + G4(3) + G6(32) + G7(192) = **275/277 = 99.3%**. Only G5's 2 videos lack CVS.

**Instrument supervision coverage:** CholecT50 instrument labels cover G1(3) + G2(42) + G3(3) + G5(2) = 50 videos. Cholec80 tool-presence (6-tool mapping: Grasper→grasper, Bipolar→bipolar, Hook→hook, Scissors→scissors, Clipper→clipper, Irrigator→irrigator; SpecimenBag dropped) additionally covers G4(3) + G6(32) = 35 videos. **Total instrument training videos: 85** (70% increase over CholecT50 alone).

Notes:
- Final coverage levels are determined by actual label availability in `registry.json`
- Cholec80 tool-presence consists of 1fps 7-tool binary annotations (184,498 rows), of which 6 tools map directly to CholecT50 instruments

### 5.3 Video ID Registry and Leakage Control

**This is not an engineering detail — it is the prerequisite for scientifically credible multi-source results.**

A canonical `registry.json` is constructed with one record per physical video:

```json
{
  "canonical_id": "VID01",
  "in_cholec80": true,
  "in_cholect50": true,
  "in_endoscapes": false,
  "has_cholec80_cvs": true,
  "cholec80_tool_presence": true,
  "endoscapes_public_id": null,
  "labels_available": ["triplet", "instrument", "verb", "target", "phase", "cvs_cholec80", "tool_presence"],
  "coverage_group": "G2",
  "split": "train",
  "frame_counts": {"cholec80": 1733, "cholect50": 1733}
}
```

**Split assignment principles:**

- CAMMA recommended combined split strategy (Walimbe et al., MICCAI 2025)
- All overlaps resolved at the physical video ID level, not the dataset ID level
- Endoscapes test split has zero overlap with Cholec80/CholecT50 (confirmed by data audit)
- Cholec80 removes videos overlapping with CholecT50 test after merging
- Cholec80-CVS inherits split assignments from Cholec80 automatically

**Final split (CAMMA combined split strategy):** 168 train / 48 val / 61 test

### 5.4 Reserved Datasets (Journal Extension)

CholecTrack20, CholecInstanceSeg, CholecSeg8k, AutoLaparo, GraSP, and PhaKIR are not included in the NeurIPS training label pool. Reserved for:

| Dataset | Journal extension purpose |
|---|---|
| CholecTrack20 | Tool persistence memory (instrument presence duration as change prediction signal) |
| CholecInstanceSeg | Instance-level spatial reasoning |
| CholecSeg8k | Scene parsing auxiliary supervision |
| AutoLaparo | Cross-procedure transfer (hysterectomy → cholecystectomy) |
| GraSP | Cross-platform pretraining (robotic → laparoscopic) |
| PhaKIR | Multi-center generalization validation |

### 5.5 Task Boundary Declaration

This project is explicitly positioned as **event-centric forecasting with an observer world model**, not end-to-end autonomous surgical navigation, nor a full interventional world model.

The data audit explicitly supports:
- Phase-aware next-action-change prediction
- Anatomy-aware state estimation
- Action-change time forecasting
- Observer-world-model forecasting conditioned on observed/predicted action tokens
- Tool-memory modeling

**Not supported:** Interventional world-model claims (counterfactual reasoning, policy optimization, action selection), complete policy learning, path planning, or autonomous control. The current data does not provide simulator-level interventional ground truth, making causal validation infeasible. The observer world model conditions on actions as inputs (observed or predicted), not as outputs (selected by a policy). This boundary is explicitly declared in the paper — **the precise, bounded claim (observer dynamics) is validated through ablation and is more defensible than a full world-model claim.**

---

## 6. Method: SurgCast

### 6.1 Overall Architecture

```
Input frame sequence (t-15, ..., t), 1fps
        │
        ▼
[Module A] Frozen DINOv3 ViT-B/16 ──→ 768-d frame features (offline extraction, stored in HDF5)
        │
        ▼
    Linear(768→512) + LayerNorm + Positional Encoding
        │
        ▼
[Module B] Causal Temporal Transformer (6 layers, 8 heads, dim=512)
        │
        ├──→ h_t (current latent surgical state, 512-d)
        │
        ├──→ [Action Token Encoder]
        │         ├──→ Coarse: instrument (6-d) + phase (7-d) → Linear(13,64) → 64-d   (85 videos)
        │         ├──→ Fine: triplet-set via Set Transformer → 64-d                      (50 videos)
        │         └──→ a_t = Fuse(coarse, fine, mask) → 64-d
        │
        ├──→ [Module C] Observer World Model (Event-Conditioned Transition)
        │         │
        │         ├──→ Output 1: p(τ | h_t, a_t, d_t) — Dual Hazard Heads [When]
        │         │         ├──→ shared trunk: Linear(h+a+d, 256) → GELU
        │         │         ├──→ λ_inst(k): instrument-set change hazard (K=20)
        │         │         └──→ λ_group(k): triplet-group change hazard (K=20)
        │         │
        │         ├──→ Output 2: p(a⁺ | h_t, a_t) — NextActionHead [What-next]
        │         │         └──→ MLP([h_t; a_t], 256) → inst + phase + group predictions
        │         │
        │         ├──→ Output 3: p(z⁺ | h_t, a_t, a⁺, τ) — EventDyn [Future state]
        │         │         └──→ FiLM-conditioned MLP → μ⁺ (512-d) + log σ² (512-d)
        │         │
        │         └──→ Multi-task heads: decode from h_t (current) and μ⁺ (post-change)
        │                   ├──→ Triplet-Group Head ──→ ŷ_triplet-group     (BCE, multi-hot)
        │                   ├──→ Instrument Head ──→ ŷ_instrument            (BCE)
        │                   ├──→ Phase Head ──→ ŷ_phase                      (CE)
        │                   ├──→ Safety/CVS Head ──→ ŷ_cvs                   (Ordinal BCE, source-calibrated)
        │                   └──→ Anatomy-Presence Head ──→ ŷ_anatomy          (BCE, 5-class)
        │
        └──→ [Module D] Structured Prior Regularization (optional regularizer)
                  │
                  ├──→ Evidence-gating: KL weight adaptively scaled by cell count / posterior entropy
                  ├──→ Task-specific prior: Categorical for phase, factorized Bernoulli for multi-label
                  └──→ L_prior = w_evidence · Σ_task L_prior^task
```

### 6.2 Module A: Visual Feature Extraction (Offline)

**Choice: DINOv3 ViT-B/16 (main experiment)**

| Item | Detail |
|---|---|
| Model | DINOv3 ViT-B/16, frozen, 86M params (`facebook/dinov3-vitb16-pretrain-lvd1689m`) |
| Input | Raw frames resized to 518×518 |
| Output | 768-d CLS token per frame |
| Storage | One HDF5 file per dataset, indexed by video_id |
| Total frames | ~253,000 (Cholec80 184K + CholecT50 incremental ~10K + Endoscapes ~59K) |
| Extraction time | ~1 hour (2×A100) |
| Storage size | ~777 MB |

**DINOv3 vs DINOv2 rationale:**
- DINOv3 distills from a 7B teacher (vs DINOv2 ~1B teacher), producing stronger representations at the same parameter count
- Introduces Gram Anchoring to resolve DINOv2's known dense feature degradation during long training
- Trained on 1.7B images (vs DINOv2 142M), with RoPE positional encoding for flexible resolution
- ViT-B/16 outputs 768-d, same as DINOv2 ViT-B/14 — no downstream architecture changes needed

**Why ViT-B rather than ViT-L:**
- Under medical imaging domain gap, ViT-L shows minimal marginal returns (literature reports only +0.7% Dice on MRI segmentation)
- ViT-L's extra capacity primarily encodes natural image fine-grained features, with limited benefit for surgical endoscopic scenes
- With only ~253K downstream frames, the information bottleneck is in downstream data, not feature dimensionality
- Using ViT-B as the main method highlights that the method contribution (observer world model, hazard modeling) does not depend on the strongest backbone

**Ablation comparisons (A10):** LemonFM (surgery-domain-specific), DINOv2 ViT-B/14, DINOv3 ViT-L/16, ImageNet ResNet-50.

**Backbone decision threshold:** If LemonFM exceeds DINOv3-B by ≥1.5 points on validation Group-Change-mAP or by ≥2.0 points on B2a CVS AUC, LemonFM becomes the default backbone for the final submission; otherwise DINOv3-B remains the default. This decision is made at the Week 4 Go/No-Go checkpoint (see Section 12).

**Why frozen:**
1. Isolates the scientific question — all contributions are concentrated on temporal prediction and missing-label handling
2. Saves computation — features need only be extracted once for all experiments
3. Stabilizes optimization — frozen backbone avoids gradient conflicts from heterogeneous data sources during joint training

### 6.3 Module B: Causal Temporal Transformer

| Parameter | Value | Rationale |
|---|---|---|
| Layers | 6 | Sufficient modeling capacity for 16-step sequences |
| Hidden dim | 512 | Projected from 768-d, parameter-efficient |
| Attention heads | 8 | Standard configuration |
| Sequence length T | 16 frames (16 seconds) | Covers typical action durations |
| Attention mask | Causal (lower triangular) | No future information leakage |
| Positional encoding | Learnable | Handles uniform 1fps sampling |
| Dropout | 0.1 | Standard regularization |

**Input processing:** 768-d feature → Linear(768, 512) → LayerNorm → + positional encoding

**Output:** Latent surgical state h_t ∈ ℝ^512 at each timestep

**Conditional extension — long-context summary token:** If state-age covariates (Section 6.4.2) still leave >2s TTC MAE on long-TTC samples (TTC > 16s) after initial training, add a 17th learnable summary token prepended to the 16-frame sequence. This token attends to all 16 frames via causal masking (the summary token is at position 0, attending only to itself; all other tokens additionally attend to it) and provides a compressed long-context signal to the hazard head. This is a lightweight fallback (1 extra token, ~512 params) that supplements the 16s window with a learned global bias. **Implementation decision deferred to Week 7 based on TTC error analysis.**

### 6.4 Module C: Observer World Model — Action-Conditioned Event Dynamics

This is the core modeling component. Module C implements an observer world model that jointly produces three outputs: (1) **When** — time to next change via dual hazard heads, (2) **What-next** — post-change action prediction, and (3) **Future state** — event-conditioned latent state transition with heteroscedastic uncertainty. The dynamics model conditions on observed/predicted action tokens (teacher-forced during training, predicted at inference), connecting "what is happening now" (action tokens) with "what will happen next" (event forecasting) through latent imagination.

#### 6.4.1 Action Token Encoder

Action tokens encode the current surgical action state, providing the dynamics model with explicit action conditioning.

**Coarse action token** (high coverage, 85 videos):

- Input: instrument presence (6-d binary) + phase (7-d one-hot) = 13-d
- Projection: Linear(13, 64)
- Coverage: G1–G6 (85 videos with instrument + phase labels)

**Fine action token** (high semantics, 50 videos):

- Input: active triplet set (variable size, each triplet → 64-d learned embedding)
- Encoder: Set Transformer (2-head, 64-d, 1 ISAB + PMA) → fixed 64-d representation
- Handles 47.4% multi-instance frames naturally (set-valued input)
- Coverage: G1–G3, G5 (50 CholecT50 videos only)

**Mask-aware fusion:**

```
a_t^coarse = Linear(13, 64)(I_t ⊕ p_t)
a_t^fine   = SetTransformer({emb(r) : r ∈ R_t})  # only when triplet labels available
a_t        = MLP([a_t^coarse; a_t^fine]) if fine available, else MLP(a_t^coarse)
```

- Output: a_t ∈ R^64
- When neither coarse nor fine tokens are available (G7, 192 Endoscapes-only videos): a_t is a learned default embedding; dynamics losses (L_dyn, L_next) are masked

**Teacher forcing + scheduled sampling:**

- Training: GT action labels with probability ρ, else sg(predicted action logits from task heads)
- ρ annealed from 0.9 → 0.3 (cosine schedule over training epochs)
- Critical: at inference, all action tokens come from predictions (ρ=0); the annealing schedule ensures a smooth transition from teacher-forced training to fully predicted inference
- sg(·) = stop-gradient on predicted logits to prevent gradient flow from the dynamics loss back through the task heads (avoids destabilizing task head training)

#### 6.4.2 Event-Conditioned Transition — Main Method (Version B)

Three jointly modeled outputs from the observer world model:

**Output 1: When — Time to next change (Phase-Gated Dual Hazard Heads)**

```
p(τ | h_t, a_t, d_t, age_t) = PhaseGatedDualHazardHead(h_t, a_t, d_t, age_t)
```

**State-age covariates:** The hazard head is augmented with a state-age feature encoding temporal persistence of the current surgical state:

```
age_t = [age_inst, age_phase, stable_run_length]
age_embed = Linear(3, 16)(age_t)
```

- `age_inst`: seconds since last instrument-set change (clamped to [0, 30])
- `age_phase`: seconds since last phase transition (clamped to [0, 120])
- `stable_run_length`: consecutive frames with identical instrument multi-hot (clamped to [0, 30])
- **Data support:** The 16s context window limits temporal context; state-age provides survival-analysis-style duration covariates beyond the window. Data audit shows instrument-set changes occur at ~15s intervals on average — state-age explicitly encodes how long the current state has persisted, a standard covariate in discrete-time survival models (cf. DeepSurv)

**Hazard input:** `[h_t (512-d); a_t (64-d); d_t (2-d); age_embed (16-d)] = 594-d` (see 6.4.4 for d_t definition)

- Shared trunk: Linear(594, 256) → GELU
- Two event-specific heads with **phase-gated residual experts**

**Phase-gated hazard architecture:** Instead of a single Linear(256, K=20) per event type, the hazard head uses a base head + 7 phase-specific residual experts:

```
# Base hazard logits (always active)
z_base(k) = Linear_base(256, K=20)(trunk_out)   # raw logits

# Phase residual experts (7 phases × 2 event types = 14 experts)
r_p(k) = Linear_expert_p(256, K=20)(trunk_out)   # p ∈ {1,...,7}

# Soft phase routing
w_p(h_t) = softmax(Linear_route(512, 7)(h_t))_p

# Final hazard (per event type, e.g., instrument-set)
λ_inst(k | h_t) = σ(z_base(k) + Σ_p w_p · r_p(k))
λ_group(k | h_t) = σ(z_base_group(k) + Σ_p w_p · r_p^group(k))
```

- **Data support:** HeiChole phase-instrument co-occurrence analysis shows strongly phase-dependent hazard profiles (e.g., CalotTriangleDissection exhibits high instrument-change frequency while ClippingCutting is brief with stereotyped transitions). Phase-gated experts capture phase-specific hazard shapes without hard-coding phase boundaries
- **Low-frequency phase handling:** Phases with <100 training frames (e.g., TrocarPlacement) share the base head; their expert weights w_p are regularized via L2 toward zero during early training to prevent overfitting
- **Routing:** w_p is computed from the 512-d latent state h_t (not predicted phase logits), avoiding circular dependency with the Phase Head

**Why hazard modeling:**

| Approach | Problem |
|---|---|
| MSE regression | Sensitive to distribution skew, cannot handle right censoring |
| Binned classification | Loses ordinal information, sensitive to bin boundary choices |
| Ordinal regression | Better than classification, but lacks explicit survival structure |
| **Discrete-time hazard** | **Naturally handles right censoring, preserves ordinal structure, directly outputs survival function** |

**K=20 non-uniform interval definition:**

Future time is discretized into K=20 non-uniform intervals covering 30 seconds:

```
Intervals I_k: (0,1], (1,2], (2,3], (3,4], (4,5], (5,6], (6,7], (7,8], (8,9], (9,10],
               (10,12], (12,14], (14,16], (16,18], (18,20], (20,22], (22,24], (24,26], (26,28], (28,30]
```

The first 10 intervals are per-second (width 1s, fine resolution); the last 10 are per-2-seconds (width 2s, coarse resolution). Defined as left-open right-closed intervals, ensuring non-overlapping probability mass coverage.

**Why K=20 non-uniform bins rather than K=15 uniform:**
- Expected inter-change intervals are ~15–30 seconds (change density 2–4/min); K=15 covers only 15 seconds, causing heavy right censoring (changes beyond the window), weakening effective hazard training signal
- Non-uniform bins maintain high precision at the near end (clinically, "will there be a change within 5 seconds" matters far more than "28s vs 30s") while saving bins at the far end
- K=20 is more parameter-efficient than K=30 uniform bins while covering the same time range

**Design rationale:** Instrument-set change (~4.1/min) and triplet-group change (~2–4/min) have different event frequencies and semantics — within-instrument action changes (group change without instrument change) and instrument entry/exit are two events requiring independent prediction. The shared trunk ensures representation sharing; independent heads allow event-specific calibration.

**TTC target computation:** The input context window is 16 seconds, but TTC targets scan the full future of the original video (not limited to the 16s input window), with a prediction range of 30 seconds. For each anchor timestep t:
- Scan from t+1 in the original video for the next occurrence of instrument-set change and group-level change
- Map TTC values to the corresponding discrete interval I_k
- If no change occurs within min(30s, remaining_video_time), mark as right-censored

**Survival function:** Probability of no change in the first k intervals (separately for each event type):

```
S_inst(k | h_t) = Π_{j=1}^{k} (1 - λ_inst(j | h_t))
S_group(k | h_t) = Π_{j=1}^{k} (1 - λ_group(j | h_t))
```

**Cumulative incidence:** Probability of change within the first k intervals:

```
F_inst(k | h_t) = 1 - S_inst(k | h_t)
F_group(k | h_t) = 1 - S_group(k | h_t)
```

**Hazard training loss:** Standard discrete-time survival negative log-likelihood computed separately for instrument-set change and triplet-group change.

For a sample with observed change in interval k* (instrument-set change example):

```
L_hazard^inst = -log λ_inst(k* | h_t) - Σ_{j=1}^{k*-1} log(1 - λ_inst(j | h_t))
```

For right-censored samples (no change within observation window):

```
L_hazard^inst = -Σ_{j=1}^{K} log(1 - λ_inst(j | h_t))
```

L_hazard^group is computed symmetrically. Total hazard loss:

```
L_hazard = L_hazard^inst + η_group · L_hazard^group
```

where η_group is the relative weight for group hazard (default 1.0, swept over {0.5, 1.0, 2.0} in ablations). This loss naturally handles the "no change within window" case — it is modeled as a censored observation.

**Early warning decision boundary:** At inference:

```
Alert at time t if F_inst(k_warn | h_t) > τ_inst  or  F_group(k_warn | h_t) > τ_group
```

**TTC expected value with non-uniform intervals:** TTC expectation = Σ_k mid(I_k) · P(T ∈ I_k | h_t), where mid(I_k) is the midpoint of interval k:

```
Interval midpoints: (0,1]→0.5, (1,2]→1.5, ..., (9,10]→9.5, (10,12]→11, (12,14]→13, ..., (28,30]→29
```

TTC MAE = |E[T] - T_true|, where T_true is the actual TTC. Brier score is computed at interval right endpoints: Brier@k = (F(k|h_t) - 𝟙[T ≤ right(I_k)])². TTC metrics are computed separately for each event type (instrument-set / triplet-group).

**Output 2: What-next — Post-change action prediction (delta-state for instruments)**

```
p(a_t^+ | h_t, a_t) = NextActionHead(h_t, a_t)
```

- Architecture: MLP([h_t; a_t], 256) → GELU → three output branches
- **Instrument branch (delta-state prediction):** Instead of predicting the absolute post-change instrument set, the instrument branch predicts edit operations — which instruments are added and which are removed:
  ```
  delta_add = σ(Linear(256, 6))     # probability of each instrument appearing
  delta_remove = σ(Linear(256, 6))  # probability of each instrument disappearing
  y_inst_plus_pred = clamp((y_inst_current + delta_add) · (1 - delta_remove), 0, 1)
  ```
  - GT targets: `gt_add = max(0, y_inst_plus - y_inst_current)`, `gt_remove = max(0, y_inst_current - y_inst_plus)`
  - **Rationale:** At change points, most instruments remain unchanged; only 1–2 instruments typically enter or exit. Edit-based prediction aligns with the event-centric framing — "what changes" rather than "what exists after change." This also provides interpretable output: the model explicitly predicts which instrument enters/exits
  - Loss: `L_next_inst = BCE(y_inst_plus_pred, y_inst_plus_gt) + α_delta · BCE(delta_add, gt_add) + β_delta · BCE(delta_remove, gt_remove)` (α_delta = β_delta = 0.5)
- **Phase branch:** Linear(256, 7) → CE (direct prediction, unchanged)
- **Triplet-group branch:** Linear(256, G) → BCE (direct prediction, unchanged — groups are too abstract for delta-state)
- Supervised with GT post-change action labels when available (labels at the actual change point)
- Teacher forced during training (same ρ schedule as action token encoder)
- Loss: L_next = m_inst · L_next_inst + m_pha · CE(ŷ_phase^+, y_phase^+) + m_tri · BCE(ŷ_group^+, y_group^+)

**Output 3: Future state — Event-conditioned state transition**

```
μ^+, log σ² = EventDyn(h_t, a_t, a_t^+, τ_embed)
```

- τ_embed: learnable 64-d embedding (K=20 embeddings, one per hazard interval; ~1.3K params)
- **Training (teacher forcing):** τ_embed = Embed(k*), where k* is the GT change interval index. Same scheduled sampling ρ as action tokens: with probability ρ, use GT interval; else use sg(argmax(λ(k))) from the hazard head output
- **Inference:** τ_embed = Embed(argmax(λ(k))) — selected from the hazard head's predicted most-likely interval (no GT available)
- Input: [h_t; a_t; a_t^+; τ_embed] → conditioned via FiLM
- **FiLM conditioning:** `μ^+ = h_t + γ ⊙ r + β`, where r = MLP_r([a_t; a_t^+; τ_embed]), γ, β = CondNet(a_t, a_t^+, τ_embed)
- Output: μ^+ (512-d predicted post-change state) and log σ² (512-d per-dimension uncertainty)
- **Posterior target:** z^{+*} = sg(h_{τ*}) — encoder output at the true change point, stop-gradient

**Training supervision (naturally available from existing pipeline):**

For each anchor timestep t, the next change point t* is detected by scanning forward in the original video until the instrument multi-hot vector (for instrument-set change) or triplet-group multi-hot vector (for group change) differs from the value at time t. The **post-change frame** is defined as the first frame at which the label vector differs: t* = min{t' > t : label(t') ≠ label(t)}.

- τ* = t* - t (actual TTC in seconds; mapped to hazard interval k* for τ_embed teacher forcing)
- a_t^{+*} = ActionToken(labels at t*) — post-change action token constructed from GT labels at the change point frame
- z_t^{+*} = sg(h_{t*}) — stop-gradient encoder output at the change point frame (the CausalTransformer processes the full video, so h_{t*} is available within the same forward pass for sequences containing t*)

**Right-censored case:** If no change occurs within min(30s, remaining_video_time), the anchor is right-censored for hazard loss. L_dyn and L_next are masked for this anchor (no post-change target exists). The hazard loss still receives the right-censored supervision signal: L_hazard = -Σ_{j=1}^{K} log(1 - λ(j)).

#### 6.4.3 Heteroscedastic NLL Loss

The dynamics loss replaces MSE alignment with heteroscedastic negative log-likelihood:

```
L_dyn = 0.5 * exp(-log σ²) * ‖z_t^{+*} - μ^+‖² + 0.5 * log σ²
```

- MSE treats all dimensions equally; heteroscedastic NLL lets the model express per-dimension confidence
- The log σ² regularization term prevents trivially large σ (the model cannot simply predict infinite uncertainty to reduce the reconstruction penalty)
- σ is now directly trained as a probabilistic quantity with a proper likelihood interpretation, not indirectly inferred from prediction residuals
- Dimensions the model is uncertain about contribute less to the loss (weighted by exp(-log σ²)), while their uncertainty is penalized by the log σ² term — the model must trade off reconstruction accuracy against uncertainty magnitude

#### 6.4.4 Change Magnitude Signal for Hazard

The hazard head input is augmented with a predicted change magnitude signal d_t:

```
d_t = [KL(p_inst^+ ‖ p_inst), KL(p_group^+ ‖ p_group)]
```

- p_inst, p_group: current-state predictions from task heads (instrument and triplet-group logits at time t)
- p_inst^+, p_group^+: post-change predictions from NextActionHead (Output 2)
- d_t captures "how much the predicted future differs from the current state" — a 2-d vector
- **d_dim = 2** throughout; this is the minimal version. An optional per-horizon version (d_dim = 8) is tested in ablation A_new
- Hazard sees both "imagination blurriness" (σ from dynamics) and "predicted semantic change magnitude" (d_t) — when d_t is large, the model predicts a significant action change, informing the hazard that a transition is likely imminent
- d_t is computed with stop-gradient on both p and p^+ to prevent hazard gradients from flowing back through the task heads

#### 6.4.5 Version A: Fixed-Horizon Action-Conditioned Dynamics (Ablation)

Minimal upgrade from the previous action-free Module C, kept as an ablation comparison (A_dynamics):

```
μ_Δ, log σ²_Δ = Dyn(h_t, a_t, e_Δ),  Δ ∈ {1, 3, 5, 10}
```

- Input: [h_t; a_t; e_Δ] (512 + 64 + 64 = 640-d)
- `e_Δ` = learnable 64-d horizon embedding (one per Δ, 4 total)
- Shared MLP: Linear(640, 512) → GELU → Linear(512, 1024), output split into μ_Δ (512-d) and log σ²_Δ (512-d)
- Same action token encoder and teacher forcing as Version B
- Same heteroscedastic NLL loss (replacing MSE): L_dyn = Σ_Δ [0.5 · exp(-log σ²_Δ) · ‖sg(h_{t+Δ}) - μ_Δ‖² + 0.5 · log σ²_Δ]
- Retains the multi-horizon design from the previous Module C, but adds action conditioning
- σ_agg = [√(mean(exp(log σ²_{t+1}))), ..., √(mean(exp(log σ²_{t+10})))] ∈ R⁴ — same aggregation as before
- **Ablation A_dynamics compares Version A (fixed-horizon) vs Version B (event-conditioned) to isolate the value of event-conditioned dynamics**

#### 6.4.6 Multi-Task Head Decoding

Five prediction heads decode from both h_t (current state) and μ^+ (predicted post-change state):

| Head | Structure | Output | Loss | Notes |
|---|---|---|---|---|
| **Triplet-Group Head** | Linear(512, G), G≈15–20, per-group sigmoid | **G-dim multi-hot** | **BCE** | 47.4% of frames have >1 active triplet; multi_hot_target[g]=1 when frame has an active triplet belonging to group g |
| Instrument Head | Linear(512, 6) | 6-dim multi-label | BCE | Predicts instrument presence; directly serves safety alerts |
| Phase Head | Linear(512, 7) | Single-label probability | CE | 7-phase classification |
| Safety/CVS Head | Linear(519, 6), source-calibrated | 6-dim ordinal CVS prediction | Ordinal BCE | Input is 512-d + 2-d source embedding + 5-d sg(anat_probs); per criterion 2 logits: P(≥1), P(≥2); source-specific calibration + anatomy injection |
| **Anatomy-Presence Head** | Linear(512, 5), per-class sigmoid | **5-dim multi-label** | **BCE** | gallbladder/cystic duct/cystic artery/cystic plate/hepatocystic triangle presence; only G1/G3/G4/G7 have bbox → converted to anatomy presence labels |

**Dual decoding:**
- **Current-state decoding:** heads(h_t) — same as before, produces current-state predictions for all task losses
- **Post-change decoding:** heads(μ^+) — evaluated at the first change point for "What" metrics (Post-change Inst mAP, Post-change Group mAP)
- **Post-change loss:** same task losses but evaluated on post-change labels at the true change point; contributes to L_task when change point supervision is available

**CVS Head — Ordinal BCE with Anatomy Injection:**
- CVS Head uses `Linear(519, 6)` (including 2-d source embedding + 5-d stop-gradient anatomy probs), outputting 2 logits per criterion corresponding to cumulative probabilities P(score ≥ 1) and P(score ≥ 2)
- **Anatomy injection:** CVS Head input is extended to `[h_t (512-d); s_src (2-d); sg(anat_probs) (5-d)] = 519-d`, where `anat_probs = σ(AnatomyHead(h_t))` are the predicted anatomy-presence probabilities with stop-gradient to prevent CVS loss from distorting anatomy predictions
  - **Rationale:** CVS criterion achievement is directly conditioned on anatomy visibility — "two structures" (C1) requires cystic duct + cystic artery visibility; "hepatocystic triangle" (C2) requires triangle exposure. Injecting anatomy probs as input provides the CVS Head with explicit anatomical context
  - **Consistency regularizer:** `L_consist = Σ_t max(0, CVS_C1_prob(t) - min(cystic_duct_prob(t), cystic_artery_prob(t)))`, penalizing the model if it predicts C1 (two structures identified) > 0 while both structures have low predicted visibility. Active only on frames with both CVS and anatomy supervision
- Ordinal BCE loss: `L_cvs = Σ_c Σ_{k∈{1,2}} BCE(σ(logit_{c,k}), 𝟙[score_c ≥ k])`
- Inference reconstruction: predicted_score_c = σ(logit_{c,1}) + σ(logit_{c,2}), range [0, 2]
- Endoscapes adaptation: continuous CVS scores activate only the first threshold (≥0.5→1); second threshold loss is masked
- Parameter change: ~55K → ~60K (negligible)

**CVS Head source-specific calibration:**
- Cholec80-CVS (0/1/2 surgeon score) and Endoscapes (continuous CVS score ≥0.5 binarized) have different annotation protocols, with only 6 dual-annotated videos for consistency verification
- Source embedding s_src is a 2-d one-hot (Cholec80-CVS=[1,0], Endoscapes=[0,1]). This is equivalent to per-source bias and temperature correction
- At inference: for videos without CVS labels (G5), use Cholec80-CVS source embedding (G5 belongs to CholecT50, semantically closer)
- Parameter increment: 2×6 + 6 = 18 extra parameters (negligible)

**Triplet-Group Head — BCE (not CE):**
- 47.4% of CholecT50 frames have >1 active triplet; multiple groups can be simultaneously active. CE (softmax) assumes mutual exclusivity, contradicting the multi-label nature of the data
- Output uses per-group sigmoid (not softmax); loss is `BCE(sigmoid(logits), multi_hot_target)`
- Action-change anticipation is defined as **set-to-set transition prediction** — predicting the transition from the current active action set to the next. This naturally accommodates the 47.4% of frames with concurrent actions
- Compositionality is the representation-level inductive bias (justifies multi-hot BCE over CE); triplet-group is the benchmark-level secondary evaluation target. The main text does not develop a separate compositional prediction formulation

**Instrument Head training data expansion:**
- CholecT50 instrument labels cover G1–G3 + G5 = 50 videos
- Cholec80 tool-presence (6-tool mapping) additionally covers G4 + G6 = 35 videos
- **Instrument Head training videos: 50 → 85** (70% increase)
- Tool mapping: Grasper→grasper, Bipolar→bipolar, Hook→hook, Scissors→scissors, Clipper→clipper, Irrigator→irrigator (SpecimenBag dropped — no corresponding CholecT50 category)

**Instrument Head rationale:**
1. Instrument prediction directly serves safety-critical anticipation — predicting clipper appearance is more precise than indirect inference through triplet-group clustering
2. Instrument-set change is the primary TTC target (6 classes vs 15–20 groups, cleaner signal); triplet-group change is secondary
3. Minimal parameter increment (~3K parameters)
4. Cross-dataset instrument supervision transfer (Cholec80 tool-presence → CholecT50 instrument) is a methodological contribution

Note: Verb Head (10 classes) and Target Head (15 classes) are not added as independent prediction heads, because verb and target have strong combinatorial constraints with instrument. Independent BCE prediction would discard these relationships. Verb and target semantic information is implicitly leveraged through triplet-group semantic embedding clustering (see Section 7.3).

**Anatomy-Presence Head:**

**Motivation:** Endoscapes provides anatomy bounding box annotations for 201 videos. Converting bboxes to anatomy-presence auxiliary supervision introduces anatomy context into the safety branch.

**Implementation:**
- Extract frame-level presence labels for 5 anatomical structures from Endoscapes bbox annotations: gallbladder, cystic duct, cystic artery, cystic plate, hepatocystic triangle
- Presence definition: frame's bbox list contains a bbox of the corresponding class → 1, else → 0
- Head structure: Linear(512, 5), per-class sigmoid
- Sparse frame-level observation mask: supervision is available only on bbox-annotated frames; loss is activated via observation mask m_anat:
  ```
  L_anatomy = Σ_t Σ_c m_anat(t,c) · BCE(ŷ_anat(t,c), y_anat(t,c))
  ```
  where m_anat(t,c) = 1 only on bbox-annotated frames. This distinguishes presence=0 (structure absent — frame has bbox annotations but no bbox of this class) from unobserved (frame has no bbox annotations, excluded from loss)
- Applicable videos: G1(3) + G3(3) + G4(3) + G7(192) = 201 videos (but not every frame has bbox annotations)
- Parameter increment: ~2.5K (negligible)

**Narrative value:** Elevates Endoscapes utilization from "another CVS annotation set" to "CVS + anatomy context." Anatomical structure visibility directly relates to CVS achievement — if cystic duct and cystic artery are not visible, the two_structures criterion cannot be achieved. The Anatomy-Presence Head provides complementary anatomical signal for the CVS Head.

#### 6.4.7 Hazard Ablation Comparisons

| TTC modeling approach | Role |
|---|---|
| MSE regression | Baseline |
| Binned classification ([1-3s, 3-5s, 5-10s, 10s+]) | Baseline |
| Ordinal regression | Intermediate |
| **Discrete-time hazard (main method)** | — |

### 6.5 Module D: Structured Prior Regularization (Optional)

**This module is an optional regularization component, not a core method contribution.** If ablations show gains <1.5pp (Change-mAP), this module is relegated to the appendix.

**Interaction with phase-gated hazard (Section 6.4.2):** The phase-gated hazard head already captures phase-specific event profiles through learned residual experts. If A_phasegate ablation shows >3pp gain over the base-only hazard, Module D's phase-conditional prior provides diminishing marginal value and moves entirely to appendix. The two mechanisms address overlapping concerns (phase-dependent prediction shaping) through different means — Module D via explicit distributional regularization, phase-gated hazard via implicit routing. The Week 4 Go/No-Go checkpoint evaluates both simultaneously.

**Idea:** Surgical procedures have strong combinatorial constraints — specific actions are legal only during specific phases, and specific instruments appear only during specific phases. This structure provides a stronger inductive bias than "ignore" or "uniform smoothing" for missing labels.

#### 6.5.1 Three-Layer Design

**Layer 1: Static Procedure Prior (built offline, fixed during training)**

**All priors are strictly limited to training-set videos to prevent data leakage.**

| Prior | Data source | Training videos |
|---|---|---:|
| P_static(triplet-group \| phase) | CholecT50-train | 35 |
| P_static(triplet-group_{t+1} \| phase_t, triplet-group_t) | CholecT50-train | 35 |
| P_static(instrument \| phase) | CholecT50-train + Cholec80-adjusted-train (6-tool mapping) | ~67 |
| P_static(phase_{t+1} \| phase_t) | Cholec80-adjusted-train | 36 |
| P_static(cvs_ready \| phase, triplet-group) | CholecT50-train ∩ Cholec80-train | **31** |

Notes:
- P(triplet-group | phase) uses **per-group Bernoulli distribution** (consistent with BCE multi-hot), not softmax
- P(instrument | phase) extends to Cholec80-adjusted-train tool-presence, ~67 training videos total
- P(cvs_ready | phase, triplet-group) statistical basis: **31 training videos**
- Procedure priors learned from 31 training videos are sufficient to regularize predictions across 277 videos, reflecting the strong inductive bias of surgical procedure structure

Endoscapes mapping: G1's 3 + G4's 3 = 6 videos with dual CVS annotations serve as cross-annotation consistency validation samples.

Stored as lookup table: `static_prior.pkl`.

**Layer 2: Context-Modulated Prior (learnable, updated during training)**

On top of the static prior, the current latent state h_t provides context modulation. Task-specific distribution forms ensure mathematical consistency between prior distributions and prediction head output spaces:

**Phase (single-label categorical):**
```
q_prior^phase(y_miss | y_obs, h_t) = softmax(α · log P_static(y_miss | y_obs) + β · g_φ^phase(h_t))
L_prior^phase = KL(Cat(p_θ^phase) ‖ Cat(q_prior^phase))
```

**Multi-label tasks (triplet-group / instrument / anatomy / CVS-binary):**
```
q_prior^ml_c(y_miss | y_obs, h_t) = σ(α · logit(P_static_c(y_miss | y_obs)) + β · g_φ^ml(h_t)_c)
L_prior^ml = Σ_c KL(Bern(p_θ,c) ‖ Bern(q_prior,c))
```

where logit(p) = log(p/(1-p)) maps Bernoulli parameters to logit space (corresponding to sigmoid output), ensuring additive combination in logit space.

Shared components:
- P_static(y_miss | y_obs) is the Layer 1 lookup table output
- g_φ(h_t) is a lightweight MLP: Linear(512, 256) → GELU → Linear(256, C) (phase and multi-label tasks share the hidden layer; output layers are independent)
- α, β are learnable scalars (initialized α=1.0, β=0.1, so training starts dominated by the static prior); optional extension to a linear function of σ_agg (see Section 6.5.2.1)
- C is the number of classes in the masked dimension

**Why task-specific distributions:** Triplet-group/instrument/anatomy prediction heads use BCE (per-class sigmoid), with output space being independent Bernoulli, not mutually exclusive categorical. Applying a softmax prior to multi-label tasks would force mutual exclusivity constraints, contradicting the 47.4% of frames with multiple active groups.

**Intuition:** The static prior tells the model "during CalotTriangleDissection, grasper-retract-gallbladder is the most common action." Context modulation further tells the model "but given the current visual context, grasper-dissect-cystic-duct should have higher probability." For multi-label tasks, this modulation acts independently on each class's Bernoulli parameter.

**Layer 3: Ablation verification**

| Ablation config | Question addressed |
|---|---|
| Uniform prior (equivalent to label smoothing) | Is structure useful? |
| Static prior only (α=1, β=0) | Is context modulation useful? |
| Static + context prior (complete) | Main method |
| Self-distillation prior (teacher = EMA of model) | Domain structure vs model structure? |
| No prior (mask-and-ignore) | Is prior regularization useful at all? |

#### 6.5.2 Coverage Dropout During Training

For highest-coverage samples (with triplet + phase + CVS labels simultaneously), one known dimension is randomly masked with probability p_drop = 0.3, and the structured prior serves as soft supervision for the masked dimension.

**Procedure:**

1. Sample a highest-coverage sample, original mask m = [1, 1, 1, 1] (has triplet, instrument, phase, CVS)
2. With probability p_drop, choose to mask phase: m' = [1, 1, 0, 1]
3. For the masked phase dimension, compute q_prior(phase | triplet-group, h_t)
4. Model's phase prediction p_θ(phase | h_t) is aligned with q_prior via KL divergence

**Regularization loss (evidence-gated, task-specific):**

```
L_prior = w_evidence(y_obs) · Σ_task L_prior^task
```

where L_prior^task uses task-specific distributions: phase uses KL(Cat‖Cat), multi-label tasks use Σ_c KL(Bern‖Bern).

**Evidence-Gating mechanism:**

Prior statistics come from 31 training overlap videos. For the (phase × triplet-group × CVS) joint space, many cells have very few samples. Blindly applying equal-strength KL constraints to all cells risks misleading the model in low-support regions where the prior may be inaccurate.

**Evidence-gating rule:**
```
w_evidence(y_obs) = min(1.0, count(cell) / N_sufficient)
```

where `count(cell)` is the training-set frame count for the corresponding (phase, observed_labels) combination, and `N_sufficient` is the sufficient-sample threshold (default 50, swept over {20, 50, 100}).

**Hierarchical fallback strategy:**
- count(phase, group, cvs_level) ≥ N_sufficient → use full joint prior, w=1.0
- count < N_sufficient but count(phase, group) ≥ N_sufficient → fall back to marginal P(cvs | phase, group)
- count(phase, group) < N_sufficient → fall back to phase-level P(group | phase)
- count(phase) < N_sufficient (nearly impossible) → fall back to global marginal P(group)

**Why this is not label smoothing:** Label smoothing uses a uniform distribution, context-independent. Our prior is conditioned on observed labels and current latent state; the distribution shape varies per sample.

**Why this is not self-distillation:** Self-distillation's teacher signal comes from the model itself (EMA), potentially amplifying model bias. Our prior derives from the objective statistical structure of surgical procedures, independent of model quality.

#### 6.5.2.1 σ-Gated Prior Strength (C↔D Lightweight Connection) — Optional Extension

> **Priority: optional extension (lower than evidence-gating).** Evidence-gating (6.6.2) is implemented first. σ-gating is added only if evidence-gating is validated and time permits. Otherwise, α, β remain ordinary learnable scalars + evidence-gating.

**Motivation:** Module C's transition uncertainty σ_agg encodes whether the current state is near a change point. When transition is uncertain (near change point), the model is uncertain about the next state classification and should rely more on the structured prior; when transition is certain (stable action execution), context suffices for accurate prediction and prior weight can decrease.

**Modification:** Replace Layer 2's scalar α, β with linear functions of σ:

```
Phase:       q_prior^phase = softmax(α(σ_t) · log P_static + β(σ_t) · g_φ(h_t))
Multi-label: q_prior^ml_c  = σ(α(σ_t) · logit(P_static_c) + β(σ_t) · g_φ(h_t)_c)

α(σ_t) = α_0 + α_1 · σ̄_t
β(σ_t) = β_0 + β_1 · σ̄_t
```

where σ̄_t = mean(σ_agg) is the scalar mean of the 4-d σ_agg. α_0, α_1, β_0, β_1 are 4 learnable parameters (replacing the original 2).

**Initialization:** α_0 = 1.0, α_1 = 0.5, β_0 = 0.1, β_1 = -0.05. This causes α to increase under high uncertainty (prior weight rises) and β to decrease (context modulation weight drops).

#### 6.5.3 Ontology Bridge

**Phase alignment:** Cholec80 and CholecT50 phase names are similar but not identical (e.g., `GallbladderRetraction` vs `gallbladder-extraction`). A shared coarse phase space (7 phases) is used with dataset-specific mapping rules.

**Note:** `GallbladderRetraction` (Cholec80) and `gallbladder-extraction` (CholecT50) are approximate semantic mappings, not exact correspondences. Explicit validation on the 45 overlap videos is required during preprocessing — checking whether these two labels align on the timeline.

**CVS alignment (ordinal + source calibration):** The CVS Head uses ordinal regression + source-specific affine calibration, outputting P(≥1) and P(≥2) cumulative probabilities per criterion. Alignment scheme for the two data sources:
- **Cholec80-CVS:** Original 0/1/2 three-level scores map directly to two cumulative thresholds — score ≥ 1 activates the first threshold, score ≥ 2 activates the second. Both threshold losses participate in training. Source embedding = [1, 0].
- **Endoscapes CVS:** Continuous scores activate only the first threshold (C1/C2/C3 ≥ 0.5 → 1); second threshold loss is masked (Endoscapes lacks 0/1/2 level distinction). Source embedding = [0, 1].
- **Source calibration purpose:** The two data sources have different annotation protocols (surgeon 0/1/2 score vs binarized continuous score). Source embeddings let the model learn source-specific decision boundaries, mitigating annotation-style shift.
- **Consistency validation:** On G1(3) + G4(3) = 6 dual-CVS videos, require binarized (≥1) threshold frame-level agreement >80%.
- **Unified inference:** predicted_score_c = σ(logit_{c,1}) + σ(logit_{c,2}), range [0, 2]. Cholec80-CVS source embedding is used as default for new videos.

### 6.6 Total Loss Function

```
L_total = L_task + λ_dyn · L_dyn + λ_hazard · L_hazard + λ_next · L_next + λ_prior · L_prior + λ_rank · L_rank + λ_consist · L_consist
```

Component definitions:

```
L_task = m_tri · L_triplet + m_inst · L_instrument + m_pha · L_phase + m_cvs · L_cvs + m_anat · L_anatomy
    (label-conditional multi-task loss, m ∈ {0,1} is visibility mask)
    (L_triplet, L_instrument are BCE; L_cvs is ordinal BCE; L_phase is CE)
    (L_anatomy = Σ_t Σ_c m_anat(t,c) · BCE(ŷ_anat(t,c), y_anat(t,c)),
     where m_anat(t,c) is frame-level observation mask (=1 only on bbox-annotated frames),
     distinguishing presence=0 (structure absent) from unobserved (no annotation))

L_dyn = 0.5 · exp(-log σ²) · ‖z_t^{+*} - μ^+‖² + 0.5 · log σ²
    (Heteroscedastic NLL for event-conditioned state prediction (see 6.4.3);
     replaces L_align MSE; active for videos with change-point supervision)

L_hazard = L_hazard^inst + η_group · L_hazard^group
    (Dual-event discrete-time survival loss (see 6.4.2), K=20 non-uniform bins,
     separately modeling instrument-set change and triplet-group change;
     L_hazard^inst active for G1–G6 (85 videos with instrument labels);
     L_hazard^group active for G1–G3+G5 (50 videos with triplet labels);
     hazard head input is now action-conditioned: [h_t; a_t; d_t; age_embed])

L_next = m_inst · L_next_inst + m_pha · CE(ŷ_phase^+, y_phase^+) + m_tri · BCE(ŷ_group^+, y_group^+)
    (Next-action prediction loss at change point (see 6.4.2 Output 2);
     L_next_inst uses delta-state formulation: BCE(y_inst_plus_pred, y_inst_plus_gt)
       + α_delta · BCE(delta_add, gt_add) + β_delta · BCE(delta_remove, gt_remove);
     CE for phase, BCE for group; active when change-point labels available)

L_rank = (1/|P|) · Σ_{(i,j)∈P} log(1 + exp(-(E[T_i] - E[T_j])))
    (Pairwise ranking loss for C-index optimization, closing the optimization-evaluation gap;
     P = set of concordant pairs (i,j) where T_true_i < T_true_j within each batch;
     E[T] = Σ_k mid(I_k) · P(T ∈ I_k | h_t) is the predicted TTC expectation;
     computed separately for instrument-set and triplet-group changes, then summed;
     inspired by DeepSurv's Cox partial likelihood but adapted for discrete-time hazard;
     P includes: (a) uncensored pairs where T_true_i < T_true_j (both observed),
     and (b) mixed pairs where uncensored T_true_i < C_j (uncensored event before
     censored observation window). Pairs where both are censored or where
     censored C_j ≤ T_true_i are excluded)

L_consist = Σ_t max(0, CVS_C1_prob(t) - min(cystic_duct_prob(t), cystic_artery_prob(t)))
    (Anatomy→CVS consistency regularizer (see 6.4.6);
     penalizes CVS C1 prediction exceeding minimum anatomy visibility;
     active only on frames with both CVS and anatomy supervision)

L_prior = w_evidence · Σ_task L_prior^task
    (Evidence-gated structured prior regularization, computed only when coverage dropout is active)
    (Task-specific distributions:
     L_prior^phase = KL(Cat(p_θ^phase) ‖ Cat(q_prior^phase))  — categorical prior
     L_prior^ml = Σ_c KL(Bern(p_θ,c) ‖ Bern(q_prior,c))  — factorized Bernoulli prior for multi-label tasks
     Applicable to triplet-group/instrument/anatomy/CVS-binary)
```

**Hyperparameter initial values:** λ_dyn = 0.5 (replaces λ_align), λ_hazard = 1.0, λ_next = 0.3, λ_prior = 0.3, λ_rank = 0.1, λ_consist = 0.05, ρ_init = 0.9, ρ_final = 0.3 (teacher forcing annealing), α_delta = β_delta = 0.5

### 6.7 Model Size

| Component | Parameters | Notes |
|---|---|---|
| Input projection (768→512) | ~400K | |
| Causal Transformer (6 layers) | ~9.5M | |
| **Action Token Encoder** | **~60K** | Coarse projection Linear(13,64) + fine Set Transformer (~40K: 2-head, 64-d, 1 ISAB + PMA) + fusion MLP |
| **EventDyn MLP (FiLM-conditioned)** | **~950K** | Wider input than previous 820K due to action token + event conditioning |
| τ embedding (K=20) | ~1.3K | 20 × 64-d learnable embeddings for hazard-to-dynamics coupling |
| **NextActionHead (delta-state)** | **~30K** | MLP([h_t; a_t], 256) → Linear(256, phase+group) + delta_add Linear(256,6) + delta_remove Linear(256,6) |
| Prediction heads (triplet-group + instrument + phase + CVS ordinal + anatomy) | ~63K | +~2.5K anatomy head, +18 CVS source calibration, CVS input 519-d (anatomy injection) |
| **State-age projection** | **~1K** | Linear(3, 16) for age_embed |
| **Phase-gated hazard head** | **~230K** | Shared trunk Linear(594,256) + 2×base Linear(256,20) + 14 expert Linear(256,20) + routing Linear(512,7) |
| Prior modulation MLP | ~140K | |
| Evidence-gating lookup | ~0 | Precomputed lookup table, no learnable parameters |
| Context α, β (optional σ-gated: α_0, α_1, β_0, β_1) | 2 (optional: 4) | Optional extension as linear function of σ (see 6.5.2.1) |
| **Total** | **~11.6M trainable** | ~400K increase from action token encoder + delta-state head + phase-gated hazard + state-age |

Including the frozen DINOv3 86M parameters, total inference parameters are ~97.6M, but only ~11.6M require training. Training on 2×A100 40GB is straightforward.

---

## 7. Action Change Definition and Robustness

### 7.1 Why Change Definition Is Critical

TTC quality depends entirely on "what counts as a change." Data audit provides precise change frequency baselines:

| Granularity | Changes/min | Changes/video | Character |
|---|---|---|---|
| Phase transition | 0.17 | 5.8 | Too sparse for continuous prediction |
| Instrument-set change | 4.07 | 136.7 | Moderate frequency, clean signal |
| Target-set change | 4.71 | 158.2 | Moderate frequency |
| Verb-set change | 5.36 | 180.3 | Higher frequency |
| Triplet-set change | 6.17 | 207.6 | Highest frequency, heavy label flicker noise |

CholecT50 triplet annotations are per-instance at 1fps; adjacent frames may exhibit label flicker. Much of the ~6.2/min triplet-set change frequency is noise. After triplet-group clustering (compressing 100 classes to 15–20 groups), expected group-set change frequency drops to ~2–4/min (to be confirmed in preprocessing), yielding ~60–120 change points per video — sufficient to support TTC training.

### 7.2 Three Change Definitions

| Definition | Description | Expected change density | Use |
|---|---|---|---|
| Strict change | Any triplet dimension changes | Extremely high, full of flicker | Internal validation only |
| **Instrument-set change (Primary)** | **Instrument multi-label vector changes (new instrument appears or existing disappears)** | **~4.1/min** | **Primary TTC target, cluster-independent** |
| **Group-level change (Secondary)** | Multi-hot group vector Hamming distance > 0 (any group activation state flips) | Moderate (~2–4/min) | **Secondary semantic stress-test, finer-grained** |
| Debounced change | Group-level change persisting ≥3 seconds before confirmation | Lower | Ablation comparison, verifying debouncing impact |

**Three-layer benchmark design:**

1. **Instrument-set change:** Completely cluster-independent, semantically clear (appearance/disappearance of 6 instrument classes). Reviewers cannot challenge the target definition as artificial. **Sole primary metric.**
2. **Triplet-group change:** Finer-grained, better captures clinical semantics (action combination changes), but depends on clustering definition. **Secondary semantic stress-test.**
3. **Clipping event:** Clinically critical event, binary detection. Safety benchmark.

**Why not only instrument-set change:** Instrument-set change is relatively coarse — the same instrument can perform different actions (grasper-retract vs grasper-dissect), and these changes are meaningful for surgical progress but invisible to instrument-set change. Triplet-group change captures this additional granularity.

### 7.3 Triplet-Group Construction

**Method: co-occurrence statistics + semantic embedding hybrid.**

1. **Co-occurrence matrix (statistical signal):** Compute a 100×100 cross-video co-occurrence matrix (frequency of two triplets appearing in the same frame), normalized to similarity matrix S_cooc.

2. **Semantic embeddings (semantic signal):** Use sentence-transformers (all-MiniLM-L6-v2, local execution, fully reproducible) to generate embedding vectors for the 100 triplet names:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')  # executed once offline
triplet_names = [
    "grasper, retract, gallbladder",
    "clipper, clip, cystic-artery",
    ...  # 100 triplets
]
embeddings = model.encode(triplet_names)  # 100 × 384
S_semantic = cosine_similarity(embeddings)  # 100 × 100
```

3. **Hybrid similarity matrix:**

```python
S_mixed = α * S_cooc + (1 - α) * S_semantic  # α tuned on validation or default 0.5
```

4. Hierarchical clustering (Ward linkage) on S_mixed, selecting the cutoff height yielding 15–20 clusters.

5. **Validation:** Each group should correspond to interpretable surgical semantics.

**Benefits of semantic augmentation:**
- Semantically similar triplets are clustered together (even with low co-occurrence in data)
- Statistically co-occurring but semantically different triplets are separated
- Produces more clinically meaningful change definitions

**Ablation (A14):** Pure co-occurrence vs pure semantic vs hybrid clustering, evaluating Change-mAP and group interpretability.

### 7.4 Required Baseline Statistics

**Known (from data audit, directly cited in the paper):**
- Triplet-set changes: ~208 per video, ~6.2/min
- Instrument-set changes: ~137 per video, ~4.1/min
- Phase transitions: ~5.8 per video
- Triplet distribution heavily long-tailed: top 3 triplets = 55.67%
- Multi-instance frame ratio: 47.4%

**To be computed after clustering (Step 2):**
- Average group-set change points per video
- Average inter-change interval (seconds), distribution (median, IQR)
- Change point distribution across phases
- How many flicker events debounced change filters vs group-set change

**Group-set change density expected range:** Based on instrument-set change (~4.1/min) and triplet-set change (~6.2/min), group-set change frequency should fall in the 2–4/min range. **If measured below 1.5/min (i.e., <50 changes per video), reduce group count to 10–12 or relax the change definition.**

---

## 8. Training Strategy

### 8.1 Coverage-Aware Batching

Training design prioritizes the core anticipation task (instrument-set change TTC and post-change state prediction), not safety-only recognition. CVS and anatomy heads are auxiliary objectives that benefit from multi-source data but do not drive the training schedule.

Each batch (batch_size=64) samples according to 7-group proportions:

| Group | Videos | Active losses | Batch % | Samples/batch |
|---|---:|---|---:|---:|
| G1 Triple intersection | 3 | L_triplet + L_instrument + L_phase + L_cvs(C80) + L_cvs(Endo) + L_anatomy + L_hazard + L_prior | 5% | 3 |
| G2 CholecT50∩Cholec80 | 42 | L_triplet + L_instrument + L_phase + L_cvs(C80) + L_hazard + L_prior(30% prob) | **35%** | **22** |
| G3 CholecT50∩Endoscapes | 3 | L_triplet + L_instrument + L_phase + L_cvs(Endo) + L_anatomy + L_hazard | 5% | 3 |
| G4 Cholec80∩Endoscapes | 3 | L_phase + L_instrument(tool) + L_cvs(C80) + L_cvs(Endo) + L_anatomy + **L_hazard^inst** + **L_dyn(inst)** + **L_next(inst)** | 4% | 3 |
| G5 CholecT50 only | 2 | L_triplet + L_instrument + L_phase + L_hazard | 3% | 2 |
| G6 Cholec80 only | 32 | L_phase + L_instrument(tool) + L_cvs(C80) + **L_hazard^inst** + **L_dyn(inst)** + **L_next(inst)** | **23%** | **15** |
| G7 Endoscapes only | 192 | L_cvs(Endo) + L_anatomy | **25%** | **16** |

G7 batch weight is set to 25% (not higher) because G7 has only CVS+bbox labels — excessive G7 weight would pull the shared temporal trunk toward safety-state classification rather than change forecasting. Action-rich groups (G1–G6) account for 75% of batches, ensuring the trunk prioritizes learning change forecasting.

**Coverage highlights:**
- L_cvs activated: G1+G2+G3+G4+G6+G7 = **97%** (275/277 videos have CVS)
- L_instrument activated: G1+G2+G3+G4+G5+G6 = **60%** (85/277 videos have instrument labels)
- L_anatomy activated: G1+G3+G4+G7 = 201 videos (73%); frame-level observation mask m_anat(t,c) is 1 only on bbox-annotated frames
- L_hazard^inst activated (instrument labels → instrument-set change points): G1–G6 = **85 videos**. Instrument-set change is the primary TTC target; extending hazard training from 50 to 85 videos significantly strengthens the primary metric's training signal
- L_hazard^group activated (triplet labels → triplet-group change points): G1+G2+G3+G5 = 50 videos (requires triplet annotations to define group-level change)
- L_dyn and L_next active: **85 videos** (G1–G6) for instrument-set change dynamics; **50 videos** (G1–G3, G5) for triplet-group change dynamics. For G4/G6, the EventDyn target z^{+*} is the encoder output at the next instrument-set change point; a_t^+ is constructed from coarse labels (instrument + phase) at that frame. G7 (192 Endoscapes-only): no action tokens or instrument labels available, dynamics loss masked — these videos contribute only through L_cvs + L_anatomy

### 8.2 Class Imbalance Handling

**Triplet-Group Head (BCE):** Triplet distribution is heavily long-tailed (top 3 = 55.67%); group-level distribution may be similarly imbalanced. Training uses per-group `pos_weight` inversely proportional to positive frequency: `pos_weight_g = (N - N_g) / N_g`, where N_g is the training-set frame count where group g is positive. Ablation compares uniform BCE vs pos_weight BCE vs Focal loss (γ=2).

**Instrument Head (BCE):** Instrument distribution is extremely imbalanced — clipper appears ~2% of the time while grasper/hook appear >80%. Per-class `pos_weight` ensures the low-frequency but safety-critical clipper receives sufficient gradient signal.

**CVS Head (Ordinal BCE):** No additional weighting. CVS positive sparsity reflects clinical reality (CVS is genuinely not achieved during most of the dissection), and forced upsampling would distort the prior distribution. Ordinal regression already mitigates information loss from binarization by preserving 0→1→2 ordinal information.

**Phase Head (CE):** No additional weighting; phase distribution is relatively uniform.

### 8.3 Sequence Construction

Sliding windows of length 16 with stride 8 are extracted from each video. Each subsequence is a training sample.

**Dual TTC target computation:** For each timestep t, TTC targets are computed separately for the two change event types:

1. **Instrument-set TTC:** Scan from t+1 in the original video's global frames for the next instrument multi-label vector change (new instrument appears or existing disappears), map TTC to K=20 non-uniform intervals
2. **Group-level TTC:** Scan from t+1 in the original video's global frames for the next multi-hot group vector change (Hamming distance > 0), similarly map to K=20 non-uniform intervals
3. **Right censoring:** If no change of the corresponding type occurs within min(30s, remaining_video_time), mark as right-censored

**Anchor protocol:**
- **Training:** Every valid timestep t in each 16-frame chunk serves as an anchor for hazard loss computation. For dynamics loss (L_dyn), supervision requires a future change point within the video; L_dyn is masked when no change point exists. For Version A fixed-horizon dynamics, L_dyn^{Δ=10} is activated only when t ≤ T−10 (i.e., anchor position + 10 is within the observation window)
- **Inference:** Single current-timestep anchor (last frame t=T of the sequence), outputting hazard function and TTC expected value

### 8.4 Training Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4, cosine decay to 1e-6 |
| Weight decay | 0.05 |
| Warmup | 5 epochs |
| Total epochs | 100 |
| Sequence length | 16 frames (16 sec) |
| Batch size | 64 |
| λ_dyn | 0.5 (replaces λ_align) |
| λ_hazard | 1.0 |
| λ_next | 0.3 |
| λ_prior | 0.3 |
| ρ_init (teacher forcing) | 0.9 |
| ρ_final (teacher forcing) | 0.3 |
| Teacher forcing schedule | Cosine annealing over training epochs |
| Coverage dropout rate | 0.3 (highest-coverage samples only) |
| Hazard time bins K | 20 (non-uniform bins, covering 30 seconds) |
| GPU | 2 × A100 40GB |
| Estimated training time | ~6 hours / 100 epochs |

---

## 9. Evaluation Protocol

### 9.1 Why a Stricter Evaluation Protocol

We propose a stricter event-centric anticipation protocol that complements existing dense-step and state-change paradigms. Existing surgical anticipation works (SuPRA, SWAG, etc.) evaluate dense-step future prediction ("what phase/instrument at time t+Δ"), which is severely inflated by temporal inertia. Our event-centric protocol uses change points as evaluation anchors, eliminating inertia inflation. The protocol applies to all baselines (not a closed proprietary metric) and will be open-sourced as part of the methodological contribution.

**Change-conditioned evaluation:** For all dense-step metrics, performance is additionally reported stratified by change vs non-change frames. On non-change frames, the copy-current baseline naturally performs well; only change-frame performance reflects genuine predictive ability.
- **Change frames:** frames where instrument-set or triplet-group state changes at the next timestep
- **Non-change frames:** frames where state remains unchanged at the next timestep
- **Reporting:** each dense metric reports three columns — All / Change-only / Non-change-only
- **Expected pattern:** Copy-current baseline achieves ~100% on non-change frames, ~0% on change-only frames; a good predictive model should significantly outperform copy-current on change-only frames

**Relationship to existing evaluation systems:**
- **CholecT50 Triplet Recognition Challenge:** Evaluates current-frame triplet recognition. We report dense-step mAP as a secondary metric for compatibility but not as the primary metric.
- **Cholec80 Phase Recognition:** Evaluates current-frame phase classification. We report Phase Acc @Δ for compatibility, also secondary.
- **Primary metrics (Event-AP, TTC metrics, Post-change mAP) are newly defined** — no historical comparison data exists; baselines (Section 10.1) establish the reference frame.

### 9.2 Data Partitioning

Following the CAMMA recommended combined split strategy (Walimbe et al., MICCAI 2025), splits are unified at the physical video ID level, ensuring the same physical video belongs to the same split across all datasets.

| Coverage group | Total | Train | Val | Test | Notes |
|---|---:|---:|---:|---:|---|
| G1 (triple intersection) | 3 | 3 | 0 | 0 | Very few samples; all in train to maximize annotation utilization |
| G2 (CholecT50∩Cholec80) | 42 | 28 | 5 | 9 | Core evaluation group; test videos have all label dimensions |
| G3 (CholecT50∩Endoscapes) | 3 | 3 | 0 | 0 | VID96/103/110 all assigned to train by CAMMA |
| G4 (Cholec80∩Endoscapes) | 3 | 3 | 0 | 0 | Very few samples; all in train |
| G5 (CholecT50 only) | 2 | 1 | 0 | 1 | VID111 in test (no CVS; used for B1, not B2) |
| G6 (Cholec80 only) | 32 | 17 | 4 | 11 | Only phase + tool-presence + CVS(C80) |
| G7 (Endoscapes only) | 192 | 113 | 39 | 40 | Only CVS(Endo) + bbox |
| **Total** | **277** | **168** | **48** | **61** | |

The CAMMA combined split strategy produced a larger test set than estimated (61 vs 45), and all evaluation tiers meet or exceed original estimates. G3's 3 videos are all in train (CAMMA's CholecT50 official split assigns VID96/103/110 to training), which sacrifices one cross-coverage test sample but maximizes CholecT50+Endoscapes cross-dataset training.

**Number consistency principle (mandatory before paper writing):** All numbers appearing in the paper (video counts, frame counts, split distributions, prior statistics, coverage rates) must be **auto-generated** from `registry.json` / preprocessing logs / split files — never hand-written. The recommendation: implement `scripts/generate_paper_stats.py` to produce a JSON/LaTeX macro file from the registry and split, with all tables and text referencing this file to eliminate inconsistency from manual copying.

### 9.3 Stratified Evaluation by Test Set

**Core principle: each metric is computed only on test videos that have the corresponding ground truth.**

| Evaluation tier | Test video source | Count | Evaluable metrics | Notes |
|---|---|---:|---|---|
| **Tier 1: Core** | CholecT50-test (G2+G5) | 10 | Event-AP-inst/group @5s, Post-change Inst/Group mAP, TTC full suite (inst+group), C-index, Brier, Dense-mAP, Safety B1 | **Main table (Table 1) evaluation basis; metrics organized as When/Whether/What** |
| **Tier 2a: CVS Safety (Auxiliary)** | **Endoscapes-test (G7) + Cholec80-test (G6)** | **51** | **CVS criterion AUC, CVS MAE** | **Primary CVS evaluation set; auxiliary category** |
| **Tier 2b: CVS-at-Clipping (Appendix)** | **Tier 1 videos with CVS (G2)** | **9** | **CVS MAE@clip, Early Warning Quality** | **Moved to appendix (9 videos insufficient statistical basis for main text)** |
| **Tier 3: Phase** | Cholec80-test (G2+G6) | 20 | Phase Acc @Δ | Expanded evaluation base for coarse-grained generalization |
| **Tier 4: Instrument** | CholecT50-test + Cholec80-test (G2+G5+G6) | 21 | Instrument Anticipation mAP, **Event-AP-inst @5s** | Expanded instrument-set event detection evaluation |
| **Tier 5: Anatomy** | **Endoscapes-test (G7)** | **40** | **Anatomy-Presence mAP** | Validates bbox → anatomy presence learning; evaluated only on bbox-annotated frames |
| **Tier 6: HeiChole External** | **HeiChole (external)** | **24** | **Phase Acc @Δ, Instrument mAP, Action transfer metrics** | Cross-center zero-shot + optional few-shot; validates RQ3. Evaluates phase/instrument/action transfer only (no CVS). Uses intersection-only ontology metrics; explicitly handles 25/50 fps and extra-abdominal frames |

**Safety evaluation tiers:**
- **B1 (main text):** Clipping Event Anticipation on Tier 1 (10 videos)
- **B2a (main text):** CVS state accuracy on Endoscapes-test + Cholec80-test (51 videos) — statistically stable primary CVS evaluation
- **B2b (appendix):** CVS-at-clipping on 9 G2 test videos. 9 videos is insufficient statistical basis for main text; serves as appendix supplementary evidence

### 9.4 Justification for Core Metric Evaluation on 10 Videos

This is a fact that must be addressed directly, not avoided.

**Why this is not a deficiency:**

1. **Dictated by task definition:** Action-change anticipation at the triplet-group level requires triplet ground truth, which only CholecT50 provides. This is the annotation reality for this level of granularity, not an experimental design flaw.
2. **Consistent with existing literature:** The CholecT50 Triplet Recognition Challenge also evaluates on 10 test videos and is accepted by the MICCAI/TMI community.
3. **Training value shown via secondary metrics:** The remaining 51 test videos, while unable to evaluate core metrics, can demonstrate multi-source training transfer through Phase Acc (Tier 3) and CVS AUC (Tier 2a). If adding G6/G7 training data improves Phase Acc and CVS AUC, this indirectly demonstrates the data's contribution to overall model representation.

**How to strengthen persuasiveness:**

1. Report **per-video** Change-mAP and C-index-inst / C-index-group (not just means; show distributions)
2. Report mean ± std over 5 random seeds
3. For key comparisons (Full SurgCast vs CholecT50-Only), conduct **paired permutation test** (p < 0.05)
4. Appendix shows per-test-video hazard heatmaps + change point qualitative visualizations
5. Appendix supplements with CholecT50 official k-fold cross-val results (action branch only), proving conclusions are not split-dependent

### 9.5 Phase-Level Change Auxiliary Evaluation (Appendix)

Phase-level TTC is reported as an auxiliary appendix analysis rather than a main benchmark, since the primary contribution is fine-grained action-change forecasting at the instrument-set and triplet-group level. Phase transitions (~6/video) are too sparse for robust TTC evaluation, and the aggregation from triplet-level hazard to phase-level hazard is underspecified.

**Definition:** Phase change point = the moment when the phase label changes. A typical cholecystectomy has 5–6 phase changes.

**Evaluation metrics (parallel to Tier 1 TTC metrics):**

| Metric | Definition |
|---|---|
| Phase-TTC MAE | Phase change time-to-next-change mean absolute error |
| Phase-TTC C-index | Phase change concordance index (single event, no inst/group split needed) |
| Phase-TTC Brier @5s | Phase change probability calibration (single event) |

**Note:** Phase change is much coarser than triplet-group change (~6 phase changes vs ~50+ group changes per procedure), with stronger sequential ordering constraints. Phase-TTC should be significantly better than triplet-level TTC — if not, the model's coarse workflow understanding is problematic.

### 9.6 Cross-Dataset Transfer Evaluation

**This is the core validation dimension for the overlap-safe multi-source protocol (Contribution 3).**

Table 1's row-by-row ablation demonstrates this (CholecT50-Only → incrementally adding data sources → Full SurgCast):

- **All methods are evaluated on the same fixed Tier 1 test set (10 CholecT50-test videos).** Training data varies; the test set is completely fixed.
- **Incremental training data order** demonstrates cross-dataset transfer value:

| Row | Training data | Training videos | What is added |
|---|---|---:|---|
| 1 | CholecT50 only | 35 | Baseline |
| 2 | + Cholec80 phase | 55 | Phase supervision extends to G6 |
| 3 | + Cholec80 tool-presence | 55 (same videos, new label dimension) | Instrument supervision: 50→85 videos |
| 4 | + Cholec80-CVS | 55 (same videos, new label dimension) | CVS supervision introduced |
| 5 | + Endoscapes | 168 | CVS + bbox massively expanded |
| 6 | + Structured prior | 168 (same data, method change) | Prior regularization |
| 7 | + Context modulation (Full) | 168 (same data, method change) | Complete method |

**Key expectations:** Row 2→3 should improve Instrument mAP (Tier 4); row 4→5 should improve CVS AUC (Tier 2a); row 1→7 should show significant total improvement on Tier 1 core metrics.

### 9.7 Evaluation Code Release

As part of the stricter event-centric protocol, evaluation code will be open-sourced with the paper, including:
- Change point extraction scripts (triplet-group level + instrument-set level + phase level)
- Change-mAP computation (per-group AP → mean)
- TTC metric computation (MAE, C-index, Brier score from hazard output)
- Change-conditioned evaluation scripts (stratifying dense metrics into All / Change-only / Non-change-only)
- Safety B1/B2 evaluation
- Per-video result export + permutation test
- Canonical registry, overlap-safe split, phase/instrument ontology map, event-label extraction code (required for reproducibility, not optional engineering details)

### 9.8 Primary Metric System

Metrics are organized into three categories: A = Primary (instrument-set change), B = Secondary (group/phase/dense), C = Auxiliary (CVS/anatomy/clipping).

#### A. Primary: Instrument-Set Change Anticipation

Instrument-set change is the sole primary benchmark, independent of clustering.

**A1. Event Timing (When — how long until the next change)**

| Metric | Definition | Significance |
|---|---|---|
| **TTC-inst MAE ↓** | Instrument-set change time-to-next-change mean absolute error (seconds), from λ_inst head | TTC prediction accuracy (instrument-level) |
| **C-index-inst ↑** | Concordance index measuring TTC ranking consistency (instrument-level) | Discriminative ability (correctly distinguishing "fast change" from "slow change") |
| **Brier-inst @5s ↓** | Integrated Brier Score evaluating TTC survival function calibration, k ∈ {5, 10, 20}; main table reports @5s, appendix supplements @10s and @20s | Probability calibration quality |

**A2. Event Detection (Whether — will a change occur within the horizon)**

| Metric | Definition | Significance |
|---|---|---|
| **Event-AP-inst @5s ↑** | **Binary classification AP for "will instrument-set change occur within k seconds," k ∈ {5, 10, 20}; main table reports @5s** | **Event detection ability, cluster-independent** |
| Event-AUROC-inst @5s ↑ | Same as above, AUROC; appendix supplements @10s and @20s | Threshold-independent event detection |

**A3. Post-Change State Prediction (What — what will the instrument set be after the change)**

| Metric | Definition | Significance |
|---|---|---|
| **Post-change Inst mAP ↑** | **Multi-label prediction mAP for the post-change instrument set, evaluated at the first real instrument-set change point** | **Post-change state prediction ability** |
| Post-change Inst macro-F1 ↑ | Same as above, macro-F1 | Class-balanced post-change prediction |

**Why A1/A2/A3 split:** The former "Inst-Change-mAP" name was ambiguous — reviewers cannot immediately distinguish "event detection," "TTC ordering," and "post-change state prediction." The new naming makes each metric's evaluation target self-evident: A1 = when it changes, A2 = whether it changes within the window, A3 = what it changes to.

**Future-state Acc @Δ moved to appendix:** This metric evaluates state prediction Δ seconds after a change occurs, which is not fully consistent with the event-centric mainline (predicting the first change and its post-change state). It could lead reviewers to perceive a slide toward dense-step evaluation. The main text focuses on Post-change Inst mAP (evaluated at the first real change point); Future-state Acc @Δ serves as appendix supplementary analysis.

#### B. Secondary: Group / Phase / Dense Metrics

Triplet-group change is a secondary semantic stress-test.

| Metric | Definition | Significance |
|---|---|---|
| **TTC-group MAE ↓** | Triplet-group change time-to-next-change MAE (seconds), from λ_group head | TTC prediction accuracy (group-level) |
| **C-index-group ↑** | Concordance index (group-level) | Discriminative ability |
| **Brier-group @5s ↓** | Group change Integrated Brier Score | Probability calibration |
| **Event-AP-group @5s ↑** | Detection AP for triplet-group change within k seconds | Secondary event detection |
| **Post-change Group mAP ↑** | Prediction mAP for post-change triplet-group set at group-level change points | Secondary semantic stress-test |
| Dense-step mAP | Triplet-group prediction mAP across all timesteps; stratified as All / Change-only / Non-change-only | Comparable with existing literature; change-conditioned stratification exposes temporal inertia inflation |
| Phase Accuracy @Δ | Phase classification accuracy at future Δ seconds | Phase prediction capability |
| Instrument Anticipation mAP | Instrument multi-label prediction mAP at future Δ seconds | Instrument prediction capability |

#### C. Auxiliary: Safety / CVS / Anatomy

Data audit reveals a critical fact: "CVS fully achieved" (total score ≥5) is extremely rare in Cholec80-CVS — only 23 annotation rows across 16 videos. If one simply defines "unsafe = imminent clipping + CVS not achieved," nearly all pre-clipping frames are "unsafe" — positive rate approaches 100%, False Alarm Rate becomes meaningless, and safety evaluation degenerates to trivial judgment.

**Solution: decompose safety evaluation into two independent sub-tasks rather than merging into a single binary judgment.**

**Sub-task B1: Clipping Event Anticipation (core safety alert)**

Directly defined using CholecT50 original instrument/verb labels, independent of CVS:

```python
def is_clipping_event(t, labels):
    """Per-instance scan; any instance containing clipper or clip/cut verb is clipping"""
    for ann in labels[t]:
        instrument_id = ann[1]
        verb_id = ann[7]
        if instrument_id == 4:    # clipper
            return True
        if verb_id in [4, 5]:     # clip, cut
            return True
    return False
```

Evaluation metrics:

| Metric | Definition |
|---|---|
| **Clipping Detection Rate @k** | Among real clipping events, proportion correctly alerted k seconds in advance (k ∈ {5, 10}) |
| **Clipping False Alarm Rate** | Proportion of "imminent clipping" alerts where no clipping occurs within k seconds |
| **Clipping PR-AUC** | Area under the PR curve at various thresholds |

Alert trigger: Instrument Head predicts clipper probability > τ_inst, or model-predicted group-set contains a clip-related group.

**B1 ground truth scope:** All 50 CholecT50 videos can be evaluated (requires only instrument/verb labels, independent of CVS). Test set: all 10 CholecT50-test videos.

**Sub-task B2a: CVS State Accuracy (primary safety evaluation)**

Evaluates model CVS state estimation accuracy on all test videos with CVS ground truth.

| Metric | Definition |
|---|---|
| **CVS criterion-wise AUC** | Independent binary classification AUC per criterion (≥1 threshold) |
| **CVS MAE** | Full-procedure absolute error between model-predicted CVS score (ordinal reconstructed 0–2) and ground truth |
| **CVS calibration** | Reliability diagram evaluating predicted probability calibration |

**B2a ground truth scope:** Endoscapes-test (40 videos, G7) + Cholec80-test (11 videos, G6) = **51 test videos**. This is a statistically stable CVS evaluation benchmark.

**Sub-task B2b: CVS-at-Clipping (Appendix supplementary analysis)**

Moved to appendix. Evaluates model CVS estimation at the clinically most critical window — known clipping timepoints on CholecT50-test videos with CVS. 9 videos have insufficient statistical basis for main text; serves as stress-test supplementary evidence.

| Metric | Definition |
|---|---|
| **CVS MAE at clipping** | Absolute error between model-predicted CVS score and ground truth at the actual clipping moment |
| **Early Warning Quality** | Temporal consistency of model CVS prediction within the k-second window before clipping (does it stably reflect CVS achievement/non-achievement) |

**B2b ground truth scope:** 9 G2 test videos (with both triplet + CVS(C80)).

---

## 10. Baselines and Ablations

### 10.1 Required Baselines

| Baseline | Description | Confound eliminated |
|---|---|---|
| **Copy-Current** | Predict next step = current step, TTC = ∞ | Exposes dense-step inflation; scores 0 on Change-mAP |
| **CholecT50-Only** | Single-dataset training (full triplet + phase) | Proves multi-source has benefit |
| **Naive Multi-Source** | Four-dataset joint training, mask-and-ignore | Proves structured prior has benefit |
| **Multi-Source + Label Masking** | Joint training + label-conditional loss (no prior) | Isolates prior's incremental contribution |
| **Multi-Source + Uniform Prior** | Joint training + uniform smoothing (equivalent to label smoothing) | Proves structure beats smoothing |
| **Multi-Source + Self-Distillation** | Joint training + EMA teacher | Proves domain prior beats model prior |
| **Anticipation Transformer** | Published surgical anticipation method (SuPRA or similar) | Comparison with anticipation SOTA |
| **MML-SurgAdapt / SPML style** | Partial-annotation multi-task learning on similar three-dataset setup (reproduced or adapted to our data protocol) | Head-to-head with latest partial-label multi-task approach |
| **SurgFUTR style** | State-change learning baseline (future prediction recast as state-change detection + cross-dataset benchmark) | Head-to-head with latest state-change future prediction |
| **Direct Hazard (no rollout)** | h_t directly inputs hazard head, skipping Module C latent transition | Isolates latent transition's incremental value |
| **Observer WM w/o Uncertainty** | σ replaced with zeros; dynamics produces state predictions only | Isolates uncertainty→hazard signal channel value |
| **Full SurgCast** | Complete method | — |

**Baseline grouping:**
1. **Trivial:** Copy-Current
2. **External:** SurgFUTR-style, MML-SurgAdapt-style, Anticipation Transformer
3. **Data ablation:** CholecT50-Only → +Cholec80 → +CVS → +Endoscapes
4. **Method ablation:** +Label masking → +Action-free rollout → +Action-conditioned (Ver. A) → +Event-conditioned observer WM (Ver. B) → Direct Hazard / Observer WM w/o Uncertainty → +Structured prior → Full SurgCast

**Key baseline rationale:**
- MML-SurgAdapt has already published partial-label multi-task learning on CholecT50+Cholec80+Endoscapes; not comparing would be seen as avoiding the most relevant prior work
- SurgFUTR has already recast future prediction as state-change learning with a cross-dataset benchmark; not comparing would appear to compete only against "old-era phase anticipation"
- Reproduction strategy: each baseline classified via the Baseline Reproducibility Ladder (Section 10.2)

**Key design principle:** Internal ablation baselines (Copy-Current through Self-Distillation) use **the exact same** encoder + temporal transformer architecture, changing only data sources, loss design, and prior type. This ensures performance gains can be attributed to the method, not model capacity. External baselines use original methods evaluated on our data protocol and metrics for fair comparison.

### 10.2 Baseline Reproducibility Ladder

External baseline reproduction quality directly affects comparison fairness and credibility. Each baseline is explicitly annotated with its reproduction tier:

| Reproduction Tier | Definition | Applicable when | Paper annotation |
|---|---|---|---|
| **Tier A (Exact)** | Official open-source code + original training protocol; only adapted to our data split and evaluation metrics | Complete runnable code available | No special annotation needed |
| **Tier B (Faithful)** | No official code but complete method description; core modules faithfully reproduced on our backbone/temporal encoder | Detailed method description available (architecture, loss, hyperparameters) | Annotated "faithfully reproduced on our backbone" |
| **Tier C (Style)** | Insufficient method description or architectural incompatibility; only core idea/loss design reproduced, explicitly noted as "style baseline" | Missing key details or incompatible architecture | Annotated with "-style" suffix (e.g., "MML-SurgAdapt-style") |

**Expected tiers per baseline:**
- MML-SurgAdapt: expected Tier B or C depending on code availability
- SurgFUTR: expected Tier B or C depending on code availability
- Anticipation Transformer (SuPRA): expected Tier A or B
- Internal ablation baselines (Copy-Current through Self-Distillation): all Tier A (fully implemented by us)

### 10.3 Result Table Design

**Table 1: Main Results — Action-Change Anticipation (two-segment design)**

**Caption:** "Core action-change results on the fixed Tier-1 CholecT50 test set (10 videos). Segment 1 (data ablation) fixes the method to Full SurgCast and varies training data; Segment 2 (method ablation) fixes training data to Full (all 4 datasets) and varies method components. Metrics organized as: A1 = When (event timing), A2 = Whether (event detection within horizon), A3 = What (post-change state prediction). Phase Acc evaluated on Tier-3 Cholec80-test (20 videos). Dense-mAP reported as All/Change-only/Non-change-only (change-conditioned evaluation). Numbers are mean ± std over 5 seeds."

**Segment 1: Data Ablation (fixed method = Full SurgCast)**

| Training Data | Event-AP-inst @5s ↑ | Post-change Inst mAP ↑ | TTC-inst MAE ↓ | C-index-inst ↑ | Brier-inst @5s ↓ | Dense-mAP | Phase Acc |
|---|---|---|---|---|---|---|---|
| CholecT50-Only | — | — | — | — | — | — | — |
| + Cholec80 (phase + tool-presence) | — | — | — | — | — | — | — |
| + Cholec80-CVS | — | — | — | — | — | — | — |
| + Endoscapes (Full data) | — | — | — | — | — | — | — |

**Segment 2: Method Ablation (fixed data = Full, all 4 datasets)**

| Method | Event-AP-inst @5s ↑ | Post-change Inst mAP ↑ | TTC-inst MAE ↓ | C-index-inst ↑ | Brier-inst @5s ↓ | Event-AP-group @5s ↑ | Post-change Group mAP ↑ | TTC-group MAE ↓ | Dense-mAP | Phase Acc |
|---|---|---|---|---|---|---|---|---|---|---|
| Copy-Current | 0.0 | 0.0 | ∞ | — | — | 0.0 | 0.0 | ∞ | high | — |
| SurgFUTR-style | — | — | — | — | — | — | — | — | — | — |
| MML-SurgAdapt-style | — | — | — | — | — | — | — | — | — | — |
| Direct Hazard (h_t → hazard, no rollout) | — | — | — | — | — | — | — | — | — | — |
| + Observer WM (Ver. B, event-conditioned) | — | — | — | — | — | — | — | — | — | — |
| + Action-conditioned dynamics | — | — | — | — | — | — | — | — | — | — |
| + Phase-gated hazard | — | — | — | — | — | — | — | — | — | — |
| + Delta-state prediction | — | — | — | — | — | — | — | — | — | — |
| + Ranking loss + state-age | — | — | — | — | — | — | — | — | — | — |
| **Full SurgCast** | **—** | **—** | **—** | **—** | **—** | **—** | **—** | **—** | — | — |

**Reading guide:** Segment 1 answers "does more data help?" (RQ2 data dimension); Segment 2 answers "does each method component contribute?" (RQ1 + RQ2 method dimension). If both segments show monotonic improvement, the core thesis holds. External baselines serve as independent reference frames.

**Table 2: Structured Prior Ablation**

**Caption:** "Structured prior ablation on Tier-1 CholecT50 test set (10 videos). All variants use identical encoder, temporal transformer, and dual hazard heads; only the prior regularization differs. Task-specific prior distributions used by default (categorical for phase, factorized Bernoulli for multi-label tasks). Numbers are mean ± std over 5 seeds."

| Prior Type | Change-mAP | C-index-inst ↑ | C-index-group ↑ | Brier-inst @5s ↓ | Brier-group @5s ↓ |
|---|---|---|---|---|---|
| No prior (mask-and-ignore) | — | — | — | — | — |
| Uniform prior | — | — | — | — | — |
| Static procedure prior | — | — | — | — | — |
| Self-distillation prior | — | — | — | — | — |
| **Static + context-modulated (Ours)** | **—** | **—** | **—** | **—** | **—** |

**Table 3: Safety-Critical Anticipation (B1 + B2a)**

**Caption:** "Safety-critical anticipation results. B1 (clipping detection) evaluated on Tier-1 CholecT50 test set (10 videos). B2a (CVS state accuracy) evaluated on Tier-2a test set (51 videos: Endoscapes-test + Cholec80-test). B2b (CVS-at-clipping) results reported in Appendix. Numbers are mean ± std over 5 seeds."

| Method | Clip Det. @5s | Clip Det. @10s | Clip FA Rate | CVS C1-AUC (B2a) | CVS C2-AUC (B2a) | CVS C3-AUC (B2a) |
|---|---|---|---|---|---|---|
| CholecT50-Only (no CVS) | — | — | — | N/A | N/A | N/A |
| + Cholec80-CVS only | — | — | — | — | — | — |
| + Endoscapes CVS | — | — | — | — | — | — |
| + Anatomy-Presence Head | — | — | — | — | — | — |
| **Full SurgCast** | **—** | **—** | **—** | **—** | **—** | **—** |

**Evaluation scope:**
- B1 metrics evaluated on **Tier 1 test set** (10 CholecT50-test videos)
- B2a metrics (CVS criterion AUC) evaluated on **Tier 2a test set** (51 videos: Endoscapes-test + Cholec80-test) — **primary CVS evaluation**
- B2b (CVS MAE@clip, Early Warning) in appendix — 9 videos' statistical basis too weak for main text
- Clipping PR-AUC in appendix

### 10.4 Ablation Checklist

**Main-text ablations (10 ablations, corresponding to Table 1 Segment 2 rows):**

| ID | Ablation | Question addressed | Priority |
|---|---|---|---|
| A_direct | Rollout vs Direct Hazard: h_t directly inputs hazard head, skipping Module C | **Core test:** Is observer world model better than direct TTC decoding? | Critical |
| A_action1 | Action-free vs detached predicted vs GT/predicted mixed (teacher forcing) | Does action conditioning help? Is teacher forcing necessary? | Critical |
| A_dynamics | Fixed-horizon action-conditioned (Ver. A) vs event-conditioned (Ver. B) | Which dynamics formulation is stronger? | Critical |
| A_phasegate | Phase-gated hazard → base-only hazard (remove residual experts + routing) | Does phase-specific hazard shaping improve TTC prediction? | High |
| A_delta | Delta-state instrument prediction → direct post-change set prediction | Does edit-based prediction improve post-change Inst mAP? | High |
| A_rank | Remove pairwise ranking loss (λ_rank = 0) | Does ranking loss close the C-index optimization gap? | High |
| A_age | Remove state-age covariates (age_embed zeroed) | Do duration covariates improve TTC prediction on long-TTC samples? | High |
| A_nll | MSE alignment vs heteroscedastic NLL | Does learned uncertainty improve over MSE? | High |
| A_dualhaz | Dual hazard heads → single hazard head (single-event modeling) | Is dual-event TTC modeling better than single-event? | Medium |
| A10a | DINOv3 ViT-B/16 → LemonFM (surgery-domain ConvNeXt-L, 1536-d) | Is domain-specific pretraining better than general SSL? | Medium |

**Appendix ablations (remaining):**

| ID | Ablation | Question addressed |
|---|---|---|
| A_determ | Observer WM with vs without learned uncertainty: σ replaced with zeros | Value of the heteroscedastic uncertainty → hazard signal channel? |
| A8 | Hazard head → MSE regression | Advantage of hazard modeling? |
| A9 | Hazard head → binned classification | Hazard vs ordinary classification? |
| A_new | Hazard head without d_t input (remove change magnitude signal) | Incremental value of predicted change magnitude? |
| A_action2 | Instrument-only vs inst+phase vs full factorized action token | What action token detail level matters? |
| A_action3 | Group token (multi-hot) vs factorized triplet (Set Transformer) | Does compositional triplet encoding matter? |
| A_film | EventDyn FiLM conditioning → concatenation+MLP | Does FiLM improve over naive concatenation? |
| A_multistep | 1-step rollout (Δ=1 only) vs multi-horizon rollout (Δ={1,3,5,10}) | Incremental value of multi-horizon design? (Version A only) |
| A1 | Remove phase head | How much does phase prior help change anticipation? |
| A2 | Remove CVS head (no Cholec80-CVS or Endoscapes CVS) | Is CVS data useful? |
| A3 | Structured prior → uniform | How much is structure worth? |
| A4 | Structured prior → self-distillation | Domain prior vs model prior? |
| A5 | Static prior only (β=0) | Is context modulation useful? |
| A6 | Remove dynamics loss (L_dyn = 0) | Does explicit NLL supervision matter? |
| A6' | Shared transition → 3 independent MLPs | Is parameter sharing useful? |
| A7 | Remove temporal module (single-frame MLP) | Is temporal modeling necessary? |
| A10b | DINOv3 ViT-B/16 → DINOv3 ViT-L/16 (304M, 1024-d) | Larger general backbone? |
| A10c | DINOv3 ViT-B/16 → DINOv2 ViT-B/14 | DINOv3 vs DINOv2? |
| A10d | DINOv3 ViT-B/16 → ImageNet ResNet-50 | Importance of strong backbone? |
| A11 | Change definition: strict vs group vs debounced | Sensitivity of change definition? |
| A12 | Sequence length: 8 / 16 / 32 | Impact of temporal window? |
| A13 | Coverage dropout rate: 0 / 0.1 / 0.3 / 0.5 | Optimal dropout rate? |
| A14 | Pure co-occurrence vs semantic-enhanced clustering | Does semantic info improve groups? |
| A15 | Remove Instrument Head | Are decomposed labels useful? |
| A16 | Cholec80-CVS only vs Endoscapes CVS only vs both combined | CVS set complementarity? |
| A17 | ±Cholec80 tool-presence for Instrument Head | Cross-dataset instrument transfer? |
| A18 | CVS Head: ordinal regression vs binary BCE vs MSE | CVS scoring approach? |
| A19 | G4 videos (Cholec80∩Endoscapes) contribution | New coverage group value? |
| A20 | CVS official pipeline (85% truncation + 5fps) vs custom full-procedure 1fps | Full CVS coverage value? |
| A_σgate | σ-gated prior → ordinary scalar α, β | Is σ-modulated prior useful? (optional) |
| A_evidence | Evidence-gated prior → uniform KL weight | Incremental value of evidence-gating? |
| A_src | CVS Head without source embedding | Is source calibration useful? |
| A_anat | Remove Anatomy-Presence Head | Incremental value of anatomy presence? |
| A_consist | Remove L_consist (anatomy→CVS consistency regularizer) | Does consistency constraint help CVS? |
| A_K | K=20 non-uniform bins → K=15 uniform bins | Extended range + non-uniform bins? |
| A_aux | No auxiliary → +CVS → +CVS+anatomy | Auxiliary task contribution? |
| A_xval | CholecT50 official k-fold cross-val (action branch only) | Split-independence? |
| A_srcln | Source-Aware ConditionalLN (P2, appendix-only): per-source LayerNorm affine params in transformer | Does source-specific normalization improve heterogeneous training? |

**Main-text ablation rationale:** The 10 main-text ablations map directly to Table 1 Segment 2 rows, each isolating a single method component. A_direct, A_action1, and A_dynamics are the most critical — they validate the three claims of the observer world model: (1) dynamics-based forecasting > direct hazard, (2) action conditioning matters, (3) event-conditioned > fixed-horizon. A_phasegate, A_delta, A_rank, and A_age validate the new components introduced by the data audit. Remaining ~30 ablations are in the appendix.

**A10 ablation group (including backbone decision threshold):**

A10 simultaneously serves the backbone decision checkpoint (Section 6.2): if LemonFM exceeds DINOv3-B by ≥1.5pp on validation Group-Change-mAP or ≥2.0pp on B2a CVS AUC, then switch default backbone.

LemonFM (`visurg/LemonFM`) is a surgery-domain foundation model released in 2025, based on ConvNeXt-Large architecture (1536-d output), pretrained on the LEMON dataset (938 hours of surgical video, 35 procedure types including cholecystectomy) using augmented knowledge distillation. On Cholec80 phase recognition, it improves +9.5pp Jaccard over general backbones.

A10 ablation group narrative value:
- If LemonFM > DINOv3-B → "domain-specific pretraining matters more than general SSL scale," supporting future improvement with surgery-domain backbones
- If DINOv3-B ≈ DINOv3-L → validates the hypothesis that ViT-L marginal returns are limited under medical imaging domain gap
- If structured prior consistently helps across all backbones → method is backbone-robust, core contribution is in temporal modeling
- A10a in main text (most informative); A10b–A10d in appendix

LemonFM engineering notes:
- LemonFM outputs 1536-d (vs main method 768-d); input projection changed to `Linear(1536→512)`
- LemonFM uses ConvNeXt-Large with global average pooling (no CLS token); extraction method adjusted accordingly
- Requires an additional HDF5 feature file (~253K × 1536 × 4 bytes ≈ 1.55 GB)
- Except for the input projection layer, downstream architecture is completely unchanged, ensuring ablation fairness

---

## 11. Paper Structure and Narrative

### 11.1 Title

> **SurgCast: Event-Centric Forecasting of Surgical Action Changes under Heterogeneous Partial Supervision**

Alternative shorter version:

> **Forecasting Surgical Action Changes: When Will the Next Instrument Change Occur and What Will Follow?**

The title conveys the problem (event-centric forecasting of action changes) and the setting (heterogeneous partial supervision), letting reviewers understand the contribution level from the title alone. Method details (observer world model, hazard) are deliberately excluded.

### 11.2 Main Text Structure (9-page limit)

| Section | Pages | Content |
|---|---|---|
| Abstract | 0.3 | One paragraph: problem, method, key results |
| Introduction | 1.5 | Problem motivation → existing method limitations → contributions |
| Related Work | 1.0 | Positioning table + differentiation vs anticipation / partial-label literature |
| Method | 3.0 | Modules B–D, with Figure 2 as the centerpiece |
| Experiments | 2.5 | Table 1 + Table 2 + Table 3 + qualitative visualization |
| Conclusion | 0.5 | |
| Appendix | Unlimited | Full ablations, phase-conditional analysis, implementation details |

### 11.3 Core Figure Design

**Figure 1 (page 1, problem setting + positioning):**

Left half: Data coverage structure matrix (7 groups). Horizontal axis: video IDs (grouped by G1–G7). Vertical axis: label dimensions (triplet, instrument, phase, tool-presence, CVS-Cholec80, CVS-Endoscapes, anatomy bbox). Color blocks show "which videos have which labels" — highlighting the 7-group mutually exclusive coverage structure and 99.3% CVS coverage.

Right half: A surgical video timeline example annotating change points, clipping events, and CVS states, visually demonstrating "what we predict" and "how unsafe transitions are defined."

**Figure 2 (pages 3–4, method architecture):**

Complete architecture diagram. Left: frozen DINOv3 ViT-B/16 + Causal Transformer → h_t. Center-left: Action Token Encoder path (coarse: instrument+phase → 64-d for 85 videos; fine: triplet-set via Set Transformer → 64-d for 50 videos; mask-aware fusion → a_t). Center: Module C Observer World Model with three outputs — (1) Phase-Gated Dual Hazard Heads receiving [h_t; a_t; d_t; age_embed] → λ_inst/λ_group (K=20) with base + 7 phase residual experts per event type and soft routing, (2) NextActionHead with delta-state instrument branch (delta_add/delta_remove) → post-change action prediction a^+, (3) EventDyn with FiLM conditioning → μ^+ + log σ². Teacher forcing mechanism shown with ρ annealing (GT → predicted). Right: Multi-task heads decoding from both h_t (current) and μ^+ (post-change), highlighting Instrument Head + Anatomy-Presence Head + CVS Head with anatomy injection (sg(anat_probs) input). Lower right: Module D Structured Prior Regularization (static prior lookup + context modulation + evidence-gating). Loss diagram includes L_rank (pairwise ranking) and L_consist (anatomy→CVS consistency).

**Figure 3 (page 7, qualitative results):**

Selected 1–2 test videos showing:
- Timeline with ground truth change points + clipping events
- Model TTC prediction (hazard function heatmap)
- Instrument Head clipper prediction probability curve
- CVS state and safety alert trigger timepoints
- Comparison with copy-current / CholecT50-only baselines

### 11.4 Core Narrative (5-sentence version)

1. Existing surgical anticipation metrics — including dense per-second prediction and implicit state-change detection — are severely inflated by temporal inertia (instrument-set change only ~4.07/min, triplet-set change only ~6.17/min), unable to distinguish genuinely predictive models from trivial copy-current baselines. At 1fps, the vast majority of adjacent frame pairs are identity transitions.

2. We propose a stricter event-centric anticipation protocol with instrument-set change as the primary benchmark, supplemented by change-conditioned evaluation that stratifies performance on change vs non-change frames, complementing and validating existing dense-step and state-change paradigms. The protocol applies to all methods and is not a closed proprietary metric.

3. The model implements an observer world model that jointly produces event timing (discrete-time hazard TTC), post-change action prediction, and event-conditioned latent state transitions with heteroscedastic uncertainty. Action-conditioned dynamics — conditioning on observed/predicted action tokens via teacher forcing — connects "what is happening now" with "when will it change" and "what will follow," while the learned uncertainty directly drives hazard estimation as an imagination confidence signal.

4. On the training side, we construct an overlap-safe multi-source protocol (4 datasets, 277 videos), jointly leveraging dispersed multi-source supervisory signals through leakage-safe splits and label-conditional masking — phase, tool, CVS, and anatomy labels, though dispersed across datasets, collectively enhance the latent surgical state representation.

5. Cross-center generalization is validated on HeiChole (24 publicly labeled external videos) through phase/instrument/action transfer, using intersection-only ontology metrics to ensure fair evaluation.

### 11.5 Contribution Ordering

**Three mainlines (contributions claimed in the Introduction):**

| Mainline | Contribution | Validation |
|---|---|---|
| **Mainline 1: Stricter Event-Centric Benchmark** | Stricter event-centric anticipation benchmark with instrument-set change as primary, supplemented by change-conditioned evaluation (change vs non-change frame stratification), eliminating temporal inertia inflation of dense-step metrics | Tier 1 evaluation + Copy-Current baseline + Change-conditioned comparison (RQ1) |
| **Mainline 2: Action-Conditioned Observer World Model for Event Forecasting** | Observer world model with action-conditioned event dynamics: jointly models event timing (dual hazard), post-change action prediction, and event-conditioned latent state transition with heteroscedastic uncertainty as imagination confidence. Teacher-forced action tokens with scheduled sampling ensure smooth training-to-inference transition | A_action1, A_dynamics, A_nll, A_direct, A_determ, A_dualhaz, hazard vs regression/classification (RQ1+RQ2 method validation) |
| **Mainline 3: Heterogeneous Partial-Supervision Protocol + External Validation** | Leakage-safe overlap-safe multi-source training (4 datasets, 277 videos) + HeiChole cross-center zero-shot validation (24 publicly labeled videos) | Tier 6 HeiChole + incremental data ablation (RQ2 + RQ3) |

**Implementation details (appear as method details or ablations, not listed as contributions):**

| Detail | Role | Location |
|---|---|---|
| DINOv3 / LemonFM backbone | Ablation A10 addresses backbone choice | Method implementation + Ablation |
| Shared transition MLP + horizon conditioning | Ablation A6' addresses parameter sharing value | Method 6.4 |
| Ordinal CVS Head + source calibration | Ablation A18 / A_src | Method 6.4 |
| Cross-dataset instrument transfer (tool-presence) | Ablation A17 | Method 6.4 |
| Anatomy-Presence Head (bbox utilization) | Ablation A_anat | Method 6.4 |
| Structured prior (optional regularizer) | Ablation A3–A5; if gain <1.5pp, relegated to appendix | Method 6.5 or Appendix |
| σ-gated prior strength | Optional, ablation A_σgate | Appendix |

---

## 12. Engineering Plan and Timeline (10 Weeks)

### Phase 0: Data Infrastructure (Week 1–2)

| Day | Task | Output | Milestone |
|---|---|---|---|
| D1–D2 | Video ID registry + Cholec80-CVS download | registry.json | |
| D3–D5 | CholecT50 preprocessing + semantic embedding + hybrid clustering + change point annotation | 50 npz + group definition + change statistics | **Immediately check change density** |
| D6–D7.5 | Cholec80 preprocessing (incl. tool-presence extraction and 6-tool mapping) + Cholec80-CVS preprocessing (incl. malformed interval handling) + Phase ontology verification | 80 npz + CVS frame-level labels + instrument_mapped | |
| D8–D9 | Endoscapes preprocessing + 6-video CVS consistency verification + bbox → anatomy-presence label extraction | ~201 npz (with anatomy_presence field) | |
| D10–D11 | Procedure graph + static prior (incl. factored distributions and CVS prior) + evidence-gating weight table | static_prior.pkl + evidence_weights.pkl | |
| D12–D12.5 | DINOv3 + LemonFM feature extraction | 3+3 HDF5 files | |
| D12.5–D13 | HeiChole data acquisition + ontology mapping + DINOv3 feature extraction | HeiChole HDF5 + ontology map | |
| D13–D14 | DataLoader + CoverageAwareSampler | Complete data pipeline | **M1: Data pipeline end-to-end running** |

### Phase 1: Core Model (Week 3–4)

| Day | Task | Output | Milestone |
|---|---|---|---|
| D15–D17 | Causal Transformer + multi-horizon heads (incl. Instrument Head + Anatomy-Presence Head + source-calibrated CVS Head with anatomy injection) | model.py | |
| D17–D18 | State-age feature extraction + ranking loss implementation | state_age.py, ranking_loss.py | |
| D18–D21 | Action Token Encoder + Observer World Model (Version B + Version A ablation variant) + phase-gated hazard head | action_encoder.py, observer_wm.py (hazard integrated into Module C) | |
| D21–D22 | Delta-state NextActionHead implementation | next_action_head.py | |
| D22–D23 | Training loop + label-conditional loss (incl. L_rank + L_consist) | train.py | |
| D24–D25 | Train Baseline 1: CholecT50-only | Results | |
| D26–D27 | Train Baseline 2: + Cholec80 + Cholec80-CVS | Results | |
| D28 | Train Baseline 3: + Endoscapes | Results | **M2: Incremental trend verified** |

**Go/No-Go Checkpoint (end of Week 4):**
If the incremental table's Change-mAP does not show an increasing trend, stop and check the data pipeline. If the pipeline is correct but the increment is negative, reassess the problem formulation.

**Phase-Gated Hazard Checkpoint (concurrent, end of Week 4):** Run A_phasegate ablation (phase-gated vs base-only hazard). If phase-gated hazard shows >3pp gain over base-only, Module D (structured prior) moves entirely to appendix — the two mechanisms address overlapping concerns and phase-gated hazard is simpler.

**Backbone Decision Checkpoint (concurrent, end of Week 4):** Compare LemonFM and DINOv3-B on validation set. If LemonFM exceeds DINOv3-B by ≥1.5pp on validation Group-Change-mAP or ≥2.0pp on B2a validation CVS AUC, LemonFM becomes the default backbone; otherwise DINOv3-B remains.

**Safety Ground Truth Check (concurrent, end of Week 4):**
Separately count B1, B2a, B2b ground truth events: B1 clipping events across all CholecT50 videos; B2a CVS annotation frames on Endoscapes-test + Cholec80-test (51 videos); B2b clipping-moment CVS frames on G2 test (9 videos). If B1 clipping events < 30, Table 3's B1 column needs downgrading; B2a's 51-video base ensures statistical stability; if B2b frames are insufficient, B2b becomes qualitative only (does not affect B2a).

### Phase 2: Prior + Full Integration (Week 5–6)

| Day | Task | Output | Milestone |
|---|---|---|---|
| D29–D30 | Anatomy→CVS injection tuning + L_consist validation | Integrated into CVS Head | |
| D30–D31 | Structured prior (static only) + evidence-gating implementation | prior.py | |
| D31–D32 | Context-modulated prior implementation | Integrated into train.py | |
| D33 | Coverage dropout implementation | Integrated into dataloader | |
| D34–D36 | Full SurgCast training + hyperparameter tuning (incl. λ_rank, λ_consist, α_delta, β_delta) | Complete model results | |
| D37–D38 | Event-centric evaluation code (instrument-set change as primary metric) | evaluate.py | |
| D39–D40 | Safety-critical evaluation code (B1 + B2a on 51 videos + B2b on 9 videos) | Safety results | |
| D41–D42 | Small-scale prior ablation (verify prior + evidence-gating effectiveness) | Static vs uniform vs no-prior vs ±evidence-gating | **M3: Prior effectiveness verified** |

**Go/No-Go Checkpoint (mid-Week 6):**
If structured prior gain over mask-and-ignore is < 1.5pp (Change-mAP), reduce the prior's weight in the paper and fall back to event-centric forecasting + hazard TTC as the main contribution.

### Phase 3: Ablations + Analysis (Week 7–8)

| Day | Task | Output |
|---|---|---|
| D43–D46 | Main-text ablations: A_direct, A_action1, A_dynamics, A_phasegate, A_delta, A_rank, A_age, A_nll, A_dualhaz, A10a (~4 hours each) | Table 1 Segment 2 rows + Table 2 complete |
| D46.5 | Conditional: if TTC MAE >2s on long-TTC samples, implement long-context summary token (Section 6.3) and re-evaluate | Optional 17th token | |
| D47–D48 | External baseline training: MML-SurgAdapt-style + SurgFUTR-style | Table 1 external baseline rows |
| D49–D50 | Appendix ablations: A_determ, A8–A9, A_new, A_action2, A_action3, A_film, A_multistep, A1–A7, A10b–A10d, A11–A20, A_evidence, A_σgate, A_src, A_anat, A_consist, A_K, A_aux, A_srcln (A10b/c/d require additional feature extraction) | Supplementary materials |
| D49.5 | HeiChole zero-shot evaluation (Tier 6: Phase Acc @Δ, Instrument mAP, Action transfer — 24 videos, no CVS) | Table 4 HeiChole row |
| D49.75 | HeiChole optional few-shot (5 videos fine-tune + re-eval on phase/instrument/action) | Table 4 few-shot row (optional) |
| D50 | CholecT50 official cross-val (action branch, A_xval) | Appendix table |
| D51–D52 | Phase-conditional analysis | Analysis table |
| D53–D54 | Qualitative visualization (Figure 3) | Hazard heatmaps + instrument prediction curves + timelines |
| D55 | Safety-critical results compilation | Table 3 |
| D56 | All results double-check | Final data |

**M4: All experimental data collected**

### Phase 4: Paper Writing (Week 9–10)

| Day | Task | Output |
|---|---|---|
| D57–D58 | Figure 1 (problem setting) + Figure 2 (architecture) | 2 core figures |
| D59–D61 | Method section | 3 pages |
| D62–D63 | Experiments section | 2.5 pages |
| D64–D65 | Introduction + Related Work | 2.5 pages |
| D66–D67 | Abstract + Conclusion + Appendix | Complete draft |
| D68–D70 | Internal review + revision + submission | **M5: Final submission** |

---

## 13. Risks and Mitigation

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Latent transition gains marginal (A_direct shows <2pp) | Medium | Medium (narrative already adjusted — latent transition is not the sole core claim) | If A_direct shows latent transition < direct hazard + 2pp, position it as auxiliary representation-learning regularizer; fall back to event-centric protocol + hazard modeling + multi-source as three mainlines |
| Structured prior gains marginal (<1.5pp) | Medium-high | Low (already demoted to optional) | Prior is already optional; marginal gains do not affect core narrative. If phase-gated hazard shows >3pp, Module D moves entirely to appendix |
| Phase-gated hazard overfits on low-frequency phases | Medium | Medium | Low-frequency phase experts (TrocarPlacement, etc.) regularized via L2 toward zero; soft routing from h_t prevents hard phase assignment; base hazard always active as fallback. Monitored at Week 4 checkpoint |
| Ranking loss batch-size sensitivity | Medium | Medium | Pair count = O(B²); with B=16 and moderate right-censoring, valid pairs per batch may be small, causing noisy L_rank gradients. Mitigation: (1) λ_rank = 0.1 (low weight limits noise propagation); (2) monitor effective pair count per batch and increase B if median < 20 pairs; (3) fallback: accumulate pairs across gradient-accumulation steps or disable L_rank if C-index gain < 0.5pp |
| Delta-state class imbalance at group-only change points | Medium | Low | ~34% of change points are triplet-group-only (instrument set unchanged), where delta_add = delta_remove = 0. This biases the delta head toward predicting no instrument change. Mitigation: (1) L_next_inst is active only at instrument-set change points (masked at group-only changes); (2) monitor delta head recall on instrument-change vs group-only change points separately; (3) if recall gap > 15pp, add focal weighting to delta BCE terms |
| Change-mAP very low for all methods (<5%) | Medium | Medium | Adjust change definition (use debounced), allow ±1s tolerance, report multiple thresholds; instrument-set change as primary provides cleaner metrics |
| TTC derailed by label flicker noise | Medium | High | Group-level change instead of strict change; debounced change as fallback; K=20 non-uniform bins extend coverage, reducing right-censoring ratio |
| Cholec80-CVS and Endoscapes CVS inconsistent | Medium | Medium→Low | Source-specific calibration directly mitigates annotation protocol differences; quantify consistency on 6 dual-CVS videos; if severely inconsistent, use the two CVS sets separately |
| Phase ontology alignment failure | Low | High | Check timeline alignment on 45 overlap videos per-phase; if needed, fall back to 6-phase mapping |
| Hazard head training instability | Low | Medium | Pretrain other heads first, then add hazard head; layered learning rates |
| Reviewers find contributions insufficient | Medium→Low | High | Three focused mainlines + MML-SurgAdapt/SurgFUTR head-to-head comparison; no longer stacking 6 parallel claims |
| Reviewers challenge benchmark as artificially constructed | Medium→Low | Medium | Three-layer benchmark with instrument-set change completely cluster-independent, eliminating "entirely defined by clustering" concern |
| DINOv3 has domain gap on surgical images | Medium | Medium | LemonFM as mandatory baseline (not optional ablation); if LemonFM significantly outperforms DINOv3, switch main backbone |
| Unsafe transition events insufficient | Low | Medium | Count at Week 4; if < 30 events, Table 3 becomes qualitative display |
| CVS evaluation statistical stability insufficient | Medium→Low | Medium | B2a evaluates on 51 test videos (vs previously 9), greatly improved statistical stability |
| Cholec80 tool-presence vs CholecT50 instrument semantic mismatch | Medium | Low | Ablation A17 quantifies impact; if large, use only CholecT50's 50 videos |
| CVS positive sparsity causes insufficient CVS Head training signal | Medium-high | Medium | Ordinal regression + source calibration; Endoscapes CVS supplements positive examples; Anatomy-Presence Head provides complementary anatomical signal |
| Custom Cholec80-CVS parsing inconsistent with official pipeline | Low | Low | Compare on 6 Endoscapes overlap videos; ablation A20 quantifies differences |
| Group-set change frequency too high or too low | Medium | Medium | Measure immediately at Phase 0 Day 3–5 and adjust group count; if <1.5/min, reduce to 10–12 groups; instrument-set change as primary reduces group frequency dependency |
| External baselines unreproducible | Medium | Medium | Follow Baseline Reproducibility Ladder (Section 10.2): Tier A (official code) → Tier B (faithful reproduction) → Tier C (style baseline, explicitly annotated) |
| CholecT50 single-split results questioned | Low | Low | Supplement with CholecT50 official cross-val results (action branch only) in appendix |
| HeiChole data acquisition or preprocessing issues | Low | Medium | Public dataset (24 labeled videos); if ontology mapping is difficult, fall back to phase+instrument subset evaluation; handle 25/50 fps mixed frame rates and extra-abdominal white frames |
| HeiChole zero-shot performance very poor | Medium | Medium | If zero-shot Phase Acc / Instrument mAP < random baseline, try few-shot (5 videos fine-tune); if still poor, report as negative result and analyze causes |
| Teacher forcing schedule sensitivity | Medium | Medium | Sweep ρ schedule (constant vs cosine vs linear); fallback to detached-only (ρ=0, no teacher forcing) if schedule is fragile |
| Set Transformer marginal gain over multi-hot linear | Medium | Low | A_action3 ablation directly tests this; fallback to multi-hot + linear projection if Set Transformer gain < 0.5pp |
| Action tokens missing for G7 (192 videos) | Low (by design) | Low | Mask-aware fusion handles this: G7 uses learned default embedding; L_dyn and L_next masked for G7. G7 contributes only CVS + anatomy supervision |

### Fallback Plan

If structured prior fails (M3 checkpoint: gain < 1.5pp Change-mAP), **switch to Fallback version without hesitation.**

**Fallback paper title:**
> Stricter Event-Centric Anticipation of Surgical Instrument Changes under Multi-Source Supervision

**Fallback contribution reordering:**

| Priority | Contribution | Main-text location |
|---|---|---|
| Main contribution 1 | Stricter event-centric protocol + instrument-set change benchmark | Introduction + Section 3 |
| Main contribution 2 | Overlap-safe multi-source training + HeiChole external validation (24 videos) | Section 3 + Section 5 |
| Main contribution 3 | Discrete-time hazard modeling for TTC prediction (without observer WM) | Section 4 |
| Intermediate fallback | Version A (fixed-horizon action-conditioned dynamics) if Version B event-conditioned fails | Method 6.4 |
| Deepest fallback | No rollout (hazard-only from h_t); structured prior in appendix | Appendix |

---

## 14. Expected Contributions

**Three contribution pillars (claimed in the Introduction):**

1. **Stricter Event-Centric Benchmark:** A stricter event-centric anticipation benchmark for surgical video, with instrument-set change as the primary target, supplemented by change-conditioned evaluation. Eliminates temporal inertia inflation in dense-step metrics. Applicable to all methods; open-sourced with the paper.

2. **Action-Conditioned Observer World Model for Event Forecasting:** An observer world model with event-conditioned dynamics that jointly predicts event timing (phase-gated discrete-time hazard TTC with state-age covariates and pairwise ranking loss for C-index optimization), post-change action state (delta-state instrument prediction for edit-based change forecasting), and post-change latent state with heteroscedastic uncertainty. Factorized action tokens (coarse: instrument+phase for 85 videos; fine: triplet-set via Set Transformer for 50 videos) with teacher-forced scheduled sampling enable action-conditioned dynamics under heterogeneous coverage. Anatomy→CVS injection with consistency regularization strengthens safety-critical prediction. ~11.6M trainable parameters.

3. **Heterogeneous Partial-Supervision Protocol + External Validation:** A leakage-safe overlap-safe multi-source training protocol (4 datasets, 277 videos, 7 coverage groups). Cross-center generalization validated on HeiChole (24 publicly labeled external videos). The protocol, registry, and evaluation code are open-sourced.

**Implementation details vs main contributions:** DINOv3/LemonFM backbone choice, ordinal CVS with source calibration, cross-dataset tool-presence transfer, anatomy-presence head, anatomy→CVS injection, and structured prior regularization are implementation details addressed through ablations — they are not main contributions. Phase-gated hazard, delta-state prediction, state-age covariates, and pairwise ranking loss are method contributions validated through main-text ablations (A_phasegate, A_delta, A_age, A_rank).

---

## 15. Journal Extension Path

### NeurIPS → TMI

| Extension | New data | Time |
|---|---|---|
| CholecTrack20 tool persistence memory | 14 overlap videos with trajectory duration | 1.5 weeks |
| CholecInstanceSeg instance-level reasoning | Fine tool geometry | 1 week |
| Cross-procedure transfer (AutoLaparo hysterectomy) | 21 videos | 1 week |
| PhaKIR multi-center generalization | 8 videos | 0.5 weeks |
| More complete clinical safety evaluation | Extended CVS analysis + dual-CVS consistency study | 3 days |
| Label embedding space approach | Verb/target semantics into prediction space | 1 week |
| LLM knowledge-enhanced prior (exploratory) | Surgical textbook knowledge → soft constraint | 1 week |
| Source-Aware ConditionalLN (P2-1) | Per-source LayerNorm affine params in transformer; changes trunk philosophy but may improve heterogeneous training | 0.5 weeks |
| FiLM→Cross-Attention for EventDyn (P2-2) | Replace FiLM conditioning with cross-attention over action/event tokens; higher capacity but more complex | 1 week |
| Full ablation coverage | Complete coverage | 2 weeks |

### TMI → Nature Subsidiary

| Additional requirement | Notes |
|---|---|
| All datasets utilized | Including GraSP cross-platform pretraining |
| Clinical expert evaluation | 2–3 surgeons Likert-scale qualitative assessment |
| Larger claim | "Dispersed annotations in the surgical data ecosystem can be systematically integrated" |
| Clinical impact narrative | Clinical significance of safety-critical forecasting |
| Estimated additional time | 3–4 months |

---

## 16. Final Checklist

### Pre-Work Confirmations

- [ ] DINOv3 ViT-B/16 weights downloaded and verified (`facebook/dinov3-vitb16-pretrain-lvd1689m`, requires `transformers>=4.56.0`)
- [ ] LemonFM weights downloaded and verified (`visurg/LemonFM`, for ablation A10a)
- [ ] Local paths for all four datasets match the data audit report (CholecT50, Cholec80, Endoscapes, Cholec80-CVS)
- [ ] Cholec80-CVS surgeons_annotations.xlsx downloaded from Figshare
- [ ] sentence-transformers (all-MiniLM-L6-v2) installed and runnable locally
- [ ] CAMMA overlap mapping files obtained
- [ ] 2×A100 environment configured (PyTorch, HDF5, etc.)
- [ ] Git repository established; experiment logging framework set up (W&B or TensorBoard)
- [ ] Confirmed: NOT using Cholec80-CVS official preprocessing pipeline's 85% truncation
- [ ] Confirmed: NOT using Cholec80-CVS official 50/15/15 split
- [ ] Read data audit report Section 3.2 (CVS detailed statistics) and Section 3.3 (CholecT50 annotation format)
- [ ] Confirmed Endoscapes bbox annotation anatomy class list (verify 5 classes are covered)
- [ ] Checked whether MML-SurgAdapt / SurgFUTR have open-source code for baseline reproduction
- [ ] Confirmed CholecT50 official cross-val protocol (CAMMA repository k-fold scheme)
- [ ] Confirmed baseline reproduction strategy (Tier A/B/C ladder) and recorded each baseline's reproduction tier
- [ ] HeiChole data downloaded and verified (24 publicly labeled videos, phase + instrument + action + skill annotations, no CVS)
- [ ] HeiChole ontology mapping confirmed (phase/instrument/action → shared coarse category space, intersection-only metrics)
- [ ] HeiChole DINOv3 features extracted and stored as HDF5 (handle 25/50 fps mixed frame rates)

### Key Milestones

- [ ] **M1 (end of Week 2):** Data pipeline end-to-end running, one batch output correct
- [ ] **M2 (end of Week 4):** Incremental table first 4 rows have numbers, trend correct → Go/No-Go; safety event count complete
- [ ] **M2.5 (end of Week 4):** HeiChole features extracted (24 publicly labeled videos), phase/instrument/action zero-shot evaluation ready
- [ ] **M3 (mid-Week 6):** Structured prior small-scale verification → Go/No-Go
- [ ] **M4 (end of Week 8):** All experimental data collected
- [ ] **M5 (end of Week 10):** Paper submitted

### Per-Experiment Logging Requirements

- [ ] Configuration file (complete hyperparameters)
- [ ] Random seed
- [ ] Training loss curves
- [ ] Validation metrics per epoch
- [ ] GPU memory peak
- [ ] Total training time

---

## Appendix Note: What Was Consolidated

This document is a clean, reader-facing consolidation of the SurgCast research proposal (formerly maintained as a versioned working document through v8.0). The following types of content were removed or restructured:

- **Changelogs (v3.1→v8.0):** Eight changelog tables totaling ~130 lines were removed. The *reasoning* behind design decisions was distilled into Section 4 (Design Rationale); the *what-changed* narrative was removed.
- **Version markers:** All `[v5.0]`, `[v5.3]`, `[v5.4]`, `[v6.0]`, `[v7.0]`, `[v8.0]` tags throughout the document were removed. The text reads as the current and only version.
- **Chinese section headers and inline Chinese:** All Chinese was translated to English for the NeurIPS 2026 venue.
- **Patch-note language:** Internal process narration ("upgraded from...", "demoted from co-primary to...") was replaced with direct statements of the current design and its justification.
- **Evolution tracking sections:** Sections documenting how specific components changed across versions (e.g., safety evaluation structure evolution, coverage structure correction history) were consolidated into the current final stance.

The original working document (`docs/surgcast-proposal.md`) is preserved unchanged for reference.
