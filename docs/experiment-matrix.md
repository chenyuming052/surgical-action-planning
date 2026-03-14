# Experiment Matrix

## Main Table Order (v6.0)

### Trivial
1. Copy-Current

### External Baselines
2. SurgFUTR-style
3. MML-SurgAdapt-style

### Data Ablation
4. CholecT50-Only
5. + Cholec80 phase
6. + Cholec80 tool-presence
7. + Cholec80-CVS
8. + Endoscapes

### Method Ablation
9. + Label-conditional masking
10. + Latent rollout (Module C)
11. **Direct Hazard (no rollout)** — h_t directly to hazard head, skip Module C [v6.0]
12. **Latent Rollout w/o Uncertainty** — σ_agg replaced with zeros [v6.0]
13. + Structured prior (static + evidence-gated) — optional regularizer
14. + Context-modulated prior (Full SurgCast)

## Tier 6: HeiChole External Validation [v6.0]
- 33 videos, zero-shot (no HeiChole data in training)
- Primary metrics: Inst-Change-mAP, TTC-inst MAE, Phase Acc @Δ
- Optional: few-shot (5 videos fine-tune + re-eval)

## Ablation Priority (v6.0)

### Main Text (core hypothesis tests)
| Ablation | Tests | Hypothesis |
|---|---|---|
| **A_direct** | Rollout vs Direct Hazard | H2: latent rollout > direct TTC decoding |
| **A_determ** | With vs without learned uncertainty (σ_agg → zeros) | Uncertainty → hazard signal value |
| **A_dualhaz** | Dual hazard heads vs single head | Dual-event TTC modeling value |
| A8 | Hazard head → MSE regression | Hazard modeling advantage |
| A9 | Hazard head → binned classification | Hazard vs classification |
| A6' | Shared transition → independent MLPs | Parameter sharing value |
| A_new | Hazard head without σ_agg input (516-d → 512-d) | σ_agg → hazard signal |
| A10a | DINOv3 ViT-B/16 → LemonFM | Domain-specific backbone |

### Appendix (prior ablations — optional regularizer)
| Ablation | Tests | Note |
|---|---|---|
| A3 | Structured prior → uniform | Structure value |
| A4 | Structured prior → self-distillation | Domain vs model prior |
| A5 | Static prior only (β=0) | Context modulation value |
| A_evidence | Evidence-gated → uniform KL weight | Evidence-gating value |
| A_σgate ⏳ | σ-gated prior → plain scalars | σ modulation value |

### Appendix (other)
| Ablation | Tests |
|---|---|
| A_multistep | 1-step vs multi-horizon rollout (Δ={1} vs {1,3,5,10}) |
| A_aux | No auxiliary → +CVS → +CVS+anatomy |
| A_src | CVS Head ± source embedding |
| A_anat | ± Anatomy-Presence Head |
| A_K | K=20 non-uniform → K=15 uniform bins |
| A_xval | CholecT50 official k-fold cross-val |
| A1-A2 | ± phase head, ± CVS head |
| A6-A7 | ± latent alignment, ± temporal module |
| A10b-d | Backbone variants |
| A11-A15 | Change definition, sequence length, etc. |
| A16-A20 | CVS source, tool-presence, etc. |

## Stop / Go rules
- 如果 group change density < 1.5/min，group 数降到 10-12
- 如果 LemonFM 在 val Group-C-mAP > DINOv3-B 1.5pp 或 B2a CVS AUC > 2.0pp，切换 backbone
- 如果 structured prior 对 Change-mAP 提升 < 1.5pp，prior 降级为 appendix detail（不影响核心叙事）
- **[v6.0] 如果 A_direct 显示 rollout < direct hazard + 2pp，H2 不成立，退守 protocol + multi-source**
- **[v6.0] 如果 HeiChole 零样本 < random baseline，尝试 few-shot；若仍差，报告为 negative result**
