# Experiment Matrix

## Main Table Order
1. Copy-Current
2. SurgFUTR-style
3. MML-SurgAdapt-style
4. CholecT50-Only
5. + Cholec80 phase
6. + Cholec80 tool-presence
7. + Cholec80-CVS
8. + Endoscapes
9. + Label-conditional masking
10. + Latent transition
11. + Structured prior (static + evidence-gated)
12. + Context-modulated prior (Full SurgCast)

## Stop / Go rules
- 如果 group change density < 1.5/min，group 数降到 10-12
- 如果 LemonFM 在 val Group-C-mAP > DINOv3-B 1.5pp 或 B2a CVS AUC > 2.0pp，切换 backbone
- 如果 structured prior 对 Change-mAP 提升 < 1.5pp，论文主贡献退守到 hazard + protocol
