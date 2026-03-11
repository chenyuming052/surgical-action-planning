# SurgCast Scaffold

这个脚手架对应 SurgCast v5.4 的工程起点，目标是把研究计划落成可执行代码工程。

## 推荐实现顺序
1. scripts/build_registry.py
2. scripts/preprocess_cholect50.py
3. scripts/preprocess_cholec80.py
4. scripts/preprocess_cholec80_cvs.py
5. scripts/preprocess_endoscapes.py
6. scripts/build_priors.py
7. scripts/extract_features.py
8. scripts/train.py
9. scripts/evaluate.py

## 最重要的工程约束
- 先做 canonical registry，再做任何 split 或 feature 提取
- Cholec80-CVS 不使用官方 85% 截断和 50/15/15 split
- 所有 prior 只能由训练集视频统计得到
- Change 任务分成 instrument-set 与 triplet-group 双事件 TTC
- Anatomy 只在 bbox-annotated frames 上启用 observation mask
