# SurgCast: 异质缺失监督下的手术动作变化预见

## Surgical Action-Change Anticipation under Heterogeneous Missing Supervision

**版本：v5.4 (Method Closure)**
**目标会议：NeurIPS 2026**
**后续拓展：TMI / Nature 子刊**

---

## 变更日志（v3.1 → v4.0）

| 变更项 | 说明 |
|---|---|
| **新增 Cholec80-CVS 数据集** | 为 Cholec80 全部 80 个视频提供 CVS 三准则标注，从根本上解决 safety evaluation ground truth 稀疏问题 |
| **覆盖结构重构** | 最高覆盖组从 6 个视频扩展到 45 个视频（triplet + phase + CVS），数据集从 3 个变为 4 个 |
| **CholecT50 分解标签利用** | 新增 Instrument Head（6 类），利用原始 instrument/verb 标签重定义 safety 事件 |
| **语义嵌入辅助聚类** | Triplet-group 构建改为共现统计 + 语义嵌入混合方案 |
| **Safety 定义重构** | Unsafe Transition 的 ground truth 从依赖聚类的 CLIP_LIKE_GROUPS 改为直接使用 instrument/verb 原始标签 |
| **Structured Prior 增强** | 新增分解条件分布 P(instrument|phase)、P(cvs|phase, triplet-group) |
| **数据工程步骤更新** | 新增 Cholec80-CVS 预处理步骤和 CVS ontology 对齐 |
| **覆盖数字修正** | 修正 Cholec80 独有视频数（35 而非 30）等数据统计 |

## 变更日志（v4.0 → v5.0）

| 变更项 | 说明 |
|---|---|
| **覆盖结构表重构** | 替换为 7 组互斥覆盖表（修正组间不互斥、遗漏 3 个视频、多处数字错误）；总计 277 个视频 |
| **Triplet-Group Head: CE → BCE** | 47.4% 帧有多活跃 triplet，改为 per-group sigmoid + BCE 多标签预测；change 定义改为 multi-hot Hamming 距离 |
| **CVS 覆盖声明修正** | VID96/103/110 不在 Cholec80 中无 Cholec80-CVS；拆分为 G1（3 个两套 CVS）和 G3（3 个仅 Endoscapes CVS） |
| **Endoscapes-only 数量修正** | 201-9=192（非 ~150） |
| **新增 G4 覆盖组** | VID67/71/72（Cholec80∩Endoscapes，不在 CholecT50）纳入覆盖表 |
| **G5 修正** | 仅含 VID92/111（无任何 CVS），VID96/103/110 已移至 G3 |
| **特征帧数修正** | CholecT50 增量 ≈10K（非 ~37K），总帧数 ~253K，存储 ~777MB |
| **Prior 数据泄漏修正** | 所有 prior 严格限定训练集视频：P(cvs\|phase, triplet-group) 基数从 45 降到 31 个训练视频 |
| **Safety 测试集修正** | Safety 评估在 9 个测试视频上进行（VID111 无 CVS ground truth） |
| **Cholec80-CVS 预处理缺陷修正** | Step 3.5 新增处理 3 个畸形区间（final < initial）和 63 个越界区间 |
| **Cholec80 tool-presence 利用** | 80 个视频的 7-tool 二值标注用于扩展 Instrument Head 训练数据（6 工具映射），训练视频从 50 增到 85 个 |
| **Batch 比例重构** | 7 组覆盖等级的采样比例和激活损失更新 |
| **新增消融 A17-A20** | Cholec80 tool-presence 价值（A17）、CVS ordinal vs binary vs MSE（A18）、G4 贡献（A19）、CVS 全程 1fps vs 官方截断（A20） |
| **CVS 对齐阈值明确** | Cholec80-CVS ≥1→1, Endoscapes ≥0.5→1，G1 的 3 个视频验证一致性 |
| **创新点排序更新** | 5 个创新点按 NeurIPS 卖点重新排序；新增 multi-label set prediction 叙事 |
| **风险表更新** | CVS 一致性验证视频明确为 6 个；新增 tool-presence 语义差异风险 |

### v4.1-origin 改进（纳入 v5.0）

| 变更项 | 说明 |
|---|---|
| **Safety 评估分解（B1+B2）** | 将 Safety-Critical Anticipation 拆为 B1（Clipping Event Anticipation，50 视频可评）和 B2（CVS State Accuracy，9 测试视频），解决 CVS 稀疏导致的评估退化问题 |
| **CVS Head 序数回归** | CVS Head 从 `Linear(512,3)` + BCE 改为 `Linear(512,6)` + ordinal BCE，每 criterion 输出 P(≥1) 和 P(≥2) 两个累积概率，保留 0/1/2 序数结构 |
| **BCE 类别不平衡处理** | 新增 Section 6.1.1：Triplet-Group Head 和 Instrument Head 使用 per-group/per-class pos_weight；Focal loss (γ=2) 作为消融替代 |
| **Change 频率审计数据** | Section 5.1 补充精确 change 频率（phase ~0.17/min, instrument ~4.07/min, triplet-set ~6.17/min）；Section 5.4 拆分为已知基线 vs 聚类后待测 |
| **CVS 预处理策略声明** | Section 3.1 和 Step 3.5 明确声明不使用官方 pipeline（85% 截断 + 5fps 采样）且不使用官方 50/15/15 split |

## 变更日志（v5.0 → v5.1）

| 变更项 | 说明 |
|---|---|
| **Backbone 升级：DINOv2 → DINOv3** | 主方法从 DINOv2 ViT-B/14 升级为 DINOv3 ViT-B/16（同 86M frozen params，768-d 输出）。DINOv3 从 7B teacher 蒸馏，引入 Gram Anchoring 解决 dense feature 退化，训练数据 1.7B 图片（vs DINOv2 142M）。选择 ViT-B 而非 ViT-L 的理由：医学影像 domain gap 下 ViT-L 边际收益极小（文献报告 +0.7% Dice），且下游数据仅 ~253K 帧 |
| **新增消融 A10：Backbone 全面对比** | A10 从单一对比扩展为 4 级消融：DINOv3 ViT-B/16（主方法）vs LemonFM（手术域专用）vs DINOv2 ViT-B/14 vs ImageNet ResNet-50，回答 domain-specific pretraining vs general SSL vs backbone scale 三个问题 |
| **新增 LemonFM 作为 domain-specific 消融** | LemonFM（visurg/LemonFM, ConvNeXt-Large, 1536-d）在 938h 手术视频（含胆囊切除）上预训练，Cholec80 phase recognition +9.5pp Jaccard。作为消融 A10b 验证 domain gap 影响 |
| **风险表更新** | backbone 相关风险条目从 "DINOv2 在手术图像上特征不够好" 更新为反映 DINOv3 + domain-specific 消融的缓解策略 |

## 变更日志（v5.2 → v5.3：Reviewer Feedback Integration）

| 变更项 | 说明 | 对应建议 |
|---|---|---|
| **创新点收敛为三条主线** | 6 个并列创新点 → 3 条主线（问题+benchmark、建模、学习范式）+ 实现细化；DINOv3/LemonFM/ordinal CVS/tool-presence transfer/shared MLP 降级为实现细节和消融 | 总体建议 |
| **Safety 评估重构为 2+1 层** | B2 从仅 9 个测试视频改为：B2a（CVS state accuracy on Endoscapes-test，~30 视频）为主结果 + B2b（CVS-at-clipping on 9 overlapped test videos）为临床 stress test | 建议 #1 |
| **Instrument-set change 提升为 co-primary benchmark** | 从辅助 TTC 目标提升为与 triplet-group change 并列的 co-primary；三层 benchmark: instrument-set change（不依赖聚类）→ triplet-group change（细粒度）→ clipping event（临床关键） | 建议 #2 |
| **Hazard 时间范围扩展** | K=15 均匀 bins → K=20 非均匀 bins（1-10 秒逐秒 + 15/20/25/30 秒）；Transition horizon Δ={1,3,5} → {1,3,5,10}；σ_agg 从 R³ → R⁴ | 建议 #3 |
| **G7 batch 权重下调** | G7 从 40% → 25%，释放的 15% 分配给 action-rich 组（G2/G6）；可选 task balancing / gradient balancing | 建议 #4 |
| **CVS Head 增加 source-specific calibration** | 新增 source embedding（2-d one-hot: Cholec80-CVS / Endoscapes）→ 可学习 affine（per-source bias + temperature），缓解标注协议差异 | 建议 #5 |
| **Structured Prior 新增 evidence-gating** | 按 cell count / posterior entropy 调节 KL 权重：低支持区域弱约束、高支持区域强约束；优先级高于 σ-gating | 建议 #6 |
| **DINOv3 明确不作为贡献** | LemonFM 升级为 must-run baseline（非可选消融）；若 LemonFM > DINOv3-B 则切换主 backbone | 建议 #7 |
| **新增两类关键 baseline** | MML-SurgAdapt/SPML 风格（partial-label multi-task）+ SurgFUTR 风格（state-change learning）baseline | 建议 #8 |
| **新增 CholecT50 official cross-val 补充实验** | 在 CholecT50 subset 上补 official k-fold cross-val 结果（action branch），证明结论非 split-dependent | 建议 #9 |
| **Endoscapes bbox 轻量利用** | 新增 Anatomy-Presence Head（5 类多标签 BCE：gallbladder/cystic duct/cystic artery/cystic plate/hepatocystic triangle），利用 Endoscapes bbox 转化的 anatomy presence 标签 | 建议 #10 |
| **Related Work 更新** | 新增 SuPRA/SWAG、MML-SurgAdapt、SurgFUTR 定位对比；主 claim 从"first multi-source missing-label surgical forecasting"收窄为更具体的命题 | 总体建议 |

## 变更日志（v5.3 → v5.4：Method Closure）

| 变更项 | 说明 |
|---|---|
| **Dual-event hazard modeling** | 为 instrument-set change 与 triplet-group change 分别建模 TTC，补齐 co-primary benchmark 的训练闭环 |
| **Task-specific structured prior** | 将 prior 从统一 softmax 改为 phase 的 categorical prior 与 multi-label tasks 的 factorized Bernoulli prior |
| **Sparse anatomy supervision masking** | Anatomy-Presence Head 仅在 bbox-annotated frames 上激活，避免将未标注帧误当作负例 |
| **Hazard interval formalization** | 将非均匀时间 bins 明确定义为离散区间，并统一 TTC / censoring / anchor protocol |
| **Phase-level TTC demoted to appendix** | 避免主线过散，聚焦 action-change forecasting 主命题 |
| **Baseline reproducibility ladder** | 增加 external baseline 的 exact / faithful / style reproduction 分级说明 |
| **Backbone decision threshold** | LemonFM vs DINOv3-B 切换门槛量化（Group-C-mAP ≥1.5pp 或 CVS AUC ≥2.0pp） |
| **Exact split backfill** | 标注所有预估数字，Phase 0 后回填精确值 |
| **Sharper main claim** | 一句话定义更新为 event-centric anticipation 表述 |
| **Defensive table captions** | 所有主表 caption 显式标注评估集/seed 数/tier |
| **New benchmark acknowledgment in Intro** | Introduction 提前承认新任务定义，设定 reviewer 预期 |

---

## 〇、一句话定义这篇论文

**[v5.4] SurgCast研究腹腔镜胆囊切除中的事件中心型预见：在异质缺失监督下，预测下一次动作变化何时发生、变化后会是什么动作集合，以及这个转移是否具备解剖安全性。**

**EN:** SurgCast studies event-centric anticipation in laparoscopic cholecystectomy: predicting when the next action change will occur, what action set will follow, and whether the transition is anatomically safe, under heterogeneous missing supervision across multiple partially overlapping datasets.

---

## 一、研究问题与动机

### 1.1 为什么要做 action-change anticipation

现有手术视频理解的文献主流做两件事：当前帧识别（recognition）和未来状态预测（anticipation）。Recognition 已经被大量工作覆盖（TeCNO、Trans-SVNet、Rendezvous 等）。Anticipation 方向也在快速发展——SuPRA 做 joint recognition and anticipation，SWAG 做 long-term workflow anticipation，SurgFUTR 把 future prediction 改写为 state-change learning，还有基于图/超图的短时 action prediction。与此同时，partial-label multi-task learning 方向的 MML-SurgAdapt 已在 CholecT50+Cholec80+Endoscapes 上做了多数据集联合训练。

这些近期工作已经把 surgical anticipation 往两个方向推进：(1) SuPRA/SWAG/SurgFUTR 把 anticipation 从简单的 dense prediction 往 workflow forecasting 和 state-change learning 发展；(2) MML-SurgAdapt/SPML 在多源 partial-annotation 设定下做了 multi-task learning。**但即使在这些最新工作之后，仍然没有工作同时做到：** event-centric change-time prediction（而非 dense-step 或隐式 state-change）、显式的 discrete-time survival modeling for TTC、safety-aware CVS 状态预测、以及利用手术流程结构化约束的 evidence-gated prior for heterogeneous missing labels。

几乎所有现有 anticipation 方法（包括最新的）仍然在以下范式之一中工作：**"预测未来第 Δ 秒的动作类别是什么"**（dense-step），或"检测 state 是否发生变化"（binary change detection）。

这个定义有一个根本缺陷：手术视频中，连续多秒的动作往往是相同的。一个 copy-current baseline（直接复制当前动作作为预测）就能在 dense per-second 指标上拿到很高分。这意味着现有指标大量被"temporal inertia"inflate，真正反映预测能力的信号被淹没了。

**我们提出一个更锐利的问题：action-change anticipation。** 不是"未来第 Δ 秒是什么"，而是：

1. **When：** 距离下一次动作变化还有多久？（time-to-next-change, TTC）
2. **What：** 变化之后的状态是什么？（future state after change）
3. **Whether safe：** 这个变化是否与当前解剖安全状态兼容？（safety compatibility）

这三个子问题合在一起构成了一个完整的手术决策支持场景——比单纯的"预测下一秒 triplet"更有临床意义，也更能区分真正有预测能力的模型和 trivial baselines。

**[v5.4] New Task, New Benchmark：** Action-change anticipation 是本文定义的新任务，因此不存在现成的评估协议或 benchmark。我们提出的三层 event-centric benchmark（instrument-set change / triplet-group change / clipping event）本身即是方法论贡献——它消除了 dense-step metrics 被 temporal inertia inflate 的根本问题，同时通过 instrument-set change 层确保 benchmark 不完全依赖 triplet-group 聚类的特定选择。Dense-step metrics（per-second triplet-group mAP, phase accuracy @Δ）作为 secondary metrics 保留，确保与现有文献的向后兼容性。

### 1.2 为什么这个问题需要多源数据

Action-change anticipation 天然需要四种信息：

- **动作语义**（当前在做什么、历史动作模式）→ CholecT50 提供 triplet 标注（instrument × verb × target）
- **流程上下文**（手术进行到哪个阶段，该阶段的典型动作持续时间）→ Cholec80 提供全程 phase 标注
- **解剖安全状态**（关键结构是否充分暴露，是否具备安全转移条件）→ Cholec80-CVS 提供全程 CVS 标注 + Endoscapes2023 提供 ROI 窗口内的 CVS 标注与解剖 bbox

**没有任何单一数据集同时提供全部标注。** CholecT50 有 triplet 和 phase 但没有 CVS/anatomy；Cholec80 只有 phase 和 tool presence；Cholec80-CVS 为 Cholec80 追加了 CVS 但没有 triplet；Endoscapes 有 anatomy bbox 和 CVS 但仅覆盖手术的一个时间窗口。

因此，要做好 action-change anticipation，必须联合利用这些分散的监督信号。但这些数据集之间存在复杂的视频重叠、标签本体不完全一致、时间粒度不同——不能简单合并。

**这就是本文的核心方法问题：如何在异质缺失监督（heterogeneous missing supervision）下，利用手术流程的结构化先验，让不完整的多源标签互相增强？**

### 1.3 三个核心假设

**H1：Action-change anticipation 是比 dense-step prediction 更合适的 benchmark。**
因为它消除了 temporal inertia 的 inflation，迫使模型展示真正的预测能力。

**H2：缺失标签应该被结构化利用，而不是简单忽略。**
因为手术流程具有强组合约束——特定阶段下只有特定动作合法，高风险动作只应在特定解剖条件下发生。这些约束为缺失标签提供了比 uniform prior 更强的归纳偏置。

**H3：Event-centric 评估比 dense-step 评估更有信息量。**
Dense 指标被 copy-current 策略 inflate；以 change point 和 safety window 为中心的评估才能反映真实的 forecasting 能力。

---

## 二、与现有工作的定位

### 2.1 定位表

| 维度 | SuPRA / SWAG | MML-SurgAdapt / SPML | SurgFUTR | SurgCast（本文） |
|---|---|---|---|---|
| 预测对象 | 未来 phase / instrument | 多任务识别（当前帧） | 未来 state-change | **动作变化时间 + 变化后状态集合 + 安全兼容性** |
| 是否显式建模 change time | 否 | 否 | 部分（state-change 检测） | **是（discrete-time hazard，生存函数输出）** |
| 是否处理 heterogeneous missing labels | 否（单一数据集） | **是（partial-annotation multi-task）** | 部分（跨数据集 benchmark） | **是（evidence-gated structured prior）** |
| 是否多源训练 | 否 | **是（CholecT50+Cholec80+Endoscapes）** | 是 | **是（4 数据源，leakage-safe protocol）** |
| 是否 safety-aware | 否 | 否 | 否 | **是（CVS state + unsafe transition warning）** |
| 评估是否 event-centric | 否（dense per-second） | 否（per-frame recognition） | 部分 | **是（change-point mAP + TTC + safety detection）** |
| 对缺失标签的处理 | 不涉及 | Partial-label masking | 数据集特定 head | **Evidence-gated procedure-aware structured prior** |
| 时间建模 | Dense future step | 无（当前帧） | Future state prediction | **Discrete-time survival analysis** |

### 2.2 本文的差异化定位

**与 SuPRA / SWAG 的区别：** 这些工作将 anticipation 定义为 dense future step prediction（"Δ 秒后是什么 phase/instrument"）。我们将预测目标从 dense step 转向 event-centric change forecasting，同时引入 safety 维度。

**与 MML-SurgAdapt / SPML 的区别：** MML-SurgAdapt 已在 CholecT50+Cholec80+Endoscapes 上做 partial-annotation multi-task learning，但其目标是当前帧识别（recognition），不涉及 anticipation 或 temporal forecasting。我们的 heterogeneous missing supervision 框架服务于 forecasting 任务，且通过 evidence-gated structured prior（而非单纯的 partial-label masking）利用手术流程的组合约束。

**与 SurgFUTR 的区别：** SurgFUTR 将 future prediction 改写为 state-change learning 并建立了跨数据集 benchmark。我们的核心差异在于：(a) 用显式的 discrete-time survival modeling（hazard function + survival function）建模 time-to-change，而非隐式的 state-change 检测；(b) 引入 safety-aware CVS state prediction 作为 anticipation 的额外维度；(c) 用 evidence-gated structured prior 处理缺失标签，而非数据集特定 head。

**总结：** 本文真正能稳稳守住的命题是——**在 leakage-safe 的多源异质缺失监督下，预测下一次动作变化的时间（discrete-time hazard TTC）、变化后的动作集合（multi-label post-change state）、以及 critical transition 前的 safety state（CVS），并用 evidence-gated structured prior 利用手术流程的组合约束为缺失标签生成自适应软监督。** 这个定位与现有工作有清楚的边界。

---

## 三、数据资源与实验协议

### 3.1 四个核心数据集

| 数据集 | 提供的监督维度 | 视频数 | 帧数 | 覆盖范围 |
|---|---|---|---|---|
| CholecT50 | triplet (instrument-verb-target) + instrument + verb + target + phase | 50 | 100,863 | 全程 |
| Cholec80 | phase（密集标注）+ tool presence（7 类二值） | 80 | 184,498 | 全程 |
| Cholec80-CVS | CVS 三准则评分（0/1/2） | 80 | 覆盖 Preparation + CalotTriangleDissection 阶段 | Preparation → ClippingCutting 前 |
| Endoscapes2023 | anatomy bbox + CVS scores | 201 | 58,813 | ROI 窗口（dissection → first clip） |

**CholecT50 标签维度详解：** CholecT50 的每帧 JSON 提供 5 个独立的标注维度：triplet（100 类多标签）、instrument（6 类多标签）、verb（10 类多标签）、target（15 类多标签）、phase（7 类单标签）。v3.1 仅使用了 triplet（聚类后）和 phase，本版本新增利用 instrument 和 verb 维度。

**Cholec80-CVS 说明：** Cholec80-CVS 是 Universidad de los Andes 团队（Ríos et al., Scientific Data 2023）为 Cholec80 全部 80 个视频追加的 CVS 标注层。标注以时间段形式提供（起止帧 + 三个 criterion 各自的 0/1/2 评分），覆盖每个视频中 Preparation 和 Calot's Triangle Dissection 阶段。它不是独立的视频数据集，而是 Cholec80 的标注扩展。

**Cholec80-CVS 预处理策略声明：** 我们不使用官方预处理 pipeline（`annotations_2_labels.py`）。官方 pipeline 丢弃 pre-clip/cut 窗口的前 85%，仅保留最后 15% 并以 5fps 采样——这种截断服务于"clipping 前最后时刻的 CVS 识别"任务，不适合 anticipation 任务。我们需要观察 CVS 从不达标到达标的完整时间演变，因此直接从原始 XLSX 生成覆盖完整 pre-clip/cut 阶段的 1fps 逐帧标签。同样，我们不采用 Cholec80-CVS 官方 50/15/15 split（与 CAMMA combined split 存在大量交叉泄漏），CVS 标注的 split 完全由 CAMMA combined strategy 中各视频的 canonical ID 决定。

### 3.2 覆盖结构

四个数据集之间存在复杂的视频重叠关系。经数据审计验证，以下 7 组互斥覆盖构成完整的 277 个视频集合：

| 组 | 标签配置 | 视频 | 数量 | 占比 |
|---|---|---|---:|---:|
| **G1 三重交叉** | triplet+inst+verb+phase+CVS(C80)+CVS(Endo)+bbox | VID66,68,70 | 3 | 1.1% |
| **G2 CholecT50∩Cholec80** | triplet+inst+verb+phase+CVS(C80)+tool-presence | 45 个重叠中去掉 G1 的 3 个 | 42 | 15.2% |
| **G3 CholecT50∩Endoscapes** | triplet+inst+verb+phase+CVS(Endo)+bbox | VID96,103,110 | 3 | 1.1% |
| **G4 Cholec80∩Endoscapes** | phase+tool-presence+CVS(C80)+CVS(Endo)+bbox | VID67,71,72 | 3 | 1.1% |
| **G5 CholecT50独有** | triplet+inst+verb+phase | VID92,111 | 2 | 0.7% |
| **G6 Cholec80独有** | phase+tool-presence+CVS(C80) | 80−45−3=32 | 32 | 11.6% |
| **G7 Endoscapes独有** | CVS(Endo)+bbox | 201−9=192 | 192 | 69.3% |
| **总计** | | | **277** | **100%** |

**v4.0 → v5.0 覆盖结构修正要点：**
- v4 的 5 行覆盖表存在组间不互斥、遗漏 3 个视频（VID67/71/72）、多处数字错误
- VID96/103/110 不在 Cholec80 中，无 Cholec80-CVS 标注，不应标"无CVS"或"混合最高覆盖"；现归为 G3（仅有 Endoscapes CVS）
- 仅 G5 的 2 个视频（VID92/111）无任何 CVS 标注
- Endoscapes 独有视频为 201−9=192（非 v4 的 ~150）
- G1 的 3 个视频（VID66/68/70）同时拥有两套 CVS（Cholec80-CVS + Endoscapes），是跨标注体系一致性验证的核心样本
- G4 的 3 个视频（VID67/71/72）在 v4 中被遗漏，现提供 phase + tool-presence + 双套 CVS

**CVS 覆盖率：** 有 CVS 标注的视频组：G1(3) + G2(42) + G3(3) + G4(3) + G6(32) + G7(192) = **275/277 = 99.3%**。仅 G5 的 2 个视频无 CVS——这是一个远优于 v4 声称的覆盖论点。

**Instrument 监督覆盖：** CholecT50 的 instrument 标签覆盖 G1(3) + G2(42) + G3(3) + G5(2) = 50 个视频。Cholec80 的 tool-presence（6 工具映射：Grasper→grasper, Bipolar→bipolar, Hook→hook, Scissors→scissors, Clipper→clipper, Irrigator→irrigator，丢弃 SpecimenBag）额外覆盖 G4(3) + G6(32) = 35 个视频。**Instrument 训练视频总计 85 个**（较 v4 增加 70%）。

说明：
- 最终覆盖等级以注册表 `registry.json` 中的实际标签可用性为准
- Cholec80 tool-presence 为 1fps 采样的 7-tool 二值标注（184,498 行），其中 6 工具与 CholecT50 instrument 直接对应

### 3.3 视频 ID 注册表与泄漏控制

**这不是工程细节，而是让多源实验结果科学可信的前提。**

构建 `registry.json`，每个物理录像一条记录：

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

**Split 分配原则：**

- 采用 CAMMA 推荐的 split combination 策略（Walimbe et al., MICCAI 2025）
- 所有重叠在物理录像 ID 层面解决，不是数据集 ID 层面
- Endoscapes test split 与 Cholec80/CholecT50 零重叠（已由数据审计确认）
- Cholec80 在合并后移除与 CholecT50 test 重叠的视频
- Cholec80-CVS 作为 Cholec80 的标注层，自动继承 Cholec80 的 split 分配

**最终 split 预估：** ~191 train / ~41 val / ~45 test

### 3.4 保留数据集（期刊拓展用）

CholecTrack20、CholecInstanceSeg、CholecSeg8k、AutoLaparo、GraSP 不进入 NeurIPS 版本的训练标签池。保留用于：

| 数据集 | 期刊拓展用途 |
|---|---|
| CholecTrack20 | Tool persistence memory（器械在场时长作为 change prediction 信号） |
| CholecInstanceSeg | Instance-level 空间推理 |
| CholecSeg8k | 场景解析辅助监督 |
| AutoLaparo | 跨术式迁移实验（子宫切除 → 胆囊切除） |
| GraSP | 跨平台预训练（机器人 → 腹腔镜） |
| PhaKIR / HeiChole | 多中心泛化验证 |

### 3.5 任务边界声明

本项目明确定位为 **planning-relevant forecasting**，不是端到端自主手术导航。可用监督支持：

- 阶段感知的下一状态预测
- 解剖安全状态估计
- 动作变化预见

不支持：完整的策略学习、路径规划、自主控制。这个边界会在论文中明确声明。

---

## 四、方法设计：SurgCast

### 4.1 总体架构

```
输入帧序列 (t-15, ..., t), 1fps
        │
        ▼
[Module A] 冻结 DINOv3 ViT-B/16 ──→ 768-d frame features (离线提取, 存HDF5)
        │
        ▼
    Linear(768→512) + LayerNorm + Positional Encoding
        │
        ▼
[Module B] Causal Temporal Transformer (6 layers, 8 heads, dim=512)
        │
        ├──→ h_t (当前 latent surgical state, 512-d)
        │
        ├──→ [Module C] Latent State Transition + Multi-Head Prediction
        │         │
        │         g_Δ(h_t) → ĥ_{t+Δ}, Δ ∈ {1, 3, 5, 10}
        │         │
        │         ├──→ Triplet-Group Head ──→ ŷ_triplet-group     (BCE, multi-hot)
        │         ├──→ Instrument Head ──→ ŷ_instrument            (BCE)
        │         ├──→ Phase Head ──→ ŷ_phase                      (CE)
        │         ├──→ Safety/CVS Head ──→ ŷ_cvs                   (Ordinal BCE, source-calibrated)
        │         └──→ Anatomy-Presence Head ──→ ŷ_anatomy          (BCE, 5-class)  [v5.3]
        │
        ├──→ [Module D] Dual Discrete-Time Hazard Heads (TTC)  [v5.4]
        │         │
        │         ├──→ shared trunk: Linear(516,256)→GELU
        │         ├──→ λ_inst(k | h_t): instrument-set change hazard (K=20 非均匀区间)
        │         └──→ λ_group(k | h_t): triplet-group change hazard (K=20 非均匀区间)
        │
        └──→ [Module E] Structured Prior Regularization (核心创新)
                  │
                  ├──→ Evidence-gating: KL 权重按 cell count / posterior entropy 自适应调节
                  ├──→ Task-specific prior: Cat for phase, factorized Bern for multi-label  [v5.4]
                  └──→ L_prior = w_evidence · Σ_task L_prior^task
```

### 4.2 Module A：视觉特征提取（离线）

**选择：DINOv3 ViT-B/16（主实验）**

| 项目 | 说明 |
|---|---|
| 模型 | DINOv3 ViT-B/16, frozen, 86M params (`facebook/dinov3-vitb16-pretrain-lvd1689m`) |
| 输入 | 原始帧 resize 到 518×518 |
| 输出 | 768-d CLS token per frame |
| 存储 | 每个数据集一个 HDF5 文件，按 video_id 索引 |
| 总帧数 | ~253,000（Cholec80 184K + CholecT50 增量 ~10K + Endoscapes ~59K） |
| 提取时间 | ~1 小时（2×A100） |
| 存储量 | ~777 MB |

**DINOv3 vs DINOv2 升级理由：**
- DINOv3 从 7B teacher 蒸馏（vs DINOv2 ~1B teacher），同参数量下表征更强
- 引入 Gram Anchoring 解决 DINOv2 已知的 dense feature 长训练退化问题
- 训练数据 1.7B 图片（vs DINOv2 142M），RoPE 位置编码支持灵活分辨率
- ViT-B/16 输出 768-d，与 DINOv2 ViT-B/14 维度相同，下游架构无需改动

**为什么选 ViT-B 而非 ViT-L：**
- 医学影像 domain gap 下 ViT-L 边际收益极小（文献报告 MRI 分割仅 +0.7% Dice）
- ViT-L 多出的容量主要编码自然图像细粒度特征，对手术内窥镜场景增量有限
- 下游数据 ~253K 帧，信息瓶颈在下游数据而非特征维度
- ViT-B 作为主方法更凸显方法贡献（structured prior 等）不依赖最强 backbone

**消融中对比（A10）：** LemonFM（手术域专用）、DINOv2 ViT-B/14、DINOv3 ViT-L/16、ImageNet ResNet-50。

**[v5.4] Backbone Decision Threshold：** If LemonFM exceeds DINOv3-B by ≥1.5 points on validation Group-Change-mAP or by ≥2.0 points on B2a CVS AUC, LemonFM becomes the default backbone for the final NeurIPS submission; otherwise DINOv3-B remains the default. This decision is made at the Week 4 Go/No-Go checkpoint (see Section 10 Timeline).

**为什么冻结：**
1. 隔离科学问题——贡献全部集中在时序预测和缺失标签处理
2. 节省计算——所有实验只需提取一次特征
3. 稳定优化——异质数据集联合训练时，冻结 backbone 避免不同数据源的梯度冲突

### 4.3 Module B：Causal Temporal Transformer

| 参数 | 值 | 理由 |
|---|---|---|
| 层数 | 6 | 16 步序列的充分建模能力 |
| 隐藏维度 | 512 | 从 768-d 投影，参数可控 |
| 注意力头数 | 8 | 标准配置 |
| 序列长度 T | 16 frames (16 秒) | 覆盖典型动作持续时间 |
| 注意力掩码 | Causal (下三角) | 不泄露未来信息 |
| 位置编码 | Learnable | 处理均匀 1fps 采样 |
| Dropout | 0.1 | 标准正则化 |

**输入处理：** 768-d feature → Linear(768, 512) → LayerNorm → + positional encoding

**输出：** 每个时间步的 latent surgical state h_t ∈ ℝ^512

### 4.4 Module C：Latent State Transition + Multi-Head Prediction

**目的：** 我们通过一个共享的状态转移函数在学到的状态空间中建模手术动态，学习手术状态如何随时间演化的统一表示。转移模型同时产生未来状态预测和校准的不确定性估计——当模型对下一状态确信时，当前动作可能继续；当不确信时，动作变化正在临近。这一转移不确定性作为额外的信号通道流入离散时间风险头，将"会发生什么"（状态预测）与"何时发生"（变化点检测）连接起来。

**共享转移 MLP + Horizon Conditioning：**

用 1 个共享 MLP 替代多个独立 MLP_Δ，通过可学习的 horizon embedding 条件化预测范围：

```
ĥ_{t+Δ} = g(h_t, e_Δ),  Δ ∈ {1, 3, 5, 10}
```

- `e_Δ` = 可学习的 64 维 horizon embedding（每个 Δ 一个，共 4 个）
- 共享 MLP 结构：Linear(576, 512) → GELU → Linear(512, 1024)
- 隐藏层 512 维（与 h_t 维度匹配），输出 1024 维，分裂为两部分：
  - **状态预测**：前 512 维 → ĥ_{t+Δ}（未来状态估计）
  - **对数方差**：后 512 维 → log σ²_{t+Δ}（逐维转移不确定性）
- 参数量：576×512 + 512 + 512×1024 + 1024 ≈ **~820K**（MLP 参数不变，仅多 1 个 64-d embedding = +256 params）

**[v5.3] 为什么新增 Δ=10：** 预期 inter-change interval 约 15-30 秒（change density 2-4/min），Δ={1,3,5} 只覆盖短时窗口（≤5 秒），对 10-20 秒级别的 change 帮助不足。Δ=10 填补中程预测空白。注意 Δ=10 要求序列长度 T=16 中有 ≥6 步可用于 latent alignment 监督（t+10 必须在观测窗口内）；对序列末尾不满足此条件的样本，Δ=10 的 L_align 自动 mask。

**设计决策——为什么不用迭代 rollout：** 10 步自回归 rollout 会累积误差，需要中间监督或 teacher forcing，增加训练复杂度。Horizon conditioning 以极小风险获得参数共享的绝大部分收益。

**Heteroscedastic 转移不确定性：**

每次前向传播同时预测转移不确定性 σ_{t+Δ}，编码"模型对该未来状态的预测有多不确定"。跨 horizon 聚合为 4 维向量：

```
σ_agg = [√(mean(σ²_{t+1})), √(mean(σ²_{t+3})), √(mean(σ²_{t+5})), √(mean(σ²_{t+10}))]  ∈ R⁴
```

每个分量为对应 horizon 的 512 维不确定性向量的均方根均值，得到 1 个标量/horizon。**[v5.3] 新增 Δ=10 的不确定性分量**，提供中程预测信心信号。

**Latent alignment loss：**

```
L_align = Σ_Δ ‖ĥ_{t+Δ} - sg(h_{t+Δ})‖²
```

其中 sg(·) 是 stop-gradient，防止目标端表示坍塌。L_align 仅作用于 512 维状态预测部分；不确定性输出通过 alignment loss 的梯度间接学习——预测残差大的区域自然对应高不确定性。

**四个预测头（从 ĥ_{t+Δ} 解码）：**

| 头 | 结构 | 输出 | 损失 | 说明 |
|---|---|---|---|---|
| **Triplet-Group Head** | Linear(512, G), G≈15-20, per-group sigmoid | **G 维 multi-hot** | **BCE** | **[v5.0 修正] 47.4% 帧有 >1 活跃 triplet，多 group 可同时激活；multi_hot_target[g]=1 当帧有属于 group g 的活跃 triplet** |
| Instrument Head | Linear(512, 6) | 6 维多标签 | BCE | 预测器械存在，直接用于 safety alert |
| Phase Head | Linear(512, 7) | 单标签概率 | CE | 7 阶段单分类 |
| Safety/CVS Head | Linear(514, 6), source-calibrated | 6 维 ordinal CVS 预测 | Ordinal BCE | **[v5.3] 输入 512-d + 2-d source embedding；每 criterion 2 logits: P(≥1), P(≥2)；source-specific affine calibration** |
| **Anatomy-Presence Head** | Linear(512, 5), per-class sigmoid | **5 维多标签** | **BCE** | **[v5.3 新增] gallbladder/cystic duct/cystic artery/cystic plate/hepatocystic triangle 存在性；仅 G1/G3/G4/G7 有 bbox → 转化为 anatomy presence 标签** |

**v5.0+ CVS Head 修正（BCE → Ordinal BCE）：**
- **问题**：v5.0 使用 `Linear(512, 3)` + 独立 BCE，将 CVS 0/1/2 三级评分二值化为 {0,1}，丢失了"部分满足 vs 完全满足"的序数信息。审计报告确认 CVS"完全达标"（三准则总分≥5）极其稀少（仅 23 行标注，16 个视频），而"部分满足"信号更丰富——cystic_plate 有 83 个正例行，two_structures 有 287 个正例行。二值化丢弃了这一关键区分。
- **修正**：CVS Head 改为 `Linear(514, 6)`（含 2-d source embedding），每个 criterion 输出 2 个 logits，分别对应 P(score ≥ 1) 和 P(score ≥ 2) 的累积概率。
- **Ordinal BCE loss**：`L_cvs = Σ_c Σ_{k∈{1,2}} BCE(σ(logit_{c,k}), 𝟙[score_c ≥ k])`
- **推理时重建**：predicted_score_c = σ(logit_{c,1}) + σ(logit_{c,2})，取值范围 [0, 2]
- **Endoscapes 适配**：Endoscapes CVS 为连续分值，仅激活第一个阈值（≥0.5→1），第二个阈值 loss masked
- **参数量变化**：从 ~55K 增至 ~58K（negligible）

**[v5.3] CVS Head Source-Specific Calibration：**
- **问题**：Cholec80-CVS（0/1/2 surgeon score）和 Endoscapes（连续 CVS score ≥0.5 二值化）标注协议不同，仅 6 个双标视频用于一致性验证。共用一个 ordinal head 时，annotation-style shift 可能降低泛化。
- **方案**：CVS Head 输入从 512-d 扩展为 [h_{t+Δ}; s_src]（514-d），其中 s_src 为 2-d one-hot source embedding（Cholec80-CVS=\[1,0\], Endoscapes=\[0,1\]）。效果等价于给每个数据源一组独立的 bias 和 temperature 校正。
- **推理时**：对无 CVS 标签的视频（G5），使用 Cholec80-CVS 的 source embedding（因为 G5 属于 CholecT50，语义更近）。
- **参数增量**：2×6 + 6 = 18 个额外参数（negligible）。
- **消融**：A_src（去掉 source embedding → 退化到 v5.2 共享 CVS Head），验证 source calibration 的增量。

**v5.0 Triplet-Group Head 修正（CE → BCE）：**
- **问题**：v4 使用 CE（softmax + cross-entropy），假设单标签互斥。但 CholecT50 中 47.4% 的帧有 >1 个活跃 triplet，多个 group 可同时激活，CE 与数据多标签性质矛盾。
- **修正**：输出改为 per-group sigmoid（非 softmax），损失改为 `BCE(sigmoid(logits), multi_hot_target)`。
- **multi_hot_target 构建**：`multi_hot_target[g] = 1` 当该帧有任何属于 group g 的活跃 triplet。
- **叙事价值**：我们将 action-change anticipation 定义为 **set-to-set transition prediction**——预测从当前活跃动作集合到下一个活跃动作集合的转变。这自然容纳了手术中 47.4% 帧存在多个并发动作的事实，比单标签分类更忠实于临床实际。

**Instrument Head 训练数据扩展（v5.0）：**
- CholecT50 的 instrument 标签覆盖 G1-G3 + G5 共 50 个视频
- Cholec80 的 tool-presence（6 工具映射）额外覆盖 G4 + G6 共 35 个视频
- **Instrument Head 训练视频从 50 个扩展到 85 个**（增加 70%）
- 工具映射：Grasper→grasper, Bipolar→bipolar, Hook→hook, Scissors→scissors, Clipper→clipper, Irrigator→irrigator（丢弃 SpecimenBag，因 CholecT50 无对应类别）

**Instrument Head 的理由：**
1. Instrument 预测直接服务于 Safety-Critical Anticipation——预测 clipper 即将出现比通过 triplet-group 聚类间接推断更精确
2. Instrument-set change is one of the two co-primary TTC targets, alongside triplet-group change（6 个类别 vs 15-20 个 group，更干净的信号）
3. 参数增量极小（~3K 参数）
4. 跨数据集 instrument supervision transfer（Cholec80 tool-presence → CholecT50 instrument）是本文的方法创新之一

注意：Verb Head（10 类）和 Target Head（15 类）不作为独立预测头加入，因为 verb 和 target 与 instrument 有强组合约束，独立 BCE 预测会丢失这些关系。Verb 和 target 的语义信息通过 triplet-group 的语义嵌入聚类方案被隐式利用（见 Section 5.3）。

**[v5.3 新增] Anatomy-Presence Head：**

**动机：** Endoscapes 提供 201 个视频的 anatomy bounding box 标注，但当前 proposal 未在模型中使用 bbox——这意味着 Endoscapes 对模型的贡献仅限于 CVS 标注。将 bbox 转化为 anatomy-presence 辅助监督，可以在 safety branch 中引入 anatomy context，叙事更完整。

**实现：**
- 从 Endoscapes bbox 标注中提取 5 类解剖结构的帧级存在性标签：gallbladder、cystic duct、cystic artery、cystic plate、hepatocystic triangle
- 存在性定义：该帧的 bbox 列表中包含对应类别的 bbox → 1，否则 → 0
- Head 结构：Linear(512, 5)，per-class sigmoid
- **[v5.4] Sparse frame-level observation mask：** Anatomy-Presence supervision is available on bbox-annotated frames from Endoscapes-derived videos; loss is activated only on annotated frames via an observation mask m_anat:
  ```
  L_anatomy = Σ_t Σ_c m_anat(t,c) · BCE(ŷ_anat(t,c), y_anat(t,c))
  ```
  where m_anat(t,c) = 1 only on bbox-annotated frames. 明确区分 presence=0（structure absent，该帧有 bbox 标注但无该类别 bbox）与 unobserved（该帧无 bbox 标注，不参与 loss 计算）。
- 涉及视频：G1(3) + G3(3) + G4(3) + G7(192) = 201 个视频，但非每帧都有 bbox 标注
- 参数增量：~2.5K（negligible）

**叙事价值：** 将 Endoscapes 的利用从"另一套 CVS 标注"提升为"CVS + anatomy context"。解剖结构的可见性直接关系到 CVS 达标判断——如果 cystic duct 和 cystic artery 不可见，CVS 的 two_structures criterion 不可能达标。Anatomy-Presence Head 为 CVS Head 提供了互补的解剖学信号。

**消融：** A_anat（去掉 Anatomy-Presence Head），验证 bbox → anatomy presence 的增量价值。

### 4.5 Module D：Dual Discrete-Time Hazard Heads（TTC 建模）

**这是本文在建模层面最有辨识度的组件。**

#### 4.5.1 为什么用 hazard modeling 而不是直接回归或分类

| 方式 | 问题 |
|---|---|
| MSE 回归 | 对分布偏斜敏感，无法处理右截断 |
| Binned 分类 | 丢失序数信息，bin 边界选择敏感 |
| Ordinal regression | 比分类好，但没有显式的生存结构 |
| **Discrete-time hazard** | **天然处理右截断、保留序数结构、直接输出生存函数** |

#### 4.5.2 形式化定义

**[v5.3→v5.4 修正] 非均匀时间区间离散化：** 将未来时间离散化为 K=20 个非均匀**区间**，覆盖未来 30 秒：

```
Intervals I_k: (0,1], (1,2], (2,3], (3,4], (4,5], (5,6], (6,7], (7,8], (8,9], (9,10],
               (10,12], (12,14], (14,16], (16,18], (18,20], (20,22], (22,24], (24,26], (26,28], (28,30]
```

前 10 个区间逐秒（宽度 1s，精细分辨率），后 10 个区间每 2 秒（宽度 2s，粗分辨率）。**[v5.4] 明确定义为闭右开左区间而非时间点，确保概率质量的无重叠覆盖。**

**为什么从 K=15 均匀 bins 改为 K=20 非均匀 bins：**
- 预期 inter-change interval 约 15-30 秒（change density 2-4/min），K=15 仅覆盖 15 秒，导致大量右删失样本（change 发生在窗口外），hazard head 有效训练信号偏弱
- 非均匀 bins 在近端保持高精度（临床上"5 秒内是否有 change"比"28 秒 vs 30 秒"重要得多），远端节省 bins 数量
- K=20 相比 K=30 均匀 bins 更节省参数，同时覆盖相同时间范围

**[v5.4] Dual-Event Hazard Function：** 为 instrument-set change 和 triplet-group change 分别建模 TTC，补齐 co-primary benchmark 的训练闭环：

```
z_t = GELU(W_trunk · [h_t; σ_agg] + b_trunk)     # shared trunk: Linear(516, 256)
λ_inst(k | h_t) = σ(W_inst · z_t + b_inst)_k       # instrument-set hazard: Linear(256, K=20)
λ_group(k | h_t) = σ(W_group · z_t + b_group)_k    # triplet-group hazard: Linear(256, K=20)
```

其中共享 trunk 为 Linear(516, 256) → GELU，输入为 [h_t; σ_agg]（512 维状态 + 4 维转移不确定性）。两个 event-specific heads 各为 Linear(256, K=20)。σ 是 sigmoid。σ_agg 来自 Module C 的 heteroscedastic 输出（见 4.4）。

**设计理由：** Instrument-set change（~4.1/min）和 triplet-group change（~2-4/min）的事件频率和语义不同——同一器械下不同动作的变化（group change without instrument change）和器械出入（instrument change）是两种需要独立预测的事件。共享 trunk 确保表征共享，独立 heads 允许 event-specific 校准。

**TTC target 来源：** [v5.4] 输入上下文窗口为 16 秒，但 TTC 目标的扫描范围覆盖原始视频的全局未来帧（不限于 16s 输入窗口），预测范围为 30 秒。对每个 anchor 时间步 t：
- 从 t+1 开始在原始视频中扫描 instrument-set change 和 group-level change 的下一次发生位置
- 将 TTC 值映射到对应的离散区间 I_k
- 如果在 min(30s, remaining_video_time) 内未发生 change，标记为右删失

**不确定性→风险的直觉：** 当转移模型对未来状态确信（低 σ）时，当前动作可能继续；当不确信（高 σ）时，动作变化正在临近。这为 hazard 估计提供了一个正交于状态表征的信号通道。

**Survival function：** 在前 k 个区间内不发生 change 的概率（分别对两种事件）：

```
S_inst(k | h_t) = Π_{j=1}^{k} (1 - λ_inst(j | h_t))
S_group(k | h_t) = Π_{j=1}^{k} (1 - λ_group(j | h_t))
```

**Cumulative incidence：** 在前 k 个区间内发生 change 的概率：

```
F_inst(k | h_t) = 1 - S_inst(k | h_t)
F_group(k | h_t) = 1 - S_group(k | h_t)
```

#### 4.5.3 训练损失

**[v5.4] 双事件 hazard loss：** 对 instrument-set change 和 triplet-group change 分别计算 discrete-time survival 的标准 negative log-likelihood：

对于观察到 change 发生在第 k* 个区间的样本（以 instrument-set change 为例）：

```
L_hazard^inst = -log λ_inst(k* | h_t) - Σ_{j=1}^{k*-1} log(1 - λ_inst(j | h_t))
```

对于在观察窗口内没有发生 change 的样本（右删失）：

```
L_hazard^inst = -Σ_{j=1}^{K} log(1 - λ_inst(j | h_t))
```

triplet-group change 的 L_hazard^group 形式完全对称。总 hazard loss 为：

```
L_hazard = L_hazard^inst + η_group · L_hazard^group
```

其中 η_group 为 group hazard 的相对权重（默认 1.0，消融中扫 {0.5, 1.0, 2.0}）。两个损失独立计算各自的 TTC targets 和 censoring flags。

这个损失天然处理了"窗口内没有变化"的情况——不需要人为指定一个 target class，而是把它建模为被截断的观测。

#### 4.5.4 Early Warning 的决策边界

在推理时，可以对每种事件类型设置阈值 τ：

```
Alert at time t if F_inst(k_warn | h_t) > τ_inst  or  F_group(k_warn | h_t) > τ_group
```

即"如果模型认为未来 k_warn 秒内发生 instrument-set 或 group-level change 的概率超过各自阈值，则发出预警"。这直接给出了一个可调节的、event-type-specific 的 alarm system。

**[v5.3→v5.4] 非均匀区间的 TTC 计算：** TTC 期望值 = Σ_k mid(I_k) · P(T ∈ I_k | h_t)，其中 mid(I_k) 是第 k 个区间的中点：

```
区间中点: (0,1]→0.5, (1,2]→1.5, ..., (9,10]→9.5, (10,12]→11, (12,14]→13, ..., (28,30]→29
```

TTC MAE = |E[T] - T_true|，其中 T_true 为实际 TTC 值。Brier score 按区间右端点计算：Brier@k = (F(k|h_t) - 𝟙[T ≤ right(I_k)])²。**[v5.4] 对每种事件类型（instrument-set / triplet-group）分别计算 TTC 指标。**

#### 4.5.5 消融对比

| TTC 建模方式 | 作为消融对比 |
|---|---|
| MSE 回归 | 基线 |
| Binned 分类（[1-3s, 3-5s, 5-10s, 10s+]） | 基线 |
| Ordinal regression | 中间方案 |
| **Discrete-time hazard（主方法）** | — |

### 4.6 Module E：Structured Prior Regularization（方法论核心）

**核心思想：** 手术流程具有强组合约束——特定阶段下只有特定动作合法，特定器械只在特定阶段出现。这个结构为缺失标签提供了比"忽略"或"uniform smoothing"更强的归纳偏置。

#### 4.6.1 三层设计

**Layer 1：Static Procedure Prior（离线构建，训练中固定）**

**⚠ v5.0 关键修正：所有 prior 严格限定于训练集视频，杜绝数据泄漏。** v4 从"45 个最高覆盖视频"构建 prior，包含了 val/test 视频——这是数据泄漏。

| Prior | 数据源 | 训练集视频数 |
|---|---|---:|
| P_static(triplet-group \| phase) | CholecT50-train | 35 |
| P_static(triplet-group_{t+1} \| phase_t, triplet-group_t) | CholecT50-train | 35 |
| P_static(instrument \| phase) | CholecT50-train + Cholec80-adjusted-train（6 工具映射） | ~67 |
| P_static(phase_{t+1} \| phase_t) | Cholec80-adjusted-train | 36 |
| P_static(cvs_ready \| phase, triplet-group) | CholecT50-train ∩ Cholec80-train | **31** |

说明：
- P(triplet-group | phase) 改为 **per-group Bernoulli 分布**（与 BCE multi-hot 一致），不再是 softmax 分布
- P(instrument | phase) 扩展到 Cholec80-adjusted-train 的 tool-presence，总计约 67 个训练视频
- P(cvs_ready | phase, triplet-group) 的统计基数从 v4 的 45 降到 **31 个训练视频**
- 从 31 个训练视频中学到的 procedure prior 足以正则化 277 个视频的预测，体现手术流程结构的强归纳偏置

Endoscapes 映射：对于 G1 的 3 个视频和 G4 的 3 个视频（共 6 个同时有两套 CVS 的视频），可以验证 Cholec80-CVS 和 Endoscapes CVS 的一致性。

存储为查找表：`static_prior.pkl`。

**Layer 2：Context-Modulated Prior（可学习，训练中更新）[v5.4 Task-Specific]**

在 static prior 基础上，用当前 latent state h_t 做上下文调制。**[v5.4] 按任务类型使用不同分布形式，确保 prior 分布与预测 head 的输出空间数学一致：**

**Phase（单标签 categorical）：**
```
q_prior^phase(y_miss | y_obs, h_t) = softmax(α · log P_static(y_miss | y_obs) + β · g_φ^phase(h_t))
L_prior^phase = KL(Cat(p_θ^phase) ‖ Cat(q_prior^phase))
```

**Multi-label tasks（triplet-group / instrument / anatomy / CVS-binary）：**
```
q_prior^ml_c(y_miss | y_obs, h_t) = σ(α · logit(P_static_c(y_miss | y_obs)) + β · g_φ^ml(h_t)_c)
L_prior^ml = Σ_c KL(Bern(p_θ,c) ‖ Bern(q_prior,c))
```

其中 logit(p) = log(p/(1-p)) 将 Bernoulli 参数映射到 logit 空间（与 sigmoid 输出对应），确保加法组合在 logit 空间进行。

共享组件：
- P_static(y_miss | y_obs) 是 Layer 1 的查找表输出
- g_φ(h_t) 是一个轻量 MLP：Linear(512, 256) → GELU → Linear(256, C)（phase 和 multi-label 任务共享 hidden layer，输出层独立）
- α, β 是可学习的标量（初始化 α=1.0, β=0.1，让训练初期以 static prior 为主）；可选扩展为 σ_agg 的线性函数，见 4.6.2.1
- C 是被 mask 维度的类别数

**[v5.4] 为什么需要 task-specific 分布：** v5.3 对所有任务使用统一 softmax，但 triplet-group/instrument/anatomy 的预测 head 使用 BCE（per-class sigmoid），输出空间是独立 Bernoulli，而非互斥 categorical。对 multi-label 任务施加 softmax prior 会强制互斥约束，与数据中 47.4% 帧有多活跃 group 的事实矛盾。Task-specific 分布消除了这一数学不一致。

**直觉：** Static prior 告诉模型"在 CalotTriangleDissection 阶段，grasper-retract-gallbladder 是最常见的动作"。Context modulation 进一步告诉模型"但根据当前视觉上下文，grasper-dissect-cystic-duct 的概率应该更高"。对 multi-label 任务，这个调制独立作用于每个类别的 Bernoulli 参数。

**Layer 3：消融验证**

| 消融配置 | 回答的问题 |
|---|---|
| Uniform prior (等价于 label smoothing) | Structure 是否有用？ |
| Static prior only (α=1, β=0) | Context modulation 是否有用？ |
| Static + context prior (完整) | 主方法 |
| Self-distillation prior (teacher = EMA of model) | Domain structure vs model structure？ |
| No prior (mask-and-ignore) | Prior regularization 本身是否有用？ |

#### 4.6.2 训练时的 Coverage Dropout

对最高覆盖样本（同时有 triplet + phase + CVS 标签），以概率 p_drop = 0.3 随机 mask 掉一个已知维度，然后用 structured prior 作为被 mask 维度的软监督目标。

**流程：**

1. 采样一个最高覆盖样本，原始 mask m = [1, 1, 1, 1]（有 triplet, 有 instrument, 有 phase, 有 CVS）
2. 以 p_drop 概率选择 mask 掉 phase：m' = [1, 1, 0, 1]
3. 对被 mask 的 phase 维度，计算 q_prior(phase | triplet-group, h_t)
4. 模型对 phase 的预测 p_θ(phase | h_t) 用 KL divergence 与 q_prior 对齐

**Regularization loss（evidence-gated, task-specific）：**

```
L_prior = w_evidence(y_obs) · Σ_task L_prior^task
```

**[v5.4]** 其中 L_prior^task 使用 task-specific 分布（见 4.6.1 Layer 2）：phase 使用 KL(Cat‖Cat)，multi-label 任务使用 Σ_c KL(Bern‖Bern)。

**[v5.3 新增] Evidence-Gating 机制：**

Prior 统计基数来自 31 个训练 overlap videos。对 (phase × triplet-group × CVS) 联合空间，很多 cell 的样本数极少。盲目对所有 cell 施加等强度 KL 约束，低支持区域的 prior 可能不准确，反而误导模型。

**Evidence-gating 规则：**
```
w_evidence(y_obs) = min(1.0, count(cell) / N_sufficient)
```

其中 `count(cell)` 是训练集中对应 (phase, observed_labels) 组合的帧数，`N_sufficient` 为充分样本阈值（默认 50，消融中扫 {20, 50, 100}）。

**层次回退策略：**
- count(phase, group, cvs_level) ≥ N_sufficient → 使用完整联合 prior，w=1.0
- count(phase, group, cvs_level) < N_sufficient 但 count(phase, group) ≥ N_sufficient → 回退到边际 P(cvs | phase, group)
- count(phase, group) < N_sufficient → 回退到 phase-level P(group | phase)
- count(phase) < N_sufficient（几乎不会发生）→ 回退到 global marginal P(group)

**与 σ-gating 的关系：** Evidence-gating 解决的是"prior 本身可不可信"（数据支撑问题），σ-gating 解决的是"当前时刻是否需要 prior"（模型确信度问题）。两者正交，可以组合：`w_final = w_evidence · w_σ`。但 evidence-gating 优先级更高——没有数据支撑的 prior 不论 σ 多高都不应强加。

**为什么这不是 label smoothing：** Label smoothing 用 uniform distribution，与上下文无关。我们的 prior 是 conditioned on observed labels and current latent state，分布形状因样本而异。

**为什么这不是 self-distillation：** Self-distillation 的 teacher 信号来自模型自身（EMA），可能放大模型偏差。我们的 prior 来自手术流程的客观统计结构，不依赖模型质量。

#### 4.6.2.1 σ-Gated Prior Strength（C↔E 轻量连接）⏳

> **⏳ 优先级：可选扩展（低于 evidence-gating）。** Evidence-gating（4.6.2）优先实现。σ-gating 仅在 evidence-gating 已验证有效且有富余时间时叠加。若时间不足，保持 α, β 为普通可学习标量 + evidence-gating 即可。

**动机：** Module C 的转移不确定性 σ_agg 编码了"当前状态是否临近变化点"。这个信号对 Module E 的 prior 调制有天然的指导意义：当转移不确定（临近变化点）时，模型对下一状态的分类不确定，应更多依赖结构化 prior；当转移确定（动作稳定执行中）时，上下文足以准确预测，prior 权重可降低。

**改动：** 将 Layer 2 的标量 α, β 替换为 σ 的线性函数（适用于 phase 的 softmax 形式和 multi-label 的 sigmoid 形式）：

```
Phase:       q_prior^phase = softmax(α(σ_t) · log P_static + β(σ_t) · g_φ(h_t))
Multi-label: q_prior^ml_c  = σ(α(σ_t) · logit(P_static_c) + β(σ_t) · g_φ(h_t)_c)

α(σ_t) = α_0 + α_1 · σ̄_t
β(σ_t) = β_0 + β_1 · σ̄_t
```

其中 σ̄_t = mean(σ_agg) 是 4 维 σ_agg 的标量均值（**[v5.3] 从 3 维扩展为 4 维**），α_0, α_1, β_0, β_1 为 4 个可学习参数（替代原来的 2 个）。

**初始化：** α_0 = 1.0, α_1 = 0.5, β_0 = 0.1, β_1 = -0.05。使得高不确定性时 α 增大（prior 权重上升）、β 减小（context modulation 权重下降）。

**叙事价值：** 这在 Module C（dynamics）和 Module E（prior）之间建立了有原则的信息流——转移模型的不确定性估计直接调节 prior 强度，而非将两者作为独立 loss 项。叙事从"transition + prior 分别工作"升级为"transition uncertainty 自适应地调控 prior 信任程度"。

**参数与复杂度增量：** 从 2 个标量 → 4 个标量，可忽略。

**消融：** 见 A_σgate（Section 7.4）。

#### 4.6.3 Ontology Bridge

**Phase 对齐：** Cholec80 和 CholecT50 的 phase 名称相似但不完全一致（如 `GallbladderRetraction` vs `gallbladder-extraction`）。使用一个共享的 coarse phase space（7 个 phase），加上数据集特定的映射规则。

**注意：** `GallbladderRetraction`（Cholec80）与 `gallbladder-extraction`（CholecT50）仅为近似语义映射，不是精确对应。在 Phase 0 中需对 CholecT50 的 45 个重叠视频做显式验证——检查这两个标签在时间轴上是否真的对齐。

**CVS 对齐（Ordinal 版本 + Source Calibration）：** CVS Head 使用 ordinal regression + source-specific affine calibration（v5.3），每个 criterion 输出 P(≥1) 和 P(≥2) 两个累积概率。两个数据源的对齐方案：
- **Cholec80-CVS**：原始 0/1/2 三级评分直接映射到两个累积阈值——score ≥ 1 激活第一阈值，score ≥ 2 激活第二阈值。两个阈值的 loss 均参与训练。Source embedding = \[1, 0\]。
- **Endoscapes CVS**：连续评分仅激活第一个阈值（C1/C2/C3 ≥ 0.5 → 1），第二个阈值 loss masked（Endoscapes 无 0/1/2 级别区分）。Source embedding = \[0, 1\]。
- **Source calibration 的作用**：两个数据源的标注协议不同（surgeon 0/1/2 score vs 连续分值二值化），source embedding 让模型学习每个数据源特定的 decision boundary，缓解 annotation-style shift。
- **一致性验证**：在 G1(3) + G4(3) = 6 个双套 CVS 视频上，要求二值化（≥1）阈值的帧级一致率 >80%。
- **推理统一**：predicted_score_c = σ(logit_{c,1}) + σ(logit_{c,2})，取值 [0, 2]。对新视频使用 Cholec80-CVS source embedding（默认）。

### 4.7 总损失函数

```
L_total = L_task + λ_align · L_align + λ_hazard · L_hazard + λ_prior · L_prior
```

各项定义：

```
L_task = m_tri · L_triplet + m_inst · L_instrument + m_pha · L_phase + m_cvs · L_cvs + m_anat · L_anatomy
    (标签条件化多任务损失，m ∈ {0,1} 是可见性 mask)
    (L_triplet、L_instrument 为 BCE loss；L_cvs 为 ordinal BCE loss；L_phase 为 CE loss)
    (v5.0 修正：L_triplet 从 CE 改为 BCE，因 triplet-group 为 multi-hot 多标签预测)
    ([v5.4] L_anatomy = Σ_t Σ_c m_anat(t,c) · BCE(ŷ_anat(t,c), y_anat(t,c))，
     其中 m_anat(t,c) 是帧级观测 mask（仅 bbox-annotated frames 上为 1），
     区分 presence=0（structure absent）与 unobserved（no annotation）)

L_align = Σ_Δ ‖ĥ_{t+Δ} - sg(h_{t+Δ})‖²
    (Latent transition alignment, Δ ∈ {1, 3, 5, 10})

L_hazard = L_hazard^inst + η_group · L_hazard^group
    ([v5.4] 双事件离散时间生存损失（见 4.5.3），K=20 非均匀区间，分别对 instrument-set change 和 triplet-group change 建模)

L_prior = w_evidence · Σ_task L_prior^task
    (Evidence-gated structured prior regularization，仅在 coverage dropout 激活时计算)
    ([v5.4] 按任务类型使用不同分布形式：
     L_prior^phase = KL(Cat(p_θ^phase) ‖ Cat(q_prior^phase))  — categorical prior
     L_prior^ml = Σ_c KL(Bern(p_θ,c) ‖ Bern(q_prior,c))  — factorized Bernoulli prior for multi-label tasks
     适用于 triplet-group/instrument/anatomy/CVS-binary)
```

**超参数初始值：** λ_align = 0.5, λ_hazard = 1.0, λ_prior = 0.3

### 4.8 模型规模

| 组件 | 参数量 | 说明 |
|---|---|---|
| Input projection (768→512) | ~400K | |
| Causal Transformer (6 layers) | ~9.5M | |
| **Shared Transition MLP** | **~820K** | **[v5.2] Linear(576,512)+Linear(512,1024)** |
| Horizon embeddings (×4) | 256 | **[v5.3] 4 × 64-d 可学习 embedding（Δ={1,3,5,10}）** |
| Prediction heads (triplet-group + instrument + phase + CVS ordinal + anatomy) | ~61K | **[v5.3] +~2.5K anatomy head, +18 CVS source calibration** |
| Hazard head (dual) | ~142K | **[v5.4] shared trunk 516→256（~132K）+ 2×Linear(256,20)（~10K）; dual heads for instrument-set and triplet-group change** |
| Prior modulation MLP | ~140K | |
| Evidence-gating lookup | ~0 | **[v5.3] 预计算查找表，无可学习参数** |
| Context α, β (⏳ σ-gated: α_0, α_1, β_0, β_1) | 2 (⏳ 4) | **[v5.2] 可选扩展为 σ 的线性函数，见 4.6.2.1** |
| **总计** | **~11.2M trainable** | **与 v5.2 基本持平** |

加上冻结的 DINOv3 86M 参数，推理时总参数 ~97M，但只有 ~11.2M 需要训练。在 2×A100 40GB 上训练毫无压力。v5.3 新增的 anatomy head / source calibration / evidence-gating / Δ=10 embedding 总计不到 3K 参数。**[v5.4] dual hazard heads 增加 ~4K 参数（额外一个 Linear(256,20) head），task-specific prior 无额外参数。总模型规模与 v5.3 持平。**

---

## 五、Action Change 的定义与稳健性处理

### 5.1 为什么 change 定义至关重要

TTC 的质量完全取决于"什么算一次 change"。数据审计提供了精确的 change 频率基线：

| 粒度 | 每分钟变化次数 | 每视频变化次数 | 特点 |
|---|---|---|---|
| Phase transition | 0.17 | 5.8 | 太稀疏，不适合连续预测 |
| Instrument-set change | 4.07 | 136.7 | 中等频率，信号干净 |
| Target-set change | 4.71 | 158.2 | 中等频率 |
| Verb-set change | 5.36 | 180.3 | 较高频率 |
| Triplet-set change | 6.17 | 207.6 | 最高频率，但含大量 flicker 噪声 |

CholecT50 的 triplet 标注是 1fps 的逐实例标注，相邻帧之间可能出现 label flicker。Triplet-set 的 ~6.2/min 频率中包含大量此类噪声。通过 triplet-group 聚类将 100 类压缩到 15-20 个 group 后，预期 group-set change 频率会降至 ~2-4/min（需在 Step 2 中实测确认），对应每视频约 60-120 个 change point——足够支撑 TTC 训练。

### 5.2 三种 change 定义

| 定义 | 说明 | 预期 change density | 用途 |
|---|---|---|---|
| Strict change | 任何一个 triplet 维度变化即 change | 极高，充满 flicker | 内部验证用，不作为主任务 |
| **Instrument-set change（co-primary）** | **instrument 多标签向量发生变化（新器械出现或旧器械消失）** | **~4.1/min** | **[v5.3] co-primary TTC 目标，不依赖聚类** |
| **Group-level change（co-primary）** | multi-hot group 向量的 Hamming 距离 > 0（任一 group 激活状态翻转） | 中等（~2-4/min） | **co-primary TTC 目标，细粒度** |
| Debounced change | group-level 变化后持续 ≥ 3 秒才确认 | 较低 | 消融对比，验证 debouncing 的影响 |

**[v5.3] 三层 Benchmark 设计：** 将 benchmark 从单一依赖 triplet-group 聚类改为三层递进结构：

1. **Instrument-set change**：完全不依赖聚类，语义清楚（6 类器械的出现/消失），审稿人无法质疑目标定义的人为性。作为 sanity benchmark 和 co-primary 指标。
2. **Triplet-group change**：更细粒度，更能体现 clinical semantics（动作组合的变化），但依赖聚类定义。作为 co-primary 指标。
3. **Clipping event**：临床关键事件，binary 检测。Safety benchmark。

**为什么不只用 instrument-set change：** Instrument-set change 粒度较粗——同一器械下可以执行不同动作（grasper-retract vs grasper-dissect），这些变化对手术进程有实际意义但不会被 instrument-set change 捕捉。Triplet-group change 补充了这一层次。

### 5.3 Triplet-Group 的构建（v4.0 语义增强方案）

**v3.1 方案：** 纯共现聚类。从 CholecT50 的 100 个 triplet 类别出发，计算共现矩阵 → 层次聚类 → 15-20 个 group。

**v4.0 增强方案：** 共现统计 + 语义嵌入混合。

1. **共现矩阵（统计信号）：** 计算 100×100 的跨视频共现矩阵（两个 triplet 在同一帧同时出现的频率），归一化为相似度矩阵 S_cooc。

2. **语义嵌入（语义信号）：** 使用 sentence-transformers（all-MiniLM-L6-v2，本地运行，完全可复现）对 100 个 triplet 名称生成嵌入向量：

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')  # 离线执行一次
triplet_names = [
    "grasper, retract, gallbladder",
    "clipper, clip, cystic-artery",
    ...  # 100 个 triplet
]
embeddings = model.encode(triplet_names)  # 100 × 384
S_semantic = cosine_similarity(embeddings)  # 100 × 100
```

3. **混合相似度矩阵：**

```python
S_mixed = α * S_cooc + (1 - α) * S_semantic  # α 通过验证集调优或默认 0.5
```

4. 对 S_mixed 做层次聚类（Ward linkage），选取产生 15-20 个 cluster 的截断高度。

5. **检验：** 每个 group 应对应可解释的手术语义。

**语义增强的好处：**
- 语义相似的 triplet 会被聚在一起（即使数据中共现不多）
- 纯统计共现较多但语义不同的 triplet 会被分开
- 让 change 定义更有临床意义

**消融对比（A14）：** 纯共现聚类 vs 纯语义聚类 vs 混合聚类，评估 Change-mAP 和 group 可解释性。

### 5.4 必须报告的基础统计

以下统计量分为已知基线（审计报告提供）和聚类后待测两部分。

**已知（审计报告提供，论文中直接引用）：**
- Triplet-set changes：每视频 ~208 次，每分钟 ~6.2 次
- Instrument-set changes：每视频 ~137 次，每分钟 ~4.1 次
- Phase transitions：每视频 ~5.8 次
- Triplet 分布严重长尾：top 3 triplet 占 55.67%
- 多实例帧比例：47.4%

**需在 Phase 0 Step 2 中计算（聚类后才有）：**
- 平均每个视频有多少个 group-set change point
- 平均 inter-change interval（秒），及其分布（中位数、四分位距）
- Change point 在不同 phase 中的分布
- Debounced change 相比 group-set change 过滤掉了多少 flicker

**Group-set change density 预期范围：** 基于 instrument-set change（~4.1/min）和 triplet-set change（~6.2/min），group-set change 频率应在 2-4/min 范围内。**⚠ 如果实测低于 1.5/min（即每视频 < 50 个 change），需降低 group 数量至 10-12 个或放宽 change 定义。**

---

## 六、训练方案

### 6.1 Coverage-Aware Batching

每个 batch（batch_size=64）按以下 7 组比例采样：

| 组 | 视频数 | 激活损失 | Batch % | 样本/batch |
|---|---:|---|---:|---:|
| G1 三重交叉 | 3 | L_triplet + L_instrument + L_phase + L_cvs(C80) + L_cvs(Endo) + L_anatomy + L_hazard + L_prior | 5% | 3 |
| G2 CholecT50∩Cholec80 | 42 | L_triplet + L_instrument + L_phase + L_cvs(C80) + L_hazard + L_prior（30%概率） | **35%** | **22** |
| G3 CholecT50∩Endoscapes | 3 | L_triplet + L_instrument + L_phase + L_cvs(Endo) + L_anatomy + L_hazard | 5% | 3 |
| G4 Cholec80∩Endoscapes | 3 | L_phase + L_instrument(tool) + L_cvs(C80) + L_cvs(Endo) + L_anatomy | 4% | 3 |
| G5 CholecT50独有 | 2 | L_triplet + L_instrument + L_phase + L_hazard | 3% | 2 |
| G6 Cholec80独有 | 32 | L_phase + L_instrument(tool) + L_cvs(C80) | **23%** | **15** |
| G7 Endoscapes独有 | 192 | L_cvs(Endo) + L_anatomy | **25%** | **16** |

**[v5.3] G7 batch 权重从 40% 下调到 25%，释放的 15% 分配给 action-rich 组：**
- **问题**：G7 (192 videos) 仅有 CVS+bbox 标签，无 triplet/phase/instrument。40% 的 batch 权重意味着共享 temporal trunk 近一半梯度来自 CVS-only 数据，会把主干拉向 safety-state classification 而非 change forecasting——而主任务是后者。
- **调整**：G2 从 28% → 35%（核心评估组，有全部标签维度）；G6 从 15% → 23%（有 phase + instrument + CVS，action-rich）；G7 从 40% → 25%。
- **G7 的 L_anatomy 新增**：G7 虽然 CVS/anatomy-only，但现在同时训练 CVS Head 和 Anatomy-Presence Head，利用率提升。
- **可选增强**：如果主干仍被 G7 拉偏，可引入 GradNorm / task balancing 动态调节各组的 loss 权重。

**v5.0+ 覆盖率亮点：**
- 激活 L_cvs 的样本：G1 + G2 + G3 + G4 + G6 + G7 = **97%**（275/277 视频有 CVS）
- 激活 L_instrument 的样本：G1 + G2 + G3 + G4 + G5 + G6 = **60%**（85/277 视频有 instrument 标注）
- **[v5.4] 激活 L_anatomy 的样本：G1 + G3 + G4 + G7 = 201 个视频（73%）；帧级 observation mask m_anat(t,c) 仅在 bbox-annotated frames 上为 1**
- 激活 L_hazard 的样本（有 triplet 定义 change point）：G1 + G2 + G3 + G5 = 50 个视频
- **action-rich 组（G1-G6）现在占 batch 的 75%**（vs v5.2 的 60%），确保主干优先学习 change forecasting

### 6.1.1 类别不平衡处理

**Triplet-Group Head（BCE）：** 审计报告确认 triplet 分布严重长尾（top 3 占 55.67%），group 层面可能同样不均衡。训练中对 L_triplet-group 使用 per-group `pos_weight`，权重与该 group 正例频率成反比：`pos_weight_g = (N - N_g) / N_g`，其中 N_g 为训练集中 group g 为正的帧数。消融中对比 uniform BCE vs pos_weight BCE vs Focal loss（γ=2）。

**Instrument Head（BCE）：** 器械分布极不均衡——clipper 出现率约 2%，而 grasper/hook 出现率 >80%。使用 per-class `pos_weight`，确保低频但 safety-critical 的 clipper 获得足够梯度信号。

**CVS Head（Ordinal BCE）：** 不额外加权。CVS 正例稀疏反映了临床现实（CVS 在大部分 dissection 过程中确实不达标），强行上采样正例会扭曲先验分布。ordinal regression 已经通过保留 0→1→2 的序数信息缓解了二值化带来的信号丢失。

**Phase Head（CE）：** 不额外加权，各 phase 分布相对均匀。

### 6.2 序列构造

对每个视频，以 stride=8 的滑动窗口切出长度为 16 的子序列。每个子序列是一个训练样本。

**[v5.4] Dual TTC 目标的计算：** 对每个时间步 t，分别计算两种 change event 的 TTC targets：

1. **Instrument-set TTC：** 从 t+1 开始在原始视频全局帧中扫描 instrument 多标签向量的下一次变化（新器械出现或旧器械消失），将 TTC 映射到 K=20 非均匀区间
2. **Group-level TTC：** 从 t+1 开始在原始视频全局帧中扫描 multi-hot group 向量的下一次 Hamming 距离 > 0 变化，同样映射到 K=20 非均匀区间
3. **右删失：** 如果在 min(30s, remaining_video_time) 内未发生对应类型的 change，标记为右删失

**Anchor Protocol [v5.4]：**
- **训练时：** 每个 16-frame chunk 中的每个有效时间步 t 都作为 anchor 计算 hazard loss。对 Δ=10 的 latent alignment，仅在 t ≤ T−10（即 anchor 位置 + 10 在观测窗口内）时激活 L_align^{Δ=10}
- **推理时：** 使用单个 current-timestep anchor（序列最后一帧 t=T），输出 hazard function 和 TTC 期望值

### 6.3 训练超参数

| 参数 | 值 |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4, cosine decay to 1e-6 |
| Weight decay | 0.05 |
| Warmup | 5 epochs |
| Total epochs | 100 |
| Sequence length | 16 frames (16 sec) |
| Batch size | 64 |
| λ_align | 0.5 |
| λ_hazard | 1.0 |
| λ_prior | 0.3 |
| Coverage dropout rate | 0.3（仅最高覆盖样本） |
| Hazard time bins K | 20 (非均匀 bins，覆盖未来 30 秒) |
| GPU | 2 × A100 40GB |
| 预估训练时间 | ~6 小时 / 100 epochs |

---

## 七、评估方案

### 7.0 评估协议：数据集、测试集与评估范围

**本节显式规定"用什么数据评估什么"，是所有实验结果可解读的前提。**

#### 7.0.1 为什么没有现成 benchmark

Action-change anticipation 是本文定义的新任务（Innovation #1），不存在已有的 leaderboard 或标准 benchmark。现有手术 anticipation 工作（SuPRA、SWAG 等）评估的是 dense-step future prediction（"未来 Δ 秒是什么 phase/instrument"），与 event-centric change-point 预测有本质区别。因此本文需要自建评估协议，并将其作为方法论贡献的一部分开源。

**与现有评估体系的关系：**
- **CholecT50 Triplet Recognition Challenge**：评估当前帧 triplet 识别。我们报告 Dense-step mAP 作为 secondary metric 与之对接，但不作为主指标。
- **Cholec80 Phase Recognition**：评估当前帧 phase 分类。我们报告 Phase Acc @Δ 对接，同为 secondary。
- **本文主指标（Change-mAP、TTC 组）为全新定义**，无历史数据可比，通过 baselines（Section 7.2）建立参照系。

#### 7.0.2 数据划分

遵循 CAMMA 推荐的 combined split 策略（Walimbe et al., MICCAI 2025），在物理视频 ID 层面统一划分，确保同一物理视频在所有数据集中归属相同 split。

**预估划分（以 registry.json 为准）：**

| 覆盖组 | 总计 | Train | Val | Test | 说明 |
|---|---:|---:|---:|---:|---|
| G1（三重交叉） | 3 | 2 | 1 | 0 | 样本极少，全部用于 train/val 以最大化标注利用 |
| G2（CholecT50∩Cholec80） | 42 | 28 | 6 | 8 | 核心评估组，test videos 有全部标签维度 |
| G3（CholecT50∩Endoscapes） | 3 | 2 | 0 | 1 | VID96/103/110 之一进 test |
| G4（Cholec80∩Endoscapes） | 3 | 2 | 1 | 0 | 样本极少，优先 train |
| G5（CholecT50独有） | 2 | 1 | 0 | 1 | VID111 进 test（无CVS，用于 B1 不用于 B2） |
| G6（Cholec80独有） | 32 | 22 | 5 | 5 | 只有 phase + tool-presence + CVS(C80) |
| G7（Endoscapes独有） | 192 | 134 | 28 | 30 | 只有 CVS(Endo) + bbox |
| **总计** | **277** | **~191** | **~41** | **~45** |  |

**⚠ 注意：** 上表为合理预估，实际分配以 CAMMA combined split 的 canonical assignment 为准。Phase 0（数据基础设施）完成 registry.json 后回填精确数字。

**[v5.4] Exact Count Backfill Protocol：** 本文所有标记 "~" 或 "[exact TBD]" 的数字为 Phase 0 前的合理预估。Phase 0 完成 `registry.json` 后，将用精确数字统一替换。受影响的位置包括：Section 3.3 split 预估、Section 7.0.3 Tier 表的测试视频数、以及所有依赖 split 的统计量。

#### 7.0.3 测试集的分层评估范围

**核心原则：每个指标只在有对应 ground truth 的测试视频上计算。**

| 评估层级 | 测试视频来源 | 预估数量 | 可评指标 | 说明 |
|---|---|---:|---|---|
| **Tier 1: Core** | CholecT50-test (G2+G3+G5) | ~10 [exact TBD] | Change-mAP (instrument-set + triplet-group), TTC 全组 (inst+group), Future-state Acc, Dense-mAP, Safety B1 | **主表 (Table 1) 的评估基准** |
| **Tier 2a: CVS Safety (主)** | **Endoscapes-test (G7) + Cholec80-test (G6)** | **~35 [exact TBD]** | **CVS criterion AUC, CVS MAE** | **[v5.3] 主 CVS 评估集，统计基础更强** |
| **Tier 2b: CVS-at-Clipping (临床 stress test)** | **Tier 1 中有 CVS 的 (G2)** | **~8 [exact TBD]** | **CVS MAE@clip, Early Warning Quality** | **[v5.3] 临床关键窗口补充证据** |
| **Tier 3: Phase** | Cholec80-test (G2+G6) | ~13 [exact TBD] | Phase Acc @Δ | 扩大评估基数，验证粗粒度泛化；**[v5.4] Phase-level TTC 移至 appendix** |
| **Tier 4: Instrument** | CholecT50-test + Cholec80-test (G2+G3+G5+G6) | ~15 [exact TBD] | Instrument Anticipation mAP, **Instrument-set Change-mAP** | **[v5.3] 新增 instrument-set change 评估** |
| **Tier 5: Anatomy** | **Endoscapes-test (G7)** | **~30 [exact TBD]** | **Anatomy-Presence mAP** | **[v5.3 新增] 验证 bbox → anatomy presence 的学习效果；[v5.4] 仅在 bbox-annotated frames 上评估** |

**[v5.3] Safety 评估从 B2(9 videos) 重构为 2+1 层：**
- **B2a（主结果）**：CVS state accuracy on Endoscapes-test + Cholec80-test（~35 videos），这是更大的外部 safety-state 测试集，统计稳定性远强于 9 个视频
- **B2b（临床 stress test）**：CVS-at-clipping on the ~8 个 G2 测试视频，作为"小样本但临床关键窗口"的补充强化证据
- **为什么 B2a 更合理**：Endoscapes 本身有 201 个 ROI-window 视频、58,813 帧、强调 anatomy/CVS，天然适合做 safety-state 评估的主战场；9 个视频上的统计容易被审稿人质疑

#### 7.0.4 核心指标仅在 ~10 个视频上评估的正当性

这是本文必须直面的事实，而非回避的弱点。

**为什么不是缺陷：**

1. **任务定义决定的**：Action-change anticipation 在 triplet-group 层面定义 change point，只有 CholecT50 提供 triplet ground truth。这不是实验设计缺陷，而是该精细度任务的标注现实。
2. **与现有文献一致**：CholecT50 Triplet Recognition Challenge 同样在 ~10 个测试视频上评估，已被 MICCAI/TMI 社区接受。
3. **训练价值体现在 secondary metrics**：其他 ~35 个测试视频虽不能评核心指标，但可以通过 Phase Acc（Tier 3）和 CVS AUC（Tier 5）证明多源训练的迁移价值。如果加了 G6/G7 训练数据后 Phase Acc 和 CVS AUC 提升 → 间接证明这些数据对模型整体表征的贡献。

**如何增强说服力：**

1. 报告 **per-video** Change-mAP 和 C-index-inst / C-index-group（不仅是均值，展示分布）
2. 报告 5 次随机种子的均值 ± std
3. 对关键对比（Full SurgCast vs CholecT50-Only）做 **paired permutation test**（p < 0.05）
4. Appendix 展示每个测试视频的 hazard heatmap + change point 定性可视化
5. **[v5.3 新增] Appendix 补充 CholecT50 official k-fold cross-val 结果**（仅 action branch），证明结论不是某个 split 偶然。这是最省事的 split-robustness 验证——不需要对全部 277 视频做 cross-val（多源泄漏控制极其复杂），只需在 CholecT50 subset 上用 CAMMA 推荐的 k-fold 分法即可。

#### 7.0.5 Phase-Level Change 辅助评估（Appendix）

**[v5.2 新增, v5.4 降级为 appendix] 在 Cholec80-test 上构造 phase change point，评估粗粒度 TTC。**

**[v5.4] Phase-level TTC is reported as an auxiliary appendix analysis rather than a main benchmark, since the primary contribution of SurgCast is fine-grained action-change forecasting at the instrument-set and triplet-group level. Phase transitions (~6/video) are too sparse for robust TTC evaluation, and the aggregation from the triplet-level hazard head to phase-level hazard is underspecified. Retaining it in appendix preserves backward compatibility without distracting from the main contribution.**

**动机：** Tier 1 仅 ~10 个视频。Phase change point 可从全部 Cholec80-test (~13 videos) 上获取，扩大 TTC 评估的视频基数。

**定义：** Phase change point = phase 标签发生跳变的时刻。一个典型胆囊切除有 5-6 次 phase 变化。

**评估指标（与 Tier 1 TTC 指标平行）：**

| 指标 | 定义 |
|---|---|
| Phase-TTC MAE | Phase change 的 time-to-next-change 平均绝对误差 |
| Phase-TTC C-index | Phase change 的 concordance index（单 event，无需 inst/group 拆分） |
| Phase-TTC Brier @5s | Phase change 的概率校准（单 event，无需 inst/group 拆分） |

**注意：** Phase change 比 triplet-group change 粒度粗得多（一个手术 ~6 次 phase change vs ~50+ 次 triplet-group change），且相邻 phase 有更强的顺序约束。因此 Phase-TTC 应显著优于 triplet-level TTC——如果不是，说明模型的粗粒度 workflow 理解有问题。

**Hazard head 适配：** Phase-TTC 不需要额外的 hazard head。We report phase-level TTC as a coarse appendix analysis using phase change points extracted from ground-truth phase trajectories. 在推理时，将 triplet-level change 聚合到 phase-level：如果模型预测未来 k 秒内 phase 会变（通过 phase head 输出在不同 horizon 上的变化），直接从 hazard head 的 survival function 中提取。

**放置位置：** **[v5.4] 单独一个小表放 appendix（不再列入 Table 1 或 Tier 3 主指标）。**

#### 7.0.6 Cross-Dataset Transfer 评估

**这是 heterogeneous missing supervision（Innovation #2）的核心验证维度。**

Table 1 的逐行 ablation 已经体现了这一点（CholecT50-Only → 逐步加数据源 → Full SurgCast），但需显式声明：

- **所有方法在同一个 Tier 1 测试集（~10 CholecT50-test 视频）上评估核心指标。** 训练数据量不同，测试集完全固定。
- **训练数据逐步增加的顺序**体现 cross-dataset transfer 的增量价值：

| 行号 | 训练数据 | 训练视频数 | 新增什么 |
|---|---|---:|---|
| 1 | CholecT50 only | ~35 | Baseline |
| 2 | + Cholec80 phase | ~71 | Phase supervision 扩展到 G6 |
| 3 | + Cholec80 tool-presence | ~71 (同上，新增标签维度) | Instrument supervision 从 50→85 视频 |
| 4 | + Cholec80-CVS | ~71 (同上，新增标签维度) | CVS supervision 引入 |
| 5 | + Endoscapes | ~191 | CVS + bbox 大量扩充 |
| 6 | + Structured prior | ~191 (同上，方法改变) | Prior regularization |
| 7 | + Context modulation (Full) | ~191 (同上，方法改变) | 完整方法 |

**关键预期：** 行 2→3 应提升 Instrument mAP（Tier 4），行 4→5 应提升 CVS AUC（Tier 2/5），行 1→7 应在 Tier 1 核心指标上有显著总增益。

#### 7.0.7 评估代码开源

作为新任务定义的一部分，评估代码将随论文开源，包括：
- Change point 提取脚本（triplet-group level + phase level）
- Change-mAP 计算（per-group AP → mean）
- TTC 指标计算（MAE, C-index, Brier score from hazard output）
- Safety B1/B2 评估
- Per-video 结果导出 + permutation test

### 7.1 主指标体系

#### A. Action-Change Anticipation（核心指标组）

**[v5.3] 双层 Change-mAP：instrument-set + triplet-group 并列 co-primary**

| 指标 | 定义 | 意义 |
|---|---|---|
| **Inst-Change-mAP** | **[v5.3 co-primary] 仅在 instrument-set change point 上评估的 instrument 多标签预测 mAP** | **不依赖聚类的 change 预测能力** |
| **Group-Change-mAP** | 仅在 group-level change point 上评估的 triplet-group 多标签预测 mAP（per-group AP 再平均） | 细粒度 change 预测能力 |
| **TTC-inst MAE** | **[v5.4] instrument-set change 的 time-to-next-change 平均绝对误差（秒），由 λ_inst head 输出** | TTC 预测精度（instrument-level） |
| **TTC-group MAE** | **[v5.4] triplet-group change 的 time-to-next-change 平均绝对误差（秒），由 λ_group head 输出** | TTC 预测精度（group-level） |
| **C-index-inst / C-index-group** | concordance index，衡量 TTC 排序一致性；**[v5.4] 主表分列报告 instrument-level（C-index-inst）和 group-level（C-index-group）** | 区分力（是否能正确区分"快变"和"慢变"） |
| **Brier-inst @5s / Brier-group @5s** | 对"未来 k 秒内是否发生 change"的概率预测的 Brier score，k ∈ {5, 10, 20}；**[v5.4] 主表分列报告 inst 和 group（@5s），appendix 补充 @10s 和 @20s** | 概率校准质量（**[v5.3] k 范围扩展至 20s 以匹配 K=20 bins**） |
| Future-state Acc @Δ | 在 change 发生后 Δ 秒的 triplet-group 预测准确率 | 预测变化后状态的能力 |

#### B. Safety-Critical Anticipation

**v5.0+ 重构：分层 safety 评估，适应 CVS 稀疏现实。**

数据审计揭示了一个关键事实：Cholec80-CVS 中"CVS 完全达标"（三准则总分≥5）极其罕见——仅 23 行标注，16 个视频。即使二值化后（任何 criterion ≥1 为正），cystic_plate 也仅 83 行正例。如果简单定义"unsafe = 即将 clipping + CVS 不达标"，几乎所有 clipping 前的帧都是"unsafe"——正例率接近 100%，False Alarm Rate 失去意义，safety evaluation 退化为 trivial 判定。

**解决方案：将 safety evaluation 分解为两个独立子任务，而非合并为一个二值判定。**

**子任务 B1：Clipping Event Anticipation（核心安全预警）**

直接使用 CholecT50 原始 instrument/verb 标签定义，不依赖 CVS：

```python
def is_clipping_event(t, labels):
    """逐实例扫描，任一实例包含 clipper 或 clip/cut verb 即为 clipping"""
    for ann in labels[t]:
        instrument_id = ann[1]
        verb_id = ann[7]
        if instrument_id == 4:    # clipper
            return True
        if verb_id in [4, 5]:     # clip, cut
            return True
    return False
```

评估指标：

| 指标 | 定义 |
|---|---|
| **Clipping Detection Rate @k** | 在真实 clipping events 中，模型在 k 秒前正确预警的比例（k ∈ {5, 10}） |
| **Clipping False Alarm Rate** | 模型发出"即将 clipping"预警但 k 秒内未发生的比例 |
| **Clipping PR-AUC** | 不同阈值下的 PR 曲线下面积 |

预警触发条件：Instrument Head 预测 clipper 概率 > τ_inst，或模型预测的 group-set 包含 clip-related group。

**B1 Ground truth 可用范围：** 所有 50 个 CholecT50 视频均可评估（仅需 instrument/verb 标签，不依赖 CVS）。测试集为 CholecT50-test 的全部 10 个视频。

**子任务 B2a：CVS State Accuracy（主 safety 评估）[v5.3 重构]**

在所有有 CVS ground truth 的测试视频上评估模型对 CVS 状态的估计准确性。

评估指标：

| 指标 | 定义 |
|---|---|
| **CVS criterion-wise AUC** | 对每个 criterion 独立评估二分类 AUC（≥1 阈值） |
| **CVS MAE** | 模型预测的 CVS score（ordinal 重建值 0-2）与 ground truth 的全程绝对误差 |
| **CVS calibration** | Reliability diagram，评估预测概率的校准质量 |

**B2a Ground truth 可用范围：** Endoscapes-test (~30 videos, G7) + Cholec80-test (~5 videos, G6) = **~35 个测试视频**。这是一个统计稳定的 CVS 评估基准。

**子任务 B2b：CVS-at-Clipping（临床 stress test）[v5.3 重构]**

在已知 clipping 将要发生的时间点上（CholecT50-test 中有 CVS 的视频），评估模型在临床最关键窗口的 CVS 估计。

评估指标：

| 指标 | 定义 |
|---|---|
| **CVS MAE at clipping** | 在真实 clipping 发生时刻，模型预测的 CVS score 与 ground truth 的绝对误差 |
| **Early Warning Quality** | 在 clipping 前 k 秒窗口内，模型 CVS 预测的时序一致性（是否稳定反映 CVS 达标/未达标） |

**B2b Ground truth 可用范围：** G2 测试视频中的 ~8 个（同时有 triplet + CVS(C80)），作为临床特定 stress test。

**[v5.3] 为什么重构为 2+1 层：**
1. v5.2 的 B2 仅在 9 个测试视频上评估 CVS，统计稳定性偏弱，容易被审稿人质疑
2. B2a 把 safety 的主故事放在更大的外部测试集（~35 videos）上，统计基础更扎实
3. B2b 把"小样本临床关键窗口"定位为补充强化证据，而非唯一证据
4. B1 回答"模型能否提前预测 clipping"，B2a 回答"模型对 CVS 状态的全程感知是否准确"，B2b 回答"在最关键时刻是否准确"——三者互补
5. 审稿人可以分别判断三个能力——B2a 的统计可信度让审稿人更容易接受

#### C. 次要参考指标（secondary）

| 指标 | 定义 | 意义 |
|---|---|---|
| Dense-step mAP | 所有时间步上的 triplet-group 预测 mAP | 与现有文献可比，但非主指标 |
| Phase Accuracy @Δ | 未来 Δ 秒的 phase 分类准确率 | Phase 预测能力 |
| Instrument Anticipation mAP | 未来 Δ 秒的 instrument 多标签预测 mAP | Instrument 预测能力 |

### 7.2 必须包含的 Baselines

| Baseline | 说明 | 消除什么 confound |
|---|---|---|
| **Copy-Current** | 预测下一步 = 当前步，TTC = ∞ | 暴露 dense-step 的 inflation；在 Change-mAP 上得分为 0 |
| **CholecT50-Only** | 单数据集训练（完整 triplet + phase） | 证明多源有增益 |
| **Naive Multi-Source** | 四数据集联合训练，mask-and-ignore | 证明 structured prior 有增益 |
| **Multi-Source + Label Masking** | 联合训练 + 标签条件 loss（无 prior） | 隔离 prior 的增量贡献 |
| **Multi-Source + Uniform Prior** | 联合训练 + uniform smoothing（等价于 label smoothing） | 证明 structure 比 smoothing 好 |
| **Multi-Source + Self-Distillation** | 联合训练 + EMA teacher | 证明 domain prior 比 model prior 好 |
| **Anticipation Transformer** | 已发表的手术 anticipation 方法（SuPRA 或同类） | 与 anticipation SOTA 对比 |
| **MML-SurgAdapt / SPML 风格** | **[v5.3 新增] 在同类三数据集上做 partial-annotation multi-task learning（复现或适配到我们的数据 protocol）** | **与最新 partial-label multi-task 正面交锋** |
| **SurgFUTR 风格** | **[v5.3 新增] state-change learning baseline（将 future prediction 改写为 state-change detection + 跨数据集 benchmark）** | **与最新 state-change future prediction 正面交锋** |
| **Full SurgCast** | 完整方法 | — |

**[v5.3] 新增两类关键 baseline 的理由：**
- MML-SurgAdapt 已经在 CholecT50+Cholec80+Endoscapes 上发表了 partial-label multi-task learning，不与之对比会被审稿人认为回避了最相关的先前工作
- SurgFUTR 已经把 future prediction 改写为 state-change learning 并建立了跨数据集 benchmark，不与之对比会被认为在和"旧时代的 phase anticipation"比
- 两个 baseline 的复现策略：按 Baseline Reproducibility Ladder（见 7.2.1）分级处理

#### 7.2.1 Baseline Reproducibility Ladder [v5.4 新增]

外部 baseline 的复现质量直接影响对比的公平性和可信度。为每个 baseline 明确标注 reproduction tier：

| Reproduction Tier | 定义 | 适用条件 | 论文中标注 |
|---|---|---|---|
| **Tier A (Exact)** | 使用官方开源代码 + 原始训练协议，仅适配到我们的 data split 和评估指标 | 原文提供完整可运行代码 | 无需特殊标注 |
| **Tier B (Faithful)** | 无官方代码但方法描述完整；核心模块在我们的 backbone/temporal encoder 上忠实复现 | 原文有详细方法描述（网络结构、loss、超参数） | 标注 "faithfully reproduced on our backbone" |
| **Tier C (Style)** | 方法描述不充分或架构差异过大；仅复现核心思想/loss 设计，明确注明为 "style baseline" | 原文缺少关键细节或架构不兼容 | 标注 "-style" 后缀（如 "MML-SurgAdapt-style"） |

**具体 baseline 的预期 tier：**
- MML-SurgAdapt：视代码开源情况，预期 Tier B 或 Tier C
- SurgFUTR：视代码开源情况，预期 Tier B 或 Tier C
- Anticipation Transformer (SuPRA)：预期 Tier A 或 Tier B
- 内部 ablation baselines（Copy-Current 到 Self-Distillation）：均为 Tier A（完全由我们实现）

**关键设计：** 内部 ablation baselines（Copy-Current 到 Self-Distillation）使用 **完全相同的** encoder + temporal transformer 架构，只改变数据源、loss 设计和 prior 类型，确保增益可以被归因到方法而非模型容量。外部 baselines（Anticipation Transformer、MML-SurgAdapt、SurgFUTR）使用原文方法，在我们的数据 protocol 和评估指标上评估，确保对比公平。

### 7.3 结果表设计

**Table 1：Main Results — Action-Change Anticipation（主表，一张表讲完故事）**

**Caption:** "Core action-change results on the fixed Tier-1 CholecT50 test set (~10 videos). All methods are evaluated on the same test videos; rows differ only in training data and method components. Phase Acc evaluated on Tier-3 Cholec80-test (~13 videos). Numbers are mean ± std over 5 seeds."

| Method | Inst-C-mAP | Group-C-mAP | TTC-inst MAE ↓ | TTC-group MAE ↓ | C-index-inst ↑ | C-index-group ↑ | Brier-inst @5s ↓ | Brier-group @5s ↓ | Dense-mAP | Phase Acc |
|---|---|---|---|---|---|---|---|---|---|---|
| Copy-Current | 0.0 | 0.0 | ∞ | ∞ | — | — | — | — | high | — |
| SurgFUTR-style | — | — | — | — | — | — | — | — | — | — |
| MML-SurgAdapt-style | — | — | — | — | — | — | — | — | — | — |
| CholecT50-Only | — | — | — | — | — | — | — | — | — | — |
| + Cholec80 phase | — | — | — | — | — | — | — | — | — | — |
| + Cholec80 tool-presence | — | — | — | — | — | — | — | — | — | — |
| + Cholec80-CVS | — | — | — | — | — | — | — | — | — | — |
| + Endoscapes | — | — | — | — | — | — | — | — | — | — |
| + Label-conditional masking | — | — | — | — | — | — | — | — | — | — |
| + Latent transition | — | — | — | — | — | — | — | — | — | — |
| + Structured prior (static + evidence-gated) | — | — | — | — | — | — | — | — | — | — |
| **+ Context-modulated prior (Full SurgCast)** | **—** | **—** | **—** | **—** | **—** | **—** | **—** | **—** | — | — |

**[v5.3→v5.4] 表格变化：**
- 新增 Inst-C-mAP 列（instrument-set Change-mAP），与 Group-C-mAP 并列为 co-primary
- **[v5.4] TTC MAE 拆分为 TTC-inst MAE 和 TTC-group MAE，对应双事件 hazard heads**
- **[v5.4.1] TTC C-index 拆分为 C-index-inst 和 C-index-group；Brier @5s 拆分为 Brier-inst @5s 和 Brier-group @5s，与 7.1 定义完全对齐**
- 新增 SurgFUTR-style 和 MML-SurgAdapt-style 外部 baseline
- Structured prior 行标注 evidence-gated
- **[v5.4] 新增 defensive table caption，显式标注评估集、seed 数和 tier**

如果每一行都比上一行高，核心命题成立。外部 baseline 的放置位置不影响增量趋势——它们作为独立参照系。

**Table 2：Structured Prior Ablation（消融表）**

**Caption:** "Structured prior ablation on Tier-1 CholecT50 test set (~10 videos). All variants use identical encoder, temporal transformer, and dual hazard heads; only the prior regularization differs. Task-specific prior distributions used by default (categorical for phase, factorized Bernoulli for multi-label tasks). Numbers are mean ± std over 5 seeds."

| Prior Type | Change-mAP | C-index-inst ↑ | C-index-group ↑ | Brier-inst @5s ↓ | Brier-group @5s ↓ |
|---|---|---|---|---|---|
| No prior (mask-and-ignore) | — | — | — | — | — |
| Uniform prior | — | — | — | — | — |
| Static procedure prior | — | — | — | — | — |
| Self-distillation prior | — | — | — | — | — |
| **Static + context-modulated (Ours)** | **—** | **—** | **—** | **—** | **—** |

**Table 3：Safety-Critical Anticipation（B1 + B2a + B2b）[v5.3 重构]**

**Caption:** "Safety-critical anticipation results. B1 (clipping detection) evaluated on Tier-1 CholecT50 test set (~10 videos). B2a (CVS state accuracy) evaluated on Tier-2a test set (~35 videos: Endoscapes-test + Cholec80-test). B2b (CVS-at-clipping) evaluated on Tier-2b (~8 G2 test videos). Numbers are mean ± std over 5 seeds."

| Method | Clip Det. @5s | Clip Det. @10s | Clip FA Rate | CVS C1-AUC (B2a) | CVS C2-AUC (B2a) | CVS C3-AUC (B2a) | CVS MAE@clip (B2b) |
|---|---|---|---|---|---|---|---|
| CholecT50-Only (no CVS) | — | — | — | N/A | N/A | N/A | N/A |
| + Cholec80-CVS only | — | — | — | — | — | — | — |
| + Endoscapes CVS | — | — | — | — | — | — | — |
| + Anatomy-Presence Head | — | — | — | — | — | — | — |
| **Full SurgCast** | **—** | **—** | **—** | **—** | **—** | **—** | **—** |

**评估范围：**
- B1 指标在 **Tier 1 测试集**（~10 CholecT50-test 视频）上评估
- B2a 指标（CVS criterion AUC）在 **Tier 2a 测试集**（~35 videos: Endoscapes-test + Cholec80-test）上评估——**主 CVS 评估**
- B2b 指标（CVS MAE@clip, Early Warning）在 **Tier 2b 测试集**（~8 个 G2 测试视频）上评估——临床 stress test
- Clipping PR-AUC 放 appendix
- **[v5.3] Anatomy-Presence Head 作为一行加入**，验证 bbox → anatomy presence 对 CVS 的间接贡献

**Phase-Conditional Analysis（分析表，可放 appendix）**

按 7 个 phase 分别报告 Change-mAP，分析：
- CVS 标注覆盖的阶段（Preparation + CalotTriangleDissection）是否增益更大
- 不覆盖的阶段是否也有增益（证明 knowledge transfer）

### 7.4 消融实验清单

| 编号 | 消融内容 | 回答的问题 |
|---|---|---|
| A1 | 去掉 phase head | Phase 先验对 change anticipation 有多大帮助？ |
| A2 | 去掉 CVS head (不使用 Cholec80-CVS 和 Endoscapes CVS) | CVS 数据是否有用？ |
| A3 | Structured prior → uniform | Structure 的价值多大？ |
| A4 | Structured prior → self-distillation | Domain prior vs model prior？ |
| A5 | Static prior only (β=0) | Context modulation 是否有用？ |
| A6 | 去掉 latent alignment | Latent forecasting 比直接分类好多少？ |
| **A6'** | **Shared transition → 3 independent MLPs（退化到 v5.1 方案）** | **[v5.2 新增] 参数共享和 horizon conditioning 是否有用？** |
| **A_new** | **Hazard head 去掉 σ_agg input（输入从 516-d 退化到 512-d）** | **[v5.2 新增, v5.3 更新] 转移不确定性对 TTC 预测的增量？** |
| A7 | 去掉 temporal module（单帧 MLP） | 时序建模是否必要？ |
| A8 | Hazard head → MSE regression | Hazard 建模的优势？ |
| A9 | Hazard head → binned classification | Hazard vs 普通分类？ |
| A10a | DINOv3 ViT-B/16 → **LemonFM**（手术域专用，ConvNeXt-L, 1536-d） | **[v5.1 新增] Domain-specific 预训练是否优于通用 SSL？** |
| A10b | DINOv3 ViT-B/16 → DINOv3 ViT-L/16（304M, 1024-d） | **[v5.1 新增] 更大通用 backbone 在 domain gap 下是否有用？** |
| A10c | DINOv3 ViT-B/16 → DINOv2 ViT-B/14 | **[v5.1 新增] DINOv3 vs DINOv2 的增量？** |
| A10d | DINOv3 ViT-B/16 → ImageNet ResNet-50 | 强 backbone 的重要性？ |
| A11 | Change 定义：strict vs group vs debounced | Change 定义的敏感性？ |
| A12 | 序列长度：8 / 16 / 32 | 时序窗口的影响？ |
| A13 | Coverage dropout rate：0 / 0.1 / 0.3 / 0.5 | 最优 dropout 率？ |
| **A14** | **纯共现聚类 vs 语义增强聚类** | **[v4.0 新增] 语义信息是否改善 group 定义？** |
| **A15** | **去掉 Instrument Head** | **[v4.0 新增] 分解标签是否有用？** |
| **A16** | **Cholec80-CVS only vs Endoscapes CVS only vs 两者联合** | **[v4.0 新增] 两套 CVS 标注的互补性？** |
| **A17** | **±Cholec80 tool-presence 对 Instrument Head** | **[v5.0 新增] 跨数据集 instrument 迁移价值？** |
| **A18** | **CVS Head: ordinal regression vs binary BCE vs MSE** | **[v5.0+ 新增] CVS 评分建模方式的影响？** |
| **A19** | **G4 视频（Cholec80∩Endoscapes）的贡献** | **[v5.0 新增] 新增覆盖组是否有价值？** |
| **A20** | **CVS 官方 pipeline（85% 截断+5fps）vs 自行全程 1fps** | **[v5.0+ 新增] 全程 CVS 覆盖对 anticipation 的价值？** |
| **A_σgate** ⏳ | **σ-gated prior → 普通标量 α, β（退化到无 C↔E 连接）** | **[v5.2 新增，可选] σ 调制 prior 强度是否有用？** |
| **A_evidence** | **Evidence-gated prior → uniform KL weight（去掉 evidence-gating）** | **[v5.3 新增] evidence-gating 的增量价值？** |
| **A_src** | **CVS Head 去掉 source embedding（退化到 v5.2 共享 Head）** | **[v5.3 新增] source calibration 是否有用？** |
| **A_anat** | **去掉 Anatomy-Presence Head** | **[v5.3 新增] bbox → anatomy presence 的增量？** |
| **A_K** | **K=20 非均匀 bins → K=15 均匀 bins（退化到 v5.2）** | **[v5.3 新增] 扩展时间范围和非均匀 bins 是否有用？** |
| **A_dualhaz** | **Dual hazard heads → single hazard head（退化到 v5.3 单事件建模）** | **[v5.4 新增] 双事件 TTC 建模是否优于单事件？** |
| **A_xval** | **CholecT50 official k-fold cross-val（action branch only）** | **[v5.3 新增] 补充结果，证明结论非 split-dependent** |

主文放 A1-A5（structured prior 相关）、**A_evidence（evidence-gating，关键消融）**、A6'（transition 参数共享）、**A_new（不确定性→hazard 信号，最关键消融）**、A8-A9（hazard 相关）、A10a（domain-specific backbone）、A16（CVS 来源）、A18（CVS ordinal vs binary）、**A_dualhaz（dual-head vs single-head hazard，验证 co-primary TTC 训练闭环）**。A_σgate 视实现情况决定放主文或 appendix。**A_src、A_anat、A_K、A_xval 放 appendix。** 其余放 appendix。

**A_new 是最具说服力的消融：** 如果去掉 σ_agg 输入显著降低 TTC 指标，直接证明"转移不确定性是 change prediction 的有效信号"——这是 shared transition + heteroscedastic 设计的核心价值命题。

**A10 消融组说明（含 [v5.4] backbone decision threshold）：**

**[v5.4]** A10 同时服务于 backbone decision checkpoint（Section 4.2）：如果 LemonFM 在 validation Group-Change-mAP 上超过 DINOv3-B ≥1.5pp 或在 B2a CVS AUC 上超过 ≥2.0pp，则切换默认 backbone。

LemonFM（`visurg/LemonFM`）是 2025 年发布的手术域专用基础模型，基于 ConvNeXt-Large 架构（1536-d 输出），在 LEMON 数据集（938 小时手术视频，35 种术式，含胆囊切除）上使用 augmented knowledge distillation 预训练。在 Cholec80 phase recognition 上比通用 backbone 提升 +9.5pp Jaccard。

A10 消融组的叙事价值：
- 如果 LemonFM > DINOv3-B → "domain-specific pretraining matters more than general SSL scale"，支持未来用手术域 backbone 进一步提升
- 如果 DINOv3-B ≈ DINOv3-L → 验证了医学影像 domain gap 下 ViT-L 边际收益有限的假设
- 如果 structured prior 在所有 backbone 上带来一致增益 → 方法对 backbone 鲁棒，核心贡献在时序建模
- A10a 放主文（最有信息量），A10b-A10d 放 appendix

LemonFM 消融的工程注意事项：
- LemonFM 输出 1536-d（vs 主方法 768-d），需将 input projection 改为 `Linear(1536→512)`
- LemonFM 使用 ConvNeXt-Large，通过 global average pooling 输出全局特征（无 CLS token），提取方式需相应调整
- 需额外提取一份 HDF5 特征文件（~253K × 1536 × 4 bytes ≈ 1.55 GB）
- 除 input projection 层外，下游架构完全不变，确保消融的公平性

---

## 八、论文结构与叙事

### 8.1 标题

> **SurgCast: Event-Centric Surgical State Forecasting via Procedure-Aware Structured Prior under Heterogeneous Missing Supervision**

备选更简洁版本：

> **Anticipating Surgical Action Changes under Heterogeneous Missing Supervision**

### 8.2 主文结构约束

**严格限制在 9 页内，结构如下：**

| 部分 | 页数 | 内容 |
|---|---|---|
| Abstract | 0.3 | 一段话讲清问题、方法、关键结果 |
| Introduction | 1.5 | 问题动机 → 现有方法的不足 → 本文贡献 |
| Related Work | 1.0 | 定位表 + 与 anticipation / partial-label 文献的差异 |
| Method | 3.0 | Module B-E，以 Figure 2 为核心 |
| Experiments | 2.5 | Table 1 + Table 2 + Table 3 + 定性可视化 |
| Conclusion | 0.5 | |
| Appendix | 不限 | 完整消融、Phase-Conditional 分析、实现细节 |

### 8.3 核心 Figure 设计

**Figure 1（第 1 页，问题设定 + 定位）：**

左半：数据覆盖结构矩阵（7 组覆盖）。横轴是视频 ID（按 G1-G7 分组），纵轴是标签维度（triplet, instrument, phase, tool-presence, CVS-Cholec80, CVS-Endoscapes, anatomy bbox）。用颜色块展示"哪些视频有哪些标签"——突出 7 组的互斥覆盖结构和 99.3% 的 CVS 覆盖率。

右半：一个手术视频的时间线示例，标注 change points、clipping events 和 CVS 状态，直观展示"我们预测的是什么"以及"unsafe transition 是怎么定义的"。

**Figure 2（第 3-4 页，方法架构）：**

完整架构图。左：冻结 DINOv3 ViT-B/16 + Causal Transformer。中：**Shared Transition MLP + Horizon Embedding（Δ={1,3,5,10}）**（突出参数共享和 σ 输出分支）+ Multi-horizon heads（突出 Instrument Head + **Anatomy-Presence Head [v5.3]** + **source-calibrated CVS Head [v5.3]**）。右上：**Dual Discrete-Time Hazard Heads**（**标注 shared trunk + λ_inst / λ_group 双 head，σ_agg ∈ R⁴ 输入，K=20 非均匀 bins [v5.4]**）。右下：Structured Prior Regularization（static prior 查表 + context modulation + **evidence-gating [v5.3]**），展示分解条件分布。

**Figure 3（第 7 页，定性结果）：**

选 1-2 个测试视频，展示：
- 时间线上的 ground truth change points + clipping events
- 模型的 TTC 预测（hazard function 热力图）
- Instrument Head 的 clipper 预测概率曲线
- CVS 状态与 safety alert 的触发时间点
- 与 copy-current / CholecT50-only baseline 的对比

### 8.4 核心叙事（5 句话版本，v5.3 更新）

1. 手术 AI 的实际价值在于预见（anticipation）而非识别（recognition），但现有 anticipation 工作——包括近期的 SuPRA/SWAG（dense future step）和 SurgFUTR（state-change learning）——评估的仍然是 dense per-second 或隐式 state-change 预测，被 temporal inertia 严重 inflate，且不涉及 safety 维度。

2. 我们定义了 action-change anticipation：预测下一次动作变化何时发生（dual discrete-time hazard TTC for instrument-set and triplet-group changes）、变化后的动作集合是什么（multi-label post-change state）、以及变化是否安全（CVS state at critical transition）——并提出了三层 event-centric benchmark（instrument-set / triplet-group / clipping event），消除对单一聚类定义的依赖。Phase-level TTC 作为辅助分析放入 appendix。

3. 这个任务天然需要整合分散在 CholecT50、Cholec80、Cholec80-CVS、Endoscapes 中的 action、workflow、anatomy-safety 信息，但这些数据集标签覆盖不均匀且有视频重叠——与 MML-SurgAdapt 等已有 partial-label 工作不同，我们处理的不是当前帧识别而是 temporal forecasting，且通过 evidence-gated structured prior 而非简单 label masking 利用手术流程的组合约束。

4. 我们提出 SurgCast，核心方法创新聚焦三条主线：(a) 三层 event-centric benchmark 消除 temporal inertia inflation 和聚类依赖；(b) discrete-time hazard TTC + 转移不确定性作为变化点检测的正交信号；(c) evidence-gated structured prior——按证据强弱自适应调节 KL 约束，低支持区域弱约束、高支持区域强约束，不盲信稀疏统计。

5. 实验表明，结构化的跨数据集学习在 event-centric 指标上显著超过单数据集训练、naive multi-task baselines、以及最新的 partial-label（MML-SurgAdapt 风格）和 state-change（SurgFUTR 风格）方法，且 instrument 预测、CVS 监督和 anatomy presence 对 safety-critical anticipation 有明确贡献。

### 8.5 创新点排序与 NeurIPS 卖点（v5.3 重构）

**[v5.3] 从 6 个并列创新点收敛为 3 条主线 + 实现细化。** 避免"很多合理工程拼在一起"的印象，聚焦为一个清楚的机器学习命题。

**三条主线（论文 Introduction 中并列声称的贡献）：**

| 主线 | 贡献 | NeurIPS 卖点 |
|---|---|---|
| **主线 1：问题与 Benchmark** | Action-change anticipation 问题定义 + event-centric 评估（三层 benchmark：instrument-set / triplet-group / clipping event） | 揭示 dense-step 指标被 temporal inertia inflate 的根本缺陷；三层 benchmark 不依赖单一聚类定义 |
| **主线 2：建模** | Multi-label post-change state prediction + discrete-time hazard TTC + uncertainty-informed change detection | 生存分析引入手术 anticipation；转移不确定性→hazard 的信号通道（A_new 直接验证） |
| **主线 3：学习范式** | Evidence-gated structured prior for heterogeneous missing supervision | 按证据强弱自适应使用 prior（不盲信稀疏统计），31 个训练视频的 prior 正则化 277 个视频 |

**实现细化（论文中作为方法细节或消融出现，不与主贡献并列）：**

| 细节 | 角色 | 位置 |
|---|---|---|
| DINOv3 / LemonFM backbone | 消融 A10 回答 backbone 选择 | Method 实现细节 + Ablation |
| Shared transition MLP + horizon conditioning | 消融 A6' 回答参数共享价值 | Method 4.4 |
| Ordinal CVS Head + source calibration | 消融 A18 / A_src | Method 4.4 |
| Cross-dataset instrument transfer (tool-presence) | 消融 A17 | Method 4.4 |
| Anatomy-Presence Head (bbox utilization) | 消融 A_anat | Method 4.4 |
| σ-gated prior strength ⏳ | 可选，消融 A_σgate | Appendix |

---

## 九、数据工程详细实施步骤

### Step 1：视频 ID 注册表（2 天）

| 子任务 | 具体操作 |
|---|---|
| Cholec80 ↔ CholecT50 | 45 个重叠视频通过 VID 编号直接匹配 |
| Cholec80/CholecT50 ↔ Endoscapes | CAMMA mapping_to_endoscapes.json + endoscapes_vid_id_map.csv |
| Cholec80-CVS 关联 | Cholec80-CVS 覆盖全部 80 个 Cholec80 视频，自动继承 canonical ID |
| Split 分配 | CAMMA 推荐策略，确保 test 无泄漏 |
| 输出 | `registry.json` |
| 验证 | Test set 视频不出现在任何 train/val 中 |

### Step 2：CholecT50 预处理（3 天）

| 子任务 | 具体操作 |
|---|---|
| 加载标签 | 解析 50 个 JSON，提取 triplet（100维）、instrument（6维）、verb（10维）、target（15维）、phase |
| **语义嵌入生成** | **[v4.0 新增] 用 sentence-transformers 对 100 个 triplet 名称生成 384-d 嵌入** |
| Triplet-group 聚类 | **混合相似度矩阵（共现 + 语义）→ 层次聚类 → 15-20 个 group** |
| Change point 标注 | 对每帧计算 group-level change 和 instrument-level change，应用 debouncing |
| TTC 目标计算 | **[v5.4] 对每帧 t，分别扫描 instrument-set change 和 group-level change 的下一次发生时间，记录双 TTC targets（ttc_inst, ttc_group）。TTC 值映射到 K=20 非均匀区间 (0,1], (1,2], ..., (9,10], (10,12], ..., (28,30]。扫描范围为原始视频的全局未来帧（不限于 16s 输入窗口），最远 30s；超出 min(30s, remaining_video_time) 标记为右删失** |
| Change point 统计 | 计算每视频 change 数、inter-change interval 分布 |
| **Clipping event 标注** | **[v4.0 新增] 对每帧标注 is_clipping = (clipper==1 or verb_clip==1 or verb_cut==1)** |
| 输出 | 每个视频一个 `.npz`：frames, triplets, instruments, verbs, targets, triplet_groups, phase, ttc_target_inst, ttc_target_group, is_change_inst, is_change_group, is_censored_inst, is_censored_group, is_clipping |

### Step 3：Cholec80 预处理（2.5 天）

| 子任务 | 具体操作 |
|---|---|
| 帧-标签对齐 | 通过原始帧 ID（0, 25, 50, ...）做 phase 标注 → 1fps 对齐 |
| 文件名偏移修正 | 图像 1-indexed（video01_000001.png），标注 0-indexed |
| Phase 标签标准化 | 统一到 coarse 7-phase ontology |
| **Tool-presence 提取** | **[v5.0 新增] 解析 80 个视频的 tool-presence 标注（7-tool 二值，1fps）；映射 6 工具到 CholecT50 instrument 类别（丢弃 SpecimenBag）；为 G4+G6 的 35 个非 CholecT50 视频生成 instrument 标签** |
| 输出 | 每个视频一个 `.npz`：frames, phase_ids, **tool_presence（7维）, instrument_mapped（6维）** |

### Step 3.5：Cholec80-CVS 预处理（1.5 天）[v4.0 新增, v5.0 增强]

**预处理策略声明：**
1. **直接 XLSX 解析**：从原始 `surgeons_annotations.xlsx` 生成覆盖完整 pre-clip/cut 阶段的 1fps 逐帧标签
2. **不使用官方 pipeline 截断**：官方 `annotations_2_labels.py` 丢弃前 85% 并以 5fps 采样（产出 62,760 帧），不适合 anticipation 任务
3. **不使用官方 50/15/15 split**：与 CAMMA combined split 存在大量交叉泄漏，CVS 标注 split 完全由 canonical video ID 决定

| 子任务 | 具体操作 |
|---|---|
| 下载标注 | 从 Figshare 下载 surgeons_annotations.xlsx |
| 时间段 → 逐帧转换 | XLSX 中帧号为原始 25fps → 除以 25 转为秒数 → 映射到 1fps 帧 |
| 帧号偏移对齐 | 注意 Cholec80 图像 1-indexed vs 标注 0-indexed 的偏移 |
| **畸形区间处理** | **[v5.0 新增] 丢弃 3 个 final < initial 的区间（含 VID06/VID48 的 2 个正例行），记录日志；注意 VID06 在 CholecT50-test 中，丢失的正例影响 safety ground truth，需量化** |
| **越界区间截断** | **[v5.0 新增] 截断 63 个超出 ClippingCutting 边界的区间至 ClippingCutting 阶段的起始帧** |
| **三级评分保留** | **[v5.0+] 不立即二值化——保留 0/1/2 三级评分用于 ordinal CVS Head。同时计算 cvs_score = Σ min(c, 1)（取值 0-3）用于 safety evaluation** |
| Phase 范围验证 | 确认标注仅覆盖 Preparation + CalotTriangleDissection 阶段 |
| **Phase ontology 验证** | **对 45 个重叠视频，检查 CholecT50 的 gallbladder-extraction 与 Cholec80 的 GallbladderRetraction 在时间轴上是否对齐** |
| 输出 | 每个视频一个 `.npz`：frames, cvs_c1(0/1/2), cvs_c2(0/1/2), cvs_c3(0/1/2), cvs_score(0-3), has_cvs_label(bool per frame) |
| 验证 | 与 Cholec80 的帧数一致；确认 has_cvs_label 仅在 Preparation + CalotTriangleDissection 阶段为 True；G1 的 3 个 + G4 的 3 个 = 6 个双套 CVS 视频上对比两套 CVS 的一致性（≥1 阈值帧级一致率 >80%） |

### Step 4：Endoscapes 预处理（2.5 天）[v5.3 增加 anatomy 提取]

| 子任务 | 具体操作 |
|---|---|
| CVS 提取 | 从 all_metadata.csv 提取 (C1, C2, C3) 向量 |
| 228 帧缺失处理 | 对 test split 用文件系统列表确定完整帧集 |
| 视频 ID 映射 | Endoscapes public ID → canonical ID |
| vids.txt 解析 | 科学计数法 → float → int |
| **[v5.3] Anatomy-presence 提取** | **从 bbox 标注中提取 5 类解剖结构帧级存在性：gallbladder, cystic_duct, cystic_artery, cystic_plate, hepatocystic_triangle。presence[c]=1 当该帧有类别 c 的 bbox** |
| **[v5.4] Anatomy observation mask 提取** | **生成 anatomy_obs_mask(5维 bool)：该帧有 bbox 标注 → mask=1（不论是否有对应类别的 bbox）；无标注帧 → mask=0。区分 presence=0（structure absent）与 unobserved（no annotation）** |
| 输出 | 每个视频一个 `.npz`：frames, cvs_scores, in_roi, **anatomy_presence(5维), anatomy_obs_mask(5维)** |

### Step 5：Procedure Graph 构建（2 天）

| 子任务 | 具体操作 |
|---|---|
| **⚠ 严格限定训练集** | **[v5.0 修正] 所有 prior 统计仅使用训练集视频，杜绝 val/test 泄漏** |
| 条件分布计算 | P(triplet-group \| phase) 来自 **CholecT50-train 35 videos**（per-group Bernoulli） |
| | P(instrument \| phase) 来自 **CholecT50-train + Cholec80-adjusted-train ~67 videos** |
| | P(phase_{t+1} \| phase_t) 来自 **Cholec80-adjusted-train 36 videos** |
| | **P(cvs_ready \| phase, triplet-group) 来自 CholecT50-train ∩ Cholec80-train 31 videos** |
| Endoscapes 映射 | G1(3) + G4(3) = 6 个双套 CVS 视频验证一致性及 anatomy ↔ CVS 对应关系 |
| **层次平滑** | **[v5.0+] 对 P(triplet-group\|phase)：(phase, group) 格中样本 <5 → fallback 到 marginal P(group)；phase 中总样本 <5 → fallback 到 global P(group)。确保无零概率** |
| **Laplace 平滑** | **[v5.0+] per-group Bernoulli 参数使用 Laplace 平滑 (+1/+2)：p_g = (N_g + 1) / (N + 2)，避免零概率 group** |
| **[v5.3] Evidence-gating 权重表** | **对每个 (phase, group, cvs_level) cell 计算训练集帧数 count → w_evidence = min(1.0, count / N_sufficient)；层次回退阈值标注** |
| 验证 | per-group Bernoulli 参数合理，所有训练中出现的 (phase, group) 对有节点；**确认无零概率格**；**evidence weights 分布合理（检查低支持 cell 占比）** |
| 输出 | `static_prior.pkl` + **`evidence_weights.pkl`** |

### Step 6：DINOv3 特征提取（1-2 天）

| 子任务 | 具体操作 |
|---|---|
| 主方法模型 | DINOv3 ViT-B/16, frozen (`facebook/dinov3-vitb16-pretrain-lvd1689m`) |
| 输入 | 518×518 resize |
| 输出 | 768-d CLS token per frame |
| 总帧数 | **~253K**（Cholec80 184K + CholecT50 增量 ~10K + Endoscapes ~59K） |
| 存储 | 3 个 HDF5 文件（cholec80.h5, cholect50.h5, endoscapes.h5）；注：45 个重叠视频的帧在 cholec80.h5 中已包含，cholect50.h5 仅存 5 个独有视频 |
| 消融 backbone | 同时提取 LemonFM（1536-d, GAP）特征，存为独立 HDF5 文件组；DINOv2 ViT-B/14 和 ResNet-50 特征在消融阶段按需提取 |
| 验证 | 特征维度正确，帧数与 registry 一致 |
| 依赖 | `transformers>=4.56.0`（DINOv3 支持）；LemonFM 权重从 `visurg/LemonFM` 下载 |

### Step 7：DataLoader 实现（2 天）

| 子任务 | 具体操作 |
|---|---|
| SequenceDataset | 从 HDF5 读取特征，滑动窗口切序列 |
| CoverageAwareSampler | 按 7 组比例采样：G1(5%)/G2(35%)/G3(5%)/G4(4%)/G5(3%)/G6(23%)/G7(25%) **[v5.3 调整]** |
| CoverageDropout | 最高覆盖样本以 0.3 概率 mask 一个维度 |
| Collate function | 输出：features, labels（含 instrument/verb）, visibility_masks, ttc_targets_inst, ttc_targets_group, censoring_flags_inst, censoring_flags_group, cvs_targets, cvs_source_ids, anatomy_presence, anatomy_obs_mask, is_clipping **[v5.4 新增 dual TTC targets, anatomy_obs_mask]** |
| 验证 | 一个 batch 的所有字段形状和数值范围正确 |

---

## 十、时间线（10 周）

### Phase 0：数据基础设施（Week 1-2）

| 天 | 任务 | 产出 | 里程碑 |
|---|---|---|---|
| D1-D2 | 视频 ID 注册表 + **Cholec80-CVS 下载** | registry.json | |
| D3-D5 | CholecT50 预处理 + **语义嵌入 + 混合聚类** + change point 标注 | 50 个 npz + group定义 + change统计 | **⚠ 立即检查 change density** |
| D6-D7.5 | Cholec80 预处理（**含 tool-presence 提取和 6 工具映射**）+ **Cholec80-CVS 预处理（含畸形区间处理）+ Phase ontology 验证** | 80 个 npz + CVS 逐帧标签 + instrument_mapped | |
| D8-D9 | Endoscapes 预处理 + **6 个重叠视频 CVS 一致性验证** + **[v5.3] bbox → anatomy-presence 标签提取** | ~201 个 npz（含 anatomy_presence 字段） | |
| D10-D11 | Procedure graph + static prior（**含分解分布和 CVS 先验**）+ **[v5.3] evidence-gating 权重表** | static_prior.pkl + evidence_weights.pkl | |
| D12-D12.5 | DINOv3 + LemonFM 特征提取 | 3+3 个 HDF5 | |
| D13-D14 | DataLoader + CoverageAwareSampler | 完整数据管线 | **M1: 数据管线端到端运行** |

### Phase 1：核心模型（Week 3-4）

| 天 | 任务 | 产出 | 里程碑 |
|---|---|---|---|
| D15-D17 | Causal Transformer + multi-horizon heads（**含 Instrument Head + [v5.3] Anatomy-Presence Head + source-calibrated CVS Head**） | model.py | |
| D18-D19 | Latent transition module | transition.py | |
| D20-D21 | Discrete-time hazard head | hazard.py | |
| D22-D23 | 训练循环 + 标签条件 loss | train.py | |
| D24-D25 | 训练 Baseline 1：CholecT50-only | 结果 | |
| D26-D27 | 训练 Baseline 2：+ Cholec80 + Cholec80-CVS | 结果 | |
| D28 | 训练 Baseline 3：+ Endoscapes | 结果 | **M2: 增量趋势验证** |

**⚡ Go/No-Go 检查点（Week 4 末）：**
如果增量表的 Change-mAP 没有递增趋势，停下来检查数据管线。如果数据管线正确但增量为负，重新评估问题设定。

**⚡ Backbone Decision Checkpoint（Week 4 末同步进行）：** [v5.4] 在 validation set 上比较 LemonFM 和 DINOv3-B。如果 LemonFM 在 validation Group-Change-mAP 上超过 DINOv3-B ≥1.5pp，或在 B2a validation CVS AUC 上超过 ≥2.0pp，则 LemonFM 成为最终 NeurIPS 提交的默认 backbone；否则 DINOv3-B 保持默认。

**⚡ Safety Ground Truth 检查（Week 4 末同步进行）：**
分别统计 B1、B2a、B2b 的 ground truth 事件数：B1 在全部 CholecT50 视频上统计 clipping events 数量；B2a 在 Endoscapes-test + Cholec80-test (~35 videos) 上统计 CVS 标注帧数；B2b 在 G2 测试视频 (~8 videos) 上统计 clipping 时刻有 CVS 标注的帧数。如果 B1 的 clipping events 少于 30 个，Table 3 的 B1 列需降级；**B2a 基于 ~35 个视频，统计稳定性不再是瓶颈**；如果 B2b 的 CVS-at-clipping 帧数不足，B2b 改为定性展示（不影响 B2a 主结果）。

### Phase 2：核心创新（Week 5-6）

| 天 | 任务 | 产出 | 里程碑 |
|---|---|---|---|
| D29-D30 | Structured prior (static only) + **[v5.3] evidence-gating** 实现 | prior.py | |
| D31-D32 | Context-modulated prior 实现 | 集成到 train.py | |
| D33 | Coverage dropout 实现 | 集成到 dataloader | |
| D34-D36 | 完整 SurgCast 训练 + 超参数调整 | 完整模型结果 | |
| D37-D38 | Event-centric evaluation 代码（**[v5.3] 含 instrument-set change co-primary 指标**） | evaluate.py | |
| D39-D40 | Safety-critical evaluation 代码（**B1 + [v5.3] B2a on ~35 videos + B2b on ~8 videos**） | safety 结果 | |
| D41-D42 | 小规模 prior ablation（验证 prior + evidence-gating 有效性） | Static vs uniform vs no-prior vs ±evidence-gating | **M3: Prior 有效性验证** |

**⚡ Go/No-Go 检查点（Week 6 中）：**
如果 structured prior 相比 mask-and-ignore 的增益 < 1.5 个点（Change-mAP），考虑降低 prior 在论文中的权重，退守到 event-centric forecasting + hazard TTC 作为主贡献。

### Phase 3：消融 + 分析（Week 7-8）

| 天 | 任务 | 产出 |
|---|---|---|
| D43-D45 | 主文消融 A1-A5, A_evidence, A8-A9, A16（每个约 4 小时训练） | Table 2 完整 |
| D46-D47 | **[v5.3] 外部 baseline 训练：MML-SurgAdapt-style + SurgFUTR-style** | Table 1 外部 baseline 行 |
| D48-D49 | Appendix 消融 A6-A7, A10b-A10d, A11-A15, A_src, A_anat, A_K（A10b/c/d 需额外特征提取） | 补充材料 |
| D50 | **[v5.3] CholecT50 official cross-val（action branch, A_xval）** | Appendix 补充表 |
| D51-D52 | Phase-Conditional Analysis | 分析表 |
| D53-D54 | 定性可视化（Figure 3） | Hazard 热力图 + Instrument 预测曲线 + 时间线 |
| D55 | Safety-critical 结果整理 | Table 3 |
| D56 | 所有结果 double-check | 最终数据 |

**M4: 所有实验数据收集完毕**

### Phase 4：论文撰写（Week 9-10）

| 天 | 任务 | 产出 |
|---|---|---|
| D57-D58 | Figure 1 (问题设定) + Figure 2 (架构) | 2 张核心 Figure |
| D59-D61 | Method section | 3 页 |
| D62-D63 | Experiments section | 2.5 页 |
| D64-D65 | Introduction + Related Work | 2.5 页 |
| D66-D67 | Abstract + Conclusion + Appendix | 完整初稿 |
| D68-D70 | 内部审阅 + 修改 + 提交 | **M5: Final submission** |

---

## 十一、风险与缓解

| 风险 | 概率 | 影响 | 缓解策略 |
|---|---|---|---|
| **Structured prior 增益 marginal (<2 pts)** | 中高 | 高 | Week 6 检查点决策；退守 event-centric + hazard TTC 作为主贡献；**[v5.3] evidence-gating 降低了盲信稀疏 prior 的风险，可能改善增益** |
| **Change-mAP 所有方法极低 (<5%)** | 中 | 中 | 调整 change 定义（用 debounced），允许 ±1s 容差，报告多个阈值；**[v5.3] instrument-set change 作为 co-primary 提供更干净的指标** |
| **TTC 被 label flicker 噪声拖死** | 中 | 高 | Group-level change 而非 strict change；debounced change 作为备选；**[v5.3] K=20 非均匀 bins 扩展覆盖范围，减少右删失比例** |
| **Cholec80-CVS 与 Endoscapes CVS 不一致** | 中 | 中→低 | **[v5.3] source-specific calibration 直接缓解标注协议差异；** 在 6 个双套 CVS 视频上量化一致性；如果严重不一致，分别使用两套 CVS |
| **Phase ontology 对齐失败** | 低 | 高 | 在 45 个重叠视频上逐 phase 检查时间轴对齐；必要时退化为 6-phase mapping |
| **Hazard head 训练不稳定** | 低 | 中 | 预训练其他 heads 先，hazard head 后接入；学习率分层 |
| **审稿人认为贡献不够** | 中→低 | 高 | **[v5.3] 三条主线聚焦 + MML-SurgAdapt/SurgFUTR baseline 正面对比；不再堆叠 6 个并列创新** |
| **审稿人质疑 benchmark 人为构造** | 中→低 | 中 | **[v5.3 新增] 三层 benchmark 中 instrument-set change 完全不依赖聚类，消除"全靠 clustering 定义"的质疑** |
| **DINOv3 在手术图像上存在 domain gap** | 中 | 中 | **[v5.3] LemonFM 升级为 must-run baseline（非可选消融）；** 如果 LemonFM 显著优于 DINOv3，切换主方法 backbone |
| **Unsafe transition 事件数不足** | 低（已有45个视频） | 中 | Week 4 末统计；如果 < 30 个事件，Table 3 改为定性展示 |
| **CVS 评估统计稳定性不足** | 中→低 | 中 | **[v5.3] B2a 在 ~35 个测试视频上评估（vs v5.2 的 9 个），统计稳定性大幅提升** |
| **Cholec80 tool-presence 与 CholecT50 instrument 标签语义差异** | 中 | 低 | 消融 A17 量化差异影响；如差异大则仅用 CholecT50 的 50 个视频 |
| **CVS 正例稀疏导致 CVS Head 训练信号不足** | 中高 | 中 | 使用 ordinal regression + source calibration；利用 Endoscapes CVS 补充正例；**Anatomy-Presence Head 提供互补解剖信号** |
| **Cholec80-CVS 自行解析与官方 pipeline 结果不一致** | 低 | 低 | 在 6 个 Endoscapes 重叠视频上对比；消融 A20 量化差异 |
| **Group-set change 频率过高或过低** | 中 | 中 | Phase 0 Day 3-5 实测后调整 group 数量；如 <1.5/min 则降低 group 数至 10-12；**instrument-set change 作为 co-primary 降低对 group 频率的依赖** |
| **外部 baseline 无法复现** | 中 | 中 | **[v5.4] 按 Baseline Reproducibility Ladder（Section 7.2.1）分三级处理：Tier A（官方代码）→ Tier B（忠实复现）→ Tier C（风格 baseline，明确标注）** |
| **CholecT50 single-split 结果被质疑** | 低 | 低 | **[v5.3 新增] 补充 CholecT50 official cross-val 结果（action branch only），放 appendix** |

### 退守计划

如果 structured prior 失效（M3 检查点判定：增益 < 1.5 个点 Change-mAP），**不犹豫，直接切换到 Fallback 版本。**

**Fallback 论文标题：**
> Anticipating Surgical Action Changes: Hazard-Based Forecasting under Multi-Source Supervision

**Fallback 贡献重排：**

| 优先级 | 贡献 | 对应主文位置 |
|---|---|---|
| 主贡献 1 | Action-change anticipation 问题定义 + event-centric 评估协议 | Introduction + Section 3 |
| 主贡献 2 | Discrete-time hazard modeling 用于 TTC 预测 | Section 4（升级为方法核心） |
| 主贡献 3 | Leakage-safe multi-dataset protocol（4 数据集）+ 多源训练增益验证 | Section 3 + Section 5 |
| 降级内容 | Structured prior 作为 exploratory ablation | Appendix 或 Section 5 的一个子段落 |

**预估 Fallback 中稿率：** NeurIPS 20-30%（仍可投 ICLR 2027 或 MICCAI 2026 作为 Plan B）

---

## 十二、期刊拓展路径

### NeurIPS → TMI

| 拓展内容 | 新增数据 | 时间 |
|---|---|---|
| CholecTrack20 tool persistence memory | 14 个重叠视频的 trajectory duration | 1.5 周 |
| CholecInstanceSeg instance-level reasoning | Fine tool geometry | 1 周 |
| 跨术式迁移（AutoLaparo hysterectomy） | 21 个视频 | 1 周 |
| PhaKIR / HeiChole 多中心泛化验证 | 8+33 个视频 | 1 周 |
| 更完整的 clinical safety evaluation | 扩展 CVS 分析 + 两套 CVS 一致性研究 | 3 天 |
| Label embedding space 方案 | Verb/target 语义融入预测空间 | 1 周 |
| LLM 知识增强 prior（探索性） | 手术教科书知识 → soft constraint | 1 周 |
| 全量消融 | 完整覆盖 | 2 周 |

### TMI → Nature 子刊

| 额外需求 | 说明 |
|---|---|
| 全部数据集使用 | 包括 GraSP 跨平台预训练 |
| 临床专家评估 | 2-3 名外科医生 Likert-scale 定性评价 |
| 更大的 claim | "分散标注的手术数据生态系统可以被系统整合" |
| Clinical impact narrative | Safety-Critical Forecasting 的临床意义解读 |
| 预估额外时间 | 3-4 个月 |

---

## 十三、最终 Checklist

### 开工前必须确认

- [ ] DINOv3 ViT-B/16 权重已下载并验证（`facebook/dinov3-vitb16-pretrain-lvd1689m`，需 `transformers>=4.56.0`）
- [ ] LemonFM 权重已下载并验证（`visurg/LemonFM`，用于消融 A10a）
- [ ] 四个数据集的本地路径与数据审计报告一致（CholecT50, Cholec80, Endoscapes, Cholec80-CVS）
- [ ] Cholec80-CVS 的 surgeons_annotations.xlsx 已从 Figshare 下载
- [ ] sentence-transformers (all-MiniLM-L6-v2) 已安装并可本地运行
- [ ] CAMMA overlap mapping 文件已获取
- [ ] 2×A100 机器环境配置完成（PyTorch, HDF5, etc.）
- [ ] Git 仓库已建立，实验 logging 框架已搭建（W&B 或 TensorBoard）
- [ ] **已确认不使用 Cholec80-CVS 官方预处理 pipeline 的 85% 截断**
- [ ] **已确认不使用 Cholec80-CVS 官方 50/15/15 split**
- [ ] **已阅读审计报告 Section 3.2（CVS 详细统计）和 Section 3.3（CholecT50 标注格式）**
- [ ] **[v5.3] 已确认 Endoscapes bbox 标注中的 anatomy 类别列表（验证 5 类是否覆盖）**
- [ ] **[v5.3] 已检查 MML-SurgAdapt / SurgFUTR 是否有开源代码可用于 baseline 复现**
- [ ] **[v5.3] 已确认 CholecT50 official cross-val protocol（CAMMA 仓库 k-fold 分法）**
- [ ] **[v5.4] 已确认 baseline 复现策略（Tier A/B/C ladder）并记录每个 baseline 的 reproduction tier**

### 关键里程碑

- [ ] **M1 (Week 2 末)：** 数据管线端到端运行，一个 batch 输出正确
- [ ] **M2 (Week 4 末)：** 增量表前 4 行有数字，趋势正确 → Go/No-Go；Safety 事件数统计完成
- [ ] **M3 (Week 6 中)：** Structured prior 小规模验证 → Go/No-Go
- [ ] **M4 (Week 8 末)：** 所有实验数据收集完毕
- [ ] **M5 (Week 10 末)：** 论文提交

### 每次实验必须记录

- [ ] 配置文件（完整超参数）
- [ ] 随机种子
- [ ] 训练 loss 曲线
- [ ] Validation 指标逐 epoch 变化
- [ ] GPU 显存峰值
- [ ] 训练总时间
- [ ] 任何异常现象（NaN、梯度爆炸、指标突变）
