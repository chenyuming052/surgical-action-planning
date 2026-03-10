# SurgCast: 异质缺失监督下的手术动作变化预见

## Surgical Action-Change Anticipation under Heterogeneous Missing Supervision

**版本：v5.1 (Backbone Update)**
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

---

## 〇、一句话定义这篇论文

**在腹腔镜胆囊切除术视频中，预测下一次手术动作变化何时发生、变化后的状态是什么、以及这个变化是否安全——而训练数据来自四个标签覆盖不均匀且有视频重叠的公开数据集。**

---

## 一、研究问题与动机

### 1.1 为什么要做 action-change anticipation

现有手术视频理解的文献主流做两件事：当前帧识别（recognition）和未来状态预测（anticipation）。Recognition 已经被大量工作覆盖（TeCNO、Trans-SVNet、Rendezvous 等）。Anticipation 方向也在快速发展——SuPRA 做 joint recognition and anticipation，SWAG 做 long-term workflow anticipation，还有基于图/超图的短时 action prediction。

但几乎所有现有 anticipation 方法都把问题定义为：**"预测未来第 Δ 秒的动作类别是什么。"**

这个定义有一个根本缺陷：手术视频中，连续多秒的动作往往是相同的。一个 copy-current baseline（直接复制当前动作作为预测）就能在 dense per-second 指标上拿到很高分。这意味着现有指标大量被"temporal inertia"inflate，真正反映预测能力的信号被淹没了。

**我们提出一个更锐利的问题：action-change anticipation。** 不是"未来第 Δ 秒是什么"，而是：

1. **When：** 距离下一次动作变化还有多久？（time-to-next-change, TTC）
2. **What：** 变化之后的状态是什么？（future state after change）
3. **Whether safe：** 这个变化是否与当前解剖安全状态兼容？（safety compatibility）

这三个子问题合在一起构成了一个完整的手术决策支持场景——比单纯的"预测下一秒 triplet"更有临床意义，也更能区分真正有预测能力的模型和 trivial baselines。

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

| 维度 | SuPRA / SWAG 等现有 Anticipation | SurgCast（本文） |
|---|---|---|
| 预测对象 | 未来 phase / 未来 instrument | **动作变化时间 + 变化后状态 + 安全兼容性** |
| 是否显式建模 change time | 否 | **是（discrete-time hazard）** |
| 是否处理 heterogeneous missing labels | 否（单一数据集，标签完整） | **是（四个数据集，标签覆盖不均）** |
| 是否多源训练 | 否或简单多任务 | **是（结构化先验正则化）** |
| 是否 safety-aware | 否 | **是（unsafe transition warning）** |
| 评估是否 event-centric | 否（dense per-second） | **是（change-point mAP + safety detection）** |
| 对缺失标签的处理 | 不涉及 | **Procedure-aware structured prior** |

### 2.2 本文的不同之处

与 SuPRA、SWAG 等工作相比，本文的核心差异不在于 anticipation 本身（预测未来在手术领域已有先例），而在于：

1. **预测目标的重新定义**：从"未来 Δ 秒是什么"转向"下一次变化何时发生、变成什么、是否安全"
2. **训练范式的差异**：在标签异质缺失条件下训练，而非假设标签完整
3. **评估哲学的差异**：event-centric 指标替代 dense-step 指标作为主指标

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

**最终 split 预估：** ~191 train / ~53 val / ~65 test

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
        │         g_Δ(h_t) → ĥ_{t+Δ}, Δ ∈ {1, 3, 5}
        │         │
        │         ├──→ Triplet-Group Head ──→ ŷ_triplet-group     (BCE, multi-hot)
        │         ├──→ Instrument Head ──→ ŷ_instrument            (BCE)
        │         ├──→ Phase Head ──→ ŷ_phase                      (CE)
        │         └──→ Safety/CVS Head ──→ ŷ_cvs                   (Ordinal BCE)
        │
        ├──→ [Module D] Discrete-Time Hazard Head (TTC)
        │         │
        │         └──→ h(k | h_t) = P(T=k | T≥k, h_t), k=1,...,K
        │
        └──→ [Module E] Structured Prior Regularization (核心创新)
                  │
                  └──→ L_prior = KL(p_θ(y_miss | h_t) ‖ q_prior(y_miss | y_obs, h_t))
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

用 1 个共享 MLP 替代 3 个独立 MLP_Δ，通过可学习的 horizon embedding 条件化预测范围：

```
ĥ_{t+Δ} = g(h_t, e_Δ),  Δ ∈ {1, 3, 5}
```

- `e_Δ` = 可学习的 64 维 horizon embedding（每个 Δ 一个，共 3 个）
- 共享 MLP 结构：Linear(576, 512) → GELU → Linear(512, 1024)
- 隐藏层 512 维（与 h_t 维度匹配），输出 1024 维，分裂为两部分：
  - **状态预测**：前 512 维 → ĥ_{t+Δ}（未来状态估计）
  - **对数方差**：后 512 维 → log σ²_{t+Δ}（逐维转移不确定性）
- 参数量：576×512 + 512 + 512×1024 + 1024 ≈ **~820K**（vs 原 3 个独立 MLP ~1.6M，节省 ~780K）

**设计决策——为什么不用迭代 rollout：** 5 步自回归 rollout 会累积误差，需要中间监督或 teacher forcing，增加训练复杂度。Horizon conditioning 以极小风险获得参数共享的绝大部分收益。

**Heteroscedastic 转移不确定性：**

每次前向传播同时预测转移不确定性 σ_{t+Δ}，编码"模型对该未来状态的预测有多不确定"。跨 horizon 聚合为 3 维向量：

```
σ_agg = [√(mean(σ²_{t+1})), √(mean(σ²_{t+3})), √(mean(σ²_{t+5}))]  ∈ R³
```

每个分量为对应 horizon 的 512 维不确定性向量的均方根均值，得到 1 个标量/horizon。

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
| Safety/CVS Head | Linear(512, 6) | 6 维 ordinal CVS 预测 | Ordinal BCE | **[v5.0+] 每 criterion 2 logits: P(≥1), P(≥2)；保留 0/1/2 序数结构** |

**v5.0+ CVS Head 修正（BCE → Ordinal BCE）：**
- **问题**：v5.0 使用 `Linear(512, 3)` + 独立 BCE，将 CVS 0/1/2 三级评分二值化为 {0,1}，丢失了"部分满足 vs 完全满足"的序数信息。审计报告确认 CVS"完全达标"（三准则总分≥5）极其稀少（仅 23 行标注，16 个视频），而"部分满足"信号更丰富——cystic_plate 有 83 个正例行，two_structures 有 287 个正例行。二值化丢弃了这一关键区分。
- **修正**：CVS Head 改为 `Linear(512, 6)`，每个 criterion 输出 2 个 logits，分别对应 P(score ≥ 1) 和 P(score ≥ 2) 的累积概率。
- **Ordinal BCE loss**：`L_cvs = Σ_c Σ_{k∈{1,2}} BCE(σ(logit_{c,k}), 𝟙[score_c ≥ k])`
- **推理时重建**：predicted_score_c = σ(logit_{c,1}) + σ(logit_{c,2})，取值范围 [0, 2]
- **Endoscapes 适配**：Endoscapes CVS 为连续分值，仅激活第一个阈值（≥0.5→1），第二个阈值 loss masked
- **参数量变化**：从 ~55K 增至 ~58K（negligible）

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
2. Instrument change 是比 triplet-group change 更干净的信号（6 个类别 vs 15-20 个 group），可作为辅助 TTC 目标
3. 参数增量极小（~3K 参数）
4. 跨数据集 instrument supervision transfer（Cholec80 tool-presence → CholecT50 instrument）是本文的方法创新之一

注意：Verb Head（10 类）和 Target Head（15 类）不作为独立预测头加入，因为 verb 和 target 与 instrument 有强组合约束，独立 BCE 预测会丢失这些关系。Verb 和 target 的语义信息通过 triplet-group 的语义嵌入聚类方案被隐式利用（见 Section 5.3）。

### 4.5 Module D：Discrete-Time Hazard Head（TTC 建模）

**这是本文在建模层面最有辨识度的组件。**

#### 4.5.1 为什么用 hazard modeling 而不是直接回归或分类

| 方式 | 问题 |
|---|---|
| MSE 回归 | 对分布偏斜敏感，无法处理右截断 |
| Binned 分类 | 丢失序数信息，bin 边界选择敏感 |
| Ordinal regression | 比分类好，但没有显式的生存结构 |
| **Discrete-time hazard** | **天然处理右截断、保留序数结构、直接输出生存函数** |

#### 4.5.2 形式化定义

将未来时间离散化为 K 个 bin：k = 1, 2, ..., K（例如 K=15，每个 bin 对应 1 秒，覆盖未来 15 秒）。

**Hazard function：** 在 h_t 条件下，第 k 个 bin 发生 change 的条件概率：

```
λ(k | h_t) = P(T = k | T ≥ k, h_t) = σ(f_hazard([h_t; σ_agg])_k)
```

其中 f_hazard 是一个 MLP：Linear(515, 256) → GELU → Linear(256, K)，σ 是 sigmoid。输入为 [h_t; σ_agg]（512 维状态 + 3 维转移不确定性），σ_agg 来自 Module C 的 heteroscedastic 输出（见 4.4）。

**不确定性→风险的直觉：** 当转移模型对未来状态确信（低 σ）时，当前动作可能继续；当不确信（高 σ）时，动作变化正在临近。这为 hazard 估计提供了一个正交于状态表征的信号通道。

**Survival function：** 在未来 k 秒内不发生 change 的概率：

```
S(k | h_t) = Π_{j=1}^{k} (1 - λ(j | h_t))
```

**Cumulative incidence：** 在未来 k 秒内发生 change 的概率：

```
F(k | h_t) = 1 - S(k | h_t)
```

#### 4.5.3 训练损失

使用 discrete-time survival 的标准 negative log-likelihood：

对于观察到 change 发生在第 k* 个 bin 的样本：

```
L_hazard = -log λ(k* | h_t) - Σ_{j=1}^{k*-1} log(1 - λ(j | h_t))
```

对于在观察窗口内没有发生 change 的样本（右截断）：

```
L_hazard = -Σ_{j=1}^{K} log(1 - λ(j | h_t))
```

这个损失天然处理了"窗口内没有变化"的情况——不需要人为指定一个 target class，而是把它建模为被截断的观测。

#### 4.5.4 Early Warning 的决策边界

在推理时，可以设置阈值 τ：

```
Alert at time t if F(k_warn | h_t) > τ
```

即"如果模型认为未来 k_warn 秒内发生 change 的概率超过 τ，则发出预警"。这直接给出了一个可调节的 alarm system。

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

**Layer 2：Context-Modulated Prior（可学习，训练中更新）**

在 static prior 基础上，用当前 latent state h_t 做上下文调制：

```
q_prior(y_miss | y_obs, h_t) = softmax(α · log P_static(y_miss | y_obs) + β · g_φ(h_t))
```

其中：
- P_static(y_miss | y_obs) 是 Layer 1 的查找表输出
- g_φ(h_t) 是一个轻量 MLP：Linear(512, 256) → GELU → Linear(256, C)
- α, β 是可学习的标量（初始化 α=1.0, β=0.1，让训练初期以 static prior 为主）；可选扩展为 σ_agg 的线性函数，见 4.6.2.1
- C 是被 mask 维度的类别数

**直觉：** Static prior 告诉模型"在 CalotTriangleDissection 阶段，grasper-retract-gallbladder 是最常见的动作"。Context modulation 进一步告诉模型"但根据当前视觉上下文，grasper-dissect-cystic-duct 的概率应该更高"。

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

**Regularization loss：**

```
L_prior = D_KL(p_θ(y_miss | h_t) ‖ q_prior(y_miss | y_obs, h_t))
```

**为什么这不是 label smoothing：** Label smoothing 用 uniform distribution，与上下文无关。我们的 prior 是 conditioned on observed labels and current latent state，分布形状因样本而异。

**为什么这不是 self-distillation：** Self-distillation 的 teacher 信号来自模型自身（EMA），可能放大模型偏差。我们的 prior 来自手术流程的客观统计结构，不依赖模型质量。

#### 4.6.2.1 σ-Gated Prior Strength（C↔E 轻量连接）⏳

> **⏳ 优先级：可选扩展。** 仅在核心实验（Phase 2-3）完成且有富余时间时实现。若时间不足，保持 α, β 为普通可学习标量即可，不影响主方法完整性。

**动机：** Module C 的转移不确定性 σ_agg 编码了"当前状态是否临近变化点"。这个信号对 Module E 的 prior 调制有天然的指导意义：当转移不确定（临近变化点）时，模型对下一状态的分类不确定，应更多依赖结构化 prior；当转移确定（动作稳定执行中）时，上下文足以准确预测，prior 权重可降低。

**改动：** 将 Layer 2 的标量 α, β 替换为 σ 的线性函数：

```
q_prior(y_miss | y_obs, h_t) = softmax(α(σ_t) · log P_static + β(σ_t) · g_φ(h_t))

α(σ_t) = α_0 + α_1 · σ̄_t
β(σ_t) = β_0 + β_1 · σ̄_t
```

其中 σ̄_t = mean(σ_agg) 是 3 维 σ_agg 的标量均值，α_0, α_1, β_0, β_1 为 4 个可学习参数（替代原来的 2 个）。

**初始化：** α_0 = 1.0, α_1 = 0.5, β_0 = 0.1, β_1 = -0.05。使得高不确定性时 α 增大（prior 权重上升）、β 减小（context modulation 权重下降）。

**叙事价值：** 这在 Module C（dynamics）和 Module E（prior）之间建立了有原则的信息流——转移模型的不确定性估计直接调节 prior 强度，而非将两者作为独立 loss 项。叙事从"transition + prior 分别工作"升级为"transition uncertainty 自适应地调控 prior 信任程度"。

**参数与复杂度增量：** 从 2 个标量 → 4 个标量，可忽略。

**消融：** 见 A_σgate（Section 7.4）。

#### 4.6.3 Ontology Bridge

**Phase 对齐：** Cholec80 和 CholecT50 的 phase 名称相似但不完全一致（如 `GallbladderRetraction` vs `gallbladder-extraction`）。使用一个共享的 coarse phase space（7 个 phase），加上数据集特定的映射规则。

**注意：** `GallbladderRetraction`（Cholec80）与 `gallbladder-extraction`（CholecT50）仅为近似语义映射，不是精确对应。在 Phase 0 中需对 CholecT50 的 45 个重叠视频做显式验证——检查这两个标签在时间轴上是否真的对齐。

**CVS 对齐（Ordinal 版本）：** CVS Head 使用 ordinal regression，每个 criterion 输出 P(≥1) 和 P(≥2) 两个累积概率。两个数据源的对齐方案：
- **Cholec80-CVS**：原始 0/1/2 三级评分直接映射到两个累积阈值——score ≥ 1 激活第一阈值，score ≥ 2 激活第二阈值。两个阈值的 loss 均参与训练。
- **Endoscapes CVS**：连续评分仅激活第一个阈值（C1/C2/C3 ≥ 0.5 → 1），第二个阈值 loss masked（Endoscapes 无 0/1/2 级别区分）。
- **一致性验证**：在 G1(3) + G4(3) = 6 个双套 CVS 视频上，要求二值化（≥1）阈值的帧级一致率 >80%。
- **推理统一**：predicted_score_c = σ(logit_{c,1}) + σ(logit_{c,2})，取值 [0, 2]，两个数据源共享同一个 CVS Head

### 4.7 总损失函数

```
L_total = L_task + λ_align · L_align + λ_hazard · L_hazard + λ_prior · L_prior
```

各项定义：

```
L_task = m_tri · L_triplet + m_inst · L_instrument + m_pha · L_phase + m_cvs · L_cvs
    (标签条件化多任务损失，m ∈ {0,1} 是可见性 mask)
    (L_triplet 和 L_instrument 为 BCE loss；L_cvs 为 ordinal BCE loss；L_phase 为 CE loss)
    (v5.0 修正：L_triplet 从 CE 改为 BCE，因 triplet-group 为 multi-hot 多标签预测)

L_align = Σ_Δ ‖ĥ_{t+Δ} - sg(h_{t+Δ})‖²
    (Latent transition alignment)

L_hazard = 离散时间生存损失（见 4.5.3）

L_prior = D_KL(p_θ(y_miss | h_t) ‖ q_prior(y_miss | y_obs, h_t))
    (Structured prior regularization，仅在 coverage dropout 激活时计算)
```

**超参数初始值：** λ_align = 0.5, λ_hazard = 1.0, λ_prior = 0.3

### 4.8 模型规模

| 组件 | 参数量 | 说明 |
|---|---|---|
| Input projection (768→512) | ~400K | |
| Causal Transformer (6 layers) | ~9.5M | |
| **Shared Transition MLP** | **~820K** | **[v5.2] Linear(576,512)+Linear(512,1024)，替代 3 个独立 MLP (~1.6M)，节省 ~780K** |
| Horizon embeddings (×3) | 192 | **[v5.2] 3 × 64-d 可学习 embedding** |
| Prediction heads (triplet-group + instrument + phase + CVS ordinal) | ~58K | |
| Hazard head | ~141K | **[v5.2] 输入 515→256→15，+~1.5K** |
| Prior modulation MLP | ~140K | |
| Context α, β (⏳ σ-gated: α_0, α_1, β_0, β_1) | 2 (⏳ 4) | **[v5.2] 可选扩展为 σ 的线性函数，见 4.6.2.1** |
| **总计** | **~11.2M trainable** | **[v5.2] 比 v5.1 减少 ~780K 参数** |

加上冻结的 DINOv3 86M 参数，推理时总参数 ~97M，但只有 ~11.2M 需要训练。在 2×A100 40GB 上训练毫无压力。参数量减少的同时引入了更强的归纳偏置（跨 horizon 参数共享 + 不确定性估计）。

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
| **Group-level change（主任务）** | multi-hot group 向量的 Hamming 距离 > 0（任一 group 激活状态翻转） | 中等 | **主要 TTC 目标** |
| Debounced change | group-level 变化后持续 ≥ 3 秒才确认 | 较低 | 消融对比，验证 debouncing 的影响 |

**v4.0 新增辅助 change 定义：Instrument-level change。** 当 instrument 的多标签向量发生变化（有新器械出现或旧器械消失）即为一次 instrument change。因为只有 6 个类别，信号比 triplet-group change 更干净。用作辅助 TTC 目标，不作为主评估指标。

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
| G1 三重交叉 | 3 | L_triplet + L_instrument + L_phase + L_cvs(C80) + L_cvs(Endo) + L_hazard + L_prior | 5% | 3 |
| G2 CholecT50∩Cholec80 | 42 | L_triplet + L_instrument + L_phase + L_cvs(C80) + L_hazard + L_prior（30%概率） | 28% | 18 |
| G3 CholecT50∩Endoscapes | 3 | L_triplet + L_instrument + L_phase + L_cvs(Endo) + L_hazard | 5% | 3 |
| G4 Cholec80∩Endoscapes | 3 | L_phase + L_instrument(tool) + L_cvs(C80) + L_cvs(Endo) | 4% | 3 |
| G5 CholecT50独有 | 2 | L_triplet + L_instrument + L_phase + L_hazard | 3% | 2 |
| G6 Cholec80独有 | 32 | L_phase + L_instrument(tool) + L_cvs(C80) | 15% | 10 |
| G7 Endoscapes独有 | 192 | L_cvs(Endo) | 40% | 25 |

**v5.0 覆盖率亮点：**
- 激活 L_cvs 的样本：G1 + G2 + G3 + G4 + G6 + G7 = **97%**（275/277 视频有 CVS）
- 激活 L_instrument 的样本：G1 + G2 + G3 + G4 + G5 + G6 = **60%**（85/277 视频有 instrument 标注）
- G6 的激活损失从 v4 的 `L_phase + L_cvs` 改为 `L_phase + L_instrument(tool) + L_cvs`（新增 Cholec80 tool-presence）

### 6.1.1 类别不平衡处理

**Triplet-Group Head（BCE）：** 审计报告确认 triplet 分布严重长尾（top 3 占 55.67%），group 层面可能同样不均衡。训练中对 L_triplet-group 使用 per-group `pos_weight`，权重与该 group 正例频率成反比：`pos_weight_g = (N - N_g) / N_g`，其中 N_g 为训练集中 group g 为正的帧数。消融中对比 uniform BCE vs pos_weight BCE vs Focal loss（γ=2）。

**Instrument Head（BCE）：** 器械分布极不均衡——clipper 出现率约 2%，而 grasper/hook 出现率 >80%。使用 per-class `pos_weight`，确保低频但 safety-critical 的 clipper 获得足够梯度信号。

**CVS Head（Ordinal BCE）：** 不额外加权。CVS 正例稀疏反映了临床现实（CVS 在大部分 dissection 过程中确实不达标），强行上采样正例会扭曲先验分布。ordinal regression 已经通过保留 0→1→2 的序数信息缓解了二值化带来的信号丢失。

**Phase Head（CE）：** 不额外加权，各 phase 分布相对均匀。

### 6.2 序列构造

对每个视频，以 stride=8 的滑动窗口切出长度为 16 的子序列。每个子序列是一个训练样本。

TTC 目标的计算：对每个时间步 t，从 t+1 开始扫描 triplet-group 序列，找到第一个 group-level change 的位置，记为 TTC target。如果在序列末尾仍未发生 change，标记为右截断。

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
| Hazard time bins K | 15 (覆盖未来 15 秒) |
| GPU | 2 × A100 40GB |
| 预估训练时间 | ~6 小时 / 100 epochs |

---

## 七、评估方案

### 7.1 主指标体系

#### A. Action-Change Anticipation（核心指标组）

| 指标 | 定义 | 意义 |
|---|---|---|
| **Change-mAP** | 仅在 group-level change point 上评估的 triplet-group 多标签预测 mAP（per-group AP 再平均） | 消除 temporal inertia inflation |
| **TTC MAE** | time-to-next-change 的平均绝对误差（秒） | TTC 预测精度 |
| **TTC C-index** | concordance index，衡量 TTC 排序一致性 | 区分力（是否能正确区分"快变"和"慢变"） |
| **TTC Brier Score** | 对"未来 k 秒内是否发生 change"的概率预测的 Brier score，k ∈ {3, 5, 10} | 概率校准质量（hazard head 的核心卖点） |
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

**子任务 B2：CVS State Accuracy at Critical Moments**

在已知 clipping 将要发生的时间点上，评估模型对 CVS 状态的估计准确性。

评估指标：

| 指标 | 定义 |
|---|---|
| **CVS criterion-wise AUC** | 对每个 criterion 独立评估二分类 AUC（≥1 阈值） |
| **CVS MAE at clipping** | 在真实 clipping 发生时刻，模型预测的 CVS score（ordinal 重建值 0-2）与 ground truth 的绝对误差 |
| **Early Warning Quality** | 在 clipping 前 k 秒窗口内，模型 CVS 预测的时序一致性（是否稳定反映 CVS 达标/未达标） |

**B2 Ground truth 可用范围：** G1 + G2 共 45 个视频有 triplet + CVS(C80)。测试集为 CholecT50-test 中的 9 个视频（VID111 属 G5，无 CVS ground truth，排除）。论文中明确注明此限制。

**为什么分解而非合并：**
1. 合并定义（unsafe = clipping + CVS 不达标）导致近乎退化的二值判定（CVS 几乎从未完全达标 → ~100% 正例率 → False Alarm Rate 无意义）
2. 分解后两个子任务各自有意义：B1 回答"模型能否提前预测 clipping"，B2 回答"模型对 CVS 状态的感知是否准确"
3. 分解允许在没有 CVS 标签的视频上也能评估 B1（所有 CholecT50 视频都可以评 B1，测试集从 9 扩展到 10 个视频）
4. 审稿人可以分别判断两个能力——这对 NeurIPS 审稿尤其重要

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
| **Anticipation Transformer** | 已发表的手术 anticipation 方法 | 与领域 SOTA 对比 |
| **Full SurgCast** | 完整方法 | — |

**关键设计：** 所有 baselines 使用 **完全相同的** encoder + temporal transformer 架构，只改变数据源、loss 设计和 prior 类型。这确保增益可以被归因到方法而非模型容量。

### 7.3 结果表设计

**Table 1：Main Results — Action-Change Anticipation（主表，一张表讲完故事）**

| Method | Change-mAP | TTC MAE ↓ | TTC C-index ↑ | Brier @5s ↓ | Dense-mAP | Phase Acc |
|---|---|---|---|---|---|---|
| Copy-Current | 0.0 | ∞ | — | — | high | — |
| CholecT50-Only | — | — | — | — | — | — |
| + Cholec80 phase | — | — | — | — | — | — |
| + Cholec80-CVS | — | — | — | — | — | — |
| + Endoscapes | — | — | — | — | — | — |
| + Label-conditional masking | — | — | — | — | — | — |
| + Latent transition | — | — | — | — | — | — |
| + Structured prior (static) | — | — | — | — | — | — |
| **+ Context-modulated prior (Full SurgCast)** | **—** | **—** | **—** | **—** | — | — |

如果每一行都比上一行高，核心命题成立。

**Table 2：Structured Prior Ablation（消融表）**

| Prior Type | Change-mAP | TTC C-index | Brier @5s ↓ |
|---|---|---|---|
| No prior (mask-and-ignore) | — | — | — |
| Uniform prior | — | — | — |
| Static procedure prior | — | — | — |
| Self-distillation prior | — | — | — |
| **Static + context-modulated (Ours)** | **—** | **—** | **—** |

**Table 3：Safety-Critical Anticipation（B1 + B2）**

| Method | Clip Det. @5s | Clip Det. @10s | Clip FA Rate | CVS C1-AUC | CVS C2-AUC | CVS C3-AUC | Early Warning |
|---|---|---|---|---|---|---|---|
| CholecT50-Only (no CVS) | — | — | — | N/A | N/A | N/A | N/A |
| + Cholec80-CVS only | — | — | — | — | — | — | — |
| + Endoscapes CVS | — | — | — | — | — | — | — |
| **Full SurgCast** | **—** | **—** | **—** | **—** | **—** | **—** | **—** |

B1 指标（Clip Det., Clip FA）在 CholecT50-test 全部 10 个视频上评估；B2 指标（CVS AUC, Early Warning）在 9 个测试视频上评估（排除 VID111）。Clipping PR-AUC 和 CVS MAE@clip 放 appendix。

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
| **A_new** | **Hazard head 去掉 σ_agg input（输入从 515-d 退化到 512-d）** | **[v5.2 新增] 转移不确定性对 TTC 预测的增量？** |
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

主文放 A1-A5（structured prior 相关）、A6'（transition 参数共享）、**A_new（不确定性→hazard 信号，最关键消融）**、A8-A9（hazard 相关）、A10a（domain-specific backbone）、A16（CVS 来源）、A18（CVS ordinal vs binary）。A_σgate 视实现情况决定放主文或 appendix。其余放 appendix。

**A_new 是最具说服力的消融：** 如果去掉 σ_agg 输入显著降低 TTC 指标，直接证明"转移不确定性是 change prediction 的有效信号"——这是 shared transition + heteroscedastic 设计的核心价值命题。

**A10 消融组说明：**

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

完整架构图。左：冻结 DINOv3 ViT-B/16 + Causal Transformer。中：**Shared Transition MLP + Horizon Embedding**（突出参数共享和 σ 输出分支）+ Multi-horizon heads（突出 Instrument Head 的新增）。右上：Discrete-Time Hazard Head（**标注 σ_agg 输入**）。右下：Structured Prior Regularization（static prior 查表 + context modulation），展示分解条件分布。

**Figure 3（第 7 页，定性结果）：**

选 1-2 个测试视频，展示：
- 时间线上的 ground truth change points + clipping events
- 模型的 TTC 预测（hazard function 热力图）
- Instrument Head 的 clipper 预测概率曲线
- CVS 状态与 safety alert 的触发时间点
- 与 copy-current / CholecT50-only baseline 的对比

### 8.4 核心叙事（5 句话版本）

1. 手术 AI 的实际价值在于预见（anticipation）而非识别（recognition），但现有工作评估的是 dense per-second prediction，被 temporal inertia 严重 inflate。

2. 我们定义了 action-change anticipation：预测下一次动作变化何时发生、变化后状态是什么、以及变化是否安全——并提出了以 change point 和 safety window 为中心的 event-centric 评估。

3. 这个任务天然需要整合分散在 CholecT50、Cholec80、Cholec80-CVS、Endoscapes 中的 action、workflow、anatomy-safety 信息，但这些数据集标签覆盖不均匀且有视频重叠。

4. 我们提出 SurgCast，包含两个核心建模创新：(a) procedure-aware structured prior regularization——利用手术流程的组合约束为缺失标签生成上下文自适应的软监督；(b) 带 horizon conditioning 的共享状态转移模型——同时预测未来状态和校准的转移不确定性，该不确定性直接流入 discrete-time hazard head 作为变化点检测的正交信号。

5. 实验表明，结构化的跨数据集学习在 event-centric 指标上显著超过单数据集训练和 naive multi-task baselines，且直接的 instrument 预测和 CVS 监督对 safety-critical anticipation 有明确贡献。

### 8.5 创新点排序与 NeurIPS 卖点（v5.0）

| 优先级 | 创新点 | NeurIPS 卖点 |
|---|---|---|
| **1** | Action-change anticipation 问题定义 + event-centric 评估 | 揭示现有 dense-step 指标被 temporal inertia inflate 的根本缺陷 |
| **2** | Heterogeneous missing supervision 框架 | 7 个覆盖等级、4 个数据源、277 个视频的系统整合 |
| **3** | Discrete-time hazard TTC + **uncertainty-informed change detection** | 生存分析方法论引入手术 anticipation；**转移不确定性作为变化点检测的正交信号**（A_new 消融直接验证） |
| **4** | Structured procedure prior | 31 个训练视频的 prior 正则化 277 个视频的预测 |
| **5** | **Shared latent dynamics model with horizon conditioning** | **参数减少 ~1M，跨 horizon 共享动态知识，heteroscedastic 输出连接状态预测与变化检测** |
| **6** | Cross-dataset instrument supervision transfer | Cholec80 tool-presence 到 CholecT50 instrument 的跨数据集知识迁移 |

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
| TTC 目标计算 | 对每帧 t，扫描后续帧找到下一个 change，记录 TTC |
| Change point 统计 | 计算每视频 change 数、inter-change interval 分布 |
| **Clipping event 标注** | **[v4.0 新增] 对每帧标注 is_clipping = (clipper==1 or verb_clip==1 or verb_cut==1)** |
| 输出 | 每个视频一个 `.npz`：frames, triplets, instruments, verbs, targets, triplet_groups, phase, ttc_target, is_change, is_censored, is_clipping |

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

### Step 4：Endoscapes 预处理（2 天）

| 子任务 | 具体操作 |
|---|---|
| CVS 提取 | 从 all_metadata.csv 提取 (C1, C2, C3) 向量 |
| 228 帧缺失处理 | 对 test split 用文件系统列表确定完整帧集 |
| 视频 ID 映射 | Endoscapes public ID → canonical ID |
| vids.txt 解析 | 科学计数法 → float → int |
| 输出 | 每个视频一个 `.npz`：frames, cvs_scores, in_roi |

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
| 验证 | per-group Bernoulli 参数合理，所有训练中出现的 (phase, group) 对有节点；**确认无零概率格** |
| 输出 | `static_prior.pkl` |

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
| CoverageAwareSampler | 按 7 组比例采样：G1(5%)/G2(28%)/G3(5%)/G4(4%)/G5(3%)/G6(15%)/G7(40%) |
| CoverageDropout | 最高覆盖样本以 0.3 概率 mask 一个维度 |
| Collate function | 输出：features, labels（含 instrument/verb）, visibility_masks, ttc_targets, censoring_flags, cvs_targets, is_clipping |
| 验证 | 一个 batch 的所有字段形状和数值范围正确 |

---

## 十、时间线（10 周）

### Phase 0：数据基础设施（Week 1-2）

| 天 | 任务 | 产出 | 里程碑 |
|---|---|---|---|
| D1-D2 | 视频 ID 注册表 + **Cholec80-CVS 下载** | registry.json | |
| D3-D5 | CholecT50 预处理 + **语义嵌入 + 混合聚类** + change point 标注 | 50 个 npz + group定义 + change统计 | **⚠ 立即检查 change density** |
| D6-D7.5 | Cholec80 预处理（**含 tool-presence 提取和 6 工具映射**）+ **Cholec80-CVS 预处理（含畸形区间处理）+ Phase ontology 验证** | 80 个 npz + CVS 逐帧标签 + instrument_mapped | |
| D8-D9 | Endoscapes 预处理 + **6 个重叠视频 CVS 一致性验证** | ~201 个 npz | |
| D10-D11 | Procedure graph + static prior（**含分解分布和 CVS 先验**） | static_prior.pkl | |
| D12-D12.5 | DINOv3 + LemonFM 特征提取 | 3+3 个 HDF5 | |
| D13-D14 | DataLoader + CoverageAwareSampler | 完整数据管线 | **M1: 数据管线端到端运行** |

### Phase 1：核心模型（Week 3-4）

| 天 | 任务 | 产出 | 里程碑 |
|---|---|---|---|
| D15-D17 | Causal Transformer + multi-horizon heads（**含 Instrument Head**） | model.py | |
| D18-D19 | Latent transition module | transition.py | |
| D20-D21 | Discrete-time hazard head | hazard.py | |
| D22-D23 | 训练循环 + 标签条件 loss | train.py | |
| D24-D25 | 训练 Baseline 1：CholecT50-only | 结果 | |
| D26-D27 | 训练 Baseline 2：+ Cholec80 + Cholec80-CVS | 结果 | |
| D28 | 训练 Baseline 3：+ Endoscapes | 结果 | **M2: 增量趋势验证** |

**⚡ Go/No-Go 检查点（Week 4 末）：**
如果增量表的 Change-mAP 没有递增趋势，停下来检查数据管线。如果数据管线正确但增量为负，重新评估问题设定。

**⚡ Safety Ground Truth 检查（Week 4 末同步进行）：**
分别统计 B1 和 B2 的 ground truth 事件数：B1 在全部 CholecT50 视频上统计 clipping events 数量；B2 在 G1+G2 共 45 个视频（训练集中 31 个有完整 triplet+CVS）上统计 clipping 时刻有 CVS 标注的帧数。B1 测试集为 10 个视频，B2 测试集为 9 个视频（排除 VID111）。如果 B1 的 clipping events 少于 30 个，Table 3 的 B1 列需降级；如果 B2 的 CVS-at-clipping 帧数不足，B2 改为定性展示。

### Phase 2：核心创新（Week 5-6）

| 天 | 任务 | 产出 | 里程碑 |
|---|---|---|---|
| D29-D30 | Structured prior (static only) 实现 | prior.py | |
| D31-D32 | Context-modulated prior 实现 | 集成到 train.py | |
| D33 | Coverage dropout 实现 | 集成到 dataloader | |
| D34-D36 | 完整 SurgCast 训练 + 超参数调整 | 完整模型结果 | |
| D37-D38 | Event-centric evaluation 代码 | evaluate.py | |
| D39-D40 | Safety-critical evaluation 代码（**B1: Clipping Anticipation + B2: CVS State Accuracy**） | safety 结果 | |
| D41-D42 | 小规模 prior ablation（验证 prior 有效性） | Static vs uniform vs no-prior | **M3: Prior 有效性验证** |

**⚡ Go/No-Go 检查点（Week 6 中）：**
如果 structured prior 相比 mask-and-ignore 的增益 < 1.5 个点（Change-mAP），考虑降低 prior 在论文中的权重，退守到 event-centric forecasting + hazard TTC 作为主贡献。

### Phase 3：消融 + 分析（Week 7-8）

| 天 | 任务 | 产出 |
|---|---|---|
| D43-D46 | 主文消融 A1-A5, A8-A9, A16（每个约 4 小时训练） | Table 2 完整 |
| D47-D49 | Appendix 消融 A6-A7, A10b-A10d, A11-A15（A10b/c/d 需额外特征提取） | 补充材料 |
| D49-D50 | Phase-Conditional Analysis | 分析表 |
| D51-D52 | 定性可视化（Figure 3） | Hazard 热力图 + Instrument 预测曲线 + 时间线 |
| D53-D54 | Safety-critical 结果整理 | Table 3 |
| D55-D56 | 所有结果 double-check | 最终数据 |

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
| **Structured prior 增益 marginal (<2 pts)** | 中高 | 高 | Week 6 检查点决策；退守 event-centric + hazard TTC 作为主贡献；注意 prior 基数从 v4 的 45 降到 31 个训练视频，可能影响先验质量 |
| **Change-mAP 所有方法极低 (<5%)** | 中 | 中 | 调整 change 定义（用 debounced），允许 ±1s 容差，报告多个阈值 |
| **TTC 被 label flicker 噪声拖死** | 中 | 高 | Group-level change 而非 strict change；debounced change 作为备选 |
| **Cholec80-CVS 与 Endoscapes CVS 不一致** | 中 | 中 | 在 G1 的 3 个 + G4 的 3 个 = 6 个双套 CVS 视频上量化一致性；如果严重不一致，分别使用两套 CVS 而非合并 |
| **Phase ontology 对齐失败** | 低 | 高 | 在 45 个重叠视频上逐 phase 检查时间轴对齐；必要时退化为 6-phase mapping |
| **Hazard head 训练不稳定** | 低 | 中 | 预训练其他 heads 先，hazard head 后接入；学习率分层 |
| **审稿人认为贡献不够** | 中 | 高 | 确保 TTC + hazard modeling 本身有足够辨识度；safety 结果（45 个视频的 ground truth）作为强 selling point |
| **DINOv3 在手术图像上存在 domain gap** | 中 | 中 | 消融 A10a 加入 LemonFM（手术域专用 backbone）对比；如果 LemonFM 显著优于 DINOv3，考虑切换主方法 backbone 或在论文中强调方法对 backbone 的鲁棒性 |
| **Unsafe transition 事件数不足** | 低（已有45个视频） | 中 | Week 4 末统计；如果 < 30 个事件，Table 3 改为定性展示 |
| **Cholec80 tool-presence 与 CholecT50 instrument 标签语义差异** | 中 | 低 | Cholec80 tool-presence 为帧级二值存在标注，CholecT50 instrument 标注与 triplet 绑定，粒度可能不完全一致。消融 A17 量化差异影响；如差异大则仅用 CholecT50 的 50 个视频 |
| **CVS 正例稀疏导致 CVS Head 训练信号不足** | 中高 | 中 | 使用 ordinal regression 保留更多序数信息；利用 Endoscapes CVS 补充正例；B2 评估聚焦 criterion-wise AUC 而非全达标率 |
| **Cholec80-CVS 自行解析与官方 pipeline 结果不一致** | 低 | 低 | 在 6 个 Endoscapes 重叠视频上对比自行解析结果与 Endoscapes CVS 的一致性；消融 A20 量化全程 1fps vs 官方截断的差异 |
| **Group-set change 频率过高或过低** | 中 | 中 | Phase 0 Day 3-5 实测后调整 group 数量；如 <1.5/min 则降低 group 数至 10-12；预留 debounced change 作为备选 |

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
