# SurgCast: 异质缺失监督下的手术动作变化预见

## Surgical Action-Change Anticipation under Heterogeneous Missing Supervision

**版本：v4.0 (Post-Discussion Revision)**
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
| Cholec80 | phase（密集标注）+ tool presence | 80 | 184,498 | 全程 |
| Cholec80-CVS | CVS 三准则评分（0/1/2） | 80 | 覆盖 Preparation + CalotTriangleDissection 阶段 | Preparation → ClippingCutting 前 |
| Endoscapes2023 | anatomy bbox + CVS scores | 201 | 58,813 | ROI 窗口（dissection → first clip） |

**CholecT50 标签维度详解：** CholecT50 的每帧 JSON 提供 5 个独立的标注维度：triplet（100 类多标签）、instrument（6 类多标签）、verb（10 类多标签）、target（15 类多标签）、phase（7 类单标签）。v3.1 仅使用了 triplet（聚类后）和 phase，本版本新增利用 instrument 和 verb 维度。

**Cholec80-CVS 说明：** Cholec80-CVS 是 Universidad de los Andes 团队（Ríos et al., Scientific Data 2023）为 Cholec80 全部 80 个视频追加的 CVS 标注层。标注以时间段形式提供（起止帧 + 三个 criterion 各自的 0/1/2 评分），覆盖每个视频中 Preparation 和 Calot's Triangle Dissection 阶段。它不是独立的视频数据集，而是 Cholec80 的标注扩展。

### 3.2 覆盖结构

纳入 Cholec80-CVS 后，覆盖结构发生根本性升级。CholecT50 中有 45 个视频同时存在于 Cholec80 中，因此这 45 个视频现在同时拥有 triplet + phase + CVS 三重标注。

| 覆盖等级 | 标签配置 | 视频来源 | 视频数 | 约占比 |
|---|---|---|---|---|
| **最高覆盖** | triplet ✓ + instrument ✓ + verb ✓ + phase ✓ + CVS(Cholec80-CVS) ✓ | CholecT50 ∩ Cholec80 | **45** | ~16% |
| 高覆盖（无CVS） | triplet ✓ + instrument ✓ + verb ✓ + phase ✓ | CholecT50 独有（VID92/96/103/110/111） | 5 | ~2% |
| Phase + CVS | phase ✓ + CVS(Cholec80-CVS) ✓ | Cholec80 独有（不在 CholecT50 也不在 Endoscapes 中） | ~29 | ~10% |
| Endoscapes-only | anatomy bbox ✓ + CVS(Endoscapes) ✓ | Endoscapes 非重叠视频 | ~150 | ~54% |
| 混合最高覆盖 | triplet ✓ + instrument ✓ + verb ✓ + phase ✓ + CVS(两套) ✓ + anatomy bbox ✓ | Endoscapes 与 CholecT50 重叠的 6 个视频 | **6** | ~2% |

**v3.1 → v4.0 的关键变化：** 原来只有 6 个视频同时具备 action + safety 信息，现在有 **45 个**（最高覆盖组）。这意味着：
- Structured prior 中 P(CVS | phase, triplet-group) 的统计从 6 个视频扩展到 45 个
- Safety-Critical Evaluation 的 ground truth 样本量增加约 7.5 倍
- Coverage-aware batching 中激活 L_cvs 的样本占比大幅提升

说明：
- 最高覆盖的 45 个视频是 structured prior 统计的核心数据源（triplet + phase + CVS 同源）
- Cholec80 独有的约 29 个视频提供 phase + CVS（Cholec80 共 80 个，减去与 CholecT50 重叠的 45 个，再减去与 Endoscapes 重叠的约 6 个）
- 混合最高覆盖的 6 个视频同时拥有所有标签维度，包括两套 CVS（Cholec80-CVS + Endoscapes），是跨标注体系一致性验证的宝贵样本
- CholecT50 独有的 5 个视频（VID92/96/103/110/111）不在 Cholec80 中，因此没有 Cholec80-CVS 标注
- 最终覆盖等级以注册表 `registry.json` 中的实际标签可用性为准

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
  "endoscapes_public_id": null,
  "labels_available": ["triplet", "instrument", "verb", "target", "phase", "cvs_cholec80"],
  "coverage_level": "highest",
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
[Module A] 冻结 DINOv2 ViT-B/14 ──→ 768-d frame features (离线提取, 存HDF5)
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
        │         ├──→ Triplet-Group Head ──→ ŷ_triplet-group     (CE)
        │         ├──→ Instrument Head ──→ ŷ_instrument            (BCE)  [v4.0 新增]
        │         ├──→ Phase Head ──→ ŷ_phase                      (CE)
        │         └──→ Safety/CVS Head ──→ ŷ_cvs                   (BCE)
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

**选择：DINOv2 ViT-B/14（主实验）**

| 项目 | 说明 |
|---|---|
| 模型 | DINOv2 ViT-B/14, frozen, 86M params |
| 输入 | 原始帧 resize 到 518×518 |
| 输出 | 768-d CLS token per frame |
| 存储 | 每个数据集一个 HDF5 文件，按 video_id 索引 |
| 总帧数 | ~280,000（Cholec80 184K + CholecT50 增量 ~37K + Endoscapes ~59K） |
| 提取时间 | ~1 小时（2×A100） |
| 存储量 | ~820 MB |

**消融中对比：** ImageNet-pretrained ResNet-50。

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

**目的：** 让模型在学到的状态空间中预测未来，而不是直接做分类。

对每个预测 horizon Δ ∈ {1, 3, 5} 秒：

```
ĥ_{t+Δ} = g_Δ(h_t) = MLP_Δ(h_t)
```

每个 MLP_Δ 结构：Linear(512, 512) → GELU → Linear(512, 512)

**Latent alignment loss：**

```
L_align = Σ_Δ ‖ĥ_{t+Δ} - sg(h_{t+Δ})‖²
```

其中 sg(·) 是 stop-gradient，防止目标端表示坍塌。

**四个预测头（从 ĥ_{t+Δ} 解码）：**

| 头 | 结构 | 输出 | 损失 | 说明 |
|---|---|---|---|---|
| Triplet-Group Head | Linear(512, G), G≈15-20 | 单标签概率 | CE | 预测变化后的主状态 |
| **Instrument Head** | **Linear(512, 6)** | **6 维多标签** | **BCE** | **[v4.0 新增] 预测器械存在，直接用于 safety alert** |
| Phase Head | Linear(512, 7) | 单标签概率 | CE | 7 阶段单分类 |
| Safety/CVS Head | Linear(512, 3) | 3 维 CVS 预测 | BCE | 3 个 CVS criterion 独立预测 |

**v4.0 新增 Instrument Head 的理由：**
1. CholecT50 的 instrument 标签（6 类）一直存在但 v3.1 未使用
2. Instrument 预测直接服务于 Safety-Critical Anticipation——预测 clipper 即将出现比通过 triplet-group 聚类间接推断更精确
3. Instrument change 是比 triplet-group change 更干净的信号（6 个类别 vs 15-20 个 group），可作为辅助 TTC 目标
4. 参数增量极小（~3K 参数）

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
λ(k | h_t) = P(T = k | T ≥ k, h_t) = σ(f_hazard(h_t)_k)
```

其中 f_hazard 是一个 MLP：Linear(512, 256) → GELU → Linear(256, K)，σ 是 sigmoid。

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

**核心来源：CholecT50 ∩ Cholec80 的 45 个最高覆盖视频。** 这 45 个视频同时拥有 triplet + instrument + verb + phase + CVS(Cholec80-CVS) 标注，是构建完整 procedure prior 的最佳数据源。

从这些视频中统计以下条件分布：

| 条件分布 | 数据来源 | 说明 |
|---|---|---|
| P_static(triplet-group \| phase) | CholecT50 全部 50 个视频 | 每个 phase 下的 triplet-group 经验分布 |
| P_static(triplet-group_{t+1} \| phase_t, triplet-group_t) | CholecT50 全部 50 个视频 | 完整状态转移概率 |
| P_static(instrument \| phase) | CholecT50 全部 50 个视频 | **[v4.0 新增]** 每个 phase 下的 instrument 分布 |
| P_static(phase_{t+1} \| phase_t) | Cholec80 全 80 个视频 | phase 转移矩阵（样本更充分） |
| P_static(cvs_ready \| phase, triplet-group) | **45 个最高覆盖视频** | **[v4.0 新增]** CVS 达标概率与 (phase, action) 的关联 |

**v4.0 关键改进：** P(cvs_ready | phase, triplet-group) 的统计从 6 个混合覆盖视频扩展到 45 个最高覆盖视频。例如，可以精确统计"在 CalotTriangleDissection 阶段，当 triplet-group 是 clipper-related 时，CVS 达标的概率是多少"——这直接支撑 unsafe transition 的先验判断。

Endoscapes 映射：对于 6 个混合覆盖视频，可以同时验证 Cholec80-CVS 和 Endoscapes CVS 的一致性，以及 anatomy bbox 与 CVS 状态的对应关系。

存储为查找表：`static_prior.pkl`。

**Layer 2：Context-Modulated Prior（可学习，训练中更新）**

在 static prior 基础上，用当前 latent state h_t 做上下文调制：

```
q_prior(y_miss | y_obs, h_t) = softmax(α · log P_static(y_miss | y_obs) + β · g_φ(h_t))
```

其中：
- P_static(y_miss | y_obs) 是 Layer 1 的查找表输出
- g_φ(h_t) 是一个轻量 MLP：Linear(512, 256) → GELU → Linear(256, C)
- α, β 是可学习的标量（初始化 α=1.0, β=0.1，让训练初期以 static prior 为主）
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

#### 4.6.3 Ontology Bridge

**Phase 对齐：** Cholec80 和 CholecT50 的 phase 名称相似但不完全一致（如 `GallbladderRetraction` vs `gallbladder-extraction`）。使用一个共享的 coarse phase space（7 个 phase），加上数据集特定的映射规则。

**注意：** `GallbladderRetraction`（Cholec80）与 `gallbladder-extraction`（CholecT50）仅为近似语义映射，不是精确对应。在 Phase 0 中需对 CholecT50 的 45 个重叠视频做显式验证——检查这两个标签在时间轴上是否真的对齐。

**CVS 对齐：** Cholec80-CVS 使用 0/1/2 三级评分，Endoscapes 使用二值/连续评分。统一方案：将 Cholec80-CVS 的评分二值化（≥1 映射为 1，表示 criterion 部分或完全满足），使两个数据源的 CVS 均为三维二值向量 (C1, C2, C3) ∈ {0,1}³，共享同一个 Safety/CVS Head。

### 4.7 总损失函数

```
L_total = L_task + λ_align · L_align + λ_hazard · L_hazard + λ_prior · L_prior
```

各项定义：

```
L_task = m_tri · L_triplet + m_inst · L_instrument + m_pha · L_phase + m_cvs · L_cvs
    (标签条件化多任务损失，m ∈ {0,1} 是可见性 mask)
    (L_triplet 和 L_phase 均为 CE loss；L_instrument 和 L_cvs 为 BCE)

L_align = Σ_Δ ‖ĥ_{t+Δ} - sg(h_{t+Δ})‖²
    (Latent transition alignment)

L_hazard = 离散时间生存损失（见 4.5.3）

L_prior = D_KL(p_θ(y_miss | h_t) ‖ q_prior(y_miss | y_obs, h_t))
    (Structured prior regularization，仅在 coverage dropout 激活时计算)
```

**超参数初始值：** λ_align = 0.5, λ_hazard = 1.0, λ_prior = 0.3

### 4.8 模型规模

| 组件 | 参数量 |
|---|---|
| Input projection (768→512) | ~400K |
| Causal Transformer (6 layers) | ~9.5M |
| Transition MLPs (×3 horizons) | ~1.6M |
| Prediction heads (triplet-group + instrument + phase + CVS) | ~55K |
| Hazard head | ~140K |
| Prior modulation MLP | ~140K |
| Context α, β | 2 |
| **总计** | **~12M trainable** |

加上冻结的 DINOv2 86M 参数，推理时总参数 ~98M，但只有 12M 需要训练。在 2×A100 40GB 上训练毫无压力。

---

## 五、Action Change 的定义与稳健性处理

### 5.1 为什么 change 定义至关重要

TTC 的质量完全取决于"什么算一次 change"。CholecT50 的 triplet 标注是 1fps 的二值向量，相邻帧之间可能出现 label flicker（如 triplet 在连续三帧中 1-0-1）。如果把每一次 flicker 都算作 change，TTC 目标会充满噪声。

### 5.2 三种 change 定义

| 定义 | 说明 | 预期 change density | 用途 |
|---|---|---|---|
| Strict change | 任何一个 triplet 维度变化即 change | 极高，充满 flicker | 内部验证用，不作为主任务 |
| **Group-level change（主任务）** | triplet-group 变化才算 change | 中等 | **主要 TTC 目标** |
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

在论文中必须包含以下信息（让审稿人判断 TTC 任务的合理性）：

- 平均每个视频有多少个 group-level change point
- 平均 inter-change interval（秒），及其分布（中位数、四分位距）
- Change point 在不同 phase 中的分布
- Debounced change 相比 group-level change 过滤掉了多少 flicker
- Instrument-level change 的密度对比

**⚠ 必须在 Phase 0 第一天完成的检查：** 在 CholecT50 上完成 triplet-group 聚类并统计 change density。如果每个视频平均 change point 少于 20 个，认真考虑调整 group 粒度或 change 定义。

---

## 六、训练方案

### 6.1 Coverage-Aware Batching

每个 batch（batch_size=64）按以下比例采样：

| 覆盖等级 | Batch 占比 | 样本数/batch | 激活的 loss |
|---|---|---|---|
| 最高覆盖（45 videos: triplet + instrument + phase + CVS） | 35% | 22 | L_triplet + L_instrument + L_phase + L_cvs + L_hazard + L_prior（30%概率） |
| Phase + CVS（~29 videos: Cholec80 独有） | 20% | 13 | L_phase + L_cvs |
| CholecT50-only（5 videos: triplet + instrument + phase, 无CVS） | 10% | 6 | L_triplet + L_instrument + L_phase + L_hazard |
| Endoscapes（~150 videos: anatomy + CVS） | 35% | 23 | L_cvs |

注意：6 个混合最高覆盖视频归入最高覆盖组训练，且额外可利用 Endoscapes 的 anatomy bbox。

**v3.1 → v4.0 的关键变化：** 激活 L_cvs 的样本从只有 Endoscapes 的 35% 扩展到 55%（最高覆盖 35% + Phase+CVS 20%），CVS 训练信号大幅增强。

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
| **Change-mAP** | 仅在 group-level change point 上评估的 triplet-group 预测 mAP | 消除 temporal inertia inflation |
| **TTC MAE** | time-to-next-change 的平均绝对误差（秒） | TTC 预测精度 |
| **TTC C-index** | concordance index，衡量 TTC 排序一致性 | 区分力（是否能正确区分"快变"和"慢变"） |
| **TTC Brier Score** | 对"未来 k 秒内是否发生 change"的概率预测的 Brier score，k ∈ {3, 5, 10} | 概率校准质量（hazard head 的核心卖点） |
| Future-state Acc @Δ | 在 change 发生后 Δ 秒的 triplet-group 预测准确率 | 预测变化后状态的能力 |

#### B. Safety-Critical Anticipation

**v4.0 重构：直接使用 instrument/verb 原始标签定义 safety 事件**

v3.1 的 safety 事件定义依赖聚类后的 CLIP_LIKE_GROUPS（需要人工检查聚类结果确认），v4.0 改为直接使用 CholecT50 的原始标签，更精确、更可复现。

**Unsafe Transition 的 Ground Truth 定义：**

```python
# 配置参数（在 config.yaml 中固化）
CVS_READY_RULE = "all"    # C1 ≥ threshold AND C2 ≥ threshold AND C3 ≥ threshold
CVS_THRESHOLD = 0.5       # 二值化后的阈值
WARNING_HORIZONS = [5, 10, 15]  # 预警窗口（秒）

def is_clipping_event(t, labels):
    """直接使用 CholecT50 原始 instrument/verb 标签判定"""
    return (labels['instrument'][t]['clipper'] == 1 or 
            labels['verb'][t]['clip'] == 1 or 
            labels['verb'][t]['cut'] == 1)

def is_unsafe_transition(t, future_window_k, labels, cvs):
    """Ground truth: 未来 k 秒内出现 clipping 且当前 CVS 不达标"""
    future_clip = any(is_clipping_event(t+i, labels) 
                      for i in range(1, future_window_k + 1))
    cvs_not_ready = not (cvs[t]['C1'] >= CVS_THRESHOLD and 
                         cvs[t]['C2'] >= CVS_THRESHOLD and 
                         cvs[t]['C3'] >= CVS_THRESHOLD)
    return future_clip and cvs_not_ready
```

**模型预测端的触发规则：**
1. 在时间步 t，如果 Instrument Head 预测 clipper 的概率 > τ_inst（可调阈值），发出"即将 clipping"预警
2. 如果同时 CVS Head 预测不满足 CVS_READY_RULE，升级为 **Unsafe Transition Alert**

**v4.0 改进的优势：**
- Ground truth 不再依赖聚类质量（去除了 CLIP_LIKE_GROUPS 的间接推断）
- 预测端直接使用 Instrument Head（6 类分类器比在 15-20 个 group 中识别 clip-like 更直接）
- 完全可复现：只依赖原始标签中的 instrument 和 verb 字段

**Ground truth 可用范围：** 45 个最高覆盖视频（同时有 triplet/instrument/verb + CVS）均可构建完整的 unsafe transition ground truth。这比 v3.1 的 6 个视频扩大了约 7.5 倍。

**评估指标：**

| 指标 | 定义 |
|---|---|
| **Detection Rate (Recall)** | 在真实发生的 clipping events 中，模型在 k 秒前正确预警的比例 |
| **False Alarm Rate** | 模型发出"即将 clipping"警告但实际未发生的比例 |
| Precision | 预警中真正触发了 clipping 的比例 |
| PR-AUC | 不同阈值 τ 下的 Precision-Recall 曲线下面积 |

主文报告 Detection Rate @5s/@10s 和 False Alarm Rate；Precision 和 PR-AUC 放 appendix。

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

**Table 3：Safety-Critical Anticipation**

| Method | Detection @5s | Detection @10s | False Alarm Rate |
|---|---|---|---|
| CholecT50-Only (no CVS) | — | — | N/A |
| + Cholec80-CVS only | — | — | — |
| + Endoscapes CVS | — | — | — |
| **Full SurgCast** | **—** | **—** | **—** |

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
| A7 | 去掉 temporal module（单帧 MLP） | 时序建模是否必要？ |
| A8 | Hazard head → MSE regression | Hazard 建模的优势？ |
| A9 | Hazard head → binned classification | Hazard vs 普通分类？ |
| A10 | DINOv2 → ImageNet ResNet-50 | 强 backbone 的重要性？ |
| A11 | Change 定义：strict vs group vs debounced | Change 定义的敏感性？ |
| A12 | 序列长度：8 / 16 / 32 | 时序窗口的影响？ |
| A13 | Coverage dropout rate：0 / 0.1 / 0.3 / 0.5 | 最优 dropout 率？ |
| **A14** | **纯共现聚类 vs 语义增强聚类** | **[v4.0 新增] 语义信息是否改善 group 定义？** |
| **A15** | **去掉 Instrument Head** | **[v4.0 新增] 分解标签是否有用？** |
| **A16** | **Cholec80-CVS only vs Endoscapes CVS only vs 两者联合** | **[v4.0 新增] 两套 CVS 标注的互补性？** |

主文放 A1-A5（structured prior 相关）、A8-A9（hazard 相关）和 A16（CVS 来源）。其余放 appendix。

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

左半：数据覆盖结构矩阵。横轴是视频 ID，纵轴是标签维度（triplet, instrument, phase, CVS-Cholec80, CVS-Endoscapes, anatomy）。用颜色块展示"哪些视频有哪些标签"——突出 45 个最高覆盖视频的三重标注。

右半：一个手术视频的时间线示例，标注 change points、clipping events 和 CVS 状态，直观展示"我们预测的是什么"以及"unsafe transition 是怎么定义的"。

**Figure 2（第 3-4 页，方法架构）：**

完整架构图。左：冻结 DINOv2 + Causal Transformer。中：Latent Transition + Multi-horizon heads（突出 Instrument Head 的新增）。右上：Discrete-Time Hazard Head。右下：Structured Prior Regularization（static prior 查表 + context modulation），展示分解条件分布。

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

4. 我们提出 SurgCast，核心方法是 procedure-aware structured prior regularization——利用手术流程的组合约束（包括 instrument-phase 对应关系和 CVS-action 安全约束）为缺失标签生成上下文自适应的软监督，结合 discrete-time hazard modeling 实现 TTC 预测。

5. 实验表明，结构化的跨数据集学习在 event-centric 指标上显著超过单数据集训练和 naive multi-task baselines，且直接的 instrument 预测和 CVS 监督对 safety-critical anticipation 有明确贡献。

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

### Step 3：Cholec80 预处理（2 天）

| 子任务 | 具体操作 |
|---|---|
| 帧-标签对齐 | 通过原始帧 ID（0, 25, 50, ...）做 phase 标注 → 1fps 对齐 |
| 文件名偏移修正 | 图像 1-indexed（video01_000001.png），标注 0-indexed |
| Phase 标签标准化 | 统一到 coarse 7-phase ontology |
| 输出 | 每个视频一个 `.npz`：frames, phase_ids |

### Step 3.5：Cholec80-CVS 预处理（1 天）[v4.0 新增]

| 子任务 | 具体操作 |
|---|---|
| 下载标注 | 从 Figshare 下载 surgeons_annotations.xlsx |
| 时间段 → 逐帧转换 | XLSX 中帧号为原始 25fps → 除以 25 转为秒数 → 映射到 1fps 帧 |
| 帧号偏移对齐 | 注意 Cholec80 图像 1-indexed vs 标注 0-indexed 的偏移 |
| CVS 评分二值化 | 三级评分（0/1/2）→ 二值化（≥1 映射为 1） |
| Phase 范围验证 | 确认标注仅覆盖 Preparation + CalotTriangleDissection 阶段 |
| **Phase ontology 验证** | **对 45 个重叠视频，检查 CholecT50 的 gallbladder-extraction 与 Cholec80 的 GallbladderRetraction 在时间轴上是否对齐** |
| 输出 | 每个视频一个 `.npz`：frames, cvs_c1, cvs_c2, cvs_c3, cvs_binary（三维二值向量） |
| 验证 | 与 Cholec80 的帧数一致；6 个 Endoscapes 重叠视频上对比两套 CVS 的一致性 |

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
| 核心数据源 | **45 个最高覆盖视频（triplet + instrument + phase + CVS 同源标注）** |
| Phase transition 补充 | Cholec80 全 80 个视频（用于 phase 转移矩阵统计） |
| 条件分布计算 | P(triplet-group \| phase) 来自 CholecT50 50 videos |
| | P(instrument \| phase) 来自 CholecT50 50 videos **[v4.0 新增]** |
| | P(phase_{t+1} \| phase_t) 来自 Cholec80 全 80 videos |
| | **P(cvs_ready \| phase, triplet-group) 来自 45 个最高覆盖视频 [v4.0 新增]** |
| Endoscapes 映射 | 6 个混合覆盖视频验证两套 CVS 一致性及 anatomy ↔ CVS 对应关系 |
| 验证 | 转移矩阵行和为 1，所有训练中出现的 (phase, group) 对有节点 |
| 输出 | `static_prior.pkl` |

### Step 6：DINOv2 特征提取（1 天）

| 子任务 | 具体操作 |
|---|---|
| 模型 | DINOv2 ViT-B/14, frozen |
| 输入 | 518×518 resize |
| 输出 | 768-d CLS token per frame |
| 存储 | 3 个 HDF5 文件（cholec80.h5, cholect50.h5, endoscapes.h5） |
| 验证 | 特征维度正确，帧数与 registry 一致 |

### Step 7：DataLoader 实现（2 天）

| 子任务 | 具体操作 |
|---|---|
| SequenceDataset | 从 HDF5 读取特征，滑动窗口切序列 |
| CoverageAwareSampler | 按 35/20/10/35 比例采样最高/Phase+CVS/CholecT50-only/Endoscapes |
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
| D6-D7 | Cholec80 预处理 + **Cholec80-CVS 预处理 + Phase ontology 验证** | 80 个 npz + CVS 逐帧标签 | |
| D8-D9 | Endoscapes 预处理 + **6 个重叠视频 CVS 一致性验证** | ~201 个 npz | |
| D10-D11 | Procedure graph + static prior（**含分解分布和 CVS 先验**） | static_prior.pkl | |
| D12 | DINOv2 特征提取 | 3 个 HDF5 | |
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
统计 45 个最高覆盖视频中的 unsafe transition 事件数。如果少于 30 个事件，Table 3 的实验设计需要降级或改为定性展示。

### Phase 2：核心创新（Week 5-6）

| 天 | 任务 | 产出 | 里程碑 |
|---|---|---|---|
| D29-D30 | Structured prior (static only) 实现 | prior.py | |
| D31-D32 | Context-modulated prior 实现 | 集成到 train.py | |
| D33 | Coverage dropout 实现 | 集成到 dataloader | |
| D34-D36 | 完整 SurgCast 训练 + 超参数调整 | 完整模型结果 | |
| D37-D38 | Event-centric evaluation 代码 | evaluate.py | |
| D39-D40 | Safety-critical evaluation 代码（**基于 instrument/verb 标签**） | safety 结果 | |
| D41-D42 | 小规模 prior ablation（验证 prior 有效性） | Static vs uniform vs no-prior | **M3: Prior 有效性验证** |

**⚡ Go/No-Go 检查点（Week 6 中）：**
如果 structured prior 相比 mask-and-ignore 的增益 < 1.5 个点（Change-mAP），考虑降低 prior 在论文中的权重，退守到 event-centric forecasting + hazard TTC 作为主贡献。

### Phase 3：消融 + 分析（Week 7-8）

| 天 | 任务 | 产出 |
|---|---|---|
| D43-D46 | 主文消融 A1-A5, A8-A9, A16（每个约 4 小时训练） | Table 2 完整 |
| D47-D48 | Appendix 消融 A6-A7, A10-A15 | 补充材料 |
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
| **Structured prior 增益 marginal (<2 pts)** | 中高 | 高 | Week 6 检查点决策；退守 event-centric + hazard TTC 作为主贡献 |
| **Change-mAP 所有方法极低 (<5%)** | 中 | 中 | 调整 change 定义（用 debounced），允许 ±1s 容差，报告多个阈值 |
| **TTC 被 label flicker 噪声拖死** | 中 | 高 | Group-level change 而非 strict change；debounced change 作为备选 |
| **Cholec80-CVS 与 Endoscapes CVS 不一致** | 中 | 中 | 在 6 个重叠视频上量化一致性；如果严重不一致，分别使用两套 CVS 而非合并 |
| **Phase ontology 对齐失败** | 低 | 高 | 在 45 个重叠视频上逐 phase 检查时间轴对齐；必要时退化为 6-phase mapping |
| **Hazard head 训练不稳定** | 低 | 中 | 预训练其他 heads 先，hazard head 后接入；学习率分层 |
| **审稿人认为贡献不够** | 中 | 高 | 确保 TTC + hazard modeling 本身有足够辨识度；safety 结果（45 个视频的 ground truth）作为强 selling point |
| **DINOv2 在手术图像上特征不够好** | 低 | 中 | 消融中加 ImageNet ResNet-50 对比，证明方法 backbone-agnostic |
| **Unsafe transition 事件数不足** | 低（已有45个视频） | 中 | Week 4 末统计；如果 < 30 个事件，Table 3 改为定性展示 |

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

- [ ] DINOv2 ViT-B/14 权重已下载并验证
- [ ] 四个数据集的本地路径与数据审计报告一致（CholecT50, Cholec80, Endoscapes, Cholec80-CVS）
- [ ] Cholec80-CVS 的 surgeons_annotations.xlsx 已从 Figshare 下载
- [ ] sentence-transformers (all-MiniLM-L6-v2) 已安装并可本地运行
- [ ] CAMMA overlap mapping 文件已获取
- [ ] 2×A100 机器环境配置完成（PyTorch, HDF5, etc.）
- [ ] Git 仓库已建立，实验 logging 框架已搭建（W&B 或 TensorBoard）

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
