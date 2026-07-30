# SE-GSCL 论文级实验矩阵与云端执行说明

## 1. 实验目标与公平性约束

论文实验围绕三个问题展开。第一，所提出的持续语义学习机制能否在工况依次到达时兼顾新工况可塑性和旧工况稳定性；第二，全局故障身份、物理引导局部症状、语义保护与可靠性融合分别产生多大贡献；第三，连续语义 token 能否驱动冻结大模型完成诊断，并在生成解释时保持诊断结论和证据的一致性。

所有方法使用相同的数据划分、编码器容量、优化轮数和随机种子。归一化统计仅由第一个工况训练集拟合；每个阶段的测试标签只用于训练结束后的指标计算；回放方法使用相同的每类容量；文本编码器和 Qwen2.5-7B 主体保持冻结。正式结果使用随机种子 42、52 和 62，报告均值与总体标准差。P3 同时报告大模型原始输出和确定性语义控制后的输出，不以控制后的指标替代模型原生能力。

## 2. 数据集与工况序列

主数据集为 CWRU 四分类协议，类别为 Normal、InnerRace、Ball 和 OuterRace，按负载工况 D0→D1→D2→D3 顺序持续学习。第二数据集采用 HUSTbearing 全九分类协议，包括健康状态、内圈/外圈/滚动体的中度与重度故障，以及中度和重度复合故障；三轴振动窗口长度为 2048、步长为 1024，选取 20→40→60→80 Hz 作为持续工况序列。

HUSTbearing 的局部物理语义使用 ER-16K 轴承参数计算 BPFI、BPFO、BSF 和 FTF 相对转频比。九类本体为每类配置三条故障描述和三项局部症状，使第二数据集同时满足“工程数据—文本语义对齐”和“物理症状监督”的输入要求，而不是仅进行普通标签分类。

本地协议审计在上述四个转速下识别到 36 条记录，每个转速完整包含九类状态。由于官方文件在每个“类别—转速”组合下仅提供一条长记录，无法执行跨轴承 ID 划分；实验采用按时间顺序的 train/validation/test 连续分块，并在相邻分块间跳过至少一个窗口长度的保护带，避免重叠窗口跨集合泄漏。论文中应如实说明这一数据集固有限制，不将其表述为跨轴承泛化实验。

## 3. 实验矩阵

### 3.1 持续学习基线与 P1 消融

| 组别 | 方法 | 作用 |
|---|---|---|
| 基线 | Sequential fine-tuning | 仅用当前工况更新，测量灾难性遗忘下限 |
| 基线 | LwF-style relation distillation | 不回放原始旧样本，保持更新前的类别关系分布 |
| 基线 | Balanced experience replay | 类别—工况平衡回放，不使用跨工况对比和关系保持 |
| 完整方法 | SE-GSCL | 平衡回放、跨工况同类约束、故障/工况解耦和历史关系保持 |
| 消融 | w/o cross-condition | 移除跨工况同类对比约束 |
| 消融 | w/o relation | 移除旧样本—语义原型关系保持 |
| 消融 | w/o decorrelation | 移除故障分支与工况分支去相关约束 |

### 3.2 P2 局部语义与融合消融

| 方法 | 局部症状 | 物理软目标 | 语义保护 | 自适应融合 |
|---|---:|---:|---:|---:|
| Global only | 否 | 否 | 否 | 否 |
| Weak local fixed | 是 | 否 | 否 | 否 |
| Physics fixed | 是 | 是 | 否 | 否 |
| Semantic guard fixed | 是 | 是 | 是 | 否 |
| SE-GSCL full | 是 | 是 | 是 | 是 |

P2 需分别报告 global、local 和 fused 的平均 Accuracy 与 Balanced Accuracy，并报告局部分支平均权重、局部覆盖率以及物理症状匹配指标。该设计用于区分“增加一个分支”与“物理监督、语义稳定和可靠性门控真正有效”。

### 3.3 P3 连续提示与解释消融

| 方法 | 连续向量 token | 辅助类别约束 | 诊断锁定 |
|---|---:|---:|---:|
| Structured text | 否 | 否 | 不适用 |
| Continuous w/o auxiliary | 是 | 否 | 不适用 |
| Continuous full | 是 | 是 | 不适用 |
| Explanation unlocked | 是 | 是 | 否 |
| Explanation locked | 是 | 是 | 是 |

连续提示分类报告 Accuracy、Balanced Accuracy、各类 Recall、各工况 Accuracy、合法标签率和与上游融合诊断的一致率。解释实验报告 JSON 解析率、证据落地率、证据类别一致性、维护策略一致性、诊断保持率和语义控制修复率。

## 4. 持续学习指标

设第 \(i\) 个训练阶段后在第 \(j\) 个工况上的得分为 \(a_{i,j}\)，共 \(T\) 个工况。

- 最终平均准确率：最后阶段在全部工况上的平均得分。
- 平均增量准确率：每个阶段仅对当时已经见过的工况求平均，再对各阶段求平均。
- 平均遗忘：对每个旧工况计算历史最佳得分与最终得分之差，再取平均。
- 最大遗忘：所有旧工况遗忘量的最大值。
- 后向迁移：最终得分相对该工况首次学习后得分的平均变化，负值表示遗忘。
- 旧工况保持率：旧工况最终得分与首次学习后得分之比的平均值。

论文主表优先使用 Balanced Accuracy，普通 Accuracy 作为补充，以降低 HUSTbearing 类别样本量差异造成的偏置。

## 5. 云端执行

以下命令假设仓库位于 `/mnt/workspace/fdllm/se_gscl_impl`，Qwen 位于 `/mnt/workspace/fdllm/models/Qwen/Qwen2___5-7B-Instruct`。

```bash
cd /mnt/workspace/fdllm/se_gscl_impl
export PYTHONPATH=src:.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
QWEN=/mnt/workspace/fdllm/models/Qwen/Qwen2___5-7B-Instruct
```

先为 HUSTbearing 生成全局故障文本和局部症状缓存。CWRU 已有缓存时无需重复执行。

```bash
python scripts/cache_text_embeddings.py \
  --model "$QWEN" \
  --ontology configs/semantics/hustbearing_faults_9.json \
  --output-dir results/semantic_cache/qwen25_7b_hust9_v1 \
  --device cuda --dtype bfloat16 --local-files-only

python scripts/cache_symptom_embeddings.py \
  --model "$QWEN" \
  --ontology configs/semantics/hustbearing_faults_9.json \
  --output-dir results/semantic_cache/qwen25_7b_hust9_symptoms_v1 \
  --device cuda --dtype bfloat16 --local-files-only
```

先用默认 dry-run 检查将要运行的命令；确认后加入 `--execute`。

```bash
python scripts/run_paper_p1_matrix.py \
  --dataset cwru4 \
  --data-root /mnt/workspace/fdllm/data/CWRU \
  --device cuda

python scripts/run_paper_p1_matrix.py \
  --dataset cwru4 \
  --data-root /mnt/workspace/fdllm/data/CWRU \
  --device cuda --execute
```

HUSTbearing 根目录必须包含官方 `raw data` 目录：

```bash
python scripts/run_paper_p1_matrix.py \
  --dataset hustbearing \
  --data-root /mnt/workspace/fdllm/data/HUSTbearing \
  --device cuda --execute
```

完成某数据集的 P1 后，运行 P2/P3。首次建议分别执行，便于定位显存或依赖问题。

```bash
python scripts/run_paper_downstream_matrix.py \
  --dataset cwru4 \
  --data-root /mnt/workspace/fdllm/data/CWRU \
  --model "$QWEN" \
  --stage p2 --device cuda --dtype bfloat16 \
  --local-files-only --execute

python scripts/run_paper_downstream_matrix.py \
  --dataset cwru4 \
  --data-root /mnt/workspace/fdllm/data/CWRU \
  --model "$QWEN" \
  --stage p3 --device cuda --dtype bfloat16 \
  --prompt-epochs 10 --local-files-only --execute
```

将 `cwru4` 和数据路径替换为 `hustbearing` 与 HUSTbearing 根目录即可运行第二数据集。脚本支持断点续跑：存在完整报告时自动跳过；只有明确需要覆盖时才添加 `--force`。

全部实验完成后生成论文表格：

```bash
python scripts/summarize_paper_matrix.py
```

输出位于 `results/paper_matrix/summary`，包括 P1/P2/P3 的逐种子表、均值/标准差表、完成数量及缺失报告清单。汇总器在实验未完整时返回状态 `incomplete`，防止把部分结果误当作完整论文矩阵。

## 6. 论文结果组织

主文建议保留四张表：两数据集持续学习基线比较、P1 消融、P2 分层语义消融、P3 连续提示与解释一致性。正文图展示工况序列准确率曲线、遗忘热图、全局/局部/融合比较和连续提示混淆矩阵。外部方法复现（如 EWC、DER++ 及旋转机械专用持续学习方法）应作为下一批横向基线接入相同协议；在其代码和超参数未完成公平复现前，不与当前内部基线混写。
