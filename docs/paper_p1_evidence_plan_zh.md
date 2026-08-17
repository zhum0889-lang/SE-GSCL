# P1 论文证据链推进计划

## 目标

本轮不继续扩展 LLM 功能，优先回答两个审稿问题：

1. 文本语义锚点是否比仅由首域信号学习的类别锚点更有效？
2. SE-GSCL 相对经验回放的增益是否对工况到达顺序稳定？

## 实验 A：语义锚点消融

固定 18 域 bearing-major 顺序、编码器、回放预算、训练轮数和随机种子，只替换锚点来源：

- `se_gscl_full`：冻结 LLM 文本经可训练投影器得到故障语义锚点；
- `wo_text_semantics`：首域训练信号编码器和自由类别锚点，随后冻结类别锚点；
- `experience_replay`：仅采用类别-域平衡经验回放；
- `wo_cross_condition`、`wo_relation`、`wo_decorrelation`：核心损失消融。

云端命令：

```bash
cd /mnt/workspace/fdllm/se_gscl_impl

python scripts/run_paper_p1_matrix.py \
  --dataset multidomain8_disjoint18 \
  --data-root /mnt/workspace/fdllm/data/MultiDomainBearing \
  --text-cache results/semantic_cache/qwen25_7b_bearing4_faceted_v2 \
  --output-root results/paper_matrix/p1_semantic_evidence \
  --order bearing_major \
  --seeds 42,52,62 \
  --jobs experience_replay,se_gscl_full,wo_text_semantics,wo_cross_condition,wo_relation,wo_decorrelation \
  --device cuda \
  --execute \
  --visualize
```

## 实验 B：域顺序稳健性

只比较最强通用基线和完整方法，避免无必要地重复全部消融。三种顺序为：

- `bearing_major`：按轴承族、环境族和速度组依次到达；
- `condition_major`：同一环境与速度组合跨轴承族交替到达；
- `reverse`：与默认顺序完全相反。

```bash
for ORDER in condition_major reverse; do
  python scripts/run_paper_p1_matrix.py \
    --dataset multidomain8_disjoint18 \
    --data-root /mnt/workspace/fdllm/data/MultiDomainBearing \
    --text-cache results/semantic_cache/qwen25_7b_bearing4_faceted_v2 \
    --output-root results/paper_matrix/p1_order_robustness \
    --order "$ORDER" \
    --seeds 42,52,62 \
    --jobs experience_replay,se_gscl_full \
    --device cuda \
    --execute \
    --visualize
done
```

## 判定标准

- 文本语义有效：`se_gscl_full` 在最终平均平衡准确率、平均增量准确率或遗忘指标上稳定优于 `wo_text_semantics`；
- 持续机制有效：`se_gscl_full` 稳定优于 `experience_replay`，且不是只在单一域顺序或单个种子成立；
- 若文本锚点没有带来增益，应降低论文中“LLM语义先验”的贡献强度，并检查语义文本质量、投影器训练和类别原型几何；
- 若不同域顺序下结论反转，应将域顺序敏感性作为主要局限，优先优化回放与关系保持，而不是继续进入 P2/P3。

## 暂缓内容

在上述两项证据成立前，暂缓 RAG、自反思、解释生成和更大 LLM。`se_gscl_temporal` 仍作为编码器候选，只有在与相同编码器的 Replay 基线公平比较后，才可替换论文主模型。
