# MultiDomainBearing 18 域主实验协议

## 1. 协议选择

论文主实验使用 `multidomain8_disjoint18`。该协议由 3 类轴承、3 个互斥环境族和 2 个转速组构成，共形成 `3 x 3 x 2 = 18` 个连续工况域。

- 轴承：6204、N204/NJ204、30204。
- 环境族：基准/松动 `H-L={H,L}`，不平衡 `U={U1,U2,U3}`，不对中 `M={M1,M2,M3}`。
- 转速组：低速 `slow={600,800,1000 rpm}`，高速 `fast={1200,1400,1600 rpm}`。

域编号定义为 `d = 6b + 2e + s`，其中 `b`、`e` 和 `s` 分别表示轴承、环境族和转速组索引。每条原始记录只属于一个域，避免同一信号文件被重复用于多个持续学习阶段。

原始环境状态不会被丢弃。例如，`U2` 样本的域标签为不平衡环境族 `U`，但样本工况上下文仍记录 `state_U2`。这使持续学习评估采用稳定的 18 域粒度，同时允许后续语义提示区分具体环境强度。

## 2. 域定义

| 域 | 轴承 | 环境族 | 转速组 |
|---:|---|---|---|
| 0/1 | 6204 | H-L | slow/fast |
| 2/3 | 6204 | U | slow/fast |
| 4/5 | 6204 | M | slow/fast |
| 6/7 | N204/NJ204 | H-L | slow/fast |
| 8/9 | N204/NJ204 | U | slow/fast |
| 10/11 | N204/NJ204 | M | slow/fast |
| 12/13 | 30204 | H-L | slow/fast |
| 14/15 | 30204 | U | slow/fast |
| 16/17 | 30204 | M | slow/fast |

## 3. 数据划分与统计口径

每个域包含 Normal、InnerRace、Ball 和 OuterRace 四类状态。域内以精确转速作为分组单位，将三个转速分别分配给训练、验证和测试集。四类故障使用一致的转速划分，且同一原始文件的滑动窗口不会跨集合出现。数据划分固定使用协议种子 1729；随机种子 42、52、62 仅控制参数初始化、批次顺序和回放抽样。

回放容量设为每类 20 个样本，共 80 个样本。由于主协议最多有 18 个已见域，该容量可在最终阶段为每个“类别×已见域”保留至少一个代表样本。所有使用回放的比较方法采用相同容量。

主结果报告三随机种子的均值和标准差，包括最终平均平衡准确率、平均增量准确率、平均遗忘、最大遗忘、后向迁移和旧域保持率。主表比较顺序微调、LwF、类别-域平衡回放和 SE-GSCL；消融实验分别移除跨工况对比、历史关系保持和故障-工况去相关约束。

## 4. 辅助协议

- `multidomain8`：保留 Risca 等复合环境组构造，用作重叠敏感复现实验。该设置的 A/B/C 环境组共享部分 H、U1 和 M1 原始记录，不作为独立域泛化的主要证据。
- `multidomain8_atomic`：将 8 种环境状态分别建域，共 48 域，用作长序列压力测试，不再作为论文主实验。

## 5. 云端运行

主对比运行：

```bash
cd /mnt/workspace/fdllm/se_gscl_impl
DATA_ROOT=/mnt/workspace/fdllm/data/MultiDomainBearing \
bash scripts/run_cloud_multidomain18_main.sh
```

运行全部消融时：

```bash
cd /mnt/workspace/fdllm/se_gscl_impl
DATA_ROOT=/mnt/workspace/fdllm/data/MultiDomainBearing \
JOBS=finetune,lwf_relation,experience_replay,se_gscl_full,wo_cross_condition,wo_relation,wo_decorrelation \
bash scripts/run_cloud_multidomain18_main.sh
```
