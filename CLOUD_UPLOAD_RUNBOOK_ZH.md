# SE-GSCL 云端 Git 部署与运行说明

## 1. 首次拉取

在云端工作区执行：

```bash
cd /mnt/workspace/fdllm
git clone https://github.com/zhum0889-lang/SE-GSCL.git se_gscl_impl
cd se_gscl_impl
```

如果仓库为私有仓库，GitHub 要求登录时应使用个人访问令牌，
不要把令牌写入脚本、配置或仓库。

后续同步代码：

```bash
cd /mnt/workspace/fdllm/se_gscl_impl
git pull --ff-only origin main
```

仓库不跟踪 CWRU、Qwen 权重、运行结果和 Python 缓存。默认复用：

```text
/mnt/workspace/fdllm/data/CWRU
/mnt/workspace/fdllm/models/Qwen/
```

## 2. 检查依赖

```bash
python -c "import torch, transformers, numpy, scipy, sklearn"
```

缺少依赖时：

```bash
pip install -r requirements.txt
```

## 3. 一键运行 P1 Smoke

```bash
bash scripts/run_cloud_p1_smoke.sh
```

脚本依次执行：

1. 检查 PyTorch、Transformers、CUDA 和本地 Qwen2.5-7B；
2. 运行单元测试；
3. 审计 CWRU4 数据；
4. 首次运行时生成冻结 Qwen 文本语义缓存；
5. 执行 CWRU 工况 0 到工况 1 的两阶段训练；
6. 保存冻结原型、旧关系快照、轻量专家和实验报告。

## 4. 路径不同时

通过环境变量覆盖默认值：

```bash
QWEN_PATH=/actual/path/Qwen2.5-7B-Instruct \
CWRU_ROOT=/actual/path/CWRU \
bash scripts/run_cloud_p1_smoke.sh
```

显存不支持 BF16 时：

```bash
DTYPE=float16 bash scripts/run_cloud_p1_smoke.sh
```

更换随机种子或训练轮数：

```bash
SEED=43 INITIAL_EPOCHS=10 CONTINUAL_EPOCHS=10 \
bash scripts/run_cloud_p1_smoke.sh
```

## 5. 输出

文本语义缓存：

```text
results/semantic_cache/qwen25_7b_bearing4_v1/
```

训练结果：

```text
results/cloud_p1/cwru4_d0_d1_qwen7b_seed*/
```

当前脚本用于验证两工况 P1 闭环。完整四工况、多种子和基线消融需要后续的顺序实验 runner，不能将当前 Smoke 数值直接作为论文结果。

## 6. 四工况快速对比

依次运行顺序微调、平衡回放和完整方法：

```bash
bash scripts/run_cloud_p1_sequence.sh
```

默认每个策略执行 10 个初始化 epoch 和每个新工况 10 个持续 epoch。
快速连通性检查可以缩短为：

```bash
INITIAL_EPOCHS=3 CONTINUAL_EPOCHS=3 \
bash scripts/run_cloud_p1_sequence.sh
```

完成后终端会打印并保存：

```text
results/cloud_p1_sequence/<run_id>/comparison.json
```

将该 JSON 返回即可继续分析。单种子快速对比仍不是论文正式结果。
