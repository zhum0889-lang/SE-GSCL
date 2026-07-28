# SE-GSCL 云端上传与运行说明

## 1. 解压

将压缩包上传到 `/mnt/workspace/fdllm`，然后执行：

```bash
cd /mnt/workspace/fdllm
unzip se_gscl_cloud_upload.zip
cd se_gscl_impl
```

压缩包不包含 CWRU、Qwen 权重、运行结果和 Python 缓存。默认复用：

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
