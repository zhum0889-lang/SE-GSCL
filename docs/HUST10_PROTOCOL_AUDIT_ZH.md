# HUSTbearing 十工况持续学习协议

## 1. 主实验序列

HUSTbearing 主实验采用十个固定转速工况，按下列顺序持续到达：

`20 -> 25 -> 30 -> 35 -> 40 -> 60 -> 65 -> 70 -> 75 -> 80 Hz`

每个工况包含 9 类设备状态，共应形成 `10 x 9 = 90` 个原始记录。`VS_0_40_0Hz` 变速记录暂不加入主训练序列，保留用于后续开放工况或非平稳转速泛化实验。

## 2. 窗口与防泄漏划分

默认窗口长度为 2048，步长为 1024，每个原始文件最多保留 60 个窗口。由于相邻窗口存在 50% 重叠，训练、验证和测试区间之间各设置 2 个窗口的保护带。单个完整文件预计得到：

- 训练集：34 个窗口
- 验证集：8 个窗口
- 测试集：10 个窗口
- 保护带排除：8 个窗口

全部 90 个文件预计产生 5400 个候选窗口，其中训练 3060、验证 720、测试 900，另有 720 个边界窗口被排除。归一化参数只能由首个工况的训练划分拟合，不能使用验证集、测试集或未来工况。

## 3. 本地审计

审计脚本只读取 `.xls` 文件头部的 `Total Data Rows`，不会加载约 1.2 GB 的完整信号：

```powershell
python scripts/audit_hustbearing_protocol.py `
  --data-root "E:\mypaper\research-paper-pipeline\cross-condition-cl-llm\data\raw data-20260723T123200Z-1-001\raw data"
```

通过标准为：

- `protocol_ready = true`
- `complete_domain_class_grid = true`
- `expected_records = observed_records = 90`
- 无缺失或重复的工况—类别组合
- 每个文件均能产生非空训练、验证和测试划分

输出位于 `results/protocol_audit/hustbearing10/`，包括逐文件 CSV 和汇总 JSON。

## 4. 云端审计

```bash
cd /mnt/workspace/fdllm/se_gscl_impl

python scripts/audit_hustbearing_protocol.py \
  --data-root /mnt/workspace/fdllm/data/hustbearing \
  --output-dir results/protocol_audit/hustbearing10
```

只有审计通过后才运行十工况 P1。后续每个随机种子必须输出 `10 x 10` 准确率矩阵，并据此计算最终平均准确率、平均增量准确率、平均遗忘、最大遗忘、后向迁移和旧工况保持率。

```bash
python scripts/run_paper_p1_matrix.py \
  --dataset hustbearing \
  --data-root /mnt/workspace/fdllm/data/hustbearing \
  --text-cache results/semantic_cache/qwen25_7b_hust9_v1 \
  --output-root results/paper_matrix_hust10/p1 \
  --seeds 42,52,62 \
  --device cuda \
  --execute
```

该入口会自动再次执行协议审计。完整 P1 矩阵包含 7 种策略或消融设置、3 个随机种子，共 21 个 `p1_report.json`：

```bash
find results/paper_matrix_hust10/p1/hustbearing \
  -name p1_report.json | wc -l

find results/paper_matrix_hust10/p1/hustbearing \
  -name accuracy_matrix_seen_only.csv | wc -l
```

两项计数均应为 21。不要复用原四工况目录作为输出目录，否则已有报告会触发跳过逻辑。
