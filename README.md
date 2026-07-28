# SE-GSCL Implementation

This directory is a clean implementation workspace copied from the reusable
parts of `E:\故障检测大模型\llm\fdllm`. The source repository is unchanged.

Repository:

```text
https://github.com/zhum0889-lang/SE-GSCL
```

Cloud deployment and update commands are documented in
`CLOUD_UPLOAD_RUNBOOK_ZH.md`.

## P0 protocol fixes

- leakage-resistant file-group split with guarded blocked fallback for CWRU;
- bearing-ID group split for Paderborn when at least three bearings per class are available;
- normalization fitted only on the first domain's training split;
- persistent capacity-bounded replay memory;
- seen-domain-only accuracy matrices;
- multi-seed execution and mean/std summaries.
- unified record manifests with source, bearing, sensor, label, domain, speed,
  torque, force, and content hash metadata.

## New SE-GSCL package

The new implementation is being built under `src/se_gscl`; `src/fdllm_repro`
remains available as the runnable baseline. The current P0/P1 implementation provides:

```text
src/se_gscl/data/       unified manifest schema and audit export
src/se_gscl/models/     multi-scale token encoder and fault/condition branches
src/se_gscl/continual/  class-domain batch sampling and old-relation snapshots
src/se_gscl/losses/     alignment, cross-condition, decorrelation, relation losses
src/se_gscl/semantics/  frozen Qwen encoding, caches, projection, prototype bank
src/se_gscl/training/   staged P1 global-semantic trainer
configs/datasets/       frozen CWRU4/CWRU10/Paderborn/HUST protocols
configs/experiments/    staged MVP experiment settings
scripts/prepare_dataset.py
scripts/smoke_specialist.py
scripts/cache_text_embeddings.py
scripts/train_p1_global.py
```

Create a pre-windowing audit manifest with:

```powershell
$env:PYTHONPATH="src;."
& "C:\Users\14352\miniconda3\python.exe" .\scripts\prepare_dataset.py `
  --dataset cwru4 `
  --data-root .\data\CWRU `
  --output-dir .\results\data_audit\cwru4
```

Paderborn supports the four official condition tuples used in the experiment
plan and reads the `vibration_1` channel. The local workspace currently contains
only healthy bearing `K001`; it is sufficient for parser connectivity but not
for a three-class paper experiment. Download multiple healthy, inner-race, and
outer-race bearing IDs before formal bearing-group evaluation.

Verify the new specialist tensor path on real CWRU windows:

```powershell
$env:PYTHONPATH="src;."
& "C:\Users\14352\miniconda3\python.exe" .\scripts\smoke_specialist.py `
  --dataset cwru4 `
  --data-root .\data\CWRU `
  --domains 0,1 `
  --token-dim 64 `
  --num-tokens 16
```

This command uses random frozen prototypes to test shapes and gradients only.
It is not a training result. Formal P1 replaces them with cached hidden-state
representations from the declared frozen text model.

## P1 real semantic smoke

Build a versioned cache from the frozen Qwen text encoder:

```powershell
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
& "C:\Users\14352\miniconda3\python.exe" .\scripts\cache_text_embeddings.py `
  --model "Qwen/Qwen2.5-0.5B-Instruct" `
  --ontology .\configs\semantics\bearing_faults_4.json `
  --output-dir .\results\semantic_cache\qwen25_05b_bearing4_v1 `
  --cache-dir "E:\故障检测大模型\llm\fdllm\models\hf_cache" `
  --local-files-only
```

Run initial-domain alignment followed by one continual domain update:

```powershell
& "C:\Users\14352\miniconda3\python.exe" .\scripts\train_p1_global.py `
  --dataset cwru4 `
  --data-root .\data\CWRU `
  --text-cache .\results\semantic_cache\qwen25_05b_bearing4_v1 `
  --domains 0,1 `
  --output-dir .\results\p1_global_smoke\cwru4_d0_d1_qwen05b_centered `
  --device cpu
```

The initial stage updates the signal specialist and text projector while Qwen
remains frozen. The projected class prototypes are then frozen. The continual
stage updates only the signal specialist using current-domain samples,
class-balanced replay, cross-condition positives, and pre-update global
relation snapshots. The one-epoch output is a connectivity check, not a paper
benchmark.

On the cloud workspace, replace `--model` with
`/mnt/workspace/fdllm/models/Qwen/Qwen2.5-7B-Instruct`, use
`--device cuda --dtype bfloat16`, and keep the same ontology and output schema.

## Layout

```text
implementation/
  src/fdllm_repro/                 copied signal/data/text modules
  experiments/continual_fdllm/     copied and revised continual modules
  tests/                            protocol unit tests
  results/                          generated experiment outputs
```

## Smoke run

```powershell
$env:PYTHONPATH="src;."
& "C:\Users\14352\miniconda3\python.exe" -m experiments.continual_fdllm.run_p0_protocol `
  --dataset cwru4 `
  --data-root ".\data\CWRU" `
  --domain-order 0,1 `
  --seeds 42 `
  --strategies no_replay,random_replay `
  --max-windows-per-file 30 `
  --smoke `
  --device cuda
```

## Formal run

```powershell
$env:PYTHONPATH="src;."
& "C:\Users\14352\miniconda3\python.exe" -m experiments.continual_fdllm.run_p0_protocol `
  --data-root ".\data\CWRU" `
  --domain-order 0,1,2,3 `
  --seeds 42,43,44,45,46 `
  --strategies no_replay,random_replay,balanced_semantic_replay `
  --memory-ratio 0.10 `
  --memory-max 400 `
  --device cuda
```

Every run writes split audits, frozen normalization statistics, persistent
memory snapshots, seen-only accuracy matrices, checkpoints, per-seed metrics,
and aggregate mean/std tables.

## Dataset download

The selected cross-condition datasets can be downloaded reproducibly:

```powershell
& "C:\Users\14352\miniconda3\python.exe" .\scripts\download_datasets.py `
  --dataset all `
  --mode pilot
```

`pilot` downloads three representative Paderborn bearing archives (healthy,
outer-race fault, and inner-race fault) plus the public HUSTbearing folder.
Use `--dataset paderborn --mode full` for all 32 Paderborn archives (about
5 GB). Downloads are stored under `data/Paderborn` and `data/HUSTbearing` and
are excluded from Git. Paderborn downloads resume from `.part` files.

To inspect the planned transfer without downloading:

```powershell
& "C:\Users\14352\miniconda3\python.exe" .\scripts\download_datasets.py `
  --dataset all `
  --mode pilot `
  --dry-run
```

## HUSTbearing local smoke run

The HUSTbearing adapter reads the dataset's tab-separated `.xls` text exports,
keeps the X/Y/Z vibration channels, and parses the nine health states and speed
domains from filenames.

```powershell
$env:PYTHONPATH="src;."
& "C:\Users\14352\miniconda3\python.exe" `
  -m experiments.continual_fdllm.run_p0_protocol `
  --dataset hustbearing `
  --data-root "..\data\raw data-20260723T123200Z-1-001" `
  --domain-order 20,25 `
  --seeds 42 `
  --strategies no_replay,balanced_semantic_replay `
  --window-size 2048 `
  --step-size 1024 `
  --max-windows-per-file 60 `
  --epochs 15 `
  --device cuda `
  --out-dir results\hust_continual_smoke
```

This milestone uses deterministic hashing-based text prototypes only as a
pipeline placeholder. Formal experiments must replace them with frozen Qwen
text representations. Every run records this limitation in
`text_prototypes.json`.

## Verified smoke output

The first real-data connectivity run is stored under:

```text
results/p0_smoke/20260720_203101
```

It used two domains, one seed, one training epoch, and 96 samples per split.
Its scores are only a pipeline check, not a paper result.
