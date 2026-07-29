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

## P1 result visualization

The four-condition sequence runner now generates paper-facing PNG and vector
PDF figures automatically. Existing sequence results can be visualized without
retraining:

```bash
python scripts/visualize_p1_results.py \
  --root results/cloud_p1_sequence/<run_id> \
  --class-names Normal,InnerRace,Ball,OuterRace
```

Reports that predate sample-level output export produce strategy summaries,
stage curves, domain comparisons, class-recall heatmaps, accuracy matrices, and
loss curves. New runs also save `stage_outputs_after_domain_*.npz` and add
normalized confusion matrices, specialist-confidence diagnostics, and joint
t-SNE plots of signal embeddings and frozen text prototypes. These are P1
semantic-prototype classifier results; generative LLM diagnosis is not yet
enabled.

## P2 local symptom semantics

P2 adds hierarchical physical evidence without updating Qwen. The ontology now
separates global fault identity from local symptoms such as periodic impacts,
characteristic-frequency harmonics, rotational sidebands, and resonance
bursts. A shared frozen Qwen text projector maps both levels into the P1
semantic space.

First create a P2-ready P1 checkpoint. The current P1 runner saves
`projected_text_bank.pt` in addition to the frozen class prototypes:

```bash
STRATEGIES=full bash scripts/run_cloud_p1_sequence.sh
```

Then run the local symptom probe:

```bash
bash scripts/run_cloud_p2_local.sh
```

The main probe settings can be overridden without editing code, for example:

```bash
LOCAL_WEIGHT=0.2 TOP_TOKENS=6 ADAPTER_EPOCHS=8 \
  bash scripts/run_cloud_p2_local.sh
```

The probe freezes the P1 specialist, trains only a lightweight local-token
adapter on the initial condition, and evaluates all four conditions. It writes
global, local, and fused metrics to `p2_report.json`; sample-level Top-k fault
candidates, uncertainty, agreement, and Top symptom evidence are written to
`evaluation_predictions.jsonl`. This structured packet is the input contract
for the later continuous-prompt Qwen stage. The same command also generates
PNG and editable-vector PDF figures under `figures/`, including branch
comparisons, fused confusion matrices, uncertainty and branch agreement, and a
fault-to-symptom evidence heatmap.

## P2.1 physics-guided local alignment

P2.1 grounds the twelve local textual symptoms with scale-invariant signal
attributes. For CWRU, the loader reads the sample-level shaft speed from each
MAT file and uses the drive-end bearing kinematics to calculate Hilbert-envelope
BPFI, BPFO, and BSF order prominence and sidebands. Time-domain stationarity,
impulsiveness, and high-frequency resonance attributes complement the order
features.

The physical attributes are robustly normalized with statistics fitted only on
the initial-condition training split. They form class-gated soft symptom
targets. P2.1 freezes the P1 specialist and global semantic projector, while an
independent local symptom projector and token adapter are optimized using:

```text
local class loss + physics-guided symptom loss + prototype anchor loss
```

Run the cloud probe with:

```bash
bash scripts/run_cloud_p21_physics.sh
```

The runner reuses the latest P2-ready P1 `full` checkpoint. Its report adds
weighted symptom error, true-class symptom error, and physical/semantic Top-1
agreement for every condition. The generated
`p21_physical_target_vs_prediction` figure compares physics-derived targets
with learned symptom probabilities and is the primary audit for explanation
grounding. This remains a local-semantic experiment; Qwen text generation is
not enabled.

## P2.2 semantic guard and adaptive fusion

P2.2 addresses the two failure modes exposed by P2.1: unconstrained symptom
prototype drift and fixed-weight fusion. The Qwen-derived symptom prototypes
are frozen as semantic anchors. A trainable residual with norm below `0.2` is
added to each prototype, preventing the local branch from turning text
prototypes into unrestricted classifier weights.

Within each ground-truth fault class, a KL loss aligns the relative
distribution of the three predicted symptoms with the physics-derived soft
distribution. Training uses only the initial condition, while its validation
split controls early stopping. After the best checkpoint is restored, the same
validation split calibrates a conservative reliability gate from branch
entropy, Top-1 margin, and global/local agreement. Test labels are never used
for checkpoint selection or fusion calibration.

Run:

```bash
bash scripts/run_cloud_p22_semantic_guard.sh
```

The report records the selected epoch, validation gate, sample-level fusion
weights, physical grounding metrics, and final anchor cosine similarity. The
additional `p22_semantic_guard_and_reliability_gate` figure audits training,
model selection, and local-branch activation across conditions.

After the single full run is validated, the controlled P2.2 ablations can be
launched with:

```bash
bash scripts/run_cloud_p22_ablation.sh
```

It compares semantic guarding without within-class ranking, semantic guarding
with fixed fusion, and the complete P2.2 configuration. P2.1 remains the
unconstrained-projector reference.

If training finishes but figure generation stops because `matplotlib` is
missing, install it into the same Python environment and regenerate figures
without repeating training:

```bash
python -m pip install "matplotlib>=3.7"
P22_DIR="$(find results/cloud_p22_semantic_guard -mindepth 1 -maxdepth 1 \
  -type d | sort | tail -n 1)"
python scripts/visualize_p2_results.py \
  --root "$P22_DIR" \
  --formats png,pdf \
  --dpi 300
```

## P3.0 frozen-Qwen diagnostic generation baseline

P3.0 validates the complete diagnostic output chain before training a
continuous-vector prompt adapter. It reads the P2.2 semantic diagnostic
packets and supplies Qwen with only Top-k fault probabilities, branch
agreement, uncertainty, physically grounded soft symptom evidence, and a
restricted maintenance-action set. Ground-truth labels and correctness flags
are removed before prompt construction.

Qwen remains frozen and deterministically returns a constrained JSON
diagnosis. The evaluator reports JSON/schema validity, candidate-label
validity, accuracy, agreement with the upstream specialist, evidence
grounding, maintenance-action validity, and whether uncertain samples
explicitly acknowledge uncertainty.

P3.0.1 further separates supporting evidence from counter-evidence and carries
the fault-class provenance of every symptom into the prompt. Its evaluator
therefore distinguishes simple name copying from diagnosis-consistent
evidence use. Maintenance actions are also checked against a conservative
policy: uncertain cases require verification, confident healthy cases continue
monitoring, and confident fault cases require scheduled inspection.

P3.0.2 adds a deterministic semantic consistency controller after generation.
Qwen still selects a diagnosis and writes the natural-language explanation.
The controller checks only ontology-verifiable fields: candidate validity,
support/counter-evidence polarity, uncertainty, confidence level, and the
maintenance policy. Every repair is recorded. Reports preserve both raw LLM
metrics and controlled-system metrics so reliability gains are not attributed
to the LLM alone.

Existing P3.0.1 generations can be audited without loading Qwen again:

```bash
python scripts/apply_p3_semantic_control.py --p3-dir <existing_p3_dir>
```

## P3.1 direct continuous semantic prompting

P3.1 tests whether Qwen can consume engineering semantics as continuous input
embeddings instead of receiving fault names and symptom descriptions as text.
The P2.2 runner now exports source-condition training and validation features
separately from test outputs. The prompt adapter is trained only on the source
condition and selected by source validation loss; test labels are never used
for training or checkpoint selection.

Each adapter input concatenates the 256-dimensional fuzzy symptom embedding,
the four-class fused posterior, normalized entropy, Top-1/Top-2 margin, and
global/local agreement. A shared low-rank adapter maps this 263-dimensional
context to four Qwen input tokens. Qwen remains fully frozen. The first probe
asks Qwen to generate exactly one fault label, isolating whether continuous
vectors carry diagnostic information before explanation generation is added.
The fuzzy semantic block is normalized per sample with an L2 norm, while
posterior and reliability values retain their bounded physical scale. No
source-domain per-dimension standardization is used, because low-variance
source features can amplify cross-condition shift.

First rerun P2.2 once with the current code to create
`p2_prompt_train.npz` and `p2_prompt_validation.npz`, then launch the small
cloud probe:

```bash
bash scripts/run_cloud_p22_semantic_guard.sh
bash scripts/run_cloud_p31_continuous_prompt.sh
```

The default P3.1 command uses 32 training, 16 validation, and 32 stratified
test samples for one epoch. These are connectivity settings, not paper
results. After the direct-vector path is validated, set the sample limits to
zero and increase the epoch count for the full source-to-cross-condition run.

Run a 32-sample condition-balanced probe:

```bash
bash scripts/run_cloud_p3_frozen_qwen.sh
```

This is a structured-text baseline, not the final direct-vector method. Once
its output contract is reliable, P3.1 will learn a lightweight adapter that
maps the saved 256-dimensional fuzzy semantic embeddings in `p2_outputs.npz`
to continuous Qwen prompt tokens, while keeping the Qwen backbone frozen.
