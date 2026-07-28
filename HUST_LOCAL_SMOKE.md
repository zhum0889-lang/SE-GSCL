# HUSTbearing Local Pipeline Smoke Test

## Scope

This milestone validates the local path from downloaded HUSTbearing signals to
continual semantic classification. It does not load Qwen2.5-7B and is not a
formal paper result.

## Data audit

- Source directory: `cross-condition-cl-llm/data/raw data-20260723T123200Z-1-001/raw data`
- Files: 99
- Size: about 1.18 GB
- Design: 9 health states by 11 operating conditions
- Record length: 262,144 rows
- Sampling rate: 25.6 kHz
- Input channels: X, Y, and Z vibration

The `.xls` files are tab-separated text exports rather than binary Excel
workbooks. Their numeric columns are time, tachometer, X, Y, and Z. The adapter
uses X/Y/Z and obtains the authoritative speed condition from each filename.

## Implemented chain

```text
HUST text export
-> filename label and condition parser
-> three-axis float32 signal
-> guarded time-block split
-> 2048-point sliding windows
-> first-domain training normalization
-> three-channel ConvLSTM
-> placeholder text prototypes
-> InfoNCE prototype alignment
-> fuzzy semantic probabilities
-> sequential condition training
-> replay memory
-> class prediction, confidence, and metrics
```

The nine labels are Healthy and medium/severe InnerRace, OuterRace, Ball, and
Compound faults. The variable-speed files use domain ID 0 and are excluded from
the first smoke run.

## Verified tests

Seven unit tests pass, including HUST filename parsing, three-axis window
preservation, ConvLSTM three-channel input, guarded split leakage checks,
normalization, persistent memory, and continual accuracy-matrix validation.

## Smoke results

### Single-domain diagnostic check

Run:

```text
results/hust_single_domain/20260724_105529
```

Configuration: 20 Hz, seed 42, 20 epochs, up to 60 windows per file.

```text
Accuracy: 55.56%
Macro-F1: 53.66%
```

This is above the 9-class random level of 11.11%, so the signal-to-semantic
classification path is learning from real data.

### Two-domain continual check

Run:

```text
results/hust_continual_smoke/20260724_110024
```

Configuration: 20 Hz -> 25 Hz, seed 42, 15 epochs per task.

| Strategy | Final D20 | Final D25 | ACC |
|---|---:|---:|---:|
| No replay | 52.22% | 74.44% | 63.33% |
| Balanced semantic replay | 57.78% | 72.22% | 65.00% |

Semantic replay improved old-condition retention by 5.56 percentage points but
reduced new-condition accuracy by 2.22 points. This is an early indication of
the stability-plasticity tradeoff, not a formal superiority claim.

## Outputs

Each strategy directory contains:

- `accuracy_matrix.csv`
- `episode_metrics.csv`
- `final_predictions.csv`
- `final_confusion_matrix.csv`
- `final_model.pt`
- versioned replay-memory snapshots

`final_predictions.csv` provides the true class, predicted class, confidence,
entropy, top-1/top-2 margin, and all nine class probabilities for each window.

## Known limitations

1. The current text encoder is a deterministic hashing placeholder with class
   anchors. It verifies interfaces but does not represent the formal Qwen
   semantic space.
2. HUSTbearing has one long record for each class-condition pair. Guarded
   contiguous blocks prevent overlapping-window leakage, but do not create
   independent physical trials.
3. Only one seed and two conditions were evaluated.
4. Confidence is not calibrated; current probabilities are suitable for
   ranking but not yet for reliability claims.

## Next implementation step

Replace the placeholder prototypes with frozen Qwen text hidden states, cache
the nine fault prototypes, and retain the current signal encoder and continual
protocol. After static Qwen-aligned classification is verified, add
fault/condition decoupled projectors and the previous-stage semantic
distribution preservation loss.
