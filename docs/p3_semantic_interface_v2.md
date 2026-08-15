# P3 faceted semantic interface v2

## Purpose

The revised interface addresses two different bottlenecks instead of treating
them as one problem:

1. The frozen text branch now encodes five balanced semantic facets for every
   fault class: identity, mechanism, signal signature, condition response, and
   disambiguation evidence.
2. P2 does not immediately average those texts into one hard class decision.
   It preserves the global class posterior, distributes each class probability
   over its descriptions, and exports a fuzzy identity embedding to P3.

The local fuzzy symptom embedding remains a separate input. P3 therefore sees
global fuzzy identity, local fuzzy symptoms, fused posterior, reliability, and
optional observable condition features.

## Cloud cache preparation

```bash
cd /mnt/workspace/fdllm/se_gscl_impl

MODEL=/mnt/workspace/fdllm/models/Qwen/Qwen2___5-7B-Instruct
ONTOLOGY=configs/semantics/bearing_faults_4_v2.json

python scripts/cache_text_embeddings.py \
  --model "$MODEL" \
  --ontology "$ONTOLOGY" \
  --output-dir results/semantic_cache/qwen25_7b_bearing4_faceted_v2 \
  --device cuda \
  --dtype bfloat16 \
  --batch-size 8 \
  --local-files-only

python scripts/cache_symptom_embeddings.py \
  --model "$MODEL" \
  --ontology "$ONTOLOGY" \
  --output-dir results/semantic_cache/qwen25_7b_bearing4_symptoms_physics_v2 \
  --device cuda \
  --dtype bfloat16 \
  --batch-size 8 \
  --local-files-only

python scripts/audit_semantic_ontology.py \
  --ontology "$ONTOLOGY" \
  --text-cache results/semantic_cache/qwen25_7b_bearing4_faceted_v2 \
  --output results/semantic_audit/bearing_faults_4_v2_qwen.json
```

Because the global prototype bank changes, P1, P2, and P3 must be rerun in
order. Reusing a v1 P1 projector with the v2 cache is not a valid comparison.

## Required ablations

| Job | Global fuzzy identity | Local fuzzy symptoms | Condition | LLM tuning |
|---|---:|---:|---:|---|
| `continuous_no_fuzzy_identity` | no | yes | no | frozen |
| `continuous_identity_only` | yes | no | no | frozen |
| `continuous_no_condition` | yes | yes | no | frozen |
| `continuous_full` | yes | yes | yes | frozen |
| `continuous_full_lora` | yes | yes | yes | LoRA |

The main claim is supported only if adding global fuzzy identity improves the
paired correction-corruption balance across all three seeds. Text quality must
also be reported with the ontology audit and the frozen-embedding separation
statistics; description count alone is not evidence of better semantics.
