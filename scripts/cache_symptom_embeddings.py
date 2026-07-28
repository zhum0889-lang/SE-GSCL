"""Cache frozen Qwen hidden states for hierarchical fault symptoms."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from se_gscl.semantics import (  # noqa: E402
    FrozenDecoderTextEncoder,
    SymptomEmbeddingCache,
    read_ontology,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ontology", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ontology = read_ontology(args.ontology)
    texts: list[str] = []
    class_ids: list[int] = []
    class_names: list[str] = []
    symptom_ids: list[str] = []
    symptom_names: list[str] = []
    for row in ontology["classes"]:
        class_names.append(str(row["name"]))
        symptoms = row.get("symptoms", [])
        if not symptoms:
            raise ValueError(f"Class {row['name']} has no local symptoms.")
        for symptom in symptoms:
            symptom_ids.append(str(symptom["id"]))
            symptom_names.append(str(symptom["name"]))
            texts.append(str(symptom["description"]))
            class_ids.append(int(row["id"]))

    encoder = FrozenDecoderTextEncoder(
        args.model,
        device=args.device,
        dtype=args.dtype,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    embeddings = encoder.encode(
        texts,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    import torch

    cache = SymptomEmbeddingCache(
        embeddings=embeddings,
        class_ids=torch.tensor(class_ids, dtype=torch.long),
        class_names=tuple(class_names),
        symptom_ids=tuple(symptom_ids),
        symptom_names=tuple(symptom_names),
        texts=tuple(texts),
        model_id=args.model,
        ontology=str(ontology["ontology"]),
        version=str(ontology["version"]),
    )
    output = cache.save(args.output_dir)
    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(output.resolve()),
                "model": args.model,
                "classes": len(class_names),
                "symptoms": cache.num_symptoms,
                "hidden_size": cache.hidden_size,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
