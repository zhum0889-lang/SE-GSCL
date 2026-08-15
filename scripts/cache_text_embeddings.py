"""Cache frozen Qwen hidden states for versioned fault descriptions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from se_gscl.semantics.text_cache import TextEmbeddingCache, read_ontology  # noqa: E402
from se_gscl.semantics.text_encoder import FrozenDecoderTextEncoder  # noqa: E402


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
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ontology = read_ontology(args.ontology)
    texts: list[str] = []
    class_ids: list[int] = []
    class_names: list[str] = []
    description_ids: list[str] = []
    description_types: list[str] = []
    class_summaries: list[str] = []
    for row in ontology["classes"]:
        class_names.append(str(row["name"]))
        class_summaries.append(str(row.get("prompt_summary", "")))
        for description in row["descriptions"]:
            texts.append(str(description["text"]))
            class_ids.append(int(row["id"]))
            description_ids.append(str(description["id"]))
            description_types.append(str(description["type"]))

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

    cache = TextEmbeddingCache(
        embeddings=embeddings,
        class_ids=torch.tensor(class_ids, dtype=torch.long),
        class_names=tuple(class_names),
        texts=tuple(texts),
        model_id=args.model,
        ontology=str(ontology["ontology"]),
        version=str(ontology["version"]),
        description_ids=tuple(description_ids),
        description_types=tuple(description_types),
        class_summaries=tuple(class_summaries),
    )
    output = cache.save(args.output_dir)
    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(output.resolve()),
                "model": args.model,
                "classes": len(class_names),
                "descriptions": len(texts),
                "description_types": sorted(set(description_types)),
                "hidden_size": cache.hidden_size,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
