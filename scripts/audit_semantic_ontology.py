"""Audit faceted fault texts and optional frozen-encoder separability."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from se_gscl.semantics import TextEmbeddingCache, read_ontology  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ontology", required=True)
    parser.add_argument("--text-cache")
    parser.add_argument("--output")
    return parser.parse_args()


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _lexical_overlap(rows: list[tuple[int, str]]) -> dict[str, float]:
    within: list[float] = []
    between: list[float] = []
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            left_tokens = _tokens(rows[left][1])
            right_tokens = _tokens(rows[right][1])
            union = left_tokens | right_tokens
            score = len(left_tokens & right_tokens) / max(1, len(union))
            target = within if rows[left][0] == rows[right][0] else between
            target.append(score)
    return {
        "mean_within_class_jaccard": float(np.mean(within)) if within else 0.0,
        "mean_between_class_jaccard": float(np.mean(between)) if between else 0.0,
    }


def _embedding_audit(cache: TextEmbeddingCache) -> dict[str, object]:
    values = cache.embeddings.float().cpu().numpy()
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)
    centers = cache.class_means().float().cpu().numpy()
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-8)
    center_cosine = centers @ centers.T
    similarities = values @ values.T
    np.fill_diagonal(similarities, -np.inf)
    nearest = similarities.argmax(axis=1)
    ids = cache.class_ids.cpu().numpy()
    nearest_same_class = ids[nearest] == ids
    same: list[float] = []
    different: list[float] = []
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            target = same if ids[left] == ids[right] else different
            target.append(float(values[left] @ values[right]))
    return {
        "hidden_size": cache.hidden_size,
        "description_nearest_neighbor_same_class_rate": float(
            nearest_same_class.mean()
        ),
        "mean_within_class_cosine": float(np.mean(same)) if same else 0.0,
        "mean_between_class_cosine": (
            float(np.mean(different)) if different else 0.0
        ),
        "within_between_margin": (
            float(np.mean(same) - np.mean(different))
            if same and different
            else 0.0
        ),
        "class_center_cosine": center_cosine.tolist(),
    }


def main() -> int:
    args = parse_args()
    ontology = read_ontology(args.ontology)
    rows: list[tuple[int, str]] = []
    classes = []
    all_texts: list[str] = []
    for row in ontology["classes"]:
        descriptions = list(row["descriptions"])
        texts = [str(value["text"]) for value in descriptions]
        all_texts.extend(texts)
        rows.extend((int(row["id"]), text) for text in texts)
        classes.append(
            {
                "id": int(row["id"]),
                "name": str(row["name"]),
                "descriptions": len(descriptions),
                "description_types": sorted(
                    str(value["type"]) for value in descriptions
                ),
                "symptoms": len(row.get("symptoms", ())),
                "prompt_summary_words": len(
                    _tokens(str(row.get("prompt_summary", "")))
                ),
            }
        )
    normalized = [" ".join(sorted(_tokens(text))) for text in all_texts]
    result: dict[str, object] = {
        "status": "ok",
        "ontology": ontology["ontology"],
        "version": ontology["version"],
        "classes": classes,
        "description_count": len(all_texts),
        "duplicate_normalized_descriptions": len(normalized) - len(set(normalized)),
        "lexical_separation": _lexical_overlap(rows),
    }
    if args.text_cache:
        cache = TextEmbeddingCache.load(args.text_cache)
        if cache.ontology != ontology["ontology"] or cache.version != ontology["version"]:
            raise ValueError(
                "Text cache ontology/version does not match the requested ontology."
            )
        result["embedding_separation"] = _embedding_audit(cache)
    rendered = json.dumps(result, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
