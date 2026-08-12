"""Audit raw-record reuse in the MultiDomainBearing 18-domain protocol.

The published compound-domain construction intentionally reuses some
environment recordings across domain definitions. This utility quantifies that
reuse before continual-learning results are interpreted as independent-domain
evidence. It does not modify data or splits.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fdllm_repro.datasets import RawRecord, load_records  # noqa: E402


def _parse_domains(value: str) -> list[int]:
    domains = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not domains or len(set(domains)) != len(domains):
        raise ValueError("--domains must contain unique integer domain IDs.")
    return domains


def _domain_name(domain: int) -> str:
    bearings = ("6204", "N204/NJ204", "30204")
    environments = ("A", "B", "C")
    speed_groups = ("slow", "fast")
    if domain < 0 or domain >= 18:
        return f"domain_{domain}"
    return (
        f"D{domain}: {bearings[domain // 6]} | "
        f"env_{environments[(domain % 6) // 2]} | "
        f"{speed_groups[domain % 2]}"
    )


def _source_id(record: RawRecord) -> str:
    return str(record.source_record_id or record.file_id)


def _pair_rows(
    sources_by_domain: dict[int, set[str]],
    domains: Iterable[int],
) -> list[dict[str, object]]:
    ordered = list(domains)
    rows: list[dict[str, object]] = []
    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1 :]:
            overlap = sources_by_domain[left] & sources_by_domain[right]
            union = sources_by_domain[left] | sources_by_domain[right]
            if overlap:
                rows.append(
                    {
                        "left_domain": left,
                        "right_domain": right,
                        "left_name": _domain_name(left),
                        "right_name": _domain_name(right),
                        "shared_source_records": len(overlap),
                        "jaccard": len(overlap) / max(1, len(union)),
                        "examples": sorted(overlap)[:5],
                    }
                )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--domains", default=",".join(str(i) for i in range(18)))
    parser.add_argument("--sampling-rate", type=int, choices=(8000, 16000), default=8000)
    parser.add_argument(
        "--fail-on-overlap",
        action="store_true",
        help="Return a non-zero status when one raw record occurs in multiple domains.",
    )
    args = parser.parse_args()

    domains = _parse_domains(args.domains)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records("multidomain8" if args.sampling_rate == 8000 else "multidomain16", Path(args.data_root), domains=domains)
    sources_by_domain: dict[int, set[str]] = {domain: set() for domain in domains}
    classes_by_domain: dict[int, Counter[str]] = {domain: Counter() for domain in domains}
    domains_by_source: dict[str, set[int]] = defaultdict(set)
    for record in records:
        domain = int(record.domain_id)
        source = _source_id(record)
        sources_by_domain.setdefault(domain, set()).add(source)
        classes_by_domain.setdefault(domain, Counter())[record.label_name] += 1
        domains_by_source[source].add(domain)

    repeated = {
        source: sorted(source_domains)
        for source, source_domains in domains_by_source.items()
        if len(source_domains) > 1
    }
    pairs = _pair_rows(sources_by_domain, domains)
    domain_rows = [
        {
            "domain": domain,
            "name": _domain_name(domain),
            "source_records": len(sources_by_domain[domain]),
            "class_records": dict(sorted(classes_by_domain[domain].items())),
        }
        for domain in domains
    ]
    report = {
        "status": "warning" if repeated else "ok",
        "dataset": "multidomain8" if args.sampling_rate == 8000 else "multidomain16",
        "domains": domains,
        "raw_records_after_domain_expansion": len(records),
        "unique_source_records": len(domains_by_source),
        "reused_source_records": len(repeated),
        "reused_source_fraction": len(repeated) / max(1, len(domains_by_source)),
        "overlapping_domain_pairs": len(pairs),
        "domain_summary": domain_rows,
        "pairwise_overlap": pairs,
        "reused_source_examples": [
            {"source_record_id": source, "domains": source_domains}
            for source, source_domains in sorted(repeated.items())[:30]
        ],
        "interpretation": (
            "Repeated source records make compound domains statistically dependent. "
            "Report this protocol as an overlap-sensitive reproduction, and use "
            "source-disjoint evaluation for any claim of independent-domain generalization."
            if repeated
            else "No raw recording is shared across requested domains."
        ),
    }
    path = output_dir / "multidomain_overlap_audit.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote audit: {path}")
    return 2 if args.fail_on_overlap and repeated else 0


if __name__ == "__main__":
    raise SystemExit(main())
