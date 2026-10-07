#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from physioagent.kg.scope.filter import DEFAULT_PROFILE, load_scope, profiles

KG_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA = KG_DIR / "data"

TOP_K_COEXISTING = 20
SIMILARITY_THRESHOLD = 5
LARGE_SET = 10


def load_concept_sets(path: Path) -> list[tuple[str, ...]]:
    sets: list[tuple[str, ...]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            concepts = set(record.get("conditions", [])) | set(record.get("procedures", [])) \
                | set(record.get("drugs", []))
            if concepts:
                sets.append(tuple(sorted(concepts)))
    return sets


def coexistence(sets: Sequence[tuple[str, ...]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for concepts in sets:
        for a in concepts:
            for b in concepts:
                if a != b:
                    counts[a][b] += 1
    return {a: dict(c) for a, c in counts.items()}


def top_coexisting(counts: dict[str, dict[str, int]],
                   k: int = TOP_K_COEXISTING) -> dict[str, list[str]]:
    top: dict[str, list[str]] = {}
    for concept, neighbours in counts.items():
        ranked = sorted(neighbours.items(), key=lambda kv: (-kv[1], kv[0]))
        top[concept] = [name for name, _ in ranked[:k]]
    return top


def filter_similar_sets(distinct: Sequence[tuple[tuple[str, ...], int]],
                        threshold: int = SIMILARITY_THRESHOLD,
                        large: int = LARGE_SET) -> list[tuple[str, ...]]:
    vocab = {name: 1 << i
             for i, name in enumerate(sorted({n for s, _ in distinct for n in s}))}
    masks = [(mask_of(s, vocab), s) for s, _ in distinct]

    kept: list[tuple[str, ...]] = []
    processed: set[int] = set()
    for i, (mask_i, set_i) in enumerate(masks):
        if i in processed:
            continue
        if len(set_i) > large:
            for j, (mask_j, set_j) in enumerate(masks):
                if i != j and len(set_j) > large \
                        and bin(mask_i ^ mask_j).count("1") < threshold:
                    processed.add(j)
        kept.append(set_i)
    return kept


def mask_of(concepts: Iterable[str], vocab: dict[str, int]) -> int:
    mask = 0
    for name in concepts:
        mask |= vocab[name]
    return mask


def write_seed_concept_sets(data_dir: Path, scope) -> Path:
    path = data_dir / "seed_concept_sets.json"
    path.write_text(
        json.dumps({theme: list(terms) for theme, terms in scope.seed_terms.items()},
                   indent=2), encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA),
                    help="where build_visits wrote visits_all.jsonl")
    ap.add_argument("--profile", default=DEFAULT_PROFILE, choices=profiles())
    ap.add_argument("--top-k", type=int, default=TOP_K_COEXISTING)
    ap.add_argument("--threshold", type=int, default=SIMILARITY_THRESHOLD)
    ap.add_argument("--seeds-only", action="store_true",
                    help="rewrite seed_concept_sets.json from the ontology and stop. "
                         "The visit artifacts need MIMIC and a cluster; this one needs "
                         "neither, so an ontology edit does not have to wait for a job.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    scope = load_scope(args.profile)

    if args.seeds_only:
        data_dir.mkdir(parents=True, exist_ok=True)
        path = write_seed_concept_sets(data_dir, scope)
        print(f"{len(scope.seed_terms)} themes, "
              f"{sum(len(t) for t in scope.seed_terms.values())} terms -> {path}")
        return 0

    visits_path = data_dir / "visits_all.jsonl"
    if not visits_path.exists():
        raise SystemExit(f"{visits_path} not found -- run build_visits.py first")

    sets = load_concept_sets(visits_path)
    print(f"{len(sets):,} admissions with at least one in-scope concept")

    counts = coexistence(sets)
    top = top_coexisting(counts, args.top_k)
    print(f"{len(top)} concepts have at least one co-occurring concept")

    distinct = Counter(sets)
    ordered = sorted(distinct.items(), key=lambda kv: (-kv[1], kv[0]))
    print(f"{len(ordered):,} distinct concept sets "
          f"({len(sets) / max(len(ordered), 1):.1f} admissions each on average)")

    filtered = filter_similar_sets(ordered, args.threshold)
    print(f"{len(filtered):,} sets survive the similarity filter "
          f"-> that is the S3 LLM call count")

    (data_dir / "all_visit_concepts.json").write_text(
        json.dumps([list(s) for s in sets]), encoding="utf-8")
    (data_dir / "all_top_coexisting_concepts.json").write_text(
        json.dumps(top, indent=2), encoding="utf-8")
    (data_dir / f"filtered_concept_sets_{args.threshold}.json").write_text(
        json.dumps([list(s) for s in filtered], indent=2), encoding="utf-8")
    write_seed_concept_sets(data_dir, scope)

    sizes = Counter(len(s) for s in sets)
    report = {
        "profile": args.profile,
        "admissions": len(sets),
        "distinct_sets": len(ordered),
        "filtered_sets": len(filtered),
        "llm_calls_s3": len(filtered) + len(scope.seed_terms),
        "concepts_with_neighbours": len(top),
        "set_size_histogram": {str(k): v for k, v in sorted(sizes.items())},
        "most_common_sets": [
            {"count": n, "concepts": list(s)} for s, n in ordered[:10]
        ],
        "top_k": args.top_k,
        "similarity_threshold": args.threshold,
    }
    (data_dir / "concept_sets_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nS3 will make {report['llm_calls_s3']:,} calls "
          f"({len(filtered):,} visit sets + {len(scope.seed_terms)} seed themes)")
    print(f"report -> {data_dir / 'concept_sets_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
