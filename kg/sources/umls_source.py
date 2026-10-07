#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Optional

from physioagent.kg.scope.filter import DEFAULT_PROFILE, Scope, load_scope, profiles
from physioagent.kg.sources.umls_prepare import DEFAULT_OUT, normalise, seed_names

KG_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA = KG_DIR / "data"

MAX_LENGTH = 4
MAX_PATHS = 20
MAX_NODES = 10000
TOP_N = 20


class UmlsSourceError(RuntimeError):
    pass


REVERSED = "<"


def read_graph(path: Path) -> dict[str, list[tuple[str, str]]]:
    graph: dict[str, list[tuple[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            head, relation, tail = parts
            graph[head].append((relation, tail))
            graph[tail].append((REVERSED + relation, head))
    return dict(graph)


def read_two_col(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                out[parts[0]] = parts[1]
    return out


def read_seed_cui(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                out[parts[0]] = parts[1]
    return out


def load_artifacts(umls_dir: Path) -> tuple[dict[str, list[tuple[str, str]]],
                                            dict[str, str], dict[str, str]]:
    graph_path = umls_dir / "umls_graph.tsv"
    seed_path = umls_dir / "seed_cui.tsv"
    term_path = umls_dir / "cui_term.tsv"
    for path in (graph_path, seed_path, term_path):
        if not path.exists():
            raise UmlsSourceError(
                f"{path} not found. Run umls_prepare.py against a UMLS release first.")
    return read_graph(graph_path), read_seed_cui(seed_path), read_two_col(term_path)


def _flip(relation: str) -> str:
    return relation[len(REVERSED):] if relation.startswith(REVERSED) \
        else REVERSED + relation


def _reverse(path: list[str]) -> list[str]:
    out = path[::-1]
    for index in range(1, len(out), 2):
        out[index] = _flip(out[index])
    return out


def _stitch(forward: list[str], backward: list[str]) -> list[str]:
    return forward + _reverse(backward)[1:]


def walk(graph: dict[str, list[tuple[str, str]]], start: str, end: str,
         max_length: int = MAX_LENGTH, max_paths: int = MAX_PATHS,
         max_nodes: int = MAX_NODES) -> list[list[str]]:
    if start == end or start not in graph or end not in graph:
        return []

    half = (max_length + 1) // 2
    queue_f: deque[list[str]] = deque([[start]])
    queue_b: deque[list[str]] = deque([[end]])
    seen_f: dict[str, list[str]] = {start: [start]}
    seen_b: dict[str, list[str]] = {end: [end]}
    paths: list[list[str]] = []
    explored = 0

    while (queue_f or queue_b) and len(paths) < max_paths and explored < max_nodes:
        for queue, seen, other, forward in ((queue_f, seen_f, seen_b, True),
                                            (queue_b, seen_b, seen_f, False)):
            if not queue:
                continue
            path = queue.popleft()
            node = path[-1]
            if node in other:
                joined = _stitch(path, other[node]) if forward \
                    else _stitch(other[node], path)
                if len(joined) // 2 <= max_length and joined not in paths:
                    paths.append(joined)
                if len(paths) >= max_paths:
                    break
            if len(path) // 2 >= half:
                continue
            for relation, neighbour in graph.get(node, ()):
                if neighbour in seen:
                    continue
                seen[neighbour] = path + [relation, neighbour]
                queue.append(seen[neighbour])
                explored += 1
                if explored >= max_nodes:
                    break
    return paths


def path_triples(path: list[str], terms: dict[str, str],
                 names: dict[str, str]) -> list[tuple[str, str, str, str, str, int]]:
    out = []
    for index in range(0, len(path) - 2, 2):
        head_cui, relation, tail_cui = path[index], path[index + 1], path[index + 2]
        if relation.startswith(REVERSED):
            relation = relation[len(REVERSED):]
            head_cui, tail_cui = tail_cui, head_cui
        head = names.get(head_cui) or terms.get(head_cui)
        tail = names.get(tail_cui) or terms.get(tail_cui)
        if not head or not tail or head == tail:
            continue
        out.append((head.lower(), relation, tail.lower(), head_cui, tail_cui,
                    index // 2 + 1))
    return out


def build_pairs(scope: Scope, top_coexisting: Optional[dict[str, list[str]]],
                top_n: int = TOP_N) -> list[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    if top_coexisting:
        for concept, neighbours in top_coexisting.items():
            for neighbour in list(neighbours)[:top_n]:
                if concept != neighbour:
                    pairs.add(tuple(sorted((concept, neighbour))))  # type: ignore[arg-type]
    else:
        names = sorted(scope.names())
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                pairs.add((left, right))
    for terms in scope.seed_terms.values():
        ordered = sorted(set(terms))
        for i, left in enumerate(ordered):
            for right in ordered[i + 1:]:
                pairs.add((left, right))
    return sorted(pairs)


def run(pairs: Iterable[tuple[str, str]], graph: dict[str, list[tuple[str, str]]],
        seed_cui: dict[str, str], terms: dict[str, str], names: dict[str, str],
        max_length: int, max_paths: int, max_nodes: int,
        progress_every: int = 200) -> tuple[list[dict[str, Any]], Counter]:
    stats: Counter = Counter()
    triples: dict[tuple[str, str, str], dict[str, Any]] = {}
    pairs = list(pairs)
    for index, (left, right) in enumerate(pairs, 1):
        start = seed_cui.get(normalise(left))
        end = seed_cui.get(normalise(right))
        if not start or not end:
            stats["pairs_unresolved"] += 1
            continue
        found = walk(graph, start, end, max_length, max_paths, max_nodes)
        if not found:
            stats["pairs_no_path"] += 1
            continue
        stats["pairs_with_path"] += 1
        stats["paths"] += len(found)
        for rank, path in enumerate(found):
            for head, relation, tail, head_cui, tail_cui, hop in path_triples(
                    path, terms, names):
                key = (head, relation, tail)
                record = triples.get(key)
                if record is None:
                    triples[key] = {
                        "head": head, "relation": relation, "tail": tail,
                        "head_cui": head_cui, "tail_cui": tail_cui,
                        "pairs": [[left, right]], "hop": hop, "path_rank": rank,
                    }
                    stats["triples"] += 1
                elif [left, right] not in record["pairs"]:
                    record["pairs"].append([left, right])
                    record["hop"] = min(record["hop"], hop)
        if progress_every and index % progress_every == 0:
            print(f"  [{index}/{len(pairs)}] {stats['triples']:,} triples "
                  f"from {stats['pairs_with_path']:,} connected pairs", flush=True)
    return list(triples.values()), stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default=DEFAULT_PROFILE, choices=profiles())
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA))
    ap.add_argument("--umls-dir", default=str(DEFAULT_OUT),
                    help="where umls_prepare.py wrote its tables")
    ap.add_argument("--max-length", type=int, default=MAX_LENGTH)
    ap.add_argument("--max-paths", type=int, default=MAX_PATHS)
    ap.add_argument("--max-nodes", type=int, default=MAX_NODES)
    ap.add_argument("--top-n", type=int, default=TOP_N,
                    help="neighbours per concept taken from S1's co-occurrence list")
    ap.add_argument("--limit", type=int, help="first N pairs only, for a smoke run")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve pairs to CUIs and stop. Says how many pairs the walk "
                         "would actually be able to start from before it runs.")
    args = ap.parse_args()

    scope = load_scope(args.profile)
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        graph, seed_cui, terms = load_artifacts(Path(args.umls_dir))
    except UmlsSourceError as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        return 1
    edges = sum(len(v) for v in graph.values()) // 2
    print(f"graph      : {len(graph):,} CUIs, {edges:,} edges")

    top_path = data_dir / "all_top_coexisting_concepts.json"
    top_coexisting = json.loads(top_path.read_text(encoding="utf-8")) \
        if top_path.exists() else None
    if top_coexisting is None:
        print(f"WARNING: {top_path.name} not found (S1 has not run here). Falling back to "
              f"every in-scope pair, which is neither cheap nor EHR-grounded.")

    pairs = build_pairs(scope, top_coexisting, args.top_n)
    if args.limit:
        pairs = pairs[:args.limit]

    names: dict[str, str] = {}
    for name in seed_names(scope):
        cui = seed_cui.get(normalise(name))
        if cui:
            names.setdefault(cui, name)

    resolvable = sum(1 for left, right in pairs
                     if normalise(left) in seed_cui and normalise(right) in seed_cui)
    print(f"pairs      : {len(pairs):,} ({resolvable:,} with both endpoints resolved)")
    if args.dry_run:
        missing = sorted({side for pair in pairs for side in pair
                          if normalise(side) not in seed_cui})
        print(f"unresolved concepts: {len(missing)}")
        for name in missing[:20]:
            print(f"  {name}")
        return 0

    records, stats = run(pairs, graph, seed_cui, terms, names,
                         args.max_length, args.max_paths, args.max_nodes)

    jsonl = data_dir / "kg_from_umls.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    lines = {f"{r['head']}\t{r['relation']}\t{r['tail']}" for r in records}
    (data_dir / "kg_from_umls.txt").write_text("\n".join(sorted(lines)) + "\n",
                                               encoding="utf-8")

    entities = {r["head"] for r in records} | {r["tail"] for r in records}
    by_hop = Counter(r["hop"] for r in records)
    report = {
        "profile": args.profile,
        "graph_cuis": len(graph),
        "graph_edges": edges,
        "pairs": len(pairs),
        "pairs_with_path": stats["pairs_with_path"],
        "pairs_no_path": stats["pairs_no_path"],
        "pairs_unresolved": stats["pairs_unresolved"],
        "paths": stats["paths"],
        "triples": len(records),
        "distinct_edges": len(lines),
        "entities": len(entities),
        "triples_by_hop": {str(k): v for k, v in sorted(by_hop.items())},
        "max_length": args.max_length,
        "max_paths": args.max_paths,
        "max_nodes": args.max_nodes,
    }
    (data_dir / "umls_report.json").write_text(json.dumps(report, indent=2) + "\n",
                                               encoding="utf-8")

    print(f"\n{len(records):,} triples ({len(lines):,} distinct edges), "
          f"{len(entities):,} entities")
    print(f"  connected : {stats['pairs_with_path']:,} pairs, "
          f"{stats['pairs_no_path']:,} with no path, "
          f"{stats['pairs_unresolved']:,} unresolved")
    print(f"  by hop    : {dict(sorted(by_hop.items()))}")
    print(f"\n-> {jsonl}\n-> {data_dir / 'kg_from_umls.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
