#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from physioagent.kg.guideline.clusters import norm_name
from physioagent.kg.retrieval.trajectory import load_graph, paths_between

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DATA = PACKAGE_ROOT / "kg" / "data"
KG = DATA / "kg_refined.jsonl"
GUIDELINE_GRAPH = DATA / "guideline_graph.json"
TRAJECTORIES = DATA / "trajectories_dev.json"
OUT = DATA / "guideline_path_probe.json"

CLUSTER_PREFIX = "guideline:"
HOPS = (3, 4)
LOOP_DRUG = re.compile(r"furosemide|bumetanide|torsemide|torasemide|ethacryn|loop diuretic", re.I)


def combined_graph(base, guideline: dict[str, Any]):
    graph = base.copy()
    by_norm = {norm_name(n): n for n in graph.nodes}
    concept_node = {}
    for concept in guideline["nodes"]["concepts"]:
        concept_node[concept["id"]] = by_norm.get(norm_name(concept["name"]), concept["name"])
    for edge in guideline["edges"]["has_dose_statement"]:
        head = concept_node[edge["from"]]
        cluster = CLUSTER_PREFIX + edge["to"].removeprefix("cluster:")
        if not graph.has_edge(head, cluster):
            graph.add_edge(head, cluster, relation="has guideline dose statement", head=head,
                           sources=["guideline"])
    return graph


def loop_clusters(guideline: dict[str, Any]) -> set[str]:
    out = set()
    for node in guideline["nodes"]["clusters"]:
        if any(LOOP_DRUG.search(m["text"]) for m in node["members"]):
            out.add(CLUSTER_PREFIX + node["id"].removeprefix("cluster:"))
    return out


def patient_pairs(trajectories: dict[str, Any]) -> dict[str, list[tuple[str, str]]]:
    rows = {}
    for key, entry in trajectories.items():
        concepts = set()
        for aspect in ("risk", "protective"):
            for path in entry.get(aspect) or []:
                nodes = path.get("nodes") or []
                if nodes:
                    concepts.update((nodes[0], nodes[-1]))
        rows[key] = sorted(tuple(sorted(p)) for p in combinations(sorted(concepts), 2))
    return rows


def probe(graph, pairs: set[tuple[str, str]], hops: int) -> dict[tuple[str, str], list]:
    return {pair: [p.nodes for p in paths_between(graph, pair[0], pair[1], cutoff=hops)]
            for pair in pairs}


def through_cluster(nodes: list[str]) -> bool:
    return any(n.startswith(CLUSTER_PREFIX) for n in nodes)


def summarise(rows: dict[str, list[tuple]], base: dict, comb: dict, loop: set[str]
              ) -> dict[str, Any]:
    pairs = set(base)
    base_paths = sum(len(v) for v in base.values())
    comb_paths = sum(len(v) for v in comb.values())
    via = sum(1 for v in comb.values() for p in v if through_cluster(p))
    lost = sum(1 for pair in pairs if base[pair] and comb[pair]
               and all(through_cluster(p) for p in comb[pair]))
    shorter = sum(1 for pair in pairs if base[pair] and comb[pair]
                  and len(comb[pair][0]) < len(base[pair][0]))
    new_pairs = sum(1 for pair in pairs if not base[pair] and comb[pair])
    rows_via = sum(1 for row in rows.values()
                   if any(through_cluster(p) for pair in row for p in comb[pair]))
    rows_loop = sum(1 for row in rows.values()
                    if any(any(n in loop for n in p) for pair in row for p in comb[pair]))
    return {
        "pairs": len(pairs),
        "pairs_with_path": {"base": sum(1 for v in base.values() if v),
                            "with_guideline": sum(1 for v in comb.values() if v)},
        "paths": {"base": base_paths, "with_guideline": comb_paths,
                  "through_cluster": via,
                  "share_through_cluster": round(via / comb_paths, 4) if comb_paths else 0.0},
        "pairs_newly_connected": new_pairs,
        "pairs_whose_paths_got_shorter": shorter,
        "pairs_where_all_paths_now_go_through_clusters": lost,
        "rows": len(rows),
        "rows_with_a_path_through_a_cluster": rows_via,
        "rows_with_a_path_through_a_loop_diuretic_cluster": rows_loop,
        "clusters_on_paths": len({n for v in comb.values() for p in v for n in p
                                  if n.startswith(CLUSTER_PREFIX)}),
    }


RELATION_TEXT = "has guideline dose statement"
NEUML = "NeuML/pubmedbert-base-embeddings"


def ranking_similarity(graph, guideline: dict[str, Any]):
    import numpy as np
    from sentence_transformers import SentenceTransformer

    from physioagent.kg.retrieval.trajectory import (
        PROTECTIVE_QUERY,
        RISK_QUERY,
        EmbeddingSimilarity,
        load_embedding_similarity,
    )

    needed = set(graph.nodes()) | {d.get("relation") for _, _, d in graph.edges(data=True)}
    needed.discard(None)
    similarity = load_embedding_similarity(DATA, NEUML, (RISK_QUERY, PROTECTIVE_QUERY), None,
                                           None, needed, local=True)
    model = SentenceTransformer(NEUML)
    for node in guideline["nodes"]["clusters"]:
        texts = [m["text"] for m in node["members"]]
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        name = CLUSTER_PREFIX + node["id"].removeprefix("cluster:")
        similarity.vectors[name] = EmbeddingSimilarity._unit(np.mean(vectors, axis=0))
    similarity.vectors[RELATION_TEXT] = EmbeddingSimilarity._unit(
        model.encode([RELATION_TEXT], show_progress_bar=False)[0])
    return similarity


def evidence_block(graph, pairs: list[tuple[str, str]], similarity, hops: int,
                   stages: dict[str, bool] | None = None) -> list[dict]:
    from physioagent.eval.evidence import EvidenceProvider
    from physioagent.kg.retrieval.trajectory import (
        ConstantSeverity,
        score_paths,
        top_rho,
    )

    paths = [p for pair in pairs for p in paths_between(graph, pair[0], pair[1], cutoff=hops)]
    scored = score_paths(paths, similarity, ConstantSeverity())
    entry = {"risk": [p.to_json() for p in top_rho(scored, "risk")],
             "protective": [p.to_json() for p in top_rho(scored, "protective")]}
    if stages is not None:
        stages["candidates"] = any(through_cluster(p.nodes) for p in paths)
        stages["top_rho"] = any(through_cluster(p["nodes"])
                                for aspect in ("risk", "protective") for p in entry[aspect])
    return [path for _, path in EvidenceProvider({}, timed=False)._select(entry)]


LINKS = DATA / "guideline_concept_links.jsonl"


def cluster_requirements(mode: str) -> dict[str, set[str]]:
    needs: dict[str, set[str]] = defaultdict(set)
    for line in LINKS.open(encoding="utf-8"):
        link = json.loads(line)
        if link["status"] != "accepted":
            continue
        if mode == "conditions" and link["kind"] != "condition":
            continue
        needs[CLUSTER_PREFIX + link["cluster_id"]].add(norm_name(link["concept"]))
    return needs


def patient_view(graph, pairs: list[tuple[str, str]], needs: dict[str, set[str]]):
    import networkx as nx

    have = {norm_name(c) for pair in pairs for c in pair}
    blocked = {c for c, req in needs.items() if not req <= have}
    return nx.subgraph_view(graph, filter_node=lambda n: n not in blocked)


def rank_probe(base_graph, graph, guideline: dict[str, Any], rows: dict[str, list],
               hops: int, mode: str = "none") -> dict[str, Any]:
    similarity = ranking_similarity(graph, guideline)
    loop = loop_clusters(guideline)
    needs = cluster_requirements(mode) if mode != "none" else {}
    with_cluster = with_loop = 0
    per_block, kept_kg = [], []
    reach: Counter = Counter()
    stage_counts: Counter = Counter()
    for pairs in rows.values():
        view = patient_view(graph, pairs, needs) if needs else graph
        stages: dict[str, bool] = {}
        block = evidence_block(view, pairs, similarity, hops, stages)
        stage_counts.update(k for k, v in stages.items() if v)
        base = evidence_block(base_graph, pairs, similarity, hops)
        clusters = {n for p in block for n in p["nodes"] if n.startswith(CLUSTER_PREFIX)}
        reach.update(clusters)
        per_block.append(sum(1 for p in block if through_cluster(p["nodes"])))
        with_cluster += bool(clusters)
        with_loop += bool(clusters & loop)
        base_keys = {tuple(p["nodes"]) for p in base}
        if base_keys:
            kept_kg.append(len(base_keys & {tuple(p["nodes"]) for p in block}) / len(base_keys))
    n = len(rows)
    by_id = {CLUSTER_PREFIX + c["id"].removeprefix("cluster:"): c
             for c in guideline["nodes"]["clusters"]}
    return {
        "rows": n,
        "rows_with_a_cluster_among_candidate_paths": stage_counts["candidates"],
        "rows_with_a_cluster_after_top_rho": stage_counts["top_rho"],
        "rows_with_a_cluster_in_block": with_cluster,
        "rows_with_a_loop_diuretic_cluster_in_block": with_loop,
        "cluster_paths_per_block_mean": round(sum(per_block) / n, 3) if n else 0.0,
        "share_of_base_block_paths_kept": round(sum(kept_kg) / len(kept_kg), 4) if kept_kg else 0.0,
        "distinct_clusters_in_blocks": len(reach),
        "top_clusters": [{"cluster": c, "rows": k, "sources": by_id[c]["sources"],
                          "text": by_id[c]["members"][0]["text"][:160]}
                         for c, k in reach.most_common(10)],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Local probe: do patient concept pairs reach "
                                             "guideline clusters through the KG?")
    ap.add_argument("--limit", type=int, default=0, help="first N dev rows only")
    ap.add_argument("--rank", action="store_true",
                    help="also run the pipeline ranking (top-rho + evidence selection)")
    ap.add_argument("--filters", nargs="+", default=["none"],
                    choices=("none", "conditions", "all"),
                    help="cluster traversal rule: none, every condition concept present, "
                         "or every linked concept present in the patient")
    args = ap.parse_args()
    base_graph = load_graph(KG)
    guideline = json.loads(GUIDELINE_GRAPH.read_text(encoding="utf-8"))
    graph = combined_graph(base_graph, guideline)
    loop = loop_clusters(guideline)
    trajectories = json.loads(TRAJECTORIES.read_text(encoding="utf-8"))
    rows = patient_pairs(trajectories)
    if args.limit:
        rows = dict(list(rows.items())[:args.limit])
    pairs = {pair for row in rows.values() for pair in row}
    print(f"KG {base_graph.number_of_nodes()} nodes / {base_graph.number_of_edges()} edges; "
          f"with guideline {graph.number_of_nodes()} / {graph.number_of_edges()}; "
          f"loop-diuretic clusters {len(loop)}; dev rows {len(rows)}; distinct pairs {len(pairs)}")
    result: dict[str, Any] = {"proxy": "patient concepts = endpoints of kept paths in "
                                       "trajectories_dev.json; all pairs among them"}
    for hops in HOPS:
        base = probe(base_graph, pairs, hops)
        comb = probe(graph, pairs, hops)
        result[f"hops_{hops}"] = summarise(rows, base, comb, loop)
        print(f"\nhops {hops}: {json.dumps(result[f'hops_{hops}'], indent=1)}")
    if args.rank:
        for mode in args.filters:
            key = f"ranked_hops_3_filter_{mode}"
            result[key] = rank_probe(base_graph, graph, guideline, rows, 3, mode)
            print(f"\nafter ranking, filter {mode}: "
                  f"{json.dumps({k: v for k, v in result[key].items() if k != 'top_clusters'})}")
            for c in result[key]["top_clusters"][:5]:
                print(f"   {c['rows']:4d} {c['cluster']} {c['sources']} {c['text'][:90]}")
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8", newline="\n")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
