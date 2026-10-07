#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from physioagent.kg.guideline.axis_eval import Mention, build_mentions
from physioagent.kg.guideline.clusters import norm_name

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DATA = PACKAGE_ROOT / "kg" / "data"
PASSAGES = DATA / "guideline_passages.jsonl"
CLUSTERS = DATA / "guideline_clusters.jsonl"
LINKS = DATA / "guideline_concept_links.jsonl"
GRAPH_OUT = DATA / "guideline_graph.json"


def build_graph(mentions: list[Mention], cluster_of: dict[str, str],
                links: list[dict[str, Any]]) -> dict[str, Any]:
    members: dict[str, list[Mention]] = defaultdict(list)
    for m in mentions:
        members[cluster_of[m.mention_id]].append(m)
    accepted = [l for l in links if l["status"] == "accepted"]
    concepts = sorted({l["concept"] for l in accepted})
    return {
        "nodes": {
            "concepts": [{"id": f"concept:{norm_name(c)}", "name": c} for c in concepts],
            "clusters": [{"id": f"cluster:{cid}", "value": ms[0].value,
                          "sources": sorted({m.source_id for m in ms}),
                          "members": [{"mention_id": m.mention_id, "passage_id": m.passage_id,
                                       "source_id": m.source_id, "text": m.text} for m in ms]}
                         for cid, ms in sorted(members.items())],
        },
        "edges": {
            "has_dose_statement": [{"from": f"concept:{norm_name(l['concept'])}",
                                    "to": f"cluster:{l['cluster_id']}", "term": l["term"],
                                    "mention_id": l["mention_id"]} for l in accepted],
        },
    }


def main() -> int:
    argparse.ArgumentParser(description="Guideline subgraph: concepts and dose-statement "
                                        "clusters").parse_args()
    passages = [json.loads(l) for l in PASSAGES.open(encoding="utf-8")]
    mentions = build_mentions(passages)
    cluster_of = {json.loads(l)["mention_id"]: json.loads(l)["cluster_id"]
                  for l in CLUSTERS.open(encoding="utf-8")}
    missing = [m.mention_id for m in mentions if m.mention_id not in cluster_of]
    if missing:
        raise SystemExit(f"{len(missing)} mentions have no cluster; rerun clusters.py")
    links = [json.loads(l) for l in LINKS.open(encoding="utf-8")]
    graph = build_graph(mentions, cluster_of, links)
    GRAPH_OUT.write_text(json.dumps(graph, ensure_ascii=False, indent=1), encoding="utf-8",
                         newline="\n")
    print(f"concept nodes {len(graph['nodes']['concepts'])}; "
          f"cluster nodes {len(graph['nodes']['clusters'])}; "
          f"concept->cluster edges {len(graph['edges']['has_dose_statement'])}")
    print(f"-> {GRAPH_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
