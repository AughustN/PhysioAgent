#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

KG_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = KG_DIR / "data"

SOURCES = {
    "pubmed": "kg_from_pubmed.jsonl",
    "umls": "kg_from_umls.jsonl",
}


JUNK_FILTERED_FLOWS = frozenset({"pubmed"})

_JUNK = (
    ("case_report", re.compile(r"\b\d+[\s-]?year[\s-]?old\b")),
    ("cohort_phrase", re.compile(
        r"^(?:the\s+)?patients?\b(?:\s+(?:with|without|on|in|at|undergoing|receiving"
        r"|already|who|suffering|treated|requiring)\b|\s*$)")),
    ("quantity", re.compile(
        r"\b\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?\s*"
        r"(?:mg|mcg|ug|ml|l|g|kg|mmol|meq|mmhg|mg/dl|ml/min|%|hours?|hrs?)\b"
        r"|^\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?\s+\w+\s+per\b"
    )),
)


def junk_reason(entity: str) -> Optional[str]:
    text = entity.strip().lower()
    for reason, pattern in _JUNK:
        if pattern.search(text):
            return reason
    return None


def read_flow(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def edge_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (str(record.get("head", "")).strip().lower(),
            str(record.get("relation", "")).strip().lower(),
            str(record.get("tail", "")).strip().lower())


def flow_provenance(flow: str, record: dict[str, Any]) -> dict[str, Any]:
    if flow == "pubmed":
        return {"flow": flow, "pmid": record.get("pmid"), "unit": record.get("unit"),
                "anchored_query": record.get("anchored_query"),
                "anchor_substituted": record.get("anchor_substituted") or []}
    return {"flow": flow, **{k: v for k, v in record.items()
                             if k not in ("head", "relation", "tail")}}


def combine(data_dir: Path, flows: Iterable[str],
            drop_junk: bool = True) -> tuple[list[dict[str, Any]], dict]:
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    per_flow = Counter()
    dropped = Counter()
    dropped_examples: dict[str, list[str]] = defaultdict(list)
    for flow in flows:
        records = read_flow(data_dir / SOURCES[flow])
        per_flow[flow] = len(records)
        for record in records:
            key = edge_key(record)
            if not all(key):
                continue
            if drop_junk and flow in JUNK_FILTERED_FLOWS:
                reasons = [r for r in (junk_reason(key[0]), junk_reason(key[2])) if r]
                if reasons:
                    dropped[reasons[0]] += 1
                    bad = key[0] if junk_reason(key[0]) else key[2]
                    if len(dropped_examples[reasons[0]]) < 8 \
                            and bad not in dropped_examples[reasons[0]]:
                        dropped_examples[reasons[0]].append(bad)
                    continue
            entry = merged.setdefault(key, {"head": key[0], "relation": key[1],
                                            "tail": key[2], "sources": [],
                                            "provenance": []})
            if flow not in entry["sources"]:
                entry["sources"].append(flow)
            entry["provenance"].append(flow_provenance(flow, record))

    edges = sorted(merged.values(), key=lambda e: (e["head"], e["relation"], e["tail"]))
    for edge in edges:
        edge["n_sources_raw"] = len(edge["sources"])

    entities = {e["head"] for e in edges} | {e["tail"] for e in edges}
    report = {
        "flows_read": dict(per_flow),
        "edges": len(edges),
        "entities": len(entities),
        "relations": len({e["relation"] for e in edges}),
        "edges_by_flow": dict(Counter(f for e in edges for f in e["sources"])),
        "edges_by_identical_phrasing": dict(Counter(e["n_sources_raw"] for e in edges)),
        "agreement_is_computed_in": "refine_report.json",
        "dropped_junk_edges": dict(dropped),
        "dropped_junk_examples": {k: v for k, v in dropped_examples.items()},
    }
    return edges, report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA))
    ap.add_argument("--flows", nargs="*", choices=sorted(SOURCES), default=sorted(SOURCES),
                    help="which flows to merge; a missing artifact is skipped")
    ap.add_argument("--keep-junk", action="store_true",
                    help="keep edges whose endpoint is a dose string, a case-report "
                         "subject, or a patient-cohort phrase. Off by default because "
                         "those are not concepts; on for the ablation that asks what "
                         "the filter cost.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    present = [f for f in args.flows if (data_dir / SOURCES[f]).exists()]
    if not present:
        raise SystemExit(f"no source artifacts in {data_dir}; run a source flow first")
    missing = [f for f in args.flows if f not in present]
    if missing:
        print(f"not present, skipped: {', '.join(missing)}")

    edges, report = combine(data_dir, present, drop_junk=not args.keep_junk)

    with (data_dir / "kg_raw.jsonl").open("w", encoding="utf-8") as handle:
        for edge in edges:
            handle.write(json.dumps(edge, ensure_ascii=False) + "\n")
    (data_dir / "kg_raw.txt").write_text(
        "\n".join(f"{e['head']}\t{e['relation']}\t{e['tail']}" for e in edges) + "\n",
        encoding="utf-8")
    (data_dir / "kg_raw_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{report['edges']} distinct edges, {report['entities']} entities, "
          f"{report['relations']} relations")
    print(f"  read       : {report['flows_read']}")
    print(f"  per flow   : {report['edges_by_flow']}")
    print(f"  identical  : {report['edges_by_identical_phrasing']} "
          f"(flows phrasing an edge the same way -- NOT agreement; see refine)")
    if report["dropped_junk_edges"]:
        print(f"  junk       : {report['dropped_junk_edges']} edges dropped")
        for reason, examples in report["dropped_junk_examples"].items():
            print(f"    {reason}: {', '.join(examples[:4])}")
    print(f"\n-> {data_dir / 'kg_raw.jsonl'}\n-> {data_dir / 'kg_raw.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
