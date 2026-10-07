#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator

from physioagent.kg.scope.filter import (
    DEFAULT_PROFILE,
    Scope,
    load_scope,
    profiles,
    umls_gate,
)

KG_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA = KG_DIR / "data"
DEFAULT_OUT = DEFAULT_DATA / "umls"

GATE_MODES = ("strict", "mesh-any", "sty-mesh-optional", "sty-or-mesh")

REL_MAPPING = {
    "AQ": "allowed qualifier",
    "CHD": "has child",
    "DEL": "deleted concept",
    "PAR": "has parent",
    "QB": "can be qualified by",
    "RB": "has a broader relationship",
    "RL": "alike",
    "RN": "has a narrower relationship",
    "RO": "has relationship",
    "RQ": "related and possibly synonymous",
    "RU": "related, unspecified",
    "SY": "source asserted synonymy",
    "XR": "not related, no mapping",
    "": "empty relationship",
}

DROP_REL = ("SY", "RQ", "RL", "AQ", "QB", "XR", "DEL", "SIB")

DEFAULT_SAB = ("MSH", "SNOMEDCT_US", "NCI", "RXNORM", "ATC")

_PUNCT = re.compile(r"[^a-z0-9]+")


def normalise(text: str) -> str:
    return " ".join(_PUNCT.sub(" ", text.lower()).split())


def rrf_rows(path: Path) -> Iterator[list[str]]:
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            yield line.rstrip("\n").split("|")


def seed_names(scope: Scope) -> list[str]:
    names = list(scope.names())
    for terms in scope.seed_terms.values():
        names.extend(terms)
    return sorted(set(names))


def read_sty(path: Path, allowed: frozenset[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for fields in rrf_rows(path):
        if len(fields) < 4:
            continue
        cui, sty = fields[0], fields[3]
        if sty in allowed:
            out[cui].append(sty)
    return dict(out)


def read_mesh_trees(path: Path, prefixes: tuple[str, ...],
                    any_tree: bool = False) -> set[str]:
    hits: set[str] = set()
    if not prefixes and not any_tree:
        return hits
    for fields in rrf_rows(path):
        if len(fields) < 11 or fields[8] != "MN" or fields[9] != "MSH":
            continue
        if any_tree or any(fields[10].startswith(prefix) for prefix in prefixes):
            hits.add(fields[0])
    return hits


def gate_cuis(sty: dict[str, list[str]], mesh: set[str], mode: str) -> set[str]:
    if mode == "sty-or-mesh":
        return set(sty) | mesh
    if mode == "sty-mesh-optional":
        return set(sty)
    return {cui for cui in sty if cui in mesh}


def read_conso(path: Path, keep: set[str], wanted_names: set[str],
               languages: tuple[str, ...] = ("ENG",)) -> tuple[dict[str, str],
                                                               dict[str, tuple[str, str]]]:
    preferred: dict[str, str] = {}
    fallback: dict[str, str] = {}
    seeds: dict[str, tuple[str, str]] = {}
    for fields in rrf_rows(path):
        if len(fields) < 15 or fields[1] not in languages:
            continue
        cui, string = fields[0], fields[14]
        if not string:
            continue
        if wanted_names:
            key = normalise(string)
            if key in wanted_names and key not in seeds:
                seeds[key] = (cui, string)
        if cui not in keep:
            continue
        if fields[2] == "P" and fields[6] == "Y":
            preferred.setdefault(cui, string)
        else:
            fallback.setdefault(cui, string)
    for cui, string in fallback.items():
        preferred.setdefault(cui, string)
    return preferred, seeds


def rel_label(rel: str, rela: str) -> str:
    if rela:
        return rela.replace("_", " ").strip().lower()
    return REL_MAPPING.get(rel, rel.lower())


def read_rel(path: Path, keep: set[str], sabs: tuple[str, ...],
             drop_rel: tuple[str, ...]) -> tuple[list[tuple[str, str, str]], Counter]:
    edges: set[tuple[str, str, str]] = set()
    dropped: Counter = Counter()
    sab_set = frozenset(sabs)
    drop_set = frozenset(drop_rel)
    for fields in rrf_rows(path):
        if len(fields) < 15:
            continue
        cui1, rel, cui2, rela, sab, suppress = (fields[0], fields[3], fields[4],
                                                fields[7], fields[10], fields[14])
        if cui1 == cui2:
            dropped["self_loop"] += 1
            continue
        if suppress == "Y":
            dropped["suppressed"] += 1
            continue
        if sab_set and sab not in sab_set:
            dropped["sab"] += 1
            continue
        if rel in drop_set:
            dropped["relation"] += 1
            continue
        if cui1 not in keep or cui2 not in keep:
            dropped["out_of_gate"] += 1
            continue
        edges.add((cui2, rel_label(rel, rela), cui1))
    return sorted(edges), dropped


def write_tsv(path: Path, rows: Iterable[tuple[str, ...]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write("\t".join(row) + "\n")
            count += 1
    return count


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--umls", required=True,
                    help="the META/ directory of a UMLS release (holds MRCONSO.RRF)")
    ap.add_argument("--profile", default=DEFAULT_PROFILE, choices=profiles())
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--gate-mode", default="mesh-any", choices=GATE_MODES)
    ap.add_argument("--sab", default=",".join(DEFAULT_SAB),
                    help="source vocabularies to keep edges from; empty string = all")
    ap.add_argument("--keep-rel", default="",
                    help="comma-separated REL codes to keep that are dropped by default "
                         f"(default drops: {','.join(DROP_REL)})")
    args = ap.parse_args()

    meta = Path(args.umls)
    files = {name: meta / f"{name}.RRF" for name in ("MRSTY", "MRSAT", "MRCONSO", "MRREL")}
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        print("ABORT: missing UMLS files:\n  " + "\n  ".join(missing), file=sys.stderr)
        return 1

    scope = load_scope(args.profile)
    gate = umls_gate()
    allowed_sty = frozenset(gate["semantic_types"])
    prefixes = gate["mesh_prefixes"]
    sabs = tuple(s for s in args.sab.split(",") if s)
    keep_rel = frozenset(s for s in args.keep_rel.split(",") if s)
    drop_rel = tuple(r for r in DROP_REL if r not in keep_rel)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"gate mode  : {args.gate_mode}")
    print(f"sty allowed: {len(allowed_sty)}   mesh prefixes: {len(prefixes)}")
    print(f"[1/4] {files['MRSTY'].name} ...", flush=True)
    sty = read_sty(files["MRSTY"], allowed_sty)
    print(f"      {len(sty):,} CUIs with an allowed semantic type")

    print(f"[2/4] {files['MRSAT'].name} ...", flush=True)
    any_tree = args.gate_mode == "mesh-any"
    mesh = read_mesh_trees(files["MRSAT"], prefixes, any_tree=any_tree)
    print(f"      {len(mesh):,} CUIs "
          f"{'with any MeSH tree number' if any_tree else 'under a whitelisted MeSH tree'}")

    keep = gate_cuis(sty, mesh, args.gate_mode)
    print(f"      {len(keep):,} CUIs clear the gate")

    names = seed_names(scope)
    wanted = {normalise(name) for name in names}
    print(f"[3/4] {files['MRCONSO'].name} ({len(names)} seed names) ...", flush=True)
    terms, seeds = read_conso(files["MRCONSO"], keep, wanted)
    keep |= {cui for cui, _ in seeds.values()}
    print(f"      {len(terms):,} kept CUIs named; "
          f"{len(seeds)}/{len(names)} seed names resolved")

    print(f"[4/4] {files['MRREL'].name} ...", flush=True)
    edges, dropped = read_rel(files["MRREL"], keep, sabs, drop_rel)
    print(f"      {len(edges):,} edges kept, dropped {dict(dropped)}")

    for cui, string in seeds.values():
        terms.setdefault(cui, string)

    n_sty = write_tsv(out_dir / "cui_sty.tsv",
                      ((cui, ",".join(sorted(v))) for cui, v in sorted(sty.items())))
    n_term = write_tsv(out_dir / "cui_term.tsv", sorted(terms.items()))
    n_seed = write_tsv(out_dir / "seed_cui.tsv",
                       ((key, cui, string) for key, (cui, string) in sorted(seeds.items())))
    n_edge = write_tsv(out_dir / "umls_graph.tsv", edges)

    unresolved = sorted(name for name in names if normalise(name) not in seeds)
    report = {
        "profile": args.profile,
        "gate_mode": args.gate_mode,
        "sab": list(sabs),
        "dropped_relations": sorted(drop_rel),
        "cuis_with_allowed_sty": len(sty),
        "cuis_in_mesh_trees": len(mesh),
        "cuis_kept": len(keep),
        "seed_names": len(names),
        "seed_names_resolved": len(seeds),
        "seed_names_unresolved": unresolved,
        "edges": n_edge,
        "edge_rows_dropped": dict(dropped),
        "rows_written": {"cui_sty.tsv": n_sty, "cui_term.tsv": n_term,
                         "seed_cui.tsv": n_seed, "umls_graph.tsv": n_edge},
    }
    (out_dir / "umls_prepare_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"\n-> {out_dir}")
    if unresolved:
        print(f"WARNING: {len(unresolved)} seed names did not resolve to a CUI; "
              f"first few: {unresolved[:5]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
