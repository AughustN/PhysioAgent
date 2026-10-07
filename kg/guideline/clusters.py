#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from physioagent.kg.guideline.axis_eval import Mention, build_mentions, embed

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DATA = PACKAGE_ROOT / "kg" / "data"
PASSAGES = DATA / "guideline_passages.jsonl"
SEED_CUI = DATA / "umls" / "seed_cui.tsv"
SYNONYMS = DATA / "umls" / "seed_synonyms.tsv"
CHILD_TERMS = DATA / "umls" / "seed_child_terms.tsv"
ONTOLOGY = PACKAGE_ROOT / "kg" / "scope" / "ontology.yaml"
RESOURCES = PACKAGE_ROOT / "kg" / "resources"
MRCONSO = PACKAGE_ROOT.parent / "umls-2026AA-metathesaurus-full" / "2026AA" / "META" / "MRCONSO.RRF"
OUT = DATA / "guideline_clusters.jsonl"
SUMMARY = DATA / "guideline_clusters_summary.json"

THRESHOLD = 0.75
SENSITIVITY = (0.6, 0.75, 0.85)
MIN_TERM_CHARS = 3
MIN_SINGLE_WORD_CHARS = 5
ACRONYM = re.compile(r"^[A-Z][A-Z0-9]{1,5}$")
QUALIFIER = re.compile(r"\s*(\([^)]*\)|\[[^\]]*\]|,?\s*NOS|,?\s*unspecified)\s*$", re.I)

SPOT_CHECKS = (
    ("acute and unspecified renal failure", "kdigo_aki_2012"),
    ("chronic kidney disease", "kdigo_ckd_2024"),
    ("congestive heart failure nonhypertensive", "acc_aha_hfsa_2022"),
    ("other liver diseases", "easl_decompensated_cirrhosis_2018"),
    ("high ceiling diuretics", "dailymed_furosemide"),
)


def read_seeds(path: Path = SEED_CUI) -> dict[str, str]:
    seeds = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            seeds[parts[1]] = parts[0]
    return seeds


def clean_term(term: str) -> str:
    previous = None
    while previous != term:
        previous, term = term, QUALIFIER.sub("", term).strip()
    return term


def keep_term(term: str) -> bool:
    if ACRONYM.match(term):
        return True
    return len(term) >= MIN_TERM_CHARS and bool(re.search(r"[A-Za-z]{3}", term))


def build_synonyms(mrconso: Path, seeds: dict[str, str]) -> list[tuple[str, str, str]]:
    rows = set()
    with mrconso.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.split("|")
            if fields[0] not in seeds or fields[1] != "ENG" or fields[16] != "N":
                continue
            term = clean_term(fields[14])
            if keep_term(term):
                rows.add((seeds[fields[0]], fields[0], term))
    for cui, concept in seeds.items():
        rows.add((concept, cui, concept))
    return sorted(rows)


def load_synonyms() -> list[tuple[str, str, str]]:
    if not SYNONYMS.exists():
        rows = build_synonyms(MRCONSO, read_seeds())
        SYNONYMS.write_text("".join(f"{c}\t{u}\t{t}\n" for c, u, t in rows), encoding="utf-8",
                            newline="\n")
    return [tuple(line.split("\t")) for line in
            SYNONYMS.read_text(encoding="utf-8").splitlines() if line]


def norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def scope_codes(ontology: Path = ONTOLOGY) -> tuple[dict[str, str], dict[str, str]]:
    seeds = {norm_name(c): c for c in read_seeds().values()}
    data = yaml.safe_load(ontology.read_text(encoding="utf-8"))
    ccs, atc = {}, {}
    for group in data["groups"].values():
        for code, name in (group.get("conditions") or {}).items():
            if norm_name(str(name)) in seeds:
                ccs[str(code)] = seeds[norm_name(str(name))]
        for code, name in (group.get("drugs") or {}).items():
            if norm_name(str(name)) in seeds:
                atc[str(code)] = seeds[norm_name(str(name))]
    return ccs, atc


def build_child_terms(mrconso: Path = MRCONSO) -> list[tuple[str, str, str]]:
    ccs, atc = scope_codes()
    icd: dict[tuple[str, str], str] = {}
    for sab, path in (("ICD10CM", RESOURCES / "ICD10CM_to_CCSCM.csv"),
                      ("ICD9CM", RESOURCES / "ICD9CM_to_CCSCM.csv")):
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                code, category = row[f"{sab}"], row["CCSCM"].strip()
                if category in ccs:
                    icd[(sab, code.strip())] = ccs[category]
    cui_concept: dict[str, set[str]] = defaultdict(set)
    with mrconso.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.split("|")
            concept = icd.get((fields[11], fields[13]))
            if concept and fields[1] == "ENG":
                cui_concept[fields[0]].add(concept)
    rows = set()
    with mrconso.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.split("|")
            if fields[0] not in cui_concept or fields[1] != "ENG" or fields[16] != "N":
                continue
            term = clean_term(fields[14])
            if keep_term(term):
                for concept in cui_concept[fields[0]]:
                    rows.add((concept, f"UMLS:{fields[0]}", term))
    with (RESOURCES / "drug_to_atc3.csv").open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#") or line.startswith("pattern,"):
                continue
            parts = line.rstrip("\n").split(",")
            if len(parts) >= 2 and parts[1].strip() in atc:
                rows.add((atc[parts[1].strip()], f"ATC:{parts[1].strip()}", f"re:{parts[0]}"))
    return sorted(rows)


def load_child_terms() -> list[tuple[str, str, str]]:
    if not CHILD_TERMS.exists():
        rows = build_child_terms()
        CHILD_TERMS.write_text("".join(f"{c}\t{u}\t{t}\n" for c, u, t in rows),
                               encoding="utf-8", newline="\n")
    return [tuple(line.split("\t")) for line in
            CHILD_TERMS.read_text(encoding="utf-8").splitlines() if line]


def term_pattern(term: str) -> re.Pattern:
    if term.startswith("re:"):
        return re.compile(term[3:], re.I)
    body = re.escape(term).replace(r"\ ", r"\s+")
    flags = 0 if ACRONYM.match(term) else re.I
    return re.compile(rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])", flags)


def matchable(term: str) -> bool:
    if term.startswith("re:") or ACRONYM.match(term) or " " in term.strip():
        return True
    return len(term) >= MIN_SINGLE_WORD_CHARS


def link_concepts(mentions: list[Mention], synonyms: list[tuple[str, str, str]]
                  ) -> dict[str, list[dict[str, str]]]:
    patterns = [(concept, cui, term, term_pattern(term)) for concept, cui, term in synonyms
                if matchable(term)]
    links: dict[str, list[dict[str, str]]] = {}
    for m in mentions:
        found: dict[str, dict[str, str]] = {}
        for where, text in (("text", m.text), ("section", m.section)):
            for concept, cui, term in longest_matches(text, patterns):
                if concept not in found:
                    found[concept] = {"concept": concept, "cui": cui, "term": term,
                                      "where": where}
        links[m.mention_id] = sorted(found.values(), key=lambda x: x["concept"])
    return links


def longest_matches(text: str, patterns: list[tuple]) -> list[tuple[str, str, str]]:
    spans = [(match.start(), match.end(), concept, cui, term)
             for concept, cui, term, pattern in patterns
             for match in pattern.finditer(text) if match.end() > match.start()]
    kept = []
    for start, end, concept, cui, term in spans:
        covered = any(s <= start and end <= e and (e - s) > (end - start) and c != concept
                      for s, e, c, _, _ in spans)
        if not covered:
            kept.append((start, end, concept, cui, term))
    kept.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    return [(concept, cui, term) for _, _, concept, cui, term in kept]


def cluster(mentions: list[Mention], vectors, threshold: float) -> list[int]:
    import numpy as np
    from scipy.cluster.hierarchy import fcluster, linkage

    by_value: dict[tuple, list[int]] = defaultdict(list)
    for i, m in enumerate(mentions):
        by_value[m.value].append(i)
    labels = [-1] * len(mentions)
    next_id = 0
    for value in sorted(by_value, key=lambda v: by_value[v][0]):
        members = by_value[value]
        if len(members) == 1:
            groups = [1]
        else:
            tree = linkage(np.asarray(vectors[members], dtype="float64"), method="complete",
                           metric="cosine")
            groups = list(fcluster(tree, t=1.0 - threshold, criterion="distance"))
        local: dict[int, int] = {}
        for index, group in zip(members, groups):
            if group not in local:
                local[group] = next_id
                next_id += 1
            labels[index] = local[group]
    return labels


def describe(mentions: list[Mention], labels: list[int]) -> dict[str, Any]:
    members: dict[int, list[Mention]] = defaultdict(list)
    for m, label in zip(mentions, labels):
        members[label].append(m)
    sizes = [len(v) for v in members.values()]
    return {"clusters": len(members),
            "multi_member": sum(1 for s in sizes if s > 1),
            "multi_source": sum(1 for v in members.values() if len({m.source_id for m in v}) > 1),
            "largest": max(sizes) if sizes else 0,
            "singletons": sum(1 for s in sizes if s == 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Two-axis clusters of guideline dose mentions, "
                                             "linked to scope concepts")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    args = ap.parse_args()

    passages = [json.loads(l) for l in PASSAGES.open(encoding="utf-8")]
    mentions = build_mentions(passages)
    vectors = embed([m.masked for m in mentions])
    sensitivity = {str(t): describe(mentions, cluster(mentions, vectors, t)) for t in SENSITIVITY}
    labels = cluster(mentions, vectors, args.threshold)
    synonyms = load_synonyms() + load_child_terms()
    links = link_concepts(mentions, synonyms)

    with OUT.open("w", encoding="utf-8", newline="\n") as handle:
        for m, label in zip(mentions, labels):
            handle.write(json.dumps({"mention_id": m.mention_id, "passage_id": m.passage_id,
                                     "source_id": m.source_id, "cluster_id": f"c{label:05d}",
                                     "value": m.value, "basis_sources": m.value_basis_sources,
                                     "concepts": links[m.mention_id], "text": m.text},
                                    ensure_ascii=False) + "\n")

    by_cluster: dict[int, list[Mention]] = defaultdict(list)
    for m, label in zip(mentions, labels):
        by_cluster[label].append(m)
    concept_mentions = Counter(l["concept"] for v in links.values() for l in v)
    concept_where = Counter((l["concept"], l["where"]) for v in links.values() for l in v)
    spot = {}
    for concept, source in SPOT_CHECKS:
        spot[f"{concept} -> {source}"] = any(
            m.source_id == source and any(l["concept"] == concept for l in links[m.mention_id])
            for m in mentions)
    summary = {
        "threshold": args.threshold,
        "mentions": len(mentions),
        "synonym_terms": len(synonyms),
        "sensitivity": sensitivity,
        "mentions_with_any_concept": sum(1 for v in links.values() if v),
        "concepts_linked": len(concept_mentions),
        "scope_concepts": len(read_seeds()),
        "top_concepts": concept_mentions.most_common(15),
        "linked_only_via_section": sorted(c for c in concept_mentions
                                          if concept_where[(c, "text")] == 0),
        "spot_checks": spot,
        "clusters": {f"c{k:05d}": {"size": len(v), "sources": sorted({m.source_id for m in v}),
                                   "value": v[0].value,
                                   "concepts": sorted({l["concept"] for m in v
                                                       for l in links[m.mention_id]})}
                     for k, v in sorted(by_cluster.items())},
    }
    SUMMARY.write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8",
                       newline="\n")
    print(f"mentions {len(mentions)}; synonym terms {len(synonyms)}")
    for t, s in sensitivity.items():
        print(f"  threshold {t}: {s}")
    print(f"mentions with a concept: {summary['mentions_with_any_concept']}/{len(mentions)}; "
          f"concepts linked {summary['concepts_linked']}/{summary['scope_concepts']}")
    print(f"top concepts: {summary['top_concepts'][:10]}")
    for check, ok in spot.items():
        print(f"  spot check {'OK  ' if ok else 'FAIL'} {check}")
    print(f"-> {OUT}\n-> {SUMMARY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
