#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import statistics
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from physioagent.kg.guideline.axis_eval import build_mentions
from physioagent.kg.guideline.clusters import (
    CHILD_TERMS,
    SYNONYMS,
    read_seeds,
    scope_codes,
    term_pattern,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DATA = PACKAGE_ROOT / "kg" / "data"
PASSAGES = DATA / "guideline_passages.jsonl"
CLUSTERS = DATA / "guideline_clusters.jsonl"
OUT_DIR = DATA / "guideline_link_eval"
LINKS_OUT = DATA / "guideline_concept_links.jsonl"
SEED = 20261004
WINDOW_WORDS = 15
DESCRIPTION_TERMS = 5
MEMBERSHIP_TERMS = 15
LABEL_QUOTAS = {("text", "condition"): 75, ("text", "drug"): 75,
                ("section", "condition"): 25, ("section", "drug"): 25}
FOLDS = 5
BOOTSTRAP = 2000
MODELS = ("neuml", "medcpt")
NEUML = "NeuML/pubmedbert-base-embeddings"
MEDCPT_QUERY = "ncbi/MedCPT-Query-Encoder"
MEDCPT_ARTICLE = "ncbi/MedCPT-Article-Encoder"

JUDGE_SYSTEM = """You check whether a sentence from a clinical guideline or drug label refers
to a given medical concept. The concept is a patient condition or a drug class used to
describe patients in a hospital record. Answer true only if the sentence is about that
condition (or patients who have it) or about drugs of that class. Answer false if the
matched words mean something else here, for example a laboratory measurement ("urine
albumin", "serum potassium") instead of a drug, an anatomical or generic word ("tubular
epithelium"), or a different condition than the concept. Reply as JSON:
{"refers": bool, "reason": "<one sentence>"}"""


def window(text: str, term: str, size: int = WINDOW_WORDS) -> str:
    match = term_pattern(term).search(text)
    if not match:
        return " ".join(text.split()[:2 * size])
    before = text[:match.start()].split()[-size:]
    after = text[match.end():].split()[:size]
    return " ".join(before + [match.group(0)] + after)


def link_context(text: str, section: str, link: dict[str, str]) -> str:
    if link["where"] == "text":
        return window(text, link["term"])
    return f"{section} || {' '.join(text.split()[:2 * WINDOW_WORDS])}"


def plain_term(term: str) -> str:
    if term.startswith("re:"):
        return re.sub(r"\\b|[\\()\[\]?*+^$|]", "", term[3:]).strip()
    return term


def concept_terms() -> dict[str, list[str]]:
    terms: dict[str, list[str]] = defaultdict(list)
    for path in (SYNONYMS, CHILD_TERMS):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            concept, _, term = line.split("\t")
            term = plain_term(term)
            if term and term.lower() not in {t.lower() for t in terms[concept]}:
                terms[concept].append(term)
    return terms


def descriptions() -> dict[str, str]:
    terms = concept_terms()
    out = {}
    for concept in read_seeds().values():
        extra = sorted((t for t in terms[concept] if t.lower() != concept.lower()),
                       key=lambda t: (len(t), t.lower()))[:DESCRIPTION_TERMS]
        out[concept] = "; ".join([concept] + extra)
    return out


def memberships(matched: dict[str, Counter]) -> dict[str, str]:
    terms = concept_terms()
    out = {}
    for concept in read_seeds().values():
        seen: list[str] = []
        for term, _ in matched.get(concept, Counter()).most_common():
            term = plain_term(term)
            if term.lower() not in {s.lower() for s in seen} and term.lower() != concept.lower():
                seen.append(term)
        for term in sorted(terms[concept], key=lambda t: (len(t), t.lower())):
            if len(seen) >= MEMBERSHIP_TERMS:
                break
            if term.lower() not in {s.lower() for s in seen} and term.lower() != concept.lower():
                seen.append(term)
        out[concept] = "; ".join(seen[:MEMBERSHIP_TERMS])
    return out


def score_neuml(contexts: list[str], descs: list[str]) -> list[float]:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(NEUML)
    a = model.encode(contexts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    b = model.encode(descs, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    return [float(x) for x in np.sum(np.asarray(a) * np.asarray(b), axis=1)]


def score_medcpt(contexts: list[str], descs: list[str]) -> list[float]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    def encode(name: str, texts: list[str], max_length: int) -> "torch.Tensor":
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name).eval()
        chunks = []
        with torch.no_grad():
            for i in range(0, len(texts), 32):
                batch = tokenizer(texts[i:i + 32], truncation=True, padding=True,
                                  max_length=max_length, return_tensors="pt")
                chunks.append(model(**batch).last_hidden_state[:, 0, :])
        return torch.cat(chunks)

    queries = encode(MEDCPT_QUERY, descs, 64)
    articles = encode(MEDCPT_ARTICLE, contexts, 512)
    return [float(x) for x in (queries * articles).sum(dim=1)]


def judge_prompt(concept: str, members: str, source: str, section: str, text: str) -> str:
    return (f"Concept: {concept}\n"
            f"This concept groups, among others: {members}\n\n"
            f"Sentence (source: {source}; section: {section or '-'}):\n{text}")


def judge(rows: list[dict[str, Any]], path: Path, workers: int) -> dict[str, dict[str, Any]]:
    from physioagent.config.loader import load_settings
    from physioagent.llm.client import LLMClient

    done: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            if "error" not in row:
                done[row["label_key"]] = row
    todo = [r for r in rows if r["label_key"] not in done]
    client = LLMClient.from_settings(load_settings())
    lock = threading.Lock()

    def one(row: dict[str, Any]) -> dict[str, Any]:
        meta: dict[str, Any] = {}
        error = None
        for _ in range(3):
            try:
                out = client.chat_json(JUDGE_SYSTEM, row["prompt"], meta)
                return {"label_key": row["label_key"], "model": meta.get("model"),
                        "refers": bool(out.get("refers")),
                        "reason": str(out.get("reason", ""))[:300]}
            except Exception as exc:
                error = exc
        return {"label_key": row["label_key"], "error": str(error)[:300]}

    with ThreadPoolExecutor(max_workers=workers) as pool, \
            path.open("a", encoding="utf-8", newline="\n") as handle:
        for future in as_completed([pool.submit(one, r) for r in todo]):
            row = future.result()
            with lock:
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                if "error" not in row:
                    done[row["label_key"]] = row
    return done


def f1(pred: list[bool], gold: list[bool]) -> dict[str, float]:
    tp = sum(p and g for p, g in zip(pred, gold))
    fp = sum(p and not g for p, g in zip(pred, gold))
    fn = sum(g and not p for p, g in zip(pred, gold))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(score, 4),
            "tp": tp, "fp": fp, "fn": fn}


def best_threshold(scores: list[float], gold: list[bool]) -> float:
    candidates = sorted(set(scores)) + [min(scores) - 1.0]
    return max(candidates, key=lambda t: (f1([s >= t for s in scores], gold)["f1"], -t))


def cross_validate(scores: list[float], gold: list[bool]) -> dict[str, Any]:
    rng = random.Random(SEED)
    order = list(range(len(scores)))
    rng.shuffle(order)
    folds = [order[k::FOLDS] for k in range(FOLDS)]
    pred = [False] * len(scores)
    thresholds = []
    for k in range(FOLDS):
        train = [i for f, fold in enumerate(folds) if f != k for i in fold]
        t = best_threshold([scores[i] for i in train], [gold[i] for i in train])
        thresholds.append(round(t, 4))
        for i in folds[k]:
            pred[i] = scores[i] >= t
    point = f1(pred, gold)
    boots = []
    for _ in range(BOOTSTRAP):
        idx = [rng.randrange(len(scores)) for _ in scores]
        boots.append(f1([pred[i] for i in idx], [gold[i] for i in idx])["f1"])
    boots.sort()
    return {**point, "f1_ci95": [boots[int(0.025 * BOOTSTRAP)], boots[int(0.975 * BOOTSTRAP)]],
            "cv_thresholds": thresholds, "threshold": round(statistics.median(thresholds), 4)}


def link_status(link: dict[str, Any]) -> str:
    return "accepted" if link["where"] == "text" else "weak"


def stratified_sample(links: list[dict[str, Any]], quotas: dict[tuple, int]) -> list[dict]:
    rng = random.Random(SEED)
    cells: dict[tuple, list[int]] = defaultdict(list)
    for i, l in enumerate(links):
        cells[(l["where"], l["kind"])].append(i)
    chosen = []
    for key in sorted(quotas):
        pool = cells.get(key, [])[:]
        rng.shuffle(pool)
        chosen += pool[:quotas[key]]
    return [links[i] for i in chosen]


def main() -> int:
    ap = argparse.ArgumentParser(description="Status and silver evaluation of concept links")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--embeddings", action="store_true",
                    help="also score links with NeuML and MedCPT and cross-validate a filter")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    passages = [json.loads(l) for l in PASSAGES.open(encoding="utf-8")]
    sections = {m.mention_id: m.section for m in build_mentions(passages)}
    _, atc = scope_codes()
    drug_concepts = set(atc.values())
    rows_in = [json.loads(line) for line in CLUSTERS.open(encoding="utf-8")]
    matched: dict[str, Counter] = defaultdict(Counter)
    for row in rows_in:
        for link in row["concepts"]:
            matched[link["concept"]][link["term"]] += 1
    members = memberships(matched)
    descs = descriptions()

    links = []
    for row in rows_in:
        section = sections.get(row["mention_id"], "")
        for link in row["concepts"]:
            concept = link["concept"]
            prompt = judge_prompt(concept, members[concept], row["source_id"], section,
                                  row["text"])
            links.append({**link, "mention_id": row["mention_id"], "source_id": row["source_id"],
                          "cluster_id": row["cluster_id"],
                          "kind": "drug" if concept in drug_concepts else "condition",
                          "context": link_context(row["text"], section, link),
                          "description": descs[concept], "members": members[concept],
                          "prompt": prompt,
                          "label_key": hashlib.sha256(prompt.encode()).hexdigest()[:24]})
    print(f"candidate links: {len(links)} "
          f"{dict(Counter((l['where'], l['kind']) for l in links))}")

    sample = stratified_sample(links, LABEL_QUOTAS)
    labels = judge(sample, OUT_DIR / "llm_labels.jsonl", args.workers)
    rows = [l for l in sample if l["label_key"] in labels]
    population = Counter((l["where"], l["kind"]) for l in links)
    per_cell = {}
    for key in sorted(LABEL_QUOTAS):
        cell = [labels[l["label_key"]]["refers"] for l in rows if (l["where"], l["kind"]) == key]
        if cell:
            per_cell[f"{key[0]}|{key[1]}"] = {"labelled": len(cell),
                                              "precision": round(sum(cell) / len(cell), 4),
                                              "links_in_corpus": population[key]}
    accepted_keys = [k for k in LABEL_QUOTAS if k[0] == "text"]
    weighted = sum(per_cell[f"{k[0]}|{k[1]}"]["precision"] * population[k]
                   for k in accepted_keys) / sum(population[k] for k in accepted_keys)
    results: dict[str, Any] = {"n_labelled": len(rows), "per_cell": per_cell,
                               "accepted_precision_weighted": round(weighted, 4)}

    if args.embeddings:
        for name, scorer in (("neuml", score_neuml), ("medcpt", score_medcpt)):
            for link, value in zip(links, scorer([l["context"] for l in links],
                                                 [l["description"] for l in links])):
                link[f"score_{name}"] = round(value, 4)
        gold = [labels[l["label_key"]]["refers"] for l in rows]
        results["term_only"] = f1([True] * len(rows), gold)
        for name in MODELS:
            results[name] = cross_validate([l[f"score_{name}"] for l in rows], gold)
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8",
                                          newline="\n")

    with LINKS_OUT.open("w", encoding="utf-8", newline="\n") as handle:
        for l in links:
            out = {"mention_id": l["mention_id"], "cluster_id": l["cluster_id"],
                   "source_id": l["source_id"], "concept": l["concept"], "cui": l["cui"],
                   "term": l["term"], "where": l["where"], "kind": l["kind"],
                   "status": link_status(l)}
            for name in MODELS:
                if f"score_{name}" in l:
                    out[f"score_{name}"] = l[f"score_{name}"]
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
    with (OUT_DIR / "links_for_review.csv").open("w", encoding="utf-8-sig", newline="") as h:
        writer = csv.writer(h)
        writer.writerow(["concept", "concept_includes", "term", "where", "kind", "source",
                         "context", "llm_refers", "llm_reason", "human_refers", "note"])
        for l in rows:
            lab = labels[l["label_key"]]
            writer.writerow([l["concept"], l["members"], l["term"], l["where"], l["kind"],
                             l["source_id"], l["context"], lab["refers"], lab["reason"], "", ""])

    status = Counter(link_status(l) for l in links)
    print(json.dumps(results, indent=1))
    print(f"link status: {dict(status)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
