#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence

import numpy as np

from physioagent.kg.retrieval.nli import (DEFAULT_MODEL as NLI_MODEL, PROTECTIVE_HYPOTHESIS,
                                          RISK_HYPOTHESIS, NLIRefiner)
from physioagent.kg.retrieval.trajectory import PROTECTIVE_QUERY, RISK_QUERY

KG_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA = KG_DIR / "data"
DEFAULT_ENCODER = "NeuML/pubmedbert-base-embeddings"

LAMBDA = 0.7
TAU1 = 0.5
TAU2 = 0.05
MAX_K = 10
MIN_K = 1
PREMISE_MAX_LENGTH = 512

ASPECTS = {
    "risk": (RISK_QUERY, RISK_HYPOTHESIS),
    "protective": (PROTECTIVE_QUERY, PROTECTIVE_HYPOTHESIS),
}


class Encoder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        pass


class Support(Protocol):
    def support(self, premise: str, hypothesis: str) -> float:
        pass


class SentenceEncoder:
    def __init__(self, model_name: str = DEFAULT_ENCODER, device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self._model.encode(list(texts), convert_to_numpy=True,
                                  normalize_embeddings=True, batch_size=128,
                                  show_progress_bar=False)


class NLISupport:
    def __init__(self, model_name: str = NLI_MODEL, device: Optional[str] = None):
        self._refiner = NLIRefiner(model_name=model_name, device=device).load()
        self.model_name = model_name
        self.device = self._refiner.device
        self._cache: dict[tuple[str, str], float] = {}
        self.calls = 0

    def support(self, premise: str, hypothesis: str) -> float:
        key = (premise, hypothesis)
        if key not in self._cache:
            self.calls += 1
            entail, _ = self._refiner.probabilities(
                [premise], hypothesis, max_length=PREMISE_MAX_LENGTH,
                truncation="only_first")[0]
            self._cache[key] = entail
        return self._cache[key]


def path_sentence(path: dict[str, Any]) -> str:
    triples = path.get("triples") or []
    if triples:
        return " ".join(f"{h} {r} {t}." for h, r, t in triples)
    return " ".join(path.get("nodes") or []) + "."


@dataclass
class Selection:
    picks: list[int] = field(default_factory=list)
    support: list[float] = field(default_factory=list)
    gains: list[float] = field(default_factory=list)
    stop: str = "empty"


def mmr_nli_select(sentences: Sequence[str], path_vecs: np.ndarray, query_vec: np.ndarray,
                   hypothesis: str, scorer: Support, lam: float = LAMBDA,
                   tau1: float = TAU1, tau2: float = TAU2, max_k: int = MAX_K,
                   min_k: int = MIN_K) -> tuple[Selection, np.ndarray]:
    n = len(sentences)
    sel = Selection()
    if n == 0:
        return sel, np.zeros(0)
    relevance = path_vecs @ query_vec
    redundancy = np.full(n, -np.inf)
    taken = np.zeros(n, dtype=bool)
    previous = 0.0
    while len(sel.picks) < min(max_k, n):
        penalty = np.where(np.isfinite(redundancy), redundancy, 0.0)
        gains = lam * relevance - (1 - lam) * penalty
        gains[taken] = -np.inf
        pick = int(np.argmax(gains))
        sel.picks.append(pick)
        sel.gains.append(float(gains[pick]))
        taken[pick] = True
        redundancy = np.maximum(redundancy, path_vecs @ path_vecs[pick])
        eta = scorer.support(" ".join(sentences[i] for i in sel.picks), hypothesis)
        sel.support.append(eta)
        if len(sel.picks) < min_k:
            previous = eta
            continue
        if eta >= tau1:
            sel.stop = "tau1"
            return sel, relevance
        if eta - previous < tau2:
            sel.stop = "tau2"
            return sel, relevance
        previous = eta
    sel.stop = "max_k" if len(sel.picks) >= max_k else "exhausted"
    return sel, relevance


def batched_select(rows: Sequence[str], path_vecs: np.ndarray, query_vec: np.ndarray,
                   hypothesis: str, scorer: Support, batch_size: int, lam: float, tau1: float,
                   tau2: float, max_k: int, min_k: int) -> tuple[list[int], np.ndarray, list]:
    relevance = path_vecs @ query_vec if len(rows) else np.zeros(0)
    picks: list[int] = []
    stops = []
    for start in range(0, len(rows), batch_size):
        index = list(range(start, min(start + batch_size, len(rows))))
        sel, _ = mmr_nli_select([rows[i] for i in index], path_vecs[index], query_vec,
                                hypothesis, scorer, lam, tau1, tau2, max_k, min_k)
        chosen = [index[i] for i in sel.picks]
        chosen.sort(key=lambda i: -relevance[i])
        picks += chosen
        stops.append(sel.stop)
    return picks, relevance, stops


def refine_entry(entry: dict[str, Any], encoder: Encoder, scorer: Support,
                 query_vecs: dict[str, np.ndarray], lam: float = LAMBDA,
                 tau1: float = TAU1, tau2: float = TAU2, max_k: int = MAX_K,
                 min_k: int = MIN_K, batch_size: int = 0) -> dict[str, Any]:
    out = {k: v for k, v in entry.items() if k not in ASPECTS}
    pools = {aspect: list(entry.get(aspect) or []) for aspect in ASPECTS}
    sentences = {aspect: [path_sentence(p) for p in pool] for aspect, pool in pools.items()}
    distinct = sorted({s for rows in sentences.values() for s in rows})
    vectors = dict(zip(distinct, encoder.encode(distinct))) if distinct else {}
    meta: dict[str, Any] = {}
    for aspect, (_, hypothesis) in ASPECTS.items():
        rows = sentences[aspect]
        path_vecs = np.stack([vectors[s] for s in rows]) if rows else np.zeros((0, 1))
        if batch_size:
            picks, relevance, stops = batched_select(rows, path_vecs, query_vecs[aspect],
                                                     hypothesis, scorer, batch_size, lam,
                                                     tau1, tau2, max_k, min_k)
            out[aspect] = [{**pools[aspect][i], "kst_pick": rank,
                            "relevance": round(float(relevance[i]), 4)}
                           for rank, i in enumerate(picks)]
            meta[aspect] = {"pool": len(rows), "picked": len(picks), "batches": len(stops),
                            "stops": dict(Counter(stops))}
            continue
        sel, relevance = mmr_nli_select(rows, path_vecs, query_vecs[aspect], hypothesis,
                                        scorer, lam, tau1, tau2, max_k, min_k)
        chosen = []
        for rank, index in enumerate(sel.picks):
            path = dict(pools[aspect][index])
            path["kst_pick"] = rank
            path["relevance"] = round(float(relevance[index]), 4)
            chosen.append(path)
        chosen.sort(key=lambda p: (-p["relevance"], p["kst_pick"]))
        out[aspect] = chosen
        meta[aspect] = {"pool": len(rows), "picked": len(sel.picks), "stop": sel.stop,
                        "support": [round(x, 4) for x in sel.support]}
    out["kst"] = meta
    return out


def build_report(results: dict[str, dict[str, Any]], params: dict[str, Any]
                 ) -> dict[str, Any]:
    report: dict[str, Any] = {"episodes": len(results), "params": params}
    for aspect in ASPECTS:
        metas = [r["kst"][aspect] for r in results.values()]
        sizes = [m["picked"] for m in metas]
        finals = [m["support"][-1] for m in metas if m.get("support")]
        stops: Counter = Counter()
        for m in metas:
            stops.update(m["stops"] if "stops" in m else [m["stop"]])
        report[aspect] = {
            "pool_median": statistics.median([m["pool"] for m in metas]) if metas else 0,
            "kst_size_median": statistics.median(sizes) if sizes else 0,
            "kst_size_hist": dict(sorted(Counter(sizes).items())),
            "stop_reasons": dict(stops),
            "final_support_median": round(statistics.median(finals), 4) if finals else None,
            "reached_tau1": stops["tau1"],
        }
    both = 0
    total = 0
    for r in results.values():
        risk = {tuple(p["nodes"]) for p in r["risk"]}
        prot = {tuple(p["nodes"]) for p in r["protective"]}
        both += len(risk & prot)
        total += len(risk | prot)
    report["paths_in_both_aspects"] = both
    report["distinct_kst_paths"] = total
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True,
                    help="a kg/retrieval/trajectory.py output (any similarity / refine)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    ap.add_argument("--nli-model", default=NLI_MODEL)
    ap.add_argument("--lambda", dest="lam", type=float, default=LAMBDA)
    ap.add_argument("--tau1", type=float, default=TAU1)
    ap.add_argument("--tau2", type=float, default=TAU2)
    ap.add_argument("--max-k", type=int, default=MAX_K)
    ap.add_argument("--min-k", type=int, default=MIN_K,
                    help="picks made before a stopping rule may fire (1 = the paper). "
                         "See mmr_nli_select for why 1 collapses KSTs to one path here")
    ap.add_argument("--batch-size", type=int, default=0,
                    help="0 = one MMR+NLI pass over the whole pool (paper); 8 = TRACER code")
    ap.add_argument("--limit", type=int, help="first N episodes only, for a smoke run")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    data = json.loads(Path(args.inp).read_text(encoding="utf-8"))
    episodes = list(data.items())[: args.limit] if args.limit else list(data.items())
    encoder = SentenceEncoder(args.encoder, args.device)
    scorer = NLISupport(args.nli_model, args.device)
    query_vecs = dict(zip(ASPECTS, encoder.encode([q for q, _ in ASPECTS.values()])))
    params = {"input": Path(args.inp).name, "encoder": args.encoder,
              "nli_model": args.nli_model, "lambda": args.lam, "tau1": args.tau1,
              "tau2": args.tau2, "max_k": args.max_k, "min_k": args.min_k,
              "batch_size": args.batch_size}
    print(f"episodes : {len(episodes)} from {args.inp}")
    print(f"encoder  : {args.encoder}   nli: {args.nli_model} on {scorer.device}")
    print(f"params   : lambda={args.lam} tau1={args.tau1} tau2={args.tau2} "
          f"max_k={args.max_k} min_k={args.min_k} batch={args.batch_size}")

    results: dict[str, dict[str, Any]] = {}
    started = time.time()
    for index, (episode_id, entry) in enumerate(episodes, 1):
        results[episode_id] = refine_entry(entry, encoder, scorer, query_vecs, args.lam,
                                           args.tau1, args.tau2, args.max_k, args.min_k,
                                           args.batch_size)
        if index % 100 == 0 or index == len(episodes):
            rate = index / max(time.time() - started, 1e-9)
            print(f"  {index}/{len(episodes)}  {rate * 60:.0f}/min  "
                  f"nli calls {scorer.calls}", flush=True)

    out = Path(args.out)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    report = build_report(results, params)
    report_path = out.with_name(out.stem + "_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for aspect in ASPECTS:
        r = report[aspect]
        print(f"{aspect:10s} pool~{r['pool_median']}  KST median {r['kst_size_median']}  "
              f"sizes {r['kst_size_hist']}  stops {r['stop_reasons']}  "
              f"final eta~{r['final_support_median']}")
    print(f"paths in both aspects: {report['paths_in_both_aspects']}"
          f"/{report['distinct_kst_paths']}")
    print(f"-> {out}\n-> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
