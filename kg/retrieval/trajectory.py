#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import numpy as np

from typing import Any, Callable, Iterable, Optional, Protocol, Sequence

KG_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA = KG_DIR / "data"

MAX_HOPS = 3
MAX_PATHS_PER_PAIR = 5
ALPHA = 0.8
BETA = 0.2
RHO = 0.3

RISK_QUERY = ("clinical factors that worsen fluid overload, reduce diuretic response, "
              "or require a higher furosemide dose")
PROTECTIVE_QUERY = ("clinical factors that improve diuresis, preserve kidney function, "
                    "or allow a lower furosemide dose")

STATE_CONCEPTS = {
    "is_ckd": "chronic kidney disease",
    "has_chf": "congestive heart failure; nonhypertensive",
    "has_htn": "essential hypertension",
    "aki_creat_safe": "acute and unspecified renal failure",
}
ALWAYS_CONCEPTS = ("high-ceiling diuretics",)


class Similarity(Protocol):
    def score(self, query: str, text: str) -> float: ...


_WORD = re.compile(r"[a-z0-9]+")


class LexicalSimilarity:
    def __init__(self) -> None:
        self._cache: dict[str, set[str]] = {}

    def _terms(self, text: str) -> set[str]:
        if text not in self._cache:
            self._cache[text] = set(_WORD.findall(text.lower()))
        return self._cache[text]

    def score(self, query: str, text: str) -> float:
        a, b = self._terms(query), self._terms(text)
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)


def path_sentence(triples: Sequence[tuple[str, str, str]], nodes: Sequence[str] = ()) -> str:
    if triples:
        return " ".join(f"{h} {r} {t}." for h, r, t in triples)
    return " ".join(nodes) + "."


class SentenceSimilarity:
    def __init__(self, encode: Callable[[list[str]], "np.ndarray"]) -> None:
        self.encode = encode
        self.vectors: dict[str, "np.ndarray"] = {}

    def prepare(self, texts: Iterable[str]) -> None:
        missing = sorted({t for t in texts if t not in self.vectors})
        if missing:
            for text, vector in zip(missing, self.encode(missing)):
                self.vectors[text] = np.asarray(vector, dtype=np.float32)

    def score(self, query: str, text: str) -> float:
        self.prepare([query, text])
        return float(np.dot(self.vectors[query], self.vectors[text]))

    def score_path(self, query: str, path: "KGPath") -> float:
        return self.score(query, path_sentence(path.triples, path.nodes))


def sentence_encoder(model_name: str) -> Callable[[list[str]], "np.ndarray"]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def encode(texts: list[str]) -> "np.ndarray":
        return model.encode(texts, batch_size=128, normalize_embeddings=True,
                            show_progress_bar=False, convert_to_numpy=True)

    return encode


class EmbeddingSimilarity:
    def __init__(self, vectors: dict[str, "np.ndarray"], query_vectors: dict[str, "np.ndarray"],
                 model: str) -> None:
        self.vectors = vectors
        self.query_vectors = query_vectors
        self.model = model
        self._cache: dict[tuple[str, tuple[str, ...]], float] = {}
        self.parts_missing = 0
        self.parts_seen = 0

    @staticmethod
    def _unit(vector: "np.ndarray") -> "np.ndarray":
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    def score_parts(self, query: str, parts: Sequence[str]) -> float:
        key = (query, tuple(parts))
        if key in self._cache:
            return self._cache[key]
        query_vector = self.query_vectors.get(query)
        known = []
        for part in parts:
            self.parts_seen += 1
            vector = self.vectors.get(part)
            if vector is None:
                self.parts_missing += 1
                continue
            known.append(vector)
        if query_vector is None or not known:
            score = 0.0
        else:
            centroid = self._unit(np.mean(known, axis=0))
            score = float(np.dot(centroid, query_vector))
            score = max(0.0, min(1.0, score))
        self._cache[key] = score
        return score

    def score(self, query: str, text: str) -> float:
        return self.score_parts(query, [text])


def load_embedding_similarity(data_dir: Path, model: str, queries: Sequence[str],
                              api_key: Optional[str], base_url: Optional[str],
                              needed: Optional[set[str]] = None,
                              local: bool = False) -> EmbeddingSimilarity:
    vectors: dict[str, np.ndarray] = {}
    for kind in ("entities", "relations"):
        path = data_dir / f"embeddings_{kind}_{model.replace('/', '-')}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found -- S6 writes it. The query vector and the stored "
                "vectors must come from the same embedder, so a different model's file "
                "is not a substitute.")
        for text, vector in json.loads(path.read_text(encoding="utf-8")).items():
            if needed is None or text in needed:
                vectors[text] = EmbeddingSimilarity._unit(np.asarray(vector, dtype=np.float32))

    cache_path = data_dir / f"embeddings_queries_{model.replace('/', '-')}.json"
    cached = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    missing = [q for q in queries if q not in cached]
    if missing:
        if local:
            from sentence_transformers import SentenceTransformer

            encoder = SentenceTransformer(model)
            for text, vector in zip(missing, encoder.encode(list(missing),
                                                            convert_to_numpy=True)):
                cached[text] = [float(x) for x in vector]
        elif api_key:
            from physioagent.kg.refine import openai_embedder
            embed = openai_embedder(api_key, base_url or "", model=model, per_min=None)
            for text, vector in zip(missing, embed(list(missing))):
                cached[text] = vector
        else:
            raise RuntimeError(
                f"{len(missing)} query vector(s) not cached in {cache_path.name}, no API "
                "key, and --embed-local not set. The queries are two fixed sentences, so "
                "this costs two embeddings once, not one per path.")
        cache_path.write_text(json.dumps(cached), encoding="utf-8")
        print(f"embedded {len(missing)} quer{'y' if len(missing) == 1 else 'ies'} "
              f"-> {cache_path.name}")
    query_vectors = {q: EmbeddingSimilarity._unit(np.asarray(cached[q], dtype=np.float32))
                     for q in queries}
    return EmbeddingSimilarity(vectors, query_vectors, model)


class ConstantSeverity:
    def __init__(self, value: float = 0.5) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError("severity must be normalised to [0,1]; see deviation 3")
        self.value = value

    def get(self, concept: str) -> float:
        return self.value


@dataclass
class KGPath:
    nodes: list[str]
    triples: list[tuple[str, str, str]]
    sources: list[str]
    risk: float = 0.0
    protective: float = 0.0
    from_visit: Optional[int] = None
    to_visit: Optional[int] = None
    to_index: Optional[bool] = None
    from_days: Optional[float] = None
    to_days: Optional[float] = None
    time_source: Optional[list[str]] = None

    def text(self) -> str:
        return " ".join(f"{h} {r} {t}" for h, r, t in self.triples)

    def parts(self) -> list[str]:
        out: list[str] = list(self.nodes)
        out += [relation for _, relation, _ in self.triples]
        return out

    def to_json(self) -> dict[str, Any]:
        out = {"nodes": self.nodes, "triples": [list(t) for t in self.triples],
               "sources": self.sources, "risk": round(self.risk, 4),
               "protective": round(self.protective, 4)}
        if self.from_visit is not None:
            out.update(from_visit=self.from_visit, to_visit=self.to_visit,
                       to_index=self.to_index)
        if self.from_days is not None or self.to_days is not None:
            out.update(from_days=self.from_days, to_days=self.to_days,
                       time_source=self.time_source)
        return out


NOISE_RELATIONS = frozenset({
    "has gdc value", "is value for gdc property",
    "anatomy originated from biological process", "biological process involves gene product",
})


def load_graph(path: Path):
    import networkx as nx

    graph = nx.Graph()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            edge = json.loads(line)
            head, tail = edge["head"], edge["tail"]
            if head == tail or edge.get("relation") in NOISE_RELATIONS:
                continue
            if graph.has_edge(head, tail):
                continue
            graph.add_edge(head, tail, relation=edge["relation"], head=head,
                           sources=list(edge.get("sources", [])))
    return graph


def triples_along(graph, nodes: Sequence[str]) -> tuple[list[tuple[str, str, str]],
                                                        list[str]]:
    triples: list[tuple[str, str, str]] = []
    sources: list[str] = []
    for left, right in zip(nodes, nodes[1:]):
        data = graph.edges[left, right]
        head = data["head"]
        tail = right if head == left else left
        triples.append((head, data["relation"], tail))
        for source in data.get("sources", []):
            if source not in sources:
                sources.append(source)
    return triples, sources


def paths_between(graph, source: str, target: str, cutoff: int = MAX_HOPS,
                  limit: int = MAX_PATHS_PER_PAIR) -> list[KGPath]:
    import networkx as nx

    if source not in graph or target not in graph or source == target:
        return []
    out: list[KGPath] = []
    try:
        for nodes in nx.all_shortest_paths(graph, source, target):
            if len(nodes) - 1 > cutoff:
                break
            triples, sources = triples_along(graph, nodes)
            out.append(KGPath(nodes=list(nodes), triples=triples, sources=sources))
            if len(out) >= limit:
                break
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []
    return out


def score_paths(paths: Iterable[KGPath], similarity: Similarity, severity: ConstantSeverity,
                alpha: float = ALPHA, beta: float = BETA) -> list[KGPath]:
    scored = []
    paths = list(paths)
    score_parts = getattr(similarity, "score_parts", None)
    score_path = getattr(similarity, "score_path", None)
    prepare = getattr(similarity, "prepare", None)
    if prepare is not None:
        prepare([path_sentence(p.triples, p.nodes) for p in paths])

    def sim(query: str, path: KGPath) -> float:
        if score_path is not None:
            return score_path(query, path)
        if score_parts is not None:
            return score_parts(query, path.parts())
        return similarity.score(query, path.text())

    for path in paths:
        values = [severity.get(node) for node in path.nodes]
        path_sev = sum(values) / len(values) if values else 0.0
        path.risk = alpha * sim(RISK_QUERY, path) + (1 - alpha) * path_sev
        path.protective = (beta * sim(PROTECTIVE_QUERY, path)
                           + (1 - beta) * (1 - path_sev))
        scored.append(path)
    return scored


def top_rho(paths: Sequence[KGPath], aspect: str, rho: float = RHO) -> list[KGPath]:
    if not paths:
        return []
    ranked = sorted(paths, key=lambda p: getattr(p, aspect), reverse=True)
    keep = max(1, int(round(len(ranked) * rho)))
    return ranked[:keep]


def concepts_from_state(state: dict[str, Any]) -> list[str]:
    concepts = list(ALWAYS_CONCEPTS)
    for flag, concept in STATE_CONCEPTS.items():
        value = state.get(flag)
        if value in (None, "", 0, 0.0, False):
            continue
        if concept not in concepts:
            concepts.append(concept)
    return concepts


def concepts_from_profile(profile: dict[str, Any]) -> list[str]:
    concepts: list[str] = []
    for _, _, names, _ in visits_from_profile(profile):
        for name in names:
            if name not in concepts:
                concepts.append(name)
    return concepts


@dataclass
class Pair:
    source: str
    target: str
    from_visit: Optional[int] = None
    to_visit: Optional[int] = None
    to_index: Optional[bool] = None
    from_days: Optional[float] = None
    to_days: Optional[float] = None
    time_source: Optional[list[str]] = None


def _days_before(stamp: str, t0: Optional[str]) -> Optional[float]:
    if not stamp or not t0:
        return None
    try:
        when = datetime.fromisoformat(stamp.strip().replace("T", " "))
        anchor = datetime.fromisoformat(t0.strip().replace("T", " "))
    except ValueError:
        return None
    return round((when - anchor).total_seconds() / 86400.0, 2)


def concept_time(visit: dict[str, Any], name: str) -> tuple[str, str]:
    stamp = (visit.get("concept_times") or {}).get(name)
    if stamp:
        return stamp, "event"
    return visit.get("admittime", "") or "", "admission"


def visits_from_profile(profile: dict[str, Any]
                        ) -> list[tuple[int, bool, list[str], dict[str, tuple[str, str]]]]:
    visits = []
    for key, visit in profile.items():
        if not key.startswith("visit_"):
            continue
        names: list[str] = []
        for name in (visit.get("conditions", []) + visit.get("procedures", [])
                     + visit.get("drugs", [])):
            if name not in names:
                names.append(name)
        if names:
            times = {name: concept_time(visit, name) for name in names}
            visits.append((int(key.split("_")[1]), bool(visit.get("is_index")), names,
                           times))
    return sorted(visits, key=lambda v: v[0])


def pairs_from_profile(profile: dict[str, Any]) -> list[Pair]:
    visits = visits_from_profile(profile)
    t0 = profile.get("index_t0")
    pairs: list[Pair] = []
    seen: set[frozenset[str]] = set()

    def timed(pair: Pair, source_time: tuple[str, str],
              target_time: tuple[str, str]) -> Pair:
        pair.from_days = _days_before(source_time[0], t0)
        pair.to_days = _days_before(target_time[0], t0)
        pair.time_source = [source_time[1], target_time[1]]
        return pair

    if len(visits) == 1:
        number, is_index, names, times = visits[0]
        for i, first in enumerate(names):
            for second in names[i + 1:]:
                seen.add(frozenset((first, second)))
                source, target = first, second
                if times[first][0] and times[second][0] and times[second][0] < times[first][0]:
                    source, target = second, first
                pairs.append(timed(Pair(source, target, number, number, is_index),
                                   times[source], times[target]))
        return pairs
    for (n_from, _, sources, t_from), (n_to, to_index, targets, t_to) in reversed(
            list(zip(visits, visits[1:]))):
        for source in sources:
            for target in targets:
                key = frozenset((source, target))
                if source == target or key in seen:
                    continue
                seen.add(key)
                pairs.append(timed(Pair(source, target, n_from, n_to, to_index),
                                   t_from[source], t_to[target]))
    return pairs


def pairs_from_concepts(concepts: Sequence[str]) -> list[Pair]:
    pairs: list[Pair] = []
    seen: set[frozenset[str]] = set()
    for index, source in enumerate(concepts):
        for target in concepts[index + 1:]:
            key = frozenset((source, target))
            if source == target or key in seen:
                continue
            seen.add(key)
            pairs.append(Pair(source, target))
    return pairs


def retrieve(concepts: Sequence[str], graph, similarity: Similarity,
             severity: ConstantSeverity, rho: float = RHO,
             cutoff: int = MAX_HOPS, refiner: Optional[Any] = None,
             refine_top: int = 0) -> dict[str, Any]:
    return retrieve_pairs(pairs_from_concepts(concepts), graph, similarity, severity,
                          rho, cutoff, refiner, refine_top,
                          n_concepts=len(set(concepts)))


def retrieve_pairs(pairs: Sequence[Pair], graph, similarity: Similarity,
                   severity: ConstantSeverity, rho: float = RHO,
                   cutoff: int = MAX_HOPS, refiner: Optional[Any] = None,
                   refine_top: int = 0, n_concepts: Optional[int] = None,
                   n_visits: Optional[int] = None) -> dict[str, Any]:
    paths: list[KGPath] = []
    for pair in pairs:
        for path in paths_between(graph, pair.source, pair.target, cutoff=cutoff):
            path.from_visit, path.to_visit, path.to_index = (
                pair.from_visit, pair.to_visit, pair.to_index)
            path.from_days, path.to_days, path.time_source = (
                pair.from_days, pair.to_days, pair.time_source)
            paths.append(path)
    scored = score_paths(paths, similarity, severity)
    if refiner is not None:
        risk_pool = top_rho(scored, "risk", rho)
        protective_pool = top_rho(scored, "protective", rho)
        if refine_top:
            risk_pool = risk_pool[:refine_top]
            protective_pool = protective_pool[:refine_top]
        candidates = {id(p): p for p in risk_pool}
        candidates.update({id(p): p for p in protective_pool})
        refiner.refine(list(candidates.values()))
        scored = list(candidates.values())
    if n_concepts is None:
        n_concepts = len({c for p in pairs for c in (p.source, p.target)})
    return {
        "n_concepts": n_concepts,
        "n_visits": n_visits,
        "n_pairs": len(pairs),
        "n_paths": len(scored),
        "risk": [p.to_json() for p in top_rho(scored, "risk", rho)],
        "protective": [p.to_json() for p in top_rho(scored, "protective", rho)],
    }


def build_report(results: dict[str, dict[str, Any]], graph) -> dict[str, Any]:
    lengths = Counter()
    sources = Counter()
    touching_pubmed = 0
    total_kept = 0
    for result in results.values():
        for aspect in ("risk", "protective"):
            for path in result[aspect]:
                total_kept += 1
                lengths[len(path["nodes"]) - 1] += 1
                for source in path["sources"]:
                    sources[source] += 1
                if "pubmed" in path["sources"]:
                    touching_pubmed += 1
    with_paths = sum(1 for r in results.values() if r["n_paths"])
    windows = Counter()
    for result in results.values():
        for aspect in ("risk", "protective"):
            for path in result[aspect]:
                if path.get("from_visit") is None:
                    windows["stand_in"] += 1
                elif path["from_visit"] == path["to_visit"]:
                    windows["single_visit"] += 1
                elif path.get("to_index"):
                    windows["last_prior_to_index"] += 1
                else:
                    windows["prior_to_prior"] += 1
    visit_counts = Counter(r.get("n_visits") for r in results.values())
    return {
        "kept_paths_by_window": dict(windows),
        "episodes_by_visit_count": {str(k): v for k, v in sorted(
            visit_counts.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))},
        "episodes": len(results),
        "episodes_with_any_path": with_paths,
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "pairs_total": sum(r["n_pairs"] for r in results.values()),
        "paths_found": sum(r["n_paths"] for r in results.values()),
        "paths_kept": total_kept,
        "path_hop_counts": {str(k): v for k, v in sorted(lengths.items())},
        "edges_by_source_in_kept_paths": dict(sources),
        "paths_touching_pubmed": touching_pubmed,
        "protective_paths": sum(len(r["protective"]) for r in results.values()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", default=str(DEFAULT_DATA / "kg_refined.jsonl"))
    ap.add_argument("--sequences", default=str(DEFAULT_DATA / "sequences.json"))
    ap.add_argument("--doses", help="doses.json (build_doses) instead of --sequences; "
                                    "results are keyed on dose_id, which is what "
                                    "patient_profiles --doses and run_eval_dose use")
    ap.add_argument("--profiles", help="patient_profiles.json; without it the concept "
                                       "set is derived from the episode's state flags")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=str(DEFAULT_DATA / "trajectories.json"))
    ap.add_argument("--rho", type=float, default=RHO)
    ap.add_argument("--hops", type=int, default=MAX_HOPS)
    ap.add_argument("--similarity", choices=("lexical", "embedding", "sentence"),
                    default="lexical",
                    help="how a path is scored against the two aspect queries. 'lexical' "
                         "is word overlap, the stand-in. 'embedding' is cosine in the "
                         "space S6 already embedded the graph into, which is what makes "
                         "the risk and protective rankings differ by meaning rather than "
                         "by which words the two query sentences happen to share")
    ap.add_argument("--refine", choices=("none", "nli"), default="none",
                    help="TRACER 4.3.2. 'nli' re-scores the retrieved paths with an "
                         "entailment model, so the risk/protective split reflects which "
                         "way a path argues rather than which topic it is about. Runs "
                         "locally on the GPU; see kg/retrieval/nli.py for the measurement "
                         "that motivates it")
    ap.add_argument("--refine-top", type=int, default=0, metavar="N",
                    help="with --refine nli, run entailment on only the top N paths per "
                         "aspect from the similarity ranking (0 = all of top-rho). The "
                         "block renders 8 paths, so N well above 8 changes nothing a "
                         "reader would see while cutting the transformer's work several "
                         "fold")
    ap.add_argument("--nli-model", default=None,
                    help="entailment checkpoint for --refine nli "
                         "(default: PubMedBERT-MNLI-MedNLI)")
    ap.add_argument("--embed-local", action="store_true",
                    help="embed the two aspect queries with sentence-transformers on this "
                         "machine instead of calling an endpoint. Use it with vectors "
                         "written by kg.retrieval.embed_local -- query and stored vectors "
                         "must come from the same weights")
    ap.add_argument("--embed-model", default="google/gemini-embedding-001",
                    help="must be the model whose vectors S6 wrote; a cosine across two "
                         "embedding spaces is meaningless")
    args = ap.parse_args()

    graph_path = Path(args.graph)
    if not graph_path.exists():
        raise SystemExit(f"{graph_path} not found -- run physioagent.kg.refine")
    graph = load_graph(graph_path)
    print(f"graph      : {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    source = Path(args.doses or args.sequences)
    id_key = "dose_id" if args.doses else "episode_id"
    episodes = [e for e in json.loads(source.read_text(encoding="utf-8"))
                if e["split"] == args.split]
    profiles = json.loads(Path(args.profiles).read_text(encoding="utf-8")) \
        if args.profiles else None
    print(f"rows       : {len(episodes)} in split {args.split!r} of {source.name}, "
          f"keyed on {id_key}")
    print(f"concepts   : {'patient_profiles.json' if profiles else 'state flags (stand-in)'}")

    if args.similarity == "sentence":
        similarity = SentenceSimilarity(sentence_encoder(args.embed_model))
        similarity.prepare([RISK_QUERY, PROTECTIVE_QUERY])
        print(f"similarity : whole path sentence, {args.embed_model}")
    elif args.similarity == "embedding":
        api_key = base_url = None
        if not args.embed_local:
            from physioagent.config.loader import load_settings
            from physioagent.llm.client import LLMClient
            client = LLMClient.from_settings(load_settings())
            api_key, base_url = client.api_key, client.base_url
        needed = set(graph.nodes()) | {d.get("relation") for _, _, d in graph.edges(data=True)}
        needed.discard(None)
        similarity = load_embedding_similarity(
            DEFAULT_DATA, args.embed_model, (RISK_QUERY, PROTECTIVE_QUERY),
            api_key, base_url, needed, args.embed_local)
        print(f"similarity : cosine, {args.embed_model}, "
              f"{len(similarity.vectors):,} vectors loaded")
    else:
        similarity = LexicalSimilarity()
        print("similarity : lexical overlap (stand-in)")
    severity = ConstantSeverity()

    refiner = None
    if args.refine == "nli":
        from physioagent.kg.retrieval.nli import DEFAULT_MODEL as NLI_DEFAULT, NLIRefiner

        refiner = NLIRefiner(model_name=args.nli_model or NLI_DEFAULT).load()
        print(f"refinement : entailment, {refiner.model_name}, device {refiner.device}")
    results: dict[str, dict[str, Any]] = {}
    started = time.time()
    for index, episode in enumerate(episodes, 1):
        if refiner is not None and (index % 50 == 0 or index == 1):
            rate = index / max(time.time() - started, 1e-9)
            remaining = (len(episodes) - index) / rate if rate else 0
            print(f"  {index}/{len(episodes)} episodes  {rate * 60:.1f}/min  "
                  f"~{remaining / 60:.0f} min left", flush=True)
        key = episode[id_key]
        if profiles is not None and key in profiles:
            profile = profiles[key]
            results[key] = retrieve_pairs(
                pairs_from_profile(profile), graph, similarity, severity,
                rho=args.rho, cutoff=args.hops, refiner=refiner,
                refine_top=args.refine_top,
                n_concepts=len(concepts_from_profile(profile)),
                n_visits=len(visits_from_profile(profile)))
        else:
            results[key] = retrieve(concepts_from_state(episode.get("state", {})), graph,
                                    similarity, severity, rho=args.rho, cutoff=args.hops,
                                    refiner=refiner, refine_top=args.refine_top)

    out = Path(args.out)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    report = build_report(results, graph)
    (DEFAULT_DATA / "trajectory_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{report['paths_found']} paths over {report['pairs_total']} pairs; "
          f"{report['paths_kept']} kept at rho={args.rho}")
    print(f"  episodes with a path : {report['episodes_with_any_path']}/{report['episodes']}")
    print(f"  hops                 : {report['path_hop_counts']}")
    print(f"  sources in kept paths: {report['edges_by_source_in_kept_paths']}")
    print(f"  touching PubMed      : {report['paths_touching_pubmed']}/{report['paths_kept']}")
    print(f"  protective kept      : {report['protective_paths']}")
    print(f"  kept by window       : {report['kept_paths_by_window']}")
    print(f"  rows by visit count  : {report['episodes_by_visit_count']}")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
