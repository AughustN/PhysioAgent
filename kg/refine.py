#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np

from physioagent.utils.rate_limit import RateLimiter
from physioagent.kg.combine import junk_reason
from physioagent.kg.scope.filter import DEFAULT_PROFILE, load_scope, profiles

KG_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = KG_DIR / "data"

EMBED_MODEL = "text-embedding-3-large"
EMB_DIMENSIONS = 1024
ENT_THRESHOLD = 0.14
REL_THRESHOLD = 0.14

SEARCH_MIN = 0.05
SEARCH_MAX = 0.37
SEARCH_STEPS = 32
SEARCH_SAMPLE = 4000

EMBED_BATCH = 128
EMBED_TIMEOUT_S = 120.0

EMBED_PER_MIN = 5
EMBED_RETRY_ATTEMPTS = 5
EMBED_RETRY_BASE_DELAY_S = 20.0

_ARTICLES = re.compile(r"^(the|a|an)\s+")
_PUNCT = re.compile(r"[^a-z0-9 ]+")

_PLURAL_SUFFIXES = (("ies", "y"), ("ses", "sis"), ("s", ""))


class RefineError(RuntimeError):
    pass


def normalise(text: str) -> str:
    text = _PUNCT.sub(" ", text.lower())
    text = " ".join(text.split())
    return _ARTICLES.sub("", text)


def singular_candidates(text: str) -> list[str]:
    out = []
    for suffix, replacement in _PLURAL_SUFFIXES:
        if text.endswith(suffix) and len(text) > len(suffix) + 2:
            out.append(text[: -len(suffix)] + replacement)
    return out


def normalise_tier(items: Iterable[str]) -> dict[str, str]:
    by_norm: dict[str, Counter] = defaultdict(Counter)
    for item in items:
        by_norm[normalise(item)][item] += 1

    merged: dict[str, str] = {}
    for norm in list(by_norm):
        for candidate in singular_candidates(norm):
            if candidate in by_norm and candidate != norm:
                merged[norm] = candidate
                break
    for norm, target in merged.items():
        by_norm[target].update(by_norm.pop(norm))

    canonical: dict[str, str] = {}
    for norm, surfaces in by_norm.items():
        best = surfaces.most_common(1)[0][0]
        for surface in surfaces:
            canonical[surface] = best
    return canonical


def load_cui_index(mrconso: Path, languages: tuple[str, ...] = ("ENG",),
                   wanted: Optional[set[str]] = None) -> dict[str, str]:
    index: dict[str, str] = {}
    with mrconso.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = line.split("|")
            if len(fields) < 15:
                continue
            cui, lat, string = fields[0], fields[1], fields[14]
            if lat not in languages:
                continue
            key = normalise(string)
            if not key or key in index:
                continue
            if wanted is not None and key not in wanted:
                continue
            index[key] = cui
    return index


def cui_tier(items: Iterable[str], index: dict[str, str]) -> dict[str, str]:
    by_cui: dict[str, Counter] = defaultdict(Counter)
    for item in items:
        cui = index.get(normalise(item))
        if cui:
            by_cui[cui][item] += 1
    canonical: dict[str, str] = {}
    for _, surfaces in by_cui.items():
        best = surfaces.most_common(1)[0][0]
        for surface in surfaces:
            canonical[surface] = best
    return canonical


def default_dimensions(model: str) -> Optional[int]:
    return EMB_DIMENSIONS if model.startswith("text-embedding-3") else None


def openai_embedder(api_key: str, base_url: str, model: str = EMBED_MODEL,
                    dimensions: Optional[int] = -1,
                    batch: int = EMBED_BATCH,
                    per_min: Optional[int] = EMBED_PER_MIN
                    ) -> Callable[[list[str]], list[list[float]]]:
    if dimensions == -1:
        dimensions = default_dimensions(model)
    limiter = RateLimiter(per_min=per_min, window=60.0) if per_min else None
    endpoint = base_url.rstrip("/")
    if not endpoint.endswith("/embeddings"):
        endpoint = endpoint + ("/embeddings" if endpoint.endswith("/v1")
                               else "/v1/embeddings")

    def post(texts: list[str]) -> list[list[float]]:
        body_fields: dict[str, Any] = {"model": model, "input": texts}
        if dimensions:
            body_fields["dimensions"] = dimensions
        payload = json.dumps(body_fields).encode("utf-8")
        body = None
        for attempt in range(1, EMBED_RETRY_ATTEMPTS + 1):
            if limiter is not None:
                limiter.acquire()
            request = urllib.request.Request(
                endpoint, data=payload,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {api_key}"})
            try:
                with urllib.request.urlopen(request, timeout=EMBED_TIMEOUT_S) as response:
                    body = json.loads(response.read())
                break
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504) \
                        or attempt == EMBED_RETRY_ATTEMPTS:
                    raise RefineError(f"HTTP {exc.code}: {exc.read()[:300]!r}") from exc
                delay = EMBED_RETRY_BASE_DELAY_S * attempt
                print(f"  HTTP {exc.code}; waiting {delay:.0f}s "
                      f"(attempt {attempt}/{EMBED_RETRY_ATTEMPTS})", flush=True)
                time.sleep(delay)
            except urllib.error.URLError as exc:
                if attempt == EMBED_RETRY_ATTEMPTS:
                    raise RefineError(f"embeddings endpoint unreachable: {exc}") from exc
                time.sleep(EMBED_RETRY_BASE_DELAY_S * attempt)
        if body is None:
            raise RefineError("embeddings request exhausted its retries")
        items = sorted(body["data"], key=lambda item: item.get("index", 0))
        return [item["embedding"] for item in items]

    def embed(texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), batch):
            out.extend(post(texts[start:start + batch]))
        return out

    return embed


def embed_all(texts: list[str], embed: Callable[[list[str]], list[list[float]]],
              cache_path: Optional[Path] = None) -> dict[str, list[float]]:
    cache: dict[str, list[float]] = {}
    if cache_path and cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    todo = [t for t in texts if t not in cache]

    def flush() -> None:
        if not cache_path:
            return
        temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
        temporary.write_text(json.dumps(cache), encoding="utf-8")
        temporary.replace(cache_path)

    try:
        for start in range(0, len(todo), EMBED_BATCH):
            batch = todo[start:start + EMBED_BATCH]
            for text, vector in zip(batch, embed(batch)):
                cache[text] = vector
            flush()
            print(f"  embedded {min(start + EMBED_BATCH, len(todo))}/{len(todo)}",
                  flush=True)
    except BaseException:
        flush()
        raise
    return {t: cache[t] for t in texts if t in cache}


def cluster(vectors: dict[str, list[float]], threshold: float) -> dict[str, str]:
    from sklearn.cluster import AgglomerativeClustering

    items = list(vectors)
    if len(items) < 2:
        return {item: item for item in items}
    matrix = np.asarray([vectors[i] for i in items], dtype=float)
    labels = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold,
                                     linkage="average", metric="cosine").fit_predict(matrix)
    canonical: dict[str, str] = {}
    for label in np.unique(labels):
        index = np.where(labels == label)[0]
        members = [items[i] for i in index]
        centroid = matrix[index].mean(axis=0)
        best = members[int(np.argmin(np.linalg.norm(matrix[index] - centroid, axis=1)))]
        for member in members:
            canonical[member] = best
    return canonical


def find_optimal_threshold(vectors: dict[str, list[float]], kind: str,
                           min_threshold: float = SEARCH_MIN,
                           max_threshold: float = SEARCH_MAX,
                           num_thresholds: int = SEARCH_STEPS,
                           sample_size: int = SEARCH_SAMPLE) -> tuple[float, list[dict]]:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    items = list(vectors)
    if len(items) < 3:
        return ENT_THRESHOLD, []
    if len(items) > sample_size:
        rng = np.random.default_rng(0)
        items = [items[i] for i in rng.choice(len(items), size=sample_size, replace=False)]
    matrix = np.asarray([vectors[i] for i in items], dtype=float)

    table: list[dict] = []
    best_threshold, best_score = None, -1.0
    for threshold in np.linspace(min_threshold, max_threshold, num_thresholds):
        labels = AgglomerativeClustering(n_clusters=None, distance_threshold=float(threshold),
                                         linkage="average",
                                         metric="cosine").fit_predict(matrix)
        n_labels = len(set(labels))
        if not 2 <= n_labels <= len(items) - 1:
            table.append({"threshold": round(float(threshold), 4), "clusters": n_labels,
                          "silhouette": None})
            continue
        score = float(silhouette_score(matrix, labels, metric="cosine"))
        table.append({"threshold": round(float(threshold), 4), "clusters": n_labels,
                      "silhouette": round(score, 4)})
        if score > best_score:
            best_threshold, best_score = float(threshold), score

    if best_threshold is None:
        print(f"  {kind}: no threshold scored; falling back to {ENT_THRESHOLD}")
        return ENT_THRESHOLD, table
    print(f"  {kind}: best threshold {best_threshold:.3f} "
          f"(silhouette {best_score:.3f}, {len(items)} items sampled)")
    if best_threshold > ENT_THRESHOLD * 1.5:
        print(f"  WARNING: {best_threshold:.3f} is far above the {ENT_THRESHOLD} this "
              f"pipeline was built on. Silhouette prefers big clusters and average "
              f"linkage chains into them -- read {kind}_protected_kept and the biggest "
              f"clusters in refine_mappings.json before trusting this.")
    return best_threshold, table


def protect(mapping: dict[str, str], protected: set[str]) -> tuple[dict[str, str], dict]:
    clusters: dict[str, list[str]] = defaultdict(list)
    for item, canonical in mapping.items():
        clusters[canonical].append(item)

    out = dict(mapping)
    stats = {"clusters_repointed": 0, "clusters_split": 0, "members_released": 0}
    for members in clusters.values():
        found = [m for m in members if m in protected]
        if not found:
            continue
        if len(found) == 1:
            owner = found[0]
            if mapping[owner] != owner:
                stats["clusters_repointed"] += 1
            for member in members:
                out[member] = owner
            continue
        stats["clusters_split"] += 1
        for member in members:
            if member in found:
                out[member] = member
            elif mapping[member] != member:
                out[member] = member
                stats["members_released"] += 1
    return out, stats


def compose(*layers: dict[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in {k for layer in layers for k in layer}:
        current = item
        for layer in layers:
            current = layer.get(current, current)
        result[item] = current
    return result


def refine(edges: list[dict[str, Any]], cui_index: Optional[dict[str, str]],
           embed: Optional[Callable[[list[str]], list[list[float]]]],
           cache_dir: Optional[Path] = None,
           threshold_search: bool = False,
           embed_model: Optional[str] = None,
           ent_threshold: float = ENT_THRESHOLD,
           rel_threshold: float = REL_THRESHOLD,
           protected: Optional[set[str]] = None) -> tuple[list[dict[str, Any]], dict]:
    entities = sorted({e["head"] for e in edges} | {e["tail"] for e in edges})
    relations = sorted({e["relation"] for e in edges})
    stats: dict[str, Any] = {"entities_in": len(entities), "relations_in": len(relations)}
    protected = {p.strip().lower() for p in (protected or set())}

    def run_tiers(items: list[str], threshold: float, kind: str,
                  pin: Optional[set[str]] = None) -> dict[str, str]:
        tier1 = normalise_tier(items)
        after1 = sorted(set(tier1.values()))
        stats[f"{kind}_after_string"] = len(after1)

        tier2: dict[str, str] = {}
        if cui_index:
            tier2 = cui_tier(after1, cui_index)
        after2 = sorted({tier2.get(i, i) for i in after1})
        stats[f"{kind}_after_cui"] = len(after2)

        tier3: dict[str, str] = {}
        if embed is not None and len(after2) > 1:
            slug = re.sub(r"[^a-z0-9]+", "-", (embed_model or EMBED_MODEL).lower())
            cache = (cache_dir / f"embeddings_{kind}_{slug}.json") if cache_dir else None
            stats["embed_model"] = embed_model or EMBED_MODEL
            vectors = embed_all(after2, embed, cache)
            if threshold_search:
                threshold, table = find_optimal_threshold(vectors, kind)
                stats[f"{kind}_threshold_search"] = table
            stats[f"{kind}_threshold"] = round(threshold, 4)
            tier3 = cluster(vectors, threshold)
        after3 = sorted({tier3.get(i, i) for i in after2})
        stats[f"{kind}_after_embedding"] = len(after3)
        mapping = compose(tier1, tier2, tier3)
        if pin:
            mapping, protect_stats = protect(mapping, pin)
            stats[f"{kind}_protected"] = protect_stats
            present = {p for p in pin if p in mapping}
            kept = sum(1 for p in present if mapping[p] == p)
            stats[f"{kind}_protected_kept"] = f"{kept}/{len(present)} in graph " \
                                              f"({len(pin)} declared)"
            stats[f"{kind}_after_protect"] = len({mapping.get(i, i) for i in items})
        return mapping

    print("entities ...", flush=True)
    entity_map = run_tiers(entities, ent_threshold, "entities", pin=protected)
    print("relations ...", flush=True)
    relation_map = run_tiers(relations, rel_threshold, "relations")

    seen: dict[tuple[str, str, str], dict[str, Any]] = {}
    for edge in edges:
        key = (entity_map.get(edge["head"], edge["head"]),
               relation_map.get(edge["relation"], edge["relation"]),
               entity_map.get(edge["tail"], edge["tail"]))
        if key[0] == key[2]:
            stats["self_loops_dropped"] = stats.get("self_loops_dropped", 0) + 1
            continue
        entry = seen.setdefault(key, {"head": key[0], "relation": key[1], "tail": key[2],
                                      "sources": [], "provenance": [], "merged_from": []})
        for source in edge.get("sources", []):
            if source not in entry["sources"]:
                entry["sources"].append(source)
        entry["provenance"].extend(edge.get("provenance", []))
        original = (edge["head"], edge["relation"], edge["tail"])
        if list(original) != list(key) and list(original) not in entry["merged_from"]:
            entry["merged_from"].append(list(original))

    refined = sorted(seen.values(), key=lambda e: (e["head"], e["relation"], e["tail"]))
    for edge in refined:
        edge["n_sources"] = len(edge["sources"])
    stats.update(edges_in=len(edges), edges_out=len(refined))
    stats["edges_by_agreeing_flows"] = dict(
        Counter(edge["n_sources"] for edge in refined))
    stats["multi_source_edges"] = [
        {"edge": [e["head"], e["relation"], e["tail"]], "sources": e["sources"]}
        for e in refined if e["n_sources"] > 1
    ][:50]
    return refined, {"stats": stats, "entity_map": entity_map,
                     "relation_map": relation_map}


def read_kg_raw(path: Path) -> list[dict[str, Any]]:
    edges = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                edges.append(json.loads(line))
    return edges


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA))
    ap.add_argument("--mrconso", help="path to MRCONSO.RRF; enables the CUI tier")
    ap.add_argument("--no-embeddings", action="store_true",
                    help="tiers 1 and 2 only -- deterministic, free, and fully traceable")
    ap.add_argument("--ent-threshold", type=float, default=ENT_THRESHOLD, metavar="D",
                    help=f"clustering distance for entities (default {ENT_THRESHOLD}). "
                         "A distance only means something inside one embedding space; "
                         "changing --embed-model invalidates any number tuned elsewhere.")
    ap.add_argument("--rel-threshold", type=float, default=REL_THRESHOLD, metavar="D",
                    help=f"clustering distance for relations (default {REL_THRESHOLD})")
    ap.add_argument("--profile", default=DEFAULT_PROFILE, choices=profiles(),
                    help="scope profile whose concept and seed names are pinned")
    ap.add_argument("--no-pin-scope", action="store_true",
                    help="allow canonicalisation to rename a CCS/ATC concept or a seed "
                         "term. Off by default: patient rows are looked up by those "
                         "names, so a rename breaks the EHR join rather than blurring a "
                         "distinction.")
    ap.add_argument("--threshold-search", action="store_true",
                    help="sweep the clustering threshold and keep the best silhouette, "
                         f"as TRACER does, instead of taking {ENT_THRESHOLD} on faith. "
                         "Costs no API calls -- the embeddings are already fetched -- and "
                         "writes the whole score table into refine_report.json.")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--embed-model", default=os.environ.get("EMBED_MODEL", EMBED_MODEL),
                    help=f"embedding model id (default {EMBED_MODEL}). The clustering "
                         "tier is only comparable across runs that used the SAME model, "
                         "so the one actually used is written into refine_report.json.")
    ap.add_argument("--embed-dimensions", type=int, default=-1, metavar="N",
                    help="output dimensions; 0 omits the field, -1 (default) sends it "
                         "only for text-embedding-3-* models, which are the ones that "
                         "take it.")
    ap.add_argument("--embed-batch", type=int, default=EMBED_BATCH, metavar="N",
                    help="texts per embeddings request; lower it if the provider "
                         "rejects the batch size.")
    ap.add_argument("--embed-per-min", type=int, default=EMBED_PER_MIN, metavar="N",
                    help=f"embeddings requests per minute (default {EMBED_PER_MIN}, the "
                         "gateway's measured cap -- it counts embeddings against the same "
                         "per-account limit as chat). 0 disables pacing.")
    ap.add_argument("--base-url", default=os.environ.get("EMBED_BASE_URL",
                                                         "https://api.openai.com/v1"))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    raw = data_dir / "kg_raw.jsonl"
    if not raw.exists():
        raise SystemExit(f"{raw} not found -- run physioagent.kg.combine first")
    edges = read_kg_raw(raw)
    print(f"{len(edges)} edges in")

    cui_index = None
    if args.mrconso:
        path = Path(args.mrconso)
        if not path.exists():
            raise SystemExit(f"{path} not found")
        print(f"reading {path.name} ...", flush=True)
        wanted = {normalise(s) for edge in edges
                  for s in (edge["head"], edge["tail"], edge["relation"])}
        cui_index = load_cui_index(path, wanted=wanted)
        print(f"  {len(cui_index):,} English terms indexed")

    embed = None
    if not args.no_embeddings:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise SystemExit(
                f"${args.api_key_env} is not set. Export it, or pass --no-embeddings to "
                "run the deterministic tiers only.")
        embed = openai_embedder(api_key, args.base_url, model=args.embed_model,
                                dimensions=args.embed_dimensions,
                                batch=args.embed_batch,
                                per_min=args.embed_per_min or None)
        print(f"embeddings: {args.embed_model} via {args.base_url} "
              f"({args.embed_batch}/request, {args.embed_per_min or 'un'}capped per min)")

    protected: set[str] = set()
    if not args.no_pin_scope:
        scope = load_scope(args.profile)
        protected = {n.lower() for n in scope.names()}
        protected |= {t.lower() for terms in scope.seed_terms.values() for t in terms}
        quantities = {s for edge in edges for s in (edge["head"], edge["tail"])
                      if junk_reason(s) == "quantity"}
        protected |= quantities
        print(f"pinned     : {len(protected) - len(quantities)} scope concepts and seed "
              f"terms, {len(quantities)} quantities")

    refined, result = refine(edges, cui_index, embed, data_dir,
                             threshold_search=args.threshold_search,
                             embed_model=args.embed_model if embed else None,
                             ent_threshold=args.ent_threshold,
                             rel_threshold=args.rel_threshold,
                             protected=protected)
    stats = result["stats"]

    with (data_dir / "kg_refined.jsonl").open("w", encoding="utf-8") as handle:
        for edge in refined:
            handle.write(json.dumps(edge, ensure_ascii=False) + "\n")
    (data_dir / "kg_refined.txt").write_text(
        "\n".join(f"{e['head']}\t{e['relation']}\t{e['tail']}" for e in refined) + "\n",
        encoding="utf-8")
    (data_dir / "refine_mappings.json").write_text(
        json.dumps({"entities": result["entity_map"], "relations": result["relation_map"]},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    (data_dir / "refine_report.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nentities  {stats['entities_in']} -> {stats['entities_after_string']} "
          f"(string) -> {stats['entities_after_cui']} (cui) -> "
          f"{stats['entities_after_embedding']} (embedding)")
    print(f"relations {stats['relations_in']} -> {stats['relations_after_string']} "
          f"(string) -> {stats['relations_after_cui']} (cui) -> "
          f"{stats['relations_after_embedding']} (embedding)")
    print(f"edges     {stats['edges_in']} -> {stats['edges_out']}"
          f"  (self-loops dropped: {stats.get('self_loops_dropped', 0)})")
    print(f"agreement {stats['edges_by_agreeing_flows']} (edges by number of flows, "
          f"computed AFTER canonicalisation)")
    print(f"\n-> {data_dir / 'kg_refined.txt'}\n-> {data_dir / 'kg_refined.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
