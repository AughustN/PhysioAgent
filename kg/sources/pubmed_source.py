#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from physioagent.config.loader import load_settings
from physioagent.utils.rate_limit import RateLimiter
from physioagent.kg.scope.filter import DEFAULT_PROFILE, Scope, load_scope, profiles
from physioagent.llm.client import (
    DEFAULT_ENV_FILE,
    LLMClient,
    LLMClientError,
    _load_env_values,
)

KG_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA = KG_DIR / "data"
RESOURCES = KG_DIR / "resources"

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
TOOL_NAME = "physioagent-kg"

NCBI_PER_MIN_NO_KEY = 150
NCBI_PER_MIN_WITH_KEY = 480

ABSTRACTS_PER_CONCEPT = 10
EFETCH_BATCH = 5
MAX_TRIPLES_PER_ABSTRACT = 10

DOMAIN_ANCHOR = ('("furosemide"[tiab] OR "loop diuretic"[tiab] OR "diuresis"[tiab] '
                 'OR "urine output"[tiab] OR "kidney"[tiab])')
MIN_ANCHORED_HITS = 3

MAX_ENTITY_CHARS = 80

RETRY_ATTEMPTS = 4
RETRY_BASE_DELAY_S = 2.0


class PubMedError(RuntimeError):
    pass


def env_value(name: str) -> Optional[str]:
    value = os.environ.get(name)
    if value:
        return value
    return _load_env_values(DEFAULT_ENV_FILE).get(name) or None


def _get(url: str, timeout: float = 60.0) -> bytes:
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": TOOL_NAME})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            if attempt == RETRY_ATTEMPTS:
                raise PubMedError(f"{type(exc).__name__}: {exc}") from exc
            time.sleep(RETRY_BASE_DELAY_S * (2 ** (attempt - 1)))
    raise PubMedError("unreachable")


@dataclass
class Entrez:
    api_key: Optional[str] = None
    email: Optional[str] = None
    limiter: Optional[RateLimiter] = None

    def _params(self, extra: dict[str, Any]) -> str:
        params = {"tool": TOOL_NAME, **extra}
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        return urllib.parse.urlencode(params)

    def _call(self, endpoint: str, params: dict[str, Any]) -> bytes:
        if self.limiter is not None:
            self.limiter.acquire()
        return _get(EUTILS + endpoint + "?" + self._params(params))

    def esearch(self, term: str, retmax: int) -> list[str]:
        raw = self._call("esearch.fcgi", {"db": "pubmed", "term": term, "retmax": retmax,
                                          "retmode": "json", "sort": "relevance"})
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PubMedError(f"esearch returned non-JSON: {raw[:200]!r}") from exc
        return list(payload.get("esearchresult", {}).get("idlist", []))

    def efetch(self, pmids: list[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for start in range(0, len(pmids), EFETCH_BATCH):
            batch = pmids[start:start + EFETCH_BATCH]
            raw = self._call("efetch.fcgi", {"db": "pubmed", "id": ",".join(batch),
                                             "retmode": "xml", "rettype": "abstract"})
            try:
                root = ET.fromstring(raw)
            except ET.ParseError:
                continue
            for article in root.findall(".//PubmedArticle"):
                pmid_el = article.find(".//PMID")
                if pmid_el is None or not pmid_el.text:
                    continue
                chunks = [(el.text or "").strip()
                          for el in article.findall(".//Abstract/AbstractText")]
                text = " ".join(c for c in chunks if c)
                if text:
                    out[pmid_el.text.strip()] = text
        return out


_PUNCT = re.compile(r"[^a-z0-9 ]+")


def normalise(text: str) -> str:
    return " ".join(_PUNCT.sub(" ", text.lower()).split())


def find_anchor(entity_norm: str,
                concepts_norm: dict[str, str]) -> tuple[Optional[str], str]:
    if entity_norm in concepts_norm:
        return concepts_norm[entity_norm], "exact"
    padded = f" {entity_norm} "
    best_norm: Optional[str] = None
    best_name: Optional[str] = None
    for norm, name in concepts_norm.items():
        if f" {norm} " in padded and (best_norm is None or len(norm) > len(best_norm)):
            best_norm, best_name = norm, name
    if best_name is None:
        return None, "none"
    return best_name, "substring"


def ground_triple(triple: tuple[str, str, str], abstract_norm: str,
                  concepts_norm: dict[str, str]) -> Optional[dict[str, Any]]:
    head, relation, tail = (part.strip() for part in triple)
    if not head or not relation or not tail:
        return None
    if len(head) > MAX_ENTITY_CHARS or len(tail) > MAX_ENTITY_CHARS:
        return None
    head_norm, tail_norm = normalise(head), normalise(tail)
    if not head_norm or not tail_norm or head_norm == tail_norm:
        return None

    substituted = []
    for name, norm in (("head", head_norm), ("tail", tail_norm)):
        if norm in abstract_norm:
            continue
        if norm in concepts_norm:
            substituted.append(name)
            continue
        return None

    head_anchor, head_match = find_anchor(head_norm, concepts_norm)
    tail_anchor, tail_match = find_anchor(tail_norm, concepts_norm)
    anchors = [a for a in (head_anchor, tail_anchor) if a]
    if not anchors:
        return None

    matches = [m for m in (head_match, tail_match) if m != "none"]
    return {
        "head": head.lower(),
        "relation": " ".join(relation.lower().split()),
        "tail": tail.lower(),
        "anchor_substituted": substituted,
        "anchors": anchors,
        "anchor_match": "exact" if "exact" in matches else "substring",
    }


SYSTEM_PROMPT = """You extract relationships that are STATED IN A GIVEN TEXT. You never \
add knowledge of your own: if the text does not say it, it does not exist. Return valid \
JSON only."""

USER_TEMPLATE = """Given a medical text and a list of important concepts, extract the \
relationships between concepts that the text actually states.

Rules:
- Every ENTITY1 and ENTITY2 must appear in the text. Do not introduce entities the text \
does not mention, and do not add relationships from your own knowledge.
- If an entity matches one of the given concepts, write it using the exact concept term.
- At most {max_triples} triples. Prefer the ones a clinician would call informative.
- If the text states no relationship between these concepts, return an empty list.

Example:
Text:
Asthma is a chronic respiratory condition characterized by inflammation and narrowing of \
the airways, leading to breathing difficulties. Wheezing and coughing are common symptoms. \
Inhaled corticosteroids are used for long-term control.

Concepts: [asthma, inflammation, airways, wheezing, coughing, inhaled corticosteroids]

Output:
{{"triples": [["asthma", "is a", "chronic respiratory condition"], \
["asthma", "characterized by", "inflammation of airways"], \
["inflammation", "causes", "narrowing of airways"], \
["wheezing", "is a symptom of", "asthma"], \
["coughing", "is a symptom of", "asthma"], \
["inhaled corticosteroids", "used for", "long-term control of asthma"]]}}

Text:
{text}

Concepts: {concepts}

Output:
"""


def parse_triples(response: Any) -> list[tuple[str, str, str]]:
    candidates: Iterable[Any] = ()
    if isinstance(response, list):
        candidates = response
    elif isinstance(response, dict):
        for key in ("triples", "relationships", "output", "result", "data"):
            if isinstance(response.get(key), list):
                candidates = response[key]
                break
        else:
            for value in response.values():
                if isinstance(value, list):
                    candidates = value
                    break
    triples: list[tuple[str, str, str]] = []
    for item in candidates:
        if isinstance(item, (list, tuple)) and len(item) == 3:
            triples.append(tuple(str(part) for part in item))  # type: ignore[arg-type]
        elif isinstance(item, dict):
            head = item.get("head") or item.get("subject") or item.get("entity1")
            rel = item.get("relation") or item.get("predicate") or item.get("relationship")
            tail = item.get("tail") or item.get("object") or item.get("entity2")
            if head and rel and tail:
                triples.append((str(head), str(rel), str(tail)))
    return triples


@dataclass
class Unit:
    key: str
    concepts: list[str]
    kind: str
    query_terms: list[str] = field(default_factory=list)
    co_terms: list[str] = field(default_factory=list)

    def query(self, anchored: bool) -> str:
        terms = self.query_terms or self.concepts
        body = " OR ".join(f'"{t}"[tiab]' for t in terms)
        if self.co_terms:
            companions = " OR ".join(f'"{t}"[tiab]' for t in self.co_terms)
            body = f"({body}) AND ({companions})"
        return f"({body}) AND {DOMAIN_ANCHOR}" if anchored else f"({body})"


def load_query_terms() -> dict[str, list[str]]:
    path = RESOURCES / "pubmed_terms.csv"
    table: dict[str, list[str]] = {}
    if not path.exists():
        return table
    with path.open(encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if not row or not row[0].strip() or row[0].lstrip().startswith("#"):
                continue
            if row[0].strip() == "concept":
                continue
            terms = [t.strip() for t in (row[1] if len(row) > 1 else "").split("|")
                     if t.strip()]
            if terms:
                table[row[0].strip().lower()] = terms
    return table


SET_COMPANIONS = 6
MIN_SET_COMPANIONS = 3


def build_units(scope: Scope, top_coexisting: Optional[dict[str, list[str]]],
                max_sets: int = 0) -> list[Unit]:
    query_terms = load_query_terms()
    units: list[Unit] = []
    for name in sorted(scope.names()):
        neighbours = (top_coexisting or {}).get(name, [])[:20]
        units.append(Unit(key=f"concept::{name}", kind="concept",
                          concepts=[name] + list(neighbours),
                          query_terms=query_terms.get(name, [name])))
    for theme, terms in scope.seed_terms.items():
        units.append(Unit(key=f"seed::{theme}", kind="seed_theme",
                          concepts=list(terms), query_terms=list(terms)[:6]))

    if max_sets and top_coexisting:
        in_scope = {name.lower() for name in scope.names()}
        ranked = sorted(
            ((name, [n for n in neighbours if n.lower() in in_scope])
             for name, neighbours in top_coexisting.items()
             if name.lower() in in_scope),
            key=lambda item: (-len(item[1]), item[0]))
        made = 0
        for name, neighbours in ranked:
            if made >= max_sets:
                break
            if len(neighbours) < MIN_SET_COMPANIONS:
                continue
            made += 1
            companions: list[str] = []
            for companion in neighbours[:SET_COMPANIONS]:
                companions.extend(query_terms.get(companion.lower(), [companion])[:2])
            units.append(Unit(key=f"set::{name}", kind="concept_set",
                              concepts=[name] + list(neighbours[:20]),
                              query_terms=query_terms.get(name, [name]),
                              co_terms=companions))
    return units


def process_unit(unit: Unit, entrez: Entrez, client: LLMClient, scope_norm: dict[str, str],
                 llm_limiter: Optional[RateLimiter], abstracts_per_unit: int,
                 strict: bool) -> dict[str, Any]:
    started = time.time()
    record: dict[str, Any] = {"unit": unit.key, "kind": unit.kind,
                              "anchored": True, "pmids": [], "triples": [],
                              "dropped": Counter(), "errors": []}
    try:
        pmids = entrez.esearch(unit.query(anchored=True), abstracts_per_unit)
        if len(pmids) < MIN_ANCHORED_HITS:
            record["anchored"] = False
            pmids = entrez.esearch(unit.query(anchored=False), abstracts_per_unit)
        abstracts = entrez.efetch(pmids) if pmids else {}
    except PubMedError as exc:
        record["errors"].append(f"entrez: {exc}")
        record["dropped"] = dict(record["dropped"])
        record["elapsed_s"] = round(time.time() - started, 2)
        return record

    record["pmids"] = list(abstracts)
    concepts_for_prompt = unit.concepts[:24]
    for pmid, text in abstracts.items():
        if llm_limiter is not None:
            llm_limiter.acquire()
        prompt = USER_TEMPLATE.format(max_triples=MAX_TRIPLES_PER_ABSTRACT,
                                      text=text, concepts=concepts_for_prompt)
        try:
            response = client.chat_json(SYSTEM_PROMPT, prompt)
        except (LLMClientError, Exception) as exc:  # noqa: BLE001 - one abstract must not
            record["errors"].append(f"{pmid}: {type(exc).__name__}: {exc}")
            continue
        usage = dict(getattr(client, "last_usage", None) or {})
        if usage:
            usage["pmid"] = pmid
            record.setdefault("usage", []).append(usage)
        abstract_norm = normalise(text)
        for raw in parse_triples(response)[:MAX_TRIPLES_PER_ABSTRACT]:
            grounded = ground_triple(raw, abstract_norm, scope_norm)
            if grounded is None:
                record["dropped"]["ungrounded"] += 1
                continue
            if strict and grounded["anchor_substituted"]:
                record["dropped"]["substituted"] += 1
                continue
            grounded.update(pmid=pmid, unit=unit.key, unit_kind=unit.kind,
                            anchored_query=record["anchored"])
            record["triples"].append(grounded)
    record["dropped"] = dict(record["dropped"])
    record["elapsed_s"] = round(time.time() - started, 2)
    return record


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    done: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not record.get("errors"):
                done[record["unit"]] = record
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default=DEFAULT_PROFILE, choices=profiles())
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA))
    ap.add_argument("--abstracts", type=int, default=ABSTRACTS_PER_CONCEPT)
    ap.add_argument("--limit", type=int, help="first N units only, for a smoke run")
    ap.add_argument("--max-sets", type=int, default=0, metavar="N",
                    help="also build N concept_set units: one anchor concept ANDed with "
                         "the companions it co-occurs with in this cohort, TRACER's unit "
                         "of work cut to a budget. These are what produce edges BETWEEN "
                         "two scope concepts; the concept units mostly produce edges from "
                         "one concept out to free text. Costs N x --abstracts calls. "
                         "Needs all_top_coexisting_concepts.json.")
    ap.add_argument("--only-kind", choices=("concept", "seed_theme", "concept_set"),
                    help="run only units of this kind. The seed themes are the 4 units "
                         "carrying the tier_c mechanism vocabulary -- the terms UMLS "
                         "resolved to no CUI -- and they sort last, so --limit can never "
                         "reach them. This is how you test whether the flow fills that "
                         "gap without paying for all 103 concept units first.")
    ap.add_argument("--strict", action="store_true",
                    help="also drop triples whose entity was written in the canonical "
                         "concept wording rather than the abstract's own")
    ap.add_argument("--fresh", action="store_true", help="delete the checkpoint first")
    ap.add_argument("--dry-run", action="store_true",
                    help="search and fetch only; no LLM calls. Shows how many abstracts "
                         "each concept actually has before any money is spent.")
    ap.add_argument("--rate-per-min", type=int, default=None, metavar="N",
                    help="LLM calls per minute. Defaults to the eval harness's cap (5), "
                         "which is the free tier of the gateway it was written against; "
                         "raise it to match whatever provider is actually configured. "
                         "Going over the real cap does not fail fast -- it fills the "
                         "checkpoint with 429s that look like an outage.")
    args = ap.parse_args()

    scope = load_scope(args.profile)
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    top_path = data_dir / "all_top_coexisting_concepts.json"
    top_coexisting = json.loads(top_path.read_text(encoding="utf-8")) \
        if top_path.exists() else None
    if top_coexisting is None:
        print(f"note: {top_path.name} not found; extraction runs without neighbour context")

    api_key = env_value("NCBI_API_KEY")
    entrez = Entrez(
        api_key=api_key,
        email=env_value("NCBI_EMAIL"),
        limiter=RateLimiter(per_min=NCBI_PER_MIN_WITH_KEY if api_key else NCBI_PER_MIN_NO_KEY,
                            window=60.0),
    )
    print(f"NCBI key   : {'set' if api_key else 'NOT SET (3 req/s cap)'}")

    if args.max_sets and top_coexisting is None:
        print("note: --max-sets needs all_top_coexisting_concepts.json; no sets built")
    units = build_units(scope, top_coexisting, max_sets=args.max_sets)
    if args.only_kind:
        units = [u for u in units if u.kind == args.only_kind]
    if args.limit:
        units = units[:args.limit]
    print(f"units      : {len(units)} "
          f"({sum(1 for u in units if u.kind == 'concept')} concepts + "
          f"{sum(1 for u in units if u.kind == 'seed_theme')} seed themes + "
          f"{sum(1 for u in units if u.kind == 'concept_set')} concept sets)")

    checkpoint = data_dir / "pubmed_checkpoint.jsonl"
    if args.fresh and checkpoint.exists():
        checkpoint.unlink()
    done = load_checkpoint(checkpoint)
    todo = [u for u in units if u.key not in done]
    print(f"to run     : {len(todo)} ({len(done)} already in {checkpoint.name})")

    if args.dry_run:
        for unit in todo:
            pmids = entrez.esearch(unit.query(anchored=True), args.abstracts)
            note = ""
            if len(pmids) < MIN_ANCHORED_HITS:
                pmids = entrez.esearch(unit.query(anchored=False), args.abstracts)
                note = "  (fell back to unanchored)"
            print(f"  {len(pmids):3d} abstracts  {unit.key}{note}")
        print(f"\ndry run: {len(todo)} units would cost about "
              f"{len(todo) * args.abstracts} extraction calls")
        return 0

    client = LLMClient.from_settings(load_settings())
    client.timeout = 180.0
    print(f"model      : {client.model_name} via {client.provider}")
    llm_limiter = RateLimiter(per_min=args.rate_per_min) if args.rate_per_min \
        else RateLimiter()
    per_min = args.rate_per_min or llm_limiter.per_min
    calls = sum(1 for _ in todo) * args.abstracts
    print(f"rate       : {per_min}/min  -> about {calls / max(per_min, 1) / 60:.1f} h "
          f"for {calls} calls at most")

    scope_norm = {normalise(name): name for name in scope.names()}
    for terms in scope.seed_terms.values():
        for term in terms:
            scope_norm[normalise(term)] = term

    with checkpoint.open("a", encoding="utf-8") as handle:
        for index, unit in enumerate(todo, 1):
            record = process_unit(unit, entrez, client, scope_norm, llm_limiter,
                                  args.abstracts, args.strict)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            done[unit.key] = record
            print(f"  [{index}/{len(todo)}] {unit.key}: "
                  f"{len(record['triples'])} triples from {len(record['pmids'])} abstracts"
                  f"{' (unanchored query)' if not record['anchored'] else ''}"
                  f"{'  ERRORS' if record['errors'] else ''}", flush=True)

    triples = [t for record in done.values() for t in record["triples"]]
    jsonl = data_dir / "kg_from_pubmed.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for triple in triples:
            handle.write(json.dumps(triple, ensure_ascii=False) + "\n")
    lines = {f"{t['head']}\t{t['relation']}\t{t['tail']}" for t in triples}
    (data_dir / "kg_from_pubmed.txt").write_text("\n".join(sorted(lines)) + "\n",
                                                 encoding="utf-8")

    dropped = Counter()
    for record in done.values():
        dropped.update(record.get("dropped") or {})
    entities = {t["head"] for t in triples} | {t["tail"] for t in triples}
    kept = len(triples)

    calls = [u for record in done.values() for u in (record.get("usage") or [])]
    if calls:
        prompt_tok = sum(u.get("prompt_tokens", 0) for u in calls)
        completion_tok = sum(u.get("completion_tokens", 0) for u in calls)
        total_tok = sum(u.get("total_tokens", 0) for u in calls)
        reasoning = sum((u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
                        for u in calls)
        print(f"\nbilling: {len(calls)} calls  prompt {prompt_tok:,}  "
              f"completion {completion_tok:,}  total {total_tok:,}")
        if reasoning:
            print(f"  reasoning tokens reported separately: {reasoning:,}")
        unexplained = total_tok - prompt_tok - completion_tok
        if unexplained > 0:
            print(f"  WARNING: total exceeds prompt+completion by {unexplained:,} tokens "
                  f"-- those are billed and were invisible to the cost estimate")

    print(f"\n{kept} triples ({len(lines)} distinct edges), {len(entities)} entities")
    print(f"  dropped   : {dict(dropped)}")
    substring = sum(1 for t in triples if t.get("anchor_match") == "substring")
    if kept:
        print(f"  anchors   : {kept - substring} exact, {substring} substring-only "
              f"({substring / kept:.1%} would have died under equality)")
    by_kind = Counter(t.get("unit_kind", "?") for t in triples)
    if by_kind.get("concept_set"):
        both = sum(1 for t in triples if len(t.get("anchors") or []) > 1)
        print(f"  by unit   : {dict(by_kind)}; {both} triples join two scope concepts")
    if kept + sum(dropped.values()):
        rate = kept / (kept + sum(dropped.values()))
        print(f"  grounded  : {rate:.1%} of extracted triples survived the text check")
    print(f"  unanchored: {sum(1 for r in done.values() if not r['anchored'])} units")
    print(f"  errors    : {sum(len(r['errors']) for r in done.values())}")
    print(f"\n-> {jsonl}\n-> {data_dir / 'kg_from_pubmed.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
