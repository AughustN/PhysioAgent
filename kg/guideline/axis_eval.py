#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from physioagent.kg.guideline.passages import split_sentences

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PASSAGES = PACKAGE_ROOT / "kg" / "data" / "guideline_passages.jsonl"
OUT_DIR = PACKAGE_ROOT / "kg" / "data" / "guideline_axis_eval"
ENCODER = "NeuML/pubmedbert-base-embeddings"
SEED = 20261002
N_PAIRS = 250
MIN_COSINE = 0.6
BANDS = ((0.6, 0.75), (0.75, 0.85), (0.85, 0.93), (0.93, 1.0001))
FOLDS = 5
BOOTSTRAP = 2000

DASH = re.compile(r"[‐‑‒–—−]")
NUM = r"\d+(?:\.\d+)?"
UNIT = (r"mg/kg/(?:day|d|h|hour)|mg/kg|g/kg|mg/(?:day|d|24 ?h|h|hour)|mL/(?:kg/)?h|ml/(?:kg/)?h"
        r"|mcg|µg|μg|mg|g|mmol|mEq|times|×|-fold|fold")
QUANTITY = re.compile(
    rf"(?P<cmp>up to|maximum(?: of)?|max\.?|no more than|not (?:to )?exceed|at least|minimum(?: of)?"
    rf"|≤|<=|≥|>=|<|>)?\s*(?P<lo>{NUM})(?:\s*-\s*(?P<hi>{NUM}))?\s*(?P<unit>{UNIT})(?![a-zA-Z])"
    rf"(?!\s*/\s*(?:dl|l|ml|min|mmol)\b)",
    re.I)
CMP = {"up to": "<=", "maximum": "<=", "maximum of": "<=", "max": "<=", "max.": "<=",
       "no more than": "<=", "not exceed": "<=", "not to exceed": "<=", "≤": "<=", "<=": "<=",
       "<": "<", "at least": ">=", "minimum": ">=", "minimum of": ">=", "≥": ">=", ">=": ">=",
       ">": ">"}
DAY_WORDS = re.compile(r"per day|/day|\bdaily\b|a day|/24 ?h|per 24 ?h|total daily|daily dose", re.I)
DOSE_WORDS = re.compile(r"per dose|single dose|\bbolus\b|each dose|as a single|\bonce\b|\btwice\b"
                        r"|\bo\.?d\.?(?![a-z])|\bb\.?i\.?d\b|\bt\.?i\.?d\b|\bq\.?d\b"
                        r"|every \d+ ?h|\bq\d+h\b|initial dose|second dose", re.I)

CONDITION_PATTERNS = (
    ("egfr", re.compile(rf"\b(?:e?GFR|CrCl|creatinine clearance)\s*(?P<op>≤|<=|<|≥|>=|>|of|below|above)?"
                        rf"\s*(?P<v>{NUM})", re.I)),
    ("ckd_stage", re.compile(r"\b(?:CKD\s*)?(?:stage\s*(?P<v>[1-5])|G(?P<g>[1-5][ab]?)\b)", re.I)),
    ("nyha", re.compile(r"\bNYHA\s*(?:class\s*)?(?P<v>IV|III|II|I)\b", re.I)),
    ("age", re.compile(rf"\b(?:aged?\s*(?P<op>≥|>=|>|<|≤)?\s*(?P<v>{NUM})|(?P<v2>{NUM})\s*years)", re.I)),
)
CONDITION_FLAGS = (
    ("loop_naive", re.compile(r"\bnaive\b|not (?:previously )?(?:receiving|taking|on) (?:a )?loop", re.I)),
    ("loop_exposed", re.compile(r"already (?:on|receiving|taking)|chronic(?:ally)? (?:oral )?(?:loop )?diuretic"
                                r"|home dose|previous oral dose|daily oral dose|oral maintenance", re.I)),
    ("paediatric", re.compile(r"\bp(?:a)?ediatric|\bchildren\b|\binfants?\b|\bneonat", re.I)),
    ("older_adults", re.compile(r"\belderly\b|older adults|geriatric", re.I)),
    ("acute_hf", re.compile(r"acute (?:decompensated )?heart failure|\bAHF\b|\bADHF\b|acute pulmonary (?:o)?edema", re.I)),
    ("chronic_hf", re.compile(r"chronic heart failure|\bHFrEF\b|\bHFpEF\b|\bCHF\b", re.I)),
    ("cirrhosis", re.compile(r"cirrho|\bascites\b", re.I)),
    ("aki", re.compile(r"\bAKI\b|acute kidney injury|acute renal failure", re.I)),
    ("ckd", re.compile(r"\bCKD\b|chronic kidney disease|renal impairment|renal insufficiency", re.I)),
    ("nephrotic", re.compile(r"nephrotic", re.I)),
    ("dialysis", re.compile(r"dialysis|\bhaemodialysis\b|\bhemodialysis\b", re.I)),
)

JUDGE_SYSTEM = """You compare two statements taken from clinical guidelines or drug labels about
dosing. Answer three separate questions, each strictly true or false:
1. same_question: do both statements address the same clinical decision (for example the
   initial intravenous dose, the maximum daily dose, the dose increment, the dose in renal
   impairment), regardless of the numbers they give?
2. same_population: do both apply to the same patient group (same condition, same renal
   function band, same prior loop-diuretic exposure, same age group)? A statement that names
   no group applies to the general adult population.
3. same_value: do both state the same dose amount on the same basis (per dose vs per day vs
   per kg vs multiple of another dose), ignoring formatting such as "20-40 mg" vs
   "20 to 40 mg"? If either states no dose amount, answer false.
Judge only from the text. Reply as JSON:
{"same_question": bool, "same_population": bool, "same_value": bool, "reason": "<one sentence>"}"""


@dataclass
class Mention:
    mention_id: str
    passage_id: str
    source_id: str
    kind: str
    section: str
    text: str
    masked: str
    value: tuple
    value_basis_sources: tuple
    condition: tuple


def normalise_text(text: str) -> str:
    text = DASH.sub("-", text)
    text = re.sub(rf"(?<=\d)\s+to\s+(?=\d)", "-", text)
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    text = re.sub(rf"({NUM})\s*(mg|g|mcg|µg)\s*(?:-|to)\s*({NUM})\s*\2\b", r"\1-\3 \2", text)
    text = text.replace("μ", "µ")
    return re.sub(r"\s+", " ", text)


def _unit(raw: str) -> tuple[str, Optional[str], float]:
    u = raw.lower().replace(" ", "")
    if u in ("times", "×", "-fold", "fold"):
        return "x", None, 1.0
    if u in ("g",):
        return "mg", None, 1000.0
    if u in ("mcg", "µg"):
        return "mg", None, 0.001
    if u == "mg/kg":
        return "mg/kg", None, 1.0
    if u == "g/kg":
        return "mg/kg", None, 1000.0
    if u.startswith("mg/kg/"):
        return "mg/kg", "day" if u.endswith(("day", "/d")) else "hour", 1.0
    if u.startswith("mg/"):
        return "mg", "day" if u.endswith(("day", "/d", "24h")) else "hour", 1.0
    if u.startswith("ml/"):
        return u, "hour", 1.0
    return u, None, 1.0


def basis_from_header(text: str, start: int) -> Optional[str]:
    cell = text.rfind("|", 0, start)
    head = text[cell + 1:start] if cell >= 0 else text[:start]
    if ":" not in head:
        return None
    header = head.rsplit(":", 1)[0]
    if DAY_WORDS.search(header):
        return "day"
    if re.search(r"\bdose\b", header, re.I):
        return None
    return None


def quantities(text: str, kind: str) -> list[tuple[tuple, str]]:
    norm = normalise_text(text)
    out = []
    for m in QUANTITY.finditer(norm):
        unit, basis, scale = _unit(m.group("unit"))
        lo = round(float(m.group("lo")) * scale, 4)
        hi = round(float(m.group("hi")) * scale, 4) if m.group("hi") else lo
        cmp = CMP.get((m.group("cmp") or "").lower().strip(), "=")
        source = "unit" if basis else None
        if basis is None and kind == "table":
            basis = basis_from_header(norm, m.start())
            source = "table_header" if basis else None
        if basis is None:
            window = norm[max(0, m.start() - 60):m.end() + 60]
            if DAY_WORDS.search(window):
                basis, source = "day", "sentence"
            elif DOSE_WORDS.search(window):
                basis, source = "dose", "sentence"
        if basis is None:
            basis, source = "unknown", "none"
        out.append(((unit, basis, cmp, lo, hi), source))
    return out


def condition_signature(text: str, section: str) -> tuple:
    scope = normalise_text(f"{section} || {text}")
    facts = set()
    for name, pattern in CONDITION_PATTERNS:
        for m in pattern.finditer(scope):
            groups = {k: v for k, v in m.groupdict().items() if v}
            value = groups.get("v") or groups.get("g") or groups.get("v2")
            if value is None:
                continue
            facts.add((name, groups.get("op", "") or "", value.upper()))
    for name, pattern in CONDITION_FLAGS:
        if pattern.search(scope):
            facts.add((name, "", ""))
    return tuple(sorted(facts))


def mask_doses(text: str) -> str:
    norm = normalise_text(text)
    return QUANTITY.sub(lambda m: f"{m.group('cmp') or ''} <NUM> {m.group('unit')}".strip(), norm)


def units_of(passage: dict[str, Any]) -> list[str]:
    if passage["kind"] == "table":
        lines = passage["text"].split("\n")
        head = lines[0] if lines else ""
        units = [f"{head} || {sentence}" if head else sentence
                 for line in lines[1:] for sentence in split_sentences(line)]
        return units or [passage["text"]]
    return split_sentences(passage["text"])


def build_mentions(passages: list[dict[str, Any]]) -> list[Mention]:
    mentions = []
    for p in passages:
        section = " > ".join(p.get("section_path") or [])
        for i, unit in enumerate(units_of(p)):
            found = quantities(unit, p["kind"])
            if not found:
                continue
            value = tuple(sorted({q for q, _ in found}))
            sources = tuple(sorted({s for _, s in found}))
            mentions.append(Mention(f"{p['passage_id']}#{i}", p["passage_id"], p["source_id"],
                                    p["kind"], section, unit, mask_doses(unit), value, sources,
                                    condition_signature(unit, section)))
    return mentions


def embed(texts: list[str]):
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(ENCODER)
    vectors = model.encode(texts, batch_size=64, normalize_embeddings=True,
                           show_progress_bar=False)
    return np.asarray(vectors, dtype="float32")


def sample_pairs(mentions: list[Mention], vectors) -> list[dict[str, Any]]:
    import numpy as np

    rng = random.Random(SEED)
    sims = vectors @ vectors.T
    cells: dict[tuple, list[tuple[int, int, float]]] = {}
    n = len(mentions)
    for i in range(n):
        row = sims[i]
        for j in np.nonzero(row[i + 1:] >= MIN_COSINE)[0] + i + 1:
            if mentions[i].text == mentions[j].text:
                continue
            cos = float(row[j])
            band = next(b for b in BANDS if b[0] <= cos < b[1])
            cross = mentions[i].source_id != mentions[j].source_id
            cells.setdefault((band, cross), []).append((i, j, cos))
    per_cell = N_PAIRS // len(cells) if cells else 0
    chosen = []
    leftovers = []
    for key in sorted(cells):
        pool = cells[key]
        rng.shuffle(pool)
        chosen += pool[:per_cell]
        leftovers += pool[per_cell:]
    rng.shuffle(leftovers)
    chosen += leftovers[:N_PAIRS - len(chosen)]
    pairs = []
    for k, (i, j, cos) in enumerate(chosen):
        a, b = mentions[i], mentions[j]
        pairs.append({"pair_id": k, "a": a.mention_id, "b": b.mention_id, "cosine": round(cos, 4),
                      "cross_source": a.source_id != b.source_id,
                      "band": next(f"{lo}-{min(hi, 1.0)}" for lo, hi in BANDS if lo <= cos < hi),
                      "same_value_rule": a.value == b.value,
                      "same_condition_rule": a.condition == b.condition,
                      "label_key": pair_key(a, b)})
    return pairs, {f"{k[0][0]}-{min(k[0][1], 1.0)}|{'cross' if k[1] else 'same'}": len(v)
                   for k, v in sorted(cells.items())}


def judge_prompt(a: Mention, b: Mention) -> str:
    return (f"Statement A (source: {a.source_id}; section: {a.section or '-'}):\n{a.text}\n\n"
            f"Statement B (source: {b.source_id}; section: {b.section or '-'}):\n{b.text}")


def pair_key(a: Mention, b: Mention) -> str:
    return hashlib.sha256(judge_prompt(a, b).encode("utf-8")).hexdigest()[:24]


def judge_pairs(pairs: list[dict[str, Any]], by_id: dict[str, Mention], path: Path,
                workers: int = 4) -> dict[str, dict[str, Any]]:
    from physioagent.config.loader import load_settings
    from physioagent.llm.client import LLMClient

    done: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            if row.get("label_key") and "error" not in row:
                done[row["label_key"]] = row
    todo = [p for p in pairs if p["label_key"] not in done]
    client = LLMClient.from_settings(load_settings())
    lock = threading.Lock()

    def one(pair: dict[str, Any]) -> dict[str, Any]:
        user = judge_prompt(by_id[pair["a"]], by_id[pair["b"]])
        meta: dict[str, Any] = {}
        last_error = None
        for _ in range(3):
            try:
                out = client.chat_json(JUDGE_SYSTEM, user, meta)
                return {"label_key": pair["label_key"], "model": meta.get("model"),
                        **{k: bool(out.get(k)) for k in
                           ("same_question", "same_population", "same_value")},
                        "reason": str(out.get("reason", ""))[:300]}
            except Exception as exc:
                last_error = exc
        return {"label_key": pair["label_key"], "error": str(last_error)[:300]}

    with ThreadPoolExecutor(max_workers=workers) as pool, path.open("a", encoding="utf-8",
                                                                     newline="\n") as handle:
        for future in as_completed([pool.submit(one, p) for p in todo]):
            row = future.result()
            with lock:
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                done[row["label_key"]] = row
    return done


def _f1(pred: list[bool], gold: list[bool]) -> dict[str, float]:
    tp = sum(p and g for p, g in zip(pred, gold))
    fp = sum(p and not g for p, g in zip(pred, gold))
    fn = sum(g and not p for p, g in zip(pred, gold))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def decide(system: str, pair: dict[str, Any], threshold: float) -> bool:
    if pair["cosine"] < threshold:
        return False
    if system in ("B", "C") and not pair["same_value_rule"]:
        return False
    if system == "C" and not pair["same_condition_rule"]:
        return False
    return True


def best_threshold(system: str, rows: list[dict[str, Any]]) -> float:
    candidates = sorted({r["cosine"] for r in rows} | {MIN_COSINE})
    return max(candidates, key=lambda t: (_f1([decide(system, r, t) for r in rows],
                                               [r["gold"] for r in rows])["f1"], -t))


def evaluate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rng = random.Random(SEED)
    order = rows[:]
    rng.shuffle(order)
    folds = [order[k::FOLDS] for k in range(FOLDS)]
    results: dict[str, Any] = {}
    for system in ("A", "B", "C"):
        preds: dict[int, bool] = {}
        thresholds = []
        for k in range(FOLDS):
            train = [r for f, fold in enumerate(folds) if f != k for r in fold]
            t = best_threshold(system, train)
            thresholds.append(round(t, 4))
            for r in folds[k]:
                preds[r["pair_id"]] = decide(system, r, t)
        pred = [preds[r["pair_id"]] for r in rows]
        gold = [r["gold"] for r in rows]
        point = _f1(pred, gold)
        boots = []
        for _ in range(BOOTSTRAP):
            idx = [rng.randrange(len(rows)) for _ in rows]
            boots.append(_f1([pred[i] for i in idx], [gold[i] for i in idx])["f1"])
        boots.sort()
        false_merges = [r for r, p in zip(rows, pred) if p and not r["gold"]]
        results[system] = {
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in point.items()},
            "f1_ci95": [round(boots[int(0.025 * BOOTSTRAP)], 4),
                        round(boots[int(0.975 * BOOTSTRAP)], 4)],
            "cv_thresholds": thresholds,
            "false_merges_by_cause": dict(Counter(
                "different value" if not r["llm"]["same_value"] else
                "different population" if not r["llm"]["same_population"] else
                "different question" for r in false_merges)),
        }
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="2-axis vs 3-axis grouping of guideline dose mentions")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-judge", action="store_true", help="stop after sampling pairs")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    passages = [json.loads(l) for l in PASSAGES.open(encoding="utf-8")]
    mentions = build_mentions(passages)
    by_id = {m.mention_id: m for m in mentions}
    with (OUT_DIR / "mentions.jsonl").open("w", encoding="utf-8", newline="\n") as h:
        for m in mentions:
            h.write(json.dumps(vars(m), ensure_ascii=False) + "\n")
    basis = Counter(b for m in mentions for b in m.value_basis_sources)
    print(f"mentions: {len(mentions)} from {len(passages)} passages; basis sources {dict(basis)}")
    print(f"mentions with any condition fact: "
          f"{sum(1 for m in mentions if m.condition)}/{len(mentions)}")

    vectors = embed([m.masked for m in mentions])
    pairs, cells = sample_pairs(mentions, vectors)
    print(f"candidate pairs by cell: {cells}")
    print(f"sampled pairs: {len(pairs)} {dict(Counter(p['band'] + ('|cross' if p['cross_source'] else '|same') for p in pairs))}")
    (OUT_DIR / "pairs.json").write_text(json.dumps(pairs, indent=1), encoding="utf-8",
                                        newline="\n")
    if args.no_judge:
        return 0

    labels = judge_pairs(pairs, by_id, OUT_DIR / "llm_labels.jsonl", args.workers)
    rows = []
    for p in pairs:
        lab = labels.get(p["label_key"])
        if not lab or "error" in lab:
            continue
        rows.append({**p, "llm": lab,
                     "gold": lab["same_question"] and lab["same_population"] and lab["same_value"]})
    errors = len(pairs) - len(rows)
    results = {"n_pairs": len(rows), "judge_errors": errors,
               "gold_positive": sum(r["gold"] for r in rows),
               "label_rates": {k: round(sum(r["llm"][k] for r in rows) / len(rows), 3)
                               for k in ("same_question", "same_population", "same_value")},
               "rule_vs_llm_agreement": {
                   "value": round(sum(r["same_value_rule"] == r["llm"]["same_value"] for r in rows)
                                  / len(rows), 3),
                   "condition": round(sum(r["same_condition_rule"] == r["llm"]["same_population"]
                                          for r in rows) / len(rows), 3)},
               "systems": evaluate(rows)}
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8",
                                          newline="\n")
    with (OUT_DIR / "pairs_for_review.csv").open("w", encoding="utf-8-sig", newline="") as h:
        writer = csv.writer(h)
        writer.writerow(["pair_id", "cosine", "source_a", "text_a", "source_b", "text_b",
                         "llm_same_question", "llm_same_population", "llm_same_value",
                         "llm_reason", "human_same_question", "human_same_population",
                         "human_same_value", "human_note"])
        for r in rows:
            a, b = by_id[r["a"]], by_id[r["b"]]
            writer.writerow([r["pair_id"], r["cosine"], a.source_id, a.text, b.source_id, b.text,
                             r["llm"]["same_question"], r["llm"]["same_population"],
                             r["llm"]["same_value"], r["llm"]["reason"], "", "", "", ""])
    print(json.dumps({k: v for k, v in results.items() if k != "systems"}, indent=1))
    for system, s in results["systems"].items():
        print(f"{system}: F1 {s['f1']:.3f} {s['f1_ci95']} P {s['precision']:.3f} "
              f"R {s['recall']:.3f} thresholds {s['cv_thresholds']} "
              f"false merges {s['false_merges_by_cause']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
