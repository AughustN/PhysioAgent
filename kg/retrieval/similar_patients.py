#!/usr/bin/env python
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Optional

KG_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA = KG_DIR / "data"

TOP_M = 5
RECENT_VISITS = 2
DEMOGRAPHIC_KEYS = ("age", "sex", "ckd_stage", "egfr")
TIE_NUMERIC = ("age", "egfr")


def _number(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def demographic_scales(pool: list[dict[str, Any]]) -> dict[str, float]:
    scales = {}
    for key in TIE_NUMERIC:
        values = [v for v in (_number(r["demographics"].get(key)) for r in pool)
                  if v is not None]
        sd = statistics.pstdev(values) if len(values) > 1 else 0.0
        scales[key] = sd if sd > 0 else 1.0
    return scales


def demographic_distance(a: dict[str, Any], b: dict[str, Any],
                         scales: dict[str, float]) -> float:
    sex_a, sex_b = a.get("sex"), b.get("sex")
    distance = 0.0 if (sex_a and sex_b and str(sex_a) == str(sex_b)) else 1.0
    for key in TIE_NUMERIC:
        x, y = _number(a.get(key)), _number(b.get(key))
        distance += 1.0 if x is None or y is None else abs(x - y) / scales[key]
    return distance


def recent_concepts(profile: dict[str, Any], n_visits: int = RECENT_VISITS) -> frozenset[str]:
    visits = []
    for key, visit in profile.items():
        if not key.startswith("visit_"):
            continue
        names = (visit.get("conditions", []) + visit.get("procedures", [])
                 + visit.get("drugs", []))
        if names:
            visits.append((int(key.split("_")[1]), names))
    visits.sort()
    return frozenset(name for _, names in visits[-n_visits:] for name in names)


def prior_visit_count(profile: dict[str, Any]) -> int:
    count = 0
    for key, visit in profile.items():
        if not key.startswith("visit_") or visit.get("is_index"):
            continue
        if visit.get("conditions") or visit.get("procedures") or visit.get("drugs"):
            count += 1
    return count


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def top_path(entry: Optional[dict[str, Any]]) -> Optional[tuple[str, dict[str, Any]]]:
    if not entry:
        return None
    from physioagent.eval.evidence import EvidenceProvider

    chosen = EvidenceProvider({"_": entry})._select(entry)
    if not chosen:
        return None
    return max(chosen, key=lambda row: (float(row[1].get(row[0], 0.0)),
                                        tuple(row[1].get("nodes") or [])))


def rank_neighbours(query: frozenset[str], pool: list[dict[str, Any]],
                    m: int = TOP_M, query_demographics: Optional[dict[str, Any]] = None,
                    scales: Optional[dict[str, float]] = None
                    ) -> list[tuple[float, dict[str, Any]]]:
    if query_demographics is not None and scales is None:
        scales = demographic_scales(pool)

    def key(item: tuple[float, dict[str, Any]]) -> tuple:
        score, row = item
        distance = (demographic_distance(query_demographics, row["demographics"], scales)
                    if query_demographics is not None else 0.0)
        return (-score, distance, row["dose_id"])

    scored = sorted(((jaccard(query, row["concepts"]), row) for row in pool), key=key)
    out: list[tuple[float, dict[str, Any]]] = []
    seen: set[int] = set()
    for score, row in scored:
        if row["subject_id"] in seen:
            continue
        seen.add(row["subject_id"])
        out.append((score, row))
        if len(out) == m:
            break
    return out


def build_pool(doses: list[dict[str, Any]], profiles: dict[str, Any],
               trajectories: dict[str, Any]) -> list[dict[str, Any]]:
    pool = []
    for record in doses:
        if record["split"] != "train":
            raise SystemExit(f"pool row {record['dose_id']} is in split "
                             f"{record['split']!r}; the pool must be train only -- its "
                             f"outcomes are printed into the prompt")
        profile = profiles.get(record["dose_id"])
        if profile is None:
            continue
        best = top_path(trajectories.get(record["dose_id"]))
        pool.append({
            "dose_id": record["dose_id"],
            "subject_id": int(record["subject_id"]),
            "concepts": recent_concepts(profile),
            "dose_mg": float(record["label"]["dose_mg"]),
            "demographics": {k: record["state"].get(k) for k in DEMOGRAPHIC_KEYS},
            "path": None if best is None else {"aspect": best[0], **best[1]},
        })
    return pool


def neighbours_json(ranked: list[tuple[float, dict[str, Any]]]) -> list[dict[str, Any]]:
    return [{"dose_id": row["dose_id"], "subject_id": row["subject_id"],
             "jaccard": round(score, 4), "dose_mg": row["dose_mg"],
             "demographics": row["demographics"], "path": row["path"]}
            for score, row in ranked]


class SimilarIndex:
    # Pool built once; neighbours ranked per row at call time, as TRACER's inference loop.

    def __init__(self, train: list[dict[str, Any]], pool_profiles: dict[str, Any],
                 pool_trajectories: dict[str, Any], query_profiles: dict[str, Any],
                 m: int = TOP_M) -> None:
        self.pool = build_pool(train, pool_profiles, pool_trajectories)
        self.scales = demographic_scales(self.pool)
        self.profiles = query_profiles
        self.m = m

    def lookup(self, record: dict[str, Any]) -> dict[str, Any]:
        profile = self.profiles.get(record["dose_id"])
        query_demo = {k: record["state"].get(k) for k in DEMOGRAPHIC_KEYS}
        ranked = rank_neighbours(recent_concepts(profile or {}), self.pool, self.m,
                                 query_demo, self.scales)
        return {"similar": neighbours_json(ranked),
                "query_prior_visits": prior_visit_count(profile or {}),
                "has_profile": profile is not None}

    def table(self, records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {r["dose_id"]: self.lookup(r) for r in records}


def _read(path: Path, what: str) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"{path} not found -- {what}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_index(doses: list[dict[str, Any]], split: str, data_dir: Path = DEFAULT_DATA,
               profiles: Optional[str] = None, pool_profiles: Optional[str] = None,
               pool_trajectories: Optional[str] = None, m: int = TOP_M) -> SimilarIndex:
    if split == "train":
        raise SystemExit("neighbours are found for dev/test rows; train is the pool")
    fetch = "scp it back from the cluster (kg_doses_bench.slurm, Bring back)"
    query = _read(Path(profiles or data_dir / f"patient_profiles_doses_{split}.json"), fetch)
    pool = _read(Path(pool_profiles or data_dir / "patient_profiles_doses_train.json"), fetch)
    traj = _read(Path(pool_trajectories or data_dir / "trajectories_train.json"), fetch)
    index = SimilarIndex([r for r in doses if r["split"] == "train"], pool, traj, query, m)
    if not index.pool:
        raise SystemExit("similar-patient pool is empty: no train row has a profile")
    return index


def summary(table: dict[str, dict[str, Any]]) -> dict[str, Any]:
    top = sorted(e["similar"][0]["jaccard"] for e in table.values() if e["similar"])

    def pct(q: float) -> float:
        return top[min(len(top) - 1, int(q * len(top)))] if top else float("nan")

    return {"rows": len(table),
            "rows_without_profile_ranked_on_demographics":
                sum(1 for e in table.values() if not e["has_profile"]),
            "no_earlier_admission": sum(1 for e in table.values()
                                        if e["query_prior_visits"] == 0),
            "top1_jaccard": {"p25": pct(.25), "p50": pct(.5), "p75": pct(.75)},
            "rows_with_zero_overlap_top1": sum(1 for s in top if s == 0.0)}
