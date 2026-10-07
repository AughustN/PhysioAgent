#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

KG_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA = KG_DIR / "data"
DEFAULT_COHORT = "mimic-exploration/output/usable_cohort_full.csv"

DEMOGRAPHIC_COLS = ("age", "sex", "weight_kg", "height_cm", "egfr", "ckd_stage")

FORBIDDEN_COLS = ("ground_truth_dose_mg", "nearest_candidate_mg", "in_candidate_set")

CHRONIC_CONDITIONS = frozenset({
    "chronic kidney disease",
    "diabetes mellitus without complication",
    "diabetes mellitus with complications",
    "essential hypertension",
    "hypertension with complications and secondary hypertension",
    "congestive heart failure; nonhypertensive",
})


class ProfileError(RuntimeError):
    pass


@dataclass
class Visit:
    visit_id: str
    admittime: str
    conditions: list[str] = field(default_factory=list)
    procedures: list[str] = field(default_factory=list)
    drugs: list[str] = field(default_factory=list)
    is_index: bool = False
    concept_times: dict[str, str] = field(default_factory=dict)

    def concepts(self) -> list[str]:
        return self.conditions + self.procedures + self.drugs

    def to_json(self) -> dict[str, Any]:
        return {"visit_id": self.visit_id, "admittime": self.admittime,
                "conditions": self.conditions, "procedures": self.procedures,
                "drugs": self.drugs, "is_index": self.is_index,
                "concept_times": self.concept_times}


@dataclass
class Profile:
    instance_id: str
    subject_id: str
    label: Any
    index_t0: str
    demographics: dict[str, Any]
    visits: list[Visit]

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "label": self.label,
            "subject_id": self.subject_id,
            "index_t0": self.index_t0,
            "demographics": self.demographics,
            "n_visits": len(self.visits),
            "n_prior_visits": sum(1 for v in self.visits if not v.is_index),
        }
        for number, visit in enumerate(self.visits, 1):
            out[f"visit_{number}"] = visit.to_json()
        return out


def load_cohort(path: Path, sample: Optional[int],
                seed: Optional[int]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ProfileError(f"no rows in {path}")
    if sample and sample < len(rows):
        rng = random.Random(seed)
        rows = rng.sample(rows, sample)
    return rows


def load_visit_concepts(path: Path, subjects: Optional[set[str]] = None
                        ) -> dict[str, list[dict[str, Any]]]:
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            subject = str(record["subject_id"])
            if subjects is not None and subject not in subjects:
                continue
            by_subject[subject].append(record)
    return dict(by_subject)


def index_visit_concepts(record: dict[str, Any], t0: str, prior: list[Visit],
                         policy: str) -> Visit:
    day_of_t0 = t0[:10]
    kept_procedures: list[str] = []
    kept_drugs: list[str] = []
    kept_times: dict[str, str] = {}
    for names, times_key, target in (
            (record.get("procedures", []), "procedure_times", kept_procedures),
            (record.get("drugs", []), "drug_times", kept_drugs)):
        times = record.get(times_key) or {}
        for name in names:
            stamp = times.get(name)
            if not stamp:
                continue
            cutoff = day_of_t0 if len(stamp) <= 10 else t0
            if stamp < cutoff:
                target.append(name)
                kept_times[name] = stamp

    conditions: list[str] = []
    if policy == "chronic-only":
        seen_before = {name for visit in prior for name in visit.conditions}
        conditions = [name for name in record.get("conditions", [])
                      if name in CHRONIC_CONDITIONS or name in seen_before]

    return Visit(visit_id=str(record["hadm_id"]), admittime=record.get("admittime", ""),
                 conditions=sorted(conditions), procedures=sorted(kept_procedures),
                 drugs=sorted(kept_drugs), is_index=True,
                 concept_times=dict(sorted(kept_times.items())))


def prior_concept_times(record: dict[str, Any]) -> dict[str, str]:
    names = set(record.get("procedures", [])) | set(record.get("drugs", []))
    times: dict[str, str] = {}
    for key in ("procedure_times", "drug_times"):
        for name, stamp in (record.get(key) or {}).items():
            if name in names and stamp and (name not in times or stamp < times[name]):
                times[name] = stamp
    return dict(sorted(times.items()))


def build_profile(row: dict[str, str], admissions: list[dict[str, Any]],
                  policy: str) -> Optional[Profile]:
    t0 = row["starttime"]
    index_hadm = str(row["hadm_id"])

    index_record = next((a for a in admissions if str(a["hadm_id"]) == index_hadm), None)
    prior_records = [a for a in admissions
                     if str(a["hadm_id"]) != index_hadm
                     and a.get("admittime") and a["admittime"] < t0]
    prior_records.sort(key=lambda a: a["admittime"])

    visits = [Visit(visit_id=str(a["hadm_id"]), admittime=a.get("admittime", ""),
                    conditions=list(a.get("conditions", [])),
                    procedures=list(a.get("procedures", [])),
                    drugs=list(a.get("drugs", [])),
                    concept_times=prior_concept_times(a))
              for a in prior_records]

    if index_record is not None:
        visits.append(index_visit_concepts(index_record, t0, visits, policy))

    if not any(visit.concepts() for visit in visits):
        return None

    demographics = {key: row[key] for key in DEMOGRAPHIC_COLS if row.get(key)}
    label: Any = row.get("label")
    if label is None:
        label = float(row["ground_truth_dose_mg"])
    return Profile(instance_id=row["instance_id"], subject_id=row["subject_id"],
                   label=label, index_t0=t0,
                   demographics=demographics, visits=visits)


def rows_from_episodes(path: Path, split: Optional[str] = None) -> list[dict[str, Any]]:
    episodes = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        if split and episode["split"] != split:
            continue
        state = episode.get("state", {})
        row: dict[str, Any] = {
            "instance_id": episode["episode_id"],
            "subject_id": str(episode["subject_id"]),
            "hadm_id": str(episode["hadm_id"]),
            "starttime": episode["t0"].replace("T", " "),
            "label": episode["labels"]["total_mg"],
        }
        for column in DEMOGRAPHIC_COLS:
            if state.get(column) is not None:
                row[column] = state[column]
        rows.append(row)
    return rows


def rows_from_doses(path: Path, split: Optional[str] = None) -> list[dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for record in records:
        if split and record["split"] != split:
            continue
        state = record.get("state", {})
        row: dict[str, Any] = {
            "instance_id": record["dose_id"],
            "subject_id": str(record["subject_id"]),
            "hadm_id": str(record["hadm_id"]),
            "starttime": record["t0"].replace("T", " "),
            "label": record["label"]["dose_mg"],
        }
        for column in DEMOGRAPHIC_COLS:
            if state.get(column) is not None:
                row[column] = state[column]
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", default=DEFAULT_COHORT)
    ap.add_argument("--episodes", metavar="SEQUENCES_JSON",
                    help="build one profile per EPISODE from sequences.json instead of "
                         "one per cohort row, cutting the history at that episode's own "
                         "t0 and keying on episode_id. The sequence task scores episodes, "
                         "and 7.8% of test episodes start later than their admission's "
                         "first dose; --cohort would hand those the wrong history. "
                         "--sample and --seed are ignored in this mode: the split is the "
                         "sample.")
    ap.add_argument("--doses", metavar="DOSES_JSON",
                    help="build one profile per scored admission from doses.json "
                         "(build_doses), keyed on dose_id. --sample and --seed are "
                         "ignored: the split is the sample.")
    ap.add_argument("--split", default="test",
                    help="which split to build with --episodes or --doses (default: test)")
    ap.add_argument("--visits", default=str(DEFAULT_DATA / "visits_all.jsonl"),
                    help="S0a's per-admission concept sets WITH ids. Not "
                         "all_visit_concepts.json, which drops them")
    ap.add_argument("--out", default=str(DEFAULT_DATA / "patient_profiles.json"))
    ap.add_argument("--sample", type=int, default=400,
                    help="rows to build, matching the eval sample (default 400)")
    ap.add_argument("--seed", type=int, default=20260901,
                    help="MUST match the seed the eval arms are sampled with, or the "
                         "profiles describe different patients than the scores do")
    ap.add_argument("--index-dx", choices=("none", "chronic-only"),
                    default="chronic-only",
                    help="what the index admission's ICD codes may contribute; see "
                         "rule B in the module docstring")
    ap.add_argument("--require-prior", action="store_true",
                    help="keep only instances with >=1 earlier admission, i.e. the "
                         "paper's footnote-4 cohort. Measured on the full cohort this "
                         "is 26.2%% of instances -- the flag reports what it drops")
    args = ap.parse_args()

    if args.doses:
        doses_path = Path(args.doses)
        if not doses_path.exists():
            raise SystemExit(f"{doses_path} not found -- run "
                             "physioagent.kg.ehr.build_doses")
        rows = rows_from_doses(doses_path, args.split)
        print(f"doses      : {len(rows)} in split {args.split!r} from {doses_path.name}")
    elif args.episodes:
        episodes_path = Path(args.episodes)
        if not episodes_path.exists():
            raise SystemExit(f"{episodes_path} not found -- run "
                             "physioagent.kg.ehr.build_sequences")
        rows = rows_from_episodes(episodes_path, args.split)
        print(f"episodes   : {len(rows)} in split {args.split!r} from "
              f"{episodes_path.name}")
    else:
        cohort_path = Path(args.cohort)
        if not cohort_path.is_absolute():
            cohort_path = KG_DIR.parents[1] / args.cohort
        rows = load_cohort(cohort_path, args.sample, args.seed)
        print(f"cohort     : {len(rows)} rows from {cohort_path.name}")

    for column in FORBIDDEN_COLS:
        if column in DEMOGRAPHIC_COLS:
            raise ProfileError(f"{column} is an outcome, not a demographic")

    visits_path = Path(args.visits)
    if not visits_path.exists():
        print(f"\nmissing: {visits_path}", file=sys.stderr)
        print("This file is built by S0a on the compute node. Either run this script "
              "there, or bring back a cohort-filtered copy -- not the full 191 MB, "
              "which is patient-derived data.", file=sys.stderr)
        return 2

    subjects = {row["subject_id"] for row in rows}
    visits_by_subject = load_visit_concepts(visits_path, subjects)
    print(f"visits     : {sum(len(v) for v in visits_by_subject.values())} admissions "
          f"for {len(visits_by_subject)}/{len(subjects)} sampled patients")
    profiles: dict[str, Any] = {}
    kept = dropped_no_history = 0
    for row in rows:
        profile = build_profile(row, visits_by_subject.get(row["subject_id"], []),
                                args.index_dx)
        if profile is None:
            continue
        if args.require_prior and all(v.is_index for v in profile.visits):
            dropped_no_history += 1
            continue
        profiles[profile.instance_id] = profile.to_json()
        kept += 1

    out_path = Path(args.out)
    out_path.write_text(json.dumps(profiles, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"\n{kept} profiles written")
    if args.require_prior:
        print(f"  dropped for having no prior admission: {dropped_no_history}")
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
