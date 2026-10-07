#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from physioagent.utils.plausibility import implausible_mask
from physioagent.kg.scope.filter import DEFAULT_PROFILE, load_scope, profiles

KG_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA = KG_DIR / "data"
DEFAULT_COHORT = "mimic-exploration/output/gate_b_cohort_prior.csv"

EPISODE_GAP_H = 168.0

HORIZON_H = 48.0
SLOT_H = 12.0
N_SLOTS = int(HORIZON_H // SLOT_H)

SPLIT_SEED = 20260927
DEV_FRACTION = 0.15
TEST_FRACTION = 0.15
SPLIT_NAMES = ("train", "dev", "test")

STATE_COLS = [
    "age", "sex", "weight_kg", "height_cm", "bmi", "ckd_stage", "is_ckd",
    "egfr", "creatinine_mg_dl", "urine_prior24h_ml", "urine_prior24h_present",
    "predose_sbp", "predose_dbp", "predose_hr", "systolic_bp", "diastolic_bp",
    "has_chf", "has_diabetes", "has_htn", "aki_creat_safe",
    "care_unit", "admission_type",
]

HISTORY_COLS = [
    "prior_po_daily_mg", "prior_po_days_ago", "prior_po_source",
    "prior_iv_cum_mg", "loop_naive", "naive_observable",
]

PRIOR_STATE_COLS = ["egfr", "creatinine_mg_dl", "urine_prior24h_ml", "predose_sbp",
                    "ckd_stage", "care_unit"]

FORBIDDEN_AS_FEATURE = [
    "ground_truth_dose_mg", "next_dose_mg", "hours_to_next_dose", "n_doses_adm",
    "urine_post_ml", "urine_post_hours", "urine_post_rate", "urine_post_present",
    "response_usable", "nearest_candidate_mg", "in_candidate_set",
]


def _clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.generic,)):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if pd.isna(value) if np.isscalar(value) else False:
        return None
    return value


def load_events(path: Path, drop_implausible: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["starttime"])
    if "eligible" in df.columns:
        df = df[df["eligible"].astype(bool)]
    if drop_implausible:
        df = df[~implausible_mask(df)]
    return df.sort_values(["subject_id", "hadm_id", "starttime"]).reset_index(drop=True)


def mark_episodes(df: pd.DataFrame, gap_h: float = EPISODE_GAP_H) -> pd.DataFrame:
    out = df.copy()
    gap = out["hours_since_prev_dose"]
    out["episode_start"] = gap.isna() | (gap > gap_h)
    out["episode_index"] = out.groupby("hadm_id")["episode_start"].cumsum().astype(int)
    anchors = (out[out["episode_start"]]
               .set_index(["hadm_id", "episode_index"])["starttime"].rename("t0"))
    out = out.join(anchors, on=["hadm_id", "episode_index"])
    out["hours_from_t0"] = (out["starttime"] - out["t0"]).dt.total_seconds() / 3600.0
    return out


def split_balance(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name in SPLIT_NAMES:
        rows = [r for r in records if r["split"] == name]
        if not rows:
            continue
        summary: dict[str, Any] = {
            "episodes": len(rows),
            "patients": len({r["subject_id"] for r in rows}),
            "censored_rate": round(sum(r["censored"] for r in rows) / len(rows), 4),
            "continued_rate": round(sum(
                any(s["given"] for s in r["slots"][1:] if s["observed"])
                for r in rows) / len(rows), 4),
        }
        for index in range(1, N_SLOTS):
            observed = [r["slots"][index] for r in rows if r["slots"][index]["observed"]]
            summary[f"slot{index}_occupancy"] = round(
                sum(1 for s in observed if s["given"]) / max(len(observed), 1), 4)
        for col in ("age", "bmi", "predose_sbp", "egfr"):
            values = [r["state"].get(col) for r in rows]
            values = [v for v in values if isinstance(v, (int, float))]
            summary[f"median_{col}"] = round(float(np.median(values)), 1) if values else None
        summary["has_htn_rate"] = round(sum(
            bool(r["state"].get("has_htn")) for r in rows) / len(rows), 4)
        summary["with_prior_episode_rate"] = round(sum(
            bool(r.get("prior_episodes")) for r in rows) / len(rows), 4)
        stages = Counter(str(r["state"].get("ckd_stage")) for r in rows)
        summary["ckd_stage"] = {k: round(v / len(rows), 4)
                                for k, v in sorted(stages.items())}
        out[name] = summary
    return out


def _chronological(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(episodes, key=lambda e: (e["t0"], e["episode_id"]))


def _continued(episode: dict[str, Any]) -> bool:
    return any(slot["given"] for slot in episode["slots"][1:] if slot["observed"])


def _stratum(episodes: list[dict[str, Any]]) -> tuple[str, bool, bool]:
    ordered = _chronological(episodes)
    stage = next((e["state"].get("ckd_stage") for e in ordered
                  if e["state"].get("ckd_stage")), None)
    return (str(stage), _continued(ordered[-1]), len(ordered) > 1)


def assign_split(records: list[dict[str, Any]], seed: int = SPLIT_SEED,
                 dev_frac: float = DEV_FRACTION,
                 test_frac: float = TEST_FRACTION) -> dict[int, str]:
    by_subject: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_subject[int(record["subject_id"])].append(record)

    strata: dict[tuple[str, bool, bool], list[int]] = defaultdict(list)
    for subject, episodes in by_subject.items():
        strata[_stratum(episodes)].append(subject)

    fractions = {"train": 1.0 - dev_frac - test_frac, "dev": dev_frac, "test": test_frac}
    rng = random.Random(seed)
    out: dict[int, str] = {}
    for key in sorted(strata):
        subjects = sorted(strata[key])
        rng.shuffle(subjects)
        quota = {name: frac * len(subjects) for name, frac in fractions.items()}
        filled = {name: 0 for name in fractions}
        for subject in subjects:
            name = max(filled, key=lambda n: (quota[n] - filled[n], fractions[n]))
            out[subject] = name
            filled[name] += 1
    return out


def _prior_summary(prior: dict[str, Any], t0: pd.Timestamp,
                   hadm_id: int) -> dict[str, Any]:
    cutoff_h = (t0 - pd.Timestamp(prior["t0"])).total_seconds() / 3600.0
    totals: list[Optional[float]] = []
    counts: list[Optional[int]] = []
    rates: list[float] = []
    for slot in prior["slots"]:
        start = slot["hours"][0]
        if not slot["observed"] or start >= cutoff_h:
            totals.append(None)
            counts.append(None)
            continue
        kept = [d for d in slot["doses"] if d["hours_from_t0"] < cutoff_h]
        totals.append(float(sum(d["dose_mg"] for d in kept)))
        counts.append(len(kept))
        for dose in kept:
            window = dose.get("urine_post_hours")
            if (dose.get("response_usable") and dose.get("urine_post_rate") is not None
                    and window is not None and dose["hours_from_t0"] + window <= cutoff_h):
                rates.append(float(dose["urine_post_rate"]))
    known = [t for t in totals if t is not None]
    return {
        "episode_id": prior["episode_id"],
        "days_before": round(cutoff_h / 24.0, 1),
        "same_admission": int(prior["hadm_id"]) == int(hadm_id),
        "slot_total_mg": totals,
        "slot_n_doses": counts,
        "total_48h_mg": float(sum(known)),
        "n_doses_48h": int(sum(c for c in counts if c is not None)),
        "continued": any(t for t in totals[1:] if t),
        "median_urine_post_rate_ml_h": round(float(np.median(rates)), 1) if rates else None,
        "truncated_at_t0": cutoff_h < HORIZON_H,
        "state": {col: prior["state"].get(col) for col in PRIOR_STATE_COLS},
    }


def attach_prior_episodes(records: list[dict[str, Any]]) -> None:
    by_subject: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_subject[int(record["subject_id"])].append(record)
    for episodes in by_subject.values():
        ordered = _chronological(episodes)
        for index, record in enumerate(ordered):
            t0 = pd.Timestamp(record["t0"])
            record["prior_episodes"] = [
                _prior_summary(prior, t0, record["hadm_id"])
                for prior in ordered[:index] if pd.Timestamp(prior["t0"]) < t0
            ]


def select_eval_targets(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    last: dict[int, str] = {}
    for record in _chronological(records):
        last[int(record["subject_id"])] = record["episode_id"]
    kept = [r for r in records
            if r["split"] == "train" or last[int(r["subject_id"])] == r["episode_id"]]
    return kept, len(records) - len(kept)


def load_episode_records(path: Path) -> list[dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if any("prior_episodes" in r for r in records):
        raise SystemExit(f"{path} is already a one-episode-per-patient artifact; "
                         "re-split from the cohort CSV or from a full episode file")
    for record in records:
        record["split"] = ""
    return records


def load_discharge_times(mimic_root: Optional[Path]) -> Optional[pd.DataFrame]:
    if mimic_root is None:
        return None
    from physioagent.kg.ehr.build_visits import _table_path

    path = _table_path(Path(mimic_root), "hosp", "admissions")
    adm = pd.read_csv(path, usecols=["hadm_id", "dischtime", "deathtime"],
                      parse_dates=["dischtime", "deathtime"])
    adm["hadm_id"] = adm["hadm_id"].astype("int64")
    return adm.set_index("hadm_id")


def episode_end(dischtime: Any, deathtime: Any) -> Optional[pd.Timestamp]:
    if not pd.isna(deathtime):
        return deathtime
    if not pd.isna(dischtime):
        return dischtime
    return None


def _slot_of(hours: float) -> int:
    return min(int(hours // SLOT_H), N_SLOTS - 1)


def build_episode(rows: pd.DataFrame,
                  end_time: Optional[pd.Timestamp]) -> dict[str, Any]:
    anchor = rows.iloc[0]
    t0 = anchor["t0"]
    in_window = rows[rows["hours_from_t0"] < HORIZON_H]

    observed_h: Optional[float] = None
    if end_time is not None and not pd.isna(end_time):
        observed_h = max(0.0, (end_time - t0).total_seconds() / 3600.0)

    slots: list[dict[str, Any]] = []
    for index in range(N_SLOTS):
        start, stop = index * SLOT_H, (index + 1) * SLOT_H
        observed = observed_h is None or observed_h > start
        doses = in_window[(in_window["hours_from_t0"] >= start)
                          & (in_window["hours_from_t0"] < stop)]
        if not observed:
            slots.append({"slot": index, "hours": [start, stop], "observed": False,
                          "given": None, "n_doses": None, "total_mg": None, "doses": []})
            continue
        slots.append({
            "slot": index,
            "hours": [start, stop],
            "observed": True,
            "given": bool(len(doses)),
            "n_doses": int(len(doses)),
            "total_mg": float(doses["ground_truth_dose_mg"].sum()),
            "doses": [
                {"hours_from_t0": round(float(r["hours_from_t0"]), 2),
                 "dose_mg": float(r["ground_truth_dose_mg"]),
                 "dose_index": int(r["dose_index"]),
                 "urine_post_ml": _clean(r.get("urine_post_ml")),
                 "urine_post_hours": _clean(r.get("urine_post_hours")),
                 "urine_post_rate": _clean(r.get("urine_post_rate")),
                 "response_usable": bool(r.get("response_usable", False)),
                 "on_infusion": bool(r.get("on_infusion", False))}
                for _, r in doses.iterrows()
            ],
        })

    observed_slots = [s for s in slots if s["observed"]]
    return {
        "episode_id": f"{int(anchor['subject_id'])}_{int(anchor['hadm_id'])}"
                      f"_e{int(anchor['episode_index'])}",
        "subject_id": int(anchor["subject_id"]),
        "hadm_id": int(anchor["hadm_id"]),
        "episode_index": int(anchor["episode_index"]),
        "instance_id": str(anchor.get("instance_id", "")),
        "t0": t0.isoformat(),
        "split": "",
        "observed_hours": None if observed_h is None else round(min(observed_h, HORIZON_H), 2),
        "censored": bool(observed_h is not None and observed_h < HORIZON_H),
        "state": {col: _clean(anchor.get(col)) for col in STATE_COLS if col in anchor},
        "history": {col: _clean(anchor.get(col)) for col in HISTORY_COLS if col in anchor},
        "slots": slots,
        "labels": {
            "given": [s["given"] for s in slots[1:]],
            "total_mg": [s["total_mg"] for s in slots],
            "n_doses": [s["n_doses"] for s in slots],
            "total_48h_mg": float(sum(s["total_mg"] for s in observed_slots)),
            "n_doses_48h": int(sum(s["n_doses"] for s in observed_slots)),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", default=DEFAULT_COHORT,
                    help="event-grain cohort CSV (one row per dose)")
    ap.add_argument("--out-dir", default=str(DEFAULT_DATA))
    ap.add_argument("--profile", default=DEFAULT_PROFILE, choices=profiles())
    ap.add_argument("--mimic-root", help="optional; enables censoring from admissions")
    ap.add_argument("--gap-hours", type=float, default=EPISODE_GAP_H,
                    help="a gap this long or longer starts a new episode")
    ap.add_argument("--keep-implausible", action="store_true",
                    help="skip the plausibility filter eval.cohort_row applies")
    ap.add_argument("--split-seed", type=int, default=SPLIT_SEED,
                    help="changing this reshuffles every split and invalidates every "
                         "number already measured against them")
    ap.add_argument("--dev-frac", type=float, default=DEV_FRACTION)
    ap.add_argument("--test-frac", type=float, default=TEST_FRACTION)
    ap.add_argument("--from-sequences",
                    help="re-split the episodes of an earlier, UNFILTERED sequences.json "
                         "instead of rebuilding them from --cohort; for machines without "
                         "MIMIC, since censoring needs --mimic-root")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_sequences:
        source = Path(args.from_sequences)
        records = load_episode_records(source)
        cohort_path = source
        dose_events = sum(len(s["doses"]) for r in records for s in r["slots"])
        censoring_known = any(r["observed_hours"] is not None for r in records)
        print(f"{len(records):,} episodes re-split from {source}")
    else:
        cohort_path = Path(args.cohort)
        if not cohort_path.exists():
            raise SystemExit(f"{cohort_path} not found")
        events = load_events(cohort_path, drop_implausible=not args.keep_implausible)
        events = mark_episodes(events, args.gap_hours)
        discharge = load_discharge_times(Path(args.mimic_root) if args.mimic_root else None)
        dose_events = int(len(events))
        censoring_known = discharge is not None

        print(f"{len(events):,} dose events / {events['hadm_id'].nunique():,} admissions "
              f"/ {events['subject_id'].nunique():,} patients")
        if discharge is None:
            print("no --mimic-root: observed_hours is null and late empty slots are "
                  "UNKNOWN, not zero")

        records = []
        for (_, _), rows in events.groupby(["hadm_id", "episode_index"], sort=True):
            anchor = rows.iloc[0]
            end_time = None
            if discharge is not None:
                hadm = int(anchor["hadm_id"])
                if hadm in discharge.index:
                    row = discharge.loc[hadm]
                    end_time = episode_end(row["dischtime"], row["deathtime"])
            records.append(build_episode(rows, end_time))

    all_episodes = len(records)
    attach_prior_episodes(records)
    splits = assign_split(records, args.split_seed, args.dev_frac, args.test_frac)
    for record in records:
        record["split"] = splits[int(record["subject_id"])]
    records, dropped_as_history = select_eval_targets(records)

    records.sort(key=lambda r: r["episode_id"])
    (out_dir / "sequences.json").write_text(json.dumps(records, indent=1), encoding="utf-8")

    occupancy = []
    for index in range(N_SLOTS):
        observed = [r["slots"][index] for r in records if r["slots"][index]["observed"]]
        occupancy.append({
            "slot": index,
            "observed_episodes": len(observed),
            "occupied": sum(1 for s in observed if s["given"]),
            "occupancy_rate": round(sum(1 for s in observed if s["given"])
                                    / max(len(observed), 1), 4),
            "median_total_mg_when_given": float(np.median(
                [s["total_mg"] for s in observed if s["given"]] or [0.0])),
        })
    filled = Counter(sum(1 for s in r["slots"] if s["observed"] and s["given"])
                     for r in records)
    gaps = []
    for record in records:
        hours = sorted(d["hours_from_t0"] for s in record["slots"] for d in s["doses"])
        gaps.extend(round(b - a, 2) for a, b in zip(hours, hours[1:]))
    inter_dose = {"n_gaps": len(gaps)}
    if gaps:
        inter_dose.update({f"p{p}_hours": round(float(np.percentile(gaps, p)), 2)
                           for p in (5, 25, 50, 75, 95)})
    doses_per_slot = Counter(s["n_doses"] for r in records for s in r["slots"]
                             if s["observed"] and s["given"])

    def _rate(rows: list[dict[str, Any]]) -> Optional[float]:
        return round(sum(_continued(r) for r in rows) / len(rows), 4) if rows else None

    train_rows = [r for r in records if r["split"] == "train"]
    after_continued = [r for r in train_rows
                       if r["prior_episodes"] and r["prior_episodes"][-1]["continued"]]
    after_stopped = [r for r in train_rows
                     if r["prior_episodes"] and not r["prior_episodes"][-1]["continued"]]
    first_course = [r for r in train_rows if not r["prior_episodes"]]
    prior_signal = {
        "continued_rate_after_a_continued_course": _rate(after_continued),
        "n_after_a_continued_course": len(after_continued),
        "continued_rate_after_a_stopped_course": _rate(after_stopped),
        "n_after_a_stopped_course": len(after_stopped),
        "continued_rate_on_a_first_course": _rate(first_course),
        "n_first_course": len(first_course),
    }

    report = {
        "cohort": str(cohort_path),
        "profile": args.profile,
        "episodes_before_selection": all_episodes,
        "episodes_dropped_as_history": dropped_as_history,
        "episodes": len(records),
        "episodes_by_split": dict(Counter(r["split"] for r in records)),
        "patients_by_split": {
            name: len({r["subject_id"] for r in records if r["split"] == name})
            for name in SPLIT_NAMES
        },
        "split_seed": args.split_seed,
        "split_fractions": {"train": round(1.0 - args.dev_frac - args.test_frac, 4),
                            "dev": args.dev_frac, "test": args.test_frac},
        "split_balance": split_balance(records),
        "prior_episodes": {
            "scored_episodes_with_prior_by_split": {
                name: sum(1 for r in records if r["split"] == name and r["prior_episodes"])
                for name in SPLIT_NAMES
            },
            "prior_episodes_truncated_at_t0": sum(
                1 for r in records for p in r["prior_episodes"] if p["truncated_at_t0"]),
            "train_continuation_given_previous_course": prior_signal,
        },
        "dose_events": dose_events,
        "gap_hours": args.gap_hours,
        "horizon_hours": HORIZON_H,
        "slot_hours": SLOT_H,
        "slot_occupancy": occupancy,
        "occupied_slots_per_episode": {str(k): v for k, v in sorted(filled.items())},
        "inter_dose_gap_hours": inter_dose,
        "doses_per_occupied_slot": {str(k): v for k, v in sorted(doses_per_slot.items())},
        "censored_episodes": sum(1 for r in records if r["censored"]),
        "episodes_without_observed_slots": sum(
            1 for r in records if not any(s["observed"] for s in r["slots"])),
        "censoring_known": censoring_known,
        "forbidden_as_feature": FORBIDDEN_AS_FEATURE,
    }
    (out_dir / "sequences_report.json").write_text(json.dumps(report, indent=2),
                                                   encoding="utf-8")

    sizes = " / ".join(f"{report['episodes_by_split'].get(n, 0):,} {n}"
                       for n in SPLIT_NAMES)
    print(f"\n{len(records):,} episodes ({sizes}, patient-disjoint, "
          f"seed {args.split_seed}); {dropped_as_history:,} dev/test episodes kept "
          f"only as prior_episodes")
    for name, row in report["split_balance"].items():
        print(f"  {name:5s} {row['episodes']:5,} ep / {row['patients']:5,} pt  "
              f"continued {row['continued_rate']:6.1%}  "
              f"slot1 {row['slot1_occupancy']:6.1%}  "
              f"prior {row['with_prior_episode_rate']:5.1%}  "
              f"age {row['median_age']:.0f}  sbp {row['median_predose_sbp']:.0f}  "
              f"htn {row['has_htn_rate']:5.1%}")
    signal = prior_signal
    print(f"  train continuation: {signal['continued_rate_after_a_continued_course']} after "
          f"a continued course (n={signal['n_after_a_continued_course']}), "
          f"{signal['continued_rate_after_a_stopped_course']} after a stopped one "
          f"(n={signal['n_after_a_stopped_course']}), "
          f"{signal['continued_rate_on_a_first_course']} on a first course "
          f"(n={signal['n_first_course']})")
    for slot in occupancy:
        print(f"  slot {slot['slot']} ({int(slot['slot'] * SLOT_H):2d}-"
              f"{int((slot['slot'] + 1) * SLOT_H):2d}h): "
              f"{slot['occupancy_rate']:6.1%} occupied, "
              f"median {slot['median_total_mg_when_given']:.0f} mg when given")
    if report["censored_episodes"]:
        print(f"  {report['censored_episodes']:,} episodes censored before 48 h")
    print(f"\nsequences -> {out_dir / 'sequences.json'}")
    print(f"report    -> {out_dir / 'sequences_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
