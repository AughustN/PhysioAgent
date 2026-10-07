#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from physioagent.kg.ehr.build_sequences import (
    DEFAULT_COHORT,
    DEFAULT_DATA,
    FORBIDDEN_AS_FEATURE,
    HISTORY_COLS,
    PRIOR_STATE_COLS,
    STATE_COLS,
    _clean,
    load_events,
)

SPLIT_SEED = 20260928
DEV_FRACTION = 0.15
TEST_FRACTION = 0.15
SPLIT_NAMES = ("train", "dev", "test")

DOSE_BANDS = ((20.0, "le20"), (40.0, "40"), (float("inf"), "ge60"))


def dose_band(mg: float) -> str:
    return next(name for edge, name in DOSE_BANDS if mg <= edge)


def _prior_admission(rows: pd.DataFrame, t0: pd.Timestamp) -> Optional[dict[str, Any]]:
    before = rows[rows["starttime"] < t0]
    if before.empty:
        return None
    first = before.iloc[0]
    doses = before["ground_truth_dose_mg"].astype(float)
    rates: list[float] = []
    for _, dose in before.iterrows():
        window = dose.get("urine_post_hours")
        rate = dose.get("urine_post_rate")
        if (bool(dose.get("response_usable")) and not pd.isna(rate) and not pd.isna(window)
                and dose["starttime"] + pd.Timedelta(hours=float(window)) <= t0):
            rates.append(float(rate))
    return {
        "hadm_id": int(first["hadm_id"]),
        "days_before": round((t0 - first["starttime"]).total_seconds() / 86400.0, 1),
        "first_dose_mg": float(doses.iloc[0]),
        "n_doses": int(len(doses)),
        "total_mg": float(doses.sum()),
        "max_dose_mg": float(doses.max()),
        "median_urine_post_rate_ml_h": round(float(np.median(rates)), 1) if rates else None,
        "truncated_at_t0": bool(len(before) < len(rows)),
        "state": {col: _clean(first.get(col)) for col in PRIOR_STATE_COLS},
    }


def build_records(events: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for subject, patient in events.groupby("subject_id", sort=True):
        by_adm = {hadm: rows.sort_values("starttime", kind="mergesort")
                  for hadm, rows in patient.groupby("hadm_id")}
        anchors = sorted((rows.iloc[0] for rows in by_adm.values()
                          if int(rows.iloc[0]["dose_index"]) == 1),
                         key=lambda r: (r["starttime"], int(r["hadm_id"])))
        for anchor in anchors:
            t0 = anchor["starttime"]
            prior = []
            for hadm, rows in by_adm.items():
                if hadm == anchor["hadm_id"] or rows.iloc[0]["starttime"] >= t0:
                    continue
                summary = _prior_admission(rows, t0)
                if summary is not None:
                    prior.append(summary)
            prior.sort(key=lambda p: -p["days_before"])
            records.append({
                "dose_id": f"{int(subject)}_{int(anchor['hadm_id'])}",
                "subject_id": int(subject),
                "hadm_id": int(anchor["hadm_id"]),
                "instance_id": str(anchor.get("instance_id", "")),
                "t0": t0.isoformat(),
                "split": "",
                "state": {col: _clean(anchor.get(col)) for col in STATE_COLS
                          if col in anchor},
                "history": {col: _clean(anchor.get(col)) for col in HISTORY_COLS
                            if col in anchor},
                "prior_admissions": prior,
                "label": {"dose_mg": float(anchor["ground_truth_dose_mg"])},
            })
    return records


def _last(records: list[dict[str, Any]]) -> dict[str, Any]:
    return max(records, key=lambda r: (r["t0"], r["hadm_id"]))


def _stratum(records: list[dict[str, Any]]) -> tuple[str, bool, str]:
    last = _last(records)
    return (str(last["state"].get("ckd_stage")), bool(last["prior_admissions"]),
            dose_band(last["label"]["dose_mg"]))


def assign_split(records: list[dict[str, Any]], seed: int = SPLIT_SEED,
                 dev_frac: float = DEV_FRACTION,
                 test_frac: float = TEST_FRACTION) -> dict[int, str]:
    by_subject: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_subject[record["subject_id"]].append(record)
    strata: dict[tuple[str, bool, str], list[int]] = defaultdict(list)
    for subject, rows in by_subject.items():
        strata[_stratum(rows)].append(subject)

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


def select_eval_targets(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    by_subject: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_subject[record["subject_id"]].append(record)
    last = {subject: _last(rows)["dose_id"] for subject, rows in by_subject.items()}
    kept = [r for r in records
            if r["split"] == "train" or last[r["subject_id"]] == r["dose_id"]]
    return kept, len(records) - len(kept)


def split_balance(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name in SPLIT_NAMES:
        rows = [r for r in records if r["split"] == name]
        if not rows:
            continue
        doses = np.array([r["label"]["dose_mg"] for r in rows])
        summary: dict[str, Any] = {
            "samples": len(rows),
            "patients": len({r["subject_id"] for r in rows}),
            "dose_median": float(np.median(doses)),
            "dose_mean": round(float(doses.mean()), 1),
            "dose_band": {band: round(float(np.mean([dose_band(d) == band for d in doses])), 4)
                          for _, band in DOSE_BANDS},
            "with_prior_admission_rate": round(
                sum(bool(r["prior_admissions"]) for r in rows) / len(rows), 4),
        }
        for col in ("age", "bmi", "predose_sbp", "egfr"):
            values = [r["state"].get(col) for r in rows]
            values = [v for v in values if isinstance(v, (int, float))]
            summary[f"median_{col}"] = round(float(np.median(values)), 1) if values else None
        for col in ("has_htn", "has_chf"):
            summary[f"{col}_rate"] = round(
                sum(bool(r["state"].get(col)) for r in rows) / len(rows), 4)
        summary["loop_naive_rate"] = round(
            sum(bool(r["history"].get("loop_naive")) for r in rows) / len(rows), 4)
        stages = Counter(str(r["state"].get("ckd_stage")) for r in rows)
        summary["ckd_stage"] = {k: round(v / len(rows), 4) for k, v in sorted(stages.items())}
        out[name] = summary
    return out


def floors(records: list[dict[str, Any]]) -> dict[str, Any]:
    train = np.array([r["label"]["dose_mg"] for r in records if r["split"] == "train"])
    median = float(np.median(train))
    out: dict[str, Any] = {"train_median_mg": median}
    for name in SPLIT_NAMES:
        rows = [r for r in records if r["split"] == name]
        if not rows:
            continue
        y = np.array([r["label"]["dose_mg"] for r in rows])
        entry: dict[str, Any] = {"n": len(rows),
                                 "mae_train_median": round(float(np.abs(y - median).mean()), 2)}
        with_prior = [r for r in rows if r["prior_admissions"]]
        if with_prior:
            yp = np.array([r["label"]["dose_mg"] for r in with_prior])
            copied = np.array([r["prior_admissions"][-1]["first_dose_mg"] for r in with_prior])
            entry["with_prior_n"] = len(with_prior)
            entry["with_prior_mae_copy_last_admission"] = round(
                float(np.abs(yp - copied).mean()), 2)
            entry["with_prior_mae_train_median"] = round(
                float(np.abs(yp - median).mean()), 2)
            entry["with_prior_exact_copy_rate"] = round(float(np.mean(yp == copied)), 4)
        out[name] = entry
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", default=DEFAULT_COHORT,
                    help="event-grain cohort CSV (one row per dose)")
    ap.add_argument("--out-dir", default=str(DEFAULT_DATA))
    ap.add_argument("--keep-implausible", action="store_true",
                    help="skip the plausibility filter eval.cohort_row applies")
    ap.add_argument("--split-seed", type=int, default=SPLIT_SEED,
                    help="changing this reshuffles every split and invalidates every "
                         "number already measured against them")
    ap.add_argument("--dev-frac", type=float, default=DEV_FRACTION)
    ap.add_argument("--test-frac", type=float, default=TEST_FRACTION)
    args = ap.parse_args()

    cohort_path = Path(args.cohort)
    if not cohort_path.exists():
        raise SystemExit(f"{cohort_path} not found")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    events = load_events(cohort_path, drop_implausible=not args.keep_implausible)
    print(f"{len(events):,} dose events / {events['hadm_id'].nunique():,} admissions "
          f"/ {events['subject_id'].nunique():,} patients")

    records = build_records(events)
    all_admissions = len(records)
    splits = assign_split(records, args.split_seed, args.dev_frac, args.test_frac)
    for record in records:
        record["split"] = splits[record["subject_id"]]
    records, dropped_as_history = select_eval_targets(records)
    records.sort(key=lambda r: r["dose_id"])
    (out_dir / "doses.json").write_text(json.dumps(records, indent=1), encoding="utf-8")

    subjects = {name: {r["subject_id"] for r in records if r["split"] == name}
                for name in SPLIT_NAMES}
    report = {
        "cohort": str(cohort_path),
        "dose_events": int(len(events)),
        "admissions_with_index_dose": all_admissions,
        "admissions_dropped_as_history": dropped_as_history,
        "samples": len(records),
        "samples_by_split": {n: sum(r["split"] == n for r in records) for n in SPLIT_NAMES},
        "patients_by_split": {n: len(s) for n, s in subjects.items()},
        "shared_patients": {"train_dev": len(subjects["train"] & subjects["dev"]),
                            "train_test": len(subjects["train"] & subjects["test"]),
                            "dev_test": len(subjects["dev"] & subjects["test"])},
        "split_seed": args.split_seed,
        "split_fractions": {"train": round(1.0 - args.dev_frac - args.test_frac, 4),
                            "dev": args.dev_frac, "test": args.test_frac},
        "split_balance": split_balance(records),
        "floors": floors(records),
        "prior_admissions_truncated_at_t0": sum(
            1 for r in records for p in r["prior_admissions"] if p["truncated_at_t0"]),
        "forbidden_as_feature": FORBIDDEN_AS_FEATURE,
    }
    (out_dir / "doses_report.json").write_text(json.dumps(report, indent=2),
                                               encoding="utf-8")

    sizes = " / ".join(f"{report['samples_by_split'][n]:,} {n}" for n in SPLIT_NAMES)
    print(f"\n{len(records):,} samples ({sizes}, seed {args.split_seed}); "
          f"{dropped_as_history:,} dev/test admissions kept only as prior_admissions")
    print(f"  shared patients: {report['shared_patients']}")
    for name, row in report["split_balance"].items():
        print(f"  {name:5s} {row['samples']:5,} / {row['patients']:5,} pt  "
              f"dose med {row['dose_median']:.0f} mean {row['dose_mean']:.1f}  "
              f"prior {row['with_prior_admission_rate']:5.1%}  "
              f"egfr {row['median_egfr']}  age {row['median_age']:.0f}")
    for name in SPLIT_NAMES:
        f = report["floors"].get(name)
        if f is None:
            continue
        print(f"  floor {name:5s} MAE train-median {f['mae_train_median']:.2f}"
              + (f"  | with prior (n={f['with_prior_n']}): copy-last "
                 f"{f['with_prior_mae_copy_last_admission']:.2f} vs median "
                 f"{f['with_prior_mae_train_median']:.2f}" if "with_prior_n" in f else ""))
    print(f"\ndoses  -> {out_dir / 'doses.json'}")
    print(f"report -> {out_dir / 'doses_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
