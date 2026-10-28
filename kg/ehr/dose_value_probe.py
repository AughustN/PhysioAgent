#!/usr/bin/env python
"""Which mg values do IV furosemide boluses actually take in MIMIC-IV?

Answers whether a 10 mg output grid fits the labels: the share of doses that are
multiples of 10, the share at 30/50/70 mg specifically (values the old ladder could not
produce), and the share that is not a multiple of 10 at all.

Read from icu/inputevents, itemid 221794, Drug Push/Bolus, unit mg, amount > 0 -- the
same events the cohort is built from. Three grains are counted, because they answer
different questions:

    all_boluses      every bolus
    first_per_hadm   first bolus of each admission (the label grain)
    first_summed_Nm  that first bolus plus every further bolus of the same admission
                     within N minutes, i.e. the dose if back-to-back pushes are one decision

Rows with statusdescription "Rewritten" are MetaVision corrections superseded by another
row; they are dropped by default and the report says how many there were.

Only aggregate counts are written. Nothing keyed to a patient leaves this script.

    python -m physioagent.kg.ehr.dose_value_probe --mimic-root <mimic-iv-3.1>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from physioagent.kg import paths
from physioagent.kg.ehr.build_visits import read_table, resolve_mimic_root

FUROSEMIDE_BOLUS_ITEMID = "221794"
BOLUS_CATEGORIES = {"Drug Push", "Bolus"}
OLD_LADDER = (10, 20, 40, 60, 80, 100, 120, 140, 160, 200)
WATCHED = (30, 50, 70)
GRID_MG = 10
GRID_MAX_MG = 200
TOP_N = 25
CHUNK = 2_000_000
EPS = 1e-6

USECOLS = ["hadm_id", "starttime", "itemid", "amount", "amountuom",
           "ordercategorydescription", "statusdescription"]


def load_boluses(root: Path, keep_rewritten: bool = False) -> tuple[pd.DataFrame, dict[str, int]]:
    kept: list[pd.DataFrame] = []
    counts = {"furosemide_rows": 0, "bolus_mg_rows": 0, "rewritten_dropped": 0}
    for chunk in read_table(root, "icu", "inputevents", USECOLS, chunksize=CHUNK):
        chunk = chunk[chunk["itemid"].eq(FUROSEMIDE_BOLUS_ITEMID)]
        counts["furosemide_rows"] += len(chunk)
        chunk = chunk[chunk["ordercategorydescription"].isin(BOLUS_CATEGORIES)
                      & chunk["amountuom"].eq("mg")]
        chunk = chunk.assign(amount=pd.to_numeric(chunk["amount"], errors="coerce"))
        chunk = chunk[chunk["amount"].gt(0)]
        counts["bolus_mg_rows"] += len(chunk)
        rewritten = chunk["statusdescription"].eq("Rewritten")
        counts["rewritten_dropped"] += 0 if keep_rewritten else int(rewritten.sum())
        if not keep_rewritten:
            chunk = chunk[~rewritten]
        kept.append(chunk[["hadm_id", "starttime", "amount"]])
    df = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(
        columns=["hadm_id", "starttime", "amount"])
    df["starttime"] = pd.to_datetime(df["starttime"], errors="coerce")
    df = df.dropna(subset=["starttime"]).sort_values(["hadm_id", "starttime"])
    return df.reset_index(drop=True), counts


def first_per_admission(df: pd.DataFrame, window_min: Optional[float] = None) -> pd.Series:
    """The first bolus of each admission; with window_min, plus every bolus of that
    admission starting within window_min minutes of it."""
    df = df[df["hadm_id"].ne("")]
    t0 = df.groupby("hadm_id")["starttime"].transform("min")
    if window_min is None:
        return df[df["starttime"].eq(t0)].groupby("hadm_id")["amount"].first()
    within = df["starttime"] <= t0 + pd.Timedelta(minutes=window_min)
    return df[within].groupby("hadm_id")["amount"].sum()


def _is_multiple(values: np.ndarray, step: float) -> np.ndarray:
    return np.abs(values - np.round(values / step) * step) < EPS


def _pct(n: int, total: int) -> float:
    return round(100.0 * n / total, 2) if total else 0.0


def summarize(amounts: pd.Series) -> dict[str, Any]:
    values = amounts.to_numpy(dtype=float)
    total = len(values)
    mult10 = _is_multiple(values, GRID_MG)
    mult5_not10 = _is_multiple(values, 5) & ~mult10
    integer = _is_multiple(values, 1)

    def share(mask: np.ndarray) -> dict[str, float]:
        n = int(mask.sum())
        return {"n": n, "pct": _pct(n, total)}

    watched = {str(v): share(np.abs(values - v) < EPS) for v in WATCHED}
    in_old_ladder = np.isin(np.round(values, 6), OLD_LADDER)
    grid = {str(v): share(np.abs(values - v) < EPS)
            for v in range(GRID_MG, GRID_MAX_MG + GRID_MG, GRID_MG)}
    rounded = pd.Series(np.round(values, 3))
    top = rounded.value_counts().head(TOP_N)
    return {
        "n": total,
        "median_mg": float(np.median(values)) if total else None,
        "multiple_of_10": share(mult10),
        "not_multiple_of_10": share(~mult10),
        "not10_multiple_of_5": share(mult5_not10),
        "not10_other_integer": share(integer & ~mult10 & ~mult5_not10),
        "not10_non_integer": share(~integer),
        "at_30_50_70": share(np.isin(np.round(values, 6), WATCHED)),
        "watched_values": watched,
        "in_old_ladder": share(in_old_ladder),
        "multiple_of_10_off_old_ladder": share(mult10 & ~in_old_ladder),
        "below_10": share(values < GRID_MG - EPS),
        "above_200": share(values > GRID_MAX_MG + EPS),
        "grid_10_to_200": grid,
        "top_values": [{"mg": float(mg), "n": int(n), "pct": _pct(int(n), total)}
                       for mg, n in top.items()],
    }


def _print(name: str, s: dict[str, Any]) -> None:
    print(f"\n=== {name}  (n = {s['n']:,}, median {s['median_mg']} mg) ===")
    for key in ("multiple_of_10", "not_multiple_of_10", "not10_multiple_of_5",
                "not10_other_integer", "not10_non_integer", "at_30_50_70",
                "in_old_ladder", "multiple_of_10_off_old_ladder", "below_10", "above_200"):
        label = "  " + key.replace("not10_", "of which ") if key.startswith("not10_") else key
        print(f"  {label:32s} {s[key]['n']:>8,}  {s[key]['pct']:6.2f}%")
    for mg, v in s["watched_values"].items():
        print(f"    = {mg:>3} mg{'':22s} {v['n']:>8,}  {v['pct']:6.2f}%")
    print("  top values:")
    for row in s["top_values"][:12]:
        print(f"    {row['mg']:>8g} mg  {row['n']:>8,}  {row['pct']:6.2f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mimic-root", help="directory holding hosp/ and icu/ "
                                         "(default: $MIMIC_ROOT)")
    ap.add_argument("--windows", type=float, nargs="*", default=[15.0, 30.0],
                    help="minutes for the summed-first-dose grains (default 15 30)")
    ap.add_argument("--keep-rewritten", action="store_true",
                    help="keep MetaVision 'Rewritten' rows (dropped by default)")
    ap.add_argument("--out", default=str(paths.EHR / "dose_value_probe.json"))
    args = ap.parse_args()

    root = resolve_mimic_root(args.mimic_root)
    print(f"mimic root : {root}")
    df, counts = load_boluses(root, args.keep_rewritten)
    print(f"furosemide inputevents rows {counts['furosemide_rows']:,}; "
          f"bolus/push in mg > 0 {counts['bolus_mg_rows']:,}; "
          f"rewritten dropped {counts['rewritten_dropped']:,}; "
          f"admissions {df['hadm_id'].nunique():,}")

    grains = {"all_boluses": df["amount"], "first_per_hadm": first_per_admission(df)}
    for window in args.windows:
        grains[f"first_summed_{window:g}m"] = first_per_admission(df, window)
    report = {"source": "icu/inputevents itemid 221794, Drug Push/Bolus, mg, amount > 0",
              "keep_rewritten": args.keep_rewritten, "row_counts": counts,
              "grains": {name: summarize(series) for name, series in grains.items()}}
    for name, s in report["grains"].items():
        _print(name, s)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
