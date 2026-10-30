"""Coverage of the guideline rules (kg/guideline/t0_rules.py) on the T0 cohort, and B1.

For one split it reports, aggregates only (cells under MIN_CELL suppressed):
  * per rule: how many patients it fires on, cannot be applied to, or lacks an input for;
  * coverage: patients with at least one dose-range rule, and with one of the rules that
    can tell patients apart (R3, R4, R5 -- R1 and R2 fire on most of a CKD cohort);
  * concordance: for each range rule, how often the dose actually ordered lies in its
    range (does practice follow the guideline where it applies?);
  * B1: the rule dose scored like an LLM arm, on the patients it answers and on all of them
    with B0 for the rest.
No LLM is involved, so this runs anywhere the cohort files are.

    python -m physioagent.eval.run_cohort_rules --split dev
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from physioagent.eval.run_cohort_dose import (
    MIN_CELL,
    distribution,
    dose_metrics,
    group_metrics,
    suppressed,
)
from physioagent.kg.ehr.dataset import COHORT_DIR, load_dataset
from physioagent.kg.guideline import t0_rules as tr

DISCRIMINATIVE = ("R3", "R4", "R5")


def _cell(n: int, total: int) -> Any:
    if n < MIN_CELL:
        return f"<{MIN_CELL}"
    return {"n": int(n), "pct": round(100.0 * n / total, 1)}


def apply_rules(X: pd.DataFrame) -> pd.DataFrame:
    """One row per patient: each rule's status, range bounds, and the B1 dose."""
    out = []
    for hadm_id, row in X.iterrows():
        firings = tr.evaluate(row)
        dose, source = tr.b1_dose(firings, row)
        rec: dict[str, Any] = {"hadm_id": hadm_id, "exposure": tr.exposure(row),
                               "b1_dose": dose, "b1_rule": source}
        for rule_id, f in firings.items():
            rec[f"{rule_id}_status"] = f.status
            rec[f"{rule_id}_low"], rec[f"{rule_id}_high"] = f.low, f.high
        out.append(rec)
    return pd.DataFrame(out).set_index("hadm_id")


def coverage_report(rules: pd.DataFrame, y: pd.Series, groups: pd.DataFrame, b0: float
                    ) -> dict[str, Any]:
    total = len(rules)
    status = {r: {s: _cell(int(c), total)
                  for s, c in rules[f"{r}_status"].value_counts().items()}
              for r in tr.RULES}
    fires = {r: rules[f"{r}_status"].eq(tr.FIRES) for r in tr.RULES}
    any_range = np.logical_or.reduce([fires[r] for r in tr.RANGE_PRIORITY])
    discriminative = np.logical_or.reduce([fires[r] for r in DISCRIMINATIVE])
    concordance = {}
    for r in tr.RANGE_PRIORITY:
        on = fires[r]
        inside = (y[on] >= rules.loc[on, f"{r}_low"]) & (y[on] <= rules.loc[on, f"{r}_high"])
        below = y[on] < rules.loc[on, f"{r}_low"]
        concordance[r] = (f"n<{MIN_CELL}" if on.sum() < MIN_CELL else
                          {"n": int(on.sum()), "label_in_range_pct": round(100 * inside.mean(), 1),
                           "label_below_range_pct": round(100 * below.mean(), 1)})
    answered = rules["b1_dose"].notna()
    frame = groups.loc[answered].assign(y=y[answered], pred=rules.loc[answered, "b1_dose"])
    imputed = rules["b1_dose"].fillna(b0).to_numpy(float)
    return {
        "n": total,
        "exposure_on_record": {k: _cell(int(v), total)
                               for k, v in rules["exposure"].value_counts().items()},
        "rule_status": status,
        "coverage": {"any_range_rule": _cell(int(any_range.sum()), total),
                     "discriminative_rule_R3_R4_R5": _cell(int(discriminative.sum()), total),
                     "b1_source_rule": {k or "none": _cell(int(v), total) for k, v in
                                        rules["b1_rule"].value_counts().items()}},
        "label_vs_rule_range": concordance,
        "b1": {
            "answered": suppressed(dose_metrics(frame["y"].to_numpy(float),
                                                frame["pred"].to_numpy(float), b0)),
            "all_targets_unanswered_as_b0": suppressed(
                dose_metrics(y.to_numpy(float), imputed, b0)),
            "pred_distribution": distribution(frame["pred"]),
            "by_group": group_metrics(frame, b0),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="dev", choices=("train", "dev", "test"))
    ap.add_argument("--split-col", default="split", choices=("split", "split_temporal"))
    ap.add_argument("--cohort-dir", type=Path, default=COHORT_DIR)
    ap.add_argument("--tag", default="")
    ap.add_argument("--save", type=Path, help="also write the report here")
    args = ap.parse_args()

    X, y, groups = load_dataset(args.split, split_col=args.split_col,
                                cohort_dir=args.cohort_dir, tag=args.tag)
    _, y_train, _ = load_dataset("train", split_col=args.split_col,
                                 cohort_dir=args.cohort_dir, tag=args.tag)
    report = {"split": f"{args.split_col}={args.split}{args.tag}",
              **coverage_report(apply_rules(X), y, groups, float(y_train.median()))}
    text = json.dumps(report, indent=2)
    print(text)
    if args.save:
        args.save.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
