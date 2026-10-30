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

--diagnose adds why B1 misses, to be read on train (dev stays for the final report):
  * per rule that set the B1 dose: the median label at each dose B1 gave, and the rank
    correlation of B1 with the label (does the rule order patients, whatever its scale?);
    with --llm-checkpoint, B1, B0 and the LLM on the same patients;
  * per range rule: the dose ordered relative to the rule's anchor (oral daily dose, last
    ward dose, last admission's first ICU bolus), split by the conditions the rule assumes;
  * R2: the label and the B1 error by eGFR band.

    python -m physioagent.eval.run_cohort_rules --split dev
    python -m physioagent.eval.run_cohort_rules --split train --diagnose
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from physioagent.eval.run_cohort_dose import (
    MIN_CELL,
    distribution,
    dose_metrics,
    group_metrics,
    read_checkpoint,
    spearman,
    suppressed,
)
from physioagent.kg.ehr.dataset import COHORT_DIR, load_dataset
from physioagent.kg.guideline import t0_rules as tr

DISCRIMINATIVE = ("R3", "R4", "R5")
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
METRIC_KEYS = ("n", "mae", "mae_b0", "skill", "bias", "within_1_step")
# (edges, labels, right): bins over the conditions each rule assumes.
ORAL_DAYS_BINS = ([0, 30, 180, tr.ORAL_CURRENT_DAYS], ["le30d", "31_180d", "181_365d"], True)
WARD_HOURS_BINS = ([tr.REDOSE_MIN_H, 6, 24, np.inf], ["2_6h", "6_24h", "ge24h"], False)
PRIOR_DAYS_BINS = ([0, 90, 365, np.inf], ["le90d", "91_365d", "gt365d"], True)
EGFR_BINS = ([-np.inf, tr.LOW_EGFR, tr.PRE_EXISTING_DYSFUNCTION_EGFR, np.inf],
             ["lt30", "30_60", "ge60"], False)


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


def _num(X: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(X.get(col, pd.Series(np.nan, index=X.index)), errors="coerce")


def _bins(values: pd.Series, spec: tuple) -> pd.Series:
    edges, labels, right = spec
    cut = pd.cut(values, edges, labels=labels, right=right, include_lowest=True)
    return cut.astype(object).where(cut.notna(), "unknown")


def _split(values: pd.Series, by: pd.Series, summary) -> dict[str, Any]:
    return {str(key): summary(part) for key, part in values.groupby(by)}


def quantiles(values: pd.Series) -> Any:
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < MIN_CELL:
        return f"n<{MIN_CELL}"
    q = values.quantile(QUANTILES)
    return {"n": int(len(values)), **{f"p{round(100 * k)}": round(float(v), 2)
                                      for k, v in q.items()}}


def relation(diff: pd.Series) -> Any:
    """Share of patients whose label is below, at, or above the anchor (diff = y - anchor)."""
    diff = diff.dropna()
    n = len(diff)
    if n < MIN_CELL:
        return f"n<{MIN_CELL}"
    return {"n": n, "lower": _cell(int((diff < 0).sum()), n),
            "equal": _cell(int((diff == 0).sum()), n), "higher": _cell(int((diff > 0).sum()), n)}


def _metrics(y: pd.Series, pred: pd.Series, b0: float) -> dict[str, Any]:
    m = suppressed(dose_metrics(y.to_numpy(float), pred.to_numpy(float), b0))
    return {k: m[k] for k in METRIC_KEYS if k in m}


def calibration(y: pd.Series, pred: pd.Series) -> dict[str, Any]:
    """The label at each dose B1 gave: a rule that orders patients shows rising medians."""
    cells: dict[str, Any] = {}
    for dose, part in y.groupby(pred):
        cells[f"{dose:g}"] = (f"n<{MIN_CELL}" if len(part) < MIN_CELL else
                              {"n": int(len(part)), "label_median": float(part.median()),
                               "label_mean": round(float(part.mean()), 1)})
    return cells


def by_source_rule(rules: pd.DataFrame, y: pd.Series, b0: float,
                   llm: Optional[pd.Series]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    b1 = rules["b1_dose"]
    for name in ("any", *tr.RANGE_PRIORITY, "none"):
        mask = (b1.notna() if name == "any" else b1.isna() if name == "none"
                else rules["b1_rule"].eq(name))
        if mask.sum() < MIN_CELL:
            out[name] = f"n<{MIN_CELL}"
            continue
        entry: dict[str, Any] = {"n": int(mask.sum())}
        if name != "none":
            entry.update(b1=_metrics(y[mask], b1[mask], b0),
                         spearman_b1=spearman(b1[mask], y[mask]),
                         calibration=calibration(y[mask], b1[mask]))
        if llm is not None:
            both = mask & llm.notna()
            if both.sum() < MIN_CELL:
                entry["same_patients_as_llm"] = f"n<{MIN_CELL}"
            else:
                same: dict[str, Any] = {"n": int(both.sum()),
                                        "llm": _metrics(y[both], llm[both], b0),
                                        "spearman_llm": spearman(llm[both], y[both])}
                if name != "none":
                    same.update(b1=_metrics(y[both], b1[both], b0),
                                spearman_b1=spearman(b1[both], y[both]))
                entry["same_patients_as_llm"] = same
        out[name] = entry
    return out


def anchors(X: pd.DataFrame, rules: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    """The dose ordered against each range rule's anchor, where the rule fires."""
    out: dict[str, Any] = {}

    on = rules["R3_status"].eq(tr.FIRES)
    daily, single = _num(X, "oral_recorded_daily_mg")[on], _num(X, "oral_recorded_mg")[on]
    ratio = y[on] / daily
    source = X.loc[on, "oral_recorded_source"].astype(str) if "oral_recorded_source" in X \
        else pd.Series("unknown", index=ratio.index)
    out["R3_label_over_oral_daily"] = {
        "all": quantiles(ratio),
        "label_over_single_oral_dose": quantiles(y[on] / single),
        "by_source": _split(ratio, source, quantiles),
        "by_days_since_oral_record": _split(
            ratio, _bins(_num(X, "oral_recorded_days")[on], ORAL_DAYS_BINS), quantiles)}

    # Anchors are rounded to 0.1 mg: inputevents amounts carry float noise (39.99998), which
    # would read as "lower" against a label on the ladder.
    on = rules["R4_status"].eq(tr.FIRES)
    last = _num(X, "ward_iv_last_mg")[on].round(1)
    diff, ratio = y[on] - last, y[on] / last
    hours = _bins(_num(X, "ward_iv_last_hours_ago")[on], WARD_HOURS_BINS)
    n_doses = _num(X, "ward_iv_n")[on].map(lambda n: "1" if n == 1 else "ge2")
    out["R4_label_vs_last_ward_dose"] = {
        "all": relation(diff),
        "label_over_last": quantiles(ratio),
        "by_hours_since_last": {"relation": _split(diff, hours, relation),
                                "label_over_last": _split(ratio, hours, quantiles)},
        "by_ward_doses": _split(diff, n_doses, relation)}

    on = rules["R5_status"].eq(tr.FIRES)
    prior = _num(X, "prior_icu_first_bolus_mg")[on].round(1)
    ratio, diff = y[on] / prior, y[on] - prior
    branch = (_num(X, "prior_icu_first_bolus_uo_6h_ml")[on] < tr.INADEQUATE_UO_6H_ML).map(
        {True: "inadequate", False: "adequate"})
    escalated = _num(X, "prior_icu_first_bolus_escalated")[on].map(
        {1.0: "escalated", 0.0: "not_escalated"}).fillna("unknown")
    days = _bins(_num(X, "prior_icu_first_bolus_days")[on], PRIOR_DAYS_BINS)
    out["R5_label_vs_prior_first_bolus"] = {
        "all": relation(diff),
        "label_over_prior": quantiles(ratio),
        "by_uo_branch": {"label_over_prior": _split(ratio, branch, quantiles),
                         "relation": _split(diff, branch, relation)},
        "by_days_since_prior_bolus": _split(ratio, days, quantiles),
        "by_prior_escalated": _split(ratio, escalated, quantiles)}
    return out


def egfr_report(X: pd.DataFrame, rules: pd.DataFrame, y: pd.Series, b0: float
                ) -> dict[str, Any]:
    """R2 says lower kidney function wants a higher dose: does practice agree, and does
    taking the high end below LOW_EGFR help B1?"""
    egfr = pd.Series([tr.kidney_egfr(row) for _, row in X.iterrows()], index=X.index,
                     dtype=float)
    band = _bins(egfr, EGFR_BINS)
    answered = rules["b1_dose"].notna()
    b1 = rules["b1_dose"]
    return {"label_by_egfr": _split(y, band, quantiles),
            "b1_by_egfr": {str(k): _metrics(y[answered & band.eq(k)],
                                            b1[answered & band.eq(k)], b0)
                           for k in sorted(band[answered].unique())}}


def diagnose_report(X: pd.DataFrame, rules: pd.DataFrame, y: pd.Series, b0: float,
                    llm: Optional[pd.Series] = None) -> dict[str, Any]:
    return {"by_source_rule": by_source_rule(rules, y, b0, llm),
            "anchors": anchors(X, rules, y),
            "R2_egfr": egfr_report(X, rules, y, b0)}


def llm_predictions(path: Path, index: pd.Index) -> pd.Series:
    rows = read_checkpoint(path)
    preds = {h: r["pred_dose_mg"] for h, r in rows.items() if r.get("status") == "ok"}
    return pd.Series([preds.get(str(h)) for h in index], index=index, dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="dev", choices=("train", "dev", "test"))
    ap.add_argument("--split-col", default="split", choices=("split", "split_temporal"))
    ap.add_argument("--cohort-dir", type=Path, default=COHORT_DIR)
    ap.add_argument("--tag", default="")
    ap.add_argument("--save", type=Path, help="also write the report here")
    ap.add_argument("--diagnose", action="store_true",
                    help="add why B1 misses (read it on train; dev is for the final report)")
    ap.add_argument("--llm-checkpoint", type=Path,
                    help="with --diagnose: an LLM arm's checkpoint of the same split, to "
                         "score B1, B0 and the LLM on the same patients")
    args = ap.parse_args()
    if args.llm_checkpoint and not args.diagnose:
        ap.error("--llm-checkpoint needs --diagnose")

    X, y, groups = load_dataset(args.split, split_col=args.split_col,
                                cohort_dir=args.cohort_dir, tag=args.tag)
    _, y_train, _ = load_dataset("train", split_col=args.split_col,
                                 cohort_dir=args.cohort_dir, tag=args.tag)
    b0 = float(y_train.median())
    rules = apply_rules(X)
    report = {"split": f"{args.split_col}={args.split}{args.tag}",
              **coverage_report(rules, y, groups, b0)}
    if args.diagnose:
        llm = llm_predictions(args.llm_checkpoint, X.index) if args.llm_checkpoint else None
        report["diagnose"] = diagnose_report(X, rules, y, b0, llm)
    text = json.dumps(report, indent=2)
    print(text)
    if args.save:
        args.save.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
