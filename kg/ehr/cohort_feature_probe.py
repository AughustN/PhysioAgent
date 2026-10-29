#!/usr/bin/env python
"""Probe 2: what the cohort spec still needs measured, on the target cohort.

Reads the S9 cohort written by `cohort_funnel_probe --save-cohort`, keeps the last qualifying
admission per patient (spec section 1), and counts:

    M1  feature coverage before t0: vitals (6 h), labs (48 h), urine output, weight
    M2  flags: anuric, loading bolus before an infusion, code status
    M3  history: prior admissions, prior furosemide by source and route, gap to the last one
    M4  in-hospital death of the target admission vs the other S9 admissions
    M5  CKD (b) tightened: outpatient creatinine + omr only, no eGFR >= 60 in between;
        prior I12/I13/403/404 without N18/585
    M6  baseline eGFR before admission, and the stage it gives vs eGFR at t0
    M7  latest oral furosemide on record: prior discharge prescription, oral given before t0
    M8  order -> administration gap of the IV order that matches the t0 dose (spec 4.6)
    B0  constant-dose baselines on the target cohort

Every rule keeps a row only when it was stored before t0 (`storetime < t0`) unless noted.
Only aggregate counts are written.

    python -m physioagent.kg.ehr.cohort_feature_probe --mimic-root <mimic-iv-3.1>
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
from physioagent.kg.ehr.cohort_funnel_probe import (
    BOLUS_CATEGORIES, CHRONIC_DAYS, CKD_PREFIXES, COHORT_FILE, CREATININE_ITEMID, DOSE_EPS,
    EGFR_CKD, FUROSEMIDE_ITEMIDS, IV_ROUTE, _code, _counts, _log, _num, _read_filtered, _ts,
    administered, classify_route, egfr_ckd_epi_2021, hadms_with_event)

VITALS = {"heart_rate": (220045,), "sbp": (220179, 220050), "map": (220181, 220052),
          "spo2": (220277,), "resp_rate": (220210,), "temperature": (223761, 223762)}
WEIGHT_ITEMIDS = {226512: 1.0, 224639: 1.0, 226531: 0.453592}   # itemid -> factor to kg
CODE_STATUS_ITEMID = 223758
VITAL_WINDOW_H = 6

LABS = {"creatinine": CREATININE_ITEMID, "sodium": "50983", "potassium": "50971",
        "bun": "51006", "bicarbonate": "50882", "chloride": "50902", "albumin": "50862",
        "hemoglobin": "51222", "magnesium": "50960", "glucose": "50931", "lactate": "50813",
        "ntprobnp": "50963"}
LAB_WINDOW_H = 48

URINE_ITEMIDS = {226559, 226560, 226561, 226584, 226563, 226564, 226565, 226567, 226557,
                 226558, 227489}
IRRIGANT_IN_ITEMID = 227488                # GU irrigant volume in: subtracted
UO_WINDOWS_H = (6, 24)
ANURIC_ML_24H = 100.0

LOADING_AFTER_H = 1
HTN_CKD_PREFIXES = {"10": ("I12", "I13"), "9": ("403", "404")}
BASELINE_MAX_DAYS = 365
ORAL_ROUTES = {"PO", "PO/NG", "NG", "ORAL"}
DISCHARGE_ACTIVE_H = 24
ORDER_WINDOW_H = 24
GAP_BINS_H = (0, 0.5, 1, 3, 6, 12, 24)
EGFR_STAGES = ((90, "G1"), (60, "G2"), (45, "G3a"), (30, "G3b"), (15, "G4"), (0, "G5"))


# --- target selection and windows ---------------------------------------------------------

def select_targets(cohort: pd.DataFrame) -> pd.DataFrame:
    """Last qualifying admission per patient (spec section 1)."""
    return cohort.sort_values(["subject_id", "t0"]).drop_duplicates("subject_id", keep="last")


def in_window(targets: pd.DataFrame, events: pd.DataFrame, hours: Optional[float],
              on: str = "hadm_id") -> pd.DataFrame:
    """Rows stored before t0 and charted in [t0 - hours, t0); hours=None means no lower bound."""
    cols = ["hadm_id", "subject_id", "t0"]
    j = targets[cols].merge(events, on=on, suffixes=("", "_ev"))
    keep = j["storetime"] < j["t0"]
    if hours is not None:
        keep &= j["charttime"] >= j["t0"] - pd.Timedelta(hours=hours)
    return j[keep]


def coverage(targets: pd.DataFrame, rows: pd.DataFrame, names: list[str]) -> dict[str, Any]:
    n = len(targets)
    have = rows.groupby("feature")["hadm_id"].nunique() if len(rows) else pd.Series(dtype=int)
    return {f: {"n": int(have.get(f, 0)), "pct": round(100.0 * have.get(f, 0) / n, 2) if n else 0.0}
            for f in names}


def urine_output(targets: pd.DataFrame, uo: pd.DataFrame, hours: float) -> pd.Series:
    """mL voided in [t0 - hours, t0) per target; NaN = nothing charted in the window."""
    rows = in_window(targets, uo, hours)
    return rows.groupby("hadm_id")["ml"].sum().reindex(targets["hadm_id"])


def loading_for_infusion(targets: pd.DataFrame, infusion: pd.DataFrame) -> set:
    after = pd.Timedelta(hours=LOADING_AFTER_H)
    return hadms_with_event(targets, infusion[["hadm_id", "starttime"]], "hadm_id",
                            lambda j: (j["starttime"] > j["t0"]) & (j["starttime"] <= j["t0"] + after))


def latest_value(targets: pd.DataFrame, rows: pd.DataFrame) -> pd.Series:
    j = in_window(targets, rows, None).sort_values("charttime")
    return j.groupby("hadm_id")["value"].last()


# --- history --------------------------------------------------------------------------------

def prior_pairs(targets: pd.DataFrame, admissions: pd.DataFrame) -> pd.DataFrame:
    """One row per (target, earlier admission of the same patient)."""
    others = admissions[["subject_id", "hadm_id", "admittime", "dischtime"]].rename(
        columns={"hadm_id": "prior_hadm", "admittime": "prior_admittime",
                 "dischtime": "prior_dischtime"})
    pairs = targets[["hadm_id", "subject_id", "admittime"]].merge(others, on="subject_id")
    return pairs[pairs["prior_admittime"] < pairs["admittime"]]


def prior_exposure(targets: pd.DataFrame, prior_events: pd.DataFrame) -> pd.Series:
    """A3: ward IV this admission > furosemide on an earlier admission > unverifiable > naive."""
    prior = targets["hadm_id"].isin(set(prior_events["hadm_id"]))
    group = np.select([targets["emar_group"].eq("escalation"), prior,
                       targets["emar_group"].eq("unverifiable")],
                      ["ward_this_admission", "prior_admission", "unverifiable"], default="naive")
    return pd.Series(group, index=targets.index)


# --- CKD ------------------------------------------------------------------------------------

def chronic_low_strict(targets: pd.DataFrame, readings: pd.DataFrame) -> set:
    """Two eGFR < 60 readings >= CHRONIC_DAYS apart before admission, with no eGFR >= 60
    between them. `readings` holds subject_id, time, egfr (normal readings included)."""
    j = targets[["hadm_id", "subject_id", "admittime"]].merge(readings, on="subject_id")
    j = j[j["time"] < j["admittime"]].sort_values(["hadm_id", "time"])
    low = j["egfr"] < EGFR_CKD
    run = (~low).astype(int).groupby(j["hadm_id"]).cumsum()
    lows = j[low].assign(run=run[low])
    span = lows.groupby(["hadm_id", "run"])["time"].agg(["min", "max"])
    ok = span[(span["max"] - span["min"]) >= pd.Timedelta(days=CHRONIC_DAYS)]
    return set(ok.index.get_level_values("hadm_id"))


def prefix_flag(dx: pd.DataFrame, prefixes: dict[str, tuple[str, ...]]) -> pd.Series:
    code, version = _code(dx["icd_code"]), dx["icd_version"].astype(str)
    hit = pd.Series(False, index=dx.index)
    for ver, pre in prefixes.items():
        hit |= version.eq(ver) & code.str.startswith(pre)
    return hit


def egfr_stage(egfr: pd.Series) -> pd.Series:
    out = pd.Series("missing", index=egfr.index)
    for floor, name in reversed(EGFR_STAGES):
        out = out.mask(egfr >= floor, name)
    return out


def baseline_egfr(targets: pd.DataFrame, readings: pd.DataFrame) -> pd.Series:
    """Most recent outpatient eGFR within BASELINE_MAX_DAYS before admission."""
    j = targets[["hadm_id", "subject_id", "admittime"]].merge(readings, on="subject_id")
    j = j[(j["time"] < j["admittime"])
          & (j["time"] >= j["admittime"] - pd.Timedelta(days=BASELINE_MAX_DAYS))]
    return j.sort_values("time").groupby("hadm_id")["egfr"].last()


# --- orders and baselines ---------------------------------------------------------------------

def matching_order(targets: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Latest IV order entered in [t0 - 24 h, t0) whose dose equals the t0 dose."""
    j = targets[["hadm_id", "t0", "dose_mg"]].merge(orders, on="hadm_id")
    w = j[(j["visible_time"] < j["t0"])
          & (j["visible_time"] >= j["t0"] - pd.Timedelta(hours=ORDER_WINDOW_H))
          & ((j["order_mg"] - j["dose_mg"]).abs() < DOSE_EPS)]
    last = w.sort_values("visible_time").drop_duplicates("hadm_id", keep="last")
    return last.assign(gap_h=(last["t0"] - last["visible_time"]).dt.total_seconds() / 3600)


def constant_baselines(dose: pd.Series) -> dict[str, dict[str, float]]:
    out = {}
    for c in (20.0, 40.0):
        err = (dose - c).abs()
        out[f"always_{int(c)}mg"] = {
            "mae_mg": round(float(err.mean()), 2),
            "mape_pct": round(float(100 * (err / dose).mean()), 2),
            "within_20pct": round(float(100 * (err <= 0.2 * dose + DOSE_EPS).mean()), 2)}
    return out


def _quantiles(series: pd.Series) -> dict[str, Optional[float]]:
    s = series.dropna()
    if s.empty:
        return {"n": 0}
    q = s.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    return {"n": int(len(s)), **{f"p{int(k * 100)}": round(float(v), 2) for k, v in q.items()}}


def _share(n: int, total: int) -> dict[str, Any]:
    return {"n": int(n), "pct": round(100.0 * n / total, 2) if total else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mimic-root", help="directory holding hosp/ and icu/ (default $MIMIC_ROOT)")
    ap.add_argument("--cohort", default=str(COHORT_FILE),
                    help="S9 cohort from cohort_funnel_probe --save-cohort")
    ap.add_argument("--out", default=str(paths.EHR / "cohort_feature_probe.json"))
    args = ap.parse_args()
    root = resolve_mimic_root(args.mimic_root)
    cohort_path = Path(args.cohort)
    if not cohort_path.exists():
        print(f"missing {cohort_path}: run cohort_funnel_probe --save-cohort first",
              file=sys.stderr)
        return 1

    s9 = pd.read_csv(cohort_path, dtype=str, keep_default_na=False)
    s9 = s9.assign(t0=_ts(s9["t0"]), admittime=_ts(s9["admittime"]), intime=_ts(s9["intime"]),
                   dose_mg=_num(s9["dose_mg"]), anchor_age=_num(s9["anchor_age"]),
                   anchor_year=_num(s9["anchor_year"]))
    targets = select_targets(s9).reset_index(drop=True)
    n = len(targets)
    hadms, subjects = set(targets["hadm_id"]), set(targets["subject_id"])
    _log(f"S9 {len(s9):,} admissions -> {n:,} targets (last qualifying admission per patient)")
    report: dict[str, Any] = {"s9_admissions": int(len(s9)), "targets": n}

    # ---- admissions, patients ------------------------------------------------------------
    _log("reading hosp/admissions, patients")
    admissions = read_table(root, "hosp", "admissions",
                            ["subject_id", "hadm_id", "admittime", "dischtime",
                             "hospital_expire_flag"])
    admissions = admissions[admissions["subject_id"].isin(subjects)].assign(
        admittime=lambda d: _ts(d["admittime"]), dischtime=lambda d: _ts(d["dischtime"]))
    patients = read_table(root, "hosp", "patients", ["subject_id", "gender", "anchor_age",
                                                      "anchor_year"])
    patients = patients[patients["subject_id"].isin(subjects)].set_index("subject_id")

    # M4: death of the target admission vs the S9 admissions not chosen
    died = admissions.set_index("hadm_id")["hospital_expire_flag"].eq("1")
    others = s9[~s9["hadm_id"].isin(hadms)]
    report["m4_in_hospital_death"] = {
        "target": _share(died.reindex(targets["hadm_id"], fill_value=False).sum(), n),
        "other_s9_admissions": _share(died.reindex(others["hadm_id"], fill_value=False).sum(),
                                      len(others))}

    # ---- M1/M2: chartevents ---------------------------------------------------------------
    _log("reading icu/chartevents (vitals, weight, code status; slow)")
    vital_of = {str(i): name for name, ids in VITALS.items() for i in ids}
    wanted = set(vital_of) | {str(i) for i in WEIGHT_ITEMIDS} | {str(CODE_STATUS_ITEMID)}
    ce = _read_filtered(root, "icu", "chartevents",
                        ["hadm_id", "itemid", "charttime", "storetime", "value", "valuenum"],
                        lambda c: c["itemid"].isin(wanted) & c["hadm_id"].isin(hadms))
    ce = ce.assign(charttime=_ts(ce["charttime"]), storetime=_ts(ce["storetime"]))
    vitals = ce[ce["itemid"].isin(set(vital_of))].assign(
        feature=lambda d: d["itemid"].map(vital_of))
    vitals = vitals[_num(vitals["valuenum"]).notna()]
    weight = ce[ce["itemid"].isin({str(i) for i in WEIGHT_ITEMIDS})].assign(feature="weight")
    weight = weight[_num(weight["valuenum"]).gt(0)]
    status = ce[ce["itemid"].eq(str(CODE_STATUS_ITEMID))]

    m1 = {f"vitals_{VITAL_WINDOW_H}h": coverage(
        targets, in_window(targets, vitals, VITAL_WINDOW_H), list(VITALS))}
    m1["weight_this_admission"] = coverage(targets, in_window(targets, weight, None),
                                           ["weight"])["weight"]

    # ---- M1: labs (target admission) + creatinine (all time, for CKD and baseline) --------
    _log("reading hosp/labevents (labs; slow)")
    lab_of = {v: k for k, v in LABS.items()}
    labs = _read_filtered(root, "hosp", "labevents",
                          ["subject_id", "hadm_id", "itemid", "charttime", "storetime",
                           "valuenum"],
                          lambda c: c["itemid"].isin(set(lab_of)) & c["subject_id"].isin(subjects))
    labs = labs.assign(charttime=_ts(labs["charttime"]), storetime=_ts(labs["storetime"]),
                       valuenum=_num(labs["valuenum"]), feature=labs["itemid"].map(lab_of))
    labs = labs.dropna(subset=["valuenum"])
    target_labs = labs[labs["hadm_id"].isin(hadms)].drop(columns="subject_id")
    m1[f"labs_{LAB_WINDOW_H}h"] = coverage(targets, in_window(targets, target_labs, LAB_WINDOW_H),
                                           list(LABS))

    # ---- M1/M2: urine output ---------------------------------------------------------------
    _log("reading icu/outputevents (urine)")
    uo_ids = {str(i) for i in URINE_ITEMIDS | {IRRIGANT_IN_ITEMID}}
    uo = _read_filtered(root, "icu", "outputevents",
                        ["hadm_id", "itemid", "charttime", "storetime", "value"],
                        lambda c: c["itemid"].isin(uo_ids) & c["hadm_id"].isin(hadms))
    uo = uo.assign(charttime=_ts(uo["charttime"]), storetime=_ts(uo["storetime"]),
                   ml=_num(uo["value"]).fillna(0.0))
    uo.loc[uo["itemid"].eq(str(IRRIGANT_IN_ITEMID)), "ml"] *= -1
    icu_hours = (targets["t0"] - targets["intime"]).dt.total_seconds() / 3600
    uo_by_h = {h: urine_output(targets, uo, h) for h in UO_WINDOWS_H}
    m1["urine_output"] = {f"charted_{h}h": _share(v.notna().sum(), n) for h, v in uo_by_h.items()}
    m1["urine_output"]["ml_6h"] = _quantiles(uo_by_h[6])
    m1["icu_hours_before_t0"] = _quantiles(icu_hours)
    report["m1_coverage"] = m1

    full_day = icu_hours.to_numpy() >= 24
    uo24 = uo_by_h[24].to_numpy()
    anuric = full_day & (np.nan_to_num(uo24, nan=np.inf) < ANURIC_ML_24H)
    m2 = {"anuric": {"eligible_24h_in_icu": int(full_day.sum()),
                     "uo_24h_below_100ml": int(anuric.sum())}}

    _log("reading icu/inputevents (furosemide)")
    furo_ids = {str(i) for i in FUROSEMIDE_ITEMIDS}
    ie = _read_filtered(root, "icu", "inputevents",
                        ["subject_id", "hadm_id", "starttime", "itemid", "amount",
                         "ordercategorydescription", "statusdescription"],
                        lambda c: c["itemid"].isin(furo_ids) & c["subject_id"].isin(subjects))
    ie = ie[ie["statusdescription"].ne("Rewritten")].assign(starttime=_ts(ie["starttime"]))
    infusion = ie[~ie["ordercategorydescription"].isin(BOLUS_CATEGORIES)]
    m2["loading_for_infusion"] = _share(len(loading_for_infusion(targets, infusion)), n)
    m2["code_status_latest"] = _counts(latest_value(targets, status).reindex(
        targets["hadm_id"]).fillna("<none>"), top=10)
    report["m2_flags"] = m2

    # ---- M3/M7: emar, pharmacy, prescriptions over every admission -------------------------
    _log("reading hosp/emar (furosemide, all admissions of the targets)")
    emar = _read_filtered(root, "hosp", "emar",
                          ["subject_id", "hadm_id", "emar_id", "pharmacy_id", "charttime",
                           "storetime", "medication", "event_txt"],
                          lambda c: c["subject_id"].isin(subjects)
                          & c["medication"].str.contains("furosemide", case=False))
    emar = emar[administered(emar["event_txt"])]
    _log("reading hosp/pharmacy (route, frequency, orders)")
    pharmacy_ids = set(emar["pharmacy_id"]) - {""}
    pharmacy = _read_filtered(
        root, "hosp", "pharmacy",
        ["pharmacy_id", "hadm_id", "medication", "route", "frequency", "entertime"],
        lambda c: c["pharmacy_id"].isin(pharmacy_ids)
        | (c["hadm_id"].isin(hadms) & c["medication"].str.contains("furosemide", case=False)))
    route_of = pharmacy.drop_duplicates("pharmacy_id").set_index("pharmacy_id")["route"]
    no_detail = pd.DataFrame(columns=["emar_id", "route", "product_description"])
    emar = classify_route(emar, route_of, no_detail).assign(
        charttime=_ts(emar["charttime"]), storetime=_ts(emar["storetime"]))
    emar["oral"] = emar["route"].str.strip().str.upper().isin(ORAL_ROUTES)

    _log("reading hosp/prescriptions (furosemide, all admissions of the targets)")
    rx = _read_filtered(root, "hosp", "prescriptions",
                        ["subject_id", "hadm_id", "pharmacy_id", "drug", "starttime",
                         "stoptime", "dose_val_rx", "dose_unit_rx", "route"],
                        lambda c: c["subject_id"].isin(subjects)
                        & c["drug"].str.contains("furosemide", case=False))
    rx = rx.assign(starttime=_ts(rx["starttime"]), stoptime=_ts(rx["stoptime"]),
                   mg=_num(rx["dose_val_rx"]).where(
                       rx["dose_unit_rx"].str.strip().str.lower().eq("mg")))

    pairs = prior_pairs(targets, admissions)
    events = pd.concat([
        ie.assign(source="icu_inputevents", iv=True)[["hadm_id", "source", "iv"]],
        emar.assign(source="emar")[["hadm_id", "source", "iv"]],
    ], ignore_index=True).rename(columns={"hadm_id": "prior_hadm"})
    prior_ev = pairs[["hadm_id", "prior_hadm"]].merge(events, on="prior_hadm")
    prior_rx = pairs[["hadm_id", "prior_hadm"]].merge(
        rx[["hadm_id"]].rename(columns={"hadm_id": "prior_hadm"}), on="prior_hadm")
    n_prior = pairs.groupby("hadm_id")["prior_hadm"].nunique().reindex(
        targets["hadm_id"]).fillna(0)
    last_prior = pairs.sort_values("prior_dischtime").groupby("hadm_id").last()
    gap_days = ((last_prior["admittime"] - last_prior["prior_dischtime"])
                .dt.total_seconds() / 86400)
    per_target = prior_ev.groupby("hadm_id").size().reindex(targets["hadm_id"]).fillna(0)
    report["m3_history"] = {
        "with_prior_admission": _share((n_prior > 0).sum(), n),
        "prior_admissions": _counts(pd.cut(n_prior, [-1, 0, 1, 4, 9, np.inf],
                                           labels=["0", "1", "2-4", "5-9", "10+"])),
        "days_since_last_discharge": _quantiles(gap_days),
        "prior_furosemide_any": _share(len(set(prior_ev["hadm_id"]) | set(prior_rx["hadm_id"])), n),
        "prior_furosemide_administered": _share(prior_ev["hadm_id"].nunique(), n),
        "prior_iv_administered": _share(prior_ev.loc[prior_ev["iv"], "hadm_id"].nunique(), n),
        "prior_by_source": {s: _share(g["hadm_id"].nunique(), n)
                            for s, g in prior_ev.groupby("source")},
        "prior_prescription_only": _share(len(set(prior_rx["hadm_id"]) - set(prior_ev["hadm_id"])), n),
        "prior_administrations_per_target": _quantiles(per_target[per_target > 0]),
    }
    report["a3_prior_exposure"] = _counts(prior_exposure(targets, prior_ev))

    # M7: latest oral furosemide on record
    oral_rx = rx[rx["route"].str.strip().str.upper().isin(ORAL_ROUTES)]
    at_discharge = pairs.merge(oral_rx.rename(columns={"hadm_id": "prior_hadm"}), on="prior_hadm")
    at_discharge = at_discharge[at_discharge["stoptime"].isna()
                                | (at_discharge["stoptime"] >= at_discharge["prior_dischtime"]
                                   - pd.Timedelta(hours=DISCHARGE_ACTIVE_H))]
    oral_now = in_window(targets, emar[emar["oral"] & emar["hadm_id"].isin(hadms)]
                         .drop(columns="subject_id"), None)
    has_dc, has_now = (targets["hadm_id"].isin(set(at_discharge["hadm_id"])),
                       targets["hadm_id"].isin(set(oral_now["hadm_id"])))
    report["m7_oral_on_record"] = {
        "prior_discharge_rx": _share(has_dc.sum(), n),
        "oral_given_this_admission_before_t0": _share(has_now.sum(), n),
        "either": _share((has_dc | has_now).sum(), n),
        "neither": _share((~(has_dc | has_now)).sum(), n),
        "prior_discharge_mg": _quantiles(at_discharge.sort_values("prior_dischtime")
                                         .groupby("hadm_id")["mg"].last()),
    }

    # M8: order -> administration gap
    ph_iv = pharmacy[pharmacy["hadm_id"].isin(hadms)
                     & pharmacy["medication"].str.contains("furosemide", case=False)
                     & pharmacy["route"].str.contains(IV_ROUTE)]
    mg_of = rx.drop_duplicates("pharmacy_id").set_index("pharmacy_id")["mg"]
    orders = pd.DataFrame({"hadm_id": ph_iv["hadm_id"], "visible_time": _ts(ph_iv["entertime"]),
                           "order_mg": ph_iv["pharmacy_id"].map(mg_of),
                           "frequency": ph_iv["frequency"]})
    match = matching_order(targets, orders)
    report["m8_order_gap"] = {
        "targets_with_matching_order": _share(len(match), n),
        "gap_hours": _quantiles(match["gap_h"]),
        "gap_bins_h": _counts(pd.cut(match["gap_h"], list(GAP_BINS_H))),
        "frequency": _counts(match["frequency"].replace("", "<blank>"), top=10),
    }

    # ---- M5/M6: CKD tightening and baseline eGFR ------------------------------------------
    _log("reading hosp/omr, diagnoses_icd")
    creat = labs[labs["feature"].eq("creatinine")]
    who = patients.loc[creat["subject_id"]]
    age = who["anchor_age"].astype(float).to_numpy() + (
        creat["charttime"].dt.year.to_numpy() - who["anchor_year"].astype(float).to_numpy())
    creat = creat.assign(egfr=egfr_ckd_epi_2021(
        creat["valuenum"], pd.Series(age, index=creat.index),
        pd.Series(who["gender"].eq("F").to_numpy(), index=creat.index)))
    outpatient = creat[creat["hadm_id"].eq("")][["subject_id", "charttime", "egfr"]].rename(
        columns={"charttime": "time"})
    omr = read_table(root, "hosp", "omr", ["subject_id", "chartdate", "result_name",
                                           "result_value"])
    omr = omr[omr["subject_id"].isin(subjects) & omr["result_name"].str.contains("egfr", case=False)]
    omr_egfr = _num(omr["result_value"].str.extract(r"^\s*>?\s*(\d+(?:\.\d+)?)\s*$")[0])
    omr_r = pd.DataFrame({"subject_id": omr["subject_id"], "time": _ts(omr["chartdate"]),
                          "egfr": omr_egfr}).dropna()
    readings = pd.concat([outpatient, omr_r], ignore_index=True).dropna()

    dx = read_table(root, "hosp", "diagnoses_icd", ["subject_id", "hadm_id", "icd_code",
                                                     "icd_version"])
    dx = dx[dx["subject_id"].isin(subjects)]
    ckd_hadm = set(dx.loc[prefix_flag(dx, CKD_PREFIXES), "hadm_id"])
    htn_hadm = set(dx.loc[prefix_flag(dx, HTN_CKD_PREFIXES), "hadm_id"])
    a = targets["hadm_id"].isin(set(pairs.loc[pairs["prior_hadm"].isin(ckd_hadm), "hadm_id"]))
    a_htn = targets["hadm_id"].isin(set(pairs.loc[pairs["prior_hadm"].isin(htn_hadm), "hadm_id"]))
    b = targets["hadm_id"].isin(chronic_low_strict(targets, readings))
    c = targets["hadm_id"].isin(ckd_hadm)
    strict = np.select([a & b, a, b, c], ["confirmed_a_and_b", "confirmed_a", "confirmed_b",
                                          "index_only"], default="no_evidence")
    report["m5_ckd"] = {
        "tier_probe1": _counts(targets["ckd_tier"]),
        "tier_strict_b": _counts(pd.Series(strict)),
        "crosstab_probe1_x_strict": pd.crosstab(targets["ckd_tier"].to_numpy(),
                                                strict).to_dict("index"),
        "prior_htn_ckd_code_without_n18_585": _share((a_htn & ~a).sum(), n),
        "of_which_currently_index_only": _share((a_htn & ~a & ~b).sum(), n),
    }

    base = baseline_egfr(targets, readings).reindex(targets["hadm_id"])
    t0_creat = in_window(targets, target_labs[target_labs["feature"].eq("creatinine")],
                         LAB_WINDOW_H).sort_values("charttime").groupby("hadm_id")["valuenum"].last()
    t0_creat = t0_creat.reindex(targets["hadm_id"])
    pt = patients.loc[targets["subject_id"]]
    t0_age = pt["anchor_age"].astype(float).to_numpy() + (
        targets["t0"].dt.year.to_numpy() - pt["anchor_year"].astype(float).to_numpy())
    egfr_t0 = egfr_ckd_epi_2021(pd.Series(t0_creat.to_numpy()), pd.Series(t0_age),
                                pd.Series(pt["gender"].eq("F").to_numpy()))
    base_stage = egfr_stage(pd.Series(base.to_numpy()))
    t0_stage = egfr_stage(egfr_t0)
    report["m6_baseline_egfr"] = {
        "with_baseline_within_365d": _share(base.notna().sum(), n),
        "baseline_stage": _counts(base_stage),
        "t0_stage": _counts(t0_stage),
        "crosstab_baseline_x_t0": pd.crosstab(base_stage, t0_stage).to_dict("index"),
    }

    # ---- B0 ----------------------------------------------------------------------------------
    dose = targets["dose_mg"].round()
    report["b0_targets"] = {"dose": _counts(dose.astype(int)),
                            "baselines": constant_baselines(dose)}

    _print(report)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _log(f"wrote {out}")
    return 0


def _print(report: dict[str, Any]) -> None:
    def block(title: str, value: Any, indent: int = 2) -> None:
        pad = " " * indent
        if isinstance(value, dict) and "p50" in value:
            print(f"{pad}{title:44s} " + "  ".join(f"{k} {v:,}" for k, v in value.items()))
        elif isinstance(value, dict) and not ({"n", "pct"} >= set(value) and "n" in value):
            print(f"{pad}{title}:")
            for k, v in value.items():
                block(str(k), v, indent + 2)
        elif isinstance(value, dict):
            print(f"{pad}{title:44s} {value['n']:>7,}  {value.get('pct', ''):>6}%")
        else:
            print(f"{pad}{title:44s} {value}")

    print(f"\n=== targets: {report['targets']:,} (from {report['s9_admissions']:,} S9 admissions) ===")
    for key, value in report.items():
        if key not in ("targets", "s9_admissions"):
            block(key, value)


if __name__ == "__main__":
    sys.exit(main())
