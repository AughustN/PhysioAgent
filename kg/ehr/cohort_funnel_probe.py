#!/usr/bin/env python
"""Probe 1: the cohort funnel, counted on the full MIMIC-IV dump before any builder exists.

Each step keeps one admission per row and reports how many admissions and patients it
keeps and drops:

    S1  furosemide IV bolus/push events (inputevents 221794, mg, amount > 0)
    S2  first ICU bolus of each admission = t0          (unit switches: event -> admission)
    S3  t0 dose on the ladder        -- off-ladder first doses drop the admission; the next
                                        bolus is NOT promoted, it already saw a response
    S4  age >= 18 at t0
    S5  no furosemide infusion running at t0
    S6  some CKD evidence: (a) N18/585 on an earlier admission, (b) eGFR < 60 twice
        >= 90 days apart before admission, or (c) N18/585 on this admission only
    S7  no dialysis (CRRT/HD/PD) in the 24 h before t0
    S8  not dialysis-dependent (Z99.2/V45.11/N18.6/585.6) -> those go to the 5D stratum
    S9  a creatinine RESULTED (storetime) in [t0 - 48 h, t0)        = main cohort
    S10 patients, the unit the 70/15/15 split is drawn over

Annotations on the S9 cohort are counted, never filtered on: CKD tier, emar group (first
in admission / escalation after ward IV / unverifiable), protocol context (transfusion,
albumin, post cardiac surgery), via ED, care unit, anchor_year_group, dose distribution.

Only aggregate counts are written to the report. --save-cohort additionally writes the S9
cohort (patient-level) under kg/data/, which is gitignored; it feeds cohort_feature_probe and
must stay on the server.

    python -m physioagent.kg.ehr.cohort_funnel_probe --mimic-root <mimic-iv-3.1> [--save-cohort]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from physioagent.kg import paths
from physioagent.kg.ehr.build_visits import read_table, resolve_mimic_root

FUROSEMIDE_BOLUS_ITEMID = 221794
FUROSEMIDE_ITEMIDS = {221794, 228340}
BOLUS_CATEGORIES = {"Drug Push", "Bolus"}
PRBC_ITEMIDS = {225168}
ALBUMIN_ITEMIDS = {220862, 220864}
INPUT_ITEMIDS = FUROSEMIDE_ITEMIDS | PRBC_ITEMIDS | ALBUMIN_ITEMIDS

DOSE_LADDER = (10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 200.0)
DOSE_EPS = 1e-3

CREATININE_ITEMID = "50912"
DIALYSIS_ITEMIDS = {225802: "crrt", 225803: "cvvhd", 225809: "cvvhdf", 225955: "scuf",
                    225441: "hd", 225805: "pd"}

CKD_PREFIXES = {"10": ("N18",), "9": ("585",)}
ESRD_CODES = {"N186", "5856"}
DIALYSIS_STATUS_CODES = {"Z992", "V4511"}
TRANSPLANT_CODES = {"Z940", "V420"}
CARDIAC_SURGERY_PREFIXES = {"9": ("361", "351", "352"), "10": ("021", "02R", "02Q")}

MIN_AGE = 18
EGFR_CKD = 60.0
CHRONIC_DAYS = 90
REQUIRED_CREAT_H = 48
CREAT_WINDOWS_H = (24, 48, 72, 168)
DIALYSIS_LOOKBACK_H = 24
PROTOCOL_BEFORE_H = 2
PROTOCOL_AFTER_H = 1
CHUNK = 2_000_000

# emar_detail.route never says IV for furosemide (only PO/NG on parent rows); the IV route
# lives in pharmacy.route, and the product name ("... vial") backs it up without a route.
IV_ROUTE = re.compile(r"^\s*IV\b", re.IGNORECASE)
INJECTABLE_PRODUCT = re.compile(r"vial|syringe|dextrose|\bml\b", re.IGNORECASE)
INHALED = re.compile(r"inhal|\bneb", re.IGNORECASE)
INHALED_ROUTES = {"IH", "NEB"}
# An IV furosemide order entered just before t0 may be the order for the t0 dose itself.
ORDER_WINDOWS_H = (1, 6, 24)

# Patient-level S9 cohort handed to cohort_feature_probe; kg/data is gitignored (DUA).
COHORT_FILE = paths.EHR / "probe_cohort.csv.gz"
COHORT_COLUMNS = ["subject_id", "hadm_id", "stay_id", "t0", "dose_mg", "admittime", "intime",
                  "gender", "anchor_age", "anchor_year", "anchor_year_group", "ckd_tier",
                  "emar_group", "protocol_any"]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _code(series: pd.Series) -> pd.Series:
    return series.str.replace(".", "", regex=False).str.replace(" ", "", regex=False).str.upper()


def _read_filtered(root: Path, module: str, name: str, usecols: list[str],
                   keep: Callable[[pd.DataFrame], pd.Series]) -> pd.DataFrame:
    parts = [chunk[keep(chunk)] for chunk in read_table(root, module, name, usecols,
                                                        chunksize=CHUNK)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=usecols)


# --- S1..S5: from events to one t0 per admission ------------------------------------------

def split_inputevents(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    df = df[df["statusdescription"].ne("Rewritten")]
    furo = df["itemid"].isin(FUROSEMIDE_ITEMIDS)
    bolus_cat = df["ordercategorydescription"].isin(BOLUS_CATEGORIES)
    bolus = df[df["itemid"].eq(FUROSEMIDE_BOLUS_ITEMID) & bolus_cat
               & df["amountuom"].eq("mg") & df["amount"].gt(0)]
    return {"bolus": bolus,
            "infusion": df[furo & ~bolus_cat],
            "prbc": df[df["itemid"].isin(PRBC_ITEMIDS)],
            "albumin": df[df["itemid"].isin(ALBUMIN_ITEMIDS)]}


def first_bolus(bolus: pd.DataFrame) -> pd.DataFrame:
    ordered = bolus.sort_values(["hadm_id", "starttime", "amount"])
    first = ordered.groupby("hadm_id", as_index=False).first()
    return first.rename(columns={"starttime": "t0", "amount": "dose_mg"})[
        ["subject_id", "hadm_id", "stay_id", "t0", "dose_mg"]]


def on_ladder(dose: pd.Series) -> pd.Series:
    values = dose.to_numpy(dtype=float)
    gap = np.abs(values[:, None] - np.asarray(DOSE_LADDER)[None, :]).min(axis=1)
    return pd.Series(gap < DOSE_EPS, index=dose.index)


def age_at_t0(cohort: pd.DataFrame, patients: pd.DataFrame) -> pd.Series:
    merged = cohort[["subject_id", "t0"]].merge(
        patients[["subject_id", "anchor_age", "anchor_year"]], on="subject_id", how="left")
    age = merged["anchor_age"] + (merged["t0"].dt.year - merged["anchor_year"])
    return pd.Series(age.to_numpy(), index=cohort.index)


def hadms_with_event(cohort: pd.DataFrame, events: pd.DataFrame, on: str,
                     when: Callable[[pd.DataFrame], pd.Series]) -> set:
    """hadm_ids of cohort rows with at least one joined event satisfying when(row)."""
    if events.empty:
        return set()
    joined = cohort[["hadm_id", "subject_id", "t0"] + (["intime"] if "intime" in cohort else [])
                    ].merge(events, on=on, how="inner", suffixes=("", "_ev"))
    return set(joined.loc[when(joined), "hadm_id"])


def infusion_running(cohort: pd.DataFrame, infusion: pd.DataFrame) -> set:
    ev = infusion[["hadm_id", "starttime", "endtime"]]
    return hadms_with_event(cohort, ev, "hadm_id",
                            lambda j: (j["starttime"] <= j["t0"])
                            & (j["endtime"].isna() | (j["endtime"] > j["t0"])))


# --- S6: CKD evidence ---------------------------------------------------------------------

def code_flags(dx: pd.DataFrame) -> pd.DataFrame:
    code, version = _code(dx["icd_code"]), dx["icd_version"].astype(str)
    ckd = pd.Series(False, index=dx.index)
    for ver, prefixes in CKD_PREFIXES.items():
        ckd |= version.eq(ver) & code.str.startswith(prefixes)
    flags = pd.DataFrame({"hadm_id": dx["hadm_id"], "ckd": ckd,
                          "esrd": code.isin(ESRD_CODES),
                          "dialysis_status": code.isin(DIALYSIS_STATUS_CODES),
                          "transplant": code.isin(TRANSPLANT_CODES)})
    return flags.groupby("hadm_id").any()


def prior_flags(cohort: pd.DataFrame, admissions: pd.DataFrame,
                flags: pd.DataFrame) -> pd.DataFrame:
    """Any flag on an admission of the same patient that started before this one."""
    index = cohort[["hadm_id", "subject_id", "admittime"]]
    others = admissions[["subject_id", "hadm_id", "admittime"]].rename(
        columns={"hadm_id": "prior_hadm", "admittime": "prior_admittime"})
    pairs = index.merge(others, on="subject_id")
    pairs = pairs[pairs["prior_admittime"] < pairs["admittime"]]
    pairs = pairs.merge(flags, left_on="prior_hadm", right_index=True, how="inner")
    out = pairs.groupby("hadm_id")[list(flags.columns)].any()
    return out.reindex(cohort["hadm_id"], fill_value=False).astype(bool)


def egfr_ckd_epi_2021(scr: pd.Series, age: pd.Series, female: pd.Series) -> pd.Series:
    scr, age = scr.astype(float), age.astype(float)
    k = np.where(female, 0.7, 0.9)
    a = np.where(female, -0.241, -0.302)
    ratio = scr / k
    egfr = (142 * np.minimum(ratio, 1) ** a * np.maximum(ratio, 1) ** -1.200
            * 0.9938 ** age * np.where(female, 1.012, 1.0))
    return pd.Series(np.where(scr > 0, egfr, np.nan), index=scr.index)


def chronic_low_egfr(cohort: pd.DataFrame, low: pd.DataFrame) -> set:
    """hadm_ids with two eGFR < 60 readings >= CHRONIC_DAYS apart, both before admission.
    `low` holds subject_id + time of readings already known to be < 60."""
    joined = cohort[["hadm_id", "subject_id", "admittime"]].merge(low, on="subject_id")
    joined = joined[joined["time"] < joined["admittime"]]
    span = joined.groupby("hadm_id")["time"].agg(["min", "max"])
    return set(span.index[(span["max"] - span["min"]) >= pd.Timedelta(days=CHRONIC_DAYS)])


def latest_creatinine(cohort: pd.DataFrame, creat: pd.DataFrame, hours: float,
                      by: str = "storetime") -> pd.DataFrame:
    """Latest creatinine per admission known before t0. by='storetime' requires the RESULT
    to exist before t0; by='charttime' only the draw, which leaks results not yet back."""
    joined = cohort[["hadm_id", "subject_id", "t0"]].merge(creat, on="subject_id")
    known = (joined[by] < joined["t0"]) & (joined["charttime"] >= joined["t0"]
                                           - pd.Timedelta(hours=hours))
    joined = joined[known].sort_values("charttime")
    return joined.groupby("hadm_id").last()[["charttime", "storetime", "valuenum"]]


def dialysis_at_t0(cohort: pd.DataFrame, dialysis: pd.DataFrame) -> set:
    lookback = pd.Timedelta(hours=DIALYSIS_LOOKBACK_H)
    return hadms_with_event(cohort, dialysis[["subject_id", "starttime", "endtime"]],
                            "subject_id",
                            lambda j: (j["starttime"] < j["t0"])
                            & (j["endtime"].isna() | (j["endtime"] >= j["t0"] - lookback)))


# --- annotations ----------------------------------------------------------------------------

def near_t0(cohort: pd.DataFrame, events: pd.DataFrame) -> set:
    before = pd.Timedelta(hours=PROTOCOL_BEFORE_H)
    after = pd.Timedelta(hours=PROTOCOL_AFTER_H)
    return hadms_with_event(cohort, events[["hadm_id", "starttime"]], "hadm_id",
                            lambda j: (j["starttime"] >= j["t0"] - before)
                            & (j["starttime"] <= j["t0"] + after))


def administered(event_txt: pd.Series) -> pd.Series:
    """emar rows where the dose was given ("Administered", not "Not Given" variants)."""
    return (event_txt.str.contains("Administered", case=False)
            & ~event_txt.str.contains(r"\bnot\b", case=False))


def classify_route(furo: pd.DataFrame, pharmacy_route: pd.Series,
                   detail: pd.DataFrame) -> pd.DataFrame:
    """Route per emar row from pharmacy, then emar_detail, then the product name."""
    d_route = (detail[detail["route"].str.strip().ne("")]
               .drop_duplicates("emar_id").set_index("emar_id")["route"])
    named = detail[detail["product_description"].str.strip().ne("")]
    product = named.groupby("emar_id")["product_description"].agg(" | ".join)
    ph = furo["pharmacy_id"].map(pharmacy_route).fillna("").str.strip()
    dr = furo["emar_id"].map(d_route).fillna("").str.strip()
    pr = furo["emar_id"].map(product).fillna("")
    route = ph.where(ph.ne(""), dr)
    source = np.select([ph.ne(""), dr.ne(""), pr.ne("")],
                       ["pharmacy", "emar_detail", "product"], default="none")
    inhaled = pr.str.contains(INHALED) | route.str.upper().isin(INHALED_ROUTES)
    iv = (route.str.contains(IV_ROUTE)
          | (route.eq("") & pr.str.contains(INJECTABLE_PRODUCT))) & ~inhaled
    return furo.assign(route=route, route_source=source, iv=iv)


def order_leak(cohort: pd.DataFrame, orders: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Per window: admissions with an IV order visible before t0, and how often the latest
    one carries exactly the t0 dose."""
    j = cohort[["hadm_id", "t0", "dose_mg"]].merge(orders, on="hadm_id")
    j = j[j["visible_time"] < j["t0"]].sort_values("visible_time")
    out = {}
    for h in ORDER_WINDOWS_H:
        w = j[j["visible_time"] >= j["t0"] - pd.Timedelta(hours=h)]
        last = w.drop_duplicates("hadm_id", keep="last")
        same = (last["order_mg"] - last["dose_mg"]).abs() < DOSE_EPS
        out[f"within_{h}h"] = {"admissions_with_iv_order": int(len(last)),
                               "last_order_dose_known": int(last["order_mg"].notna().sum()),
                               "last_order_equals_label": int(same.sum())}
    return out


def emar_group(cohort: pd.DataFrame, observable: set, ward_iv: pd.DataFrame) -> pd.Series:
    before_icu = hadms_with_event(cohort, ward_iv[["hadm_id", "charttime"]], "hadm_id",
                                  lambda j: j["charttime"] < j["intime"])
    group = np.where(~cohort["hadm_id"].isin(observable), "unverifiable",
                     np.where(cohort["hadm_id"].isin(before_icu), "escalation",
                              "first_in_admission"))
    return pd.Series(group, index=cohort.index)


def _counts(series: pd.Series, top: Optional[int] = None) -> dict[str, Any]:
    vc = series.astype(str).value_counts()
    if top:
        vc = vc.head(top)
    total = len(series)
    return {k: {"n": int(v), "pct": round(100.0 * v / total, 2) if total else 0.0}
            for k, v in vc.items()}


class Funnel:
    def __init__(self) -> None:
        self.steps: list[dict[str, Any]] = []

    def record(self, step: str, label: str, df: pd.DataFrame, unit: str = "admissions") -> None:
        n = int(df["subject_id"].nunique()) if unit == "patients" else len(df)
        prev = self.steps[-1]["n"] if self.steps and self.steps[-1]["unit"] == unit else None
        row = {"step": step, "label": label, "unit": unit, "n": n,
               "patients": int(df["subject_id"].nunique()),
               "dropped": None if prev is None else prev - n}
        self.steps.append(row)
        drop = "" if row["dropped"] is None else f"  (-{row['dropped']:,})"
        print(f"  {step:4s} {label:58s} {n:>9,} {unit:10s} {row['patients']:>8,} pts{drop}",
              flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mimic-root", help="directory holding hosp/ and icu/ (default $MIMIC_ROOT)")
    ap.add_argument("--out", default=str(paths.EHR / "cohort_funnel_probe.json"))
    ap.add_argument("--save-cohort", metavar="PATH", nargs="?", const=str(COHORT_FILE),
                    help="also write the S9 cohort (patient-level, stays on the server) for "
                         f"cohort_feature_probe; default path {COHORT_FILE}")
    args = ap.parse_args()
    root = resolve_mimic_root(args.mimic_root)
    _log(f"mimic root {root}")
    funnel = Funnel()

    # ---- S0-S1 ---------------------------------------------------------------------------
    _log("reading icu/inputevents (furosemide, PRBC, albumin)")
    wanted = {str(i) for i in INPUT_ITEMIDS}
    ie = _read_filtered(root, "icu", "inputevents",
                        ["subject_id", "hadm_id", "stay_id", "starttime", "endtime", "itemid",
                         "amount", "amountuom", "ordercategorydescription",
                         "statusdescription"],
                        lambda c: c["itemid"].isin(wanted))
    ie = ie.assign(itemid=_num(ie["itemid"]).astype(int), amount=_num(ie["amount"]),
                   starttime=_ts(ie["starttime"]), endtime=_ts(ie["endtime"]))
    parts = split_inputevents(ie)
    bolus = parts["bolus"]
    print(f"\n  S0   furosemide inputevents rows (221794 + 228340)                "
          f"{int(ie['itemid'].isin(FUROSEMIDE_ITEMIDS).sum()):>9,}")
    funnel.record("S1", "IV bolus/push, mg, amount > 0 (events)", bolus, unit="events")

    # ---- S2-S5 ---------------------------------------------------------------------------
    cohort = first_bolus(bolus)
    funnel.record("S2", "first ICU bolus per admission = t0", cohort)
    cohort = cohort[on_ladder(cohort["dose_mg"])]
    funnel.record("S3", "t0 dose on the ladder", cohort)

    _log("reading hosp/patients, admissions, icu/icustays")
    patients = read_table(root, "hosp", "patients",
                          ["subject_id", "gender", "anchor_age", "anchor_year",
                           "anchor_year_group"])
    patients = patients.assign(anchor_age=_num(patients["anchor_age"]),
                               anchor_year=_num(patients["anchor_year"]))
    admissions = read_table(root, "hosp", "admissions",
                            ["subject_id", "hadm_id", "admittime", "edregtime",
                             "admission_type"])
    admissions = admissions.assign(admittime=_ts(admissions["admittime"]))
    icustays = read_table(root, "icu", "icustays",
                          ["stay_id", "intime", "first_careunit"])
    icustays = icustays.assign(intime=_ts(icustays["intime"]))

    cohort = cohort[age_at_t0(cohort, patients) >= MIN_AGE]
    funnel.record("S4", f"age >= {MIN_AGE} at t0", cohort)
    cohort = cohort[~cohort["hadm_id"].isin(infusion_running(cohort, parts["infusion"]))]
    funnel.record("S5", "no furosemide infusion running at t0", cohort)

    cohort = (cohort.merge(admissions.drop(columns="subject_id"), on="hadm_id", how="left")
                    .merge(icustays, on="stay_id", how="left")
                    .merge(patients[["subject_id", "gender", "anchor_age", "anchor_year",
                                     "anchor_year_group"]], on="subject_id", how="left"))
    subjects = set(cohort["subject_id"])

    # ---- S6: CKD -------------------------------------------------------------------------
    _log("reading hosp/diagnoses_icd")
    dx = read_table(root, "hosp", "diagnoses_icd", ["subject_id", "hadm_id", "icd_code",
                                                     "icd_version"])
    dx = dx[dx["subject_id"].isin(subjects)]
    flags = code_flags(dx)
    prior = prior_flags(cohort, admissions[admissions["subject_id"].isin(subjects)], flags)
    index = flags.reindex(cohort["hadm_id"], fill_value=False).astype(bool)

    _log("reading hosp/labevents creatinine (largest table; this is the slow step)")
    creat = _read_filtered(root, "hosp", "labevents",
                           ["subject_id", "itemid", "charttime", "storetime", "valuenum"],
                           lambda c: c["itemid"].eq(CREATININE_ITEMID)
                           & c["subject_id"].isin(subjects))
    creat = creat.assign(charttime=_ts(creat["charttime"]), storetime=_ts(creat["storetime"]),
                         valuenum=_num(creat["valuenum"])).dropna(subset=["valuenum"])
    demo = patients.set_index("subject_id")
    who = demo.loc[creat["subject_id"]]
    creat_age = who["anchor_age"].to_numpy() + (creat["charttime"].dt.year.to_numpy()
                                                - who["anchor_year"].to_numpy())
    creat["egfr"] = egfr_ckd_epi_2021(creat["valuenum"], pd.Series(creat_age, index=creat.index),
                                      pd.Series(who["gender"].eq("F").to_numpy(),
                                                index=creat.index))
    low = creat.loc[creat["egfr"] < EGFR_CKD, ["subject_id", "charttime"]].rename(
        columns={"charttime": "time"})

    _log("reading hosp/omr (outpatient eGFR)")
    omr = read_table(root, "hosp", "omr", ["subject_id", "chartdate", "result_name",
                                           "result_value"])
    omr = omr[omr["subject_id"].isin(subjects)
              & omr["result_name"].str.contains("egfr", case=False)]
    omr_value = _num(omr["result_value"].str.extract(r"^\s*(\d+(?:\.\d+)?)\s*$")[0])
    omr_low = omr.loc[omr_value < EGFR_CKD, ["subject_id", "chartdate"]].rename(
        columns={"chartdate": "time"})
    omr_low = omr_low.assign(time=_ts(omr_low["time"]))
    chronic = chronic_low_egfr(cohort, pd.concat([low, omr_low], ignore_index=True))

    latest = {h: latest_creatinine(cohort, creat, h) for h in CREAT_WINDOWS_H}
    by_chart = latest_creatinine(cohort, creat, REQUIRED_CREAT_H, by="charttime")
    t0_creat = latest[REQUIRED_CREAT_H]
    sex_f = cohort.set_index("hadm_id")["gender"].eq("F")
    age_t0 = (cohort.set_index("hadm_id")["anchor_age"]
              + cohort.set_index("hadm_id")["t0"].dt.year
              - cohort.set_index("hadm_id")["anchor_year"])
    hadm_t0 = t0_creat.index
    egfr_at_t0 = egfr_ckd_epi_2021(t0_creat["valuenum"], age_t0.reindex(hadm_t0),
                                   sex_f.reindex(hadm_t0))

    a = prior["ckd"].to_numpy()
    b = cohort["hadm_id"].isin(chronic).to_numpy()
    c = index["ckd"].to_numpy()
    e = cohort["hadm_id"].map(egfr_at_t0)
    tier = np.select(
        [a & b, a, b, c & e.lt(EGFR_CKD).to_numpy(), c & e.ge(EGFR_CKD).to_numpy(), c],
        ["confirmed_a_and_b", "confirmed_a", "confirmed_b", "index_consistent",
         "index_discordant", "index_no_recent_creatinine"], default="none")
    cohort = cohort.assign(ckd_tier=tier)
    s5_tiers = _counts(cohort["ckd_tier"])
    cohort = cohort[cohort["ckd_tier"].ne("none")]
    funnel.record("S6", "any CKD evidence (prior code / chronic eGFR / index code)", cohort)

    # ---- S7-S8: dialysis -----------------------------------------------------------------
    _log("reading icu/procedureevents (dialysis)")
    dial_ids = {str(i) for i in DIALYSIS_ITEMIDS}
    pe = _read_filtered(root, "icu", "procedureevents",
                        ["subject_id", "itemid", "starttime", "endtime"],
                        lambda c: c["itemid"].isin(dial_ids) & c["subject_id"].isin(subjects))
    pe = pe.assign(starttime=_ts(pe["starttime"]), endtime=_ts(pe["endtime"]))
    on_dialysis = dialysis_at_t0(cohort, pe)
    cohort = cohort[~cohort["hadm_id"].isin(on_dialysis)]
    funnel.record("S7", f"no dialysis in the {DIALYSIS_LOOKBACK_H} h before t0", cohort)

    prior_c = prior.reindex(cohort["hadm_id"])
    index_c = index.reindex(cohort["hadm_id"])
    dependent = ((prior_c["dialysis_status"] | prior_c["esrd"]
                  | index_c["dialysis_status"] | index_c["esrd"]).to_numpy())
    stratum_5d = cohort[dependent]
    stratum_detail = {
        "n": int(len(stratum_5d)),
        "prior_dialysis_status_code": int(prior_c["dialysis_status"].sum()),
        "prior_esrd_code": int(prior_c["esrd"].sum()),
        "index_only": int((dependent & ~(prior_c["dialysis_status"]
                                         | prior_c["esrd"]).to_numpy()).sum()),
        "dose": _counts(stratum_5d["dose_mg"].round().astype(int)),
    }
    cohort = cohort[~dependent]
    funnel.record("S8", "not dialysis-dependent (5D -> separate stratum)", cohort)

    creat_cover = {f"within_{h}h": int(cohort["hadm_id"].isin(latest[h].index).sum())
                   for h in CREAT_WINDOWS_H}
    leak_ids = set(by_chart.index) - set(t0_creat.index)
    creat_cover["drawn_within_48h_but_not_resulted_by_t0"] = int(
        cohort["hadm_id"].isin(leak_ids).sum())
    cohort = cohort[cohort["hadm_id"].isin(t0_creat.index)]
    funnel.record("S9", f"creatinine resulted in [t0 - {REQUIRED_CREAT_H} h, t0)  = MAIN", cohort)
    funnel.record("S10", "patients (split unit, 70/15/15)", cohort, unit="patients")

    # ---- annotations on S9 ---------------------------------------------------------------
    _log("reading hosp/emar (furosemide administrations, coverage)")
    hadms = set(cohort["hadm_id"])
    observable: set = set()
    furo_rows: list[pd.DataFrame] = []
    for chunk in read_table(root, "hosp", "emar", ["hadm_id", "emar_id", "pharmacy_id",
                                                   "charttime", "medication", "event_txt"],
                            chunksize=CHUNK):
        chunk = chunk[chunk["hadm_id"].isin(hadms)]
        observable |= set(chunk["hadm_id"])
        furo_rows.append(chunk[chunk["medication"].str.contains("furosemide", case=False)])
    furo_all = pd.concat(furo_rows, ignore_index=True)
    furo = furo_all[administered(furo_all["event_txt"])]
    _log("reading hosp/emar_detail (route, product)")
    furo_ids = set(furo["emar_id"])
    detail = _read_filtered(root, "hosp", "emar_detail",
                            ["emar_id", "route", "product_description"],
                            lambda c: c["emar_id"].isin(furo_ids))
    _log("reading hosp/pharmacy (route, furosemide orders)")
    pharmacy_ids = set(furo["pharmacy_id"]) - {""}
    pharmacy = _read_filtered(
        root, "hosp", "pharmacy", ["pharmacy_id", "hadm_id", "medication", "route", "entertime"],
        lambda c: c["pharmacy_id"].isin(pharmacy_ids)
        | (c["hadm_id"].isin(hadms) & c["medication"].str.contains("furosemide", case=False)))
    pharmacy_route = pharmacy.drop_duplicates("pharmacy_id").set_index("pharmacy_id")["route"]
    furo = classify_route(furo, pharmacy_route, detail).assign(charttime=_ts(furo["charttime"]))
    ward_iv = furo[furo["iv"]]
    cohort = cohort.assign(emar_group=emar_group(cohort, observable, ward_iv))

    _log("reading hosp/prescriptions (furosemide orders)")
    rx = _read_filtered(root, "hosp", "prescriptions",
                        ["pharmacy_id", "hadm_id", "drug", "starttime", "dose_val_rx",
                         "dose_unit_rx"],
                        lambda c: c["hadm_id"].isin(hadms)
                        & c["drug"].str.contains("furosemide", case=False))
    ph_orders = pharmacy[pharmacy["hadm_id"].isin(hadms)
                         & pharmacy["medication"].str.contains("furosemide", case=False)
                         & pharmacy["route"].str.contains(IV_ROUTE)]
    rx_mg = _num(rx["dose_val_rx"]).where(rx["dose_unit_rx"].str.strip().str.lower().eq("mg"))
    dose_by_pharmacy = rx.assign(order_mg=rx_mg).drop_duplicates("pharmacy_id").set_index(
        "pharmacy_id")["order_mg"]
    orders = pd.DataFrame({
        "hadm_id": ph_orders["hadm_id"],
        "visible_time": _ts(ph_orders["entertime"]),
        "order_mg": ph_orders["pharmacy_id"].map(dose_by_pharmacy),
    })
    leak = order_leak(cohort, orders)

    _log("reading hosp/services, procedures_icd")
    services = read_table(root, "hosp", "services", ["hadm_id", "transfertime", "curr_service"])
    services = services[services["hadm_id"].isin(hadms)].assign(
        transfertime=_ts(services["transfertime"]))
    svc = cohort[["hadm_id", "t0"]].merge(services, on="hadm_id")
    svc = svc[svc["transfertime"] <= svc["t0"]].sort_values("transfertime")
    service_at_t0 = svc.groupby("hadm_id")["curr_service"].last()
    px = read_table(root, "hosp", "procedures_icd", ["hadm_id", "chartdate", "icd_code",
                                                     "icd_version"])
    px = px[px["hadm_id"].isin(hadms)]
    pcode, pver = _code(px["icd_code"]), px["icd_version"].astype(str)
    cardiac = pd.Series(False, index=px.index)
    for ver, prefixes in CARDIAC_SURGERY_PREFIXES.items():
        cardiac |= pver.eq(ver) & pcode.str.startswith(prefixes)
    px = px[cardiac].assign(chartdate=_ts(px.loc[cardiac, "chartdate"]))
    pj = cohort[["hadm_id", "t0"]].merge(px, on="hadm_id")
    surgery_before = set(pj.loc[pj["chartdate"] <= pj["t0"].dt.normalize(), "hadm_id"])
    cohort = cohort.assign(
        service_at_t0=cohort["hadm_id"].map(service_at_t0).fillna("unknown"),
        protocol_transfusion=cohort["hadm_id"].isin(near_t0(cohort, parts["prbc"])),
        protocol_albumin=cohort["hadm_id"].isin(near_t0(cohort, parts["albumin"])),
        protocol_post_cardiac_surgery=cohort["hadm_id"].isin(surgery_before)
        & (cohort["hadm_id"].map(service_at_t0).eq("CSURG")
           | cohort["first_careunit"].str.contains("CVICU", na=False)),
        via_ed=cohort["edregtime"].ne(""),
    )
    cohort["protocol_any"] = cohort[["protocol_transfusion", "protocol_albumin",
                                     "protocol_post_cardiac_surgery"]].any(axis=1)
    cohort["transplant"] = (prior.reindex(cohort["hadm_id"])["transplant"].to_numpy()
                            | index.reindex(cohort["hadm_id"])["transplant"].to_numpy())

    dose = cohort["dose_mg"].round().astype(int)
    non_protocol = cohort[~cohort["protocol_any"]]
    report = {
        "funnel": funnel.steps,
        "ckd_tier_at_s5_before_filter": s5_tiers,
        "stratum_5d": stratum_detail,
        "creatinine_coverage_at_s8": creat_cover,
        "omr_egfr_rows_for_cohort_subjects": int(len(omr)),
        "emar_furosemide_event_txt": _counts(furo_all["event_txt"], top=15),
        "emar_furosemide_route_values": _counts(furo["route"], top=15),
        "emar_furosemide_route_source": _counts(furo["route_source"]),
        "emar_furosemide_iv": int(furo["iv"].sum()),
        "iv_order_before_t0": leak,
        "main_cohort": {
            "admissions": int(len(cohort)), "patients": int(cohort["subject_id"].nunique()),
            "patients_with_2plus_admissions": int((cohort["subject_id"].value_counts() > 1).sum()),
            "median_dose_mg": float(dose.median()),
            "dose": _counts(dose),
            "majority_dose_share_pct": round(100.0 * dose.value_counts().iloc[0] / len(dose), 2),
            "ckd_tier": _counts(cohort["ckd_tier"]),
            "emar_group": _counts(cohort["emar_group"]),
            "protocol_transfusion": int(cohort["protocol_transfusion"].sum()),
            "protocol_albumin": int(cohort["protocol_albumin"].sum()),
            "protocol_post_cardiac_surgery": int(cohort["protocol_post_cardiac_surgery"].sum()),
            "protocol_any": int(cohort["protocol_any"].sum()),
            "non_protocol_admissions": int(len(non_protocol)),
            "non_protocol_median_dose_mg": round(float(non_protocol["dose_mg"].median()), 1)
            if len(non_protocol) else None,
            "transplant": int(cohort["transplant"].sum()),
            "via_ed": int(cohort["via_ed"].sum()),
            "first_careunit": _counts(cohort["first_careunit"]),
            "service_at_t0": _counts(cohort["service_at_t0"], top=10),
            "admission_type": _counts(cohort["admission_type"]),
            "anchor_year_group": _counts(cohort["anchor_year_group"]),
            "crosstab_ckd_tier_x_emar_group": pd.crosstab(
                cohort["ckd_tier"], cohort["emar_group"]).to_dict(),
        },
    }

    m = report["main_cohort"]
    print(f"\n=== main cohort: {m['admissions']:,} admissions, {m['patients']:,} patients "
          f"({m['patients_with_2plus_admissions']:,} with >= 2), median dose "
          f"{m['median_dose_mg']:g} mg, majority dose share {m['majority_dose_share_pct']}% ===")
    for key in ("ckd_tier", "emar_group", "anchor_year_group", "first_careunit"):
        print(f"  {key}:")
        for value, row in m[key].items():
            print(f"    {value:32s} {row['n']:>7,}  {row['pct']:6.2f}%")
    print("  dose:")
    for value, row in m["dose"].items():
        print(f"    {value:>5s} mg {'':24s} {row['n']:>7,}  {row['pct']:6.2f}%")
    print(f"  protocol: transfusion {m['protocol_transfusion']:,} | albumin "
          f"{m['protocol_albumin']:,} | post cardiac surgery "
          f"{m['protocol_post_cardiac_surgery']:,} | any {m['protocol_any']:,} | non-protocol "
          f"{m['non_protocol_admissions']:,} (median {m['non_protocol_median_dose_mg']} mg)")
    print(f"  via ED {m['via_ed']:,} | transplant {m['transplant']:,}")
    print(f"\n  5D stratum: {stratum_detail['n']:,} admissions")
    print(f"  creatinine coverage at S8: {creat_cover}")
    print(f"  CKD tier before S6 filter: "
          + ", ".join(f"{k} {v['n']:,}" for k, v in s5_tiers.items()))
    print(f"  emar furosemide routes (top): "
          + ", ".join(f"{k or '<blank>'} {v['n']:,}"
                      for k, v in report["emar_furosemide_route_values"].items()))
    print(f"  emar route source: "
          + ", ".join(f"{k} {v['n']:,}" for k, v in report["emar_furosemide_route_source"].items())
          + f" | classified IV {report['emar_furosemide_iv']:,}")
    print(f"  crosstab ckd_tier x emar_group: {m['crosstab_ckd_tier_x_emar_group']}")
    print(f"  IV furosemide order visible before t0 (of {m['admissions']:,} admissions):")
    for window, row in leak.items():
        print(f"    {window:12s} with order {row['admissions_with_iv_order']:>6,} | dose known "
              f"{row['last_order_dose_known']:>6,} | latest order == label "
              f"{row['last_order_equals_label']:>6,}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _log(f"wrote {out}")
    if args.save_cohort:
        cohort_out = Path(args.save_cohort)
        cohort_out.parent.mkdir(parents=True, exist_ok=True)
        cohort[COHORT_COLUMNS].to_csv(cohort_out, index=False)
        _log(f"wrote {cohort_out} (patient-level: keep on the server, never commit or share)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
