#!/usr/bin/env python
"""Feature builder (docs/feature_schema.md): model inputs for each target of cohort_build.

Every value is known before t_dec: rows need storetime < t_dec where the table has one,
date-only fields a date before the date of t_dec, and windows count back from t_dec. Labs
join on subject_id (ED draws before admission carry no hadm_id). Current-admission
furosemide comes from administered records only (emar Administered, inputevents), never
from orders (spec 4.1); prior admissions add their discharge prescriptions.

Writes next to the cohort (gitignored, patient-level, DUA):
    features{tag}.csv.gz   one row per target, the columns of dataset.FEATURE_COLUMNS
    history{tag}.csv.gz    one row per furosemide event before t_dec, with its response

    python -m physioagent.kg.ehr.cohort_features --mimic-root <mimic-iv-3.1>
    python -m physioagent.kg.ehr.cohort_features --mimic-root <...> --tag _5d
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from physioagent.kg.ehr.build_visits import read_table, resolve_mimic_root
from physioagent.kg.ehr.cohort_build import FUROSEMIDE, OUT_DIR, egfr_readings
from physioagent.kg.ehr.cohort_feature_probe import (
    CODE_STATUS_ITEMID, IRRIGANT_IN_ITEMID, LABS, ORAL_ROUTES, URINE_ITEMIDS, WEIGHT_ITEMIDS,
    baseline_egfr, egfr_stage, in_window, prefix_flag, prior_pairs)
from physioagent.kg.ehr.cohort_funnel_probe import (
    BOLUS_CATEGORIES, FUROSEMIDE_BOLUS_ITEMID, FUROSEMIDE_ITEMIDS, _log, _num, _read_filtered,
    _ts, administered, classify_route, egfr_ckd_epi_2021)
from physioagent.kg.ehr.dataset import FEATURE_COLUMNS

VITALS = {"hr": (220045,), "sbp": (220179, 220050), "dbp": (220180, 220051),
          "map": (220181, 220052), "spo2": (220277,), "resp_rate": (220210,),
          "temp_c": (223762,), "temp_f": (223761,)}
VITAL_WINDOW_H = 6
LAB_WINDOW_H = 48
UO_WINDOW_H = (6, 24)
FLUID_WINDOW_H = 24
WARD_IV_TOTAL_H = 24
RESPONSE_UO_H = 6
RESPONSE_CREAT_H = 48
ESCALATION_H = 24
OMR_WEIGHT_DAYS = 365
DISCHARGE_ACTIVE_H = 24
LB_TO_KG = 0.453592

PRIOR_DX = {"prior_hf": {"10": ("I50",), "9": ("428",)},
            "prior_cirrhosis": {"10": ("K703", "K746"), "9": ("5712", "5715")},
            "prior_diabetes": {"10": ("E08", "E09", "E10", "E11", "E13"), "9": ("250",)}}

HISTORY_COLUMNS = ["subject_id", "hadm_id", "event_hadm_id", "scope", "source", "kind",
                   "hours_before_t_dec", "dose_mg", "rate_mg_h", "duration_h", "frequency",
                   "uo_6h_after_ml", "creatinine_before", "creatinine_after_48h",
                   "escalated_24h"]


def _hours(delta: pd.Series) -> pd.Series:
    return delta.dt.total_seconds() / 3600


def at_t_dec(targets: pd.DataFrame) -> pd.DataFrame:
    """in_window and friends cut at "t0"; the feature cutoff is t_dec."""
    return targets.assign(t0=targets["t_dec"])


def latest(rows: pd.DataFrame, value: str = "valuenum") -> pd.DataFrame:
    """Last value per (hadm_id, feature), wide."""
    if rows.empty:
        return pd.DataFrame()
    return rows.sort_values("charttime").groupby(["hadm_id", "feature"])[value].last().unstack()


def code_status(value: pd.Series) -> pd.Series:
    v = value.fillna("").str.lower()
    dnr, dni = v.str.contains("dnr|do not resuscitate"), v.str.contains("dni|do not intubate")
    return pd.Series(np.select([v.str.contains("comfort"), dnr & dni, dnr, dni,
                                v.str.contains("full")],
                               ["cmo", "dnr_dni", "dnr", "dni", "full"], default="unknown"),
                     index=value.index)


# --- event windows ----------------------------------------------------------------------------

def window_sum(points: pd.DataFrame, rows: pd.DataFrame, key: str, hours: float) -> pd.Series:
    """Sum of rows.value with rows.time in (t, t + hours] per point (key, time); NaN when no
    row falls in the window."""
    out = pd.Series(np.nan, index=points.index)
    p = points.loc[points["time"].notna(), [key, "time"]]
    r = rows[[key, "time", "value"]].dropna(subset=["time"]).sort_values([key, "time"],
                                                                         kind="stable")
    if p.empty or r.empty:
        return out
    r = r.assign(cum=r.groupby(key)["value"].cumsum(), cnt=r.groupby(key).cumcount() + 1)
    r = r.sort_values("time", kind="stable")[[key, "time", "cum", "cnt"]]

    def upto(when: pd.Series) -> pd.DataFrame:
        q = p.assign(time=when).reset_index().sort_values("time")
        m = pd.merge_asof(q, r, on="time", by=key, direction="backward")
        return m.set_index("index")[["cum", "cnt"]].reindex(p.index).fillna(0.0)

    lo, hi = upto(p["time"]), upto(p["time"] + pd.Timedelta(hours=hours))
    n = hi["cnt"] - lo["cnt"]
    out.loc[p.index] = np.where(n > 0, hi["cum"] - lo["cum"], np.nan)
    return out


def value_asof(points: pd.DataFrame, rows: pd.DataFrame, key: str, offset_h: float = 0.0,
               strict: bool = True) -> pd.DataFrame:
    """Latest rows.value at or before t + offset_h (before only when strict), with its time."""
    empty = pd.DataFrame({"value": np.nan, "found": pd.NaT}, index=points.index)
    p = points.loc[points["time"].notna(), [key, "time"]]
    r = rows[[key, "time", "value"]].dropna(subset=["time"]).sort_values("time", kind="stable")
    if p.empty or r.empty:
        return empty
    r = r.assign(found=r["time"])
    q = p.assign(time=p["time"] + pd.Timedelta(hours=offset_h)).reset_index().sort_values("time")
    m = pd.merge_asof(q, r, on="time", by=key, direction="backward",
                      allow_exact_matches=not strict)
    return m.set_index("index")[["value", "found"]].reindex(points.index)


def escalated(events: pd.DataFrame, hours: float = ESCALATION_H) -> pd.Series:
    """For each IV bolus: another IV bolus of the same patient with a larger dose in
    (t, t + hours]. Non-bolus rows get NaN."""
    out = pd.Series(np.nan, index=events.index)
    iv = events[events["kind"].eq("bolus") & events["time"].notna()].sort_values("time")
    span = np.timedelta64(int(hours * 3600), "s")
    for _, g in iv.groupby("subject_id"):
        t, d = g["time"].to_numpy(), g["dose_mg"].to_numpy(dtype=float)
        lo = np.searchsorted(t, t, side="right")
        hi = np.searchsorted(t, t + span, side="right")
        later = [np.nanmax(d[a:b]) if b > a else -np.inf for a, b in zip(lo, hi)]
        out.loc[g.index] = (np.asarray(later) > d).astype(float)
    return out


# --- furosemide events --------------------------------------------------------------------------

def icu_events(ie: pd.DataFrame) -> pd.DataFrame:
    bolus = (ie["itemid"].eq(str(FUROSEMIDE_BOLUS_ITEMID))
             & ie["ordercategorydescription"].isin(BOLUS_CATEGORIES)
             & ie["amountuom"].eq("mg") & ie["amount"].gt(0))
    infusion = ~ie["ordercategorydescription"].isin(BOLUS_CATEGORIES)
    ev = ie[bolus | infusion]
    mg_h = ev["rate"].where(ev["rateuom"].str.lower().eq("mg/hour"))
    return pd.DataFrame({
        "subject_id": ev["subject_id"], "event_hadm_id": ev["hadm_id"], "source": "icu_inputevents",
        "kind": np.where(bolus[ev.index], "bolus", "infusion"), "time": ev["starttime"],
        "storetime": ev["storetime"],
        "dose_mg": ev["amount"].where(ev["amountuom"].eq("mg")),
        "rate_mg_h": mg_h.where(~bolus[ev.index]),
        "duration_h": _hours(ev["endtime"] - ev["starttime"]).where(~bolus[ev.index]),
        "frequency": ""})


def emar_events(emar: pd.DataFrame) -> pd.DataFrame:
    oral = emar["route"].str.strip().str.upper().isin(ORAL_ROUTES)
    kind = np.select([emar["iv"], oral], ["bolus", "oral"], default="other")
    return pd.DataFrame({
        "subject_id": emar["subject_id"], "event_hadm_id": emar["hadm_id"], "source": "emar",
        "kind": kind, "time": emar["charttime"], "storetime": emar["storetime"],
        "dose_mg": emar["dose_mg"], "rate_mg_h": np.nan, "duration_h": np.nan, "frequency": ""})


def discharge_events(pairs: pd.DataFrame, rx: pd.DataFrame) -> pd.DataFrame:
    """Oral furosemide prescriptions still active at the discharge of a prior admission."""
    oral = rx[rx["route"].str.strip().str.upper().isin(ORAL_ROUTES)]
    dc = (pairs[["prior_hadm", "prior_dischtime"]].drop_duplicates()
          .merge(oral, left_on="prior_hadm", right_on="hadm_id"))
    dc = dc[dc["stoptime"].isna()
            | (dc["stoptime"] >= dc["prior_dischtime"] - pd.Timedelta(hours=DISCHARGE_ACTIVE_H))]
    dc = dc.sort_values("starttime").drop_duplicates("prior_hadm", keep="last")
    return pd.DataFrame({
        "subject_id": dc["subject_id"], "event_hadm_id": dc["prior_hadm"], "source": "discharge_rx",
        "kind": "discharge_rx", "time": dc["prior_dischtime"], "storetime": dc["prior_dischtime"],
        "dose_mg": dc["mg"], "rate_mg_h": np.nan, "duration_h": np.nan,
        "frequency": dc["frequency"].fillna("")})


def attach(targets: pd.DataFrame, events: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Events of earlier admissions, and of the target admission stored before t_dec.

    On the target admission emar counts only before ICU admission: t0 is the first ICU bolus
    in inputevents, so an emar IV dose between intime and t0 is either the t0 dose charted
    twice or contradicts t0, and it would leak the label in the first case."""
    prior = pairs[["hadm_id", "prior_hadm"]].merge(
        events, left_on="prior_hadm", right_on="event_hadm_id").drop(columns="prior_hadm")
    prior = prior.assign(scope="prior_admission")
    now = targets[["hadm_id"]].merge(events, left_on="hadm_id", right_on="event_hadm_id")
    now = now.assign(scope="this_admission")
    j = pd.concat([prior, now], ignore_index=True).merge(
        targets[["hadm_id", "t_dec", "intime"]], on="hadm_id")
    keep = (j["time"] < j["t_dec"]) & (j["storetime"].isna() | (j["storetime"] < j["t_dec"]))
    in_icu_emar = (j["scope"].eq("this_admission") & j["source"].eq("emar")
                   & (j["time"] >= j["intime"]))
    j = j[keep & ~in_icu_emar].drop(columns="intime")
    return j.assign(hours_before_t_dec=_hours(j["t_dec"] - j["time"]))


# --- aggregates -------------------------------------------------------------------------------

def history_features(targets: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    idx = targets["hadm_id"]
    given = hist[~hist["kind"].eq("discharge_rx")]
    prior = given[given["scope"].eq("prior_admission")]
    now = given[given["scope"].eq("this_admission")]
    p_bolus = prior[prior["kind"].eq("bolus")].sort_values("hours_before_t_dec")
    p_iv = prior[prior["kind"].isin(["bolus", "infusion"])]
    ward = now[now["source"].eq("emar") & now["kind"].eq("bolus")].sort_values("hours_before_t_dec")
    oral_now = now[now["kind"].eq("oral")].sort_values("hours_before_t_dec")

    first_icu = (prior[prior["source"].eq("icu_inputevents") & prior["kind"].eq("bolus")]
                 .sort_values("hours_before_t_dec", ascending=False)
                 .drop_duplicates(["hadm_id", "event_hadm_id"]))   # first bolus of each one
    last_first = first_icu.sort_values("hours_before_t_dec").drop_duplicates("hadm_id")
    last_first = last_first.set_index("hadm_id")

    dc = hist[hist["kind"].eq("discharge_rx")].sort_values("hours_before_t_dec")
    dc = dc.drop_duplicates("hadm_id").set_index("hadm_id")
    on = oral_now.drop_duplicates("hadm_id").set_index("hadm_id")
    oral_mg = dc["dose_mg"].reindex(idx).to_numpy(dtype=float)
    oral_src = np.where(idx.isin(dc.index), "prior_discharge_rx",
                        np.where(idx.isin(on.index), "this_admission", "unknown"))
    oral_mg = np.where(oral_src == "this_admission",
                       on["dose_mg"].reindex(idx).to_numpy(dtype=float), oral_mg)
    oral_days = np.where(oral_src == "prior_discharge_rx",
                         (dc["hours_before_t_dec"] / 24).reindex(idx).to_numpy(dtype=float),
                         (on["hours_before_t_dec"] / 24).reindex(idx).to_numpy(dtype=float))

    in_24h = ward[ward["hours_before_t_dec"] <= WARD_IV_TOTAL_H]
    g = {
        "prior_furo_admissions": prior.groupby("hadm_id")["event_hadm_id"].nunique(),
        "prior_iv_n": p_iv.groupby("hadm_id").size(),
        "prior_iv_last_mg": p_bolus.groupby("hadm_id")["dose_mg"].first(),
        "prior_iv_median_mg": p_bolus.groupby("hadm_id")["dose_mg"].median(),
        "prior_iv_max_mg": p_bolus.groupby("hadm_id")["dose_mg"].max(),
        "prior_icu_first_bolus_mg": last_first["dose_mg"],
        "prior_icu_first_bolus_days": last_first["hours_before_t_dec"] / 24,
        "prior_icu_first_bolus_uo_6h_ml": last_first["uo_6h_after_ml"],
        "prior_icu_first_bolus_escalated": last_first["escalated_24h"],
        "prior_infusion": prior[prior["kind"].eq("infusion")].groupby("hadm_id").size() > 0,
        "prior_last_furo_days": (prior["hours_before_t_dec"] / 24).groupby(prior["hadm_id"]).min(),
        "ward_iv_n": ward.groupby("hadm_id").size(),
        "ward_iv_last_mg": ward.groupby("hadm_id")["dose_mg"].first(),
        "ward_iv_max_mg": ward.groupby("hadm_id")["dose_mg"].max(),
        "ward_iv_total_24h_mg": in_24h.groupby("hadm_id")["dose_mg"].sum(min_count=1),
        "ward_iv_last_hours_ago": ward.groupby("hadm_id")["hours_before_t_dec"].min(),
        "oral_this_adm_last_mg": on["dose_mg"],
        "icu_infusion_before": now[now["kind"].eq("infusion")].groupby("hadm_id").size() > 0,
    }
    out = pd.DataFrame({k: v.reindex(idx).to_numpy() for k, v in g.items()}, index=idx)
    counts = ["prior_furo_admissions", "prior_iv_n", "ward_iv_n"]
    flags = ["prior_infusion", "icu_infusion_before"]
    out[counts] = out[counts].fillna(0).astype(int)
    out[flags] = out[flags].eq(True).astype(int)
    out["oral_recorded_mg"] = oral_mg
    out["oral_recorded_source"] = oral_src
    out["oral_recorded_days"] = oral_days
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mimic-root", help="directory holding hosp/ and icu/ (default $MIMIC_ROOT)")
    ap.add_argument("--cohort-dir", default=str(OUT_DIR))
    ap.add_argument("--tag", default="", help='"" for the main cohort, "_5d" for the 5D stratum')
    args = ap.parse_args()
    root = resolve_mimic_root(args.mimic_root)
    cohort_dir = Path(args.cohort_dir)
    cohort_path = cohort_dir / f"cohort{args.tag}.csv.gz"
    if not cohort_path.exists():
        print(f"missing {cohort_path}: run cohort_build first", file=sys.stderr)
        return 1

    t = pd.read_csv(cohort_path, dtype=str, keep_default_na=False)
    t = t.assign(**{c: _ts(t[c]) for c in ("t0", "t_dec", "admittime", "intime")},
                 anchor_age=_num(t["anchor_age"]), anchor_year=_num(t["anchor_year"]))
    hadms, subjects = set(t["hadm_id"]), set(t["subject_id"])
    _log(f"{len(t):,} targets from {cohort_path}")
    cut = at_t_dec(t)
    f = pd.DataFrame(index=t["hadm_id"])

    # ---- D --------------------------------------------------------------------------------
    female = t["gender"].eq("F").to_numpy()
    age = (t["anchor_age"] + t["t_dec"].dt.year - t["anchor_year"]).to_numpy()
    f["age"], f["sex_female"] = age, female.astype(int)
    for col in ("via_ed", "transplant_prior"):
        f[col] = t[col].eq("True").astype(int).to_numpy()
    for col in ("admission_type", "first_careunit", "service_at_t_dec"):
        f[col] = t[col].replace("", "unknown").to_numpy()
    f["hosp_hours_before"] = _hours(t["t_dec"] - t["admittime"]).to_numpy()
    # An order entered on the ward puts t_dec before ICU admission: 0 h in the ICU.
    f["icu_hours_before"] = _hours(t["t_dec"] - t["intime"]).clip(lower=0).to_numpy()

    _log("reading hosp/admissions, diagnoses_icd")
    admissions = read_table(root, "hosp", "admissions",
                            ["subject_id", "hadm_id", "admittime", "dischtime"])
    admissions = admissions[admissions["subject_id"].isin(subjects)].assign(
        admittime=lambda d: _ts(d["admittime"]), dischtime=lambda d: _ts(d["dischtime"]))
    pairs = prior_pairs(t, admissions)
    pairs = pairs[pairs["prior_dischtime"] < t.set_index("hadm_id")["t_dec"]
                  .reindex(pairs["hadm_id"]).to_numpy()]
    f["n_prior_admissions"] = pairs.groupby("hadm_id").size().reindex(f.index).fillna(0).astype(int)
    last_dc = pairs.groupby("hadm_id")["prior_dischtime"].max()
    f["days_since_last_discharge"] = (
        (t.set_index("hadm_id")["admittime"] - last_dc.reindex(f.index)).dt.total_seconds()
        / 86400)
    dx = read_table(root, "hosp", "diagnoses_icd", ["subject_id", "hadm_id", "icd_code",
                                                     "icd_version"])
    dx = dx[dx["subject_id"].isin(subjects)]
    for col, prefixes in PRIOR_DX.items():
        hit = set(dx.loc[prefix_flag(dx, prefixes), "hadm_id"])
        with_hit = set(pairs.loc[pairs["prior_hadm"].isin(hit), "hadm_id"])
        f[col] = f.index.isin(with_hit).astype(int)

    # ---- V, weight, code status ------------------------------------------------------------
    _log("reading icu/chartevents (vitals, weight, code status; slow)")
    vital_of = {str(i): name for name, ids in VITALS.items() for i in ids}
    weight_ids = {str(i): k for i, k in WEIGHT_ITEMIDS.items()}
    wanted = set(vital_of) | set(weight_ids) | {str(CODE_STATUS_ITEMID)}
    ce = _read_filtered(root, "icu", "chartevents",
                        ["hadm_id", "itemid", "charttime", "storetime", "value", "valuenum"],
                        lambda c: c["itemid"].isin(wanted) & c["hadm_id"].isin(hadms))
    ce = ce.assign(charttime=_ts(ce["charttime"]), storetime=_ts(ce["storetime"]),
                   valuenum=_num(ce["valuenum"]))
    vitals = ce[ce["itemid"].isin(set(vital_of))].assign(
        feature=lambda d: d["itemid"].map(vital_of)).dropna(subset=["valuenum"])
    vitals.loc[vitals["feature"].eq("temp_f"), "valuenum"] = (
        (vitals.loc[vitals["feature"].eq("temp_f"), "valuenum"] - 32) * 5 / 9)
    vitals["feature"] = vitals["feature"].replace("temp_f", "temp_c")
    v6 = in_window(cut, vitals, VITAL_WINDOW_H)
    last_v = latest(v6).reindex(f.index)
    for col in ("hr", "sbp", "dbp", "map", "spo2", "resp_rate", "temp_c"):
        f[col] = last_v[col] if col in last_v else np.nan
    low = v6.groupby(["hadm_id", "feature"])["valuenum"].min().unstack().reindex(f.index)
    f["sbp_min_6h"] = low["sbp"] if "sbp" in low else np.nan
    f["map_min_6h"] = low["map"] if "map" in low else np.nan

    weight = ce[ce["itemid"].isin(set(weight_ids)) & ce["valuenum"].gt(0)]
    weight = weight.assign(feature="weight",
                           valuenum=weight["valuenum"] * weight["itemid"].map(weight_ids))
    w_icu = latest(in_window(cut, weight, None)).reindex(f.index)
    w_icu = w_icu["weight"] if "weight" in w_icu else pd.Series(np.nan, index=f.index)
    status = ce[ce["itemid"].eq(str(CODE_STATUS_ITEMID))].assign(feature="code_status")
    cs = latest(in_window(cut, status, None), value="value").reindex(f.index)
    f["code_status"] = code_status(cs["code_status"] if "code_status" in cs
                                   else pd.Series(np.nan, index=f.index))

    # ---- L, K -------------------------------------------------------------------------------
    _log("reading hosp/labevents (labs, all creatinine; slow)")
    lab_of = {v: k for k, v in LABS.items()}
    labs = _read_filtered(root, "hosp", "labevents",
                          ["subject_id", "hadm_id", "itemid", "charttime", "storetime",
                           "valuenum"],
                          lambda c: c["itemid"].isin(set(lab_of)) & c["subject_id"].isin(subjects))
    labs = labs.assign(charttime=_ts(labs["charttime"]), storetime=_ts(labs["storetime"]),
                       valuenum=_num(labs["valuenum"]), feature=labs["itemid"].map(lab_of))
    labs = labs.dropna(subset=["valuenum"])
    l48 = in_window(cut, labs, LAB_WINDOW_H, on="subject_id")
    last_l = latest(l48).reindex(f.index)
    for col in LABS:
        f[col] = last_l[col] if col in last_l else np.nan
    cr = l48[l48["feature"].eq("creatinine")].sort_values("charttime")
    first_cr = cr.groupby("hadm_id")["valuenum"].first().reindex(f.index)
    last_cr_t = cr.groupby("hadm_id")["charttime"].last().reindex(f.index)
    f["creatinine_hours_ago"] = _hours(t.set_index("hadm_id")["t_dec"] - last_cr_t)
    f["creatinine_change_48h"] = f["creatinine"] - first_cr

    creat = labs[labs["feature"].eq("creatinine")]
    pt = t.drop_duplicates("subject_id").set_index("subject_id")
    who = pt.loc[creat["subject_id"]]
    c_age = who["anchor_age"].to_numpy() + (creat["charttime"].dt.year.to_numpy()
                                            - who["anchor_year"].to_numpy())
    creat = creat.assign(egfr=egfr_ckd_epi_2021(
        creat["valuenum"], pd.Series(c_age, index=creat.index),
        pd.Series(who["gender"].eq("F").to_numpy(), index=creat.index)))
    _log("reading hosp/omr (eGFR, weight)")
    omr = read_table(root, "hosp", "omr", ["subject_id", "chartdate", "result_name",
                                           "result_value"])
    omr = omr[omr["subject_id"].isin(subjects)]
    readings = egfr_readings(creat, omr[omr["result_name"].str.contains("egfr", case=False)])
    base = baseline_egfr(t, readings).reindex(f.index)
    base_r = t[["hadm_id", "subject_id", "admittime"]].merge(readings, on="subject_id")
    base_r = base_r[(base_r["time"] < base_r["admittime"])
                    & (base_r["time"] >= base_r["admittime"] - pd.Timedelta(days=365))]
    base_t = base_r.sort_values("time").groupby("hadm_id")[["time", "admittime"]].last()
    f["egfr_t_dec"] = egfr_ckd_epi_2021(f["creatinine"], pd.Series(age, index=f.index),
                                        pd.Series(female, index=f.index))
    f["egfr_baseline"] = base
    f["egfr_baseline_days"] = ((base_t["admittime"] - base_t["time"]).dt.total_seconds()
                               / 86400).reindex(f.index)
    f["egfr_delta"] = f["egfr_t_dec"] - f["egfr_baseline"]
    f["stage_from_t_dec"] = (f["egfr_baseline"].isna() & f["egfr_t_dec"].notna()).astype(int)
    f["ckd_stage"] = egfr_stage(f["egfr_baseline"].fillna(f["egfr_t_dec"])).replace(
        "missing", "unknown").to_numpy()

    w_omr = omr[omr["result_name"].str.contains(r"^weight", case=False)]
    w_omr = pd.DataFrame({"subject_id": w_omr["subject_id"], "time": _ts(w_omr["chartdate"]),
                          "kg": _num(w_omr["result_value"]) * LB_TO_KG}).dropna()
    wj = t[["hadm_id", "subject_id", "t_dec"]].merge(w_omr, on="subject_id")
    wj = wj[(wj["time"] < wj["t_dec"].dt.normalize())
            & (wj["time"] >= wj["t_dec"].dt.normalize() - pd.Timedelta(days=OMR_WEIGHT_DAYS))]
    w_o = wj.sort_values("time").groupby("hadm_id")["kg"].last().reindex(f.index)
    f["weight_kg"] = w_icu.fillna(w_o)
    f["weight_source"] = np.where(w_icu.notna(), "icu_chart",
                                  np.where(w_o.notna(), "omr", "unknown"))

    # ---- U -----------------------------------------------------------------------------------
    _log("reading icu/outputevents (urine, all admissions of the targets)")
    uo_ids = {str(i) for i in URINE_ITEMIDS | {IRRIGANT_IN_ITEMID}}
    uo = _read_filtered(root, "icu", "outputevents",
                        ["subject_id", "hadm_id", "itemid", "charttime", "storetime", "value"],
                        lambda c: c["itemid"].isin(uo_ids) & c["subject_id"].isin(subjects))
    uo = uo.assign(charttime=_ts(uo["charttime"]), storetime=_ts(uo["storetime"]),
                   ml=_num(uo["value"]).fillna(0.0))
    uo.loc[uo["itemid"].eq(str(IRRIGANT_IN_ITEMID)), "ml"] *= -1
    now_uo = uo[uo["hadm_id"].isin(hadms)].drop(columns="subject_id")
    for h in UO_WINDOW_H:
        rows = in_window(cut, now_uo, h)
        f[f"uo_{h}h_ml"] = rows.groupby("hadm_id")["ml"].sum().reindex(f.index)
    f["uo_window_6h_h"] = np.minimum(6.0, f["icu_hours_before"])
    f["uo_rate_6h_ml_kg_h"] = (f["uo_6h_ml"] / f["weight_kg"]
                               / f["uo_window_6h_h"].where(f["uo_window_6h_h"] > 0))

    # ---- inputevents: furosemide (all admissions), fluids (target admission) ----------------
    _log("reading icu/inputevents (furosemide, fluids)")
    furo_ids = {str(i) for i in FUROSEMIDE_ITEMIDS}
    ie = _read_filtered(root, "icu", "inputevents",
                        ["subject_id", "hadm_id", "starttime", "endtime", "storetime", "itemid",
                         "amount", "amountuom", "rate", "rateuom", "ordercategorydescription",
                         "statusdescription"],
                        lambda c: (c["itemid"].isin(furo_ids) & c["subject_id"].isin(subjects))
                        | c["hadm_id"].isin(hadms))
    ie = ie[ie["statusdescription"].ne("Rewritten")]
    ie = ie.assign(starttime=_ts(ie["starttime"]), endtime=_ts(ie["endtime"]),
                   storetime=_ts(ie["storetime"]), amount=_num(ie["amount"]),
                   rate=_num(ie["rate"]))
    fluid = ie[ie["hadm_id"].isin(hadms) & ie["amountuom"].str.lower().eq("ml")]
    fluid = fluid.rename(columns={"starttime": "charttime"}).drop(columns="subject_id")
    fl = in_window(cut, fluid, FLUID_WINDOW_H)
    f["fluid_in_24h_ml"] = fl.groupby("hadm_id")["amount"].sum().reindex(f.index)

    # ---- emar (+ detail dose, pharmacy route), prescriptions --------------------------------
    _log("reading hosp/emar (coverage, furosemide administrations)")
    all_hadms = set(admissions["hadm_id"])
    observable: set = set()
    rows_: list[pd.DataFrame] = []
    for chunk in read_table(root, "hosp", "emar", ["subject_id", "hadm_id", "emar_id",
                                                   "pharmacy_id", "charttime", "storetime",
                                                   "medication", "event_txt"],
                            chunksize=2_000_000):
        observable |= set(chunk.loc[chunk["hadm_id"].isin(all_hadms), "hadm_id"])
        rows_.append(chunk[chunk["subject_id"].isin(subjects)
                           & chunk["medication"].str.contains(FUROSEMIDE, case=False)])
    emar = pd.concat(rows_, ignore_index=True)
    emar = emar[administered(emar["event_txt"])]
    _log("reading hosp/emar_detail (dose given, route, product)")
    ids = set(emar["emar_id"])
    detail = _read_filtered(root, "hosp", "emar_detail",
                            ["emar_id", "dose_given", "dose_given_unit", "route",
                             "product_description"],
                            lambda c: c["emar_id"].isin(ids))
    given = detail.assign(mg=_num(detail["dose_given"]))
    given = given[given["dose_given_unit"].str.strip().str.lower().eq("mg") & given["mg"].gt(0)]
    mg_of = given.drop_duplicates("emar_id").set_index("emar_id")["mg"]
    _log("reading hosp/pharmacy, prescriptions (route, frequency, discharge orders)")
    pharmacy = _read_filtered(
        root, "hosp", "pharmacy", ["subject_id", "pharmacy_id", "route", "frequency",
                                   "medication"],
        lambda c: c["subject_id"].isin(subjects)
        & c["medication"].str.contains(FUROSEMIDE, case=False))
    ph = pharmacy.drop_duplicates("pharmacy_id").set_index("pharmacy_id")
    emar = classify_route(emar, ph["route"], detail).assign(
        charttime=_ts(emar["charttime"]), storetime=_ts(emar["storetime"]),
        dose_mg=emar["emar_id"].map(mg_of))
    rx = _read_filtered(root, "hosp", "prescriptions",
                        ["subject_id", "hadm_id", "pharmacy_id", "drug", "starttime",
                         "stoptime", "dose_val_rx", "dose_unit_rx", "route"],
                        lambda c: c["subject_id"].isin(subjects)
                        & c["drug"].str.contains(FUROSEMIDE, case=False))
    # Orders of the target admission are never read into features (spec 4.1).
    rx = rx[~rx["hadm_id"].isin(hadms)]
    rx = rx.assign(starttime=_ts(rx["starttime"]), stoptime=_ts(rx["stoptime"]),
                   mg=_num(rx["dose_val_rx"]).where(
                       rx["dose_unit_rx"].str.strip().str.lower().eq("mg")),
                   frequency=rx["pharmacy_id"].map(ph["frequency"]))

    f["src_emar_this_adm"] = f.index.isin(observable).astype(int)
    f["src_emar_prior"] = f.index.isin(
        set(pairs.loc[pairs["prior_hadm"].isin(observable), "hadm_id"])).astype(int)

    # ---- history ------------------------------------------------------------------------------
    events = pd.concat([icu_events(ie[ie["itemid"].isin(furo_ids)]), emar_events(emar),
                        discharge_events(pairs, rx)], ignore_index=True)
    events = events.assign(time=_ts(events["time"]), storetime=_ts(events["storetime"]))
    icu_pts = events[events["source"].eq("icu_inputevents")].rename(
        columns={"event_hadm_id": "hadm_id"})
    uo_rows = uo[["hadm_id", "charttime", "ml"]].rename(
        columns={"charttime": "time", "ml": "value"})
    events["uo_6h_after_ml"] = window_sum(icu_pts, uo_rows, "hadm_id", RESPONSE_UO_H).reindex(
        events.index)
    cr_rows = creat[["subject_id", "charttime", "valuenum"]].rename(
        columns={"charttime": "time", "valuenum": "value"})
    events["creatinine_before"] = value_asof(events, cr_rows, "subject_id")["value"]
    after = value_asof(events, cr_rows, "subject_id", RESPONSE_CREAT_H, strict=False)
    events["creatinine_after_48h"] = after["value"].where(after["found"] > events["time"])
    events["escalated_24h"] = escalated(events)

    hist = attach(t, events, pairs)
    # A response counts only when its whole window closed before t_dec, in any scope: a
    # prior admission ending hours before readmission would otherwise see the t0 bolus as
    # an escalation, or creatinine drawn after t_dec. Creatinine is kept for prior ones only.
    before = hist["hours_before_t_dec"]
    hist.loc[before < RESPONSE_UO_H, "uo_6h_after_ml"] = np.nan
    hist.loc[before < ESCALATION_H, "escalated_24h"] = np.nan
    hist.loc[before < RESPONSE_CREAT_H, "creatinine_after_48h"] = np.nan
    hist.loc[hist["scope"].eq("this_admission"),
             ["creatinine_before", "creatinine_after_48h"]] = np.nan
    hist = hist.sort_values(["hadm_id", "hours_before_t_dec"], ascending=[True, False])

    h = history_features(t, hist)
    for col in h.columns:
        f[col] = h[col]

    missing = set(FEATURE_COLUMNS) - set(f.columns)
    assert not missing, f"features not built: {sorted(missing)}"
    f = f[FEATURE_COLUMNS].reset_index().merge(t[["hadm_id", "subject_id"]], on="hadm_id")
    f = f[["subject_id", "hadm_id"] + FEATURE_COLUMNS]
    f.to_csv(cohort_dir / f"features{args.tag}.csv.gz", index=False)
    hist[HISTORY_COLUMNS].to_csv(cohort_dir / f"history{args.tag}.csv.gz", index=False)
    _log(f"wrote features{args.tag}.csv.gz ({len(f):,} rows), history{args.tag}.csv.gz "
         f"({len(hist):,} events) in {cohort_dir} (patient-level: keep on the server)")

    cover = f[FEATURE_COLUMNS].notna().mean().mul(100).round(1)
    print("\n=== feature coverage (% non-missing) ===")
    for col, pct in cover.items():
        print(f"  {col:34s} {pct:6.1f}%")
    print(f"  history events by scope x kind: "
          f"{hist.groupby(['scope', 'kind']).size().to_dict()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
