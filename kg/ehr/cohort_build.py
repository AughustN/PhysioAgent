#!/usr/bin/env python
"""Cohort builder (docs/cohort_spec.md): one row per patient, its target admission, t0, t_dec,
the label, the analysis flags and the split.

Funnel as in cohort_funnel_probe, with the locked changes:

    S6  CKD (b) strict: eGFR < 60 twice >= 90 days apart before admission, from outpatient
        creatinine (labevents.hadm_id blank) and omr only, no eGFR >= 60 in between
    S9  creatinine resulted in [t_dec - 48 h, t_dec), where t_dec (spec 4.6) is the entertime
        of the latest IV furosemide order in the 24 h before t0 whose dose equals the label,
        else t0
    S10 last qualifying admission per patient = target

Flags written (spec, "Cờ"):
    A1 ckd_evidence (confirmed / index_only) + ckd_tier detail
    A2 decision_context: loading_for_infusion > protocol > routine
    A3 prior_exposure: ward_this_admission > prior_admission > unverifiable > naive
    B  transplant_prior (input), transplant_any (table 1 only)
    C  via_ed, first_careunit, service_at_t_dec, admission_type, anchor_year_group
A flags use information around and after t0; they are analysis axes and never model input.

Writes, under kg/data/ehr/cohort/ (gitignored; patient-level, stays on the server, DUA):
    cohort.csv.gz      targets of the main cohort, with split and split_temporal
    cohort_5d.csv.gz   targets of the dialysis-dependent stratum (S8), same columns, no split
    outcomes.csv.gz    post-t0 fields kept apart from the cohort, evaluation only
and the aggregate report kg/data/ehr/cohort_build_report.json.

    python -m physioagent.kg.ehr.cohort_build --mimic-root <mimic-iv-3.1>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from physioagent.kg import paths
from physioagent.kg.ehr.build_visits import read_table, resolve_mimic_root
from physioagent.kg.ehr.cohort_feature_probe import (
    chronic_low_strict, loading_for_infusion, matching_order, prior_exposure, prior_pairs,
    select_targets)
from physioagent.kg.ehr.cohort_funnel_probe import (
    CARDIAC_SURGERY_PREFIXES, CREATININE_ITEMID, DIALYSIS_ITEMIDS, DIALYSIS_LOOKBACK_H, DOSE_EPS,
    EGFR_CKD, FUROSEMIDE_ITEMIDS, INPUT_ITEMIDS, IV_ROUTE, MIN_AGE, REQUIRED_CREAT_H, Funnel,
    _code, _counts, _log, _num, _read_filtered, _ts, administered, age_at_t0, classify_route,
    code_flags, dialysis_at_t0, egfr_ckd_epi_2021, emar_group, first_bolus, infusion_running,
    latest_creatinine, near_t0, on_ladder, prior_flags, split_inputevents)

OUT_DIR = paths.EHR / "cohort"
REPORT = paths.EHR / "cohort_build_report.json"
FUROSEMIDE = r"furosemide|lasix"

SEED = 2026
SPLIT = {"train": 0.70, "dev": 0.15, "test": 0.15}
LATE_YEAR_GROUPS = {"2017 - 2019", "2020 - 2022"}

COLUMNS = ["subject_id", "hadm_id", "stay_id", "t0", "t_dec", "dose_mg", "dose_band",
           "admittime", "intime", "gender", "anchor_age", "anchor_year", "anchor_year_group",
           "ckd_tier", "ckd_evidence", "emar_group", "prior_exposure", "protocol_transfusion",
           "protocol_albumin", "protocol_post_cardiac_surgery", "loading_for_infusion",
           "decision_context", "transplant_prior", "transplant_any", "via_ed", "first_careunit",
           "service_at_t_dec", "admission_type"]


# --- CKD (spec section 3) -------------------------------------------------------------------

def egfr_readings(creat: pd.DataFrame, omr: pd.DataFrame) -> pd.DataFrame:
    """Outpatient eGFR readings (subject_id, time, egfr): creatinine with no hadm_id + omr."""
    outpatient = creat.loc[creat["hadm_id"].eq(""), ["subject_id", "charttime", "egfr"]]
    omr_egfr = _num(omr["result_value"].str.extract(r"^\s*>?\s*(\d+(?:\.\d+)?)\s*$")[0])
    omr_r = pd.DataFrame({"subject_id": omr["subject_id"], "time": _ts(omr["chartdate"]),
                          "egfr": omr_egfr})
    return pd.concat([outpatient.rename(columns={"charttime": "time"}), omr_r],
                     ignore_index=True).dropna()


def ckd_tier(a: np.ndarray, b: np.ndarray, c: np.ndarray, egfr_t0: pd.Series) -> np.ndarray:
    return np.select(
        [a & b, a, b, c & egfr_t0.lt(EGFR_CKD).to_numpy(), c & egfr_t0.ge(EGFR_CKD).to_numpy(), c],
        ["confirmed_a_and_b", "confirmed_a", "confirmed_b", "index_consistent",
         "index_discordant", "index_no_recent_creatinine"], default="none")


# --- decision time (spec 4.6) ---------------------------------------------------------------

def decision_time(cohort: pd.DataFrame, orders: pd.DataFrame) -> pd.Series:
    """Entertime of the matching IV order, else t0."""
    entered = matching_order(cohort, orders).set_index("hadm_id")["visible_time"]
    t_dec = cohort["hadm_id"].map(entered)
    return t_dec.fillna(cohort["t0"])


# --- flags and split ------------------------------------------------------------------------

def dose_band(dose: pd.Series) -> pd.Series:
    band = np.select([dose <= 20 + DOSE_EPS, dose <= 60 + DOSE_EPS], ["le20", "40_60"],
                     default="ge80")
    return pd.Series(band, index=dose.index)


def decision_context(loading: pd.Series, protocol: pd.Series) -> pd.Series:
    return pd.Series(np.select([loading, protocol], ["loading_for_infusion", "protocol"],
                               default="routine"), index=loading.index)


def stratified_split(frame: pd.DataFrame, strata: list[str], fractions: dict[str, float],
                     seed: int = SEED) -> pd.Series:
    """Shuffle each stratum with a fixed seed and cut it by `fractions` (rows sorted by
    subject_id first, so the result does not depend on input order)."""
    rng = np.random.default_rng(seed)
    out = pd.Series("", index=frame.index, dtype=object)
    shares = np.cumsum(list(fractions.values()))[:-1] / sum(fractions.values())
    for _, idx in frame.sort_values("subject_id").groupby(strata, sort=True).groups.items():
        idx = rng.permutation(np.asarray(idx))
        cuts = np.round(shares * len(idx)).astype(int)
        for name, part in zip(fractions, np.split(idx, cuts)):
            out.loc[part] = name
    return out


def temporal_split(frame: pd.DataFrame, strata: list[str], seed: int = SEED) -> pd.Series:
    """Sensitivity split: late anchor_year_group = test, the rest train/dev at 70:15."""
    late = frame["anchor_year_group"].isin(LATE_YEAR_GROUPS)
    out = pd.Series("test", index=frame.index, dtype=object)
    early = frame[~late]
    out.loc[early.index] = stratified_split(early, strata, {"train": SPLIT["train"],
                                                            "dev": SPLIT["dev"]}, seed)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mimic-root", help="directory holding hosp/ and icu/ (default $MIMIC_ROOT)")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--report", default=str(REPORT))
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    root = resolve_mimic_root(args.mimic_root)
    _log(f"mimic root {root}")
    funnel = Funnel()

    # ---- S1-S5 -----------------------------------------------------------------------------
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
    funnel.record("S1", "IV bolus/push, mg, amount > 0 (events)", parts["bolus"], unit="events")
    cohort = first_bolus(parts["bolus"])
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
                            ["subject_id", "hadm_id", "admittime", "dischtime", "edregtime",
                             "admission_type", "hospital_expire_flag"])
    admissions = admissions.assign(admittime=_ts(admissions["admittime"]),
                                   dischtime=_ts(admissions["dischtime"]))
    icustays = read_table(root, "icu", "icustays", ["stay_id", "intime", "first_careunit"])
    icustays = icustays.assign(intime=_ts(icustays["intime"]))

    cohort = cohort[age_at_t0(cohort, patients) >= MIN_AGE]
    funnel.record("S4", f"age >= {MIN_AGE} at t0", cohort)
    cohort = cohort[~cohort["hadm_id"].isin(infusion_running(cohort, parts["infusion"]))]
    funnel.record("S5", "no furosemide infusion running at t0", cohort)

    cohort = (cohort.merge(admissions.drop(columns=["subject_id", "dischtime",
                                                    "hospital_expire_flag"]),
                           on="hadm_id", how="left")
                    .merge(icustays, on="stay_id", how="left")
                    .merge(patients, on="subject_id", how="left"))
    subjects = set(cohort["subject_id"])

    # ---- S6: CKD, (b) strict ---------------------------------------------------------------
    _log("reading hosp/diagnoses_icd")
    dx = read_table(root, "hosp", "diagnoses_icd", ["subject_id", "hadm_id", "icd_code",
                                                     "icd_version"])
    flags = code_flags(dx[dx["subject_id"].isin(subjects)])
    prior = prior_flags(cohort, admissions[admissions["subject_id"].isin(subjects)], flags)
    index = flags.reindex(cohort["hadm_id"], fill_value=False).astype(bool)

    _log("reading hosp/labevents creatinine (largest table; this is the slow step)")
    creat = _read_filtered(root, "hosp", "labevents",
                           ["subject_id", "hadm_id", "itemid", "charttime", "storetime",
                            "valuenum"],
                           lambda c: c["itemid"].eq(CREATININE_ITEMID)
                           & c["subject_id"].isin(subjects))
    creat = creat.assign(charttime=_ts(creat["charttime"]), storetime=_ts(creat["storetime"]),
                         valuenum=_num(creat["valuenum"])).dropna(subset=["valuenum"])
    who = patients.set_index("subject_id").loc[creat["subject_id"]]
    creat_age = who["anchor_age"].to_numpy() + (creat["charttime"].dt.year.to_numpy()
                                                - who["anchor_year"].to_numpy())
    creat["egfr"] = egfr_ckd_epi_2021(creat["valuenum"], pd.Series(creat_age, index=creat.index),
                                      pd.Series(who["gender"].eq("F").to_numpy(),
                                                index=creat.index))

    _log("reading hosp/omr (outpatient eGFR)")
    omr = read_table(root, "hosp", "omr", ["subject_id", "chartdate", "result_name",
                                           "result_value"])
    omr = omr[omr["subject_id"].isin(subjects)
              & omr["result_name"].str.contains("egfr", case=False)]
    chronic = chronic_low_strict(cohort, egfr_readings(creat, omr))

    # Joined on subject_id: ED draws before admission carry no hadm_id (spec 4).
    creat_any = creat.drop(columns="hadm_id")
    t0_creat = latest_creatinine(cohort, creat_any, REQUIRED_CREAT_H)
    by_hadm = cohort.set_index("hadm_id")
    age_t0 = by_hadm["anchor_age"] + by_hadm["t0"].dt.year - by_hadm["anchor_year"]
    egfr_t0 = egfr_ckd_epi_2021(t0_creat["valuenum"], age_t0.reindex(t0_creat.index),
                                by_hadm["gender"].eq("F").reindex(t0_creat.index))
    cohort = cohort.assign(ckd_tier=ckd_tier(prior["ckd"].to_numpy(),
                                             cohort["hadm_id"].isin(chronic).to_numpy(),
                                             index["ckd"].to_numpy(),
                                             cohort["hadm_id"].map(egfr_t0)))
    tiers_before_s6 = _counts(cohort["ckd_tier"])
    cohort = cohort[cohort["ckd_tier"].ne("none")]
    funnel.record("S6", "CKD evidence, (b) strict", cohort)

    # ---- S7: dialysis ----------------------------------------------------------------------
    _log("reading icu/procedureevents (dialysis)")
    dial_ids = {str(i) for i in DIALYSIS_ITEMIDS}
    pe = _read_filtered(root, "icu", "procedureevents",
                        ["subject_id", "itemid", "starttime", "endtime"],
                        lambda c: c["itemid"].isin(dial_ids) & c["subject_id"].isin(subjects))
    pe = pe.assign(starttime=_ts(pe["starttime"]), endtime=_ts(pe["endtime"]))
    cohort = cohort[~cohort["hadm_id"].isin(dialysis_at_t0(cohort, pe))]
    funnel.record("S7", f"no dialysis in the {DIALYSIS_LOOKBACK_H} h before t0", cohort)

    # ---- t_dec (spec 4.6) ------------------------------------------------------------------
    _log("reading hosp/pharmacy, prescriptions (furosemide orders)")
    pharmacy = _read_filtered(
        root, "hosp", "pharmacy",
        ["subject_id", "pharmacy_id", "hadm_id", "medication", "route", "entertime"],
        lambda c: c["subject_id"].isin(subjects)
        & c["medication"].str.contains(FUROSEMIDE, case=False))
    hadms = set(cohort["hadm_id"])
    rx = _read_filtered(root, "hosp", "prescriptions",
                        ["pharmacy_id", "hadm_id", "drug", "dose_val_rx", "dose_unit_rx"],
                        lambda c: c["hadm_id"].isin(hadms)
                        & c["drug"].str.contains(FUROSEMIDE, case=False))
    rx_mg = _num(rx["dose_val_rx"]).where(rx["dose_unit_rx"].str.strip().str.lower().eq("mg"))
    mg_of = rx.assign(mg=rx_mg).drop_duplicates("pharmacy_id").set_index("pharmacy_id")["mg"]
    iv_orders = pharmacy[pharmacy["hadm_id"].isin(hadms)
                         & pharmacy["route"].str.contains(IV_ROUTE)]
    orders = pd.DataFrame({"hadm_id": iv_orders["hadm_id"],
                           "visible_time": _ts(iv_orders["entertime"]),
                           "order_mg": iv_orders["pharmacy_id"].map(mg_of)})
    cohort = cohort.assign(t_dec=decision_time(cohort, orders))

    # ---- S8-S10 ----------------------------------------------------------------------------
    prior_c = prior.reindex(cohort["hadm_id"])
    index_c = index.reindex(cohort["hadm_id"])
    dependent = (prior_c["dialysis_status"] | prior_c["esrd"]
                 | index_c["dialysis_status"] | index_c["esrd"]).to_numpy()
    stratum_5d = cohort[dependent]
    cohort = cohort[~dependent]
    funnel.record("S8", "not dialysis-dependent (5D -> separate stratum)", cohort)

    def creatinine_before_t_dec(frame: pd.DataFrame) -> pd.DataFrame:
        known = latest_creatinine(frame.assign(t0=frame["t_dec"]), creat_any, REQUIRED_CREAT_H)
        return frame[frame["hadm_id"].isin(known.index)]

    cohort = creatinine_before_t_dec(cohort)
    funnel.record("S9", f"creatinine resulted in [t_dec - {REQUIRED_CREAT_H} h, t_dec)", cohort)
    s9_admissions = len(cohort)
    targets = select_targets(cohort).reset_index(drop=True)
    funnel.record("S10", "last qualifying admission per patient = target", targets,
                  unit="patients")
    targets_5d = select_targets(creatinine_before_t_dec(stratum_5d)).reset_index(drop=True)

    # ---- flags on targets (both strata) ----------------------------------------------------
    both = pd.concat([targets.assign(stratum="main"), targets_5d.assign(stratum="5d")],
                     ignore_index=True)
    t_hadms, t_subjects = set(both["hadm_id"]), set(both["subject_id"])

    _log("reading hosp/emar (coverage, furosemide administrations)")
    observable: set = set()
    furo_rows: list[pd.DataFrame] = []
    for chunk in read_table(root, "hosp", "emar", ["subject_id", "hadm_id", "emar_id",
                                                   "pharmacy_id", "charttime", "medication",
                                                   "event_txt"], chunksize=2_000_000):
        observable |= set(chunk.loc[chunk["hadm_id"].isin(t_hadms), "hadm_id"])
        furo_rows.append(chunk[chunk["subject_id"].isin(t_subjects)
                               & chunk["medication"].str.contains(FUROSEMIDE, case=False)])
    emar = pd.concat(furo_rows, ignore_index=True)
    emar = emar[administered(emar["event_txt"])]
    route_of = pharmacy.drop_duplicates("pharmacy_id").set_index("pharmacy_id")["route"]
    no_detail = pd.DataFrame(columns=["emar_id", "route", "product_description"])
    emar = classify_route(emar, route_of, no_detail).assign(charttime=_ts(emar["charttime"]))
    both["emar_group"] = emar_group(both, observable,
                                    emar[emar["iv"] & emar["hadm_id"].isin(t_hadms)])

    furo_ie = ie[ie["itemid"].isin(FUROSEMIDE_ITEMIDS) & ie["statusdescription"].ne("Rewritten")]
    events = pd.concat([furo_ie[["hadm_id"]], emar[["hadm_id"]]], ignore_index=True)
    pairs = prior_pairs(both, admissions[admissions["subject_id"].isin(t_subjects)])
    prior_ev = pairs[["hadm_id", "prior_hadm"]].merge(
        events.rename(columns={"hadm_id": "prior_hadm"}), on="prior_hadm")
    both["prior_exposure"] = prior_exposure(both, prior_ev)

    _log("reading hosp/services, procedures_icd")
    services = read_table(root, "hosp", "services", ["hadm_id", "transfertime", "curr_service"])
    services = services[services["hadm_id"].isin(t_hadms)].assign(
        transfertime=lambda d: _ts(d["transfertime"]))
    svc = both[["hadm_id", "t_dec"]].merge(services, on="hadm_id")
    svc = svc[svc["transfertime"] <= svc["t_dec"]].sort_values("transfertime")
    service = svc.groupby("hadm_id")["curr_service"].last()
    px = read_table(root, "hosp", "procedures_icd", ["hadm_id", "chartdate", "icd_code",
                                                     "icd_version"])
    px = px[px["hadm_id"].isin(t_hadms)]
    pcode, pver = _code(px["icd_code"]), px["icd_version"].astype(str)
    cardiac = pd.Series(False, index=px.index)
    for ver, prefixes in CARDIAC_SURGERY_PREFIXES.items():
        cardiac |= pver.eq(ver) & pcode.str.startswith(prefixes)
    px = px[cardiac].assign(chartdate=_ts(px.loc[cardiac, "chartdate"]))
    pj = both[["hadm_id", "t0"]].merge(px, on="hadm_id")
    surgery_before = set(pj.loc[pj["chartdate"] <= pj["t0"].dt.normalize(), "hadm_id"])

    both = both.assign(
        service_at_t_dec=both["hadm_id"].map(service).fillna("unknown"),
        protocol_transfusion=both["hadm_id"].isin(near_t0(both, parts["prbc"])),
        protocol_albumin=both["hadm_id"].isin(near_t0(both, parts["albumin"])),
        protocol_post_cardiac_surgery=both["hadm_id"].isin(surgery_before)
        & (both["hadm_id"].map(service).eq("CSURG")
           | both["first_careunit"].str.contains("CVICU", na=False)),
        loading_for_infusion=both["hadm_id"].isin(loading_for_infusion(both, parts["infusion"])),
        transplant_prior=prior.reindex(both["hadm_id"])["transplant"].to_numpy(),
        transplant_any=(prior.reindex(both["hadm_id"])["transplant"].to_numpy()
                        | index.reindex(both["hadm_id"])["transplant"].to_numpy()),
        via_ed=both["edregtime"].ne(""),
        ckd_evidence=np.where(both["ckd_tier"].str.startswith("confirmed"), "confirmed",
                              "index_only"),
        dose_band=dose_band(both["dose_mg"]),
    )
    protocol = both[["protocol_transfusion", "protocol_albumin",
                     "protocol_post_cardiac_surgery"]].any(axis=1)
    both["decision_context"] = decision_context(both["loading_for_infusion"], protocol)

    main_t = both[both["stratum"].eq("main")].copy()
    strata = ["dose_band", "ckd_evidence"]
    main_t["split"] = stratified_split(main_t, strata, SPLIT, args.seed)
    main_t["split_temporal"] = temporal_split(main_t, strata, args.seed)
    t5 = both[both["stratum"].eq("5d")]

    died = admissions.set_index("hadm_id")["hospital_expire_flag"].eq("1")
    outcomes = pd.DataFrame({"hadm_id": both["hadm_id"], "stratum": both["stratum"],
                             "hospital_expire_flag": both["hadm_id"].map(died).fillna(False)})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    main_t[COLUMNS + ["split", "split_temporal"]].to_csv(out_dir / "cohort.csv.gz", index=False)
    t5[COLUMNS].to_csv(out_dir / "cohort_5d.csv.gz", index=False)
    outcomes.to_csv(out_dir / "outcomes.csv.gz", index=False)
    _log(f"wrote {out_dir}/cohort.csv.gz, cohort_5d.csv.gz, outcomes.csv.gz "
         "(patient-level: keep on the server, never commit or share)")

    from_order = main_t["t_dec"].lt(main_t["t0"])
    report: dict[str, Any] = {
        "funnel": funnel.steps,
        "ckd_tier_before_s6": tiers_before_s6,
        "s9_admissions": int(s9_admissions),
        "targets": int(len(main_t)),
        "targets_5d": int(len(t5)),
        "median_dose_mg": float(main_t["dose_mg"].median()),
        "dose": _counts(main_t["dose_mg"].round().astype(int)),
        "t_dec_from_order": {"n": int(from_order.sum()),
                             "pct": round(100.0 * from_order.mean(), 2)},
        "ckd_tier": _counts(main_t["ckd_tier"]),
        "ckd_evidence": _counts(main_t["ckd_evidence"]),
        "decision_context": _counts(main_t["decision_context"]),
        "prior_exposure": _counts(main_t["prior_exposure"]),
        "crosstab_prior_exposure_x_anchor_year_group": pd.crosstab(
            main_t["prior_exposure"], main_t["anchor_year_group"]).to_dict("index"),
        "transplant_prior": int(main_t["transplant_prior"].sum()),
        "transplant_any": int(main_t["transplant_any"].sum()),
        "via_ed": int(main_t["via_ed"].sum()),
        "split": _counts(main_t["split"]),
        "split_temporal": _counts(main_t["split_temporal"]),
        "crosstab_split_x_dose_band": pd.crosstab(main_t["split"],
                                                  main_t["dose_band"]).to_dict("index"),
        "in_hospital_death": int(outcomes.loc[outcomes["stratum"].eq("main"),
                                              "hospital_expire_flag"].sum()),
    }
    print(f"\n=== targets {report['targets']:,} (S9 {s9_admissions:,} admissions), 5D "
          f"{report['targets_5d']:,}, median dose {report['median_dose_mg']:g} mg, t_dec from "
          f"order {report['t_dec_from_order']['pct']}% ===")
    for key in ("ckd_tier", "ckd_evidence", "decision_context", "prior_exposure", "split",
                "split_temporal"):
        print(f"  {key}: " + ", ".join(f"{k} {v['n']:,}" for k, v in report[key].items()))
    print(f"  prior_exposure x anchor_year_group: "
          f"{report['crosstab_prior_exposure_x_anchor_year_group']}")
    print(f"  split x dose_band: {report['crosstab_split_x_dose_band']}")
    print(f"  transplant prior {report['transplant_prior']:,} / any {report['transplant_any']:,}"
          f" | via ED {report['via_ed']:,} | in-hospital death {report['in_hospital_death']:,}")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _log(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
