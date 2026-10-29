"""Model inputs for the T0 furosemide dose task (docs/feature_schema.md).

`load_dataset` joins cohort.csv.gz (label, split, analysis flags) with features.csv.gz and
returns X, y and the grouping columns metrics are sliced by. X holds FEATURE_COLUMNS only;
any FORBIDDEN_COLUMNS reaching it raises, so a label-derived or post-t0 column cannot leak in.

    from physioagent.kg.ehr.dataset import load_dataset
    X, y, groups = load_dataset("train")
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from physioagent.kg import paths

COHORT_DIR = paths.EHR / "cohort"

DEMOGRAPHIC = ["age", "sex_female", "weight_kg", "weight_source", "via_ed", "admission_type",
               "first_careunit", "service_at_t_dec", "code_status", "transplant_prior",
               "hosp_hours_before", "icu_hours_before", "fluid_in_24h_ml"]
VITALS = ["hr", "sbp", "dbp", "map", "spo2", "resp_rate", "temp_c", "sbp_min_6h", "map_min_6h"]
LABS = ["creatinine", "creatinine_hours_ago", "creatinine_change_48h", "bun", "sodium",
        "potassium", "chloride", "bicarbonate", "magnesium", "glucose", "hemoglobin", "albumin",
        "lactate", "ntprobnp"]
KIDNEY = ["egfr_t_dec", "egfr_baseline", "egfr_baseline_days", "egfr_delta", "ckd_stage",
          "stage_from_t_dec"]
URINE = ["uo_6h_ml", "uo_24h_ml", "uo_window_6h_h", "uo_rate_6h_ml_kg_h"]
THIS_ADMISSION = ["src_emar_this_adm", "ward_iv_n", "ward_iv_last_mg", "ward_iv_max_mg",
                  "ward_iv_total_24h_mg", "ward_iv_last_hours_ago", "oral_this_adm_last_mg",
                  "icu_infusion_before"]
HISTORY = ["n_prior_admissions", "days_since_last_discharge", "src_emar_prior",
           "prior_furo_admissions", "prior_iv_n", "prior_iv_last_mg", "prior_iv_median_mg",
           "prior_iv_max_mg", "prior_icu_first_bolus_mg", "prior_icu_first_bolus_days",
           "prior_icu_first_bolus_uo_6h_ml", "prior_icu_first_bolus_escalated", "prior_infusion",
           "prior_last_furo_days", "oral_recorded_mg", "oral_recorded_source",
           "oral_recorded_days", "prior_hf", "prior_cirrhosis", "prior_diabetes"]
FEATURE_COLUMNS = DEMOGRAPHIC + VITALS + LABS + KIDNEY + URINE + THIS_ADMISSION + HISTORY
CATEGORICAL = ["weight_source", "admission_type", "first_careunit", "service_at_t_dec",
               "code_status", "ckd_stage", "oral_recorded_source"]

LABEL = "dose_mg"
GROUP_COLUMNS = ["dose_band", "ckd_tier", "ckd_evidence", "emar_group", "prior_exposure",
                 "decision_context", "anchor_year_group"]
FORBIDDEN_COLUMNS = frozenset(
    [LABEL, "dose_band"]
    + ["ckd_tier", "ckd_evidence", "emar_group", "prior_exposure", "decision_context",
       "protocol_transfusion", "protocol_albumin", "protocol_post_cardiac_surgery",
       "loading_for_infusion"]
    + ["transplant_any", "anchor_year_group", "anchor_year", "anchor_age"]
    + ["t0", "t_dec", "admittime", "intime", "split", "split_temporal"]
    + ["hospital_expire_flag", "stratum"])


def check_inputs(columns) -> None:
    leaked = sorted(set(columns) & FORBIDDEN_COLUMNS)
    if leaked:
        raise ValueError(f"forbidden columns in model input: {leaked}")
    unknown = sorted(set(columns) - set(FEATURE_COLUMNS))
    if unknown:
        raise ValueError(f"columns outside FEATURE_COLUMNS: {unknown}")


def load_dataset(split: Optional[str] = None, split_col: str = "split",
                 cohort_dir: Path = COHORT_DIR, tag: str = ""
                 ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """X (FEATURE_COLUMNS, categoricals as pandas category), y (dose_mg) and the grouping
    columns, for one split ("train" / "dev" / "test") or all rows when split is None."""
    cohort_dir = Path(cohort_dir)
    cohort = pd.read_csv(cohort_dir / f"cohort{tag}.csv.gz", dtype={"hadm_id": str})
    features = pd.read_csv(cohort_dir / f"features{tag}.csv.gz", dtype={"hadm_id": str})
    # Model inputs come from features only; the cohort adds the label, split and groups.
    keep = ["hadm_id", LABEL, split_col] + GROUP_COLUMNS
    rows = cohort[keep].merge(features.drop(columns="subject_id"), on="hadm_id", how="inner",
                              validate="one_to_one")
    X = rows[FEATURE_COLUMNS].copy()
    check_inputs(X.columns)
    # Categories are fixed over the whole cohort so train, dev and test share one encoding.
    for col in CATEGORICAL:
        X[col] = X[col].fillna("unknown").astype("category")
    if split is not None:
        keep = rows[split_col].eq(split).to_numpy()
        rows, X = rows[keep], X[keep]
    index = rows["hadm_id"].to_numpy()
    return (X.set_index(index), rows[LABEL].set_axis(index),
            rows[GROUP_COLUMNS].set_axis(index))
