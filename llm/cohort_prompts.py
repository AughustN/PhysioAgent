"""Prompts for the T0 dose task on the cohort of docs/cohort_spec.md (plain arm).

The user prompt shows the LLM the same columns B2 sees (kg/ehr/dataset.FEATURE_COLUMNS),
renamed and grouped so they read without the schema. The system prompt carries no number
measured on any cohort: what a plain LLM brings is its own clinical knowledge.
"""
from __future__ import annotations

import json
import math
from typing import Any, Mapping, Optional

import pandas as pd

PLAIN_PROMPT = """You are an experienced critical-care physician choosing the first \
intravenous furosemide BOLUS for a patient with chronic kidney disease who is in the ICU. \
Return valid JSON only.

TASK
Predict the dose the treating team actually ordered at this decision point, not the
theoretically optimal dose. You are scored on agreement with real prescribing.

DOSE MENU -- your answer MUST be exactly one of these values, in mg:
  10, 20, 40, 60, 80, 100, 120, 140, 160, 200

THE PATIENT BLOCK
Everything in it was recorded before the order was written. A field that is absent was
not recorded: it does not mean normal, and it does not mean zero.
  * furosemide_this_admission covers the ward, before ICU admission. Its dose fields are
    listed only when the ward medication record exists for this admission.
  * prior_admissions summarises earlier hospital stays of this patient. "none recorded"
    means this is the first admission on record. Diagnoses are those coded at earlier
    stays; the current stay's diagnoses are not available.
  * Responses to earlier doses (urine output after a dose, whether the dose was raised
    within 24 h) are listed only when that window had closed before this decision.

OUTPUT
{"reasoning": "<at most three sentences, citing this patient's values>",
 "primary_factor": "<the single factor that decided the dose>",
 "dose_mg": <one menu value>}
"""

# (section, column, key shown to the LLM). Order inside a section is the order shown.
FIELDS = [
    ("patient", "age", "age_years"),
    ("patient", "sex_female", "sex"),
    ("patient", "weight_kg", "weight_kg"),
    ("patient", "transplant_prior", "kidney_transplant_recipient"),
    ("patient", "admission_type", "admission_type"),
    ("patient", "via_ed", "admitted_through_emergency_department"),
    ("patient", "service_at_t_dec", "hospital_service"),
    ("patient", "first_careunit", "icu_unit"),
    ("patient", "code_status", "code_status"),
    ("patient", "hosp_hours_before", "hours_since_hospital_admission"),
    ("patient", "icu_hours_before", "hours_since_icu_admission"),
    ("patient", "fluid_in_24h_ml", "charted_iv_input_last_24h_ml"),
    ("vitals_latest_6h", "hr", "heart_rate_bpm"),
    ("vitals_latest_6h", "sbp", "systolic_bp_mmhg"),
    ("vitals_latest_6h", "dbp", "diastolic_bp_mmhg"),
    ("vitals_latest_6h", "map", "mean_arterial_pressure_mmhg"),
    ("vitals_latest_6h", "sbp_min_6h", "lowest_systolic_bp_6h_mmhg"),
    ("vitals_latest_6h", "map_min_6h", "lowest_mean_arterial_pressure_6h_mmhg"),
    ("vitals_latest_6h", "spo2", "spo2_pct"),
    ("vitals_latest_6h", "resp_rate", "respiratory_rate_per_min"),
    ("vitals_latest_6h", "temp_c", "temperature_c"),
    ("labs_latest_48h", "creatinine", "creatinine_mg_dl"),
    ("labs_latest_48h", "creatinine_hours_ago", "creatinine_drawn_hours_ago"),
    ("labs_latest_48h", "creatinine_change_48h", "creatinine_change_48h_mg_dl"),
    ("labs_latest_48h", "bun", "bun_mg_dl"),
    ("labs_latest_48h", "sodium", "sodium_meq_l"),
    ("labs_latest_48h", "potassium", "potassium_meq_l"),
    ("labs_latest_48h", "chloride", "chloride_meq_l"),
    ("labs_latest_48h", "bicarbonate", "bicarbonate_meq_l"),
    ("labs_latest_48h", "magnesium", "magnesium_mg_dl"),
    ("labs_latest_48h", "glucose", "glucose_mg_dl"),
    ("labs_latest_48h", "hemoglobin", "hemoglobin_g_dl"),
    ("labs_latest_48h", "albumin", "albumin_g_dl"),
    ("labs_latest_48h", "lactate", "lactate_mmol_l"),
    ("labs_latest_48h", "ntprobnp", "nt_probnp_pg_ml"),
    ("kidney", "egfr_t_dec", "egfr_now_ml_min_1_73m2"),
    ("kidney", "egfr_baseline", "outpatient_egfr_baseline_ml_min_1_73m2"),
    ("kidney", "egfr_baseline_days", "baseline_measured_days_before_admission"),
    ("kidney", "egfr_delta", "egfr_now_minus_baseline"),
    ("kidney", "ckd_stage", "ckd_stage"),
    ("kidney", "stage_from_t_dec", "ckd_stage_from_current_egfr_only"),
    ("urine_output", "uo_6h_ml", "urine_output_last_6h_ml"),
    ("urine_output", "uo_24h_ml", "urine_output_last_24h_ml"),
    ("urine_output", "uo_window_6h_h", "icu_hours_inside_the_6h_window"),
    ("urine_output", "uo_rate_6h_ml_kg_h", "urine_output_6h_ml_per_kg_per_h"),
    ("furosemide_this_admission", "src_emar_this_adm", "ward_medication_record_available"),
    ("furosemide_this_admission", "ward_iv_n", "ward_iv_doses"),
    ("furosemide_this_admission", "ward_iv_last_mg", "ward_iv_last_dose_mg"),
    ("furosemide_this_admission", "ward_iv_last_hours_ago", "ward_iv_last_dose_hours_ago"),
    ("furosemide_this_admission", "ward_iv_max_mg", "ward_iv_largest_dose_mg"),
    ("furosemide_this_admission", "ward_iv_total_24h_mg", "ward_iv_total_last_24h_mg"),
    ("furosemide_this_admission", "oral_this_adm_last_mg", "ward_oral_last_dose_mg"),
    ("furosemide_this_admission", "icu_infusion_before", "icu_infusion_given_and_stopped"),
    ("oral_loop_diuretic", "oral_recorded_mg", "last_recorded_oral_dose_mg"),
    ("oral_loop_diuretic", "oral_recorded_source", "recorded_in"),
    ("oral_loop_diuretic", "oral_recorded_days", "recorded_days_ago"),
    ("prior_admissions", "n_prior_admissions", "n_prior_admissions"),
    ("prior_admissions", "days_since_last_discharge", "days_since_last_discharge"),
    ("prior_admissions", "src_emar_prior", "ward_medication_record_available"),
    ("prior_admissions", "prior_furo_admissions", "admissions_with_furosemide"),
    ("prior_admissions", "prior_last_furo_days", "days_since_last_furosemide"),
    ("prior_admissions", "prior_iv_n", "iv_doses"),
    ("prior_admissions", "prior_iv_last_mg", "iv_last_dose_mg"),
    ("prior_admissions", "prior_iv_median_mg", "iv_median_dose_mg"),
    ("prior_admissions", "prior_iv_max_mg", "iv_largest_dose_mg"),
    ("prior_admissions", "prior_infusion", "ever_on_continuous_infusion"),
    ("prior_admissions", "prior_icu_first_bolus_mg", "last_first_icu_bolus_mg"),
    ("prior_admissions", "prior_icu_first_bolus_days", "last_first_icu_bolus_days_ago"),
    ("prior_admissions", "prior_icu_first_bolus_uo_6h_ml",
     "urine_output_6h_after_that_bolus_ml"),
    ("prior_admissions", "prior_icu_first_bolus_escalated", "dose_raised_within_24h_after_it"),
    ("prior_admissions", "prior_hf", "heart_failure_coded"),
    ("prior_admissions", "prior_cirrhosis", "cirrhosis_coded"),
    ("prior_admissions", "prior_diabetes", "diabetes_coded"),
]
SECTIONS = list(dict.fromkeys(section for section, _, _ in FIELDS))
BOOLEAN = {"sex_female", "transplant_prior", "via_ed", "stage_from_t_dec", "src_emar_this_adm",
           "icu_infusion_before", "src_emar_prior", "prior_infusion",
           "prior_icu_first_bolus_escalated", "prior_hf", "prior_cirrhosis", "prior_diabetes"}
# Shown only when the ward record exists: without it, 0 doses means "not observable".
WARD_DOSE_COLUMNS = {"ward_iv_n", "ward_iv_last_mg", "ward_iv_last_hours_ago",
                     "ward_iv_max_mg", "ward_iv_total_24h_mg", "oral_this_adm_last_mg"}


def _missing(value: Any) -> bool:
    return value is None or (not isinstance(value, str) and bool(pd.isna(value)))


def _number(value: float) -> float | int:
    value = round(float(value), 2 if abs(value) < 1 else 1 if abs(value) < 100 else 0)
    return int(value) if value.is_integer() else value


def _value(column: str, value: Any) -> Any:
    if column == "sex_female":
        return "female" if int(value) else "male"
    if column in BOOLEAN:
        return bool(int(value))
    if isinstance(value, str):
        return value
    return _number(value)


def patient_block(row: Mapping[str, Any]) -> dict[str, Any]:
    """One feature row (FEATURE_COLUMNS) as the nested dict shown to the LLM."""
    ward_record = not _missing(row.get("src_emar_this_adm")) and int(row["src_emar_this_adm"])
    block: dict[str, Any] = {section: {} for section in SECTIONS}
    for section, column, key in FIELDS:
        value = row.get(column)
        if _missing(value) or (isinstance(value, str) and value == "unknown"):
            continue
        if column in WARD_DOSE_COLUMNS and not ward_record:
            continue
        block[section][key] = _value(column, value)
    if not block["prior_admissions"].get("n_prior_admissions"):
        block["prior_admissions"] = "none recorded"
    return {section: fields for section, fields in block.items() if fields}


def plain_system_prompt() -> str:
    return PLAIN_PROMPT


def plain_user_prompt(row: Mapping[str, Any]) -> str:
    return "\n".join(["PATIENT AT THE DECISION POINT", json.dumps(patient_block(row), indent=2),
                      "", "Answer with the JSON object described above."])


def parse_dose(response: Any) -> Optional[float]:
    """The dose_mg of a reply as a float, or None if it is missing or not a number."""
    if not isinstance(response, dict):
        return None
    try:
        dose = float(response.get("dose_mg"))
    except (TypeError, ValueError):
        return None
    return dose if math.isfinite(dose) and dose > 0 else None

