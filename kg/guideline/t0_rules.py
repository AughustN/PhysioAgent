"""Guideline rules for the first ICU IV furosemide bolus (docs/cohort_spec.md), run in code.

Each rule turns guideline statements into a condition on kg/ehr/dataset.FEATURE_COLUMNS and
an action: a dose range in mg, or a direction ("prefer the higher dose"). Every rule quotes
the passages it rests on, verbatim from kg/data/guideline/guideline_passages.jsonl. The tier
says how far the rule is from the text:

    guideline           the passages state both the condition and the dose
    operationalization  the passages state the idea; the threshold, or the mapping onto our
                        features, is ours and is spelled out in `ours`

Only adult IV statements are used; paediatric, mg/kg and cirrhosis oral regimens are out of
scope. CKD guidelines (KDIGO 2024, JSN 2023) give no IV loop-diuretic dose, so every rule
comes from heart-failure guidance or the drug label.

For one patient a rule is "fires", "not_applicable", or "unknown" (an input it needs is
missing, so the rule can be neither applied nor ruled out).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import pandas as pd

LADDER = (10, 20, 40, 60, 80, 100, 120, 140, 160, 200)
# ESC 2021 Fig. 13 calls the response adequate at a urine output >= 100-150 mL/h after 6 h.
INADEQUATE_UO_6H_ML = 600
# "the daily oral dose taken by the patient before admission": a discharge prescription
# older than this is not read as the current regimen.
ORAL_CURRENT_DAYS = 365
# DailyMed: a further dose "not sooner than 2 hours after the previous dose".
REDOSE_MIN_H = 2
# Below this eGFR, B1 takes the high end of a range (R2 says "higher", not how much).
LOW_EGFR = 30
PRE_EXISTING_DYSFUNCTION_EGFR = 60

FIRES, NOT_APPLICABLE, UNKNOWN = "fires", "not_applicable", "unknown"


@dataclass(frozen=True)
class Citation:
    passage_id: str
    quote: str


@dataclass(frozen=True)
class Rule:
    id: str
    title: str
    tier: str
    citations: tuple[Citation, ...]
    ours: str = ""


@dataclass(frozen=True)
class Firing:
    rule_id: str
    status: str
    low: Optional[int] = None
    high: Optional[int] = None
    direction: Optional[str] = None
    text: str = ""


RULES = {
    "R1": Rule(
        "R1", "Loop-diuretic naive: 20-40 mg IV", "guideline",
        (Citation("esc_hf_2021:0170", "If the patient was not on oral diuretics, a starting "
                  "dose of 20–40 mg of furosemide, or a bolus of 10–20 mg i.v. torasemide, "
                  "can be used."),
         Citation("hfa_diuretics_2019:0022", "(i) diuretic naïve patients with acute heart "
                  "failure should receive a dose of intravenous furosemide of at least 20–40 "
                  "mg furosemide equivalent."),
         Citation("dailymed_furosemide:0027", "Initial dose is 20 mg to 40 mg once given "
                  "intramuscularly or intravenously.")),
        ours="naive = no recorded oral or IV furosemide, with the ward record available for "
             "this admission and for earlier admissions if there were any. Home medication "
             "is not recorded in MIMIC-IV, so naive means naive on record."),
    "R2": Rule(
        "R2", "Pre-existing kidney dysfunction: prefer the higher dose", "operationalization",
        (Citation("hfa_diuretics_2019:0022", "The higher dose should be considered in "
                  "patients with pre-existing kidney dysfunction as it is associated with a "
                  "rightward shift in the dose–response curve."),
         Citation("aasld_ascites_2021:0013", "Patients with chronic kidney disease (CKD) in "
                  "general are treated with higher doses of loop diuretics")),
        ours=f"pre-existing dysfunction = baseline eGFR (else current) < "
             f"{PRE_EXISTING_DYSFUNCTION_EGFR}. The text gives a direction, no amount."),
    "R3": Rule(
        "R3", "On an oral loop diuretic: IV 1-2 times the daily oral dose", "guideline",
        (Citation("esc_hf_2021:0170", "Diuretic treatment should be started with an initial "
                  "i.v. dose of furosemide, or equivalent dose of bumetanide or torasemide, "
                  "corresponding to 1–2 times the daily oral dose taken by the patient before "
                  "admission."),
         Citation("hfa_diuretics_2019:0022", "(ii) Patients on an ambulatory diuretic regimen "
                  "should receive at least the pre-existing oral dose")),
        ours=f"daily oral dose = oral_recorded_daily_mg, recorded <= {ORAL_CURRENT_DAYS} days "
             "before the decision."),
    "R4": Rule(
        "R4", "IV doses already given this admission: raise the dose", "operationalization",
        (Citation("dailymed_furosemide:0031", "The dose may be raised by 20 mg and "
                  "administered not sooner than 2 hours after the previous dose until the "
                  "desired diuretic effect has been obtained."),
         Citation("esc_hf_2021:0173", "Double dose i.v. until maximum i.v. dose"),
         Citation("acc_aha_hfsa_2022:0239", "In patients hospitalized with HF when diuresis "
                  "is inadequate to relieve symptoms and signs of congestion, it is "
                  "reasonable to intensify the diuretic regimen using either: a. higher doses "
                  "of intravenous loop diuretics")),
        ours="urine output is not charted on the ward, so a further bolus after a ward IV "
             f"dose given >= {REDOSE_MIN_H} h earlier is read as a response judged "
             "inadequate; range from +20 mg (label) to double (ESC)."),
    "R5": Rule(
        "R5", "Response to the first ICU bolus of the last admission that had one",
        "operationalization",
        (Citation("esc_hf_2021:0173", "Urine output after 6 h ≥100–150 mL/h Double dose "
                  "i.v. until maximum i.v. dosea Repeat similar dose i.v. every 12 h"),),
        ours=f"ESC applies within one episode; carrying it over from an earlier admission is "
             f"ours. Inadequate = urine output < {INADEQUATE_UO_6H_ML} mL in the 6 h after "
             "that bolus: double it; otherwise repeat it."),
}
RANGE_PRIORITY = ("R4", "R3", "R5", "R1")   # most recent, most direct evidence first


def _num(row: Mapping[str, Any], col: str) -> Optional[float]:
    value = row.get(col)
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    return float(value)


def _flag(row: Mapping[str, Any], col: str) -> bool:
    return bool(_num(row, col))


def on_ladder(low: float, high: float) -> tuple[int, int]:
    """The ladder rungs inside [low, high]; the nearest rung when none is, capped at 200."""
    inside = [d for d in LADDER if low - 1e-6 <= d <= high + 1e-6]
    if inside:
        return inside[0], inside[-1]
    rung = min(LADDER, key=lambda d: abs(d - (low + high) / 2))
    return rung, rung


def exposure(row: Mapping[str, Any]) -> str:
    """'exposed', 'naive' or 'unknown' on record, from model inputs only (A3 is a label)."""
    ward_record = _flag(row, "src_emar_this_adm")
    prior_record = (_num(row, "n_prior_admissions") or 0) == 0 or _flag(row, "src_emar_prior")
    oral = str(row.get("oral_recorded_source") or "unknown") != "unknown"
    if (oral or (_num(row, "prior_iv_n") or 0) > 0 or _flag(row, "icu_infusion_before")
            or (ward_record and (_num(row, "ward_iv_n") or 0) > 0)):
        return "exposed"
    return "naive" if ward_record and prior_record else "unknown"


def r1(row: Mapping[str, Any]) -> Firing:
    state = exposure(row)
    if state == "naive":
        return Firing("R1", FIRES, 20, 40,
                      text="no oral or IV furosemide on record → IV 20–40 mg")
    return Firing("R1", NOT_APPLICABLE if state == "exposed" else UNKNOWN)


def kidney_egfr(row: Mapping[str, Any]) -> Optional[float]:
    baseline = _num(row, "egfr_baseline")
    return baseline if baseline is not None else _num(row, "egfr_t_dec")


def r2(row: Mapping[str, Any]) -> Firing:
    egfr = kidney_egfr(row)
    if egfr is None:
        return Firing("R2", UNKNOWN)
    if egfr >= PRE_EXISTING_DYSFUNCTION_EGFR:
        return Firing("R2", NOT_APPLICABLE)
    return Firing("R2", FIRES, direction="higher",
                  text=f"eGFR {egfr:.0f} (pre-existing kidney dysfunction) → "
                       "prefer the higher dose")


def r3(row: Mapping[str, Any]) -> Firing:
    if str(row.get("oral_recorded_source") or "unknown") == "unknown":
        return Firing("R3", NOT_APPLICABLE)
    daily, days = _num(row, "oral_recorded_daily_mg"), _num(row, "oral_recorded_days")
    if daily is None or days is None or days > ORAL_CURRENT_DAYS:
        return Firing("R3", UNKNOWN)
    low, high = on_ladder(daily, 2 * daily)
    return Firing("R3", FIRES, low, high,
                  text=f"oral {daily:g} mg/day recorded {days:.0f} days ago → "
                       f"IV {low}–{high} mg")


def r4(row: Mapping[str, Any]) -> Firing:
    if not _flag(row, "src_emar_this_adm"):
        return Firing("R4", UNKNOWN)
    last, ago = _num(row, "ward_iv_last_mg"), _num(row, "ward_iv_last_hours_ago")
    if not (_num(row, "ward_iv_n") or 0) or last is None:
        return Firing("R4", NOT_APPLICABLE)
    if ago is None or ago < REDOSE_MIN_H:
        return Firing("R4", NOT_APPLICABLE)
    low, high = on_ladder(last + 20, 2 * last)
    return Firing("R4", FIRES, low, high,
                  text=f"ward IV {last:g} mg {ago:.0f} h ago → IV {low}–{high} mg")


def r5(row: Mapping[str, Any]) -> Firing:
    dose, uo = _num(row, "prior_icu_first_bolus_mg"), _num(row, "prior_icu_first_bolus_uo_6h_ml")
    if dose is None:
        return Firing("R5", NOT_APPLICABLE)
    if uo is None:
        return Firing("R5", UNKNOWN)
    target = 2 * dose if uo < INADEQUATE_UO_6H_ML else dose
    low, high = on_ladder(target, target)
    verdict = "inadequate → double" if uo < INADEQUATE_UO_6H_ML else "adequate → repeat"
    return Firing("R5", FIRES, low, high,
                  text=f"last admission's first ICU bolus {dose:g} mg, urine {uo:.0f} mL in 6 h "
                       f"({verdict}) → IV {low} mg")


CHECKS = {"R1": r1, "R2": r2, "R3": r3, "R4": r4, "R5": r5}


def evaluate(row: Mapping[str, Any]) -> dict[str, Firing]:
    return {rule_id: check(row) for rule_id, check in CHECKS.items()}


def b1_dose(firings: Mapping[str, Firing], row: Mapping[str, Any]) -> tuple[Optional[int], str]:
    """B1: the range of the highest-priority rule that fires; its high end when R2 fires at an
    eGFR below LOW_EGFR, else its low end. (None, "") when no range rule fires."""
    for rule_id in RANGE_PRIORITY:
        f = firings[rule_id]
        if f.status == FIRES:
            egfr = kidney_egfr(row)
            high = firings["R2"].status == FIRES and egfr is not None and egfr < LOW_EGFR
            return (f.high if high else f.low), rule_id
    return None, ""
