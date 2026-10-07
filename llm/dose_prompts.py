from __future__ import annotations

import json
from typing import Any, Optional

from physioagent.schemas.patient import PatientProfile
from physioagent.utils.model import model_to_dict

DOSE_LADDER = [10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 200.0]

BASE_PROMPT = """You are an experienced nephrologist choosing a single intravenous \
Furosemide BOLUS dose for a CKD inpatient. Return valid JSON only.

TASK
Predict the dose an experienced clinician would actually order for this patient, not the
theoretically optimal dose. You are being scored on agreement with real prescribing.

DOSE MENU — your answer MUST be exactly one of these values, in mg:
  10, 20, 40, 60, 80, 100, 120, 140, 160, 200
The menu is not a linear scale. It doubles up to 80 mg (10 -> 20 -> 40 -> 80) and then
steps by 20 (100 -> 120 -> 160 -> 200); a move of "one step" means one menu entry, not a
fixed number of mg.

WHERE THE ORDERS ACTUALLY SIT
Small doses dominate. 20 mg and 40 mg together account for most orders; 10 mg is routine.
Doses of 100 mg and above are reserved for a small minority. Most CKD inpatients on this
drug receive 20 or 40 mg, so a low dose is the normal answer and not a cautious one.

Begin at 20 mg and move along the menu from there, in EITHER direction.

WHAT MOVES THE DOSE
Upward:
  * Advanced renal impairment. The worse the filtration, the more drug is needed to reach
    the site of action; this is the strongest single upward factor.
  * Low urine output over the last 24h — the strongest fluid-side trigger.
  * Established heart failure.
  * A patient already taking a substantial daily dose, who has adapted to it.
Downward:
  * Preserved renal function.
  * Urine output that is already adequate.
  * No heart failure.
  * No prior loop-diuretic exposure — a drug-naive patient responds to less.

WHAT TO IGNORE
Sex, height, weight, heart rate, diabetes and hypertension carry almost no signal for
this decision. Do not reason from them, and do not let them justify a move.

HOW TO COMBINE FACTORS
Factors DO NOT ACCUMULATE. Identify the single strongest one and move at most ONE step
from the starting point on its account. Two upward factors do not make two steps. Adding
up separate adjustments is the most common way to overshoot badly.

Note that these patients ALL have chronic kidney disease. Renal impairment is therefore
the norm here, not a distinguishing feature, and treating its mere presence as an upward
factor moves every single patient up by one step and answers 40 mg for everyone. Only
impairment that is severe RELATIVE to this population argues upward.

TAIL OVERRIDES — the only routes past a one-step move
  * Severe renal impairment together with low urine output: go to 80 mg or above.
  * A patient already on a substantial daily loop-diuretic dose: match or exceed it.

OUTPUT
{"dose_mg": <one menu value>, "primary_factor": "<the single factor that decided it>",
 "rationale": "<two sentences at most>"}
"""

_SECTIONS = {"plain": ""}


def system_prompt() -> str:
    return BASE_PROMPT


def system_prompt_for(arm: str) -> str:
    if arm not in _SECTIONS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(_SECTIONS)}")
    return BASE_PROMPT + _SECTIONS[arm]


def user_prompt(profile: PatientProfile, evidence_block: Optional[str] = None) -> str:
    payload: dict[str, Any] = {
        "age": profile.age,
        "ckd_stage": profile.ckd_stage,
        "egfr_ml_min_1_73m2": profile.egfr,
        "creatinine_mg_dl": profile.creatinine,
        "urine_output_prior_24h_ml": profile.urine_prior24h_ml,
        "predose_systolic_bp": profile.predose_sbp,
        "heart_failure": profile.has_chf,
        "rising_creatinine_last_7d": profile.aki_creat_safe,
        "care_unit": profile.care_unit,
        "home_oral_furosemide_mg_per_day": profile.current_furosemide_dose,
        "loop_diuretic_naive": profile.loop_naive,
    }
    payload = {k: v for k, v in payload.items() if v is not None}

    parts = ["PATIENT", json.dumps(payload, indent=2)]
    if evidence_block:
        parts += ["", evidence_block]
    parts += ["", "Answer with the JSON object described above."]
    return "\n".join(parts)


N_SLOTS = 4
SLOT_HOURS = 12

WINDOW_LADDER = [0.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 200.0,
                 240.0, 320.0, 400.0]

SCHEDULE_PROMPT = """You are an experienced nephrologist planning the first 48 hours of \
intravenous Furosemide for a CKD inpatient who has just received their first dose. Return \
valid JSON only.

TASK
The 48 hours are divided into four 12-hour windows:
  slot 0 = hours 0-12   slot 1 = hours 12-24   slot 2 = hours 24-36   slot 3 = hours 36-48
For each window, predict the TOTAL milligrams the patient actually received in it, and how
many separate doses made up that total. You are being scored on agreement with real
prescribing, not on the theoretically optimal plan.

A WINDOW TOTAL IS NOT A DOSE. 40 mg once and 20 mg twice are both a 40 mg window. Report
the sum, and report the count separately.

SLOT 0 IS ALWAYS OCCUPIED. The patient has already had a dose in it; predict its total.

MOST WINDOWS AFTER THE FIRST ARE EMPTY. In this population roughly one patient in four
receives anything in hours 12-24, one in five in hours 24-36, and one in six in hours
36-48; more than half receive nothing at all after the first 12 hours. Answering 0 mg for
a later window is the common, correct answer, not a cautious one. Predict a dose in a
later window only when something about THIS patient argues for continuing.

WHAT ARGUES FOR CONTINUING
  * Severe renal impairment with low urine output over the prior 24 h -- the drug is not
    yet achieving decongestion.
  * Established heart failure with ongoing congestion.
  * A patient already on a substantial home loop-diuretic dose, who is adapted to it and
    will not respond to a single bolus.
  * A first window whose total was small relative to the impairment.
WHAT ARGUES FOR STOPPING
  * Preserved filtration, or urine output already adequate before the dose.
  * No heart failure and no prior loop-diuretic exposure.
  * A first window that was already large.

PREVIOUS COURSES
The patient block may list this patient's earlier furosemide courses, oldest first, each
with its four window totals as they were actually given. They are a WEAK guide. In this
population a patient whose previous course continued past the first 12 hours is continued
again only slightly more often than one whose previous course stopped, and the previous
first-window total is often far from this one. Read them as how this patient has been
dosed before -- in particular whether they needed large totals -- and weigh them against
the patient's condition now. Do not copy their pattern of windows. A null window was not
observed -- the patient left hospital, or the next course had already begun.

WHAT TO IGNORE
Sex, height, weight, heart rate, diabetes and hypertension carry almost no signal here.
Do not reason from them.

MENU for a window total, in mg:
  0, 10, 20, 40, 60, 80, 100, 120, 140, 160, 200, 240, 320, 400
0 means no dose was given in that window. This menu is NOT the single-dose menu: a window
total is a sum, so it reaches higher. On real first windows the median is 40 mg, three in
four are 100 mg or less, and one in sixteen exceeds 200 mg -- usually a patient already on
a large home dose. 40 mg and 20 mg together are the most common answers by a wide margin.

HOW TO COMBINE FACTORS
Factors DO NOT ACCUMULATE. Identify the single strongest one. Two reasons to continue do
not justify filling every remaining window.

OUTPUT
{"total_mg": [<slot 0>, <slot 1>, <slot 2>, <slot 3>],
 "n_doses": [<slot 0>, <slot 1>, <slot 2>, <slot 3>],
 "p_dose": [<slot 0>, <slot 1>, <slot 2>, <slot 3>],
 "primary_factor": "<the single factor that decided the plan>",
 "rationale": "<two sentences at most>"}
All three arrays must have exactly four entries. n_doses must be 0 exactly where total_mg
is 0.

p_dose is how likely you think a dose was given in that window, from 0.00 to 1.00. It is
a SEPARATE judgement from the milligrams and must not simply mirror them: two windows you
answered 0 mg for can still differ, one at 0.35 and the other at 0.05, and that difference
is scored. Give slot 0 a p_dose of 1.00 -- it always has a dose. Use the full range; do
not round everything to 0 and 1.
"""

_SCHEDULE_KG_SECTION = """

USING THE RETRIEVED EVIDENCE
The patient block may be followed by EVIDENCE: relations between this patient's own
conditions, taken from published abstracts and medical ontologies. It is background, not
a recommendation, and it is retrieved by similarity, so some lines will be irrelevant to
the decision in front of you.
Rules, in order:
  * The patient's own numbers outrank every line of it. Evidence never overrides
    preserved filtration, adequate urine output, or an already-large first window.
  * A relation is not a dose. Nothing in it tells you a milligram value or how many
    windows to fill.
  * It does not change HOW TO COMBINE FACTORS: still name one strongest factor. Evidence
    that merely agrees with a factor you already had adds nothing and must not promote it.
  * An empty or missing block means nothing was retrieved for this patient. It is not
    evidence that the patient is stable.
"""

_SCHEDULE_SECTIONS = {"plain": "", "kg": _SCHEDULE_KG_SECTION}

_SCHEDULE_STATE_FIELDS = (
    ("age", "age"),
    ("ckd_stage", "ckd_stage"),
    ("egfr", "egfr_ml_min_1_73m2"),
    ("creatinine_mg_dl", "creatinine_mg_dl"),
    ("urine_prior24h_ml", "urine_output_prior_24h_ml"),
    ("predose_sbp", "predose_systolic_bp"),
    ("has_chf", "heart_failure"),
    ("aki_creat_safe", "rising_creatinine_last_7d"),
    ("care_unit", "care_unit"),
)
_SCHEDULE_HISTORY_FIELDS = (
    ("prior_po_daily_mg", "home_oral_furosemide_mg_per_day"),
    ("prior_iv_cum_mg", "iv_furosemide_before_this_episode_mg"),
    ("loop_naive", "loop_diuretic_naive"),
)

MAX_PRIOR_COURSES = 3


def _prior_course(prior: dict[str, Any]) -> dict[str, Any]:
    course: dict[str, Any] = {
        "days_before_this_course": prior["days_before"],
        "same_admission": prior["same_admission"],
        "window_totals_mg": prior["slot_total_mg"],
        "doses_per_window": prior["slot_n_doses"],
    }
    then = prior.get("state") or {}
    if then.get("egfr") is not None:
        course["egfr_then"] = then["egfr"]
    if prior.get("median_urine_post_rate_ml_h") is not None:
        course["urine_output_after_dose_ml_per_h"] = prior["median_urine_post_rate_ml_h"]
    if prior.get("truncated_at_t0"):
        course["cut_short_by_this_course"] = True
    return course


def schedule_system_prompt_for(arm: str) -> str:
    if arm not in _SCHEDULE_SECTIONS:
        raise ValueError(f"unknown arm {arm!r}; "
                         f"expected one of {sorted(_SCHEDULE_SECTIONS)}")
    return SCHEDULE_PROMPT + _SCHEDULE_SECTIONS[arm]


def schedule_user_prompt(state: dict[str, Any], history: dict[str, Any],
                         evidence_block: Optional[str] = None,
                         prior_episodes: Optional[list[dict[str, Any]]] = None) -> str:
    payload: dict[str, Any] = {}
    for source, name in _SCHEDULE_STATE_FIELDS:
        if state.get(source) is not None:
            payload[name] = state[source]
    for source, name in _SCHEDULE_HISTORY_FIELDS:
        if history.get(source) is not None:
            payload[name] = history[source]

    parts = ["PATIENT AT THE TIME OF THE FIRST DOSE", json.dumps(payload, indent=2)]
    if prior_episodes is not None:
        parts += ["", "PREVIOUS FUROSEMIDE COURSES"]
        if prior_episodes:
            shown = prior_episodes[-MAX_PRIOR_COURSES:]
            parts.append(json.dumps({"n_previous_courses": len(prior_episodes),
                                     "most_recent_courses_oldest_first":
                                         [_prior_course(p) for p in shown]}, indent=2))
        else:
            parts.append("none recorded")
    if evidence_block:
        parts += ["", evidence_block]
    parts += ["", "Answer with the JSON object described above."]
    return "\n".join(parts)


def parse_schedule(response: Any) -> Optional[dict[str, list[float]]]:
    if not isinstance(response, dict):
        return None
    totals = response.get("total_mg")
    counts = response.get("n_doses")
    if not isinstance(totals, (list, tuple)) or len(totals) != N_SLOTS:
        return None

    def _number(value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(number, 0.0)

    mg = [_number(v) for v in totals]

    probabilities = response.get("p_dose")
    if isinstance(probabilities, (list, tuple)) and len(probabilities) == N_SLOTS:
        p_dose = [min(max(_number(v), 0.0), 1.0) for v in probabilities]
    else:
        p_dose = None

    if isinstance(counts, (list, tuple)) and len(counts) == N_SLOTS:
        doses = [int(_number(v)) for v in counts]
    else:
        doses = [1 if value > 0 else 0 for value in mg]
    doses = [0 if value == 0 else max(count, 1) for value, count in zip(mg, doses)]
    return {"total_mg": mg, "n_doses": doses, "p_dose": p_dose}


def snap_to_ladder(value: Any) -> Optional[float]:
    try:
        dose = float(value)
    except (TypeError, ValueError):
        return None
    return min(DOSE_LADDER, key=lambda rung: abs(rung - dose))


_PRIOR_ADMISSIONS_SECTION = """PREVIOUS ADMISSIONS
The patient block may list earlier admissions in which this patient received IV
furosemide, oldest first, with the first dose, the number of doses, the largest dose and
the kidney function at the time. They are a WEAK guide. Last time's first dose is usually
not this time's: in this population the new dose is higher about half the time, the same
about a quarter, and the previous admission is typically months ago. Read them for how
large a dose this patient has needed before, and weigh that against the patient now. Do
not copy the previous first dose. "none recorded" means this is the first admission with
IV furosemide on record, which is the common case.

"""

_INDEX_KG_SECTION = """

USING THE RETRIEVED EVIDENCE
The patient block may be followed by EVIDENCE: paths in a knowledge graph between this
patient's own conditions, taken from published abstracts and medical ontologies. It is
background, not a recommendation, and it is retrieved by similarity, so some lines will
be irrelevant. Rules, in order:
  * The patient's own numbers outrank every line of it. Evidence never overrides
    preserved filtration or adequate urine output.
  * A relation between conditions is not a dose. Nothing in it tells you a milligram value.
  * It does not change HOW TO COMBINE FACTORS: still one strongest factor, at most one
    step. Evidence that merely agrees with a factor you already had adds nothing.
  * An empty or missing block means nothing was retrieved for this patient. It is not
    evidence that the patient is stable.
"""

if BASE_PROMPT.count("\nOUTPUT\n") != 1:
    raise RuntimeError("BASE_PROMPT lost its OUTPUT header; the index prompt splices there")
INDEX_DOSE_PROMPT = BASE_PROMPT.replace("\nOUTPUT\n", "\n" + _PRIOR_ADMISSIONS_SECTION
                                        + "OUTPUT\n", 1)

_INDEX_SIM_SECTION = """

USING SIMILAR PATIENTS
The patient block may end with SIMILAR PATIENTS: up to five earlier patients whose most
recent records share the most recorded conditions, procedures and drug classes with this
one, each with the first dose they actually received. Rules, in order:
  * This patient's own numbers come first. Overlapping codes do not mean overlapping kidney
    function or urine output; check those before letting a neighbour's dose count.
  * Neighbours who agree with each other and resemble this patient's numbers are a
    reasonable anchor. Neighbours who disagree carry little information.
  * The overlap score is how many recorded items they share, not how alike they are
    clinically. A high score is not a reason to copy a dose.
"""

_INDEX_SECTIONS = {"plain": "", "kg": _INDEX_KG_SECTION,
                   "kg_sim": _INDEX_KG_SECTION + _INDEX_SIM_SECTION}

_INDEX_HISTORY_FIELDS = (
    ("prior_po_daily_mg", "home_oral_furosemide_mg_per_day"),
    ("loop_naive", "loop_diuretic_naive"),
)
MAX_PRIOR_ADMISSIONS = 3


def index_system_prompt_for(arm: str) -> str:
    if arm not in _INDEX_SECTIONS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(_INDEX_SECTIONS)}")
    return INDEX_DOSE_PROMPT + _INDEX_SECTIONS[arm]


def _prior_admission(prior: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "days_before_this_admission": prior["days_before"],
        "first_dose_mg": prior["first_dose_mg"],
        "n_doses": prior["n_doses"],
        "largest_dose_mg": prior["max_dose_mg"],
    }
    then = prior.get("state") or {}
    if then.get("egfr") is not None:
        out["egfr_then"] = then["egfr"]
    if prior.get("median_urine_post_rate_ml_h") is not None:
        out["urine_output_after_dose_ml_per_h"] = prior["median_urine_post_rate_ml_h"]
    return out


def index_user_prompt(state: dict[str, Any], history: dict[str, Any],
                      prior_admissions: list[dict[str, Any]],
                      evidence_block: Optional[str] = None) -> str:
    payload: dict[str, Any] = {}
    for source, name in _SCHEDULE_STATE_FIELDS:
        if state.get(source) is not None:
            payload[name] = state[source]
    if state.get("urine_prior24h_present") == 0:
        payload.pop("urine_output_prior_24h_ml", None)
    for source, name in _INDEX_HISTORY_FIELDS:
        if history.get(source) is not None:
            payload[name] = history[source]

    parts = ["PATIENT AT THE TIME OF THE DOSE", json.dumps(payload, indent=2),
             "", "PREVIOUS ADMISSIONS WITH IV FUROSEMIDE"]
    if prior_admissions:
        shown = prior_admissions[-MAX_PRIOR_ADMISSIONS:]
        parts.append(json.dumps({"n_previous_admissions": len(prior_admissions),
                                 "most_recent_oldest_first":
                                     [_prior_admission(p) for p in shown]}, indent=2))
    else:
        parts.append("none recorded")
    if evidence_block:
        parts += ["", evidence_block]
    parts += ["", "Answer with the JSON object described above."]
    return "\n".join(parts)
