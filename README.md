# PhysioAgent

PhysioAgent predicts the first intravenous furosemide bolus a clinician gives in the ICU to a
patient with chronic kidney disease (CKD). An LLM reads the patient record as it stood when the
dose was decided and answers with one dose from a fixed ladder. Every arm is scored against
the dose actually given, on the same held-out MIMIC-IV 3.1 patients.

## Task

- **Sample:** one patient, at their last qualifying admission.
- **t0:** the first IV furosemide bolus/push of that admission in the ICU
  (`inputevents` itemid 221794). No other furosemide may precede it in the ICU: no infusion,
  and no IV dose charted in `emar` one hour or more before t0.
- **Decision time `t_dec`:** the entry time of the IV furosemide order with the same dose in
  the 24 h before t0, else t0. Every input is known before `t_dec`.
- **Label:** the mg given at t0, one of 10/20/40/60/80/100/120/140/160/200.
- **Metric:** MAE in mg (with skill = 1 − MAE / MAE of B0), MAPE, share within ±20%, REC
  curve. Metrics are also reported per analysis axis (CKD evidence, decision context, prior
  furosemide exposure) and per dose band.

The full definition is in [`docs/cohort_spec.md`](docs/cohort_spec.md) and the input
columns in [`docs/feature_schema.md`](docs/feature_schema.md).

## Cohort

| Step | Criterion | Remaining |
|---|---|---|
| S1 | IV furosemide bolus/push in the ICU, mg, amount > 0 | 95,799 boluses |
| S2 | First ICU bolus per admission = t0 | 24,761 admissions |
| S3 | t0 dose on the ladder | 24,621 |
| S4 | Age ≥ 18 at t0 | 24,621 |
| S5 | No ICU furosemide infusion started at or before t0 | 23,878 |
| S5b | No IV furosemide in `emar` between ICU admission and t0 − 1 h | 22,913 |
| S6 | CKD evidence: N18/585 code on an earlier admission, two outpatient eGFR < 60 ≥ 90 days apart with none ≥ 60 between, or a CKD code on this admission | 6,800 |
| S7 | No dialysis in the 24 h before t0 | 6,728 |
| S8 | Not dialysis-dependent (Z99.2/V45.11/N18.6/585.6; kept as a separate 5D stratum of 501 patients) | 6,122 |
| S9 | Creatinine resulted in [`t_dec` − 48 h, `t_dec`) | 5,983 admissions |
| S10 | Last qualifying admission per patient | **5,057 patients** |

Outpatient creatinine means a draw with no `hadm_id` taken outside every ED visit and
hospital stay of the patient; ED draws also carry no `hadm_id`. The split is by patient,
70/15/15 (train 3,540 / dev 758 / test 759), stratified by dose band and CKD evidence. A
temporal sensitivity split tests on `anchor_year_group` 2017–2022. The training median is
20 mg, with 50.2% of training doses at or below 20 mg.

## Arms and baselines

| | Description | Code |
|---|---|---|
| B0 | Constant training median; constant 20 mg and 40 mg are always reported alongside | `eval/run_cohort_dose.py --score` |
| B1 | Guideline rules R1–R5 applied in code | `eval/run_cohort_rules.py` |
| `plain` | LLM, patient block only | `eval/run_cohort_dose.py --arm plain` |
| `plain_calib` | `plain` plus the training dose shares (calibration ablation) | `--arm plain_calib` |
| `guideline` | `plain` plus the guideline rules that fire for the patient, values filled in; the reply names the rules it uses | `--arm guideline` |

The rules (`kg/guideline/t0_rules.py`) quote heart-failure guideline passages and the drug
label verbatim. CKD guidelines (KDIGO 2024, JSN 2023) give no IV loop-diuretic dose. Each rule
is tagged `guideline` when the passage states both the condition and the dose, or
`operationalization` when the threshold or the mapping onto our features is ours.
`--dose-logprobs` additionally reads the dose distribution from the model's next-token
log-probabilities (vLLM only).

## Repository layout

```
config/          settings.yaml: LLM endpoint (a local vLLM server by default)
docs/            cohort_spec.md, feature_schema.md
eval/            run_cohort_dose.py (LLM arms, B0, scoring), run_cohort_rules.py (B1, rule coverage)
eval/slurm/      AAU AI Cloud: setup_env.sh (once), run_cohort_dose.slurm (one arm per job)
kg/ehr/          cohort_build.py, cohort_features.py, dataset.py; the two probes the
                 builder reuses (cohort_funnel_probe.py, cohort_feature_probe.py)
kg/guideline/    t0_rules.py; manifest.yaml, load.py, passages.py rebuild the guideline
                 passages the rules quote
kg/scope/, kg/resources/   concept scope and public code mappings (ICD→CCS, ATC)
llm/             OpenAI-compatible client, cohort prompts, dose ladder
schemas/, utils/ PatientProfile, rate limiter, plausibility ranges
```

## Running

Python ≥ 3.12. Modules are imported as `physioagent.*`, so put a directory containing
`physioagent/` (this repository) on `PYTHONPATH`.

```bash
pip install -r requirements.txt

# Cohort and features (~12 min and ~15 min on a laptop CPU)
python -m physioagent.kg.ehr.cohort_build    --mimic-root <mimic-iv-3.1>
python -m physioagent.kg.ehr.cohort_features --mimic-root <mimic-iv-3.1>
python -m physioagent.kg.ehr.cohort_features --mimic-root <mimic-iv-3.1> --tag _5d

# B1 and rule coverage (no LLM)
python -m physioagent.eval.run_cohort_rules --split dev
```

The LLM arms run as Slurm jobs that start vLLM on the job's GPU (AAU AI Cloud, Singularity
image `/home/container/vllm-openai_latest.sif`). The cluster needs only this repository, the
two cohort files and the model weights, not raw MIMIC-IV:

```bash
# on the machine that built the cohort
ssh ai-fe02.srv.aau.dk 'mkdir -p physioagent/code/physioagent/kg/data/ehr/cohort'
scp kg/data/ehr/cohort/{cohort,features}.csv.gz \
    ai-fe02.srv.aau.dk:physioagent/code/physioagent/kg/data/ehr/cohort/

# on the AI Cloud front-end, with the repository in $HOME/physioagent/code/physioagent
bash $HOME/physioagent/code/physioagent/eval/slurm/setup_env.sh   # once
cd $HOME/physioagent
MODEL=RedHatAI/gemma-3-12b-it-FP8-dynamic ARM=plain sbatch code/physioagent/eval/slurm/run_cohort_dose.slurm
```

A full run is scored at the end of the job. All arms of one comparison use the same model,
GPU type (L40S) and vLLM version; the job log prints the last two.

Prompt and configuration work is done on dev. Test is run once, after everything is fixed.

## Data policy

MIMIC-IV is under the PhysioNet credentialed data use agreement. No patient-level data,
derived cohort or per-patient model output is committed: `kg/data/`, `eval/results/` and
every cohort file are git-ignored. Prompts containing MIMIC-derived values go only to a
model served on our own machine. Guideline PDFs and passages are publisher copyright and are
not distributed; `kg/guideline/manifest.yaml` pins each source by sha256 so it can be
re-downloaded.
