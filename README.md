# PhysioAgent

PhysioAgent predicts the first intravenous furosemide dose a clinician gives a patient with 
chronic kidney disease (CKD). It does this with an LLM that reads the patient record at the
moment of the decision. The LLM can also be given evidence retrieved from a medical knowledge
graph and the doses received by similar past patients. Every component is scored against
the same held-out MIMIC-IV patients, so each one has to show a lower dose error than the
version of the pipeline without it.

## Task

- **Input:** the patient at decision time t0. This covers demographics, pre-dose vitals,
  eGFR, 24-hour urine output, comorbidities, care unit, home oral loop-diuretic dose, and
  the furosemide doses from earlier admissions. Every field carries a timestamp before t0.
- **Output:** one IV furosemide dose in mg, chosen from the ladder
  10/20/40/60/80/100/120/140/160/200 mg.
- **Label:** the dose the clinician actually gave. The score therefore measures agreement
  with clinical practice, not whether the dose was optimal.
- **Metric:** mean absolute error (MAE) in mg. Comparisons between arms use a paired
  bootstrap over patients (2,000 resamples).

Each sample is the first IV bolus of one admission. Later boluses are left out because at
the every-bolus grain, copying the previous dose is exactly right 62% of the time. A model
trained on that would learn to keep the dose rather than to choose one. At the first-dose
grain, copying the last admission's dose does no better than the training median.

## Data

The source is MIMIC-IV 3.1, `inputevents` (ICU).

| Step | Boluses | Patients |
|---|---|---|
| Furosemide IV bolus/push, unit mg, amount > 0 (continuous infusions excluded) | 95,799 | 21,758 |
| Patients with a CKD code (N18.x / 585.x) | 32,356 | 6,658 |
| Weight, height and blood pressure all present | 21,127 | 4,137 |
| Physiologically impossible values removed | 20,976 | 4,119 |
| First bolus of each admission | 5,578 admissions | 4,119 |

The split assigns whole patients 70/15/15, so no patient appears in more than one split
(seed 20260928). It is stratified by CKD stage × prior admission × dose band
(≤20 / 40 / ≥60 mg). Without the dose band, the test median came out at 20 mg against
40 mg for train. Train keeps every admission (3,921 samples, 2,883 patients). Dev (625) and
test (611) keep one sample per patient: the patient's last admission. Earlier admissions
for those patients are kept as history only, never as samples. All three splits have a
40 mg median dose.

The eGFR-band median sets the bar every arm has to beat. It predicts the median training
dose for the patient's eGFR band (≤15 → 80 mg, ≤30 → 60, ≤45 → 40, above 45 → 20). This
gives an MAE of **26.13 mg** on test, against 31.01 mg for the single training median.

## Pipeline

### 1. Scope and EHR concepts (`kg/scope`, `kg/ehr`)

A whitelist of 103 concepts covers the renal, cardiovascular, fluid/electrolyte and diuretic
domains: 35 condition groups, 23 procedure groups and 45 drug classes. Diagnoses and
procedures are mapped ICD → CCS. Drugs are mapped by name to ATC level 3
(`kg/resources/drug_to_atc3.csv`), so no NDC download is needed. Codes outside the whitelist
are dropped. The output is 457,288 admissions expressed as concept sets. The knowledge graph
uses these sets for co-occurrence statistics, and the patient profiles use them as input.

### 2. Knowledge graph (`kg/sources`, `kg/combine.py`, `kg/refine.py`)

The design follows TRACER, with one deliberate change. TRACER's branch that has an LLM
generate relations from its own memory is removed, because nothing it outputs can be
checked against a source. Every edge here traces back to a guideline sentence, a PMID or a
UMLS CUI pair.

| Source | Extraction | Triples | Connectivity |
|---|---|---|---|
| Guideline | Quoted sentences checked against the source PDF, grouped into rules | 87 | None: isolated rules |
| PubMed | An LLM reads abstracts; a triple is kept only if both entities appear verbatim in the abstract | 1,549 | Partial: 78% of nodes are leaves |
| UMLS | Paths between concept pairs that co-occur in MIMIC, within a domain-restricted graph | 1,338 | Yes: 19% of nodes are leaves |

After merging and cleaning, the graph has **2,674 edges**. PubMed is queried once per
concept rather than once per admission. Each concept's top-20 co-occurring concepts decide
which pairs are worth linking. This takes about 1,000 LLM calls instead of 216,000. The
cost is that fewer abstracts discuss both concepts of a pair together.

**UMLS adds structure but not dosing knowledge.** 49% of its edges are taxonomy (is-a,
parent–child) and 25% are anatomical location. None are about treatment, mechanism, effect
or dose. UMLS has no dose-by-patient-state relation at all, so adding more UMLS sources
would not help. Dosing knowledge has to come from guidelines. The guideline flow is being
rebuilt on a fixed corpus of 12 sources (`kg/guideline/manifest.yaml`, each pinned by
sha256), cut into passages by one shared chunker, and linked to concepts by retrieval
rather than by hand.

### 3. Retrieval (`kg/retrieval`)

For each patient, TRACER's procedure is applied to the patient's own visit history:

- **Trajectories:** shortest graph paths (≤3 hops) between the concepts of adjacent
  admissions. The index admission contributes only concepts timestamped before t0. Each
  path is labelled with its time window relative to t0 and scored on two aspects: arguing
  for more diuresis, and arguing for caution.
- **Refinement (KST):** MMR selection followed by an NLI stopping rule. Applying the
  paper's rule literally collapses each set to a single path, so the shipped variant keeps
  at least 4 paths per aspect.
- **Similar patients:** the 5 training patients with the highest Jaccard overlap on their
  last two visits, each shown with the dose they received.

### 4. Prompting (`llm/dose_prompts.py`)

Every arm uses the same system prompt and the same patient block. The **plain** arm stops
there. The **kg** arms add two things: a system section with rules for using the evidence
(measurements override evidence; a relation is not a dose; a guideline range is not an
answer), and an evidence block in the user message containing the time-labelled KG paths
plus the 5 similar patients. The similar patients are added only for rows with no earlier
admission. The model is `deepseek-v4.1-flash` at temperature 0, called through a gateway
with zero data retention.

## Results

All results are on the 611 test patients. Lower MAE is better.

| Arm | MAE overall | With earlier admission (455) | No earlier admission (156) |
|---|---|---|---|
| eGFR-band median (floor) | 26.13 | 29.23 | 17.08 |
| plain | 25.11 | 28.13 | 16.31 |
| kg (timed KG + similar patients) | 26.19 | 29.85 | 15.54 |
| Timed KG + CliCARE guideline alignment | 25.97 | 29.36 | 16.06 |
| Timed KG + 4-line guideline section | — | 29.36 | 14.90 |
| **Timed KG, guideline edges removed** | **24.66** | **28.04** | **14.78** |
| Bayesian anchor (best statistical model) | 24.95 | — | — |

The LLM without evidence beats the constant training median (−5.89 mg, CI
[−7.99, −3.80]). It does not separate from the eGFR-band floor (−1.01 mg, CI
[−2.95, +0.92]).

Adding the full evidence block made results worse overall (+1.08 mg vs plain, CI
[−0.29, +2.42]). The block changed 140 of 611 answers. 103 of those moved up, but only 34
of the upward moves landed closer to the true dose. The model predicted 80 mg 152 times,
against 69 true 80 mg doses. Guideline paths were cited in 48 answers, and they pushed
toward higher doses. With the guideline edges removed, the timed-KG arm gives the lowest MAE
of any configuration (24.66 mg), and it is the best arm in both history subgroups.

Patients with no earlier admission are where evidence helps most consistently. Every KG
variant beats plain on these 156 rows, and they also beat simply copying the 5 neighbours'
median dose (18.11 mg). Patients with an earlier admission are the harder group: no arm
gets more than about 1 mg below the floor there.

The Bayesian anchor is the strongest model that uses no LLM. It starts from the eGFR-band
prior on a log2 dose scale. It then shifts toward each available observation: doses from
earlier admissions (with weights fitted separately for gaps of <30 days, 30–365 days and
>365 days), the home oral dose, and the mean dose of the 5 neighbours. Each observation's
weight is fitted on train. It improves on the floor by −1.18 mg (CI [−2.03, −0.43]), a gain
that resampling does not explain. The best LLM arm scores 0.29 mg lower, but no paired
interval has been computed between the two yet.

## Open problems

1. **Severity scoring.** TRACER's severity term is fixed at a constant. In this setup it
   pushed the model toward higher doses, so it stays off until a per-condition severity
   table is in place. The leading candidate is in-hospital mortality per CCS condition,
   which can be computed from MIMIC.
2. **Guideline knowledge has to reach the prompt in the form it was written.** The
   earlier guideline rules compressed quotes into schema fields. That lost the distinction
   between per-day and per-bolus doses, and pairwise path search never reached a dose stored
   as a leaf node. The rebuilt flow keeps passages verbatim and decides applicability by
   retrieval.
3. **Clinical notes** are not used yet. Adding the top-n notes per patient is the next step
   of TRACER that remains.

## Repository layout

```
config/       settings.yaml (LLM provider and model); per-user overrides in *.local.yaml
kg/scope/     concept whitelist (ontology.yaml) and the scope filter
kg/ehr/       MIMIC → visit concept sets, index-dose samples, patient profiles
kg/sources/   UMLS path flow, PubMed extraction flow
kg/guideline/ guideline corpus: manifest, loader, passages, clusters, concept links
kg/retrieval/ trajectories, KST refinement, NLI, similar patients
kg/resources/ public code mappings: ICD→CCS, ATC, drug-name→ATC3, PubMed query terms
kg/slurm/     cluster jobs for the heavy builds
llm/          OpenAI-compatible client and the dose prompts
schemas/      PatientProfile
utils/        rate limiter, plausibility ranges
```

The evaluation harness and the tests are kept out of this repository.

## Running

Python ≥ 3.12. Modules are imported as `physioagent.*`, so put a directory that contains
`physioagent/` (pointing at this repo) on `PYTHONPATH`.

```bash
pip install -r requirements.txt
cp .env.example .env                       # API keys; never commit .env

python -m physioagent.kg.ehr.build_visits --mimic-root <mimic-iv-3.1>
python -m physioagent.kg.sources.umls_prepare --umls <UMLS META dir>
python -m physioagent.kg.sources.umls_source
python -m physioagent.kg.sources.pubmed_source
python -m physioagent.kg.combine
python -m physioagent.kg.refine
python -m physioagent.kg.ehr.build_doses --cohort <cohort.csv>
python -m physioagent.kg.ehr.patient_profiles --doses --split test
python -m physioagent.kg.retrieval.trajectory --doses --split test --profiles <profiles.json>
```

The visit build, UMLS paths and NLI refinement run on the cluster through the jobs in
`kg/slurm/`.

## Data policy

MIMIC-IV is under the PhysioNet credentialed data use agreement. No patient-level data,
derived cohort, or per-patient output is committed: `kg/data/`, `eval/` and every cohort CSV
are git-ignored. Guideline PDFs and passages are publisher copyright and are ignored too.
Prompts containing MIMIC-derived fields are sent only to an endpoint with zero data
retention.
