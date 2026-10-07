# PhysioAgent — 48-Hour Furosemide Dosing Sequence: First Results

Companion to `mimic-exploration/report.md`, which covers the earlier single-dose task. That
task is retired along with the Pulse engine; this report covers the task that replaced it.

Every number here comes from an artifact in the repo, named at the point of use. Raw
measurements and the run-by-run log live in `eval/results/seq/first_results_20260922.md`
and `eval/results/seq/kg_arm_inputs_20260922.md`; this file is the argument those numbers
support.

---

## 1. The task

One episode is one course of furosemide for one patient. Given the patient at the moment of
the first dose, the model plans the next **48 hours as four 12-hour windows**, answering per
window with a total in milligrams, a dose count, and `p_dose`, its own probability that any
dose was given. Ground truth is what the clinicians actually did — concordance with
practice, not dose optimality.

A new course starts when the previous one has been quiet for `gap_hours = 168` (7 days), so
one admission can contribute more than one episode.

**Why the headline metric is not MAE.** The decision — *is there a dose in this window* —
is what a clinician acts on, and over slots 1-3 it is a rare positive (29.6% / 23.4% /
19.6% occupied on the test split). Accuracy and MAE both reward answering "nothing" three
times: `eval/sequence_baselines.py` measures that non-answer at macro-F1 0.431 and MAE
18.33 mg without looking at the patient. So the decision is scored with AUPRC and macro-F1,
and MAE is reported in two halves that are different questions: slot 0 is *how much to
start with*, slots 1-3 are *how much to continue with*.

**Unobserved is not zero.** A window after discharge or death carries no label; those cells
are dropped from every statistic rather than scored as 0 mg, which would pay the model for
predicting the discharge. 246 of 5,937 episodes are censored inside 48 h.

## 2. Data

`kg/data/sequences_report.json`, built by `kg/ehr/build_sequences.py`:

| | episodes | patients | slot 1 / 2 / 3 occupancy | censored |
|---|---|---|---|---|
| train | 4,161 | 2,897 | 30.1% / 23.4% / 17.1% | 4.1% |
| dev | 888 | 633 | 31.4% / 23.2% / 16.9% | 4.1% |
| test | 888 | 589 | 29.7% / 23.4% / 19.6% | 4.5% |

5,937 episodes over 20,976 dose events. The split is **patient-disjoint**, 70/15/15,
`SPLIT_SEED = 20260921`, stratified on `ckd_stage` and on whether the patient received a
further dose after slot 0. Disjointness was verified rather than assumed: 0 shared episode
ids and 0 shared subject ids between test and dev.

Dev exists so that prompts and thresholds are tuned off the reported split. Test was scored
once.

Two facts about the label worth stating: the median window total, when a window is
occupied, is 40 mg at every slot; and 3,223 episodes have exactly one occupied slot, so
"one dose and stop" is the single most common true schedule.

### Two corrections that invalidated earlier numbers

Both landed 2026-09-21, and any sequence result measured before that date is void.

1. The test split used to be *patients inside the Pulse simulator's acceptance envelope*.
   Pulse was dropped but its split rule stayed, leaving test a different population from
   train: median age 57 vs 70, BMI 25.2 vs 29.3, SBP 111 vs 130, slot-1 occupancy 25.2% vs
   30.7%. That is a label shift in the headline metric with no model in it.
2. `episode_end` took `min(deathtime, dischtime)`. MIMIC rounds `dischtime` to midnight for
   many in-hospital deaths, which ended the window before doses that demonstrably happened.
   Censoring now takes `deathtime` first and `dischtime` only as a fallback.

## 3. The knowledge graph

Two source flows, each checkable, combined by `kg/combine.py` and merged by
`kg/refine.py`:

| flow | edges read | in raw graph |
|---|---|---|
| PubMed abstracts | 1,549 | 1,357 |
| UMLS walk | 1,338 | 1,338 |

Raw: 2,695 edges over 1,559 entities and 917 relations. After merging entities and relations
that mean the same thing: **2,599 edges, 1,426 entities, 726 relations** (`kg_refined.jsonl`,
`refine_report.json`).

**Agreement between sources is designed as a signal and currently delivers nothing.**
All 2,599 edges are asserted by exactly one flow. Any future claim that rests on cross-source
corroboration has to reckon with that number first.

### Where this departs from TRACER, deliberately

- **TRACER's LLM generation branch is gone.** `llm_source.py` asks a model to invent
  relations from its own memory, and nothing it produces can be checked. The PubMed flow
  keeps a model but only as a reader: every entity in an emitted triple must appear in the
  abstract verbatim or the triple is dropped. The consequence is a graph that is small and
  citable rather than large and unfalsifiable. The concept-set stage this replaces was
  costed at 216,701 LLM calls and never paid for.
- **PubMed is queried per concept, not per visit.** TRACER's unit is a whole admission's
  concept set, which on this cohort is ~202k queries for a vocabulary admitting at most
  4,560 distinct pairs. Per concept costs ~1,000 calls.

## 4. Patient profiles (TRACER 4.3, input side)

`kg/ehr/patient_profiles.py` turns each episode into the visit history the retriever reads,
keyed by `episode_id` — not by admission, because 7.8% of episodes are a second or later
course inside one admission and an admission-keyed profile would cut their history at the
wrong moment.

Built on the cluster (`kg/slurm/kg_s0a_profiles.slurm`), both splits:

| | profiles | index visit carries concepts | concepts/profile (median) | with >=1 prior admission |
|---|---|---|---|---|
| test | 888 | 888 (100%) | 43 | 674 (75.9%) |
| dev | 888 | 888 (100%) | 40 | 674 (75.9%) |

**The leakage rule that matters.** `diagnoses_icd` is coded at discharge, so every ICD code
on the index admission was assigned *after* the dose being predicted. The index visit
therefore contributes only drugs and procedures that `build_visits` timestamped strictly
before t0 — an untimed concept is dropped, because an unknown time is not a safe time — and
its diagnoses are excluded except for a short list of chronic conditions the clinician knew
beforehand, or codes also present on an earlier admission of the same patient. Prior
admissions are unrestricted: they closed before t0.

**This changes TRACER's footnote 4.** The paper excludes single-visit patients. On the older
admission-grain cohort that would have discarded about three quarters of the evaluation set
(26.2% had prior history). On the episode grain with all prior admissions in scope, 75.9%
have history, so the paper-faithful cohort costs **24.1%** of the eval set. The deviation we
inherited from the released code is much cheaper to avoid here than the module's own
docstring implies. `--require-prior` produces that cohort.

## 5. Retrieval and the evidence block

`kg/retrieval/trajectory.py` takes each profile's concepts, finds shortest paths up to 3
hops between every pair, scores each path for two aspects, and keeps the top ρ = 0.3 of
each. On test: 240,970 pairs, 294,847 paths found, 176,910 kept, and 888/888 episodes have
at least one path.

`eval/evidence.py` renders the block that is the **entire** difference between the `kg` arm
and the control. Four decisions, each recorded in that module with the measurement that
forced it:

1. **A path in both aspects is kept once, under its higher score.** `top_rho` follows the
   paper and keeps the top ρ of each aspect independently, so 44.4% of kept test paths sit
   in both lists. Rendered raw, the model would read one relation labelled two opposite ways.
2. **Triples with relation names**, not node chains: the relation is what the graph adds
   over the patient's own condition list, which the prompt already carries.
3. **Eight paths, balanced across aspects**, four from each before either takes the rest.
4. **Eight distinct facts, not eight variants of two.** Score alone filled every slot from
   one edge's neighbourhood — four protective lines all reading
   `chronic kidney disease -[conveys a high risk for]-> X`. A path must now contribute a
   triple the block does not already state, and at most two paths may share a
   (subject, relation) pair. Distinct facts per block went 4 → 12 at the same token cost.

Result: **707 distinct blocks over 888 test episodes**, the most common covering 2.3%,
median 8 lines. With the stand-in concept set that preceded the real profiles there were
**7 distinct blocks** and the most common covered 49.7% — a kg arm built on that would have
measured a boilerplate suffix, not a knowledge graph.

## 6. Experimental setup

Both arms share `SCHEDULE_PROMPT` byte for byte; the `kg` arm appends exactly one section
(`llm/dose_prompts._SCHEDULE_KG_SECTION`) and carries the block in its user message. Same
model, same temperature 0, same episodes, scored together in one session.

- Model `gpt-oss-120b`, served through the cheaperinference gateway with `zdr: true` on
  every request. `served_model` returned `openai/gpt-oss-120b` on all 1,776 rows.
- Before committing an arm, the gateway was checked for prompt tampering: `X-Ci-Techniques`
  empty and `X-Ci-Tokens-Saved` zero on every row, and `prompt_tokens` 1,108-1,181 against
  the ~1,150 the same prompt measures elsewhere. Every row stores `served_model`,
  `prompt_tokens`, `completion_tokens` and `request_id`, so the check cannot lapse mid-run
  and a fall in prompt tokens is a visible drift alarm.
- 6 workers at `--rpm 30`. plain 888/888 ok on the first attempt; kg 888/888 ok.

## 7. Results

888 test episodes, 2,611 scorable cells over slots 1-3, 633 positives (24.2%).

| arm | AUPRC | macro-F1 | precision | recall | pred+ rate | MAE slot 0 | MAE slots 1-3 |
|---|---|---|---|---|---|---|---|
| floor | 0.242 const / **0.272 prevalence** | 0.431 | — | — | 0.242 true | 55.33 | 18.33 |
| plain | 0.2615 | 0.5132 | 0.2630 | 0.3349 | 0.3087 | 49.65 | 23.32 |
| kg | **0.2700** | **0.5337** | **0.2928** | 0.3002 | **0.2486** | 49.98 | **21.97** |

Paired bootstrap over episodes, 2,000 resamples:

| delta (kg − plain) | estimate | 95% CI |
|---|---|---|
| AUPRC | +0.0088 | [−0.0054, +0.0223] |
| macro-F1 | **+0.0205** | **[+0.0057, +0.0357]** |
| MAE slots 1-3 | **−1.34 mg** | **[−1.79, −0.90]** |

**The evidence block helps the decision and not the ranking.** macro-F1 and MAE over slots
1-3 improve beyond resampling noise. AUPRC moves the same way and its interval spans zero.

**Neither arm clears the prevalence floor on AUPRC** (0.272): plain is below it, kg is level
with it. A per-slot base rate ranks these episodes as well as the model does. That is the
honest state of the ranking task, and it is the most important open problem in this report.

**The mechanism is visible, which is why the direction is credible.** Evidence was present
for 888/888 kg episodes, and the arms disagree on 423 of 888 schedules (48%). The clearest
effect is calibration: plain answers positive on 30.9% of cells against a true 24.2%, and
the block pulls that to 24.9%, trading recall (0.335 → 0.300) for precision (0.263 → 0.293).

Both arms beat the leave-one-out floor on slot 0 and both stay worse than the "answer 0 mg"
floor on slots 1-3. Over-prediction in the later windows is reduced, not solved.

## 8. Two ablations on the aspect split

The block labels paths "ARGUES TOWARD MORE DIURESIS" / "ARGUES TOWARD LESS", from two
scores whose only aspect-specific input is a similarity to one of two query sentences. Both
ablations ran on dev, and neither cost an LLM call.

### 8.1 A better similarity makes it worse

| scorer | paths in both aspects | Spearman(risk, protective) |
|---|---|---|
| lexical overlap (shipped) | 44.2% | +0.357 |
| PubMedBERT embeddings | 62.1% | +0.664 |

Not a model failure — a property of the queries:

```
cos(RISK_QUERY, PROTECTIVE_QUERY)       +0.771   PubMedBERT
cos("furosemide dose", "CKD stage 4")   +0.169   unrelated control
```

PubMedBERT separates genuinely different content perfectly well and still puts the two
aspect queries almost on top of each other, because they describe the same subject and
differ only in **polarity**. Cosine measures topic. No similarity function fixes that, and
TRACER anticipates it: 4.3.2 uses entailment, which has direction.

### 8.2 Entailment fixes the mechanism and changes nothing

`kg/retrieval/nli.py` (PubMedBERT-MNLI-MedNLI, run on a V100 via `kg/slurm/kg_nli.slurm`)
replaces both scores with P(entailment) for parallel hypotheses that differ only in polarity.

| scorer | paths/episode | in both aspects | Spearman | score std |
|---|---|---|---|---|
| lexical | 101 | 44.2% | +0.357 | 0.0124 |
| embeddings | 91 | 62.1% | +0.664 | 0.0410 |
| NLI, top-40 pool | 25 | 38.5% | +0.254 | 0.3432 |
| NLI, full pool | 45 | **33.0%** | **+0.093** | 0.3510 |

At the full pool the two aspects are nearly independent rankings and 57.8% of paths entail
exactly one hypothesis, where similarity gave direction to none. Re-running the arm on those
trajectories (`kg-v2`, 888/888 ok):

| contrast | AUPRC | macro-F1 | MAE slots 1-3 |
|---|---|---|---|
| kg-v2 − kg-v1 | +0.0009 [−0.0145, +0.0167] | −0.0065 [−0.0215, +0.0090] | +0.10 [−0.31, +0.49] |

Every interval spans zero. **The degenerate aspect labels were not what limited the arm** —
refuted with its own instrument rather than left as a suspicion. `kg-v1` remains the primary
result because it ran first, not because it scores better; the two are statistically
indistinguishable, so preferring v1 on its slightly larger macro-F1 gain would be selection
after the fact.

## 9. Limitations

1. **AUPRC never clears the prevalence floor** in any configuration. The ranking signal —
   `p_dose` — is where the task is currently lost, and neither the graph nor the refinement
   touches it.
2. **Severity is the constant 0.5.** It is the only term entering the two aspects with
   opposite signs, so it cancels, and the aspect split rests entirely on the queries. A real
   severity table is the untested lever.
3. **Single-source edges.** All 2,599 edges have one flow behind them; cross-source
   agreement cannot currently be used as a confidence signal.
4. **The serving stack cannot be fully named.** The gateway strips seller identity by
   design; per-row `request_id` is the only audit handle. Seller quantization is uncontrolled
   noise — the same 20 dev episodes scored AUPRC 0.386 through the gateway against 0.416 on
   another provider serving the same model string. Both arms ran back-to-back in one session
   so this is noise on both rather than a between-arm bias.
5. **One model, one prompt.** Nothing here says whether the block would help a stronger
   model, or one that ranks better than the base rate to begin with.
6. **kg-v2 was fed the top-40 trajectories**, not the full-pool ones, so the arm was run at
   38.5% aspect overlap rather than 33.0%.

## 10. A scoring bug worth remembering

The first printed table said `n = 908` for an 888-episode split, for both arms. `load_arm`
globbed `eval_seq_<arm>*.jsonl` to merge shard files, and with an empty tag that also matched
`eval_seq_<arm>_limit20.jsonl` — a 20-episode **dev** smoke from the same afternoon. Dev
episodes were being scored inside the test result, and the rows gave nothing away: same arm,
same model, `status: ok`. The contaminated numbers were AUPRC 0.264 / 0.272 and macro-F1
0.513 / 0.534, close enough to the clean ones to have survived review.

Fixed by anchoring the filename to `eval_seq_<arm><tag>(_shardNofM)?.jsonl`, with three
regression tests. Every number in this report comes from the fixed loader.

## 11. Reproducing this

```bash
export PYTHONPATH=<repo>/_pkgroot

# floors (no LLM)
python -m physioagent.eval.sequence_baselines --split test

# profiles and retrieval; the cluster job does both splits
sbatch physioagent/kg/slurm/kg_s0a_profiles.slurm
python -m physioagent.kg.retrieval.trajectory --split test \
  --profiles PhysioAgent/kg/data/patient_profiles_test.json

# the two arms
python -m physioagent.eval.run_eval_seq --arm plain --split test --workers 6 --rpm 30
python -m physioagent.eval.run_eval_seq --arm kg    --split test --workers 6 --rpm 30
python -m physioagent.eval.run_eval_seq --score --split test

# ablations
python -m physioagent.kg.retrieval.embed_local
python -m physioagent.kg.retrieval.trajectory --split dev --similarity embedding \
  --embed-local --embed-model NeuML/pubmedbert-base-embeddings --profiles ... --out ...
sbatch physioagent/kg/slurm/kg_nli.slurm      # SPLITS / REFINE_TOP are env overrides
```

Checkpoints are per row and resume on `status == "ok"`, so an interrupted arm continues from
the same command. Superseded checkpoints in `eval/results/seq/` are renamed to `.txt` rather
than deleted; the suffix says why each one is out of the way.

## 12. What to do next, in order

1. **Attack the ranking, not the graph.** AUPRC is the metric that fails, `p_dose` is what
   produces it, and both ablations so far moved things the metric does not read. Measure
   `p_dose` calibration directly before building anything else.
2. **Build the severity table** — the one lever in the scoring formula never tested. A
   data-driven source (in-hospital mortality per CCS condition, computable from MIMIC) is
   defensible and stays inside the DUA.
3. **Refresh the graph.** `kg_from_pubmed.jsonl` is 1.5 h newer than `kg_raw.jsonl`: 58 real
   edges never made it in, including a bioavailability and diuretic-resistance cluster that
   is squarely on topic, and 41 edges in the graph no longer exist upstream. Re-running
   combine + refine costs one kg arm re-run, because `plain` does not read the graph.
4. **One stronger model**, to separate "the block does not help" from "this model cannot use
   it".
