#!/bin/bash
# S0a + S1 + S0b: MIMIC-IV -> in-scope concept sets -> co-occurrence -> the sets S3 pays
# for, plus the 48 h dosing sequences with censoring resolved.
#
#   sbatch physioagent/kg/slurm/kg_s0s1.slurm
#
# NOT AN ARRAY JOB, and not sized like one. This is a single-pass table scan: two small
# tables read whole, prescriptions (~20M rows) read in chunks and reduced to in-scope codes
# inside each chunk. Sharding it would only multiply the reads. No GPU (nothing here is a
# model), no LLM (nothing here calls out), so it is pure I/O against /media02.
#
# SIZING. Peak memory is one prescriptions chunk plus the accumulated per-admission sets:
# ~550k admissions x three small sets of short code strings. 8G is several times that; the
# reason not to ask for more is the same as in eval/slurm/job.slurm -- Slurm treats memory
# as consumable, and a fat ask blocks us off nodes whose CPUs are free.
#
# THE FULL DUMP IS REQUIRED. The slim build (mimic-exploration/build_slim.py) has no
# prescriptions table, and prescriptions is where every drug concept comes from. Point
# MIMIC_ROOT at the full dump; the job aborts in a second if the table is missing rather
# than producing a graph with no drugs in it.
#
# WHAT TO READ WHEN IT FINISHES, before spending anything on S3:
#   kg/data/scope_report.json         concepts_never_seen  -> scope too narrow or a
#                                                             mapping gap
#                                     unmapped_drugs_top   -> extend drug_to_atc3.csv
#                                     diagnoses_unmapped   -> the CCS crosswalks are FY2019
#                                       .backoff_would_be_in_scope and MIMIC-IV 3.1 runs to
#                                                             2022, so codes CMS added
#                                                             after Oct 2019 (N18.30/31/32,
#                                                             the CKD stage-3a/3b split)
#                                                             cannot map. This counts the
#                                                             rows a parent-prefix fallback
#                                                             would STILL recover. Since
#                                                             2026-09-20 --dx-backoff
#                                                             defaults to icd10, so this
#                                                             counts what is left after
#                                                             the fallback, and
#                                                             diagnoses.backoff_in_scope
#                                                             counts what it rescued.
#   kg/data/concept_sets_report.json  llm_calls_s3         -> the S3 bill, in calls
#   kg/data/sequences_report.json     censored_episodes    -> episodes ending before 48 h
#
#SBATCH --job-name=physioagent-kg-s0s1
#SBATCH --partition=batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# --- edit for your account ------------------------------------------------------------
WORK=/media02/tphung03
PROJECT="$WORK/PhysioAgent"
CONDA_SH="$WORK/miniconda3/etc/profile.d/conda.sh"
ENV_DIR="$WORK/conda/envs/physioagent"
# The full MIMIC-IV dump: the directory holding hosp/. NOT the slim build sitting beside
# it -- mimic-iv-3.1-slim has no prescriptions table, which is where every drug concept
# comes from, and the check below refuses to start rather than build a drug-free graph.
export MIMIC_ROOT="${MIMIC_ROOT:-$WORK/MIMIC-IV/mimic-iv-3.1}"
PROFILE="${PROFILE:-default}"
MIN_FREE_MB="${MIN_FREE_MB:-2000}"
# --------------------------------------------------------------------------------------

# /media02 is a shared 5 T export with no quotas; job 73992 died when another account
# filled it mid-run. This job writes only a few hundred MB, but it still gets killed if the
# volume is full, so fail in 1 s instead of 30 min in. Runs BEFORE conda, which itself
# writes into $HOME (= /media02 here) and dies with an empty log on a full disk.
AVAIL_MB=$(df -BM --output=avail /media02 2>/dev/null | tail -1 | tr -dc '0-9')
if [[ -z "$AVAIL_MB" ]]; then
    echo "WARNING: could not read free space on /media02; continuing anyway" >&2
elif (( AVAIL_MB < MIN_FREE_MB )); then
    echo "ABORT: /media02 has ${AVAIL_MB}M free, need >=${MIN_FREE_MB}M." >&2
    exit 1
fi

# /home/tphung03 is read-only on this cluster; conda writes ~/.conda on nearly every call.
export HOME="$WORK"
source "$CONDA_SH"
conda activate "$ENV_DIR"
cd "$PROJECT"

echo "node=$(hostname)  mimic_root=$MIMIC_ROOT  profile=$PROFILE"
if [[ ! -d "$MIMIC_ROOT/hosp" ]]; then
    echo "ABORT: no hosp/ under $MIMIC_ROOT" >&2
    exit 1
fi
for t in diagnoses_icd procedures_icd prescriptions; do
    if ! compgen -G "$MIMIC_ROOT/hosp/$t.csv*" > /dev/null; then
        echo "ABORT: $MIMIC_ROOT/hosp/$t.csv[.gz] missing (slim build?)" >&2
        exit 1
    fi
done

# Single-threaded pandas: this is I/O bound and BLAS threads would just fight each other.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python -m physioagent.kg.ehr.build_visits --profile "$PROFILE"
python -m physioagent.kg.ehr.visit_concepts --profile "$PROFILE"

# S0b runs fine on a laptop from the cohort CSV alone -- but only here can it read
# admissions.dischtime/deathtime, and without those a late empty slot is ambiguous
# between "no dose" and "already discharged". Re-run it here so the artifact carries
# censoring rather than nulls.
python -m physioagent.kg.ehr.build_sequences \
    --cohort "${COHORT:-mimic-exploration/output/gate_b_cohort_prior.csv}" \
    --mimic-root "$MIMIC_ROOT" --profile "$PROFILE"

echo
echo "=== scope_report.json (head) ==="
python - <<'PY'
import json, pathlib
d = json.loads((pathlib.Path("physioagent/kg/data/scope_report.json")).read_text())
for k in ("admissions_seen", "admissions_written", "concepts_observed"):
    print(f"{k:34s} {d[k]:,}")
print(f"{'concepts never seen':34s} {len(d['concepts_never_seen'])}")
print(f"{'unmapped drug names':34s} {d['unmapped_drug_names']:,}")
u = d["diagnoses_unmapped"]
print(f"{'unmapped dx rows (icd9/icd10)':34s} "
      f"{u['rows_by_version'].get('icd9', 0):,} / {u['rows_by_version'].get('icd10', 0):,}")
print(f"{'  backoff would map (icd10)':34s} {u['backoff_would_map'].get('icd10', 0):,}")
print(f"{'  of those, in scope':34s} {u['backoff_would_be_in_scope'].get('icd10', 0):,}")
for name, n in list(u["backoff_concepts"].items())[:5]:
    print(f"{'    ' + name[:30]:34s} {n:,}")
c = json.loads((pathlib.Path("physioagent/kg/data/concept_sets_report.json")).read_text())
print(f"{'distinct concept sets':34s} {c['distinct_sets']:,}")
print(f"{'sets after similarity filter':34s} {c['filtered_sets']:,}")
print(f"{'S3 LLM calls':34s} {c['llm_calls_s3']:,}")
q = json.loads((pathlib.Path("physioagent/kg/data/sequences_report.json")).read_text())
print(f"{'episodes':34s} {q['episodes']:,}")
print(f"{'  train / test':34s} {q['episodes_by_split'].get('train', 0):,}"
      f" / {q['episodes_by_split'].get('test', 0):,}")
print(f"{'censored before 48h':34s} {q['censored_episodes']:,}")
PY
