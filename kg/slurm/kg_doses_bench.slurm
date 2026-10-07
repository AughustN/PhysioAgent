#!/bin/bash
# Everything the index-dose benchmark needs from MIMIC, in one job:
#   profiles (S7) -> trajectories (S8), for train/dev/test of kg/data/doses.json.
#   Similar patients are ranked per row inside run_eval_dose (online, as TRACER).
#
#   sbatch physioagent/kg/slurm/kg_doses_bench.slurm
#   REBUILD_VISITS=1 sbatch ...     # only if visits_all.jsonl lacks *_times (see check)
#   REUSE_PROFILES=0 sbatch ...     # rebuild the profiles even if all three exist
#   SIMILARITY=lexical sbatch ...   # the old word-overlap scorer instead of embeddings
#
# SIMILARITY=embedding (default since 2026-09-29) scores paths by cosine in the NeuML
# PubMedBERT space, offline: it needs the three embeddings_*_NeuML-pubmedbert-base-
# embeddings.json files in kg/data (entities, relations, and the two cached queries);
# the compute node never loads a model or touches the network.
#
# The job never writes doses.json: the split the arms are scored on is the one uploaded.
# Upload it first:
#   scp PhysioAgent/kg/data/doses.json tphung03@fit-hcmus:$PROJECT/physioagent/kg/data/
#
# WHAT TO BRING BACK (kg/data/): trajectories_{train,dev,test}.json and
# patient_profiles_doses_{train,dev,test}.json -- run_eval_dose ranks similar patients
# from them. visits_all.jsonl stays on the node.
#
#SBATCH --job-name=physioagent-doses-bench
#SBATCH --partition=batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# --- edit for your account ------------------------------------------------------------
WORK=/media02/tphung03
PROJECT="$WORK/PhysioAgent"
CONDA_SH="$WORK/miniconda3/etc/profile.d/conda.sh"
ENV_DIR="$WORK/conda/envs/physioagent"
export MIMIC_ROOT="${MIMIC_ROOT:-$WORK/MIMIC-IV/mimic-iv-3.1}"
PROFILE="${PROFILE:-default}"
REBUILD_VISITS="${REBUILD_VISITS:-0}"
REUSE_PROFILES="${REUSE_PROFILES:-1}"
SIMILARITY="${SIMILARITY:-embedding}"
EMBED_MODEL=NeuML/pubmedbert-base-embeddings
# --------------------------------------------------------------------------------------

export HOME="$WORK"
source "$CONDA_SH"
conda activate "$ENV_DIR"
cd "$PROJECT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

DATA=physioagent/kg/data
DOSES=$DATA/doses.json
VISITS=$DATA/visits_all.jsonl
echo "node=$(hostname)  started $(date)"

if [[ ! -f "$DOSES" ]]; then
    echo "ABORT: $DOSES missing -- upload the file the arms are scored on" >&2
    exit 1
fi
python - <<'PY'
import json, collections, pathlib
rows = json.loads(pathlib.Path("physioagent/kg/data/doses.json").read_text())
c = collections.Counter(r["split"] for r in rows)
print("doses.json:", len(rows), "rows ", dict(sorted(c.items())))
PY

if [[ "$REBUILD_VISITS" == "1" ]]; then
    python -m physioagent.kg.ehr.build_visits --profile "$PROFILE"
fi
if [[ ! -f "$VISITS" ]]; then
    echo "ABORT: $VISITS missing -- rerun with REBUILD_VISITS=1" >&2
    exit 1
fi
# Step 0 needs event times. A visits_all.jsonl from before 2026-09-17 has none, and every
# downstream file would still build -- with every time falling back to admittime.
python - <<'PY'
import json, itertools
with open("physioagent/kg/data/visits_all.jsonl", encoding="utf-8") as h:
    rows = [json.loads(l) for l in itertools.islice(h, 20000)]
timed = sum(1 for r in rows if r.get("drug_times") or r.get("procedure_times"))
dated = sum(1 for r in rows if r.get("admittime"))
print(f"visits_all sample: {len(rows)} rows, {timed} with *_times, {dated} with admittime")
if timed == 0 or dated == 0:
    raise SystemExit("ABORT: visits_all.jsonl has no timestamps -- rerun with REBUILD_VISITS=1")
PY

if [[ "$SIMILARITY" == "embedding" ]]; then
    SLUG=${EMBED_MODEL//\//-}
    for kind in entities relations queries; do
        if [[ ! -f "$DATA/embeddings_${kind}_${SLUG}.json" ]]; then
            echo "ABORT: $DATA/embeddings_${kind}_${SLUG}.json missing -- upload it" >&2
            exit 1
        fi
    done
    SIM_ARGS=(--similarity embedding --embed-local --embed-model "$EMBED_MODEL")
else
    SIM_ARGS=(--similarity lexical)
fi

for split in train dev test; do
    if [[ "$REUSE_PROFILES" == "1" && -f "$DATA/patient_profiles_doses_${split}.json" ]]; then
        echo "=== S7 profiles $split: reusing $DATA/patient_profiles_doses_${split}.json"
        continue
    fi
    echo; echo "=== S7 profiles $split  $(date +%T) ==="
    python -m physioagent.kg.ehr.patient_profiles \
        --doses "$DOSES" --split "$split" --visits "$VISITS" \
        --out "$DATA/patient_profiles_doses_${split}.json"
done

for split in train dev test; do
    echo; echo "=== S8 trajectories $split  $(date +%T) ==="
    python -m physioagent.kg.retrieval.trajectory \
        --doses "$DOSES" --split "$split" \
        --profiles "$DATA/patient_profiles_doses_${split}.json" \
        --out "$DATA/trajectories_${split}.json" "${SIM_ARGS[@]}"
    cp "$DATA/trajectory_report.json" "$DATA/trajectory_report_${split}.json"
done

echo; echo "=== coverage ==="
python - <<'PY'
import json, pathlib, statistics
data = pathlib.Path("physioagent/kg/data")
doses = json.loads((data / "doses.json").read_text())
for split in ("train", "dev", "test"):
    ids = [r["dose_id"] for r in doses if r["split"] == split]
    prof = json.loads((data / f"patient_profiles_doses_{split}.json").read_text())
    traj = json.loads((data / f"trajectories_{split}.json").read_text())
    timed = event = total = 0
    for p in prof.values():
        for k, v in p.items():
            if k.startswith("visit_"):
                total += 1
                event += bool(v.get("concept_times"))
    with_path = sum(1 for i in ids if traj.get(i, {}).get("n_paths"))
    no_hist = sum(1 for p in prof.values() if p["n_prior_visits"] == 0)
    print(f"{split}: {len(ids)} rows | profiles {len(prof)} | no earlier admission {no_hist} "
          f"| visits with event times {event}/{total} | rows with a path {with_path}")
PY
echo; echo "done $(date)"
echo "Bring back:"
echo "  scp tphung03@fit-hcmus:$PROJECT/$DATA/{trajectories_dev,trajectories_test,patient_profiles_doses_train,patient_profiles_doses_dev,patient_profiles_doses_test,trajectories_train}.json PhysioAgent/kg/data/"
