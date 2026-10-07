#!/bin/bash
# S8 + TRACER 4.3.2 on a GPU node: retrieval with entailment refinement, both splits.
#
#   sbatch physioagent/kg/slurm/kg_nli.slurm
#   SPLITS="dev" sbatch physioagent/kg/slurm/kg_nli.slurm      # one split only
#
# WHY THIS IS NOT THE LAPTOP'S JOB
# The refiner runs PubMedBERT-MNLI-MedNLI over every candidate path of every episode. On
# the laptop that was ~444k premise-hypothesis pairs for one split; it held the GPU at 99%
# for ten minutes and was then killed by the OS for memory pressure before writing
# anything. Here it gets its own GPU and its own memory.
#
# --refine-top 40 is the other half of the fix. The evidence block renders EIGHT paths, so
# refining ~250 per episode to choose 8 spends the transformer on a tail that rho already
# ranked last. The cap is applied to the similarity ranking, per aspect, and cuts the work
# roughly six fold. Raise it if an ablation wants the uncapped behaviour; 0 means all.
#
# WHAT MUST BE ON THE NODE BEFORE THIS RUNS
#   physioagent/kg/data/kg_refined.jsonl          the graph (gitignored; scp it up)
#   physioagent/kg/data/sequences.json            the episodes being scored
#   physioagent/kg/data/patient_profiles_{split}.json   built by kg_s0a_profiles.slurm
#   a conda env with torch + transformers
#   the NLI checkpoint in $HF_HOME, because compute nodes here have no outbound network:
#     (login node) HF_HOME=/media02/tphung03/hf python -c "from transformers import
#       AutoTokenizer, AutoModelForSequenceClassification as M;
#       n='pritamdeka/PubMedBERT-MNLI-MedNLI'; AutoTokenizer.from_pretrained(n); M.from_pretrained(n)"
#
#SBATCH --job-name=physioagent-kg-nli
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# --- edit for your account ------------------------------------------------------------
WORK=/media02/tphung03
PROJECT="$WORK/PhysioAgent"
CONDA_SH="$WORK/miniconda3/etc/profile.d/conda.sh"
ENV_DIR="$WORK/conda/envs/physioagent"
export HF_HOME="${HF_HOME:-$WORK/hf}"
SPLITS="${SPLITS:-test dev}"
REFINE_TOP="${REFINE_TOP:-40}"
# --------------------------------------------------------------------------------------

export HOME="$WORK"
source "$CONDA_SH"
conda activate "$ENV_DIR"
cd "$PROJECT"

# Compute nodes here have no outbound network. Without this the first from_pretrained
# blocks on a download that cannot finish, and the job burns its wall clock doing nothing.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "node=$(hostname)  splits='$SPLITS'  refine_top=$REFINE_TOP  HF_HOME=$HF_HOME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
    echo "ABORT: no GPU visible; this job asked for --gres=gpu:1" >&2; exit 1; }

python - <<'PY' || exit 1
import sys
try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: F401
except ImportError as exc:
    sys.exit(f"ABORT: {exc}. In the env: pip install 'transformers>=4.40' torch")
print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("ABORT: torch cannot see the GPU; wrong build or wrong node")
PY

# Job 77170 failed here after queueing and loading torch, on
# "No module named 'physioagent.kg.retrieval'": the checkout on this node predates that
# package, so the scp meant to deliver it had silently landed nowhere -- scp only writes
# into a directory that already exists. Import it before anything expensive: two seconds
# against a queue slot plus a model load.
python - <<'IMPORTCHECK' || exit 1
import sys
try:
    import physioagent.kg.retrieval.trajectory  # noqa: F401
    import physioagent.kg.retrieval.nli  # noqa: F401
except ImportError as exc:
    sys.exit(f'ABORT: {exc}. physioagent/kg/retrieval/ is missing or incomplete on this '
             'node; it needs __init__.py, trajectory.py and nli.py. scp only writes into '
             'a directory that already exists, so mkdir -p it first.')
print('modules   : physioagent.kg.retrieval OK')
IMPORTCHECK

for split in $SPLITS; do
    profiles="physioagent/kg/data/patient_profiles_${split}.json"
    if [[ ! -f "$profiles" ]]; then
        echo "ABORT: $profiles missing -- run kg_s0a_profiles.slurm first" >&2
        exit 1
    fi
    echo
    echo "=== $split, entailment-refined ==="
    python -u -m physioagent.kg.retrieval.trajectory \
        --split "$split" \
        --profiles "$profiles" \
        --refine nli \
        --refine-top "$REFINE_TOP" \
        --out "physioagent/kg/data/trajectories_${split}_nli.json"
done

echo
echo "Bring the results back:"
for split in $SPLITS; do
    echo "  scp tphung03@fit-hcmus:$PROJECT/physioagent/kg/data/trajectories_${split}_nli.json PhysioAgent/kg/data/"
done
