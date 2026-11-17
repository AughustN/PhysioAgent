#!/bin/bash
# One-time setup on the AAU AI Cloud front-end (ai-fe02.srv.aau.dk). Idempotent.
#
#   bash $ROOT/code/physioagent/eval/slurm/setup_env.sh
#
# Layout under $ROOT (default $HOME/physioagent; $HOME is private to you, and MIMIC-derived
# files must stay readable by PhysioNet-credentialed people only -- never put them in a
# /home/project/ folder opened with chmod o=rw):
#
#   code/physioagent/   this repository (the folder that CONTAINS physioagent/ is code/)
#   code/physioagent/kg/data/ehr/cohort/   cohort.csv.gz + features.csv.gz, built elsewhere
#                       with kg.ehr.cohort_build / cohort_features (raw MIMIC is not needed here)
#   venv/               Python packages of this repo, on top of the vLLM container
#   hf_cache/           model weights (Gemma 3 12B FP8: ~13 GB)
#   logs/  results/
#
# The Python side runs inside the cluster's vLLM container, so the venv is created with
# --system-site-packages and only adds what the container lacks (pandas, pymupdf, ...).

set -euo pipefail

ROOT=${ROOT:-$HOME/physioagent}
SIF=${SIF:-/home/container/vllm-openai_latest.sif}
MODEL=${MODEL:-RedHatAI/gemma-3-12b-it-FP8-dynamic}

mkdir -p "$ROOT/logs" "$ROOT/results" "$ROOT/hf_cache" "$ROOT/code/physioagent/kg/data/ehr/cohort"
chmod 700 "$ROOT"

if [[ ! -x "$ROOT/venv/bin/python" ]]; then
    singularity exec "$SIF" python3 -m venv --system-site-packages "$ROOT/venv"
fi
singularity exec "$SIF" "$ROOT/venv/bin/pip" install --quiet \
    -r "$ROOT/code/physioagent/requirements.txt"

# Pull the weights once from the front-end, so a GPU job never waits on a download.
# Gated models need HF_TOKEN in ~/.bashrc (see the AI Cloud batch LLM inference guide).
export HF_HOME="$ROOT/hf_cache"
singularity exec "$SIF" "$ROOT/venv/bin/python" -c \
    "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL'))"

singularity exec "$SIF" python3 -c "import vllm; print('vllm', vllm.__version__)"
for f in cohort.csv.gz features.csv.gz; do
    [[ -f "$ROOT/code/physioagent/kg/data/ehr/cohort/$f" ]] || echo "missing: kg/data/ehr/cohort/$f (upload it)"
done
df -h "$ROOT" | tail -n 1
echo "ready: $ROOT"
