#!/bin/bash
# Score one LLM arm on the T0 dose cohort with a model served by vLLM inside this job.
# Nothing leaves the node: the model runs on the job's GPU and no API key is involved.
#
#   cd $ROOT
#   MODEL=RedHatAI/gemma-3-12b-it-FP8-dynamic LIMIT=5 sbatch code/physioagent/eval/slurm/run_cohort_dose.slurm
#   MODEL=RedHatAI/gemma-3-12b-it-FP8-dynamic sbatch code/physioagent/eval/slurm/run_cohort_dose.slurm
#   ARM=plain_calib MODEL=... sbatch ...            # calibration ablation
#   ARM=guideline LOGPROBS=1 MODEL=... sbatch ...   # rules arm, with E2 dose distribution
#
# A full run (no LIMIT) is scored at the end of the job; the aggregate report goes to
# $ROOT/results/cohort_dose_<model>/score_<arm>_<split col>_<split>.json.
#
# GPU type is pinned: a T4 has no bfloat16 (Gemma 3 overflows in float16), and the FP8
# checkpoint runs natively on L40S (Ada) but as weight-only Marlin on A40 (Ampere). Every arm
# of one comparison must run on the same GPU type and vLLM version (printed below).
# Prompt work goes on dev; test is run once, at the end.
#SBATCH --job-name=physio_dose
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

ROOT=${ROOT:-$HOME/physioagent}
SIF=${SIF:-/home/container/vllm-openai_latest.sif}
MODEL=${MODEL:?set MODEL to the Hugging Face id vLLM should serve}
ARM=${ARM:-plain}
SPLIT=${SPLIT:-dev}
SPLIT_COL=${SPLIT_COL:-split}
LIMIT=${LIMIT:-}
LOGPROBS=${LOGPROBS:-}
PORT=${PORT:-8000}
MAX_LEN=${MAX_LEN:-4096}
MAX_SEQS=${MAX_SEQS:-8}
# Explicit, because "auto" follows the checkpoint config: RedHatAI's Gemma 3 FP8 declares
# float32, which vLLM downcasts to float16, and Gemma 3 overflows in float16.
DTYPE=${DTYPE:-bfloat16}
PY="$ROOT/venv/bin/python"

# A .env file outranks the variables below (llm/client.py reads it first) and may point
# LLM_PROVIDER at an external API. MIMIC rows must not reach one.
if [[ -f "$ROOT/code/physioagent/.env" ]]; then
    echo "ABORT: $ROOT/code/physioagent/.env exists -- delete it; it overrides LLM_PROVIDER" >&2
    exit 1
fi

export HF_HOME="$ROOT/hf_cache" HF_HUB_OFFLINE=1
export LLM_PROVIDER=local_vllm LLM_MODE=local MODEL_NAME="$MODEL"
export OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1"
SLUG=${MODEL//\//-}
export PHYSIOAGENT_DOSE_RESULTS="$ROOT/results/cohort_dose_$SLUG"
mkdir -p "$PHYSIOAGENT_DOSE_RESULTS"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
singularity exec "$SIF" python3 -c "import vllm; print('vllm', vllm.__version__)"

VLLM_ARGS=()
# Gemma 3 (4B and up) and MedGemma are image-text models; prompts here are text only, so
# skip the vision memory profile.
if [[ "$MODEL" =~ gemma-3-(4b|12b|27b)|medgemma-(4b|27b-it) ]]; then
    VLLM_ARGS+=(--limit-mm-per-prompt '{"image": 0}')
fi

# Prompts carry patient values and must not reach the vLLM log. Older vLLM logs requests
# unless --disable-log-requests is given; newer vLLM logs none by default and has dropped
# that flag in favour of --enable-log-requests.
if ! singularity exec "$SIF" vllm serve --help=all 2>/dev/null | grep -q -- "--enable-log-requests"; then
    VLLM_ARGS+=(--disable-log-requests)
fi

VLLM_LOG="$ROOT/logs/vllm_${SLUG}_${SLURM_JOB_ID}.log"
singularity exec --nv "$SIF" vllm serve "$MODEL" --port "$PORT" \
    --max-model-len "$MAX_LEN" \
    --dtype "$DTYPE" \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs "$MAX_SEQS" \
    ${VLLM_ARGS[@]+"${VLLM_ARGS[@]}"} \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

# Stop waiting as soon as the server process is gone: a crash at start-up otherwise looks
# like a slow load.
for _ in $(seq 1 360); do
    curl -sf "$OPENAI_BASE_URL/models" >/dev/null && break
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ABORT: vLLM exited during start-up -- last lines of $VLLM_LOG:" >&2
        tail -n 15 "$VLLM_LOG" >&2
        exit 1
    fi
    sleep 5
done
if ! curl -sf "$OPENAI_BASE_URL/models" | grep -q "\"$MODEL\""; then
    echo "ABORT: nothing on port $PORT serves $MODEL -- see $VLLM_LOG" >&2
    exit 1
fi

cd "$ROOT/code"
singularity exec "$SIF" "$PY" -m physioagent.eval.run_cohort_dose \
    --arm "$ARM" --split "$SPLIT" --split-col "$SPLIT_COL" \
    ${LIMIT:+--limit "$LIMIT"} ${LOGPROBS:+--dose-logprobs} --workers "$MAX_SEQS" --rpm 6000

if [[ -z "$LIMIT" ]]; then
    singularity exec "$SIF" "$PY" -m physioagent.eval.run_cohort_dose \
        --arm "$ARM" --split "$SPLIT" --split-col "$SPLIT_COL" --score \
        --save "$PHYSIOAGENT_DOSE_RESULTS/score_${ARM}_${SPLIT_COL}_${SPLIT}.json"
fi
