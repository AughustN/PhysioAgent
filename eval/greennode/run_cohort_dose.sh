#!/bin/bash
# Score the plain arm on the T0 dose cohort with a model served by vLLM on this GreenNode
# notebook. Nothing leaves the notebook: the model runs on our GPU, no API key is involved.
#
#   MODEL=google/gemma-3-12b-it QUANT=fp8 LIMIT=5 bash physioagent/eval/greennode/run_cohort_dose.sh
#   MODEL=google/gemma-3-12b-it QUANT=fp8 bash ...               # full dev
#   MODEL=google/gemma-3-12b-it QUANT=fp8 SCORE=1 bash ...       # score
#
# Our slice is MIG 2g.20gb (~20 GB): a 12B model fits only with QUANT=fp8 (weights quantized
# by vLLM at load, H100 runs FP8 natively). A bf16 and an fp8 run are different scorers, so
# QUANT is part of the results folder name.
#
# Run from $CODE (the folder that CONTAINS physioagent/). Starts vLLM if nothing answers on
# $PORT, and refuses to score if the server answers with a different model. Needs
# kg/data/ehr/cohort/{cohort,features}.csv.gz (kg.ehr.cohort_build, kg.ehr.cohort_features).
# Prompt work goes on dev; test is run once, at the end.

set -euo pipefail

ROOT=/network-volume/physioagent
CODE=$ROOT/code
MODEL=${MODEL:?set MODEL to the Hugging Face id vLLM should serve}
SPLIT=${SPLIT:-dev}
SPLIT_COL=${SPLIT_COL:-split}
LIMIT=${LIMIT:-}
SCORE=${SCORE:-}
PORT=${PORT:-8000}
QUANT=${QUANT:-}
GPU_UTIL=${GPU_UTIL:-0.9}
MAX_LEN=${MAX_LEN:-4096}
MAX_SEQS=${MAX_SEQS:-8}

source /home/jovyan/physioagent/env.sh

# A .env file outranks the variables below (llm/client.py reads it first) and may point
# LLM_PROVIDER at an external API. MIMIC rows must not reach one.
if [[ -f "$CODE/physioagent/.env" ]]; then
    echo "ABORT: $CODE/physioagent/.env exists -- delete it; it overrides LLM_PROVIDER" >&2
    exit 1
fi

export LLM_PROVIDER=local_vllm LLM_MODE=local MODEL_NAME="$MODEL"
export OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1"
SLUG=${MODEL//\//-}${QUANT:+-$QUANT}
export PHYSIOAGENT_DOSE_RESULTS="$ROOT/results/cohort_dose_$SLUG"
mkdir -p "$PHYSIOAGENT_DOSE_RESULTS" "$ROOT/logs"

cd "$CODE"
if [[ -n "$SCORE" ]]; then
    python -m physioagent.eval.run_cohort_dose --split "$SPLIT" --split-col "$SPLIT_COL" \
        --score --save "$PHYSIOAGENT_DOSE_RESULTS/score_${SPLIT_COL}_${SPLIT}.json"
    exit 0
fi

VLLM_ARGS=()
if [[ -n "$QUANT" ]]; then
    VLLM_ARGS+=(--quantization "$QUANT")
fi
# Gemma 3 (4B and up) and MedGemma are image-text models; prompts here are text only, so
# skip the vision memory profile (same flag as run_gemma_dose.sh).
if [[ "$MODEL" =~ gemma-3-(4b|12b|27b)|medgemma-(4b|27b-it) ]]; then
    VLLM_ARGS+=(--limit-mm-per-prompt image=0)
fi

if ! curl -sf "$OPENAI_BASE_URL/models" >/dev/null; then
    echo "starting vLLM for $MODEL (log: $ROOT/logs/vllm_$SLUG.log)"
    nohup vllm serve "$MODEL" --port "$PORT" \
        --max-model-len "$MAX_LEN" \
        --gpu-memory-utilization "$GPU_UTIL" \
        --max-num-seqs 16 \
        ${VLLM_ARGS[@]+"${VLLM_ARGS[@]}"} \
        > "$ROOT/logs/vllm_$SLUG.log" 2>&1 &
    for _ in $(seq 1 180); do
        curl -sf "$OPENAI_BASE_URL/models" >/dev/null && break
        sleep 5
    done
fi
if ! curl -sf "$OPENAI_BASE_URL/models" | grep -q "\"$MODEL\""; then
    echo "ABORT: nothing on port $PORT serves $MODEL -- see $ROOT/logs/vllm_$SLUG.log" >&2
    exit 1
fi

python -m physioagent.eval.run_cohort_dose --split "$SPLIT" --split-col "$SPLIT_COL" \
    ${LIMIT:+--limit "$LIMIT"} --workers "$MAX_SEQS" --rpm 6000
