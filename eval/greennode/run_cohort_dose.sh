#!/bin/bash
# Score the plain arm on the T0 dose cohort with a model served by vLLM on this GreenNode
# notebook. Nothing leaves the notebook: the model runs on our GPU, no API key is involved.
#
#   MODEL=RedHatAI/gemma-3-12b-it-FP8-dynamic LIMIT=5 bash ...   # smoke: 5 dev rows
#   MODEL=RedHatAI/gemma-3-12b-it-FP8-dynamic bash ...           # full dev
#   MODEL=RedHatAI/gemma-3-12b-it-FP8-dynamic SCORE=1 bash ...   # score
#   ARM=plain_calib MODEL=... bash ...                           # calibration ablation
#   ARM=guideline LOGPROBS=1 MODEL=... bash ...                  # rules arm, with E2
#
# LOGPROBS=1 (E2) scores every menu dose with prompt_logprobs, which vLLM serves only with
# prefix caching off, so the server is started with --no-enable-prefix-caching. A server
# already running from an earlier start keeps its flags: stop it first (pkill -f "vllm serve").
#
# Our slice is MIG 2g.20gb (~20 GB). A 12B model fits only as a checkpoint already stored in
# FP8 (~13 GB). QUANT=fp8 does not help there: vLLM 0.8.5 allocates the weights in bf16
# (~24 GB for 12B) and quantizes after loading, and the out-of-memory then surfaces as
# "NVML_SUCCESS == r INTERNAL ASSERT FAILED", because NVML cannot read memory on a MIG
# slice. QUANT suits models whose bf16 weights fit. A bf16 and an fp8 run are different
# scorers, so QUANT is part of the results folder name.
#
# Run from $CODE (the folder that CONTAINS physioagent/). Starts vLLM if nothing answers on
# $PORT, and refuses to score if the server answers with a different model. Needs
# kg/data/ehr/cohort/{cohort,features}.csv.gz (kg.ehr.cohort_build, kg.ehr.cohort_features).
# Prompt work goes on dev; test is run once, at the end.

set -euo pipefail

ROOT=/network-volume/physioagent
CODE=$ROOT/code
MODEL=${MODEL:?set MODEL to the Hugging Face id vLLM should serve}
ARM=${ARM:-plain}
SPLIT=${SPLIT:-dev}
SPLIT_COL=${SPLIT_COL:-split}
LIMIT=${LIMIT:-}
SCORE=${SCORE:-}
LOGPROBS=${LOGPROBS:-}
PORT=${PORT:-8000}
QUANT=${QUANT:-}
GPU_UTIL=${GPU_UTIL:-0.9}
MAX_LEN=${MAX_LEN:-4096}
MAX_SEQS=${MAX_SEQS:-8}
# Explicit, because "auto" follows the checkpoint config: RedHatAI's Gemma 3 FP8 declares
# float32, which vLLM downcasts to float16, and Gemma 3 overflows in float16.
DTYPE=${DTYPE:-bfloat16}

# env.sh (vLLM venv, HF_HOME) lived in /home/jovyan, which a notebook restart can wipe.
# Without it, vllm must already be on PATH; the model cache then goes on the volume.
ENV_SH=${ENV_SH:-/home/jovyan/physioagent/env.sh}
if [[ -f "$ENV_SH" ]]; then
    source "$ENV_SH"
fi
export HF_HOME=${HF_HOME:-$ROOT/hf_cache}
if [[ -z "$SCORE" ]] && ! command -v vllm >/dev/null; then
    echo "ABORT: vllm not on PATH and $ENV_SH missing -- install vLLM or set ENV_SH" >&2
    exit 1
fi

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
    python -m physioagent.eval.run_cohort_dose --arm "$ARM" --split "$SPLIT" \
        --split-col "$SPLIT_COL" --score \
        --save "$PHYSIOAGENT_DOSE_RESULTS/score_${ARM}_${SPLIT_COL}_${SPLIT}.json"
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
if [[ -n "$LOGPROBS" ]]; then
    VLLM_ARGS+=(--no-enable-prefix-caching)
fi

if ! curl -sf "$OPENAI_BASE_URL/models" >/dev/null; then
    echo "starting vLLM for $MODEL (log: $ROOT/logs/vllm_$SLUG.log)"
    nohup vllm serve "$MODEL" --port "$PORT" \
        --max-model-len "$MAX_LEN" \
        --dtype "$DTYPE" \
        --gpu-memory-utilization "$GPU_UTIL" \
        --max-num-seqs "$MAX_SEQS" \
        ${VLLM_ARGS[@]+"${VLLM_ARGS[@]}"} \
        > "$ROOT/logs/vllm_$SLUG.log" 2>&1 &
    VLLM_PID=$!
    # The first start downloads the weights, so wait up to an hour -- but stop as soon as
    # the server process is gone: a crash at import time otherwise looks like a slow load.
    for _ in $(seq 1 720); do
        curl -sf "$OPENAI_BASE_URL/models" >/dev/null && break
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "ABORT: vLLM exited during start-up -- last lines of the log:" >&2
            tail -n 15 "$ROOT/logs/vllm_$SLUG.log" >&2
            exit 1
        fi
        sleep 5
    done
fi
if ! curl -sf "$OPENAI_BASE_URL/models" | grep -q "\"$MODEL\""; then
    echo "ABORT: nothing on port $PORT serves $MODEL -- see $ROOT/logs/vllm_$SLUG.log" >&2
    exit 1
fi

python -m physioagent.eval.run_cohort_dose --arm "$ARM" --split "$SPLIT" --split-col "$SPLIT_COL" \
    ${LIMIT:+--limit "$LIMIT"} ${LOGPROBS:+--dose-logprobs} --workers "$MAX_SEQS" --rpm 6000
