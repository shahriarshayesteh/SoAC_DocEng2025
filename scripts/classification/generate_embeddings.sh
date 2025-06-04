#!/usr/bin/env bash
# Generate sentence embeddings for SoAC using LLaMA with multi-GPU.
# You can override variables by passing them when running the script, e.g.,
# MODEL_ID="deepseek-ai/DeepSeek-R1-Distill-Llama-8B" ./generate_embeddings.sh

set -e
set -u

# --------- DEFAULT CONFIGS (overridable) ---------
MODEL_ID="${MODEL_ID:-meta-llama/Meta-Llama-3-8B}"
TASK_NAME="${TASK_NAME:-model_embeddings}"
OUTPUT_DIR="${OUTPUT_DIR:-embeddings/model_embeddings}"
MAX_LEN="${MAX_LEN:-1024}"
BATCH_SIZE="${BATCH_SIZE:-8}"
# --------------------------------------------------

echo "[INFO] Generating embeddings using $MODEL_ID..."

python3 src/SoACer_training/embeddings/generate_embeddings.py \
  --model_id "$MODEL_ID" \
  --task "$TASK_NAME" \
  --output_base "$OUTPUT_DIR" \
  --max_len "$MAX_LEN" \
  --batch_size "$BATCH_SIZE"

echo "[DONE] All splits processed for $MODEL_ID"
