#!/usr/bin/env bash
# run_embedding.sh
# Usage: ./run_embedding.sh <MODEL_ID> [BATCH_SIZE] [MAX_LEN] [TASK_NAME] [OUTPUT_DIR]

set -e
set -u

MODEL_ID=${1:-"meta-llama/Meta-Llama-3-8B"}
BATCH_SIZE=${2:-1}
MAX_LEN=${3:-1024}
TASK_NAME=${4:-"model_embeddings"}
OUTPUT_BASE=${5:-"ablation/embeddings"}

PYTHON="python3"
SCRIPT_PATH="src/Ablation/embedding/ablation_emb.py"

echo "[EMBEDDING] Running embedding extraction..."
echo "Model: $MODEL_ID"
echo "Batch size: $BATCH_SIZE"
echo "Max length: $MAX_LEN"
echo "Task: $TASK_NAME"
echo "Output: $OUTPUT_BASE"

$PYTHON "$SCRIPT_PATH" \
    --model_id "$MODEL_ID" \
    --batch_size "$BATCH_SIZE" \
    --max_len "$MAX_LEN" \
    --task "$TASK_NAME" \
    --output_base "$OUTPUT_BASE"

echo "[EMBEDDING] Done. Files saved to $OUTPUT_BASE/$TASK_NAME/$(basename "$MODEL_ID")/dataset_tensor"
