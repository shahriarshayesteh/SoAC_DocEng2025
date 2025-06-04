#!/usr/bin/env bash
# run_ablation.sh
# Usage: ./run_ablation.sh <MODEL_NAME> <EMBED_SIZE> <COMMON_DIM> [SEEDS]

set -e
set -u

# ───────────────────────────────────────────────
# Input Arguments
# ───────────────────────────────────────────────
MODEL_NAME=${1:-"Llama-3.2-1B"}
EMBED_SIZE=${2:-2048}
COMMON_DIM=${3:-2048}
SEEDS=${4:-12}  # Default seed is now 12

CLASS_NUM=10
BATCH_SIZE=8
EPOCHS=15
LR=2e-4
DEVICE="cuda:0"

# ───────────────────────────────────────────────
# Paths
# ───────────────────────────────────────────────
TRAIN_TEMPLATE="/data/sxs7285/Porjects_code/thesis/SoAC/websector/${MODEL_NAME}_{}"
VAL_TEST_DIR="/data/sxs7285/Porjects_code/thesis/SoAC/websector/${MODEL_NAME}_12"
RESULTS_BASE="/data/sxs7285/Porjects_code/thesis/SoAC/ablation_results/${MODEL_NAME}"

PYTHON_SCRIPT="/data/sxs7285/Porjects_code/thesis/DocEng/SoAC-DocEng/src/training/run_ablation_train_eval.py"

echo "[ABLATION] Running seed(s): $SEEDS for $MODEL_NAME"

python3 "$PYTHON_SCRIPT" \
    --seeds $SEEDS \
    --train_template "${TRAIN_TEMPLATE}/dataset_tensor/" \
    --val_test_dir "${VAL_TEST_DIR}/dataset_tensor/" \
    --results_base "$RESULTS_BASE" \
    --embed_size "$EMBED_SIZE" \
    --common_dim "$COMMON_DIM" \
    --class_num "$CLASS_NUM" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --device "$DEVICE"

echo "[ABLATION] Done. Results in: $RESULTS_BASE"
