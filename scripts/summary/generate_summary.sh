#!/usr/bin/env bash
set -e
set -u

SUMMARY_LENGTH=${1:-15}
NUM_WORKERS=${2:-12}
OUTPUT_DIR=${3:-"data/soac_summaries/sc${SUMMARY_LENGTH}"}
PYTHON="python3"

echo "[SUMMARY] Running LexRank with ${SUMMARY_LENGTH} sentences and ${NUM_WORKERS} workers..."
mkdir -p "$OUTPUT_DIR"

$PYTHON /data/sxs7285/Porjects_code/thesis/DocEng/SoAC-DocEng/src/summary/lexrank.py \
    --output_dir "$OUTPUT_DIR" \
    --splits train validation test \
    --sentences_count "$SUMMARY_LENGTH" \
    --num_workers "$NUM_WORKERS"

echo "[SUMMARY] Done. Output: $OUTPUT_DIR"






