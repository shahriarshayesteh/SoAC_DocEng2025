#!/usr/bin/env bash
set -e
set -u

# Default values
OUTPUT_DIR="data/soac_summaries"
SPLITS="train validation test"
SUMMARY_LENGTH=15
NUM_WORKERS=12

# Parse named arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --splits)
      SPLITS="$2"
      shift 2
      ;;
    --sentences_count)
      SUMMARY_LENGTH="$2"
      shift 2
      ;;
    --num_workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

PYTHON="python3"

echo "[SUMMARY] Running LexRank with ${SUMMARY_LENGTH} sentences, ${NUM_WORKERS} workers, splits: ${SPLITS}"
mkdir -p "$OUTPUT_DIR"

$PYTHON src/summary/lexrank.py \
    --output_dir "$OUTPUT_DIR" \
    --splits $SPLITS \
    --sentences_count "$SUMMARY_LENGTH" \
    --num_workers "$NUM_WORKERS"

echo "[SUMMARY] Done. Output: $OUTPUT_DIR"
