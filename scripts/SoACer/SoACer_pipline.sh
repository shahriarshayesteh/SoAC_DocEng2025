#!/bin/bash

# Usage: ./run_predict.sh input.txt output_dir/
# or:    ./run_predict.sh https://example.com output_dir/

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_file_or_url> <output_directory>"
    exit 1
fi

INPUT=$1
OUTPUT_DIR=$2

python3 src/inference/SoACer_pipeline.py --input "$INPUT" --output_dir "$OUTPUT_DIR"
