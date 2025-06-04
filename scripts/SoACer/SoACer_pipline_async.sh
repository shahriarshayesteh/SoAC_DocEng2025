#!/bin/bash

# Usage: ./run_predict.sh input_urls.txt output_dir/

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 input_urls.txt output_dir/"
    exit 1
fi

INPUT="$1"
OUTPUT="$2"

python3 main.py --input_txt "$INPUT" --output_dir "$OUTPUT"
